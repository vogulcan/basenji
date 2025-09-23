#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import glob
import json
from typing import List, Tuple, Iterable, Dict, Any

import click
import numpy as np
import pysam
import tensorflow as tf
import tqdm

from basenji import dna_io, seqnn
from cooltools.lib.numutils import set_diag

# -----------------------------
# Base config (paths relative to CWD by default)
# -----------------------------
BASE = "."
HG38_STATS_DEFAULT = f"{BASE}/data/hg38_statistics.json"
MM10_STATS_DEFAULT = f"{BASE}/data/mm10_statistics.json"

HG38_MODELS_DEFAULT = f"{BASE}/models/f*c0/train/model0_best.h5"  # human models
MM10_MODELS_DEFAULT = f"{BASE}/models/f*c0/train/model1_best.h5"  # mouse models

# -----------------------------
# Hardcoded target heads
# -----------------------------
HEADS_HG38: Dict[int, Tuple[str, str]] = {
    0: ("HFF", "HIC:HFF"),
    1: ("H1hESC", "HIC:H1hESC"),
    2: ("GM12878", "HIC:GM12878"),
    3: ("IMR90", "HIC:IMR90"),
    4: ("HCT116", "HIC:HCT116"),
}

HEADS_MM10: Dict[int, Tuple[str, str]] = {
    0: ("Hsieh2019_mESC_uC", "HIC:mESC"),
    1: ("Bonev2017_mESC", "HIC:mESC"),
    2: ("Bonev2017_CN", "HIC:cortical neuron"),
    3: ("Bonev2017_ncx_CN", "HIC:neocortex cortical neuron"),
    4: ("Bonev2017_NPC", "HIC:neural progenitor cell"),
    5: ("Bonev2017_ncx_NPC", "HIC:neocortex neural progenitor cell"),
}


# -----------------------------
# Utilities
# -----------------------------
def load_data_stats(stats_path: str) -> Dict[str, Any]:
    with open(stats_path) as f:
        s = json.load(f)
    seq_length = s["seq_length"]
    target_length = s["target_length"]  # flattened upper-tri length
    hic_diags = s["diagonal_offset"]
    target_crop = s["crop_bp"] // s["pool_width"]
    target_length1 = s["seq_length"] // s["pool_width"]
    target_length1_cropped = target_length1 - 2 * target_crop
    return {
        "seq_length": seq_length,
        "target_length": target_length,
        "hic_diags": hic_diags,
        "target_length1_cropped": target_length1_cropped,
    }


def find_models_and_params(models_glob: str) -> List[Tuple[str, str]]:
    models = sorted(glob.glob(models_glob))
    if not models:
        raise FileNotFoundError(f"No model checkpoints found with glob: {models_glob}")
    pairs = []
    for m in models:
        p = os.path.join(os.path.dirname(m), "params.json")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"params.json not found next to {m}")
        pairs.append((m, p))
    return pairs


def load_seqnn_model(model_h5: str, params_json: str, head_i: int):
    with open(params_json) as f:
        params = json.load(f)
    params_model = params["model"]
    model = seqnn.SeqNN(params_model)
    # Silence Keras summary while restoring (optional; safe if noisy)
    import contextlib

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
        devnull
    ), contextlib.redirect_stderr(devnull):
        model.restore(model_h5, head_i=head_i)
    return model


def from_upper_triu(
    vector_repr: np.ndarray, matrix_len: int, num_diags: int
) -> np.ndarray:
    """Reconstruct symmetric matrix (NaNs along masked diagonals) from upper-tri vector."""
    z = np.zeros((matrix_len, matrix_len), dtype=np.float32)
    triu_tup = np.triu_indices(matrix_len, num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)
    return z + z.T


def fasta_batches(
    fasta_path: str, batch_size: int
) -> Iterable[Tuple[List[str], List[Tuple[str, str]]]]:
    """Yield (headers, records) where records are (header, seq) uppercase."""
    with pysam.FastxFile(fasta_path) as fh:
        headers, recs = [], []
        for rec in fh:
            headers.append(rec.name)
            recs.append((rec.name, rec.sequence.upper()))
            if len(recs) == batch_size:
                yield headers, recs
                headers, recs = [], []
        if headers:
            yield headers, recs


def encode_and_stack(records: List[Tuple[str, str]], seq_length: int) -> np.ndarray:
    """Convert list of (header, seq) to batch array [B, seq_length, 4]."""
    arrs = []
    for hdr, seq in records:
        if len(seq) != seq_length:
            raise ValueError(
                f"Sequence '{hdr}' length {len(seq)} != expected {seq_length}."
            )
        arrs.append(dna_io.dna_1hot(seq))
    return np.stack(arrs, axis=0).astype(np.float32)


def resolve_target_selection(
    targets: Tuple[str, ...], HEADS: Dict[int, Tuple[str, str]]
) -> Tuple[List[int], List[str]]:
    """Validate/resolve user-specified identifiers against HEADS, return (indices, names)."""
    identifiers = [HEADS[i][0] for i in sorted(HEADS)]
    ident_set = {h.lower() for h in identifiers}
    if targets:
        requested = {t.lower() for t in targets}
        unknown = sorted(requested - ident_set)
        if unknown:
            raise click.ClickException(
                f"Unknown target(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(identifiers)}"
            )
        chosen = [
            (i, HEADS[i][0]) for i in sorted(HEADS) if HEADS[i][0].lower() in requested
        ]
    else:
        chosen = [(i, HEADS[i][0]) for i in sorted(HEADS)]
    idxs = [i for i, _ in chosen]
    names = [n for _, n in chosen]
    return idxs, names


def genome_config(genome: str):
    """Return (HEADS, head_i_default, stats_default, models_default) for genome."""
    genome = genome.lower()
    if genome == "hg38":
        return HEADS_HG38, 0, HG38_STATS_DEFAULT, HG38_MODELS_DEFAULT
    elif genome == "mm10":
        return HEADS_MM10, 1, MM10_STATS_DEFAULT, MM10_MODELS_DEFAULT
    else:
        raise click.ClickException("Invalid --genome. Use 'hg38' or 'mm10'.")

def ensemble_preds(preds_by_fold, mode="log_mean"):
    """
    preds_by_fold: list of np.ndarray of shape [B, T, C] (log(O/E))
    mode: "log_mean" (default) or "oe_mean"
    """
    stack = np.stack(preds_by_fold, axis=0)  # [F, B, T, C]
    if mode == "log_mean":
        return stack.mean(axis=0)  # mean in log space
    elif mode == "oe_mean":
        return np.log(np.clip(np.mean(np.exp(stack), axis=0), 1e-6, None))
    else:
        raise ValueError("mode must be 'log_mean' or 'oe_mean'")


# -----------------------------
# CLI
# -----------------------------
@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    help="""\
Ensemble Akita v2 models (all 8 folds) on a multi-FASTA:
- averages predictions across folds
- converts selected heads to symmetric Hi-C matrices
- writes a compressed NPZ with 'headers', 'head_names', 'mats'

Available heads (hardcoded):

  hg38:
    [0] HFF           HIC:HFF
    [1] H1hESC        HIC:H1hESC
    [2] GM12878       HIC:GM12878
    [3] IMR90         HIC:IMR90
    [4] HCT116        HIC:HCT116

  mm10:
    [0] Hsieh2019_mESC_uC    HIC:mESC
    [1] Bonev2017_mESC       HIC:mESC
    [2] Bonev2017_CN         HIC:cortical neuron
    [3] Bonev2017_ncx_CN     HIC:neocortex cortical neuron
    [4] Bonev2017_NPC        HIC:neural progenitor cell
    [5] Bonev2017_ncx_NPC    HIC:neocortex neural progenitor cell
""",
)
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--genome",
    type=click.Choice(["hg38", "mm10"], case_sensitive=False),
    default="hg38",
    show_default=True,
    help="Genome to use (sets head mapping and default model/stats).",
)
@click.option(
    "--batch-size",
    type=int,
    default=4,
    show_default=True,
    help="Batch size for inference.",
)
@click.option(
    "--out",
    "out_npz",
    default="akita_ensemble_outputs.npz",
    show_default=True,
    help="Output NPZ path.",
)
@click.option(
    "--models-glob",
    default="",
    show_default=False,
    help="Glob for model checkpoints. If omitted, set from --genome.",
)
@click.option(
    "--stats",
    "stats_path",
    default="",
    show_default=False,
    help="Path to statistics.json. If omitted, set from --genome.",
)
@click.option(
    "--targets",
    multiple=True,
    help="Target head identifiers to export (by name). If omitted, exports outputs from ALL heads for the chosen genome.",
)
def main(fasta, genome, batch_size, out_npz, models_glob, stats_path, targets):
    # Quiet TensorFlow logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")

    # Resolve genome-specific config
    HEADS, head_i, stats_default, models_default = genome_config(genome)

    # Smart defaults for --models-glob / --stats if not provided
    if not models_glob:
        models_glob = models_default
    if not stats_path:
        stats_path = stats_default

    click.echo(f"Genome: {genome}  |  head_i: {head_i}")
    click.echo(f"Models glob: {models_glob}")
    click.echo(f"Stats path:  {stats_path}")

    # Load stats
    stats = load_data_stats(stats_path)
    seq_length = stats["seq_length"]
    target_length = stats["target_length"]
    hic_diags = stats["hic_diags"]
    Lc = stats["target_length1_cropped"]

    click.echo(f"seq_length: {seq_length}")
    click.echo(f"flattened target_length: {target_length}")
    click.echo(f"matrix (cropped) size: ({Lc},{Lc})")

    # Resolve which heads to export (genome-specific)
    head_indices, head_names = resolve_target_selection(targets, HEADS)
    click.echo("Using heads:")
    for i in head_indices:
        click.echo(f"  [{i}] {HEADS[i][0]}  -  {HEADS[i][1]}")

    # Discover & load models
    pairs = find_models_and_params(models_glob)
    click.echo(f"Found {len(pairs)} checkpoints:")
    for m, _ in pairs:
        click.echo(f"  - {m}")
    models = [load_seqnn_model(m, p, head_i=head_i) for m, p in pairs]
    click.echo(f"Loaded {len(models)} models.")

    all_headers: List[str] = []
    mats: List[np.ndarray] = []  # each entry: [H, Lc, Lc] per sequence

    # Iterate FASTA in batches
    with pysam.FastxFile(fasta) as fh:
        total_seqs = sum(1 for _ in fh)
    total_batches = math.ceil(total_seqs / batch_size)

    for headers, recs in tqdm.tqdm(
        fasta_batches(fasta, batch_size),
        desc="Batches",
        unit="batch",
        total=total_batches,
    ):
        x = encode_and_stack(recs, seq_length)  # [B, L, 4]

        # predict with each model then average: [B, T, C]
        preds_by_fold = []
        for model in models[:1]:
            y = model.model.predict(x, verbose=0)  # shape [B, T, C]
            print("pred shape:", y.shape,
                  "min/max:", float(np.nanmin(y)), float(np.nanmax(y)))
            preds_by_fold.append(y.astype(np.float32))

        ensemble = ensemble_preds(preds_by_fold, mode="log_mean")  # you could also pass in from a CLI flag

        # For each sequence in batch, convert selected heads to matrices
        for i, hdr in enumerate(headers):
            head_mats = []
            for h_idx in head_indices:
                flat = ensemble[i, :, h_idx]  # using the chosen head
                mat = from_upper_triu(flat, Lc, hic_diags)
                head_mats.append(mat)
            mats.append(np.stack(head_mats, axis=0))
            all_headers.append(hdr)

    # Stack and save
    mats_arr = (
        np.stack(mats, axis=0)
        if mats
        else np.zeros((0, len(head_indices), Lc, Lc), dtype=np.float32)
    )
    headers_arr = np.array(all_headers, dtype=object)
    head_names_arr = np.array(head_names, dtype=object)

    np.savez_compressed(
        out_npz, headers=headers_arr, head_names=head_names_arr, mats=mats_arr
    )
    click.echo(
        f"Saved N={len(all_headers)}, H={len(head_indices)} matrices to: {out_npz}"
    )
    click.echo(
        "NPZ keys: 'headers' (object), 'head_names' (object), 'mats' (float32, N x H x L x L)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise click.ClickException(str(e))
