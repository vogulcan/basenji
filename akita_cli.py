#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
from typing import List, Tuple, Iterable, Dict, Any

import click
import numpy as np
import pysam
import tensorflow as tf

from basenji import dna_io, seqnn
from cooltools.lib.numutils import set_diag

# -----------------------------
# Hardcoded config
# -----------------------------
BASE = "."
DATA_STATS = f"{BASE}/data/hg38_statistics.json"
MODELS_GLOB = f"{BASE}/models/f*c0/train/model0_best.h5"

# -----------------------------
# Hardcoded human target heads
# -----------------------------
HEADS: Dict[int, Tuple[str, str]] = {
    0: ("HFF",     "HIC:HFF"),
    1: ("H1hESC",  "HIC:H1hESC"),
    2: ("GM12878", "HIC:GM12878"),
    3: ("IMR90",   "HIC:IMR90"),
    4: ("HCT116",  "HIC:HCT116"),
}
HEAD_IDENTIFIERS = [HEADS[i][0] for i in sorted(HEADS)]
HEAD_IDENTIFIERS_SET = {h.lower() for h in HEAD_IDENTIFIERS}


# -----------------------------
# Utilities
# -----------------------------
def load_data_stats(stats_path: str) -> Dict[str, Any]:
    with open(stats_path) as f:
        s = json.load(f)
    seq_length = s["seq_length"]
    target_length = s["target_length"]
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


def load_seqnn_model(model_h5: str, params_json: str):
    with open(params_json) as f:
        params = json.load(f)
    params_model = params["model"]
    model = seqnn.SeqNN(params_model)
    model.restore(model_h5, head_i=0)
    return model


def from_upper_triu(vector_repr: np.ndarray, matrix_len: int, num_diags: int) -> np.ndarray:
    z = np.zeros((matrix_len, matrix_len), dtype=np.float32)
    triu_tup = np.triu_indices(matrix_len, num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)
    return z + z.T


def fasta_batches(fasta_path: str, batch_size: int) -> Iterable[Tuple[List[str], List[Tuple[str, str]]]]:
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
    arrs = []
    for hdr, seq in records:
        if len(seq) != seq_length:
            raise ValueError(
                f"Sequence '{hdr}' length {len(seq)} != expected {seq_length}."
            )
        arrs.append(dna_io.dna_1hot(seq))
    return np.stack(arrs, axis=0).astype(np.float32)


def resolve_target_selection(targets: Tuple[str, ...]) -> Tuple[List[int], List[str]]:
    if targets:
        requested = {t.lower() for t in targets}
        unknown = sorted(requested - HEAD_IDENTIFIERS_SET)
        if unknown:
            raise click.ClickException(
                f"Unknown target(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(HEAD_IDENTIFIERS)}"
            )
        chosen = [(i, HEADS[i][0]) for i in sorted(HEADS) if HEADS[i][0].lower() in requested]
    else:
        chosen = [(i, HEADS[i][0]) for i in sorted(HEADS)]
    idxs = [i for i, _ in chosen]
    names = [n for _, n in chosen]
    return idxs, names


# -----------------------------
# CLI
# -----------------------------
@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    help=(
        "Ensemble Akita v2 HUMAN models (all 8 folds) on a multi-FASTA:\n"
        "- averages predictions across folds\n"
        "- converts selected heads to symmetric Hi-C matrices\n"
        "- writes a compressed NPZ with 'headers', 'head_names', 'mats'\n"
        "\n"
        "Available heads:\n"
        "[0] HFF      - HIC:HFF\n"
        "[1] H1hESC   - HIC:H1hESC\n"
        "[2] GM12878  - HIC:GM12878\n"
        "[3] IMR90    - HIC:IMR90\n"
        "[4] HCT116   - HIC:HCT116\n"
    ),
)
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-size", type=int, default=4, show_default=True, help="Batch size for inference.")
@click.option("--out", "out_npz", default="akita_ensemble_outputs.npz", show_default=True, help="Output NPZ path.")
@click.option("--models-glob", default=MODELS_GLOB, show_default=True, help="Glob for human model0_best.h5 files. Use model1_best.h5 for mouse.")
@click.option("--stats", "stats_path", default=DATA_STATS, show_default=True, help="Path to statistics.json. Use mm10_statistics.json for mouse in data.")
@click.option(
    "--targets",
    multiple=True,
    help="Target head identifiers to export (by name). If omitted, uses ALL heads.",
)
def main(fasta, batch_size, out_npz, models_glob, stats_path, targets):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")

    # Load stats
    stats = load_data_stats(stats_path)
    seq_length = stats["seq_length"]
    target_length = stats["target_length"]
    hic_diags = stats["hic_diags"]
    Lc = stats["target_length1_cropped"]

    click.echo(f"seq_length: {seq_length}")
    click.echo(f"flattened target_length: {target_length}")
    click.echo(f"matrix (cropped) size: ({Lc},{Lc})")

    # Resolve which heads to export
    head_indices, head_names = resolve_target_selection(targets)
    click.echo("Using heads:")
    for i in head_indices:
        click.echo(f"  [{i}] {HEADS[i][0]}  -  {HEADS[i][1]}")

    # Discover & load models
    pairs = find_models_and_params(models_glob)
    click.echo(f"Found {len(pairs)} human checkpoints:")
    for m, _ in pairs:
        click.echo(f"  - {m}")
    models = [load_seqnn_model(m, p) for m, p in pairs]
    click.echo(f"Loaded {len(models)} models.")

    all_headers: List[str] = []
    mats: List[np.ndarray] = []

    for headers, recs in fasta_batches(fasta, batch_size):
        x = encode_and_stack(recs, seq_length)

        pred_sum = None
        for model in models:
            y = model.model.predict(x, verbose=0)
            pred_sum = y if pred_sum is None else pred_sum + y
        ensemble = pred_sum / float(len(models))

        for i, hdr in enumerate(headers):
            head_mats = []
            for h_idx in head_indices:
                flat = ensemble[i, :, h_idx].astype(np.float32)
                mat = from_upper_triu(flat, Lc, hic_diags)
                head_mats.append(mat)
            mats.append(np.stack(head_mats, axis=0))
            all_headers.append(hdr)

        click.echo(f"Processed {len(all_headers)} sequences so far...")

    mats_arr = np.stack(mats, axis=0) if mats else np.zeros((0, len(head_indices), Lc, Lc), dtype=np.float32)
    headers_arr = np.array(all_headers, dtype=object)
    head_names_arr = np.array(head_names, dtype=object)

    np.savez_compressed(out_npz, headers=headers_arr, head_names=head_names_arr, mats=mats_arr)
    click.echo(f"Saved N={len(all_headers)}, H={len(head_indices)} matrices to: {out_npz}")
    click.echo("NPZ keys: 'headers', 'head_names', 'mats' (float32, N x H x L x L)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise click.ClickException(str(e))
