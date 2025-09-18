#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import csv
from typing import List, Tuple, Iterable, Dict, Any

import click
import numpy as np
import pysam
import tensorflow as tf

from basenji import dna_io, seqnn
from cooltools.lib.numutils import set_diag


# -----------------------------
# Hardcoded defaults (adjust if needed)
# -----------------------------
BASE = "/home/carlos/Clone/gnn_benchmark/basenji"
DATA_STATS = f"{BASE}/data/hg38/statistics.json"
TARGETS_TXT = f"{BASE}/data/human/targets.txt"  # human targets definition
# Human Akita v2 models (model0_* = human) across 8 folds
MODELS_GLOB = f"{BASE}/models/f*c0/train/model0_best.h5"


# -----------------------------
# Utilities
# -----------------------------
def load_data_stats(stats_path: str) -> Dict[str, Any]:
    with open(stats_path) as f:
        s = json.load(f)
    seq_length = s["seq_length"]
    target_length = s["target_length"]             # flattened upper-tri length
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


def parse_targets_txt(path: str) -> List[Dict[str, str]]:
    """
    Parse targets.txt (TSV). Returns a list of dicts with keys:
      - index (int), identifier (str), description (str)
    Keeps file order.
    """
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # normalize keys present in your file: index, identifier, description
            idx = int(row["index"])
            ident = row["identifier"]
            desc = row.get("description", "")
            out.append({"index": idx, "identifier": ident, "description": desc})
    # sort by index just in case
    out.sort(key=lambda r: r["index"])
    return out


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
    model.restore(model_h5)
    return model


def from_upper_triu(vector_repr: np.ndarray, matrix_len: int, num_diags: int) -> np.ndarray:
    """
    Reconstruct a symmetric matrix (NaNs along masked diagonals) from an
    upper-triangular vector that starts at offset `num_diags`.
    """
    z = np.zeros((matrix_len, matrix_len), dtype=np.float32)
    triu_tup = np.triu_indices(matrix_len, num_diags)
    z[triu_tup] = vector_repr
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)
    return z + z.T


def fasta_batches(fasta_path: str, batch_size: int) -> Iterable[Tuple[List[str], List[Tuple[str, str]]]]:
    """
    Yields (headers, records) where:
      - headers: list[str] FASTA headers for this batch
      - records: list[(header, seq)] for this batch
    """
    with pysam.FastxFile(fasta_path) as fh:
        headers = []
        recs = []
        for rec in fh:
            headers.append(rec.name)
            recs.append((rec.name, rec.sequence.upper()))
            if len(recs) == batch_size:
                yield headers, recs
                headers, recs = [], []
        if headers:
            yield headers, recs


def encode_and_stack(records: List[Tuple[str, str]], seq_length: int) -> np.ndarray:
    """
    Convert list of (header, seq) to a batch array [B, seq_length, 4].
    Raises if any sequence length != seq_length.
    """
    arrs = []
    for hdr, seq in records:
        if len(seq) != seq_length:
            raise ValueError(
                f"Sequence '{hdr}' length {len(seq)} != expected {seq_length}. "
                "All input sequences must match the model's seq_length."
            )
        arrs.append(dna_io.dna_1hot(seq))
    return np.stack(arrs, axis=0).astype(np.float32)


# -----------------------------
# CLI
# -----------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("fasta", type=click.Path(exists=True, dir_okay=False))
@click.option("--batch-size", type=int, default=4, show_default=True, help="Batch size for inference.")
@click.option("--out", "out_npz", default="akita_ensemble_outputs.npz", show_default=True, help="Output NPZ path.")
@click.option("--models-glob", default=MODELS_GLOB, show_default=True, help="Glob for human model0_best.h5 files.")
@click.option("--stats", "stats_path", default=DATA_STATS, show_default=True, help="Path to statistics.json.")
@click.option("--targets-file", default=TARGETS_TXT, show_default=True, help="Path to human targets.txt.")
@click.option("--targets", multiple=True, help="Target head identifiers to export (by name). If omitted, uses ALL heads.")
@click.option("--list-heads", is_flag=True, help="List available heads from targets.txt and exit.")
def main(fasta, batch_size, out_npz, models_glob, stats_path, targets_file, targets, list_heads):
    """
    Ensemble Akita v2 HUMAN models across all folds on a multi-FASTA input.

    - Reads sequences from FASTA (each must be exactly seq_length from stats).
    - Averages predictions over all discovered human models (model0_best.h5).
    - Converts chosen target heads to symmetric Hi-C matrices.
    - Saves all matrices into one compressed NPZ with metadata.
    """
    # Quiet TensorFlow logs
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

    # Parse heads
    heads = parse_targets_txt(targets_file)  # list of dicts
    if not heads:
        raise click.ClickException(f"No heads found in {targets_file}")

    # --list-heads mode
    if list_heads:
        click.echo(f"Available heads in {targets_file}:")
        for h in heads:
            click.echo(f"  [{h['index']}] {h['identifier']}  -  {h['description']}")
        return

    # Determine which heads to export
    if targets:
        # user provided identifiers
        ident2idx = {h["identifier"]: h["index"] for h in heads}
        missing = [t for t in targets if t not in ident2idx]
        if missing:
            raise click.ClickException(f"Unknown target(s): {', '.join(missing)}. "
                                       f"Use --list-heads to see available identifiers.")
        chosen = [(ident2idx[t], t) for t in targets]
    else:
        # ALL heads
        chosen = [(h["index"], h["identifier"]) for h in heads]

    chosen.sort(key=lambda x: x[0])  # keep index order
    head_indices = [idx for idx, _ in chosen]
    head_names = [name for _, name in chosen]
    click.echo("Using heads:")
    for idx, name in zip(head_indices, head_names):
        desc = next((h["description"] for h in heads if h["index"] == idx), "")
        click.echo(f"  [{idx}] {name}  -  {desc}")

    # Discover and load models
    pairs = find_models_and_params(models_glob)
    click.echo(f"Found {len(pairs)} human checkpoints:")
    for m, _ in pairs:
        click.echo(f"  - {m}")

    models = []
    for m, p in pairs:
        click.echo(f"Loading model: {m}")
        models.append(load_seqnn_model(m, p))
    click.echo(f"Loaded {len(models)} models.")

    all_headers: List[str] = []
    mats: List[np.ndarray] = []  # each entry: [H, Lc, Lc] per sequence

    # Iterate FASTA in batches
    for headers, recs in fasta_batches(fasta, batch_size):
        x = encode_and_stack(recs, seq_length)  # [B, L, 4]
        B = x.shape[0]

        # predict with each model then average: [B, T, C]
        pred_sum = None
        for model in models:
            y = model.model.predict(x, verbose=0)  # [B, target_length, num_targets]
            if pred_sum is None:
                pred_sum = y
            else:
                pred_sum += y
        ensemble = pred_sum / float(len(models))    # [B, T, C]

        # For each sequence in batch, convert selected heads to matrices
        for i, hdr in enumerate(headers):
            # build [H, Lc, Lc] for this sequence
            head_mats = []
            for h_idx in head_indices:
                flat = ensemble[i, :, h_idx].astype(np.float32)   # [T]
                mat = from_upper_triu(flat, Lc, hic_diags)        # [Lc, Lc]
                head_mats.append(mat)
            mats.append(np.stack(head_mats, axis=0))              # [H, Lc, Lc]
            all_headers.append(hdr)

        click.echo(f"Processed {len(all_headers)} sequences so far...")

    # Stack and save
    if mats:
        mats_arr = np.stack(mats, axis=0)   # [N, H, Lc, Lc]
    else:
        mats_arr = np.zeros((0, len(head_indices), Lc, Lc), dtype=np.float32)

    headers_arr = np.array(all_headers, dtype=object)
    head_names_arr = np.array(head_names, dtype=object)

    np.savez_compressed(out_npz,
                        headers=headers_arr,
                        head_names=head_names_arr,
                        mats=mats_arr)
    click.echo(f"Saved N={len(all_headers)}, H={len(head_indices)} matrices to: {out_npz}")
    click.echo("NPZ keys: 'headers' (object), 'head_names' (object), 'mats' (float32, N x H x L x L)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise click.ClickException(str(e))
