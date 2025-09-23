#!/usr/bin/env python3
"""
Akita CLI: Predict contact maps from FASTA and save [N, 448, 448] to NPZ.

Files expected (defaults can be overridden via options):
  - akita_v1_params.json
  - model_best.h5
  - akita_v1_statistics.json

Example:
  ./akita_fasta_to_npz_verbose_tqdm.py input.fa --out akita_preds.h5 --target-index 0

Notes:
  * Sequences are automatically padded with 'N' if shorter than the required length
    or center-trimmed if longer (configurable via --length-policy).
"""

import json
import math
import os
from typing import Iterator, List, Tuple

import click
import numpy as np
from cooltools.lib.numutils import set_diag
import h5py

# Akita / Basenji
from basenji import dna_io, seqnn

import tensorflow as tf
from tqdm import tqdm


# ----------------------------
# FASTA utilities (no deps)
# ----------------------------

def read_fasta(path: str) -> Iterator[Tuple[str, str]]:
    """Yield (header, sequence) tuples from a FASTA file.
    Accepts multi-line sequences; ignores empty/comment lines."""
    header = None
    seq_chunks: List[str] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks).upper().replace('U', 'T')
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, ''.join(seq_chunks).upper().replace('U', 'T')


def normalize_length(seq: str, required: int, policy: str = 'auto') -> str:
    """Return sequence with exact required length according to policy.

    policy:
      - 'auto' (default): pad with 'N' if shorter; center-trim if longer.
      - 'pad': always right-pad with 'N' if shorter; error if longer.
      - 'trim': always center-trim if longer; error if shorter.
      - 'strict': error unless already exact length.
    """
    L = len(seq)
    if L == required:
        return seq
    if policy == 'strict':
        raise ValueError(f"Sequence length {L} != required {required} (strict policy)")
    if L < required:
        if policy in ('auto', 'pad'):
            return seq + ('N' * (required - L))
        else:
            raise ValueError(f"Sequence shorter than required ({L} < {required}) under policy='{policy}'")
    # L > required
    if policy in ('auto', 'trim'):
        # center-trim
        extra = L - required
        left = extra // 2
        right = extra - left
        return seq[left: L - right]
    else:
        raise ValueError(f"Sequence longer than required ({L} > {required}) under policy='{policy}'")


# ----------------------------
# Hi-C vector -> symmetric matrix
# ----------------------------

def upper_triu_to_full(vector_repr: np.ndarray, matrix_len: int, num_diags: int) -> np.ndarray:
    """Reconstruct symmetric contact map from the Akita upper-triangular vector.

    The returned diagonals within +/- num_diags are set to NaN, matching typical Akita postprocessing.
    """
    z = np.zeros((matrix_len, matrix_len), dtype=vector_repr.dtype)
    triu_tup = np.triu_indices(matrix_len, num_diags)
    if vector_repr.shape[-1] != triu_tup[0].shape[0]:
        raise ValueError(
            f"Vector length {vector_repr.shape[-1]} doesn't match expected upper-tri entries {triu_tup[0].shape[0]}"
        )
    z[triu_tup] = vector_repr
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)
    return z + z.T


# ----------------------------
# CLI
# ----------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('fasta', type=click.Path(exists=True, dir_okay=False))
@click.option('--model-dir', default='.', show_default=True,
              help='Directory containing params/statistics/model files.')
@click.option('--params-file', default='data/akita_v1_params.json', show_default=True,
              help='Params JSON filename (within --model-dir unless absolute).')
@click.option('--stats-file', default='data/akita_v1_statistics.json', show_default=True,
              help='Dataset statistics JSON filename.')
@click.option('--weights-file', default='data/akita_v1_model_best.h5', show_default=True,
              help='Model weights file (.h5).')
@click.option('--target-index', default=0, show_default=True, type=int,
              help='Which target to extract for the contact map.')
@click.option('--batch-size', default=4, show_default=True, type=int,
              help='Batch size for model.predict.')
@click.option('--length-policy', type=click.Choice(['auto', 'pad', 'trim', 'strict']),
              default='auto', show_default=True,
              help='How to handle sequences whose length differs from the model input length.')
@click.option('--out', 'out_path', default='akita_preds.h5', show_default=True,
              help='Output HDF5 path containing dataset "preds" with shape [N, M, M].')
@click.option('--list-headers/--no-list-headers', default=False, show_default=True,
              help='Print headers as they are processed.')
@click.option('--allow-gpu/--no-allow-gpu', default=True, show_default=True,
              help='Use GPU if available (disable to force CPU).')
@click.option('--progress/--no-progress', default=True, show_default=True,
              help='Show tqdm progress bars for sequences and batches.')
def main(
    fasta: str,
    model_dir: str,
    params_file: str,
    stats_file: str,
    weights_file: str,
    target_index: int,
    batch_size: int,
    length_policy: str,
    out_path: str,
    list_headers: bool,
    allow_gpu: bool,
    progress: bool,
):
    """Run Akita on FASTA and save an NPZ with [N, M, M] predictions."""
    # Resolve paths
    if not os.path.isabs(params_file):
        params_file = os.path.join(model_dir, params_file)
    if not os.path.isabs(stats_file):
        stats_file = os.path.join(model_dir, stats_file)
    if not os.path.isabs(weights_file):
        weights_file = os.path.join(model_dir, weights_file)

    # Optional: restrict GPUs
    if not allow_gpu:
        tf.config.set_visible_devices([], 'GPU')

    # Load params & stats
    with open(params_file) as f:
        params = json.load(f)
        params_model = params['model']
    with open(stats_file) as f:
        stats = json.load(f)

    seq_length = stats['seq_length']
    pool_width = stats['pool_width']
    hic_diags = stats['diagonal_offset']
    target_crop = stats['crop_bp'] // pool_width
    target_length1 = seq_length // pool_width
    matrix_len = target_length1 - 2 * target_crop

    # Build model
    model = seqnn.SeqNN(params_model)
    model.restore(weights_file)

    # Read sequences and preprocess
    headers: List[str] = []
    batch_1hot: List[np.ndarray] = []
    mats: List[np.ndarray] = []

    total_batches = 0

    pbar_seqs = tqdm(desc="Reading & queuing", unit="seq", dynamic_ncols=True, disable=not progress, leave=False, total=sum(1 for _ in read_fasta(fasta)))
    pbar_batches = tqdm(desc="Predicting batches", unit="batch", dynamic_ncols=True, disable=not progress, leave=True, total=math.ceil(pbar_seqs.total / batch_size) if pbar_seqs.total else None)

    def flush_batch():
        nonlocal batch_1hot, mats, total_batches
        if not batch_1hot:
            return
        x = np.stack(batch_1hot, axis=0)  # [B, seqlen, 4]
        pred = model.model.predict(x, verbose=0)  # [B, T, C]
        total_batches += 1
        pbar_batches.update(1)
        # extract target vector(s) -> matrix
        # pred shape assumptions: T == stats['target_length']
        for b in range(pred.shape[0]):
            vec = pred[b, :, target_index]
            mat = upper_triu_to_full(vec, matrix_len, hic_diags)
            mats.append(mat)
        batch_1hot = []

    for header, seq in read_fasta(fasta):
        if list_headers:
            click.echo(f"Processing: {header}")
        seq_fixed = normalize_length(seq, seq_length, policy=length_policy)
        onehot = dna_io.dna_1hot(seq_fixed)
        batch_1hot.append(onehot)
        headers.append(header)
        pbar_seqs.update(1)
        if len(batch_1hot) >= batch_size:
            flush_batch()

    # final flush
    flush_batch()

    pbar_seqs.close()
    pbar_batches.close()

    preds = np.stack(mats, axis=0) if mats else np.zeros((0, matrix_len, matrix_len), dtype=np.float32)

    # Save (HDF5)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        # main tensor
        dset = f.create_dataset(
            'preds',
            data=preds,
            compression='gzip',
            compression_opts=4,
            chunks=True
        )
        # headers as variable-length UTF-8 strings
        str_dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('headers', data=np.array(headers, dtype=object), dtype=str_dt)

        # useful metadata
        f.attrs['matrix_len'] = int(matrix_len)
        f.attrs['target_index'] = int(target_index)
        f.attrs['seq_length'] = int(seq_length)
        f.attrs['pool_width'] = int(pool_width)
        f.attrs['diagonal_offset'] = int(hic_diags)

    click.echo(f"Saved predictions (HDF5): {out_path}")
    click.echo(f"Sequences processed: {len(headers)}; Batches: {total_batches}")
    click.echo(f"Shape: {preds.shape} (N, {matrix_len}, {matrix_len})")



if __name__ == '__main__':
    main()
