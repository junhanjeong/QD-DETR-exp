"""
사용법

python supplementary/visualize_gate_logs.py \
  --input /workspace/QD-DETR-exp/results/tmp/gate_logs_val_Nall.jsonl \
  --num-samples 50
"""

#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn aesthetic
sns.set_theme(style="whitegrid")


def read_jsonl_head(path: str, n: int) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n >= 0 and i >= n:
                break
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def pool_gate_value(arr_like: Any) -> float:
    # arr_like is nested list of numbers with arbitrary shape; reduce to scalar by mean
    arr = np.array(arr_like, dtype=np.float32)
    return float(arr.mean())


def build_section_series(samples: List[Dict[str, Any]]):
    if not samples:
        return [], {}, []
    num_layers = len(samples[0].get('layer_gates', []))
    gating_type = samples[0].get('gating_type', 'unknown')
    # Initialize dict: (layer_idx, gate_name) -> list of values per sample
    sections = {}
    for li in range(num_layers):
        for gname in ('gate_mha', 'gate_ffn'):
            sections[(li, gname)] = []

    qids = []
    for s in samples:
        qids.append(s.get('qid'))
        layer_gates = s.get('layer_gates', [])
        # For safety if some samples have different layer count
        for li in range(min(num_layers, len(layer_gates))):
            lg = layer_gates[li]
            for gname in ('gate_mha', 'gate_ffn'):
                val = pool_gate_value(lg[gname])
                sections[(li, gname)].append(val)
        # If a sample had fewer layers, pad with nan to keep lengths aligned
        for li in range(len(layer_gates), num_layers):
            for gname in ('gate_mha', 'gate_ffn'):
                sections[(li, gname)].append(np.nan)

    return qids, sections, gating_type


def plot_section_series(qids: List[Any], sections: Dict[Any, List[float]], gating_type: str, out_path: str):
    """Two heatmaps (left=MHA, right=FFN). Y-axis=layers, X-axis=samples, colors represent gate values."""
    if not sections:
        return None
    num_layers = max(li for (li, _), _ in sections.items()) + 1 if sections else 0
    num_samples = len(qids)
    # Build matrices
    mha_mat = np.vstack([np.array(sections[(li, 'gate_mha')], dtype=np.float32) for li in range(num_layers)])
    ffn_mat = np.vstack([np.array(sections[(li, 'gate_ffn')], dtype=np.float32) for li in range(num_layers)])
    # Common color scale
    vmin = np.nanmin(np.concatenate([mha_mat, ffn_mat]))
    vmax = np.nanmax(np.concatenate([mha_mat, ffn_mat]))
    # Use symmetric range if values cross zero
    if vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    fig, axes = plt.subplots(1, 2, figsize=(max(10, num_samples * 0.25), 2.5 + num_layers * 0.4), sharey=True)
    sns.heatmap(mha_mat, ax=axes[0], cmap='coolwarm', vmin=vmin, vmax=vmax, cbar=True,
                yticklabels=[str(li) for li in range(num_layers)], xticklabels=False)
    axes[0].set_title('MHA')
    axes[0].set_ylabel('layer')
    sns.heatmap(ffn_mat, ax=axes[1], cmap='coolwarm', vmin=vmin, vmax=vmax, cbar=True,
                yticklabels=[str(li) for li in range(num_layers)], xticklabels=False)
    axes[1].set_title('FFN')
    fig.suptitle(f"Gate heatmaps (gating_type={gating_type})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_top_bottom(qids: List[Any], sections: Dict[Any, List[float]], out_path: str, top_k: int = 10):
    # Per-sample mean across all sections
    if not qids:
        return None
    # Stack shape: (num_sections, num_samples)
    all_series = np.array([sections[k] for k in sorted(sections.keys())], dtype=np.float32)
    per_sample_mean = np.nanmean(all_series, axis=0)

    idx_sorted = np.argsort(per_sample_mean)
    bottom_idx = idx_sorted[:top_k]
    top_idx = idx_sorted[-top_k:][::-1]

    def bar_plot(indices, title, ax):
        labels = [str(qids[i]) for i in indices]
        values = per_sample_mean[indices]
        ax.bar(np.arange(len(indices)), values, color='#4C78A8')
        ax.set_xticks(np.arange(len(indices)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('mean gate')
        ax.set_title(title)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    bar_plot(top_idx, f'Top {top_k} samples by mean gate', axes[0])
    bar_plot(bottom_idx, f'Bottom {top_k} samples by mean gate', axes[1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_layerwise_means(sections: Dict[Any, List[float]], gating_type: str, out_path: str):
    """Per layer averages across samples; grouped bars for MHA/FFN with annotations and 'Layer i' labels."""
    if not sections:
        return None
    layer_indices = sorted({li for (li, _), _ in sections.items()})
    x_labels = [f'Layer {li}' for li in layer_indices]
    x = np.arange(len(layer_indices))
    mha_means, ffn_means = [], []
    for li in layer_indices:
        mha_vals = np.array(sections[(li, 'gate_mha')], dtype=np.float32)
        ffn_vals = np.array(sections[(li, 'gate_ffn')], dtype=np.float32)
        mha_means.append(float(np.nanmean(mha_vals)))
        ffn_means.append(float(np.nanmean(ffn_vals)))

    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(layer_indices)*1.5), 5))
    # Use seaborn barplot-like color palette
    palette = sns.color_palette("Set2", 2)
    # Draw bars
    bars1 = ax.bar(x - width/2, mha_means, width, label='MHA', color=palette[0])
    bars2 = ax.bar(x + width/2, ffn_means, width, label='FFN', color=palette[1])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('mean gate')
    ax.set_title(f'Layer-wise mean gates (gating_type={gating_type})')
    ax.legend()
    # Annotations
    def annotate(bars):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2., h, f"{h:.3f}", ha='center', va='bottom', fontsize=8)
    annotate(bars1)
    annotate(bars2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def plot_samples_layer_aggregated(qids: List[Any], sections: Dict[Any, List[float]], out_path: str):
    """For each sample, average across layers per gate type, then heatmap with y=[MHA, FFN] and color as value."""
    if not qids:
        return None
    layer_indices = sorted({li for (li, _), _ in sections.items()})
    num_samples = len(qids)
    mha_per_sample = np.zeros(num_samples, dtype=np.float32)
    ffn_per_sample = np.zeros(num_samples, dtype=np.float32)
    for i in range(num_samples):
        mha_vals = [sections[(li, 'gate_mha')][i] for li in layer_indices]
        ffn_vals = [sections[(li, 'gate_ffn')][i] for li in layer_indices]
        mha_per_sample[i] = np.nanmean(np.array(mha_vals, dtype=np.float32))
        ffn_per_sample[i] = np.nanmean(np.array(ffn_vals, dtype=np.float32))

    mat = np.vstack([mha_per_sample, ffn_per_sample])  # shape (2, num_samples)
    # Color scale symmetric if crossing zero
    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    if vmin < 0 < vmax:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m
    fig, ax = plt.subplots(figsize=(max(10, num_samples * 0.25), 3.5))
    sns.heatmap(mat, ax=ax, cmap='coolwarm', vmin=vmin, vmax=vmax, cbar=True,
                yticklabels=['MHA', 'FFN'], xticklabels=False)
    # Sparse x labels for qids (optional small axis below)
    step = max(1, num_samples // 20)
    ax.set_xlabel('samples (qid order)')
    ax.set_title('Per-sample mean over layers (color-coded)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Visualize gate logs from JSONL')
    parser.add_argument('--input', required=True, help='Path to gate_logs_*.jsonl')
    parser.add_argument('--num-samples', type=int, default=-1, help='Use first N samples (-1=all)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save figures (default: input dir)')
    parser.add_argument('--show', action='store_true', help='Show figures interactively')
    parser.add_argument('--topk', type=int, default=10, help='Top/Bottom K for mean gate plots')
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    samples = read_jsonl_head(args.input, args.num_samples)
    qids, sections, gating_type = build_section_series(samples)

    sec_path = os.path.join(out_dir, 'gate_sections.png')
    tb_path = os.path.join(out_dir, 'gate_top_bottom.png')
    lw_path = os.path.join(out_dir, 'gate_layerwise_means.png')
    sl_path = os.path.join(out_dir, 'gate_samples_layermean.png')
    plot_section_series(qids, sections, gating_type, sec_path)
    plot_top_bottom(qids, sections, tb_path, top_k=args.topk)
    plot_layerwise_means(sections, gating_type, lw_path)
    plot_samples_layer_aggregated(qids, sections, sl_path)

    print("\n".join([
        f"Saved: {sec_path}",
        f"Saved: {tb_path}",
        f"Saved: {lw_path}",
        f"Saved: {sl_path}",
    ]))
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
