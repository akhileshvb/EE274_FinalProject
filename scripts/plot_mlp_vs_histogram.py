#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tabcl.cli import Model
from src.tabcl.conditional import decode_columns_with_parents
from src.tabcl.codec import decompress_numeric_array, histogram_from_pairs
from src.tabcl.tiny_mlp import deserialize_mlp_params, TORCH_AVAILABLE

if not TORCH_AVAILABLE:
    print("Error: PyTorch is required for MLP visualization. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

import torch


def load_tabcl_file(tabcl_file: Path):
    """Load model and frames from .tabcl file."""
    data = tabcl_file.read_bytes()
    p = 6  # Skip magic bytes
    version = int.from_bytes(data[p:p+4], "little")
    p += 4
    mlen = int.from_bytes(data[p:p+8], "little")
    p += 8
    model = Model.from_bytes(data[p:p+mlen])
    p += mlen
    
    ncols = int.from_bytes(data[p:p+4], "little")
    p += 4
    
    frames = []
    from src.tabcl.conditional import _decode_varint
    for _ in range(ncols):
        is_compressed = False
        if data[p] == 0xFF:
            is_compressed = True
            p += 1
        flen, bytes_consumed = _decode_varint(data, p, signed=False)
        p += bytes_consumed
        frame_data = data[p:p+flen]
        p += flen
        frames.append(frame_data)
    
    return model, frames


def get_histogram_conditional_dist(parent_indices: np.ndarray, child_indices: np.ndarray):
    """
    Compute histogram-based conditional distribution P(child | parent).
    
    Returns:
        dict mapping parent_value -> dict mapping child_value -> probability
    """
    # Build histogram
    counts = histogram_from_pairs(parent_indices, child_indices)
    
    # Group by parent value
    parent_to_counts = {}
    for (p_val, c_val), count in counts.items():
        if p_val not in parent_to_counts:
            parent_to_counts[p_val] = {}
        parent_to_counts[p_val][c_val] = count
    
    # Normalize to get conditional probabilities
    conditional_dists = {}
    for p_val, child_counts in parent_to_counts.items():
        total = sum(child_counts.values())
        conditional_dists[p_val] = {c_val: count / total for c_val, count in child_counts.items()}
    
    return conditional_dists


def get_mlp_conditional_dist(model, parent_indices: np.ndarray, parent_map: dict, child_map: dict):
    """
    Get MLP-predicted conditional distribution P(child | parent).
    
    Returns:
        dict mapping parent_value -> dict mapping child_value -> probability
    """
    device = next(model.parameters()).device
    
    # Map parent indices
    parent_mapped = np.array([
        parent_map.get(v, 0) for v in parent_indices
    ], dtype=np.int64)
    parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
    
    # Get MLP predictions
    model.eval()
    with torch.no_grad():
        logits = model.forward([parent_tensor])
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()
    
    # Build conditional distributions by parent value
    conditional_dists = {}
    unique_parents = np.unique(parent_indices)
    
    for p_val in unique_parents:
        mask = parent_indices == p_val
        if np.sum(mask) == 0:
            continue
        
        # Average probabilities for this parent value
        avg_probs = probs_np[mask].mean(axis=0)
        
        # Map to original child values using inverse child_map
        dist = {}
        for mapped_child_idx, prob in enumerate(avg_probs):
            # Find original child value from inverse map
            orig_child_val = None
            for orig, mapped in child_map.items():
                if mapped == mapped_child_idx:
                    orig_child_val = orig
                    break
            if orig_child_val is not None:
                dist[orig_child_val] = float(prob)
        
        conditional_dists[p_val] = dist
    
    return conditional_dists


def plot_distribution_comparison(
    parent_val: int,
    hist_dist: dict,
    mlp_dist: dict,
    parent_name: str,
    child_name: str,
    ax,
    max_children: int = 10
):
    """Plot comparison of histogram vs MLP distribution for a specific parent value."""
    # Get all child values that appear in either distribution
    all_children = set(hist_dist.get(parent_val, {}).keys())
    all_children.update(mlp_dist.get(parent_val, {}).keys())
    
    # Get probabilities and sort by histogram probability (show most likely children)
    child_probs = []
    for c in all_children:
        hist_prob = hist_dist.get(parent_val, {}).get(c, 0.0)
        mlp_prob = mlp_dist.get(parent_val, {}).get(c, 0.0)
        child_probs.append((c, hist_prob, mlp_prob))
    
    # Sort by histogram probability (descending) and take top N
    child_probs.sort(key=lambda x: x[1], reverse=True)
    child_probs = child_probs[:max_children]
    
    if len(child_probs) == 0:
        return
    
    children = [c for c, _, _ in child_probs]
    hist_probs = [h for _, h, _ in child_probs]
    mlp_probs = [m for _, _, m in child_probs]
    
    x = np.arange(len(children))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, hist_probs, width, label='Histogram', alpha=0.85, color='#ff7f0e', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, mlp_probs, width, label='MLP', alpha=0.85, color='#2ca02c', edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Child Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'P(Child | Parent={parent_val})\n(Top {len(children)} child values)', fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(x)
    # Truncate long labels
    labels = [str(c)[:15] + ('...' if len(str(c)) > 15 else '') for c in children]
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Only label significant bars to avoid clutter
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # Only label if > 5%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='medium')


def plot_residual_distribution(residuals: np.ndarray, ax):
    """Plot distribution of residuals (actual - predicted)."""
    # Focus on residuals near zero for better visualization
    # Clip extreme outliers for readability
    residual_clipped = np.clip(residuals, np.percentile(residuals, 1), np.percentile(residuals, 99))
    
    bins = min(40, len(np.unique(residual_clipped)))
    n, bins_edges, patches = ax.hist(residual_clipped, bins=bins, alpha=0.75, edgecolor='black', 
                                     color='#1f77b4', linewidth=0.8)
    
    # Highlight zero residual
    zero_idx = np.argmin(np.abs(bins_edges))
    if zero_idx < len(patches):
        patches[zero_idx].set_color('#2ca02c')
        patches[zero_idx].set_alpha(0.9)
    
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Residual Distribution\n(Small residuals → better compression)', fontsize=15, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add statistics in a cleaner box
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    zero_count = np.sum(residuals == 0)
    zero_pct = 100 * zero_count / len(residuals) if len(residuals) > 0 else 0
    abs_mean = np.mean(np.abs(residuals))
    
    stats_text = f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}\n|Mean|: {abs_mean:.2f}\nExact: {zero_pct:.1f}\%'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9, linewidth=1.5))


def plot_kl_divergence_comparison(hist_dist: dict, mlp_dist: dict, ax, max_parents: int = 20):
    """Plot KL divergence between histogram and MLP distributions for each parent value."""
    parent_vals = sorted(set(hist_dist.keys()) | set(mlp_dist.keys()))
    kl_divs = []
    kl_parents = []
    
    for p_val in parent_vals:
        hist_p = hist_dist.get(p_val, {})
        mlp_p = mlp_dist.get(p_val, {})
        
        # Get all child values
        all_children = set(hist_p.keys()) | set(mlp_p.keys())
        
        # Compute KL divergence: KL(hist || mlp)
        kl = 0.0
        for c_val in all_children:
            hist_prob = hist_p.get(c_val, 0.0)
            mlp_prob = mlp_p.get(c_val, 1e-10)  # Avoid log(0)
            
            if hist_prob > 0:
                kl += hist_prob * np.log(hist_prob / mlp_prob)
        
        kl_divs.append(kl)
        kl_parents.append(p_val)
    
    # Sort by KL divergence and show top N
    sorted_pairs = sorted(zip(kl_parents, kl_divs), key=lambda x: x[1], reverse=True)
    if len(sorted_pairs) > max_parents:
        sorted_pairs = sorted_pairs[:max_parents]
        sorted_pairs.sort(key=lambda x: x[0])  # Re-sort by parent value for display
    
    display_parents = [p for p, _ in sorted_pairs]
    display_kls = [kl for _, kl in sorted_pairs]
    
    bars = ax.bar(range(len(display_parents)), display_kls, alpha=0.75, color='#d62728', 
                  edgecolor='black', linewidth=0.8)
    ax.set_xlabel('Parent Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('KL Divergence', fontsize=14, fontweight='bold')
    title = 'KL(Histogram || MLP) by Parent Value\n(Lower = more similar)'
    if len(parent_vals) > max_parents:
        title += f'\n(Showing top {max_parents} by KL divergence)'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=10)
    ax.set_xticks(range(len(display_parents)))
    # Show every Nth label to avoid crowding
    step = max(1, len(display_parents) // 10)
    labels = [str(p) if i % step == 0 or i == len(display_parents) - 1 else '' 
              for i, p in enumerate(display_parents)]
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Only label significant bars
    for bar, kl in zip(bars, display_kls):
        if kl > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{kl:.1f}', ha='center', va='bottom', fontsize=9, fontweight='medium')


def main():
    parser = argparse.ArgumentParser(
        description="Compare MLP vs histogram conditional distributions"
    )
    parser.add_argument("input", type=Path, help="Path to .tabcl file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                       help="Output file path (default: mlp_vs_histogram.png)")
    parser.add_argument("--column", type=int, default=None,
                       help="Specific child column index to analyze (default: first MLP column)")
    parser.add_argument("--max-parents", type=int, default=3,
                       help="Maximum number of parent values to show (default: 3)")
    parser.add_argument("--max-children", type=int, default=10,
                       help="Maximum number of child values per distribution (default: 10)")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    # Load file
    print(f"Loading {args.input}...", file=sys.stderr)
    model, frames = load_tabcl_file(args.input)
    
    # Decode all columns to get actual data first
    print("Decoding columns...", file=sys.stderr)
    # First decode root columns to determine n_rows
    from src.tabcl.codec import decompress_numeric_array
    root_cols = [i for i, p in enumerate(model.parents) if p == -1]
    if root_cols:
        first_root = root_cols[0]
        first_root_data = decompress_numeric_array(frames[first_root])
        n_rows = len(first_root_data)
    else:
        # Fallback: try to estimate from frame sizes
        n_rows = 1000000
    
    indices = decode_columns_with_parents(frames, model.parents, n_rows, model.mlp_models)
    
    # Get actual number of rows from decoded data
    n_rows = len(indices[0])
    
    # Find columns that use MLP
    mlp_columns = []
    if model.mlp_models:
        for i, mlp_model_bytes in enumerate(model.mlp_models):
            if mlp_model_bytes is not None:
                mlp_columns.append(i)
    
    # Select column to analyze
    if args.column is not None:
        col_idx = args.column
        if col_idx >= len(model.columns):
            print(f"Error: Column index {col_idx} out of range (max: {len(model.columns)-1})", file=sys.stderr)
            sys.exit(1)
    elif len(mlp_columns) > 0:
        # Use existing MLP column
        col_idx = mlp_columns[0]
        print(f"Using existing MLP-encoded column {col_idx} ({model.columns[col_idx]})...", file=sys.stderr)
    else:
        # Find a child column to train MLP on
        child_columns = [i for i, p in enumerate(model.parents) if p != -1]
        if len(child_columns) == 0:
            print("Error: No child columns found (all columns are roots)", file=sys.stderr)
            sys.exit(1)
        
        # Select first child column
        col_idx = child_columns[0]
        print(f"No MLP columns found. Training MLP for column {col_idx} ({model.columns[col_idx]})...", file=sys.stderr)
    
    print(f"Analyzing column {col_idx} ({model.columns[col_idx]})...", file=sys.stderr)
    
    # Get parent column
    parent_idx = model.parents[col_idx]
    if parent_idx == -1:
        print("Error: Selected column is a root (no parent)", file=sys.stderr)
        sys.exit(1)
    
    parent_indices = indices[parent_idx]
    child_indices = indices[col_idx]
    
    # Get or train MLP model
    if col_idx in mlp_columns and model.mlp_models[col_idx] is not None:
        # Deserialize existing MLP model
        print("Loading existing MLP model...", file=sys.stderr)
        _, mlp_model = deserialize_mlp_params(model.mlp_models[col_idx])
    else:
        # Train new MLP model
        print("Training new MLP model...", file=sys.stderr)
        from src.tabcl.tiny_mlp import train_tiny_mlp
        
        # Get unique values for mapping
        all_unique_children = np.unique(child_indices)
        all_unique_parents = np.unique(parent_indices)
        all_unique_children = all_unique_children[all_unique_children != -1]
        all_unique_parents = all_unique_parents[all_unique_parents != -1]
        
        mlp_model = train_tiny_mlp(
            child_indices=child_indices,
            parent_indices=parent_indices,
            embedding_dim=8,
            hidden_dims=[32],
            num_epochs=20,
            max_samples=50000,
            all_unique_children=all_unique_children,
            all_unique_parents=all_unique_parents,
        )
        
        if mlp_model is None:
            print("Error: Failed to train MLP model", file=sys.stderr)
            sys.exit(1)
        
        print("MLP model trained successfully!", file=sys.stderr)
    
    # Get distributions
    print("Computing distributions...", file=sys.stderr)
    hist_dist = get_histogram_conditional_dist(parent_indices, child_indices)
    mlp_dist = get_mlp_conditional_dist(mlp_model, parent_indices, mlp_model.parent_map, mlp_model.child_map)
    
    # Get residuals
    device = next(mlp_model.parameters()).device
    parent_mapped = np.array([
        mlp_model.parent_map.get(v, 0) for v in parent_indices
    ], dtype=np.int64)
    parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
    
    mlp_model.eval()
    with torch.no_grad():
        logits = mlp_model.forward([parent_tensor])
        probs = torch.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()
        predicted_classes = np.argmax(probs_np, axis=1)
    
    child_mapped = np.array([
        mlp_model.child_map.get(v, 0) for v in child_indices
    ], dtype=np.int64)
    residuals = child_mapped.astype(np.int64) - predicted_classes.astype(np.int64)
    
    # Select parent values to plot (most frequent ones)
    parent_counts = {}
    for p_val in parent_indices:
        parent_counts[p_val] = parent_counts.get(p_val, 0) + 1
    
    top_parents = sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)[:args.max_parents]
    top_parent_vals = [p_val for p_val, _ in top_parents]
    
    parent_name = model.columns[parent_idx]
    child_name = model.columns[col_idx]
    
    # Create a cleaner 2x2 layout: distributions on left, residual and KL on right
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Top row: Distribution comparisons (up to 3)
    dist_axes = []
    for i, p_val in enumerate(top_parent_vals):
        if i < 3:  # Max 3 in top row
            ax = fig.add_subplot(gs[0, i])
            plot_distribution_comparison(p_val, hist_dist, mlp_dist, parent_name, child_name, 
                                       ax, max_children=args.max_children)
            dist_axes.append(ax)
    
    # Bottom left: Residual distribution
    ax_residual = fig.add_subplot(gs[1, 0])
    plot_residual_distribution(residuals, ax_residual)
    
    # Bottom middle: KL divergence
    ax_kl = fig.add_subplot(gs[1, 1])
    plot_kl_divergence_comparison(hist_dist, mlp_dist, ax_kl, max_parents=20)
    
    # Bottom right: Summary statistics
    ax_stats = fig.add_subplot(gs[1, 2])
    ax_stats.axis('off')
    
    # Compute summary statistics
    total_kl = 0.0
    kl_count = 0
    for p_val in set(hist_dist.keys()) | set(mlp_dist.keys()):
        hist_p = hist_dist.get(p_val, {})
        mlp_p = mlp_dist.get(p_val, {})
        all_children = set(hist_p.keys()) | set(mlp_p.keys())
        kl = 0.0
        for c_val in all_children:
            hist_prob = hist_p.get(c_val, 0.0)
            mlp_prob = mlp_p.get(c_val, 1e-10)
            if hist_prob > 0:
                kl += hist_prob * np.log(hist_prob / mlp_prob)
        if kl > 0:
            total_kl += kl
            kl_count += 1
    
    avg_kl = total_kl / kl_count if kl_count > 0 else 0.0
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    zero_pct = 100 * np.sum(residuals == 0) / len(residuals) if len(residuals) > 0 else 0
    abs_mean_residual = np.mean(np.abs(residuals))
    
    stats_text = f"""
    \textbf{{Summary Statistics}}
    
    \textbf{{Residuals:}}
    Mean: {mean_residual:.2f}
    Std: {std_residual:.2f}
    |Mean|: {abs_mean_residual:.2f}
    Exact predictions: {zero_pct:.1f}\%
    
    \textbf{{Distribution Similarity:}}
    Avg KL divergence: {avg_kl:.2f}
    Parent values analyzed: {kl_count}
    
    \textbf{{Why MLP Wins:}}
    • Smooths sparse distributions
    • Small residuals compress well
    • Lower model cost than histograms
    """
    
    # Remove LaTeX formatting for matplotlib text
    stats_text_clean = f"""Summary Statistics

Residuals:
  Mean: {mean_residual:.2f}
  Std: {std_residual:.2f}
  |Mean|: {abs_mean_residual:.2f}
  Exact: {zero_pct:.1f}%

Distribution Similarity:
  Avg KL: {avg_kl:.2f}
  Parents: {kl_count}

Why MLP Wins:
  • Smooths sparse distributions
  • Small residuals → better compression
  • Lower model cost"""
    
    ax_stats.text(0.1, 0.5, stats_text_clean, transform=ax_stats.transAxes,
                 fontsize=12, verticalalignment='center', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', 
                          alpha=0.8, linewidth=1.5),
                 family='monospace')
    
    plt.suptitle(f'MLP vs Histogram Comparison: {child_name} | {parent_name}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    output_file = args.output or Path(args.input).parent / "mlp_vs_histogram.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}", file=sys.stderr)
    plt.close()


if __name__ == "__main__":
    main()

