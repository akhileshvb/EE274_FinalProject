import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Add parent directory to path to import tabcl
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tabcl.cli import Model, _load_csv
from src.tabcl.mi import compute_empirical_mi
from src.tabcl.codec import mdl_cost_fn_openzl


def plot_compression_gain_vs_mi(tabcl_file: Path, csv_file: Path = None, output_file: Path = None, delimiter: str = ","):
    """
    Plot compression gain vs mutual information for each parent-child pair.
    
    Args:
        tabcl_file: Path to .tabcl file
        csv_file: Path to original CSV file (if None, tries to infer from tabcl_file)
        output_file: Path to save the plot (if None, displays interactively)
        delimiter: CSV delimiter
    """
    if not tabcl_file.exists():
        print(f"Error: {tabcl_file} not found", file=sys.stderr)
        sys.exit(1)
    
    # Load model from .tabcl file
    try:
        data = tabcl_file.read_bytes()
        p = 6  # Skip magic bytes
        version = int.from_bytes(data[p:p+4], "little")
        p += 4
        mlen = int.from_bytes(data[p:p+8], "little")
        p += 8
        model = Model.from_bytes(data[p:p+mlen])
    except Exception as e:
        print(f"Error: Could not load model from {tabcl_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load original CSV data
    if csv_file is None:
        # Try to infer CSV file path
        csv_file = tabcl_file.with_suffix('.csv')
        if not csv_file.exists():
            # Try other common extensions
            for ext in ['.csv', '.txt']:
                csv_file = tabcl_file.with_suffix(ext)
                if csv_file.exists():
                    break
    
    if csv_file is None or not csv_file.exists():
        print(f"Error: Could not find original CSV file for {tabcl_file}", file=sys.stderr)
        print(f"Please provide --csv-file argument", file=sys.stderr)
        sys.exit(1)
    
    # Load CSV data
    try:
        df = _load_csv(csv_file, delimiter)
        table = df.astype(object).to_numpy()
    except Exception as e:
        print(f"Error: Could not load CSV from {csv_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    n_rows, n_cols = table.shape
    
    # Find all parent-child pairs
    parent_child_pairs = []
    for child_idx in range(n_cols):
        parent_idx = model.parents[child_idx]
        if parent_idx != -1:  # Has a parent
            parent_child_pairs.append((parent_idx, child_idx))
    
    if not parent_child_pairs:
        print("Warning: No parent-child pairs found (all columns are roots)", file=sys.stderr)
        return
    
    # Calculate MI and compression gain for each pair
    mi_values = []
    compression_gains = []
    edge_weights = []
    pair_labels = []
    
    # Get edge weights from model
    edge_weight_dict = {}
    for edge in model.edges:
        if len(edge) >= 3:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            edge_weight_dict[(u, v)] = w
            edge_weight_dict[(v, u)] = w  # Undirected
    
    for parent_idx, child_idx in parent_child_pairs:
        # Calculate MI between parent and child
        parent_data = table[:, parent_idx]
        child_data = table[:, child_idx]
        
        try:
            mi_nats = compute_empirical_mi(parent_data, child_data)
            mi_bits = mi_nats / np.log(2.0)
        except Exception as e:
            print(f"Warning: Could not calculate MI for pair ({parent_idx}, {child_idx}): {e}", file=sys.stderr)
            continue
        
        # Get edge weight (this represents the compression benefit)
        edge_key = (parent_idx, child_idx)
        if edge_key in edge_weight_dict:
            edge_weight = edge_weight_dict[edge_key]
        else:
            # Edge might be in reverse order or not in edges (shouldn't happen)
            edge_weight = edge_weight_dict.get((child_idx, parent_idx), 0.0)
        
        # Calculate compression gain using information-theoretic metrics
        # Since edge weights are quantized and may be the same, we need a metric that shows variation
        # Options:
        # 1. Use MI directly (but that's on x-axis)
        # 2. Use entropy-based metrics that vary
        # 3. Calculate theoretical compression ratio
        
        try:
            # Calculate entropies
            # Child entropy: H(X_child)
            child_unique, child_counts = np.unique(child_data, return_counts=True)
            child_probs = child_counts / len(child_data)
            child_entropy_bits = -np.sum(child_probs * np.log2(child_probs + 1e-10))
            
            # Parent entropy: H(X_parent)
            parent_unique, parent_counts = np.unique(parent_data, return_counts=True)
            parent_probs = parent_counts / len(parent_data)
            parent_entropy_bits = -np.sum(parent_probs * np.log2(parent_probs + 1e-10))
            
            # Joint entropy: H(X_parent, X_child)
            # Build joint distribution
            joint_counts = {}
            for i in range(len(parent_data)):
                pair = (parent_data[i], child_data[i])
                joint_counts[pair] = joint_counts.get(pair, 0) + 1
            joint_total = sum(joint_counts.values())
            joint_probs = np.array([count / joint_total for count in joint_counts.values()])
            joint_entropy_bits = -np.sum(joint_probs * np.log2(joint_probs + 1e-10))
            
            # Conditional entropy: H(X_child | X_parent) = H(X_parent, X_child) - H(X_parent)
            conditional_entropy_bits = joint_entropy_bits - parent_entropy_bits
            
            # Compression gain metrics:
            # 1. Information gain per row: I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
            #    This is already calculated as mi_bits, but we can use it for y-axis too
            # 2. Compression ratio: independent_cost / conditional_cost
            #    Independent: n * H(X_child) bits
            #    Conditional: n * H(X_child | X_parent) bits
            #    Ratio = H(X_child) / H(X_child | X_parent)
            
            # Use compression ratio as the metric
            # Higher ratio = more compression benefit
            if conditional_entropy_bits > 0:
                compression_ratio = child_entropy_bits / conditional_entropy_bits
            else:
                compression_ratio = 0.0
            
            # Alternative: Use the entropy reduction percentage
            # (H(X_child) - H(X_child | X_parent)) / H(X_child) = I(X;Y) / H(X_child)
            if child_entropy_bits > 0:
                entropy_reduction = mi_bits / child_entropy_bits
            else:
                entropy_reduction = 0.0
            
            # Use compression ratio as y-axis (shows how much better conditional encoding is)
            compression_gain = compression_ratio
            
        except Exception as e:
            print(f"Warning: Could not calculate entropies for pair ({parent_idx}, {child_idx}): {e}", file=sys.stderr)
            # Fallback: use MI as compression gain (shows correlation)
            compression_gain = mi_bits
        
        mi_values.append(mi_bits)
        compression_gains.append(compression_gain)
        edge_weights.append(edge_weight)
        pair_labels.append(f"C{parent_idx}→C{child_idx}")
    
    if not mi_values:
        print("Error: No valid MI values calculated", file=sys.stderr)
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(mi_values, compression_gains, s=150, alpha=0.6, 
                        c=edge_weights, cmap='viridis', edgecolors='black', linewidths=1.5)
    
    # Add colorbar for edge weights
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Edge Weight (bits)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Add labels for each point (if not too many)
    if len(mi_values) <= 30:
        for i, label in enumerate(pair_labels):
            ax.annotate(label, (mi_values[i], compression_gains[i]), 
                       fontsize=9, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    # Add trend line
    if len(mi_values) > 1:
        z = np.polyfit(mi_values, compression_gains, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(mi_values), max(mi_values), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.3f}')
        ax.legend(fontsize=12)
    
    ax.set_xlabel('Mutual Information I(X_parent; X_child) (bits)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Compression Ratio (H(X_child) / H(X_child|X_parent))', fontsize=16, fontweight='bold')
    ax.set_title(f'Compression Gain vs Mutual Information\n{tabcl_file.stem}', 
                fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    # Add statistics text
    correlation = np.corrcoef(mi_values, compression_gains)[0, 1] if len(mi_values) > 1 else 0
    stats_text = f'Pairs: {len(mi_values)}\nCorrelation: {correlation:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Compression gain plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot compression gain vs mutual information for parent-child column pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot compression gain vs MI
  python3 scripts/plot_compression_gain.py datasets/forest/covtype.csv.tabcl -o compression_gain.png
  
  # Specify CSV file explicitly
  python3 scripts/plot_compression_gain.py file.tabcl --csv-file file.csv -o plot.png
        """
    )
    parser.add_argument("tabcl_file", type=Path, help="Path to .tabcl file")
    parser.add_argument("--csv-file", type=Path, default=None,
                       help="Path to original CSV file (if not specified, tries to infer)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                       help="Output file path (if not specified, displays interactively)")
    parser.add_argument("--delimiter", type=str, default=",",
                       help="CSV delimiter (default: ',')")
    
    args = parser.parse_args()
    
    plot_compression_gain_vs_mi(
        args.tabcl_file,
        csv_file=args.csv_file,
        output_file=args.output,
        delimiter=args.delimiter
    )


if __name__ == "__main__":
    main()

