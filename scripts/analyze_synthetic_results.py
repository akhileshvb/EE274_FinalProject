"""
Analyze the results from the synthetic MI experiment.
Computes actual MI values and compares with compression gains.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tabcl.cli import Model, _load_csv
from src.tabcl.mi import compute_empirical_mi


def analyze_synthetic_results(csv_file: Path, tabcl_file: Path):
    """
    Analyze synthetic data: compute actual MI and compare with compression gains.
    """
    # Load data
    print(f"Loading data from {csv_file}...")
    df = _load_csv(csv_file, ",")
    table = df.astype(object).to_numpy()
    n_rows, n_cols = table.shape
    
    # Load model
    print(f"Loading model from {tabcl_file}...")
    data = tabcl_file.read_bytes()
    p = 6  # Skip magic bytes
    version = int.from_bytes(data[p:p+4], "little")
    p += 4
    mlen = int.from_bytes(data[p:p+8], "little")
    p += 8
    model = Model.from_bytes(data[p:p+mlen])
    
    # Find parent-child pairs
    parent_child_pairs = []
    for child_idx in range(n_cols):
        parent_idx = model.parents[child_idx]
        if parent_idx != -1:
            parent_child_pairs.append((parent_idx, child_idx))
    
    print(f"Found {len(parent_child_pairs)} parent-child pairs")
    
    # Get edge weights
    edge_weight_dict = {}
    for edge in model.edges:
        if len(edge) >= 3:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            edge_weight_dict[(u, v)] = w
            edge_weight_dict[(v, u)] = w
    
    # Compute actual MI and compression metrics
    results = []
    for parent_idx, child_idx in parent_child_pairs:
        parent_data = table[:, parent_idx]
        child_data = table[:, child_idx]
        
        # Compute actual MI
        try:
            mi_nats = compute_empirical_mi(parent_data, child_data)
            mi_bits = mi_nats / np.log(2.0)
        except Exception as e:
            print(f"Warning: Could not calculate MI for pair ({parent_idx}, {child_idx}): {e}")
            continue
        
        # Get edge weight
        edge_key = (parent_idx, child_idx)
        edge_weight = edge_weight_dict.get(edge_key, edge_weight_dict.get((child_idx, parent_idx), 0.0))
        
        # Calculate compression gain (savings per row)
        savings_per_row = edge_weight / n_rows if n_rows > 0 else 0.0
        
        # Calculate child entropy (for normalization)
        child_unique, child_counts = np.unique(child_data, return_counts=True)
        child_probs = child_counts / len(child_data)
        child_entropy_bits = -np.sum(child_probs * np.log2(child_probs + 1e-10))
        
        # Relative savings
        relative_savings = savings_per_row / child_entropy_bits if child_entropy_bits > 0 else 0.0
        
        results.append({
            'parent_idx': parent_idx,
            'child_idx': child_idx,
            'mi_bits': mi_bits,
            'edge_weight': edge_weight,
            'savings_per_row': savings_per_row,
            'child_entropy': child_entropy_bits,
            'relative_savings': relative_savings,
        })
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nMI (bits):")
    print(f"  Mean: {results_df['mi_bits'].mean():.3f}")
    print(f"  Std:  {results_df['mi_bits'].std():.3f}")
    print(f"  Min:  {results_df['mi_bits'].min():.3f}")
    print(f"  Max:  {results_df['mi_bits'].max():.3f}")
    
    print(f"\nCompression Savings per Row (bits):")
    print(f"  Mean: {results_df['savings_per_row'].mean():.6f}")
    print(f"  Std:  {results_df['savings_per_row'].std():.6f}")
    print(f"  Min:  {results_df['savings_per_row'].min():.6f}")
    print(f"  Max:  {results_df['savings_per_row'].max():.6f}")
    
    print(f"\nRelative Savings (fraction of child entropy):")
    print(f"  Mean: {results_df['relative_savings'].mean():.3f}")
    print(f"  Std:  {results_df['relative_savings'].std():.3f}")
    print(f"  Min:  {results_df['relative_savings'].min():.3f}")
    print(f"  Max:  {results_df['relative_savings'].max():.3f}")
    
    # Correlation analysis
    correlation = np.corrcoef(results_df['mi_bits'], results_df['savings_per_row'])[0, 1]
    relative_correlation = np.corrcoef(results_df['mi_bits'], results_df['relative_savings'])[0, 1]
    
    print(f"\nCorrelations:")
    print(f"  MI vs Savings per Row: {correlation:.3f}")
    print(f"  MI vs Relative Savings: {relative_correlation:.3f}")
    
    # Save results
    output_file = csv_file.parent / "analysis_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze synthetic MI experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv-file", type=Path, required=True,
                       help="Path to synthetic CSV file")
    parser.add_argument("--tabcl-file", type=Path, required=True,
                       help="Path to compressed .tabcl file")
    
    args = parser.parse_args()
    
    if not args.csv_file.exists():
        print(f"Error: {args.csv_file} not found", file=sys.stderr)
        sys.exit(1)
    
    if not args.tabcl_file.exists():
        print(f"Error: {args.tabcl_file} not found", file=sys.stderr)
        sys.exit(1)
    
    analyze_synthetic_results(args.csv_file, args.tabcl_file)


if __name__ == "__main__":
    main()

