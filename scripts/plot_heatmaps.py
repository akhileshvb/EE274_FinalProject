#!/usr/bin/env python3
"""
Generate heatmap visualizations for tabcl compression analysis.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import base64
import subprocess
import tempfile

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
except ImportError:
    print("Error: matplotlib, numpy, seaborn, and pandas are required. Install with: pip install matplotlib numpy seaborn pandas", file=sys.stderr)
    sys.exit(1)

# Import tabcl to read model data
try:
    from tabcl.cli import Model
    from tabcl.mi import compute_empirical_mi, compute_hashed_mi
except ImportError:
    print("Error: tabcl module not found. Make sure you're in the project directory.", file=sys.stderr)
    sys.exit(1)


def load_results(csv_file: Path) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value and value != "None" and key not in ["dataset", "input_file"]:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = None
                elif value == "None" or not value:
                    row[key] = None
            results.append(row)
    return results


def extract_model_stats(tabcl_file: Path) -> Optional[Dict]:
    """Extract statistics from a .tabcl file."""
    try:
        data = tabcl_file.read_bytes()
        if len(data) < 6 or data[:6] != b"TABCL\x00":
            return None
        
        p = 6
        version = int.from_bytes(data[p:p+4], "little")
        p += 4
        mlen = int.from_bytes(data[p:p+8], "little")
        p += 8
        
        if version >= 3:
            model = Model.from_bytes(data[p:p+mlen])
            return {
                "model": model,
                "n_cols": len(model.columns),
                "n_edges": len(model.edges),
                "edges": model.edges,
            }
    except Exception as e:
        print(f"Warning: Could not extract stats from {tabcl_file}: {e}", file=sys.stderr)
        return None
    return None


def plot_mi_heatmap(csv_file: Path, dataset_name: str, output_file: Path):
    """Plot heatmap of mutual information between all column pairs."""
    base_dir = csv_file.parent
    results = load_results(csv_file)
    
    # Find the dataset
    dataset_info = None
    for r in results:
        if r["dataset"] == dataset_name:
            dataset_info = r
            break
    
    if not dataset_info:
        print(f"Error: Dataset {dataset_name} not found in CSV", file=sys.stderr)
        return
    
    input_file = base_dir / Path(dataset_info["input_file"])
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found", file=sys.stderr)
        return
    
    # Load the CSV to compute MI
    try:
        import pandas as pd
        df = pd.read_csv(input_file, header=None, dtype=str, engine='python', on_bad_lines='skip')
        n_cols = min(df.shape[1], 20)  # Limit to 20 columns for visualization
        df = df.iloc[:, :n_cols]
        
        # Compute MI matrix
        mi_matrix = np.zeros((n_cols, n_cols))
        
        print(f"Computing MI matrix for {n_cols} columns...", file=sys.stderr)
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                x = df.iloc[:, i].values
                y = df.iloc[:, j].values
                try:
                    # Use hashed MI for speed
                    mi = compute_hashed_mi(x, y, num_buckets=4096, row_sample=min(10000, len(x)))
                    mi_matrix[i, j] = mi
                    mi_matrix[j, i] = mi  # Symmetric
                except Exception as e:
                    print(f"Warning: Could not compute MI for columns {i}, {j}: {e}", file=sys.stderr)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use seaborn for better heatmap
        sns.heatmap(mi_matrix, annot=False, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Mutual Information (nats)'},
                   square=True, linewidths=0.5, ax=ax)
        
        ax.set_xlabel("Column Index", fontsize=12)
        ax.set_ylabel("Column Index", fontsize=12)
        ax.set_title(f"Mutual Information Heatmap: {dataset_name}\n"
                    f"Shows dependency structure between columns", 
                    fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"MI heatmap saved to {output_file}", file=sys.stderr)
        plt.close()
        
    except Exception as e:
        print(f"Error creating MI heatmap: {e}", file=sys.stderr)


def plot_edge_weight_heatmap(csv_file: Path, dataset_name: str, output_file: Path):
    """Plot heatmap of edge weights (from forest) between column pairs."""
    base_dir = csv_file.parent
    results = load_results(csv_file)
    
    # Find the dataset
    dataset_info = None
    for r in results:
        if r["dataset"] == dataset_name:
            dataset_info = r
            break
    
    if not dataset_info:
        print(f"Error: Dataset {dataset_name} not found in CSV", file=sys.stderr)
        return
    
    input_file = base_dir / Path(dataset_info["input_file"])
    tabcl_file = base_dir / Path(dataset_info["input_file"]).with_suffix(".tabcl")
    
    # If .tabcl file doesn't exist, try to create it
    if not tabcl_file.exists():
        print(f"Warning: {tabcl_file} not found. Attempting to compress file...", file=sys.stderr)
        if not input_file.exists():
            print(f"Error: Input file {input_file} not found", file=sys.stderr)
            return
        
        try:
            # Determine MI mode based on file size
            file_size_mb = input_file.stat().st_size / (1024 * 1024)
            mi_mode = "hashed" if file_size_mb > 1.0 else "exact"
            
            # Compress the file
            cmd = ["python3", "-m", "tabcl.cli", "compress",
                  "--input", str(input_file),
                  "--output", str(tabcl_file),
                  "--delimiter", ",",
                  "--mi-mode", mi_mode]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0 or not tabcl_file.exists():
                print(f"Error: Failed to compress {input_file}: {result.stderr}", file=sys.stderr)
                return
            print(f"Successfully compressed {input_file}", file=sys.stderr)
        except Exception as e:
            print(f"Error: Could not compress {input_file}: {e}", file=sys.stderr)
            return
    
    stats = extract_model_stats(tabcl_file)
    if not stats:
        return
    
    model = stats["model"]
    n_cols = stats["n_cols"]
    
    # Create weight matrix
    weight_matrix = np.zeros((n_cols, n_cols))
    
    for u, v, w in model.edges:
        if u < n_cols and v < n_cols:
            weight_matrix[u, v] = w
            weight_matrix[v, u] = w  # Symmetric
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(weight_matrix, annot=False, fmt='.2f', cmap='viridis',
               cbar_kws={'label': 'Edge Weight (n*I(X;Y) - MDL_cost)'},
               square=True, linewidths=0.5, ax=ax, vmin=0)
    
    ax.set_xlabel("Column Index", fontsize=12)
    ax.set_ylabel("Column Index", fontsize=12)
    ax.set_title(f"Forest Edge Weight Heatmap: {dataset_name}\n"
                f"Shows selected dependencies in MDL-weighted forest ({stats['n_edges']} edges)",
                fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Edge weight heatmap saved to {output_file}", file=sys.stderr)
    plt.close()


def plot_parameter_sweep_heatmap(csv_file: Path, dataset_name: str, output_file: Path):
    """Plot heatmap of compression ratio vs algorithm parameters."""
    base_dir = csv_file.parent
    results = load_results(csv_file)
    
    # Find the dataset
    dataset_info = None
    for r in results:
        if r["dataset"] == dataset_name:
            dataset_info = r
            break
    
    if not dataset_info:
        print(f"Error: Dataset {dataset_name} not found in CSV", file=sys.stderr)
        return
    
    input_file = base_dir / Path(dataset_info["input_file"])
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found", file=sys.stderr)
        return
    
    # Parameter ranges to test
    rare_thresholds = [1, 2, 5, 10, 20]
    mi_buckets = [1024, 2048, 4096, 8192, 16384]
    
    print(f"Running parameter sweep for {dataset_name}...", file=sys.stderr)
    print("This may take a while...", file=sys.stderr)
    
    ratio_matrix = np.zeros((len(rare_thresholds), len(mi_buckets)))
    
    for i, rare_thresh in enumerate(rare_thresholds):
        for j, buckets in enumerate(mi_buckets):
            try:
                # Run compression with these parameters
                with tempfile.NamedTemporaryFile(suffix='.tabcl', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                
                cmd = ["python3", "-m", "tabcl.cli", "compress",
                      "--input", str(input_file),
                      "--output", str(tmp_path),
                      "--delimiter", ",",
                      "--rare-threshold", str(rare_thresh),
                      "--mi-buckets", str(buckets),
                      "--mi-mode", "hashed"]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and tmp_path.exists():
                    compressed_size = tmp_path.stat().st_size
                    original_size = input_file.stat().st_size
                    ratio = original_size / compressed_size if compressed_size > 0 else 0
                    ratio_matrix[i, j] = ratio
                    tmp_path.unlink()
                else:
                    ratio_matrix[i, j] = 0
                    
            except Exception as e:
                print(f"Warning: Failed for rare_threshold={rare_thresh}, mi_buckets={buckets}: {e}", file=sys.stderr)
                ratio_matrix[i, j] = 0
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(ratio_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=mi_buckets, yticklabels=rare_thresholds,
               cbar_kws={'label': 'Compression Ratio'},
               linewidths=0.5, ax=ax)
    
    ax.set_xlabel("MI Buckets", fontsize=12)
    ax.set_ylabel("Rare Threshold", fontsize=12)
    ax.set_title(f"Compression Ratio vs Parameters: {dataset_name}\n"
                f"Shows sensitivity to rare_threshold and mi_buckets",
                fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Parameter sweep heatmap saved to {output_file}", file=sys.stderr)
    plt.close()


def plot_dataset_characteristics_heatmap(csv_file: Path, output_file: Path):
    """Plot heatmap of compression ratio vs dataset characteristics."""
    results = load_results(csv_file)
    base_dir = csv_file.parent
    
    # Collect dataset characteristics
    dataset_chars = []
    
    for r in results:
        dataset = r["dataset"]
        input_file = base_dir / Path(r["input_file"])
        
        if not input_file.exists():
            continue
        
        try:
            import pandas as pd
            df = pd.read_csv(input_file, header=None, dtype=str, engine='python', on_bad_lines='skip')
            
            n_rows = len(df)
            n_cols = df.shape[1]
            file_size_mb = input_file.stat().st_size / (1024 * 1024)
            
            # Count numeric vs categorical (approximate)
            numeric_count = 0
            for col in df.columns:
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_count += 1
                except:
                    pass
            
            categorical_count = n_cols - numeric_count
            
            tabcl_ratio = r.get("tabcl_ratio", 0)
            
            dataset_chars.append({
                "dataset": dataset,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "file_size_mb": file_size_mb,
                "numeric_ratio": numeric_count / n_cols if n_cols > 0 else 0,
                "tabcl_ratio": tabcl_ratio,
            })
        except Exception as e:
            print(f"Warning: Could not analyze {dataset}: {e}", file=sys.stderr)
            continue
    
    if len(dataset_chars) < 2:
        print("Error: Need at least 2 datasets for characteristics heatmap", file=sys.stderr)
        return
    
    # Create correlation matrix
    df_chars = pd.DataFrame(dataset_chars)
    
    # Select numeric columns for correlation
    numeric_cols = ["n_rows", "n_cols", "file_size_mb", "numeric_ratio", "tabcl_ratio"]
    corr_matrix = df_chars[numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=0.5, ax=ax,
               cbar_kws={'label': 'Correlation Coefficient'})
    
    ax.set_title("Dataset Characteristics Correlation Matrix\n"
                f"Shows relationships between dataset properties and compression ratio",
                fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Dataset characteristics heatmap saved to {output_file}", file=sys.stderr)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create heatmap visualizations for tabcl")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent.parent / "benchmark_results.csv",
                       help="Input CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent,
                       help="Output directory for plots")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Specific dataset to analyze (for MI and edge weight heatmaps)")
    parser.add_argument("--all", action="store_true",
                       help="Generate all heatmaps for all datasets")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    results = load_results(args.input)
    if not results:
        print("Error: No results found in CSV file", file=sys.stderr)
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset characteristics heatmap (works for all datasets)
    plot_dataset_characteristics_heatmap(args.input, args.output_dir / "dataset_characteristics_heatmap.png")
    
    if args.all or args.dataset:
        datasets_to_process = [args.dataset] if args.dataset else [r["dataset"] for r in results]
        
        for dataset_name in datasets_to_process:
            # MI heatmap
            plot_mi_heatmap(args.input, dataset_name, 
                          args.output_dir / f"mi_heatmap_{dataset_name}.png")
            
            # Edge weight heatmap
            plot_edge_weight_heatmap(args.input, dataset_name,
                                    args.output_dir / f"edge_weight_heatmap_{dataset_name}.png")
            
            # Parameter sweep (optional, takes a long time)
            # Uncomment if you want to run parameter sweeps
            # plot_parameter_sweep_heatmap(args.input, dataset_name,
            #                             args.output_dir / f"parameter_sweep_{dataset_name}.png")
    else:
        print("Use --dataset <name> to generate MI/edge weight heatmaps for a specific dataset", file=sys.stderr)
        print("Use --all to generate for all datasets", file=sys.stderr)


if __name__ == "__main__":
    main()

