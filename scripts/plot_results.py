import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy", 
          file=sys.stderr)
    sys.exit(1)


# Color scheme for methods
METHOD_COLORS = {
    "tabcl": "#1f77b4",          # Blue
    "gzip": "#ff7f0e",           # Orange
    "zstd": "#2ca02c",           # Green
    "bzip2": "#d62728",          # Red
    "columnar gzip": "#9467bd",  # Purple
}

METHOD_ORDER = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]

# Dataset name mapping
DATASET_NAMES = {
    "adult_train": "Adult Income (Census)",
    "test_sample_10k": "Display Advertising",
    "yellow_tripdata_2025-01_updated": "NYC Yellow Taxicab",
    "covtype": "Forest Data",
    "jester": "Jokes",
    "business_price": "Business Price Index",
}


def load_results(csv_file: Path) -> List[Dict]:
    """Load results from CSV file."""
    results = []
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
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


def plot_compression_ratios(results: List[Dict], output_file: Path):
    """Plot compression ratios for all methods - clean version without bar labels."""
    # Map dataset names
    dataset_keys = [r["dataset"] for r in results]
    datasets = [DATASET_NAMES.get(key, key) for key in dataset_keys]
    n_datasets = len(datasets)
    n_methods = len(METHOD_ORDER)
    
    # Prepare data
    method_data = {method: [] for method in METHOD_ORDER}
    for result in results:
        for method in METHOD_ORDER:
            ratio_key = f"{method}_ratio"
            ratio = result.get(ratio_key)
            method_data[method].append(ratio if ratio is not None else 0)
    
    # Create plot with more spacing
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(n_datasets)
    width = 0.12  # Fixed width per bar for better visibility
    spacing = 0.02  # Spacing between groups
    
    for i, method in enumerate(METHOD_ORDER):
        ratios = method_data[method]
        if any(r > 0 for r in ratios):
            offset = (i - n_methods / 2) * (width + spacing) + width / 2
            color = METHOD_COLORS.get(method, "#808080")
            bars = ax.bar(x + offset, ratios, width, label=method, alpha=0.85, 
                         color=color, edgecolor="black", linewidth=0.8)
    
    ax.set_xlabel("Dataset", fontsize=18, fontweight='bold')
    ax.set_ylabel("Compression Ratio", fontsize=18, fontweight='bold')
    ax.set_title("Compression Ratio Comparison", fontsize=20, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=14, ncol=1)
    ax.tick_params(axis='y', labelsize=14)
    # Add more grid lines for easier reading
    ax.grid(axis="y", alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    # Set y-axis to show more grid lines
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Compression ratio plot saved to {output_file}", file=sys.stderr)
    plt.close()


def plot_compression_times(results: List[Dict], output_file: Path):
    """Plot compression speedup factors for all methods (relative to slowest method per dataset)."""
    # Map dataset names
    dataset_keys = [r["dataset"] for r in results]
    datasets = [DATASET_NAMES.get(key, key) for key in dataset_keys]
    n_datasets = len(datasets)
    n_methods = len(METHOD_ORDER)
    
    # Prepare data - calculate speedup factors
    method_data = {method: [] for method in METHOD_ORDER}
    for result in results:
        # Get all times for this dataset
        times = {}
        for method in METHOD_ORDER:
            time_key = f"{method}_time"
            time_val = result.get(time_key)
            if time_val and time_val > 0:
                times[method] = time_val
        
        # Find slowest time (baseline)
        if times:
            slowest_time = max(times.values())
            
            # Calculate speedup factor for each method (slowest = 1x, faster = >1x)
            for method in METHOD_ORDER:
                if method in times:
                    speedup = slowest_time / times[method]
                    method_data[method].append(speedup)
                else:
                    method_data[method].append(0)
        else:
            # No valid times for this dataset
            for method in METHOD_ORDER:
                method_data[method].append(0)
    
    # Create plot with more spacing
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(n_datasets)
    width = 0.12  # Fixed width per bar
    spacing = 0.02  # Spacing between groups
    
    for i, method in enumerate(METHOD_ORDER):
        speedups = method_data[method]
        if any(s > 0 for s in speedups):
            offset = (i - n_methods / 2) * (width + spacing) + width / 2
            color = METHOD_COLORS.get(method, "#808080")
            bars = ax.bar(x + offset, speedups, width, label=method, alpha=0.85,
                         color=color, edgecolor="black", linewidth=0.8)
    
    ax.set_xlabel("Dataset", fontsize=18, fontweight='bold')
    ax.set_ylabel("Speedup Factor", fontsize=18, fontweight='bold')
    ax.set_title("Compression Speedup Comparison", fontsize=20, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=14, ncol=1)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis="y", alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    # Add more grid lines
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Compression speedup plot saved to {output_file}", file=sys.stderr)
    plt.close()


def plot_compression_ratios_heatmap(results: List[Dict], output_file: Path):
    """Plot compression ratios as a heatmap for easier comparison."""
    # Map dataset names
    dataset_keys = [r["dataset"] for r in results]
    datasets = [DATASET_NAMES.get(key, key) for key in dataset_keys]
    
    # Prepare data matrix
    data_matrix = []
    for result in results:
        row = []
        for method in METHOD_ORDER:
            ratio_key = f"{method}_ratio"
            ratio = result.get(ratio_key)
            row.append(ratio if ratio is not None else 0)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(METHOD_ORDER, fontsize=14)
    ax.set_yticklabels(datasets, fontsize=14)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(METHOD_ORDER)):
            value = data_matrix[i, j]
            if value > 0:
                text = ax.text(j, i, f'{value:.1f}x',
                             ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title("Compression Ratio Heatmap", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel("Compression Method", fontsize=18, fontweight='bold')
    ax.set_ylabel("Dataset", fontsize=18, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Compression Ratio', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Compression ratio heatmap saved to {output_file}", file=sys.stderr)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from CSV - creates separate plots for ratios and times"
    )
    parser.add_argument(
        "--input", 
        type=Path, 
        default=Path(__file__).parent.parent / "benchmark_results.csv",
        help="Input CSV file (default: benchmark_results.csv)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path(__file__).parent.parent,
        help="Output directory for plots (default: project root)"
    )
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    results = load_results(args.input)
    if not results:
        print("Error: No results found in CSV file", file=sys.stderr)
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create separate plots
    ratio_file = args.output_dir / "benchmark_results_ratios.png"
    speedup_file = args.output_dir / "benchmark_results_times.png"
    heatmap_file = args.output_dir / "benchmark_results_ratios_heatmap.png"
    
    plot_compression_ratios(results, ratio_file)
    plot_compression_times(results, speedup_file)
    plot_compression_ratios_heatmap(results, heatmap_file)
    
    print(f"\nPlots created successfully:", file=sys.stderr)
    print(f"  - Compression ratios (bar chart): {ratio_file}", file=sys.stderr)
    print(f"  - Compression speedup: {speedup_file}", file=sys.stderr)
    print(f"  - Compression ratios (heatmap): {heatmap_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
