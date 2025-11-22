#!/usr/bin/env python3
"""
Plot benchmark results from CSV.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required. Install with: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)


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
    """Plot compression ratios for all methods."""
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    datasets = [r["dataset"] for r in results]
    
    # Prepare data
    method_data = {method: [] for method in methods}
    for result in results:
        for method in methods:
            ratio_key = f"{method}_ratio"
            ratio = result.get(ratio_key)
            method_data[method].append(ratio if ratio is not None else 0)
    
    # Create plot
    x = np.arange(len(datasets))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        ratios = method_data[method]
        # Only plot if we have at least one non-zero value
        if any(r > 0 for r in ratios):
            offset = (i - len(methods) / 2) * width + width / 2
            bars = ax.bar(x + offset, ratios, width, label=method, alpha=0.8)
            # Add value labels on bars
            for bar, ratio in zip(bars, ratios):
                if ratio > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{ratio:.2f}x', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_title("Compression Ratio Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Compression ratio plot saved to {output_file}", file=sys.stderr)


def plot_compression_times(results: List[Dict], output_file: Path):
    """Plot compression times for all methods."""
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    datasets = [r["dataset"] for r in results]
    
    # Prepare data
    method_data = {method: [] for method in methods}
    for result in results:
        for method in methods:
            time_key = f"{method}_time"
            time_val = result.get(time_key)
            method_data[method].append(time_val if time_val is not None else 0)
    
    # Create plot
    x = np.arange(len(datasets))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, method in enumerate(methods):
        times = method_data[method]
        # Only plot if we have at least one non-zero value
        if any(t > 0 for t in times):
            offset = (i - len(methods) / 2) * width + width / 2
            bars = ax.bar(x + offset, times, width, label=method, alpha=0.8)
            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                if time_val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{time_val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Compression Time (seconds)", fontsize=12)
    ax.set_title("Compression Time Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Compression time plot saved to {output_file}", file=sys.stderr)


def plot_combined(results: List[Dict], output_file: Path):
    """Plot compression ratios and times in a combined figure."""
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    datasets = [r["dataset"] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Compression Ratios
    x = np.arange(len(datasets))
    width = 0.15
    
    method_data_ratio = {method: [] for method in methods}
    for result in results:
        for method in methods:
            ratio_key = f"{method}_ratio"
            ratio = result.get(ratio_key)
            method_data_ratio[method].append(ratio if ratio is not None else 0)
    
    for i, method in enumerate(methods):
        ratios = method_data_ratio[method]
        if any(r > 0 for r in ratios):
            offset = (i - len(methods) / 2) * width + width / 2
            bars = ax1.bar(x + offset, ratios, width, label=method, alpha=0.8)
            for bar, ratio in zip(bars, ratios):
                if ratio > 0:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{ratio:.2f}x', ha='center', va='bottom', fontsize=7)
    
    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_ylabel("Compression Ratio", fontsize=12)
    ax1.set_title("Compression Ratio", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha="right")
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    
    # Plot 2: Compression Times
    method_data_time = {method: [] for method in methods}
    for result in results:
        for method in methods:
            time_key = f"{method}_time"
            time_val = result.get(time_key)
            method_data_time[method].append(time_val if time_val is not None else 0)
    
    for i, method in enumerate(methods):
        times = method_data_time[method]
        if any(t > 0 for t in times):
            offset = (i - len(methods) / 2) * width + width / 2
            bars = ax2.bar(x + offset, times, width, label=method, alpha=0.8)
            for bar, time_val in zip(bars, times):
                if time_val > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{time_val:.2f}s', ha='center', va='bottom', fontsize=7)
    
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_ylabel("Compression Time (seconds)", fontsize=12)
    ax2.set_title("Compression Time", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=15, ha="right")
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Combined plot saved to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent.parent / "benchmark_results.csv",
                       help="Input CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent,
                       help="Output directory for plots")
    parser.add_argument("--combined", action="store_true",
                       help="Create a combined plot with both ratios and times")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    results = load_results(args.input)
    if not results:
        print("Error: No results found in CSV file", file=sys.stderr)
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.combined:
        output_file = args.output_dir / "benchmark_results_combined.png"
        plot_combined(results, output_file)
    else:
        ratio_file = args.output_dir / "benchmark_results_ratios.png"
        time_file = args.output_dir / "benchmark_results_times.png"
        plot_compression_ratios(results, ratio_file)
        plot_compression_times(results, time_file)


if __name__ == "__main__":
    main()

