#!/usr/bin/env python3
"""
Advanced plots for tabcl compression analysis.
Extracts statistics from compressed files and creates visualizations.
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import base64

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
except ImportError:
    print("Error: matplotlib, numpy, and networkx are required. Install with: pip install matplotlib numpy networkx", file=sys.stderr)
    sys.exit(1)

# Import tabcl to read model data
try:
    from tabcl.cli import Model
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
            
            # Count column types
            numeric_cols = sum(1 for d in model.dicts if d is None or (isinstance(d, tuple) and len(d) == 2))
            categorical_cols = len(model.dicts) - numeric_cols
            
            # Count edges in forest
            n_edges = len(model.edges)
            n_cols = len(model.columns)
            
            # Calculate edge weights statistics
            edge_weights = [e[2] for e in model.edges if len(e) >= 3]
            
            # Count rare values (approximate from rare_meta)
            rare_count = sum(1 for r in model.rare_meta if r and r.strip())
            
            # Calculate vocabulary sizes for categorical columns
            vocab_sizes = []
            for d in model.dicts:
                if isinstance(d, list):
                    vocab_sizes.append(len(d))
                elif d is None or (isinstance(d, tuple) and len(d) == 2):
                    vocab_sizes.append(0)  # Numeric column
            
            return {
                "n_cols": n_cols,
                "n_edges": n_edges,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "edge_weights": edge_weights,
                "avg_edge_weight": np.mean(edge_weights) if edge_weights else 0,
                "max_edge_weight": max(edge_weights) if edge_weights else 0,
                "rare_count": rare_count,
                "vocab_sizes": vocab_sizes,
                "avg_vocab_size": np.mean(vocab_sizes) if vocab_sizes else 0,
                "max_vocab_size": max(vocab_sizes) if vocab_sizes else 0,
            }
    except Exception as e:
        print(f"Warning: Could not extract stats from {tabcl_file}: {e}", file=sys.stderr)
        return None


def plot_ratio_vs_size(results: List[Dict], output_file: Path):
    """Plot compression ratio vs dataset size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for method, color in zip(methods, colors):
        sizes = []
        ratios = []
        labels = []
        for r in results:
            size = r.get("original_size_bytes")
            ratio_key = f"{method}_ratio"
            ratio = r.get(ratio_key)
            if size and ratio and ratio > 0:
                sizes.append(size / (1024 * 1024))  # Convert to MB
                ratios.append(ratio)
                labels.append(r["dataset"])
        
        if sizes:
            ax.scatter(sizes, ratios, label=method, alpha=0.7, s=100, color=color)
            # Add dataset labels
            for i, label in enumerate(labels):
                ax.annotate(label, (sizes[i], ratios[i]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Dataset Size (MB)", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_title("Compression Ratio vs Dataset Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Ratio vs size plot saved to {output_file}", file=sys.stderr)


def plot_time_vs_size(results: List[Dict], output_file: Path):
    """Plot compression time vs dataset size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    for method, color in zip(methods, colors):
        sizes = []
        times = []
        labels = []
        for r in results:
            size = r.get("original_size_bytes")
            time_key = f"{method}_time"
            time_val = r.get(time_key)
            if size and time_val and time_val > 0:
                sizes.append(size / (1024 * 1024))  # Convert to MB
                times.append(time_val)
                labels.append(r["dataset"])
        
        if sizes:
            ax.scatter(sizes, times, label=method, alpha=0.7, s=100, color=color)
            # Add dataset labels for tabcl
            if method == "tabcl":
                for i, label in enumerate(labels):
                    ax.annotate(label, (sizes[i], times[i]), fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Dataset Size (MB)", fontsize=12)
    ax.set_ylabel("Compression Time (seconds)", fontsize=12)
    ax.set_title("Compression Time vs Dataset Size", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Time vs size plot saved to {output_file}", file=sys.stderr)


def plot_efficiency(results: List[Dict], output_file: Path):
    """Plot compression efficiency: ratio per second."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = [r["dataset"] for r in results]
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    
    x = np.arange(len(datasets))
    width = 0.15
    
    method_data = {method: [] for method in methods}
    for result in results:
        for method in methods:
            ratio_key = f"{method}_ratio"
            time_key = f"{method}_time"
            ratio = result.get(ratio_key)
            time_val = result.get(time_key)
            if ratio and time_val and time_val > 0:
                efficiency = ratio / time_val  # Ratio per second
                method_data[method].append(efficiency)
            else:
                method_data[method].append(0)
    
    for i, method in enumerate(methods):
        efficiencies = method_data[method]
        if any(e > 0 for e in efficiencies):
            offset = (i - len(methods) / 2) * width + width / 2
            bars = ax.bar(x + offset, efficiencies, width, label=method, alpha=0.8)
            for bar, eff in zip(bars, efficiencies):
                if eff > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{eff:.1f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Compression Efficiency (Ratio/Second)", fontsize=12)
    ax.set_title("Compression Efficiency: Ratio per Second", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Efficiency plot saved to {output_file}", file=sys.stderr)


def plot_relative_performance(results: List[Dict], output_file: Path):
    """Plot tabcl performance relative to best baseline for each dataset."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = []
    ratio_improvements = []
    time_overheads = []
    
    for r in results:
        tabcl_ratio = r.get("tabcl_ratio")
        tabcl_time = r.get("tabcl_time")
        
        if not tabcl_ratio or not tabcl_time:
            continue
        
        # Find best baseline ratio
        baseline_ratios = []
        for method in ["gzip", "zstd", "bzip2", "columnar gzip"]:
            ratio = r.get(f"{method}_ratio")
            if ratio:
                baseline_ratios.append(ratio)
        
        if baseline_ratios:
            best_baseline_ratio = max(baseline_ratios)
            ratio_improvement = (tabcl_ratio / best_baseline_ratio - 1) * 100  # Percentage improvement
            
            # Find best baseline time
            baseline_times = []
            for method in ["gzip", "zstd", "bzip2", "columnar gzip"]:
                time_val = r.get(f"{method}_time")
                if time_val and time_val > 0:
                    baseline_times.append(time_val)
            
            if baseline_times:
                best_baseline_time = min(baseline_times)
                time_overhead = (tabcl_time / best_baseline_time - 1) * 100  # Percentage overhead
                
                datasets.append(r["dataset"])
                ratio_improvements.append(ratio_improvement)
                time_overheads.append(time_overhead)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ratio_improvements, width, label="Ratio Improvement (%)", alpha=0.8, color="#2ca02c")
    bars2 = ax.bar(x + width/2, time_overheads, width, label="Time Overhead (%)", alpha=0.8, color="#d62728")
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("tabcl Performance Relative to Best Baseline", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Relative performance plot saved to {output_file}", file=sys.stderr)


def plot_forest_structure(results: List[Dict], csv_file: Path, output_dir: Path):
    """Plot forest structure for each dataset."""
    # Find .tabcl files
    base_dir = csv_file.parent
    
    for r in results:
        dataset = r["dataset"]
        input_file = r.get("input_file")
        if not input_file:
            continue
        
        tabcl_file = base_dir / Path(input_file).with_suffix(".tabcl")
        if not tabcl_file.exists():
            print(f"Warning: {tabcl_file} not found, skipping forest visualization", file=sys.stderr)
            continue
        
        stats = extract_model_stats(tabcl_file)
        if not stats:
            continue
        
        # Load model to get edges
        try:
            data = tabcl_file.read_bytes()
            p = 6
            version = int.from_bytes(data[p:p+4], "little")
            p += 4
            mlen = int.from_bytes(data[p:p+8], "little")
            p += 8
            model = Model.from_bytes(data[p:p+mlen])
            
            # Create graph
            G = nx.Graph()
            G.add_nodes_from(range(stats["n_cols"]))
            for u, v, w in model.edges:
                G.add_edge(u, v, weight=w)
            
            # Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes - color by root vs child
            node_colors = []
            for i in range(stats["n_cols"]):
                if model.parents[i] == -1:
                    node_colors.append("lightblue")  # Root nodes
                else:
                    node_colors.append("lightcoral")  # Child nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                                 node_size=500, alpha=0.9, edgecolors="black", linewidths=1.5)
            
            # Draw edges with weights as edge width
            edges = G.edges()
            weights = [G[u][v].get("weight", 1.0) for u, v in edges]
            if weights:
                max_weight = max(weights)
                edge_widths = [w / max_weight * 3 for w in weights]
            else:
                edge_widths = [1] * len(edges)
            
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                                 alpha=0.6, edge_color="gray")
            
            # Draw labels - use column names if available
            if model.columns:
                # Truncate long column names
                labels = {i: (col[:15] + "..." if len(col) > 15 else col) 
                         for i, col in enumerate(model.columns)}
            else:
                labels = {i: f"C{i}" for i in range(stats["n_cols"])}
            nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
            
            ax.set_title(f"Forest Structure: {dataset}\n"
                        f"Columns: {stats['n_cols']}, Edges: {stats['n_edges']}, "
                        f"Numeric: {stats['numeric_cols']}, Categorical: {stats['categorical_cols']}",
                        fontsize=12, fontweight="bold")
            ax.axis("off")
            
            output_file = output_dir / f"forest_{dataset}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Forest structure plot saved to {output_file}", file=sys.stderr)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not plot forest for {dataset}: {e}", file=sys.stderr)


def plot_edge_weight_distribution(results: List[Dict], csv_file: Path, output_file: Path):
    """Plot distribution of edge weights across all datasets."""
    base_dir = csv_file.parent
    all_weights = []
    dataset_labels = []
    
    for r in results:
        dataset = r["dataset"]
        input_file = r.get("input_file")
        if not input_file:
            continue
        
        tabcl_file = base_dir / Path(input_file).with_suffix(".tabcl")
        if not tabcl_file.exists():
            continue
        
        stats = extract_model_stats(tabcl_file)
        if stats and stats["edge_weights"]:
            all_weights.extend(stats["edge_weights"])
            dataset_labels.extend([dataset] * len(stats["edge_weights"]))
    
    if not all_weights:
        print("Warning: No edge weights found, skipping distribution plot", file=sys.stderr)
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(all_weights, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Edge Weight", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Edge Weight Distribution", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)
    
    # Box plot by dataset
    if dataset_labels:
        unique_datasets = list(set(dataset_labels))
        dataset_weights = {ds: [] for ds in unique_datasets}
        for weight, ds in zip(all_weights, dataset_labels):
            dataset_weights[ds].append(weight)
        
        data_to_plot = [dataset_weights[ds] for ds in unique_datasets]
        bp = ax2.boxplot(data_to_plot, labels=unique_datasets, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        
        ax2.set_ylabel("Edge Weight", fontsize=12)
        ax2.set_xlabel("Dataset", fontsize=12)
        ax2.set_title("Edge Weight by Dataset", fontsize=14, fontweight="bold")
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Edge weight distribution plot saved to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Create advanced visualizations for tabcl")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent.parent / "benchmark_results.csv",
                       help="Input CSV file")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent,
                       help="Output directory for plots")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input file {args.input} not found", file=sys.stderr)
        sys.exit(1)
    
    results = load_results(args.input)
    if not results:
        print("Error: No results found in CSV file", file=sys.stderr)
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all plots
    plot_ratio_vs_size(results, args.output_dir / "ratio_vs_size.png")
    plot_time_vs_size(results, args.output_dir / "time_vs_size.png")
    plot_efficiency(results, args.output_dir / "efficiency.png")
    plot_relative_performance(results, args.output_dir / "relative_performance.png")
    plot_edge_weight_distribution(results, args.input, args.output_dir / "edge_weight_distribution.png")
    plot_forest_structure(results, args.input, args.output_dir)


if __name__ == "__main__":
    main()

