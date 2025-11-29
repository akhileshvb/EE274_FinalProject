"""
Run synthetic dataset experiments with varying correlation levels.
Generates multiple datasets with different correlation strengths, runs compression benchmarks,
and writes results to CSV.
"""

import argparse
import sys
import subprocess
import csv
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting will be skipped", file=sys.stderr)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_mi_data import generate_tree_structured_data


def run_benchmark(csv_file: Path, outdir: Path, delimiter: str = ",", 
                  zstd_level: int = 19, mi_mode: str = "auto", workers: int = None):
    """
    Run compression benchmark on a CSV file and return results.
    
    Returns:
        Dictionary with compression results (size, time, ratio) for each method
    """
    results = {}
    
    # Get original size
    orig_size = csv_file.stat().st_size
    results['original_size'] = orig_size
    
    # Handle "auto" mi_mode - choose based on file size
    effective_mi_mode = mi_mode
    if mi_mode == "auto":
        # For small files (< 1MB), use exact; for larger, use hashed
        if orig_size < 1_000_000:
            effective_mi_mode = "exact"
        else:
            effective_mi_mode = "hashed"
    
    # tabcl (learned tree)
    tabcl_out = outdir / (csv_file.stem + ".tabcl")
    if tabcl_out.exists():
        tabcl_out.unlink()
    
    tabcl_cmd = ["tabcl", "compress", "--input", str(csv_file), 
                 "--output", str(tabcl_out), "--delimiter", delimiter,
                 "--mi-mode", effective_mi_mode, "--rare-threshold", "1"]
    if workers:
        tabcl_cmd.extend(["--workers", str(workers)])
    
    try:
        import time
        start = time.perf_counter()
        subprocess.run(tabcl_cmd, check=True, capture_output=True)
        results['tabcl_time'] = time.perf_counter() - start
        results['tabcl_size'] = tabcl_out.stat().st_size
        results['tabcl_ratio'] = orig_size / results['tabcl_size']
    except subprocess.CalledProcessError as e:
        print(f"  Warning: tabcl compression failed: {e.stderr.decode() if e.stderr else 'unknown error'}", file=sys.stderr)
        results['tabcl_time'] = None
        results['tabcl_size'] = None
        results['tabcl_ratio'] = None
    
    # tabcl (line graph baseline)
    tabcl_line_out = outdir / (csv_file.stem + ".tabcl_line")
    if tabcl_line_out.exists():
        tabcl_line_out.unlink()
    
    tabcl_line_cmd = ["tabcl", "compress", "--input", str(csv_file),
                      "--output", str(tabcl_line_out), "--delimiter", delimiter,
                      "--mi-mode", effective_mi_mode, "--rare-threshold", "1", "--use-line-graph"]
    if workers:
        tabcl_line_cmd.extend(["--workers", str(workers)])
    
    try:
        start = time.perf_counter()
        subprocess.run(tabcl_line_cmd, check=True, capture_output=True)
        results['tabcl_line_time'] = time.perf_counter() - start
        results['tabcl_line_size'] = tabcl_line_out.stat().st_size
        results['tabcl_line_ratio'] = orig_size / results['tabcl_line_size']
    except subprocess.CalledProcessError as e:
        print(f"  Warning: tabcl line graph compression failed: {e.stderr.decode() if e.stderr else 'unknown error'}", file=sys.stderr)
        results['tabcl_line_time'] = None
        results['tabcl_line_size'] = None
        results['tabcl_line_ratio'] = None
    
    # gzip
    import shutil
    if shutil.which("gzip"):
        gzip_out = outdir / (csv_file.stem + ".gz")
        if gzip_out.exists():
            gzip_out.unlink()
        try:
            start = time.perf_counter()
            subprocess.run(["gzip", "-kf", str(csv_file)], check=True, capture_output=True)
            results['gzip_time'] = time.perf_counter() - start
            produced = csv_file.parent / (csv_file.name + ".gz")
            if produced.exists() and produced != gzip_out:
                produced.replace(gzip_out)
            results['gzip_size'] = gzip_out.stat().st_size
            results['gzip_ratio'] = orig_size / results['gzip_size']
        except subprocess.CalledProcessError:
            results['gzip_time'] = None
            results['gzip_size'] = None
            results['gzip_ratio'] = None
    else:
        results['gzip_time'] = None
        results['gzip_size'] = None
        results['gzip_ratio'] = None
    
    # zstd (raw zstd on original CSV, not tabcl+zstd)
    if shutil.which("zstd"):
        zstd_out = outdir / (csv_file.stem + ".zst")
        if zstd_out.exists():
            zstd_out.unlink()
        try:
            start = time.perf_counter()
            # Compress the original CSV file directly with zstd
            subprocess.run(["zstd", "-q", f"-{zstd_level}", "-f", str(csv_file), "-o", str(zstd_out)],
                          check=True, capture_output=True)
            results['zstd_time'] = time.perf_counter() - start
            results['zstd_size'] = zstd_out.stat().st_size
            results['zstd_ratio'] = orig_size / results['zstd_size']
        except subprocess.CalledProcessError:
            results['zstd_time'] = None
            results['zstd_size'] = None
            results['zstd_ratio'] = None
    else:
        results['zstd_time'] = None
        results['zstd_size'] = None
        results['zstd_ratio'] = None
    
    return results


def plot_results(results_file: Path, output_file: Path = None):
    """
    Plot compression results vs correlation strength.
    
    Creates a plot showing compression ratio vs correlation strength.
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plots", file=sys.stderr)
        return
    
    if not results_file.exists():
        print(f"Error: Results file {results_file} not found", file=sys.stderr)
        return
    
    # Load results
    df = pd.read_csv(results_file)
    
    if len(df) == 0:
        print("Warning: No results to plot", file=sys.stderr)
        return
    
    # Sort by correlation strength
    df = df.sort_values('correlation_strength')
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.suptitle('Compression Ratio vs Correlation Strength', fontsize=20, fontweight='bold')
    
    methods = [
        ('tabcl_ratio', 'tabcl (learned tree)', 'steelblue', 'o-', 2.5),
        ('tabcl_line_ratio', 'tabcl (line graph)', 'coral', 's--', 2),
        ('gzip_ratio', 'gzip', 'green', '^-', 2),
        ('zstd_ratio', 'zstd', 'purple', 'd-', 2),
    ]
    
    for col, label, color, style, linewidth in methods:
        if col in df.columns:
            valid_data = df[df[col].notna()]
            if len(valid_data) > 0:
                ax.plot(valid_data['correlation_strength'], valid_data[col], 
                        style, label=label, color=color, linewidth=linewidth, markersize=7)
    
    ax.set_xlabel('Correlation Strength', fontsize=18, fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontsize=18, fontweight='bold')
    ax.legend(loc='best', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([df['correlation_strength'].min() - 0.05, df['correlation_strength'].max() + 0.05])
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic dataset experiments with varying correlation levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment with 5 correlation levels
  python3 scripts/run_synthetic_experiment.py --output synthetic_experiment/ --n-levels 5
  
  # Run with custom correlation range
  python3 scripts/run_synthetic_experiment.py --output synthetic_experiment/ --corr-min 0.1 --corr-max 0.9
        """
    )
    parser.add_argument("--output", type=Path, default=Path("synthetic_experiment"),
                       help="Output directory (default: synthetic_experiment)")
    parser.add_argument("--n-levels", type=int, default=10,
                       help="Number of correlation levels to test (default: 10)")
    parser.add_argument("--corr-min", type=float, default=0.0,
                       help="Minimum correlation strength (default: 0.0)")
    parser.add_argument("--corr-max", type=float, default=1.0,
                       help="Maximum correlation strength (default: 1.0)")
    parser.add_argument("--n-rows", type=int, default=10000,
                       help="Number of rows per dataset (default: 10000)")
    parser.add_argument("--tree-depth", type=int, default=3,
                       help="Tree depth (default: 3)")
    parser.add_argument("--branches", type=int, default=3,
                       help="Branches per node (default: 3)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--zstd-level", type=int, default=19,
                       help="ZSTD compression level (default: 19)")
    parser.add_argument("--mi-mode", default="auto", choices=["exact", "hashed", "auto"],
                       help="MI computation mode (default: auto)")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers for tabcl (default: auto-detect)")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip compression benchmarks (only generate datasets)")
    
    args = parser.parse_args()
    
    # Create output directory structure
    args.output.mkdir(parents=True, exist_ok=True)
    datasets_dir = args.output / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean up old datasets if they exist in the main directory
    for old_csv in args.output.glob("synthetic_corr_*.csv"):
        old_csv.unlink()
    
    print("=" * 70)
    print("Synthetic Dataset Experiment with Varying Correlation Levels")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output}")
    print(f"  Datasets directory: {datasets_dir}")
    print(f"  Correlation levels: {args.n_levels} ({args.corr_min} to {args.corr_max})")
    print(f"  Rows per dataset: {args.n_rows}")
    print(f"  Tree depth: {args.tree_depth}, Branches: {args.branches}")
    print(f"  Seed: {args.seed}")
    print()
    
    # Generate correlation levels with extra resolution near 1.0.
    if args.n_levels == 1:
        correlation_levels = [args.corr_min]
    else:
        if args.corr_max >= 0.8 and args.corr_min <= 0.8:
            n_points_low = max(2, int(args.n_levels * 0.5))
            if args.corr_min < 0.8:
                low_range = np.linspace(args.corr_min, 0.8, n_points_low, endpoint=False)
            else:
                low_range = np.array([])
            
            n_points_mid = max(3, int(args.n_levels * 0.3))
            mid_range = np.linspace(0.8, 0.95, n_points_mid, endpoint=False)
            
            n_points_high = max(6, args.n_levels - len(low_range) - len(mid_range))
            if args.corr_max >= 0.95:
                uniform_spacing = np.linspace(0, 1, n_points_high)
                high_range = 0.95 + (args.corr_max - 0.95) * (1 - np.exp(-3 * uniform_spacing)) / (1 - np.exp(-3))
                high_range = np.clip(high_range, 0.95, args.corr_max)
                if args.corr_max >= 1.0:
                    high_range[-1] = args.corr_max
            else:
                high_range = np.linspace(0.95, args.corr_max, n_points_high, endpoint=True)
            
            correlation_levels = np.concatenate([low_range, mid_range, high_range])
            correlation_levels = np.unique(correlation_levels)
            correlation_levels = np.sort(correlation_levels)
        else:
            correlation_levels = np.linspace(args.corr_min, args.corr_max, args.n_levels)
    
    # Results storage
    all_results = []
    
    # Generate datasets and run benchmarks
    for idx, corr_strength in enumerate(correlation_levels):
        print(f"[{idx+1}/{args.n_levels}] Correlation strength: {corr_strength:.3f}")
        print(f"  Generating dataset...")
        
        # Generate dataset
        columns_dict = generate_tree_structured_data(
            args.n_rows, args.tree_depth, args.branches, 
            correlation_strength=corr_strength, seed=args.seed + idx
        )
        df = pd.DataFrame(columns_dict)
        
        # Save dataset in datasets subdirectory
        csv_file = datasets_dir / f"synthetic_corr_{corr_strength:.3f}.csv"
        df.to_csv(csv_file, index=False)
        print(f"  Saved to {csv_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
        
        if not args.skip_benchmark:
            print(f"  Running compression benchmarks...")
            results = run_benchmark(
                csv_file, args.output, delimiter=",", 
                zstd_level=args.zstd_level, mi_mode=args.mi_mode, workers=args.workers
            )
            
            # Add metadata
            results['correlation_strength'] = corr_strength
            results['n_rows'] = args.n_rows
            results['n_columns'] = df.shape[1]
            results['tree_depth'] = args.tree_depth
            results['branches'] = args.branches
            results['dataset_file'] = csv_file.name
            
            all_results.append(results)
            
            # Print summary
            if results.get('tabcl_ratio'):
                print(f"    tabcl: {results['tabcl_ratio']:.2f}x compression")
            if results.get('tabcl_line_ratio'):
                print(f"    tabcl (line): {results['tabcl_line_ratio']:.2f}x compression")
            if results.get('gzip_ratio'):
                print(f"    gzip: {results['gzip_ratio']:.2f}x compression")
            if results.get('zstd_ratio'):
                print(f"    zstd: {results['zstd_ratio']:.2f}x compression")
        else:
            print(f"  Skipping benchmarks (--skip-benchmark)")
        
        print()
    
    # Write results to CSV
    if all_results and not args.skip_benchmark:
        results_file = args.output / "compression_results.csv"
        print(f"Writing results to {results_file}...")
        
        # Define column order
        columns = [
            'correlation_strength', 'n_rows', 'n_columns', 'tree_depth', 'branches',
            'original_size', 'dataset_file',
            'tabcl_size', 'tabcl_time', 'tabcl_ratio',
            'tabcl_line_size', 'tabcl_line_time', 'tabcl_line_ratio',
            'gzip_size', 'gzip_time', 'gzip_ratio',
            'zstd_size', 'zstd_time', 'zstd_ratio'
        ]
        
        # Write CSV
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for result in all_results:
                writer.writerow({col: result.get(col, '') for col in columns})
        
        print(f"Results saved to {results_file}")
        print(f"\nSummary:")
        print(f"  Total datasets: {len(all_results)}")
        print(f"  Correlation range: {args.corr_min:.3f} to {args.corr_max:.3f}")
        
        # Print summary statistics
        if all_results:
            tabcl_ratios = [r['tabcl_ratio'] for r in all_results if r.get('tabcl_ratio')]
            if tabcl_ratios:
                print(f"\nCompression ratio statistics (tabcl):")
                print(f"  Mean: {np.mean(tabcl_ratios):.2f}x")
                print(f"  Min: {np.min(tabcl_ratios):.2f}x")
                print(f"  Max: {np.max(tabcl_ratios):.2f}x")
        
        # Generate plots
        if HAS_MATPLOTLIB:
            print(f"\nGenerating plots...")
            plot_file = args.output / "compression_results.png"
            plot_results(results_file, plot_file)
        else:
            print(f"\nSkipping plots (matplotlib not available)")
    
    print(f"\n{'=' * 70}")
    print("Experiment complete!")
    print(f"{'=' * 70}")
    print(f"\nOutput files in {args.output}:")
    print(f"  - Datasets: {datasets_dir}/synthetic_corr_*.csv")
    if not args.skip_benchmark:
        print(f"  - Results: compression_results.csv")
        print(f"  - Plot: compression_results.png")
        print(f"  - Compressed files: *.tabcl, *.tabcl_line, *.gz, *.zst")


if __name__ == "__main__":
    main()
