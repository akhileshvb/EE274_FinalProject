"""
Run full experiment: generate synthetic data, compress it, and plot compression gain vs MI.
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_mi_data import generate_dataset_with_varying_mi
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Run synthetic MI experiment: generate data, compress, and plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment
  python3 scripts/run_synthetic_experiment.py --output synthetic_experiment/
        """
    )
    parser.add_argument("--output", type=Path, default=Path("synthetic_experiment"),
                       help="Output directory (default: synthetic_experiment)")
    parser.add_argument("--n-pairs", type=int, default=20,
                       help="Number of parent-child pairs (default: 20)")
    parser.add_argument("--n-rows", type=int, default=10000,
                       help="Number of rows (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--use-tree", action="store_true", default=True,
                       help="Use tree-structured dependencies (default: True)")
    parser.add_argument("--tree-depth", type=int, default=3,
                       help="Tree depth if using tree structure (default: 3)")
    parser.add_argument("--branches", type=int, default=3,
                       help="Branches per node if using tree structure (default: 3)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Synthetic MI Experiment")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print(f"\n[1/3] Generating synthetic dataset...")
    csv_file = args.output / "synthetic_mi_data.csv"
    
    if args.use_tree:
        from scripts.generate_synthetic_mi_data import generate_tree_structured_data
        columns_dict = generate_tree_structured_data(args.n_rows, args.tree_depth, args.branches, args.seed)
        df = pd.DataFrame(columns_dict)
    else:
        df = generate_dataset_with_varying_mi(args.n_rows, args.n_pairs, args.seed, use_tree=False)
    
    df.to_csv(csv_file, index=False)
    print(f"  Saved to {csv_file}")
    print(f"  Shape: {df.shape}")
    
    # Step 2: Compress with tabcl
    print(f"\n[2/3] Compressing with tabcl...")
    tabcl_file = args.output / "synthetic_mi_data.csv.tabcl"
    
    try:
        result = subprocess.run(
            ["python3", "-m", "src.tabcl.cli", "compress", 
             "--input", str(csv_file), 
             "--output", str(tabcl_file)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  Compressed to {tabcl_file}")
    except subprocess.CalledProcessError as e:
        print(f"  Error compressing: {e.stderr}")
        sys.exit(1)
    
    # Step 3: Generate compression gain plot
    print(f"\n[3/3] Generating compression gain plot...")
    plot_file = args.output / "compression_gain_synthetic.png"
    
    try:
        result = subprocess.run(
            ["python3", "scripts/plot_compression_gain.py", 
             str(tabcl_file), 
             "--csv-file", str(csv_file),
             "-o", str(plot_file)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  Plot saved to {plot_file}")
    except subprocess.CalledProcessError as e:
        print(f"  Error generating plot: {e.stderr}")
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print("Experiment complete!")
    print(f"{'=' * 60}")
    print(f"\nOutput files:")
    print(f"  - Dataset: {csv_file}")
    print(f"  - Compressed: {tabcl_file}")
    print(f"  - Plot: {plot_file}")


if __name__ == "__main__":
    main()

