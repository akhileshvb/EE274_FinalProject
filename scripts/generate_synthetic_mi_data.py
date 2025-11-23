"""
Generate synthetic data with varying mutual information between parent and child columns.
This allows us to study the relationship between MI and compression gain.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


def generate_correlated_data(n_rows: int, mi_target: float, seed: int = 0):
    """
    Generate two correlated columns with approximately target MI.
    
    Uses a deterministic mapping: child = f(parent) with controlled noise.
    Higher MI = more deterministic mapping.
    
    Args:
        n_rows: Number of rows
        mi_target: Target mutual information in bits
        seed: Random seed
    
    Returns:
        (parent, child) arrays with approximately target MI
    """
    np.random.seed(seed)
    
    # Generate parent column (uniform categorical)
    n_parent_values = 20  # More values for better MI control
    parent = np.random.randint(0, n_parent_values, size=n_rows)
    
    # Generate child column based on parent with controlled correlation
    # Strategy: child = deterministic_function(parent) + noise
    # MI is controlled by the noise level
    
    # Convert MI target to a noise probability
    # Higher MI = lower noise = more deterministic
    # For categorical data, we can approximate:
    # If child is deterministic function of parent, MI ≈ log(n_parent_values)
    # With noise, MI decreases
    
    # Use a simple model: child = (parent + offset) % n_child_values with probability p
    # Otherwise, child is random
    n_child_values = 20
    
    # Estimate noise probability from target MI
    # If deterministic: MI ≈ log2(min(n_parent, n_child))
    max_mi = np.log2(min(n_parent_values, n_child_values))
    if mi_target >= max_mi:
        noise_prob = 0.0  # Fully deterministic
    else:
        # Rough approximation: noise reduces MI
        # Use exponential decay model
        noise_prob = 1.0 - (mi_target / max_mi)
        noise_prob = max(0.0, min(0.95, noise_prob))  # Clamp
    
    child = np.zeros(n_rows, dtype=int)
    
    for i in range(n_rows):
        p_val = parent[i]
        if np.random.random() < (1.0 - noise_prob):
            # Deterministic: child = parent (with some offset for variety)
            child[i] = (p_val + (i % 3)) % n_child_values
        else:
            # Random: uniform over child values
            child[i] = np.random.randint(0, n_child_values)
    
    return parent, child


def generate_linear_correlated_data(n_rows: int, correlation: float, seed: int = 0):
    """
    Generate two linearly correlated numeric columns.
    
    Args:
        n_rows: Number of rows
        correlation: Pearson correlation coefficient (-1 to 1)
        seed: Random seed
    
    Returns:
        (parent, child) arrays with specified correlation
    """
    np.random.seed(seed)
    
    # Generate parent (normal distribution)
    parent = np.random.normal(0, 1, size=n_rows)
    
    # Generate child with specified correlation
    # child = correlation * parent + sqrt(1 - correlation^2) * noise
    noise = np.random.normal(0, 1, size=n_rows)
    child = correlation * parent + np.sqrt(1 - correlation**2) * noise
    
    return parent, child


def generate_tree_structured_data(n_rows: int, tree_depth: int, branches_per_node: int = 2, seed: int = 0):
    """
    Generate data with explicit tree-structured dependencies.
    
    Creates a tree where:
    - Root nodes are independent
    - Each level depends on the previous level
    - Strong correlations ensure high MI (95% deterministic, 5% noise)
    
    Args:
        n_rows: Number of rows
        tree_depth: Depth of the tree (number of levels)
        branches_per_node: Number of children per node
        seed: Random seed
    
    Returns:
        Dictionary of column_name -> array
    """
    np.random.seed(seed)
    columns = {}
    
    # Track all nodes by level for proper tree construction
    nodes_by_level = []
    
    # Generate root nodes (level 0) - independent
    n_roots = branches_per_node
    root_nodes = []
    for i in range(n_roots):
        # Root columns: uniform categorical
        n_values = 10
        col_name = f'root_{i}'
        columns[col_name] = np.random.randint(0, n_values, size=n_rows)
        root_nodes.append(col_name)
    nodes_by_level.append(root_nodes)
    
    # Generate dependent levels
    for level in range(1, tree_depth):
        current_level_nodes = []
        parent_nodes = nodes_by_level[level - 1]
        
        # For each parent node, create children
        for parent_name in parent_nodes:
            parent_data = columns[parent_name]
            
            # Create children for this parent
            for child_idx in range(branches_per_node):
                child_name = f'level_{level}_node_{len(current_level_nodes)}'
                
                # Generate child with strong dependency on parent
                # Use 95% deterministic mapping, 5% noise for high MI
                n_values = 10
                child = np.zeros(n_rows, dtype=int)
                
                # Create strong correlation: child = f(parent) with small noise
                for i in range(n_rows):
                    p_val = parent_data[i]
                    # Deterministic mapping with small random noise (5% noise)
                    if np.random.random() < 0.95:
                        # Strong correlation: child = parent (with some variation for uniqueness)
                        # Use a deterministic function based on parent value and row index
                        child[i] = (p_val + (i % 3) + child_idx) % n_values
                    else:
                        # Small noise
                        child[i] = np.random.randint(0, n_values)
                
                columns[child_name] = child
                current_level_nodes.append(child_name)
        
        nodes_by_level.append(current_level_nodes)
    
    return columns


def generate_dataset_with_varying_mi(n_rows: int = 10000, n_pairs: int = 20, seed: int = 0, use_tree: bool = True):
    """
    Generate a dataset with multiple parent-child pairs having varying MI values.
    
    Args:
        n_rows: Number of rows
        n_pairs: Number of parent-child pairs (if use_tree=False)
        seed: Random seed
        use_tree: If True, use tree-structured dependencies; if False, use independent pairs
    
    Returns:
        DataFrame with columns
    """
    np.random.seed(seed)
    
    if use_tree:
        # Generate tree-structured data
        # Calculate tree parameters to get roughly n_pairs dependencies
        # For a tree: total_nodes = sum(branches^level for level in range(depth))
        # Dependencies = total_nodes - roots
        tree_depth = 3  # 3 levels: root, level1, level2
        branches_per_node = 3  # 3 children per node
        
        # This gives: 3 roots, 9 level1 nodes, 27 level2 nodes = 39 nodes total
        # Dependencies: 9 + 27 = 36 parent-child pairs
        
        columns_dict = generate_tree_structured_data(n_rows, tree_depth, branches_per_node, seed)
        return pd.DataFrame(columns_dict)
    else:
        # Original approach: independent pairs
        columns = {}
        
        for i in range(n_pairs):
            # Vary MI from low to high in a controlled sweep
            if n_pairs > 1:
                # Linear spacing: 0 to ~3 bits (reasonable range for categorical)
                mi_target = (i / (n_pairs - 1)) * 3.0
            else:
                mi_target = 1.5
            
            # Use categorical correlation for all pairs (more predictable)
            parent, child = generate_correlated_data(n_rows, mi_target, seed=seed + i)
            
            columns[f'parent_{i}'] = parent
            columns[f'child_{i}'] = child
        
        return pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with varying mutual information",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset with 20 pairs, 10k rows
  python3 scripts/generate_synthetic_mi_data.py --output synthetic_data/ --n-pairs 20 --n-rows 10000
        """
    )
    parser.add_argument("--output", type=Path, default=Path("synthetic_data"),
                       help="Output directory (default: synthetic_data)")
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
    
    if args.use_tree:
        print(f"Generating tree-structured dataset (depth={args.tree_depth}, branches={args.branches}), {args.n_rows} rows...")
        columns_dict = generate_tree_structured_data(args.n_rows, args.tree_depth, args.branches, args.seed)
        df = pd.DataFrame(columns_dict)
    else:
        print(f"Generating synthetic dataset with {args.n_pairs} parent-child pairs, {args.n_rows} rows...")
        df = generate_dataset_with_varying_mi(args.n_rows, args.n_pairs, args.seed, use_tree=False)
    
    # Save to CSV
    output_file = args.output / "synthetic_mi_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Also save a summary
    summary_file = args.output / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Synthetic MI Dataset Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Rows: {args.n_rows}\n")
        f.write(f"Parent-child pairs: {args.n_pairs}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        if args.use_tree:
            f.write(f"\nTree structure:\n")
            f.write(f"  Depth: {args.tree_depth}\n")
            f.write(f"  Branches per node: {args.branches}\n")
            f.write(f"  Total columns: {len(df.columns)}\n")
        else:
            f.write(f"\nColumn pairs:\n")
            for i in range(args.n_pairs):
                f.write(f"  parent_{i} -> child_{i}\n")
    
    print(f"Summary saved to {summary_file}")


if __name__ == "__main__":
    main()

