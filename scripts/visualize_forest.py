import argparse
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
except ImportError:
    print("Error: matplotlib, networkx, and numpy are required. Install with: pip install matplotlib networkx numpy", file=sys.stderr)
    sys.exit(1)

# Add parent directory to path to import tabcl
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tabcl.cli import Model


def visualize_forest(tabcl_file: Path, output_file: Path = None, layout: str = "spring", 
                     show_labels: bool = True, show_weights: bool = True, 
                     col_names: bool = True, highlight_connected: bool = True):
    """
    Visualize the forest structure from a .tabcl file.
    
    Args:
        tabcl_file: Path to the .tabcl file
        output_file: Path to save the visualization (if None, displays interactively)
        layout: Graph layout algorithm ('spring', 'circular', 'hierarchical', 'kamada_kawai')
        show_labels: Whether to show node labels
        show_weights: Whether to show edge weights as edge width
        col_names: Whether to use column names from model (if available) instead of C0, C1, etc.
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
    
    # Create graph
    G = nx.Graph()
    n_cols = len(model.columns)
    G.add_nodes_from(range(n_cols))
    
    # Add edges with weights
    for edge in model.edges:
        if len(edge) >= 3:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            G.add_edge(u, v, weight=w)
        elif len(edge) == 2:
            u, v = int(edge[0]), int(edge[1])
            G.add_edge(u, v, weight=1.0)
    
    if len(G.edges()) == 0:
        print("Warning: No edges found in forest (all columns are independent)", file=sys.stderr)
    
    # Create figure - use larger size for dense graphs
    n_nodes = len(G.nodes())
    if n_nodes > 30:
        figsize = (18, 14)
    elif n_nodes > 15:
        figsize = (16, 12)
    else:
        figsize = (14, 10)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Choose layout with better spacing for dense graphs
    if layout == "spring":
        # Increase k (optimal distance) for better spacing in dense graphs
        k = max(3.0, np.sqrt(n_nodes) * 0.8) if n_nodes > 20 else 2.0
        pos = nx.spring_layout(G, k=k, iterations=100, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        # Try to create a hierarchical layout based on parent relationships
        try:
            # Use parent array to create hierarchy
            pos = {}
            roots = [i for i, p in enumerate(model.parents) if p == -1]
            if roots:
                # Simple hierarchical layout
                levels = {}
                def get_level(node):
                    if node in levels:
                        return levels[node]
                    if model.parents[node] == -1:
                        levels[node] = 0
                    else:
                        levels[node] = get_level(model.parents[node]) + 1
                    return levels[node]
                
                for i in range(n_cols):
                    levels[i] = get_level(i)
                
                max_level = max(levels.values()) if levels else 0
                for i in range(n_cols):
                    level = levels.get(i, 0)
                    y = max_level - level
                    # Distribute nodes at same level horizontally
                    same_level = [j for j in range(n_cols) if levels.get(j, 0) == level]
                    x = same_level.index(i) - len(same_level) / 2
                    pos[i] = (x, y)
            else:
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        except:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "force":
        # Force-directed layout with better parameters for dense graphs
        k = max(4.0, np.sqrt(n_nodes) * 1.2) if n_nodes > 20 else 3.0
        pos = nx.spring_layout(G, k=k, iterations=200, seed=42)
    else:
        k = max(3.0, np.sqrt(n_nodes) * 0.8) if n_nodes > 20 else 2.0
        pos = nx.spring_layout(G, k=k, iterations=100, seed=42)
    
    # Post-process layout to prevent node overlap
    # Increase minimum distance between nodes
    min_distance = 0.3
    max_iterations = 50
    for _ in range(max_iterations):
        moved = False
        for i in range(n_cols):
            if i not in pos:
                continue
            for j in range(i + 1, n_cols):
                if j not in pos:
                    continue
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_distance and dist > 0:
                    # Push nodes apart
                    move_x = dx * (min_distance - dist) / (2 * dist + 1e-6)
                    move_y = dy * (min_distance - dist) / (2 * dist + 1e-6)
                    pos[i] = (pos[i][0] + move_x, pos[i][1] + move_y)
                    pos[j] = (pos[j][0] - move_x, pos[j][1] - move_y)
                    moved = True
        if not moved:
            break
    
    # Draw nodes
    # Identify connected vs isolated nodes
    if G.edges():
        connected_nodes = set()
        for u, v in G.edges():
            connected_nodes.add(u)
            connected_nodes.add(v)
        # Calculate node degrees for size variation
        degrees = dict(G.degree())
    else:
        connected_nodes = set()
        degrees = {i: 0 for i in range(n_cols)}
    
    # Categorize nodes for legend
    root_connected = []
    root_isolated = []
    child_connected = []
    child_isolated = []
    
    node_colors = []
    node_sizes = []
    for i in range(n_cols):
        is_root = model.parents[i] == -1
        is_connected = i in connected_nodes
        
        # Color nodes based on whether they have a parent (child) or not (root)
        if is_root:
            if highlight_connected and is_connected:
                color = "steelblue"  # Connected root nodes (darker blue)
                root_connected.append(i)
            else:
                color = "lightblue"  # Isolated root nodes
                root_isolated.append(i)
        else:
            if highlight_connected and is_connected:
                color = "crimson"  # Connected child nodes (darker red)
                child_connected.append(i)
            else:
                color = "lightcoral"  # Isolated child nodes (shouldn't happen)
                child_isolated.append(i)
        
        node_colors.append(color)
        
        # Size based on degree (hubs are larger) and connection status
        # Make nodes large enough to fit the font size comfortably
        # Font sizes are: 20-28 for nodes, so nodes should be at least 2000-3000
        if n_nodes > 30:
            base_size = 2500  # Large enough for font size 20
        elif n_nodes > 15:
            base_size = 2800  # Large enough for font size 24
        else:
            base_size = 3200  # Large enough for font size 28
        
        degree = degrees.get(i, 0)
        if highlight_connected and is_connected:
            # Connected nodes are larger, and scale with degree
            size = base_size + degree * 300
        else:
            size = base_size * 0.9  # Isolated nodes still need to fit text
        node_sizes.append(size)
    
    # Draw edges first (so they're behind nodes)
    # Draw edges with weights as edge width
    if show_weights and G.edges():
        edges = list(G.edges())
        weights = [G[u][v].get("weight", 1.0) for u, v in edges]
        if weights:
            max_weight = max(weights)
            min_weight = min(weights)
            if max_weight > min_weight:
                # Normalize weights to [0.5, 3.0] for edge width
                edge_widths = [(w - min_weight) / (max_weight - min_weight) * 2.5 + 0.5 
                              for w in weights]
            else:
                edge_widths = [2.0] * len(edges)
            
            # Color edges dark green (varying intensity based on weight)
            if max_weight > min_weight:
                # Normalize weights to [0, 1] and map to dark green shades
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
                # Use dark green color map (darker green = higher weight)
                edge_colors = [plt.cm.Greens(0.3 + 0.7 * nw) for nw in normalized_weights]
            else:
                edge_colors = ["darkgreen"] * len(edges)
        else:
            edge_widths = [2.0] * len(edges)
            edge_colors = "gray"
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                             alpha=0.7, edge_color=edge_colors)
    
    # Draw nodes after edges (so nodes are on top)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.9, edgecolors="black", linewidths=3)
    
    # Draw edge labels after nodes but with offset to avoid overlap
    if show_weights and G.edges():
        edges = list(G.edges())
        weights = [G[u][v].get("weight", 1.0) for u, v in edges]
        if weights:
            # Add weight labels on edges - only for smaller graphs or important edges
            if len(edges) <= 15:  # Only show weights if not too many edges
                edge_labels = {(u, v): f"{w:.1f}" for (u, v), w in zip(edges, weights)}
                edge_font_size = 24 if n_nodes <= 15 else 22
                # Use networkx edge labels - it will position them correctly on edges
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, 
                                            font_size=edge_font_size,
                                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                                                     edgecolor="darkgreen", linewidth=2.0, alpha=0.95),
                                            rotate=False,  # Don't rotate labels
                                            )
            elif len(edges) <= 50:
                # For medium graphs, only label top 10% of edges by weight
                sorted_edges = sorted(zip(edges, weights), key=lambda x: x[1], reverse=True)
                top_n = max(5, len(edges) // 10)
                top_edges = sorted_edges[:top_n]
                edge_labels = {(u, v): f"{w:.1f}" for (u, v), w in top_edges}
                edge_font_size = 22 if n_nodes <= 30 else 20
                # Use networkx edge labels - it will position them correctly on edges
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, 
                                            font_size=edge_font_size,
                                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                                                     edgecolor="darkgreen", linewidth=2.0, alpha=0.95),
                                            rotate=False,  # Don't rotate labels
                                            )
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, width=2.0, alpha=0.6, edge_color="gray")
    
        # Draw labels
    if show_labels:
        if col_names and model.columns:
            # Use actual column names (truncate if too long)
            labels = {i: col[:15] + "..." if len(col) > 15 else col 
                     for i, col in enumerate(model.columns)}
        else:
            labels = {i: f"C{i}" for i in range(n_cols)}
        
        # Determine font size based on graph size
        if n_nodes > 30:
            base_font_size = 22
        elif n_nodes > 15:
            base_font_size = 26
        else:
            base_font_size = 28
        
        # Draw all labels with larger font
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=base_font_size, font_weight="bold")
    
    # Add title with statistics
    n_edges = len(G.edges())
    n_roots = sum(1 for p in model.parents if p == -1)
    n_children = n_cols - n_roots
    
    # Find connected components
    if G.edges():
        components = list(nx.connected_components(G))
        n_components = len(components)
        largest_component_size = max(len(c) for c in components) if components else 0
    else:
        n_components = n_cols  # All isolated
        largest_component_size = 1
    
    # Print statistics to terminal
    print(f"\nForest Structure Statistics for {tabcl_file.stem}:")
    print(f"  Columns: {n_cols}")
    print(f"  Edges: {n_edges}")
    print(f"  Roots: {n_roots}")
    print(f"  Connected Components: {n_components}", end="")
    if n_components < n_cols:
        print(f" (largest: {largest_component_size} columns)")
    else:
        print()
    
    if G.edges() and show_weights:
        weights = [G[u][v].get("weight", 0) for u, v in G.edges()]
        print(f"  Edge weight range: [{min(weights):.2f}, {max(weights):.2f}]")
    print()
    
    # Simple title
    ax.set_title("Forest Structure", fontsize=32, fontweight="bold", pad=25)
    ax.axis("off")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    
    if root_connected:
        legend_elements.append(Patch(facecolor='steelblue', edgecolor='black', 
                                    label=f'Root (connected) [{len(root_connected)}]'))
    if root_isolated:
        legend_elements.append(Patch(facecolor='lightblue', edgecolor='black', 
                                    label=f'Root (isolated) [{len(root_isolated)}]'))
    if child_connected:
        legend_elements.append(Patch(facecolor='crimson', edgecolor='black', 
                                    label=f'Child (connected) [{len(child_connected)}]'))
    if child_isolated:
        legend_elements.append(Patch(facecolor='lightcoral', edgecolor='black', 
                                    label=f'Child (isolated) [{len(child_isolated)}]'))
    
    if legend_elements:
        # Place legend outside the plot area to avoid covering nodes
        # Use upper left anchor but position it outside on the right side
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), 
                 frameon=True, fancybox=True, shadow=True, fontsize=24)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Forest visualization saved to {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the MST/forest graph from a .tabcl file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize and save to file
  python3 scripts/visualize_forest.py datasets/business_price/business-price-indexes-march-2022-quarter-csv.tabcl -o forest.png
  
  # Use circular layout
  python3 scripts/visualize_forest.py file.tabcl --layout circular
  
  # Hide labels and weights
  python3 scripts/visualize_forest.py file.tabcl --no-labels --no-weights
        """
    )
    parser.add_argument("input", type=Path, help="Path to .tabcl file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                       help="Output file path (if not specified, displays interactively)")
    parser.add_argument("--layout", choices=["spring", "circular", "hierarchical", "kamada_kawai", "force"],
                       default="spring", help="Graph layout algorithm (default: spring). 'force' uses force-directed layout with better spacing.")
    parser.add_argument("--no-labels", action="store_true",
                       help="Hide node labels")
    parser.add_argument("--no-weights", action="store_true",
                       help="Hide edge weights (use uniform edge width)")
    parser.add_argument("--no-col-names", action="store_true",
                       help="Use C0, C1, ... instead of column names")
    parser.add_argument("--no-highlight", action="store_true",
                       help="Don't highlight connected nodes differently")
    
    args = parser.parse_args()
    
    visualize_forest(
        args.input,
        output_file=args.output,
        layout=args.layout,
        show_labels=not args.no_labels,
        show_weights=not args.no_weights,
        col_names=not args.no_col_names,
        highlight_connected=not args.no_highlight
    )


if __name__ == "__main__":
    main()

