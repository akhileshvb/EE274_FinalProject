import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add parent directory to path to import tabcl.
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tabcl.cli import Model


def visualize_forest(tabcl_file: Path, output_file: Path = None, layout: str = "spring", 
                     show_labels: bool = True, show_weights: bool = True, 
                     col_names: bool = True, highlight_connected: bool = True):
    """Visualize the forest structure stored in a .tabcl file."""
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
    
    # Directed graph: parent -> child.
    G = nx.DiGraph()
    n_cols = len(model.columns)
    G.add_nodes_from(range(n_cols))
    
    # Add edges based on parent relationships.
    for child_idx, parent_idx in enumerate(model.parents):
        if parent_idx >= 0:  # Has a parent
            # Find the edge weight from model.edges
            weight = 1.0
            for edge in model.edges:
                if len(edge) >= 3:
                    u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
                    if (u == parent_idx and v == child_idx) or (u == child_idx and v == parent_idx):
                        weight = w
                        break
            G.add_edge(parent_idx, child_idx, weight=weight)
    
    # Also add edges from model.edges for completeness.
    for edge in model.edges:
        if len(edge) >= 3:
            u, v, w = int(edge[0]), int(edge[1]), float(edge[2])
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                # Add as undirected edge if not already in parent structure
                G.add_edge(u, v, weight=w)
        elif len(edge) == 2:
            u, v = int(edge[0]), int(edge[1])
            if not G.has_edge(u, v) and not G.has_edge(v, u):
                G.add_edge(u, v, weight=1.0)
    
    if len(G.edges()) == 0:
        print("Warning: No edges found in forest (all columns are independent)", file=sys.stderr)
    
    # Figure size scales with number of nodes.
    n_nodes = len(G.nodes())
    if n_nodes > 30:
        figsize = (18, 14)
    elif n_nodes > 15:
        figsize = (16, 12)
    else:
        figsize = (14, 10)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout is computed on an undirected copy.
    G_layout = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    if layout == "spring":
        k = max(3.0, np.sqrt(n_nodes) * 0.8) if n_nodes > 20 else 2.0
        pos = nx.spring_layout(G_layout, k=k, iterations=100, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G_layout)
    elif layout == "hierarchical":
        # Create a hierarchical layout based on parent relationships (best for showing learned tree)
        try:
            # Use parent array to create hierarchy
            pos = {}
            roots = [i for i, p in enumerate(model.parents) if p == -1]
            if roots:
                # Build level structure from parent relationships
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
                # Distribute nodes at each level horizontally
                level_nodes = {}
                for i in range(n_cols):
                    level = levels.get(i, 0)
                    if level not in level_nodes:
                        level_nodes[level] = []
                    level_nodes[level].append(i)
                
                # Position nodes: roots at top, children below
                # Use better spacing to prevent overlap - significantly increased for much wider spread
                min_node_spacing = 10.0  # Minimum horizontal spacing between nodes (significantly increased)
                vertical_spacing = 3.0  # Vertical spacing between levels (increased from 2.0)
                
                for level in sorted(level_nodes.keys()):
                    nodes_at_level = level_nodes[level]
                    y = (max_level - level) * vertical_spacing  # Roots at top, children below
                    
                    # Calculate spacing to ensure no overlap
                    # For roots (level 0), ensure they're well-spaced
                    if level == 0:
                        # Roots: space them evenly with good separation
                        total_width = max(len(nodes_at_level) * min_node_spacing, 25.0)
                        spacing = total_width / max(len(nodes_at_level), 1)
                    else:
                        # For other levels, use adaptive spacing based on number of nodes
                        # Use much wider spacing for levels with more nodes
                        total_width = max(len(nodes_at_level) * min_node_spacing, 35.0)
                        spacing = total_width / max(len(nodes_at_level), 1)
                    
                    # Center the nodes horizontally
                    start_x = -(len(nodes_at_level) - 1) * spacing / 2
                    for idx, node in enumerate(nodes_at_level):
                        x = start_x + idx * spacing
                        pos[node] = (x, y)
            else:
                # Fallback if no roots found
                pos = nx.spring_layout(G_layout, k=2, iterations=50, seed=42)
        except Exception as e:
            print(f"Warning: Hierarchical layout failed: {e}, using spring layout", file=sys.stderr)
            pos = nx.spring_layout(G_layout, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_layout)
    elif layout == "force":
        # Force-directed layout with better parameters for dense graphs
        k = max(4.0, np.sqrt(n_nodes) * 1.2) if n_nodes > 20 else 3.0
        pos = nx.spring_layout(G_layout, k=k, iterations=200, seed=42)
    else:
        k = max(3.0, np.sqrt(n_nodes) * 0.8) if n_nodes > 20 else 2.0
        pos = nx.spring_layout(G_layout, k=k, iterations=100, seed=42)
    
    # Post-process layout to prevent node overlap
    # Increase minimum distance between nodes (larger for better separation)
    # Account for node sizes - nodes are large (2500-3200 in node_size units)
    # In layout coordinates, we need at least 1.5-2.0 units of separation for good visibility
    min_distance = 1.8  # Increased from 1.0 to spread nodes out more
    max_iterations = 100  # More iterations for better separation
    for iteration in range(max_iterations):
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
                    # Push nodes apart more aggressively
                    overlap = min_distance - dist
                    move_x = (dx / (dist + 1e-6)) * overlap * 0.6  # Move each node 60% of overlap
                    move_y = (dy / (dist + 1e-6)) * overlap * 0.6
                    pos[i] = (pos[i][0] + move_x, pos[i][1] + move_y)
                    pos[j] = (pos[j][0] - move_x, pos[j][1] - move_y)
                    moved = True
        if not moved:
            break
    
    # Additional pass: ensure roots (level 0) are well-separated
    if layout == "hierarchical":
        roots = [i for i, p in enumerate(model.parents) if p == -1]
        if len(roots) > 1:
            # Sort roots by x position
            root_positions = [(i, pos[i]) for i in roots if i in pos]
            root_positions.sort(key=lambda x: x[1][0])
            
            # Ensure minimum spacing between roots (increased for more spread)
            root_min_spacing = 10.0  # Increased for much wider spacing
            for idx in range(1, len(root_positions)):
                prev_node, prev_pos = root_positions[idx - 1]
                curr_node, curr_pos = root_positions[idx]
                dx = curr_pos[0] - prev_pos[0]
                if abs(dx) < root_min_spacing:
                    # Move current root to the right
                    new_x = prev_pos[0] + root_min_spacing
                    pos[curr_node] = (new_x, curr_pos[1])
                    root_positions[idx] = (curr_node, (new_x, curr_pos[1]))
    
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
    # Draw edges with weights as edge width and arrows to show parent->child direction
    if G.edges():
        edges = list(G.edges())
        weights = [G[u][v].get("weight", 1.0) for u, v in edges]
        if weights and show_weights:
            max_weight = max(weights)
            min_weight = min(weights)
            if max_weight > min_weight:
                # Normalize weights to [1.0, 4.0] for edge width (thicker = stronger dependency)
                edge_widths = [(w - min_weight) / (max_weight - min_weight) * 3.0 + 1.0 
                              for w in weights]
            else:
                edge_widths = [2.5] * len(edges)
            
            # Color edges dark green (varying intensity based on weight)
            if max_weight > min_weight:
                # Normalize weights to [0, 1] and map to dark green shades
                normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in weights]
                # Use dark green color map (darker green = higher weight = stronger dependency)
                edge_colors = [plt.cm.Greens(0.3 + 0.7 * nw) for nw in normalized_weights]
            else:
                edge_colors = ["darkgreen"] * len(edges)
        else:
            edge_widths = [2.5] * len(edges)
            edge_colors = "darkgreen"
        
        # Draw directed edges with arrows to show parent->child relationships
        # Connect arrows directly to nodes (no margin)
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                             alpha=0.8, edge_color=edge_colors,
                             arrows=True, arrowsize=30, arrowstyle='->',
                             connectionstyle='arc3,rad=0.1',  # Slight curve for better visibility
                             node_size=node_sizes,  # Pass node sizes for proper arrow positioning
                             min_source_margin=0, min_target_margin=0)  # Connect directly to nodes
    
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
        # Draw edges without weights (uniform)
        # Connect arrows directly to nodes (no margin)
        nx.draw_networkx_edges(G, pos, ax=ax, width=2.5, alpha=0.7, edge_color="darkgreen",
                             arrows=True, arrowsize=30, arrowstyle='->',
                             connectionstyle='arc3,rad=0.1',
                             node_size=node_sizes,
                             min_source_margin=0, min_target_margin=0)  # Connect directly to nodes
    
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
    
    # Find connected components (convert to undirected for this check)
    if G.edges():
        G_undirected = G.to_undirected()
        components = list(nx.connected_components(G_undirected))
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
    
    # Title showing this is the learned tree structure
    title = "Learned Dependency Tree Structure"
    if n_components > 1:
        title += f" ({n_components} trees)"
    ax.set_title(title, fontsize=32, fontweight="bold", pad=25)
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
                       default="hierarchical", help="Graph layout algorithm (default: hierarchical). 'hierarchical' shows parent->child relationships best.")
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

