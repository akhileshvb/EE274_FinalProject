from typing import List, Tuple, Optional
import numpy as np
import networkx as nx
from .mi import estimate_edge_weight, estimate_edge_weight_hashed


def build_mdl_weighted_forest(
	table: np.ndarray,
	mdl_cost_fn,
	mi_mode: str = "exact",
	num_buckets: int = 4096,
	row_sample: Optional[int] = None,
	seed: int = 0,
) -> List[Tuple[int, int, float]]:
	"""
	Build a maximum spanning forest on columns using MDL-weighted edges.
	mi_mode: 'exact' or 'hashed'. If 'hashed', use bucketed MI with optional row_sample.
	"""
	if table.ndim != 2:
		raise ValueError("table must be 2D")
	n_rows, n_cols = table.shape

	G = nx.Graph()
	G.add_nodes_from(range(n_cols))

	for i in range(n_cols):
		for j in range(i + 1, n_cols):
			mdl_bits = mdl_cost_fn(table[:, i], table[:, j])
			if mi_mode == "hashed":
				w = estimate_edge_weight_hashed(n_rows, table[:, i], table[:, j], mdl_bits, num_buckets=num_buckets, row_sample=row_sample, seed=seed)
			else:
				w = estimate_edge_weight(n_rows, table[:, i], table[:, j], mdl_bits)
			G.add_edge(i, j, weight=w)

	mst = nx.maximum_spanning_tree(G)
	edges = [(u, v, float(d.get("weight", 0.0))) for u, v, d in mst.edges(data=True)]
	return edges
