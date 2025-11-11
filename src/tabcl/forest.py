from typing import List, Tuple, Optional, Callable
import numpy as np
import networkx as nx
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
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

	# Skip edges involving very high-cardinality columns (near-unique) - they compress better independently
	# Use very fast approximation: just check a tiny sample
	skip_high_card = set()
	rng = np.random.default_rng(seed)
	# Use tiny sample for speed - just need to identify obviously high-cardinality columns
	card_check_sample = min(500, n_rows) if n_rows > 500 else n_rows
	if card_check_sample < n_rows:
		card_idx = rng.choice(n_rows, size=card_check_sample, replace=False)
		table_sample = table[card_idx]
	else:
		table_sample = table
	
	# Quick cardinality check - skip high-cardinality columns early
	# Use approximation: if even a small sample has high uniqueness, skip the column
	for i in range(n_cols):
		col_data = table_sample[:, i]
		try:
			col_unique = len(np.unique(col_data))
		except (TypeError, ValueError):
			# For mixed types, use small sample for speed
			sample_size = min(100, len(col_data))
			if sample_size < len(col_data):
				sample_idx = rng.choice(len(col_data), size=sample_size, replace=False)
				col_unique_approx = len(set(col_data[sample_idx].tolist()))
				# Estimate: if sample has high uniqueness, column is high-cardinality
				col_unique = col_unique_approx * (len(col_data) / sample_size) if col_unique_approx > sample_size * 0.5 else col_unique_approx
			else:
				if col_data.dtype == object:
					col_unique = len(set(col_data.tolist()))
				else:
					col_unique = len(set(col_data.flatten().tolist()))
		# Skip if >50% unique in sample (columns are likely identifiers)
		threshold = int(0.5 * card_check_sample)
		if col_unique > threshold:
			skip_high_card.add(i)
	
	# Prepare edge pairs to compute
	edge_pairs = []
	for i in range(n_cols):
		for j in range(i + 1, n_cols):
			if i not in skip_high_card and j not in skip_high_card:
				edge_pairs.append((i, j))
	
	# Create a wrapper that passes sampling parameters to mdl_cost_fn if it accepts them
	def mdl_cost_wrapper(x, y, mdl_sample_size: Optional[int] = None):
		"""Use smaller sample for MDL cost to speed up computation."""
		# Use smaller sample for MDL cost estimation (it's just for ranking edges)
		effective_mdl_sample = mdl_sample_size or (row_sample // 2 if row_sample else None)
		try:
			sig = inspect.signature(mdl_cost_fn)
			if 'row_sample' in sig.parameters:
				return mdl_cost_fn(x, y, row_sample=effective_mdl_sample, seed=seed)
		except (ValueError, TypeError):
			# Function might be a builtin or not inspectable
			pass
		return mdl_cost_fn(x, y)
	
	def compute_edge_weight(i: int, j: int) -> Tuple[int, int, float]:
		"""Compute edge weight for a single pair of columns."""
		x, y = table[:, i], table[:, j]
		# Use moderate samples for speed - balance accuracy and performance
		if n_rows > 100_000:
			mdl_sample = min(row_sample or 5000, n_rows, 5000)
		elif n_rows > 10_000:
			mdl_sample = min(row_sample or 3000, n_rows, 3000)  # Moderate sample for speed
		elif n_rows > 1_000:
			mdl_sample = min(row_sample or 2000, n_rows, 2000)
		else:
			# For very small datasets, use full data
			mdl_sample = row_sample if row_sample and row_sample < n_rows else None
		mdl_bits = mdl_cost_wrapper(x, y, mdl_sample_size=mdl_sample)
		if mi_mode == "hashed":
			# Use same sample for MI as for MDL to be consistent
			w = estimate_edge_weight_hashed(n_rows, x, y, mdl_bits, num_buckets=num_buckets, row_sample=mdl_sample, seed=seed)
		else:
			w = estimate_edge_weight(n_rows, x, y, mdl_bits)
		return (i, j, w)
	
	# Parallelize edge weight computation - always use parallel for speed
	import os
	max_workers = min(len(edge_pairs), int(os.cpu_count() or 4), 16)  # Cap at 16 to avoid overhead
	
	if max_workers > 1 and len(edge_pairs) > 1:
		# Use parallel computation - always parallelize for speed
		with ThreadPoolExecutor(max_workers=max_workers) as executor:
			futures = {executor.submit(compute_edge_weight, i, j): (i, j) for i, j in edge_pairs}
			for future in as_completed(futures):
				try:
					i, j, w = future.result()
					G.add_edge(i, j, weight=w)
				except Exception as e:
					# Skip edges that fail (shouldn't happen, but be defensive)
					pass
	else:
		# Sequential only for single edge (unlikely)
		for i, j in edge_pairs:
			_, _, w = compute_edge_weight(i, j)
			G.add_edge(i, j, weight=w)

	mst = nx.maximum_spanning_tree(G)
	edges = [(u, v, float(d.get("weight", 0.0))) for u, v, d in mst.edges(data=True)]
	# Only filter out edges with negative weights - they hurt compression (MDL cost > benefit)
	# Keep all positive weight edges - even small benefits can accumulate
	edges = [(u, v, w) for u, v, w in edges if w > 0.0]
	return edges
