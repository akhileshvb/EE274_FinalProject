from typing import List, Tuple, Optional, Callable
import numpy as np
import networkx as nx
import inspect
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .mi import estimate_edge_weight, estimate_edge_weight_hashed, compute_empirical_mi


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
	# NOTE: For numeric/float columns, high cardinality is expected (continuous data),
	# so we should NOT skip them. Only skip high-cardinality categorical/string columns.
	for i in range(n_cols):
		col_data = table_sample[:, i]
		# Check if column is numeric - if so, don't skip based on cardinality
		is_numeric = False
		try:
			if isinstance(col_data, np.ndarray):
				if col_data.dtype.kind in ['f', 'i', 'u']:  # float, int, unsigned int
					is_numeric = True
				elif col_data.dtype == object:
					# For object arrays, check if values are numeric
					# Sample a few values to check
					sample_vals = col_data[:min(10, len(col_data))]
					try:
						# Try to convert to float - if it works, it's numeric
						_ = [float(v) for v in sample_vals if v is not None]
						is_numeric = True
					except (ValueError, TypeError):
						is_numeric = False
		except:
			pass
		
		# Only check cardinality for non-numeric columns
		if not is_numeric:
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

	# Debug: print all edge weights before MST
	# (commented out for production, but useful for debugging)
	# if len(G.edges()) > 0:
	#     print(f"DEBUG: Graph has {len(G.edges())} edges with weights:")
	#     for u, v, d in G.edges(data=True):
	#         print(f"  Edge ({u}, {v}): weight={d.get('weight', 0.0)}")
	
	mst = nx.maximum_spanning_tree(G)
	edges = [(u, v, float(d.get("weight", 0.0))) for u, v, d in mst.edges(data=True)]
	# Only filter out edges with negative weights - they hurt compression (MDL cost > benefit)
	# Keep all positive weight edges - even small benefits can accumulate
	edges = [(u, v, w) for u, v, w in edges if w > 0.0]
	return edges


def build_mdl_weighted_forest_two_phase(
	table: np.ndarray,
	mdl_cost_fn_fast: Callable,
	mdl_cost_fn_openzl: Callable,
	skip_high_card: set,
	mi_mode: str = "auto",
	mi_buckets: int = 4096,
	mi_sample: Optional[int] = None,
	mi_seed: int = 0,
) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]:
	"""
	Build a maximum-weight forest using a two-phase approach:
	1. Phase 1: Fast ranking with proxy MDL on all edge pairs.
	2. Phase 2: Refine top candidates using OpenZL MDL and exact MI.

	Returns:
	    Tuple[List[Tuple[int, int, float]], List[Tuple[int, int, float]]]: 
	        (phase1_edges, final_edges). 
	        phase1_edges are from fast ranking, final_edges are the refined edges after phase 2.
	This method keeps all positive-weight edges that do not create cycles, so the result
	is a maximum-weight forest (not just an MST).
	"""
	if table.ndim != 2:
		raise ValueError("table must be 2D")
	n_rows, n_cols = table.shape
	
	# Phase 1: Fast ranking with proxy MDL on all edge pairs
	# For datasets <= 10k rows, can afford to use exact MI throughout for better accuracy
	use_exact_mi_phase1 = (n_rows <= 10000)
	
	# Determine phase1_sample
	if mi_sample is None:
		if n_rows > 1_000_000:
			phase1_sample = min(100_000, n_rows // 35)
		elif n_rows > 100_000:
			phase1_sample = min(50_000, n_rows // 20)
		elif n_rows > 10_000:
			phase1_sample = min(10_000, n_rows // 2) if use_exact_mi_phase1 else min(5_000, n_rows // 4)
		elif n_rows > 1_000:
			phase1_sample = min(3_000, n_rows // 2) if use_exact_mi_phase1 else min(2_000, n_rows // 3)
		else:
			phase1_sample = None
	else:
		phase1_sample = mi_sample
	
	# Determine phase1_mi_mode
	if mi_mode == "auto":
		if n_rows <= 10000:
			phase1_mi_mode = "exact"
		else:
			phase1_mi_mode = "hashed"
	elif mi_mode == "exact":
		phase1_mi_mode = "exact"
	else:
		phase1_mi_mode = "hashed"
	
	# Determine phase1_buckets
	if phase1_mi_mode == "hashed":
		if n_rows > 1_000_000:
			phase1_buckets = min(mi_buckets * 2, 16384)
		elif n_rows > 100_000:
			phase1_buckets = min(mi_buckets, 8192)
		else:
			phase1_buckets = mi_buckets
	else:
		phase1_buckets = mi_buckets
	
	# Phase 1: compute weights for all edge pairs with proxy MDL
	def compute_fast_weight(pair):
		i, j = pair
		if i in skip_high_card or j in skip_high_card:
			return (i, j, -1.0)
		x, y = table[:, i], table[:, j]
		mdl_bits = mdl_cost_fn_fast(x, y, row_sample=phase1_sample, seed=mi_seed)
		if phase1_mi_mode == "hashed":
			w = estimate_edge_weight_hashed(n_rows, x, y, mdl_bits, num_buckets=phase1_buckets, row_sample=phase1_sample, seed=mi_seed)
		else:
			w = estimate_edge_weight(n_rows, x, y, mdl_bits)
		return (i, j, w)
	
	# Compute all edge pairs
	all_pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols) if i not in skip_high_card and j not in skip_high_card]
	max_workers = min(len(all_pairs), int(os.cpu_count() or 4), 16)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		all_fast_edges = list(executor.map(compute_fast_weight, all_pairs))
	
	# Filter negative weights and sort by weight
	all_fast_edges = [(u, v, w) for u, v, w in all_fast_edges if w > 0.0]
	all_fast_edges.sort(key=lambda x: x[2], reverse=True)
	
	# Phase 2: Refine top candidates with OpenZL MDL and EXACT MI
	max_candidates = min(len(all_fast_edges), max(n_cols * 25, 1500), 3000)
	top_candidates = all_fast_edges[:max_candidates]
	
	# Determine phase2_sample
	if n_rows <= 10000:
		phase2_sample = None
	elif n_rows > 1_000_000:
		phase2_sample = min(200_000, n_rows // 20)
	elif n_rows > 100_000:
		phase2_sample = min(100_000, n_rows // 10)
	else:
		phase2_sample = min(n_rows, max(20_000, n_rows // 5))
	
	def recompute_weight(edge):
		u, v, _ = edge
		x_full, y_full = table[:, u], table[:, v]
		rng = np.random.default_rng(mi_seed)
		if phase2_sample is not None and phase2_sample < n_rows:
			sample_idx = rng.choice(n_rows, size=phase2_sample, replace=False)
			x_sample = x_full[sample_idx]
			y_sample = y_full[sample_idx]
		else:
			x_sample = x_full
			y_sample = y_full
		# Use OpenZL MDL for accurate model cost
		mdl_bits = mdl_cost_fn_openzl(x_sample, y_sample, row_sample=None, seed=mi_seed)
		# Use EXACT MI for accurate mutual information
		mi_nats = compute_empirical_mi(x_sample, y_sample)
		mi_bits = mi_nats / np.log(2.0)
		# Edge weight: n_rows * I(X;Y) - model_cost
		w = n_rows * mi_bits - mdl_bits
		return (u, v, w)
	
	max_workers = min(len(top_candidates), int(os.cpu_count() or 4), 8)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		refined_edges = list(executor.map(recompute_weight, top_candidates))
	
	# Filter negative weights
	refined_edges = [(u, v, w) for u, v, w in refined_edges if w > 0.0]
	
	# Return phase1 edges and refined edges separately for profiling
	return all_fast_edges, refined_edges


def build_maximum_weight_forest(
	n_cols: int,
	edges: List[Tuple[int, int, float]],
) -> List[Tuple[int, int, float]]:
	"""
	Build a maximum-weight forest from a list of edges.
	Keeps all positive-weight edges that don't create cycles (greedy algorithm).
	This is optimal: we get maximum total weight while maintaining acyclic structure.
	"""
	import networkx as nx
	G = nx.Graph()
	G.add_nodes_from(range(n_cols))
	# Sort edges by weight (descending) and add them greedily - this is optimal for forests
	edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
	forest_edges = []
	for u, v, w in edges_sorted:
		# Check if adding this edge would create a cycle
		# If u and v are in different connected components, it's safe to add
		if not nx.has_path(G, u, v):
			G.add_edge(u, v, weight=w)
			forest_edges.append((u, v, w))
	
	return forest_edges
