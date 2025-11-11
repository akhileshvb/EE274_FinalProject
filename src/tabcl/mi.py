import numpy as np
import pandas as pd
from typing import Optional


def compute_empirical_mi(x: np.ndarray, y: np.ndarray) -> float:
	"""
	Empirical mutual information I(X;Y) in nats for discrete arrays.
	Optimized for performance with high-cardinality data.
	"""
	if x.shape[0] != y.shape[0]:
		raise ValueError("x and y must have same length")
	if x.size == 0:
		return 0.0

	# Use factorize for efficient encoding
	x_codes, x_uniques = pd.factorize(x, sort=False)
	y_codes, y_uniques = pd.factorize(y, sort=False)
	n = float(x_codes.shape[0])
	
	# Optimize: use numpy for faster contingency table construction
	# For high-cardinality, use sparse representation if beneficial
	x_max = x_codes.max() + 1
	y_max = y_codes.max() + 1
	
	# Build contingency table efficiently
	# Use bincount for speed when possible
	if x_max * y_max < 10_000_000:  # Dense representation if not too large
		# Build 2D contingency table
		jt = np.zeros((x_max, y_max), dtype=np.int64)
		np.add.at(jt, (x_codes, y_codes), 1)
		pxy = jt.astype(float) / n
		px = pxy.sum(axis=1, keepdims=True)
		py = pxy.sum(axis=0, keepdims=True)
	else:
		# For very high cardinality, use pandas crosstab (more memory efficient)
		jt = pd.crosstab(x_codes, y_codes)
		pxy = jt.to_numpy(dtype=float) / n
		px = pxy.sum(axis=1, keepdims=True)
		py = pxy.sum(axis=0, keepdims=True)

	with np.errstate(divide="ignore", invalid="ignore"):
		ratio = pxy / (px @ py)
		mask = pxy > 0
		mi = np.sum(pxy[mask] * np.log(ratio[mask]))
	return float(mi)


def compute_hashed_mi(
	x: np.ndarray,
	y: np.ndarray,
	num_buckets: int = 4096,
	row_sample: Optional[int] = None,
	seed: int = 0,
) -> float:
	"""
	Approximate I(X;Y) in nats using hashing and optional row sampling.
	- Values are hashed into num_buckets; we compute MI on the hashed contingency.
	- If row_sample is provided and smaller than len(x), we sample rows without replacement.
	"""
	if x.shape[0] != y.shape[0]:
		raise ValueError("x and y must have same length")
	n_total = x.shape[0]
	idx = np.arange(n_total)
	if row_sample is not None and row_sample < n_total:
		rng = np.random.default_rng(seed)
		idx = rng.choice(idx, size=row_sample, replace=False)
	xv = x[idx]
	yv = y[idx]

	# Hash to buckets - optimize for speed with vectorized operations
	# Convert to string array once, then hash
	if len(xv) > 1000:
		# For large arrays, use list comprehension (faster for large datasets)
		hx = np.array([hash(str(v)) % num_buckets for v in xv], dtype=np.int64)
		hy = np.array([hash(str(v)) % num_buckets for v in yv], dtype=np.int64)
	else:
		# For smaller arrays, vectorized approach
		hx = np.mod(np.array([hash(str(v)) for v in xv], dtype=np.int64), num_buckets)
		hy = np.mod(np.array([hash(str(v)) for v in yv], dtype=np.int64), num_buckets)

	# Build contingency table using advanced indexing (fast)
	C = np.zeros((num_buckets, num_buckets), dtype=np.int64)
	np.add.at(C, (hx, hy), 1)
	n = float(C.sum())
	if n == 0:
		return 0.0
	pxy = C.astype(float) / n
	px = pxy.sum(axis=1, keepdims=True)
	py = pxy.sum(axis=0, keepdims=True)
	with np.errstate(divide="ignore", invalid="ignore"):
		ratio = pxy / (px @ py)
		mask = pxy > 0
		mi = np.sum(pxy[mask] * np.log(ratio[mask]))
	return float(mi)


def estimate_edge_weight(n_rows: int, x: np.ndarray, y: np.ndarray, mdl_bits: float) -> float:
	"""
	Return n * I(X;Y) (in bits) minus model description length (bits).
	"""
	mi_nats = compute_empirical_mi(x, y)
	mi_bits = mi_nats / np.log(2.0)
	return n_rows * mi_bits - mdl_bits


def estimate_edge_weight_hashed(
	n_rows: int,
	x: np.ndarray,
	y: np.ndarray,
	mdl_bits: float,
	num_buckets: int = 4096,
	row_sample: Optional[int] = None,
	seed: int = 0,
) -> float:
	mi_nats = compute_hashed_mi(x, y, num_buckets=num_buckets, row_sample=row_sample, seed=seed)
	mi_bits = mi_nats / np.log(2.0)
	return n_rows * mi_bits - mdl_bits
