import numpy as np
import pandas as pd
from typing import Optional


def compute_empirical_mi(x: np.ndarray, y: np.ndarray) -> float:
	"""
	Empirical mutual information I(X;Y) in nats for discrete arrays.
	"""
	if x.shape[0] != y.shape[0]:
		raise ValueError("x and y must have same length")
	if x.size == 0:
		return 0.0

	x_codes, _ = pd.factorize(x, sort=False)
	y_codes, _ = pd.factorize(y, sort=False)
	n = float(x_codes.shape[0])

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

	# Hash to buckets; convert to integers first via string to stabilize across runs
	hx = np.mod(np.vectorize(lambda v: hash(str(v)))(xv).astype(np.int64), num_buckets)
	hy = np.mod(np.vectorize(lambda v: hash(str(v)))(yv).astype(np.int64), num_buckets)

	# Build contingency table
	C = np.zeros((num_buckets, num_buckets), dtype=np.int64)
	for a, b in zip(hx.tolist(), hy.tolist()):
		C[a, b] += 1
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
