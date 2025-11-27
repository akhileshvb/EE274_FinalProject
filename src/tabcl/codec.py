from typing import Any, Dict, Iterable, Tuple, Optional
import json
import numpy as np
import threading
from collections import Counter

try:
	import openzl.ext as zl  # type: ignore
except Exception:
	zl = None  # type: ignore


def histogram_from_pairs(x: Iterable[Any], y: Iterable[Any]) -> Dict[Tuple[Any, Any], int]:
	"""Build histogram of (x, y) pairs. Optimized for large datasets with vectorized operations."""
	# Fast path for numpy arrays - use vectorized operations when possible
	if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
		# For integer arrays with reasonable range, use bincount (much faster)
		if (x.dtype.kind in 'iu' and y.dtype.kind in 'iu' and 
		    len(x) > 1000 and x.max() < 1000000 and y.max() < 1000000 and
		    x.min() >= 0 and y.min() >= 0):
			# Use bincount for fast histogram construction
			x_max = int(x.max()) + 1
			y_max = int(y.max()) + 1
			# Flatten 2D index: idx = x * y_max + y
			flat_idx = x.astype(np.int64) * y_max + y.astype(np.int64)
			counts_flat = np.bincount(flat_idx, minlength=x_max * y_max)
			# Convert back to dictionary
			result = {}
			for i in range(x_max):
				for j in range(y_max):
					idx = i * y_max + j
					if counts_flat[idx] > 0:
						result[(i, j)] = int(counts_flat[idx])
			return result
		# For other cases, use Counter but optimize
		if len(x) < 10000:
			# Small arrays: convert to list of tuples (Counter is fast)
			return dict(Counter(zip(x, y)))
		else:
			# Larger arrays: use Counter directly on zip (memory efficient)
			return dict(Counter(zip(x, y)))
	else:
		# Non-numpy: use Counter
		return dict(Counter(zip(x, y)))


def _gamma_bits(v: int) -> int:
	if v <= 0:
		return 1
	l = int(np.floor(np.log2(v))) + 1
	return 2 * l - 1


def _golomb_bits(v: int, m: int) -> int:
	"""
	Compute Golomb encoding length in bits for value v with parameter m.
	Golomb code: unary part + binary part
	- Unary: floor(v/m) ones + one zero = floor(v/m) + 1 bits
	- Binary: k bits where k = ceil(log2(m)) for truncated binary or log2(m) for remainder
	Optimal m ≈ -ln(2)/ln(p) where p is probability parameter
	"""
	if v <= 0:
		return 1
	if m <= 1:
		return _gamma_bits(v)  # Fallback to gamma if m too small
	q = v // m
	b = int(np.ceil(np.log2(m)))
	# Truncated binary encoding for remainder
	rem = v % m
	if m != (1 << b):
		# Truncated binary: first 2^b - m values use b-1 bits, rest use b bits
		bound = (1 << b) - m
		if rem < bound:
			binary_bits = b - 1
		else:
			binary_bits = b
	else:
		binary_bits = b
	return (q + 1) + binary_bits


def _optimal_golomb_m(counts: Dict[Tuple[Any, Any], int]) -> int:
	"""
	Estimate optimal Golomb parameter m for counts.
	For geometric distribution with parameter p, optimal m = ceil(-ln(2) / ln(1-p))
	We estimate p from mean count or use heuristic.
	"""
	if not counts:
		return 256  # Default
	all_counts = list(counts.values())
	if not all_counts:
		return 256
	mean_count = np.mean(all_counts)
	if mean_count < 1:
		return 256
	# Heuristic: if mean is large, use larger m; if sparse, use smaller m
	# Optimal m for geometric with mean μ is approximately 0.693 * μ
	m_opt = max(1, int(0.693 * mean_count))
	# Round to nearest power of 2 for simpler implementation
	# Clamp to reasonable range
	m_opt = min(max(m_opt, 8), 8192)
	return m_opt


def proxy_model_bits(counts: Dict[Tuple[Any, Any], int]) -> float:
	m_hist = len(counts)
	if m_hist == 0:
		return 0.0
	# Use gamma for histogram size, Golomb for counts
	bits = _gamma_bits(m_hist)
	# Determine if counts are sparse (many small values) - use Golomb
	# Otherwise use gamma
	if m_hist > 0:
		all_counts = list(counts.values())
		mean_count = np.mean(all_counts) if all_counts else 1
		# If mean is small (sparse), Golomb is better
		if mean_count > 1 and mean_count < 100:  # Sparse regime
			golomb_m = _optimal_golomb_m(counts)
			for (_, _), c in counts.items():
				bits += _golomb_bits(c, golomb_m)
		else:
			# Not sparse enough, use gamma
			for (_, _), c in counts.items():
				bits += _gamma_bits(c + 1)
	return float(bits)


def _build_generic_compressor() -> "zl.Compressor":
	c = zl.Compressor()
	g = zl.graphs.Compress()(c)
	c.select_starting_graph(g)
	return c


def _build_ace_numeric_compressor() -> "zl.Compressor":
	"""
	Build a simple graph favoring low-cardinality numeric sequences using Tokenize->indices Compress.
	Falls back to generic graph if ACE graph construction fails.
	"""
	try:
		c = zl.Compressor()
		# Tokenize numeric stream; send indices to generic compressor; alphabet to generic as well
		tokenize = zl.nodes.Tokenize(type=zl.Type.Numeric, sort=True)
		graph = tokenize(c, alphabet=zl.graphs.Compress(), indices=zl.graphs.Compress())
		c.select_starting_graph(graph)
		return c
	except Exception:
		return _build_generic_compressor()


def _build_delta_numeric_compressor() -> "zl.Compressor":
	try:
		c = zl.Compressor()
		delta = zl.nodes.DeltaInt()
		graph = delta(c, successor=zl.graphs.Compress())
		c.select_starting_graph(graph)
		return c
	except Exception:
		return _build_generic_compressor()


# Thread-local compressor cache for MDL cost computation (reuse compressors for speed)
_compressor_cache = threading.local()

def _get_cached_ace_compressor():
	"""Get or create a cached ACE compressor for MDL cost computation."""
	if not hasattr(_compressor_cache, 'ace_compressor'):
		_compressor_cache.ace_compressor = _build_ace_numeric_compressor()
	return _compressor_cache.ace_compressor

def openzl_model_bits(counts: Dict[Tuple[Any, Any], int]) -> float:
	if zl is None:
		return proxy_model_bits(counts)
	# Encode histogram as a flat int64 triple array [a0,b0,c0, a1,b1,c1, ...]
	pairs = list(counts.items())
	m = len(pairs)
	if m == 0:
		return 0.0
	
	# For very small histograms (high correlation), use a more efficient encoding
	# This reduces model cost for highly correlated data, making edges more attractive
	# Increased threshold to 20 to capture more high-correlation cases
	if m <= 20:
		# For small histograms, use proxy_model_bits which is more efficient for small counts
		# This helps tabcl recognize high-correlation edges as beneficial
		return proxy_model_bits(counts)
	
	flat = np.empty(3 * m, dtype=np.int64)
	k = 0
	for (a, b), c in pairs:
		# stringify to stabilize mapping, then hash to int64 domain for robustness
		flat[k] = np.int64(zl.hash_name(str(a))) if hasattr(zl, "hash_name") else np.int64(abs(hash(str(a))) & ((1<<63)-1)); k += 1
		flat[k] = np.int64(zl.hash_name(str(b))) if hasattr(zl, "hash_name") else np.int64(abs(hash(str(b))) & ((1<<63)-1)); k += 1
		flat[k] = np.int64(c); k += 1
	# Use cached compressor for speed (reuse across calls)
	c = _get_cached_ace_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	out = cctx.compress([zl.Input(zl.Type.Numeric, flat)])
	return float(len(out) * 8)


def mdl_cost_fn_fast(x: np.ndarray, y: np.ndarray, row_sample: Optional[int] = None, seed: int = 0) -> float:
	"""
	Fast MDL cost estimation using proxy_model_bits (no OpenZL compression).
	Use this for forest building where we just need to rank edges.
	Optimized with vectorized histogram construction.
	"""
	# If row_sample is None, use data as-is (caller may have already sampled)
	if row_sample is not None and row_sample < len(x):
		rng = np.random.default_rng(seed)
		idx = rng.choice(len(x), size=row_sample, replace=False)
		x_sampled = x[idx]
		y_sampled = y[idx]
	else:
		x_sampled = x
		y_sampled = y
	
	# Fast path: use vectorized bincount for integer arrays
	if (isinstance(x_sampled, np.ndarray) and isinstance(y_sampled, np.ndarray) and
	    x_sampled.dtype.kind in 'iu' and y_sampled.dtype.kind in 'iu' and
	    len(x_sampled) > 1000 and x_sampled.max() < 1000000 and y_sampled.max() < 1000000 and
	    x_sampled.min() >= 0 and y_sampled.min() >= 0):
		# Direct bincount approach (fastest)
		x_max = int(x_sampled.max()) + 1
		y_max = int(y_sampled.max()) + 1
		flat_idx = x_sampled.astype(np.int64) * y_max + y_sampled.astype(np.int64)
		counts_flat = np.bincount(flat_idx, minlength=x_max * y_max)
		# Convert to dict format for proxy_model_bits
		counts = {}
		for i in range(x_max):
			for j in range(y_max):
				idx = i * y_max + j
				if counts_flat[idx] > 0:
					counts[(i, j)] = int(counts_flat[idx])
		return proxy_model_bits(counts)
	
	# Fallback to histogram_from_pairs for other cases
	counts = histogram_from_pairs(x_sampled, y_sampled)
	return proxy_model_bits(counts)


def mdl_cost_fn_openzl(x: np.ndarray, y: np.ndarray, row_sample: Optional[int] = None, seed: int = 0) -> float:
	"""
	Compute MDL cost for pairwise distribution using OpenZL compression.
	Slower but more accurate. Use for final compression, not for forest building.
	
	For continuous numeric data, we discretize/bucket the values to avoid creating
	huge histograms with many unique pairs. This makes MDL cost more reasonable.
	"""
	rng = np.random.default_rng(seed)  # Always create RNG for potential use in discretization
	if row_sample is not None and row_sample < len(x):
		idx = rng.choice(len(x), size=row_sample, replace=False)
		x_sampled = x[idx]
		y_sampled = y[idx]
	else:
		x_sampled = x
		y_sampled = y
	
	# For continuous numeric data, discretize before building histogram
	# This prevents huge histograms with many unique float pairs
	# Use quantile-based bucketing which preserves correlation structure better than hash-based
	# Hash-based bucketing destroys correlation because it randomizes values
	if isinstance(x_sampled, np.ndarray) and isinstance(y_sampled, np.ndarray):
		if x_sampled.dtype.kind == 'f' or y_sampled.dtype.kind == 'f':  # Float type
			# Use quantile-based bucketing to preserve correlation structure
			# This is better for numeric data than hash-based bucketing
			# Use fewer buckets to reduce MDL cost - 256 is enough to preserve correlation for most cases
			num_buckets = 256  # Reduced to 256 to significantly reduce MDL cost while still preserving correlation
			
			# Discretize x using quantile-based bucketing
			if x_sampled.dtype.kind == 'f':
				# For large arrays, sample to compute quantiles efficiently
				if len(x_sampled) > 10000:
					sample_idx = rng.choice(len(x_sampled), size=min(10000, len(x_sampled)), replace=False)
					x_sample_for_quantiles = x_sampled[sample_idx]
				else:
					x_sample_for_quantiles = x_sampled
				# Compute quantiles and use them as bucket boundaries
				quantiles = np.linspace(0, 100, num_buckets + 1)
				x_percentiles = np.percentile(x_sample_for_quantiles, quantiles)
				# Handle edge case where all values are the same
				if np.all(x_percentiles == x_percentiles[0]):
					x_discrete = np.zeros_like(x_sampled, dtype=np.int64)
				else:
					x_discrete = np.digitize(x_sampled, x_percentiles[1:-1], right=False)
					x_discrete = np.clip(x_discrete, 0, num_buckets - 1)
			else:
				x_discrete = x_sampled
			
			# Discretize y using quantile-based bucketing
			if y_sampled.dtype.kind == 'f':
				if len(y_sampled) > 10000:
					sample_idx = rng.choice(len(y_sampled), size=min(10000, len(y_sampled)), replace=False)
					y_sample_for_quantiles = y_sampled[sample_idx]
				else:
					y_sample_for_quantiles = y_sampled
				quantiles = np.linspace(0, 100, num_buckets + 1)
				y_percentiles = np.percentile(y_sample_for_quantiles, quantiles)
				if np.all(y_percentiles == y_percentiles[0]):
					y_discrete = np.zeros_like(y_sampled, dtype=np.int64)
				else:
					y_discrete = np.digitize(y_sampled, y_percentiles[1:-1], right=False)
					y_discrete = np.clip(y_discrete, 0, num_buckets - 1)
			else:
				y_discrete = y_sampled
			
			counts = histogram_from_pairs(x_discrete, y_discrete)
		else:
			# Integer or other types - use as-is
			counts = histogram_from_pairs(x_sampled, y_sampled)
	else:
		# Non-numpy arrays - use as-is
		counts = histogram_from_pairs(x_sampled, y_sampled)
	
	return openzl_model_bits(counts)


def generic_bytes_compress(data: bytes, min_size: int = 30) -> bytes:
	"""
	Compress bytes using OpenZL.
	
	Args:
		data: Data to compress
		min_size: Minimum size threshold - skip compression for data smaller than this
		
	Returns:
		Compressed data if beneficial, otherwise original data
	"""
	if zl is None:
		return data
	# Skip compression for very small data to avoid OpenZL decompression issues
	# OpenZL has issues decompressing data < 30 bytes reliably
	# Lowered threshold to 30 bytes for more aggressive compression
	if len(data) < min_size:
		# For very small data, compression overhead isn't worth it and may cause decompression issues
		return data
	c = _build_generic_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	compressed = cctx.compress([zl.Input(zl.Type.Serial, data)])
	# Only use compressed version if it's actually smaller (account for decompression overhead)
	# For very small compressed data (< 30 bytes), OpenZL may fail to decompress, so skip it
	if len(compressed) < len(data) and len(compressed) >= 30:
		return compressed
	# If compression didn't help or result is too small, return original
	return data


def generic_bytes_decompress(data: bytes, fallback_to_uncompressed: bool = False) -> bytes:
	"""
	Decompress data using OpenZL.
	
	Args:
		data: Data to decompress (may be compressed or uncompressed)
		fallback_to_uncompressed: If True and decompression fails, return data as-is
		
	Returns:
		Decompressed data, or original data if fallback_to_uncompressed=True and decompression fails
	"""
	if zl is None:
		return data
	if not data:
		return data  # Empty data is valid uncompressed
	
	# Very small data (< 50 bytes) is likely uncompressed to avoid OpenZL issues
	# But we should still try to decompress in case it was compressed
	try:
		dctx = zl.DCtx()
		outs = dctx.decompress(data)
		if len(outs) != 1 or outs[0].type != zl.Type.Serial:
			raise RuntimeError("Unexpected OpenZL output")
		return outs[0].content.as_bytes()
	except RuntimeError as e:
		error_msg = str(e)
		# Check for specific OpenZL errors that indicate the data might be uncompressed
		if "Internal buffer too small" in error_msg or "error code: 71" in error_msg:
			if fallback_to_uncompressed:
				# Data might be uncompressed - return as-is
				return data
			# Re-raise with context
			if len(data) < 50:
				raise RuntimeError(
					f"OpenZL decompression failed: compressed data is too small ({len(data)} bytes). "
					f"This is a known issue with OpenZL and very small compressed buffers. "
					f"Original error: {error_msg}"
				) from e
			else:
				raise RuntimeError(
					f"OpenZL decompression failed: data may be corrupted or in wrong format. "
					f"Data length: {len(data)} bytes. Original error: {error_msg}"
				) from e
		# For other errors, re-raise unless fallback is enabled
		if fallback_to_uncompressed:
			return data
		raise
	except Exception as e:
		# For any other exception, if fallback is enabled, return data as-is
		if fallback_to_uncompressed:
			return data
		raise


def compress_numeric_array(arr: np.ndarray) -> bytes:
	"""Compress a 1-D int64 numpy array using OpenZL numeric input; fallback to raw bytes."""
	if arr.dtype != np.int64:
		arr = arr.astype(np.int64, copy=False)
	if zl is None:
		return arr.tobytes(order="C")
	c = _build_generic_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	return cctx.compress([zl.Input(zl.Type.Numeric, arr)])


def compress_numeric_array_ace(arr: np.ndarray) -> bytes:
	"""Compress with ACE-friendly graph; fallback to generic numeric compression."""
	if arr.dtype != np.int64:
		arr = arr.astype(np.int64, copy=False)
	if zl is None:
		return arr.tobytes(order="C")
	c = _build_ace_numeric_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	try:
		return cctx.compress([zl.Input(zl.Type.Numeric, arr)])
	except Exception:
		return compress_numeric_array(arr)


def decompress_numeric_array(data: bytes) -> np.ndarray:
	"""Decompress a numeric frame produced by compress_numeric_array/ace; fallback interprets raw int64 bytes."""
	if zl is None:
		return np.frombuffer(data, dtype=np.int64)
	d = zl.DCtx()
	outs = d.decompress(data)
	if len(outs) != 1 or outs[0].type != zl.Type.Numeric:
		raise RuntimeError("Unexpected OpenZL output for numeric array")
	return outs[0].content.as_nparray().astype(np.int64, copy=False)


def get_generic_numeric_compressor():
	if zl is None:
		return None
	return _build_generic_compressor()


def get_ace_numeric_compressor():
	if zl is None:
		return None
	return _build_ace_numeric_compressor()


def compress_numeric_array_with(compressor, arr: np.ndarray) -> bytes:
	if arr.dtype != np.int64:
		arr = arr.astype(np.int64, copy=False)
	if zl is None or compressor is None:
		return arr.tobytes(order="C")
	cctx = zl.CCtx(); cctx.ref_compressor(compressor); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	return cctx.compress([zl.Input(zl.Type.Numeric, arr)])


_thread_local = threading.local()


def _get_thread_local_pair():
	if getattr(_thread_local, "pair", None) is None:
		if zl is None:
			_thread_local.pair = (None, None, None)
		else:
			_thread_local.pair = (_build_ace_numeric_compressor(), _build_generic_compressor(), _build_delta_numeric_compressor())
	return _thread_local.pair


def compress_numeric_array_fast(arr: np.ndarray, use_ace: bool, prefer_delta: bool = False) -> bytes:
	if arr.dtype != np.int64:
		arr = arr.astype(np.int64, copy=False)
	if zl is None:
		return arr.tobytes(order="C")
	ace_c, gen_c, del_c = _get_thread_local_pair()
	comp = del_c if prefer_delta else (ace_c if use_ace else gen_c)
	cctx = zl.CCtx(); cctx.ref_compressor(comp); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	return cctx.compress([zl.Input(zl.Type.Numeric, arr)])


def is_mostly_numeric(arr: np.ndarray, threshold: float = 0.95) -> bool:
	# Heuristic: treat as numeric if values are small non-negative ints and few unique > threshold
	if arr.size == 0:
		return True
	vals = arr
	if not np.issubdtype(vals.dtype, np.integer):
		return False
	return True
