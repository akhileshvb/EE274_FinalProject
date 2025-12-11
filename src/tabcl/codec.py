from typing import Any, Dict, Iterable, Tuple, Optional
import json
import numpy as np
import threading
from collections import Counter

try:
	import openzl.ext as zl  
except Exception:
	zl = None  


def histogram_from_pairs(x: Iterable[Any], y: Iterable[Any]) -> Dict[Tuple[Any, Any], int]:
	"""Count (x, y) pairs. Uses bincount for integer arrays when possible."""
	if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
		if (x.dtype.kind in 'iu' and y.dtype.kind in 'iu' and 
		    len(x) > 1000 and x.max() < 1000000 and y.max() < 1000000 and
		    x.min() >= 0 and y.min() >= 0):
			x_max = int(x.max()) + 1
			y_max = int(y.max()) + 1
			flat_idx = x.astype(np.int64) * y_max + y.astype(np.int64)
			counts_flat = np.bincount(flat_idx, minlength=x_max * y_max)
			result = {}
			for i in range(x_max):
				for j in range(y_max):
					idx = i * y_max + j
					if counts_flat[idx] > 0:
						result[(i, j)] = int(counts_flat[idx])
			return result
		return dict(Counter(zip(x, y)))
	else:
		return dict(Counter(zip(x, y)))


def _gamma_bits(v: int) -> int:
	if v <= 0:
		return 1
	l = int(np.floor(np.log2(v))) + 1
	return 2 * l - 1


def _golomb_bits(v: int, m: int) -> int:
	"""Golomb code length in bits."""
	if v <= 0:
		return 1
	if m <= 1:
		return _gamma_bits(v)
	q = v // m
	b = int(np.ceil(np.log2(m)))
	rem = v % m
	if m != (1 << b):
		bound = (1 << b) - m
		binary_bits = b - 1 if rem < bound else b
	else:
		binary_bits = b
	return (q + 1) + binary_bits


def _optimal_golomb_m(counts: Dict[Tuple[Any, Any], int]) -> int:
	"""Pick Golomb parameter m based on mean count."""
	if not counts:
		return 256
	all_counts = list(counts.values())
	if not all_counts:
		return 256
	mean_count = np.mean(all_counts)
	if mean_count < 1:
		return 256
	m_opt = max(1, int(0.693 * mean_count))
	m_opt = min(max(m_opt, 8), 8192)
	return m_opt


def proxy_model_bits(counts: Dict[Tuple[Any, Any], int]) -> float:
	"""Estimate MDL cost using gamma/Golomb codes."""
	m_hist = len(counts)
	if m_hist == 0:
		return 0.0
	bits = _gamma_bits(m_hist)
	if m_hist > 0:
		all_counts = list(counts.values())
		mean_count = np.mean(all_counts) if all_counts else 1
		if mean_count > 1 and mean_count < 100:
			golomb_m = _optimal_golomb_m(counts)
			for (_, _), c in counts.items():
				bits += _golomb_bits(c, golomb_m)
		else:
			for (_, _), c in counts.items():
				bits += _gamma_bits(c + 1)
	return float(bits)


def _build_generic_compressor() -> "zl.Compressor":
	c = zl.Compressor()
	g = zl.graphs.Compress()(c)
	c.select_starting_graph(g)
	return c


def _build_ace_numeric_compressor() -> "zl.Compressor":
	"""Build ACE compressor for low-cardinality numeric data. Falls back to generic if it fails."""
	try:
		c = zl.Compressor()
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


_compressor_cache = threading.local()

def _get_cached_ace_compressor():
	"""Reuse ACE compressor across calls to avoid rebuilding."""
	if not hasattr(_compressor_cache, 'ace_compressor'):
		_compressor_cache.ace_compressor = _build_ace_numeric_compressor()
	return _compressor_cache.ace_compressor

def openzl_model_bits(counts: Dict[Tuple[Any, Any], int]) -> float:
	"""MDL cost using OpenZL compression. More accurate but slower than proxy."""
	if zl is None:
		return proxy_model_bits(counts)
	pairs = list(counts.items())
	m = len(pairs)
	if m == 0:
		return 0.0
	
	# Small histograms are cheaper with proxy encoding
	if m <= 20:
		return proxy_model_bits(counts)
	
	flat = np.empty(3 * m, dtype=np.int64)
	k = 0
	for (a, b), c in pairs:
		flat[k] = np.int64(zl.hash_name(str(a))) if hasattr(zl, "hash_name") else np.int64(abs(hash(str(a))) & ((1<<63)-1)); k += 1
		flat[k] = np.int64(zl.hash_name(str(b))) if hasattr(zl, "hash_name") else np.int64(abs(hash(str(b))) & ((1<<63)-1)); k += 1
		flat[k] = np.int64(c); k += 1
	c = _get_cached_ace_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	out = cctx.compress([zl.Input(zl.Type.Numeric, flat)])
	return float(len(out) * 8)


def mdl_cost_fn_fast(x: np.ndarray, y: np.ndarray, row_sample: Optional[int] = None, seed: int = 0) -> float:
	"""Fast MDL estimate using proxy encoding. Good enough for ranking edges."""
	if row_sample is not None and row_sample < len(x):
		rng = np.random.default_rng(seed)
		idx = rng.choice(len(x), size=row_sample, replace=False)
		x_sampled = x[idx]
		y_sampled = y[idx]
	else:
		x_sampled = x
		y_sampled = y
	
	if (isinstance(x_sampled, np.ndarray) and isinstance(y_sampled, np.ndarray) and
	    x_sampled.dtype.kind in 'iu' and y_sampled.dtype.kind in 'iu' and
	    len(x_sampled) > 1000 and x_sampled.max() < 1000000 and y_sampled.max() < 1000000 and
	    x_sampled.min() >= 0 and y_sampled.min() >= 0):
		x_max = int(x_sampled.max()) + 1
		y_max = int(y_sampled.max()) + 1
		flat_idx = x_sampled.astype(np.int64) * y_max + y_sampled.astype(np.int64)
		counts_flat = np.bincount(flat_idx, minlength=x_max * y_max)
		counts = {}
		for i in range(x_max):
			for j in range(y_max):
				idx = i * y_max + j
				if counts_flat[idx] > 0:
					counts[(i, j)] = int(counts_flat[idx])
		return proxy_model_bits(counts)
	
	counts = histogram_from_pairs(x_sampled, y_sampled)
	return proxy_model_bits(counts)


def mdl_cost_fn_openzl(x: np.ndarray, y: np.ndarray, row_sample: Optional[int] = None, seed: int = 0) -> float:
	"""MDL cost using OpenZL. More accurate but slower. Use for final compression."""
	rng = np.random.default_rng(seed)
	if row_sample is not None and row_sample < len(x):
		idx = rng.choice(len(x), size=row_sample, replace=False)
		x_sampled = x[idx]
		y_sampled = y[idx]
	else:
		x_sampled = x
		y_sampled = y
	
	# Bucket floats to avoid huge histograms
	if isinstance(x_sampled, np.ndarray) and isinstance(y_sampled, np.ndarray):
		if x_sampled.dtype.kind == 'f' or y_sampled.dtype.kind == 'f':
			num_buckets = 256
			
			if x_sampled.dtype.kind == 'f':
				if len(x_sampled) > 10000:
					sample_idx = rng.choice(len(x_sampled), size=min(10000, len(x_sampled)), replace=False)
					x_sample_for_quantiles = x_sampled[sample_idx]
				else:
					x_sample_for_quantiles = x_sampled
				quantiles = np.linspace(0, 100, num_buckets + 1)
				x_percentiles = np.percentile(x_sample_for_quantiles, quantiles)
				if np.all(x_percentiles == x_percentiles[0]):
					x_discrete = np.zeros_like(x_sampled, dtype=np.int64)
				else:
					x_discrete = np.digitize(x_sampled, x_percentiles[1:-1], right=False)
					x_discrete = np.clip(x_discrete, 0, num_buckets - 1)
			else:
				x_discrete = x_sampled
			
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
			counts = histogram_from_pairs(x_sampled, y_sampled)
	else:
		counts = histogram_from_pairs(x_sampled, y_sampled)
	
	return openzl_model_bits(counts)


def generic_bytes_compress(data: bytes, min_size: int = 30) -> bytes:
	"""Compress bytes with OpenZL. Skips very small data to avoid decompression issues."""
	if zl is None:
		return data
	if len(data) < min_size:
		return data
	c = _build_generic_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	compressed = cctx.compress([zl.Input(zl.Type.Serial, data)])
	if len(compressed) < len(data) and len(compressed) >= 30:
		return compressed
	return data


def generic_bytes_decompress(data: bytes, fallback_to_uncompressed: bool = False) -> bytes:
	"""Decompress OpenZL data. Can fallback to returning original if decompression fails."""
	if zl is None:
		return data
	if not data:
		return data
	try:
		dctx = zl.DCtx()
		outs = dctx.decompress(data)
		if len(outs) != 1 or outs[0].type != zl.Type.Serial:
			raise RuntimeError("Unexpected OpenZL output")
		return outs[0].content.as_bytes()
	except RuntimeError as e:
		error_msg = str(e)
		if "Internal buffer too small" in error_msg or "error code: 71" in error_msg:
			if fallback_to_uncompressed:
				return data
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
		if fallback_to_uncompressed:
			return data
		raise
	except Exception as e:
		if fallback_to_uncompressed:
			return data
		raise


def compress_numeric_array(arr: np.ndarray) -> bytes:
	"""Compress int64 array with OpenZL. Falls back to raw bytes if OpenZL unavailable."""
	if arr.dtype != np.int64:
		arr = arr.astype(np.int64, copy=False)
	if zl is None:
		return arr.tobytes(order="C")
	c = _build_generic_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	return cctx.compress([zl.Input(zl.Type.Numeric, arr)])


def compress_numeric_array_ace(arr: np.ndarray) -> bytes:
	"""Compress with ACE. Falls back to generic if ACE fails."""
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
	"""Decompress numeric array. Falls back to raw int64 bytes if OpenZL unavailable."""
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
	"""Check if array looks numeric (integer dtype)."""
	if arr.size == 0:
		return True
	if not np.issubdtype(arr.dtype, np.integer):
		return False
	return True
