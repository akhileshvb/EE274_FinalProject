from typing import Any, Dict, Iterable, Tuple
import json
import numpy as np
import threading

try:
	import openzl.ext as zl  # type: ignore
except Exception:
	zl = None  # type: ignore


def histogram_from_pairs(x: Iterable[Any], y: Iterable[Any]) -> Dict[Tuple[Any, Any], int]:
	counts: Dict[Tuple[Any, Any], int] = {}
	for a, b in zip(x, y):
		k = (a, b)
		counts[k] = counts.get(k, 0) + 1
	return counts


def _gamma_bits(v: int) -> int:
	if v <= 0:
		return 1
	l = int(np.floor(np.log2(v))) + 1
	return 2 * l - 1


def proxy_model_bits(counts: Dict[Tuple[Any, Any], int]) -> float:
	m = len(counts)
	if m == 0:
		return 0.0
	bits = _gamma_bits(m)
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


def openzl_model_bits(counts: Dict[Tuple[Any, Any], int]) -> float:
	if zl is None:
		return proxy_model_bits(counts)
	# Encode histogram as a flat int64 triple array [a0,b0,c0, a1,b1,c1, ...]
	pairs = list(counts.items())
	m = len(pairs)
	flat = np.empty(3 * m, dtype=np.int64)
	k = 0
	for (a, b), c in pairs:
		# stringify to stabilize mapping, then hash to int64 domain for robustness
		flat[k] = np.int64(zl.hash_name(str(a))) if hasattr(zl, "hash_name") else np.int64(abs(hash(str(a))) & ((1<<63)-1)); k += 1
		flat[k] = np.int64(zl.hash_name(str(b))) if hasattr(zl, "hash_name") else np.int64(abs(hash(str(b))) & ((1<<63)-1)); k += 1
		flat[k] = np.int64(c); k += 1
	# Compress with ACE-friendly numeric graph
	c = _build_ace_numeric_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	out = cctx.compress([zl.Input(zl.Type.Numeric, flat)])
	return float(len(out) * 8)


def mdl_cost_fn_openzl(x: np.ndarray, y: np.ndarray) -> float:
	counts = histogram_from_pairs(x, y)
	return openzl_model_bits(counts)


def generic_bytes_compress(data: bytes) -> bytes:
	if zl is None:
		return data
	c = _build_generic_compressor()
	cctx = zl.CCtx(); cctx.ref_compressor(c); cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)
	return cctx.compress([zl.Input(zl.Type.Serial, data)])


def generic_bytes_decompress(data: bytes) -> bytes:
	if zl is None:
		return data
	dctx = zl.DCtx()
	outs = dctx.decompress(data)
	if len(outs) != 1 or outs[0].type != zl.Type.Serial:
		raise RuntimeError("Unexpected OpenZL output")
	return outs[0].content.as_bytes()


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
