from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .codec import (
	decompress_numeric_array,
	compress_numeric_array_fast,
	is_mostly_numeric,
)


def _encode_varint(value: int, signed: bool = False) -> bytes:
	"""
	Encode an integer as a variable-length byte sequence.
	Format: 7 bits per byte, MSB indicates continuation (1=more bytes, 0=last byte)
	If signed=True, uses zigzag encoding to support negative values.
	"""
	if not signed and value < 0:
		raise ValueError("Varint encoding only supports non-negative integers when signed=False")
	
	# Apply zigzag encoding for signed values (both positive and negative)
	if signed:
		# Zigzag encoding: 
		# - positive values become even numbers (value * 2)
		# - negative values become odd numbers (abs(value) * 2 - 1)
		# Formula: (value << 1) ^ (value >> 63) for 64-bit integers
		value = (value << 1) ^ (value >> 63)
	
	result = []
	# Encode value (which is now non-negative after zigzag if signed)
	uvalue = value & 0xFFFFFFFFFFFFFFFF  # Ensure unsigned (treat as unsigned for encoding)
	while uvalue >= 0x80:
		result.append((uvalue & 0x7F) | 0x80)
		uvalue >>= 7
	result.append(uvalue & 0x7F)
	return bytes(result)


def _decode_varint(data: bytes, offset: int, signed: bool = False) -> Tuple[int, int]:
	"""
	Decode a variable-length integer from bytes.
	Returns (value, bytes_consumed)
	If signed=True, decodes zigzag-encoded values.
	"""
	result = 0
	shift = 0
	bytes_consumed = 0
	pos = offset
	
	while pos < len(data):
		byte = data[pos]
		result |= (byte & 0x7F) << shift
		bytes_consumed += 1
		pos += 1
		if (byte & 0x80) == 0:
			break
		shift += 7
		if shift >= 64:
			raise ValueError("Varint too long")
	
	# Zigzag decoding for signed values
	if signed:
		if result & 1:
			# Odd number means negative
			result = ~(result >> 1)
		else:
			# Even number means positive
			result = result >> 1
	
	return (result, bytes_consumed)


def orient_forest(n_cols: int, edges: List[Tuple[int, int, float]]) -> List[int]:
	"""
	Orient an undirected forest into parents array.
	Returns list parents[j] = parent index or -1 if root.
	"""
	adj: Dict[int, List[int]] = {i: [] for i in range(n_cols)}
	for u, v, _ in edges:
		adj[u].append(v)
		adj[v].append(u)

	parents = [-1] * n_cols
	visited = [False] * n_cols

	for start in range(n_cols):
		if visited[start]:
			continue
		queue: deque[int] = deque([start])
		visited[start] = True
		parents[start] = -1
		while queue:
			node = queue.popleft()
			for nei in adj[node]:
				if not visited[nei]:
					visited[nei] = True
					parents[nei] = node
					queue.append(nei)
	return parents


def _should_use_ace(arr: np.ndarray, threshold: int = 2048) -> bool:
	# Use ACE when alphabet is reasonably small
	# ACE is better for low-cardinality categorical data
	if arr.size == 0:
		return True
	unique_count = int(np.unique(arr).size)
	# Use ACE if cardinality is low (good for categorical)
	# For very sparse data (many zeros), delta might be better, but ACE handles it well too
	return unique_count <= threshold


def _choose_best_compression(arr: np.ndarray, dict_info: Any, prefer_delta: bool = False) -> Tuple[bool, bool]:
	"""
	Choose the best compression method for a segment.
	Returns (use_ace, prefer_delta)
	"""
	if arr.size == 0:
		return (False, False)
	
	# For numeric columns (dict_info is None or tuple), prefer delta if values are sequential
	if prefer_delta or dict_info is None or isinstance(dict_info, tuple):
		# Check if delta encoding would be beneficial
		# Delta is good for sequential/smooth data
		if arr.size > 10:
			# Check if values are mostly sequential or have small deltas
			diffs = np.diff(np.sort(arr))
			if len(diffs) > 0:
				mean_diff = np.mean(diffs[diffs > 0]) if np.any(diffs > 0) else 0
				# If differences are small and consistent, delta is better
				if mean_diff > 0 and mean_diff < 1000 and np.std(diffs[diffs > 0]) < mean_diff * 2:
					return (False, True)
		return (False, prefer_delta)
	
	# For categorical data, use ACE if cardinality is low
	unique_count = int(np.unique(arr).size)
	cardinality_ratio = unique_count / arr.size if arr.size > 0 else 1.0
	
	# ACE is better for low-cardinality data
	# But if cardinality is very high (>50% unique), delta might be better even for categorical
	if cardinality_ratio > 0.5:
		# High cardinality - try delta
		return (False, True)
	elif unique_count <= 2048:
		# Low cardinality - use ACE
		return (True, False)
	else:
		# Medium cardinality - prefer ACE but could use delta
		return (True, False)


def _try_all_compression_methods(arr: np.ndarray, dict_info: Any) -> bytes:
	"""
	Try all available compression methods and return the smallest result.
	This is more expensive but gives better compression.
	"""
	if arr.size == 0:
		return b""
	
	from .codec import compress_numeric_array_fast
	
	# Check for constant or near-constant arrays (very compressible)
	unique_vals = np.unique(arr)
	if len(unique_vals) == 1:
		# Constant array - use generic compression (very efficient for constants)
		try:
			return compress_numeric_array_fast(arr, use_ace=False, prefer_delta=False)
		except Exception:
			pass
	
	# Try different compression strategies - be more aggressive
	candidates = []
	unique_count = int(unique_vals.size)
	
	# Check for sparse arrays (many zeros) - these compress very well
	zero_count = np.sum(arr == 0)
	zero_ratio = zero_count / arr.size if arr.size > 0 else 0
	if zero_ratio > 0.5:
		# More than 50% zeros - sparse array, generic compression handles this well
		# But also try ACE if applicable (ACE is good for sparse categorical)
		if dict_info is not None and not isinstance(dict_info, tuple):
			try:
				ace_result = compress_numeric_array_fast(arr, use_ace=True, prefer_delta=False)
				candidates.append(("ace_sparse", ace_result))
			except Exception:
				pass
		# Generic compression is usually best for sparse numeric
		try:
			generic_result = compress_numeric_array_fast(arr, use_ace=False, prefer_delta=False)
			candidates.append(("generic_sparse", generic_result))
		except Exception:
			pass
		# If we have sparse candidates, still try other methods but prioritize sparse ones
		# (Don't return early - let other methods compete)
	
	# Check if array has many repeated values (low entropy) - ACE or generic will work well
	# Use numpy for faster counting (only for reasonable sizes to avoid overhead)
	is_dominant = False
	if arr.size > 0 and arr.size < 100000:  # Only check for smaller arrays to avoid overhead
		# Reuse unique_vals from above, but get counts
		_, counts = np.unique(arr, return_counts=True)
		max_count = np.max(counts) if len(counts) > 0 else 0
		dominant_ratio = max_count / arr.size
		# If one value dominates (>40% of values), it's highly compressible
		is_dominant = dominant_ratio > 0.4
	
	# Strategy 1: ACE (if applicable) - usually best for low-cardinality categorical
	# Try ACE more aggressively - it often works well even for higher cardinality
	# Also good for arrays with dominant values (high repetition)
	if dict_info is not None and not isinstance(dict_info, tuple):
		# Try ACE for reasonable cardinality - expand range even more
		# Also try if there's a dominant value (even with higher cardinality)
		if unique_count <= 32768 or (is_dominant and unique_count <= 65536):  # Even more aggressive for dominant values
			try:
				ace_result = compress_numeric_array_fast(arr, use_ace=True, prefer_delta=False)
				candidates.append(("ace", ace_result))
			except Exception:
				pass
	
	# Strategy 2: Delta encoding (good for sequential/numeric data and runs)
	# Always try delta for numeric columns (including timestamps which are now integers)
	if dict_info is None or isinstance(dict_info, tuple):
		try:
			delta_result = compress_numeric_array_fast(arr, use_ace=False, prefer_delta=True)
			candidates.append(("delta", delta_result))
		except Exception:
			pass
	else:
		# For categorical, try delta more often - it can help even for medium cardinality
		# Delta encoding compresses runs well (consecutive identical values)
		# Be more aggressive: try delta for smaller segments and lower cardinality
		# Also check if there are many repeated values (runs) - delta is great for that
		has_runs = False
		if arr.size > 1:
			# Check for runs of identical consecutive values
			diffs = np.diff(arr)
			zero_diffs = np.sum(diffs == 0)
			# If >20% of values are part of runs, delta will help
			if zero_diffs > arr.size * 0.2:
				has_runs = True
		
		if unique_count > 20 or arr.size > 30 or has_runs:  # Even more aggressive
			try:
				delta_result = compress_numeric_array_fast(arr, use_ace=False, prefer_delta=True)
				candidates.append(("delta", delta_result))
			except Exception:
				pass
	
	# Strategy 3: Generic compression (fallback) - always try this
	try:
		generic_result = compress_numeric_array_fast(arr, use_ace=False, prefer_delta=False)
		candidates.append(("generic", generic_result))
	except Exception:
		pass
	
	# Return the smallest compression result
	if not candidates:
		# Fallback to raw bytes if all compression fails
		return arr.tobytes(order="C")
	
	# Sort by size and return the smallest
	candidates.sort(key=lambda x: len(x[1]))
	return candidates[0][1]


def encode_columns_with_parents(indices: List[np.ndarray], parents: List[int], dicts: List[Any], workers: int | None = None, use_mlp: bool = False) -> Tuple[List[bytes], List[Optional[bytes]]]:
	"""
	Encode each column:
	- roots: single numeric frame (ACE if low-cardinality)
	- child j with parent p: bucket child indices by parent token and compress each bucket separately
	  Format:
	    [b"CND\x00"][nbuckets:u32][for each bucket: parent_id:i64, frame_len:u64, frame...]
	"""
	frames: List[bytes] = [b""] * len(indices)

	def encode_root(j: int) -> None:
		arr = indices[j]
		# For root columns, try all compression methods and pick the best
		# This is worth it since roots are larger and compression quality matters more
		# Lower threshold to get better compression on more roots
		if arr.size > 10:  # Lowered threshold even more: try all methods for roots > 10 elements
			frames[j] = _try_all_compression_methods(arr, dicts[j])
		else:
			# For smaller arrays, still try multiple methods to pick the best
			if arr.size > 3:  # Lowered threshold: try multiple methods for roots > 3
				# Try ACE (if applicable) and delta/generic, pick best
				candidates = []
				prefer_delta_base = dicts[j] is None or isinstance(dicts[j], tuple)
				
				# Try ACE if applicable (use higher threshold to match _try_all_compression_methods)
				if dicts[j] is not None and not isinstance(dicts[j], tuple):
					unique_count = int(np.unique(arr).size)
					if unique_count <= 32768:  # Match threshold from _try_all_compression_methods
						try:
							candidates.append(compress_numeric_array_fast(arr, use_ace=True, prefer_delta=False))
						except Exception:
							pass
				
				# Try delta if numeric or medium/high cardinality
				if prefer_delta_base:
					try:
						candidates.append(compress_numeric_array_fast(arr, use_ace=False, prefer_delta=True))
					except Exception:
						pass
				elif dicts[j] is not None and not isinstance(dicts[j], tuple):
					unique_count = int(np.unique(arr).size)
					if unique_count > 20:  # Lower threshold further to try delta more
						try:
							candidates.append(compress_numeric_array_fast(arr, use_ace=False, prefer_delta=True))
						except Exception:
							pass
				
				# Always try generic
				try:
					candidates.append(compress_numeric_array_fast(arr, use_ace=False, prefer_delta=False))
				except Exception:
					pass
				
				if candidates:
					candidates.sort(key=len)
					frames[j] = candidates[0]
				else:
					frames[j] = compress_numeric_array_fast(arr, use_ace=False, prefer_delta=False)
			else:
				# Very small arrays (<=10), use heuristic (faster)
				prefer_delta_base = dicts[j] is None or isinstance(dicts[j], tuple)
				use_ace, prefer_delta = _choose_best_compression(arr, dicts[j], prefer_delta_base)
				frames[j] = compress_numeric_array_fast(arr, use_ace, prefer_delta=prefer_delta)

	# Encode roots in parallel
	root_ids = [j for j, p in enumerate(parents) if p == -1]
	child_ids = [j for j, p in enumerate(parents) if p != -1]
	if workers is None or workers < 1:
		workers = 1
	with ThreadPoolExecutor(max_workers=workers) as ex:
		list(ex.map(encode_root, root_ids))

	# Encode children (each independently given parent indices are fixed)
	def encode_child(j: int) -> None:
		arr = indices[j]
		p = parents[j]
		parent_ids = indices[p]
		if parent_ids.shape[0] != arr.shape[0]:
			raise ValueError("Parent and child length mismatch")
		# Vectorized grouping by parent id
		pids = parent_ids.astype(np.int64, copy=False)
		order = np.argsort(pids)
		sorted_pids = pids[order]
		sorted_vals = arr.astype(np.int64, copy=False)[order]
		uniq, starts = np.unique(sorted_pids, return_index=True)
		parts: List[bytes] = []
		parts.append(b"CND\x00")
		parts.append(int(len(uniq)).to_bytes(4, "little", signed=False))
		
		# Analyze segment sizes to decide if conditional encoding is worth it
		# If too many segments are very small, the overhead might not be worth it
		segment_sizes = [(starts[k + 1] if k + 1 < len(starts) else sorted_pids.size) - starts[k] 
		                for k in range(len(uniq))]
		avg_segment_size = sum(segment_sizes) / len(segment_sizes) if segment_sizes else 0
		small_segments = sum(1 for s in segment_sizes if s < 3)
		
		# If average segment size is very small (< 5) and we have many segments,
		# conditional encoding overhead might not be worth it
		# However, for now we'll still use it but optimize small segments
		
		for k, pid in enumerate(uniq.tolist()):
			start = starts[k]
			end = starts[k + 1] if k + 1 < len(starts) else sorted_pids.size
			segment = sorted_vals[start:end]
			
			# Choose compression method based on segment size
			# For better compression, try all methods for larger segments
			# Be extremely aggressive - try all methods for even smaller segments
			if segment.size > 3:  # Try all methods for segments > 3 (very aggressive)
				# For larger segments, try all compression methods (worth the cost for better compression)
				fb = _try_all_compression_methods(segment, dicts[j])
			elif segment.size > 1:
				# For medium segments (4-10), try ACE and delta/generic, pick best
				candidates = []
				prefer_delta_base = dicts[j] is None or isinstance(dicts[j], tuple)
				
				# Try ACE if applicable
				if dicts[j] is not None and not isinstance(dicts[j], tuple):
					unique_count = int(np.unique(segment).size)
					if unique_count <= 16384:  # Increased threshold to match _try_all_compression_methods
						try:
							ace_result = compress_numeric_array_fast(segment, use_ace=True, prefer_delta=False)
							candidates.append(ace_result)
						except Exception:
							pass
				
				# Try delta if numeric or medium/high cardinality
				# Be more aggressive: try delta for lower cardinality and smaller segments
				if prefer_delta_base or (dicts[j] is not None and not isinstance(dicts[j], tuple) and int(np.unique(segment).size) > 20):  # Lower threshold further
					try:
						delta_result = compress_numeric_array_fast(segment, use_ace=False, prefer_delta=True)
						candidates.append(delta_result)
					except Exception:
						pass
				
				# Always try generic
				try:
					generic_result = compress_numeric_array_fast(segment, use_ace=False, prefer_delta=False)
					candidates.append(generic_result)
				except Exception:
					pass
				
				if candidates:
					candidates.sort(key=len)
					fb = candidates[0]
				else:
					# Fallback
					use_ace, prefer_delta = _choose_best_compression(segment, dicts[j], prefer_delta_base)
					fb = compress_numeric_array_fast(segment, use_ace, prefer_delta=prefer_delta)
			else:
				# For very small segments (2-3), still try multiple methods for best compression
				candidates = []
				prefer_delta_base = dicts[j] is None or isinstance(dicts[j], tuple)
				
				# Try ACE if applicable
				if dicts[j] is not None and not isinstance(dicts[j], tuple):
					try:
						ace_result = compress_numeric_array_fast(segment, use_ace=True, prefer_delta=False)
						candidates.append(ace_result)
					except Exception:
						pass
				
				# Try delta if numeric
				if prefer_delta_base:
					try:
						delta_result = compress_numeric_array_fast(segment, use_ace=False, prefer_delta=True)
						candidates.append(delta_result)
					except Exception:
						pass
				
				# Always try generic
				try:
					generic_result = compress_numeric_array_fast(segment, use_ace=False, prefer_delta=False)
					candidates.append(generic_result)
				except Exception:
					pass
				
				if candidates:
					candidates.sort(key=len)
					fb = candidates[0]
				else:
					# Fallback
					use_ace, prefer_delta = _choose_best_compression(segment, dicts[j], prefer_delta_base)
					fb = compress_numeric_array_fast(segment, use_ace, prefer_delta=prefer_delta)
			
			# Use variable-length encoding for parent ID and length to save space
			# For small parent IDs and lengths, use fewer bytes
			pid_bytes = _encode_varint(pid, signed=False)  # Parent IDs are non-negative
			parts.append(pid_bytes)
			# Use varint for length (unsigned since length is always positive)
			# Note: We use a flag byte to indicate raw storage instead of negative length
			length_bytes = _encode_varint(len(fb), signed=False)
			parts.append(length_bytes)
			parts.append(fb)
		
		frames[j] = b"".join(parts)

	# MLP models list (one per column, None if histogram used)
	mlp_models: List[Optional[bytes]] = [None] * len(indices)
	
	if use_mlp:
		# Try MLP for child columns
		from .mlp_conditional import compare_mlp_vs_histogram_mdl
		from .codec import mdl_cost_fn_openzl
		
		def try_mlp_for_child(j: int) -> None:
			"""Try MLP encoding for a child column and update frame if beneficial."""
			arr = indices[j]
			p = parents[j]
			parent_ids = indices[p]
			
			# Skip if too many unique values (MLP not suitable)
			unique_children = len(np.unique(arr))
			if unique_children > 10000 or unique_children < 2:
				return
			
			# Compute actual histogram encoding cost by doing a dry-run
			# This gives us the real compressed size
			pids = parent_ids.astype(np.int64, copy=False)
			order = np.argsort(pids)
			sorted_pids = pids[order]
			sorted_vals = arr.astype(np.int64, copy=False)[order]
			uniq, starts = np.unique(sorted_pids, return_index=True)
			
			# Compute histogram model cost (metadata overhead)
			sample_size = min(10000, len(arr))
			if sample_size < len(arr):
				idx = np.random.choice(len(arr), size=sample_size, replace=False)
				x_sample = parent_ids[idx]
				y_sample = arr[idx]
			else:
				x_sample = parent_ids
				y_sample = arr
			hist_model_bits = mdl_cost_fn_openzl(x_sample, y_sample, row_sample=None, seed=0)
			
			# Compute actual histogram data cost by encoding segments
			hist_data_bits = 0.0
			# Frame header: CND marker (4 bytes) + nbuckets (4 bytes)
			hist_data_bits += 8 * 8.0
			for k, pid in enumerate(uniq.tolist()):
				start = starts[k]
				end = starts[k + 1] if k + 1 < len(starts) else sorted_pids.size
				segment = sorted_vals[start:end]
				
				# Try all compression methods and get the best
				segment_compressed = _try_all_compression_methods(segment, dicts[j])
				
				# Add overhead: varint for parent ID + varint for length
				pid_varint_len = len(_encode_varint(pid, signed=False))
				length_varint_len = len(_encode_varint(len(segment_compressed), signed=False))
				hist_data_bits += (pid_varint_len + length_varint_len + len(segment_compressed)) * 8.0
			
			hist_total_bits = hist_model_bits + hist_data_bits
			
			# Compare MLP vs histogram
			# Use a small negative margin to favor MLP when it's close (helps with learning)
			# But require it to be at least competitive
			use_mlp_flag, model, mlp_total = compare_mlp_vs_histogram_mdl(
				child_indices=arr,
				parent_indices=parent_ids,
				histogram_model_bits=hist_model_bits,
				histogram_data_bits=hist_data_bits,
				margin=-50.0,  # Small negative margin: prefer MLP if within 50 bits
			)
			
			if use_mlp_flag and model is not None:
				# Encode with MLP
				from .mlp_conditional import encode_column_with_mlp
				from .tiny_mlp import serialize_mlp_params
				
				encoded_data, mlp_model, _ = encode_column_with_mlp(
					child_indices=arr,
					parent_indices=parent_ids,
					dicts=dicts[j],
				)
				
				if encoded_data is not None and mlp_model is not None:
					# Serialize MLP model
					mlp_params = serialize_mlp_params(mlp_model)
					mlp_models[j] = mlp_params
					
					# Create frame with MLP marker
					mlp_frame = b"MLP\x00" + struct.pack("<I", len(encoded_data)) + encoded_data
					frames[j] = mlp_frame
		
		# Try MLP for child columns in parallel
		with ThreadPoolExecutor(max_workers=workers) as ex:
			list(ex.map(try_mlp_for_child, child_ids))
		
		# Encode remaining children with histogram method
		remaining_children = [j for j in child_ids if mlp_models[j] is None]
		with ThreadPoolExecutor(max_workers=workers) as ex:
			list(ex.map(encode_child, remaining_children))
	else:
		# Standard histogram encoding
		with ThreadPoolExecutor(max_workers=workers) as ex:
			list(ex.map(encode_child, child_ids))

	return frames, mlp_models


def decode_columns_with_parents(frames: List[bytes], parents: List[int], n_rows: int, mlp_models: Optional[List[Optional[bytes]]] = None) -> List[np.ndarray]:
	"""
	Inverse of encode_columns_with_parents. Reconstruct each column's indices array.
	Decodes roots first, then iteratively decodes children whose parent is already available.
	"""
	indices: List[np.ndarray] = [None] * len(frames)  # type: ignore

	undecoded: set[int] = set()
	for j, p in enumerate(parents):
		if p == -1:
			indices[j] = decompress_numeric_array(frames[j])
			if indices[j].shape[0] != n_rows:
				raise RuntimeError("Root length mismatch")
		else:
			undecoded.add(j)

	progress = True
	while undecoded and progress:
		progress = False
		ready: List[int] = []
		for j in list(undecoded):
			p = parents[j]
			if p >= 0 and indices[p] is not None:
				ready.append(j)
		for j in ready:
			p = parents[j]
			parent_ids = indices[p]
			frame = frames[j]
			
			# Check for MLP-encoded frame
			if frame[:4] == b"MLP\x00":
				if mlp_models is None or mlp_models[j] is None:
					raise RuntimeError(f"MLP model missing for column {j}")
				
				# Deserialize MLP model
				from .tiny_mlp import deserialize_mlp_params
				_, model = deserialize_mlp_params(mlp_models[j])
				
				# Decode MLP frame
				ptr = 4
				data_len = struct.unpack("<I", frame[ptr:ptr+4])[0]
				ptr += 4
				encoded_data = frame[ptr:ptr+data_len]
				
				from .mlp_conditional import decode_column_with_mlp
				indices[j] = decode_column_with_mlp(encoded_data, model, parent_ids, n_rows)
				undecoded.discard(j)
				progress = True
				continue
			
			if frame[:4] != b"CND\x00":
				indices[j] = decompress_numeric_array(frame)
				undecoded.discard(j)
				progress = True
				continue
			ptr = 4
			nb = int.from_bytes(frame[ptr:ptr+4], "little"); ptr += 4
			bucket_data: Dict[int, List[int]] = {}
			
			# Always try new format (varint) first - this is the current encoding format
			# Only fall back to old format if varint decoding fails AND we have enough bytes for old format
			save_ptr = ptr
			try:
				# Try to decode all buckets as varints (new format)
				for _ in range(nb):
					pid, bytes_consumed = _decode_varint(frame, ptr, signed=False)
					ptr += bytes_consumed
					flen, bytes_consumed = _decode_varint(frame, ptr, signed=False)
					ptr += bytes_consumed
					if ptr + flen > len(frame):
						raise IndexError("Not enough data for segment")
					fb = frame[ptr:ptr+flen]
					ptr += flen
					vals = decompress_numeric_array(fb).tolist()
					bucket_data[pid] = vals
			except (ValueError, IndexError) as e:
				# Varint decoding failed - try old format only if we have enough bytes
				# Old format: 8 bytes per pid + 8 bytes per length
				expected_old_format_size = nb * 16
				remaining_bytes = len(frame) - save_ptr
				
				if remaining_bytes >= expected_old_format_size:
					# Might be old format - try decoding as old format
					ptr = save_ptr
					bucket_data = {}
					try:
						for _ in range(nb):
							pid = int.from_bytes(frame[ptr:ptr+8], "little", signed=True)
							ptr += 8
							flen = int.from_bytes(frame[ptr:ptr+8], "little", signed=False)
							ptr += 8
							if ptr + flen > len(frame):
								raise IndexError("Not enough data for segment in old format")
							fb = frame[ptr:ptr+flen]
							ptr += flen
							vals = decompress_numeric_array(fb).tolist()
							bucket_data[pid] = vals
					except (ValueError, IndexError):
						# Both formats failed - re-raise original error
						raise e
				else:
					# Not enough bytes for old format - re-raise original error
					raise e
			
			# Reconstruct in sorted order, then unsort
			# Values in buckets are in sorted-by-parent-ID order
			pids_sorted = np.sort(parent_ids.astype(np.int64, copy=False))
			order_sorted = np.argsort(parent_ids.astype(np.int64, copy=False))
			out_sorted = np.empty(n_rows, dtype=np.int64)
			cursors: Dict[int, int] = {pid: 0 for pid in bucket_data.keys()}
			for i in range(n_rows):
				pid = int(pids_sorted[i])
				vals = bucket_data.get(pid, [])
				k = cursors.get(pid, 0)
				if k >= len(vals):
					out_sorted[i] = 0
				else:
					out_sorted[i] = vals[k]
					cursors[pid] = k + 1
			
			# Unsort to restore original order
			out = np.empty(n_rows, dtype=np.int64)
			out[order_sorted] = out_sorted
			indices[j] = out
			undecoded.discard(j)
			progress = True

	if undecoded:
		raise RuntimeError("Could not decode all columns; parent ordering issue")

	return indices
