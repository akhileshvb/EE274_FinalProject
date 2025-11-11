from typing import Dict, List, Tuple, Any
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .codec import (
	decompress_numeric_array,
	compress_numeric_array_fast,
	is_mostly_numeric,
)


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


def encode_columns_with_parents(indices: List[np.ndarray], parents: List[int], dicts: List[Any], workers: int | None = None) -> List[bytes]:
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
		prefer_delta = dicts[j] is None
		use_ace = False if prefer_delta else _should_use_ace(arr)
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
		for k, pid in enumerate(uniq.tolist()):
			start = starts[k]
			end = starts[k + 1] if k + 1 < len(starts) else sorted_pids.size
			segment = sorted_vals[start:end]
			prefer_delta = dicts[j] is None
			use_ace = False if prefer_delta else _should_use_ace(segment)
			fb = compress_numeric_array_fast(segment, use_ace, prefer_delta=prefer_delta)
			parts.append(int(pid).to_bytes(8, "little", signed=True))
			parts.append(len(fb).to_bytes(8, "little", signed=False))
			parts.append(fb)
		frames[j] = b"".join(parts)

	with ThreadPoolExecutor(max_workers=workers) as ex:
		list(ex.map(encode_child, child_ids))

	return frames


def decode_columns_with_parents(frames: List[bytes], parents: List[int], n_rows: int) -> List[np.ndarray]:
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
			if frame[:4] != b"CND\x00":
				indices[j] = decompress_numeric_array(frame)
				undecoded.discard(j)
				progress = True
				continue
			ptr = 4
			nb = int.from_bytes(frame[ptr:ptr+4], "little"); ptr += 4
			bucket_data: Dict[int, List[int]] = {}
			for _ in range(nb):
				pid = int.from_bytes(frame[ptr:ptr+8], "little", signed=True); ptr += 8
				flen = int.from_bytes(frame[ptr:ptr+8], "little", signed=False); ptr += 8
				fb = frame[ptr:ptr+flen]; ptr += flen
				vals = decompress_numeric_array(fb).tolist()
				bucket_data[pid] = vals
			
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
