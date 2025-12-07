"""Fully autoregressive MLP-based compression: p(x_j | x_<j) for all previous columns."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import struct

from .tiny_mlp import (
	train_tiny_mlp,
	compute_mlp_model_bits,
	serialize_mlp_params,
	deserialize_mlp_params,
	TinyMLP,
	TORCH_AVAILABLE,
)

if TORCH_AVAILABLE:
	import torch

try:
	import openzl.ext as zl  # type: ignore
except Exception:
	zl = None


def encode_column_autoregressive(
	child_indices: np.ndarray,
	parent_columns: List[np.ndarray],  # All previous columns
	embedding_dim: int = 4,
	hidden_dims: List[int] = [16],
	num_epochs: int = 20,
	max_samples: Optional[int] = 50000,
) -> Tuple[Optional[bytes], Optional[TinyMLP], float]:
	"""
	Encode a column using fully autoregressive MLP: p(child | parent1, parent2, ..., parentN).
	
	Args:
		child_indices: Column to encode [n_rows]
		parent_columns: List of all previous columns [n_rows] each
		embedding_dim: MLP embedding dimension
		hidden_dims: MLP hidden layer dimensions
		num_epochs: Training epochs
		max_samples: Max samples for training
		
	Returns:
		(encoded_bytes, model, total_bits)
	"""
	if not TORCH_AVAILABLE or zl is None:
		return None, None, float('inf')
	
	if len(parent_columns) == 0:
		# No parents: encode independently
		from .codec import compress_numeric_array_fast
		encoded = compress_numeric_array_fast(child_indices.astype(np.int64), use_ace=True, prefer_delta=False)
		return encoded, None, len(encoded) * 8.0
	
	# Get unique values
	all_unique_children = np.unique(child_indices)
	all_unique_children = all_unique_children[all_unique_children != -1]
	
	# Get unique values for each parent column
	all_unique_parents_list = []
	autoregressive_vocab_sizes = []
	for parent_col in parent_columns:
		unique_parents = np.unique(parent_col)
		unique_parents = unique_parents[unique_parents != -1]
		all_unique_parents_list.append(unique_parents)
		autoregressive_vocab_sizes.append(len(unique_parents))
	
	# Train autoregressive MLP
	model = train_tiny_mlp(
		child_indices=child_indices,
		parent_indices=None,  # Not used in autoregressive mode
		embedding_dim=embedding_dim,
		hidden_dims=hidden_dims,
		num_epochs=num_epochs,
		max_samples=max_samples,
		all_unique_children=all_unique_children,
		all_unique_parents=None,  # Not used in autoregressive mode
		autoregressive_parents=parent_columns,
		autoregressive_vocab_sizes=autoregressive_vocab_sizes,
	)
	
	if model is None:
		return None, None, float('inf')
	
	# Compute model cost
	model_bits = compute_mlp_model_bits(model)
	
	# Get predictions for all rows
	device = next(model.parameters()).device
	
	# Map parent columns to model's vocabulary
	parent_tensors = []
	for i, parent_col in enumerate(parent_columns):
		# Get parent map for this column (stored as parent_map_0, parent_map_1, etc.)
		if hasattr(model, f'parent_map_{i}'):
			parent_map = getattr(model, f'parent_map_{i}')
		elif i == 0 and hasattr(model, 'parent_map'):
			parent_map = model.parent_map
		else:
			# Fallback: create identity map
			unique_vals = np.unique(parent_col[parent_col != -1])
			parent_map = {v: idx for idx, v in enumerate(unique_vals)}
		
		parent_mapped = np.array([
			parent_map.get(v, 0) for v in parent_col
		], dtype=np.int64)
		vocab_size = autoregressive_vocab_sizes[i] if i < len(autoregressive_vocab_sizes) else len(parent_map)
		parent_mapped = np.clip(parent_mapped, 0, vocab_size - 1)
		parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
		parent_tensors.append(parent_tensor)
	
	model.eval()
	with torch.no_grad():
		logits = model.forward(parent_tensors)
		probs = torch.softmax(logits, dim=1)
		probs_np = probs.cpu().numpy()
		# Ensure probabilities are valid
		probs_np = np.clip(probs_np, 1e-10, 1.0)
		probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
	
	# Map child indices to model's vocabulary
	child_mapped = np.array([
		model.child_map.get(v, -1) if v != -1 else -1 for v in child_indices
	], dtype=np.int64)
	
	# Check for unmapped values
	unmapped_mask = (child_mapped == -1) & (child_indices != -1)
	if np.any(unmapped_mask):
		unmapped_count = np.sum(unmapped_mask)
		unmapped_orig_vals = np.unique(child_indices[unmapped_mask])
		raise RuntimeError(
			f"ERROR: {unmapped_count} child values could not be mapped: {unmapped_orig_vals[:10]}"
		)
	
	# Handle sentinels
	if np.any(child_indices == -1):
		sentinel_count = np.sum(child_indices == -1)
		if sentinel_count > len(child_indices) * 0.1:
			return None, None, float('inf')
	
	# Ensure all values are in valid range
	child_mapped = np.clip(child_mapped, 0, model.num_classes - 1)
	
	# Encode ranks using probability-weighted encoding (same as mlp_conditional.py)
	prob_sorted_indices = np.argsort(probs_np, axis=1)[:, ::-1]
	
	# Vectorized rank computation
	n_rows = len(child_mapped)
	match_matrix = prob_sorted_indices == child_mapped[:, np.newaxis]
	ranks = np.argmax(match_matrix, axis=1).astype(np.int32)
	no_match_mask = ~np.any(match_matrix, axis=1)
	ranks[no_match_mask] = model.num_classes - 1
	
	# Compress ranks
	from .codec import compress_numeric_array_fast
	try:
		ranks_compressed = compress_numeric_array_fast(
			ranks.astype(np.int64),
			use_ace=True,
			prefer_delta=False
		)
	except Exception:
		try:
			ranks_compressed = compress_numeric_array_fast(
				ranks.astype(np.int64),
				use_ace=False,
				prefer_delta=True
			)
		except Exception:
			ranks_compressed = ranks.astype(np.int64).tobytes()
	
	combined = ranks_compressed
	
	# Total cost = model cost + compressed ranks cost
	data_bits = len(combined) * 8.0
	total_bits = model_bits + data_bits
	
	return combined, model, total_bits


def decode_column_autoregressive(
	encoded_data: bytes,
	model: TinyMLP,
	parent_columns: List[np.ndarray],
	n_rows: int,
) -> np.ndarray:
	"""
	Decode a column using fully autoregressive MLP.
	
	Args:
		encoded_data: Compressed ranks data
		model: Trained TinyMLP model
		parent_columns: List of all previous columns [n_rows] each
		n_rows: Number of rows
		
	Returns:
		Decoded column indices [n_rows]
	"""
	if zl is None:
		raise RuntimeError("OpenZL not available for decompression")
	
	# Reconstruct MLP predictions
	device = next(model.parameters()).device
	
	# Map parent columns
	parent_tensors = []
	autoregressive_vocab_sizes = []
	for i, parent_col in enumerate(parent_columns):
		# Get parent map for this column
		if hasattr(model, f'parent_map_{i}'):
			parent_map = getattr(model, f'parent_map_{i}')
		elif i == 0 and hasattr(model, 'parent_map'):
			parent_map = model.parent_map
		else:
			# Fallback: create identity map
			unique_vals = np.unique(parent_col[parent_col != -1])
			parent_map = {v: idx for idx, v in enumerate(unique_vals)}
		
		parent_mapped = np.array([
			parent_map.get(v, 0) for v in parent_col
		], dtype=np.int64)
		# Get vocab size from model
		if hasattr(model, 'embedding_vocab_sizes') and i < len(model.embedding_vocab_sizes):
			vocab_size = model.embedding_vocab_sizes[i]
		else:
			vocab_size = len(parent_map) if parent_map else 1
		parent_mapped = np.clip(parent_mapped, 0, vocab_size - 1)
		autoregressive_vocab_sizes.append(vocab_size)
		parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
		parent_tensors.append(parent_tensor)
	
	model.eval()
	with torch.no_grad():
		logits = model.forward(parent_tensors)
		probs = torch.softmax(logits, dim=1)
		probs_np = probs.cpu().numpy()
		probs_np = np.clip(probs_np, 1e-10, 1.0)
		probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)
	
	# Decode ranks
	from .codec import decompress_numeric_array
	try:
		ranks = decompress_numeric_array(encoded_data)
	except Exception:
		if zl is not None:
			try:
				dctx = zl.DCtx()
				outs = dctx.decompress(encoded_data)
				if len(outs) != 1 or outs[0].type != zl.Type.Numeric:
					raise RuntimeError("Unexpected OpenZL output")
				ranks = outs[0].data
			except Exception:
				ranks = np.frombuffer(encoded_data, dtype=np.int64)
		else:
			ranks = np.frombuffer(encoded_data, dtype=np.int64)
	
	# Ensure ranks match expected length
	if len(ranks) != n_rows:
		if len(ranks) < n_rows:
			ranks = np.pad(ranks, (0, n_rows - len(ranks)), mode='constant')
		else:
			ranks = ranks[:n_rows]
	
	# Reconstruct child values from ranks
	prob_sorted_indices = np.argsort(probs_np, axis=1)[:, ::-1]
	ranks_clamped = np.clip(ranks, 0, model.num_classes - 1).astype(np.int32)
	row_indices = np.arange(n_rows)
	child_mapped = prob_sorted_indices[row_indices, ranks_clamped]
	child_mapped = np.clip(child_mapped, 0, model.num_classes - 1)
	
	# Map back to original values
	if hasattr(model, 'child_map') and model.child_map:
		inverse_map = {mapped: orig for orig, mapped in model.child_map.items()}
		child_indices = np.array([
			inverse_map.get(int(v), -1) for v in child_mapped
		], dtype=np.int64)
	else:
		child_indices = child_mapped
	
	return child_indices


def encode_columns_autoregressive(
	indices: List[np.ndarray],
	dicts: List[Any],
	embedding_dim: int = 4,
	hidden_dims: List[int] = [16],
	num_epochs: int = 20,
	max_samples: Optional[int] = 50000,
	max_parents: int = 8,  # Limit number of parent columns to use
) -> Tuple[List[bytes], List[Optional[bytes]]]:
	"""
	Encode columns using fully autoregressive MLP: p(x_j | x_<j).
	
	Processes columns in order (0, 1, 2, ..., n-1).
	For column j, uses the most recent max_parents columns as parents (not all previous).
	
	Args:
		indices: List of column arrays [n_rows] each
		dicts: List of dictionaries (for compatibility)
		embedding_dim: MLP embedding dimension
		hidden_dims: MLP hidden layer dimensions
		num_epochs: Training epochs
		max_samples: Max samples for training
		max_parents: Maximum number of previous columns to use as parents (default: 8)
		
	Returns:
		(frames, mlp_models) where:
		- frames: List of compressed column data
		- mlp_models: List of serialized model bytes (one per column)
	"""
	if not TORCH_AVAILABLE:
		raise ImportError("PyTorch is required for autoregressive MLP compression")
	
	n_cols = len(indices)
	if n_cols == 0:
		return [], []
	
	frames: List[bytes] = [b""] * n_cols
	mlp_models: List[Optional[bytes]] = [None] * n_cols
	
	# Process columns in order
	for j in range(n_cols):
		child_col = indices[j]
		
		# Get the most recent max_parents columns as parents (not all previous)
		# This limits model size and training time
		parent_columns = indices[max(0, j - max_parents):j]
		
		# Skip if too many unique values
		unique_children = len(np.unique(child_col[child_col != -1]))
		if unique_children > 10000 or unique_children < 2:
			# Fallback to independent encoding
			from .codec import compress_numeric_array_fast
			frames[j] = compress_numeric_array_fast(child_col.astype(np.int64), use_ace=True, prefer_delta=False)
			mlp_models[j] = None
			continue
		
		# Skip MLP if too many parents (would be too slow)
		if len(parent_columns) > max_parents:
			# Fallback to independent encoding
			from .codec import compress_numeric_array_fast
			frames[j] = compress_numeric_array_fast(child_col.astype(np.int64), use_ace=True, prefer_delta=False)
			mlp_models[j] = None
			continue
		
		# Adjust training parameters based on number of parents
		# More parents = smaller model, fewer epochs, fewer samples
		effective_embedding_dim = embedding_dim
		effective_hidden_dims = hidden_dims.copy()
		effective_epochs = num_epochs
		effective_max_samples = max_samples
		
		if len(parent_columns) > 4:
			# Reduce model size and training time for many parents
			effective_embedding_dim = max(2, embedding_dim - 1)
			effective_hidden_dims = [max(8, hd - 4) for hd in hidden_dims]
			effective_epochs = max(10, num_epochs - 5)
			effective_max_samples = min(30000, max_samples) if max_samples else 30000
		elif len(parent_columns) > 2:
			# Moderate reduction
			effective_epochs = max(15, num_epochs - 3)
			effective_max_samples = min(40000, max_samples) if max_samples else 40000
		
		# Encode with autoregressive MLP
		encoded_data, model, total_bits = encode_column_autoregressive(
			child_indices=child_col,
			parent_columns=parent_columns,
			embedding_dim=effective_embedding_dim,
			hidden_dims=effective_hidden_dims,
			num_epochs=effective_epochs,
			max_samples=effective_max_samples,
		)
		
		if encoded_data is not None and model is not None:
			# Serialize model
			mlp_params = serialize_mlp_params(model)
			mlp_models[j] = mlp_params
			
			# Create frame with MLP marker
			mlp_frame = b"ARMLP\x00" + struct.pack("<I", len(encoded_data)) + encoded_data
			frames[j] = mlp_frame
		else:
			# Fallback to independent encoding
			from .codec import compress_numeric_array_fast
			frames[j] = compress_numeric_array_fast(child_col.astype(np.int64), use_ace=True, prefer_delta=False)
			mlp_models[j] = None
	
	return frames, mlp_models


def decode_columns_autoregressive(
	frames: List[bytes],
	mlp_models: List[Optional[bytes]],
	n_rows: int,
	max_parents: int = 8,  # Must match encoding
) -> List[np.ndarray]:
	"""
	Decode columns from fully autoregressive MLP compression.
	
	Args:
		frames: List of compressed column data
		mlp_models: List of serialized model bytes
		n_rows: Number of rows
		max_parents: Maximum number of parent columns (must match encoding)
		
	Returns:
		List of decoded column arrays
	"""
	if not TORCH_AVAILABLE:
		raise ImportError("PyTorch is required for autoregressive MLP decompression")
	
	n_cols = len(frames)
	decoded_cols: List[np.ndarray] = [None] * n_cols
	
	# Process columns in order (must decode sequentially)
	for j in range(n_cols):
		frame = frames[j]
		model_bytes = mlp_models[j] if j < len(mlp_models) else None
		
		# Check for autoregressive MLP frame
		if frame[:6] == b"ARMLP\x00":
			if model_bytes is None:
				raise RuntimeError(f"MLP model missing for column {j}")
			
			# Deserialize model
			_, model = deserialize_mlp_params(model_bytes)
			
			# Decode frame
			ptr = 6
			data_len = struct.unpack("<I", frame[ptr:ptr+4])[0]
			ptr += 4
			encoded_data = frame[ptr:ptr+data_len]
			
			# Get the most recent max_parents columns as parents (same as encoding)
			parent_columns = decoded_cols[max(0, j - max_parents):j]
			
			# Decode
			decoded_cols[j] = decode_column_autoregressive(
				encoded_data=encoded_data,
				model=model,
				parent_columns=parent_columns,
				n_rows=n_rows,
			)
		else:
			# Fallback: independent decoding
			from .codec import decompress_numeric_array
			decoded_cols[j] = decompress_numeric_array(frame).astype(np.int32)
	
	return decoded_cols
