"""MLP-based conditional encoding and decoding of columns."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import struct

from .tiny_mlp import (
	train_tiny_mlp,
	compute_mlp_model_bits,
	compute_mlp_data_bits,
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


def encode_column_with_mlp(
	child_indices: np.ndarray,
	parent_indices: np.ndarray,
	dicts: Any,
	embedding_dim: int = 4,  # Smaller default to reduce model cost
	hidden_dims: List[int] = [16],  # Smaller default to reduce model cost
	num_epochs: int = 20,  # More epochs for better training
	max_samples: Optional[int] = 50000,
) -> Tuple[Optional[bytes], Optional[TinyMLP], float]:
	"""Encode a child column using an MLP conditional model.

	Returns (encoded_bytes, model, total_bits). If the MLP is not usable or not
	beneficial, returns (None, None, inf).
	"""
	if not TORCH_AVAILABLE or zl is None:
		return None, None, float('inf')
	
	# Get all unique values from full dataset to ensure mapping covers everything
	# Filter out -1 (sentinel value) if present
	all_unique_children = np.unique(child_indices)
	all_unique_parents = np.unique(parent_indices)
	
	# Remove -1 if it's present (it's a sentinel value, not a real data value)
	all_unique_children = all_unique_children[all_unique_children != -1]
	all_unique_parents = all_unique_parents[all_unique_parents != -1]
	
	# Train MLP
	model = train_tiny_mlp(
		child_indices=child_indices,
		parent_indices=parent_indices,
		embedding_dim=embedding_dim,
		hidden_dims=hidden_dims,
		num_epochs=num_epochs,
		max_samples=max_samples,
		all_unique_children=all_unique_children,
		all_unique_parents=all_unique_parents,
	)
	
	if model is None:
		return None, None, float('inf')
	
	# Compute model cost
	model_bits = compute_mlp_model_bits(model)
	
	# Compute data cost using actual rank encoding (more accurate than NLL)
	# We'll compute this after encoding, but for now use NLL as approximation
	data_bits_approx = compute_mlp_data_bits(
		model,
		parent_indices,
		child_indices,
		max_samples=None,  # Use all data for final encoding
	)
	
	# Get predictions for all rows
	device = next(model.parameters()).device
	# Map parent indices, using 0 as fallback for unmapped values
	parent_mapped = np.array([
		model.parent_map.get(v, 0) for v in parent_indices
	], dtype=np.int64)
	# Ensure parent_mapped is within valid range
	parent_mapped = np.clip(parent_mapped, 0, model.embedding_vocab_sizes[0] - 1)
	parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
	
	model.eval()
	with torch.no_grad():
		logits = model.forward([parent_tensor])
		probs = torch.softmax(logits, dim=1)
		probs_np = probs.cpu().numpy()
		# Ensure probabilities are valid (should be, but be safe)
		probs_np = np.clip(probs_np, 1e-10, 1.0)
		probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)  # Renormalize
		predicted_classes = np.argmax(probs_np, axis=1)
	
	# Map child indices to the model's child_map space
	child_mapped = np.array([
		model.child_map.get(v, -1) if v != -1 else -1 for v in child_indices
	], dtype=np.int64)
	
	# If there are unmapped values, something went wrong with the mapping:
	unmapped_mask = (child_mapped == -1) & (child_indices != -1)
	if np.any(unmapped_mask):
		unmapped_count = np.sum(unmapped_mask)
		unmapped_orig_vals = np.unique(child_indices[unmapped_mask])
		raise RuntimeError(
			f"ERROR: {unmapped_count} child values could not be mapped: {unmapped_orig_vals[:10]}. "
			f"This means all_unique_children was incomplete or the model's child_map is missing values."
		)
	
	# For rows where child_indices == -1 (sentinel), skip MLP if they are too frequent.
	if np.any(child_indices == -1):
		sentinel_count = np.sum(child_indices == -1)
		if sentinel_count > len(child_indices) * 0.1:  # More than 10% are sentinels
			# Too many sentinels - MLP won't work well
			return None, None, float('inf')
	
	# NEW APPROACH: Use probability-weighted rank encoding instead of residuals
	# This is much more efficient when MLP assigns high probability to correct values
	# For each row, sort child values by MLP probability (descending) and encode the rank of the actual value
	# This is much more efficient when MLP assigns high probability to correct values
	
	# Ensure all values are in valid range
	child_mapped = np.clip(child_mapped, 0, model.num_classes - 1)
	
	# Vectorized rank encoding: much faster than per-row loop
	# For each row, we need to find the rank of child_mapped[row_idx] in the probability-sorted list
	# Vectorized approach: argsort probabilities (descending) for each row
	prob_sorted_indices = np.argsort(probs_np, axis=1)[:, ::-1]  # [n_rows, num_classes], sorted descending
	
	# Fully vectorized rank computation using advanced indexing
	# Create a 2D array where ranks[row, child_value] = rank
	# We can do this by: for each row, create an inverse mapping
	n_rows = len(child_mapped)
	ranks = np.zeros(n_rows, dtype=np.int32)
	
	# Vectorized: for each row, find the position of child_mapped[row] in prob_sorted_indices[row]
	# We can use: ranks[row] = position where prob_sorted_indices[row, :] == child_mapped[row]
	# This is equivalent to: ranks[row] = argwhere(prob_sorted_indices[row] == child_mapped[row])[0]
	# We can vectorize this by creating a comparison matrix
	row_indices = np.arange(n_rows)
	# Create a boolean mask: [n_rows, num_classes] where mask[row, col] = (prob_sorted_indices[row, col] == child_mapped[row])
	match_matrix = prob_sorted_indices == child_mapped[:, np.newaxis]
	# Find first match for each row
	ranks = np.argmax(match_matrix, axis=1).astype(np.int32)
	# If no match found (shouldn't happen), argmax returns 0, but we want to verify
	# Check for rows where no match was found
	no_match_mask = ~np.any(match_matrix, axis=1)
	ranks[no_match_mask] = model.num_classes - 1  # Fallback for unmapped values
	
	# Compress ranks using numeric codec (ranks should be small when MLP is good)
	from .codec import compress_numeric_array_fast
	try:
		ranks_compressed = compress_numeric_array_fast(
			ranks.astype(np.int64),
			use_ace=True,  # ACE is good for small integers with low entropy
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
	# Use actual compressed size for accurate cost
	data_bits = len(combined) * 8.0
	total_bits = model_bits + data_bits
	
	return combined, model, total_bits


def decode_column_with_mlp(
	encoded_data: bytes,
	model: TinyMLP,
	parent_indices: np.ndarray,
	n_rows: int,
) -> np.ndarray:
	"""
	Decode a child column using MLP-based conditional decompression.
	
	Args:
		encoded_data: Compressed residuals data
		model: Trained TinyMLP model
		parent_indices: Parent column indices [n_rows]
		n_rows: Number of rows
	
	Returns:
		Decoded child column indices [n_rows]
	"""
	if zl is None:
		raise RuntimeError("OpenZL not available for decompression")
	
	# Step 1: Reconstruct MLP predictions for all rows
	device = next(model.parameters()).device
	# Map parent indices, using 0 as fallback for unmapped values (same as encoding)
	parent_mapped = np.array([
		model.parent_map.get(v, 0) for v in parent_indices
	], dtype=np.int64)
	# Ensure parent_mapped is within valid range
	parent_mapped = np.clip(parent_mapped, 0, model.embedding_vocab_sizes[0] - 1)
	parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
	
	model.eval()
	with torch.no_grad():
		logits = model.forward([parent_tensor])
		probs = torch.softmax(logits, dim=1)
		probs_np = probs.cpu().numpy()
		# Ensure probabilities are valid (should be, but be safe)
		probs_np = np.clip(probs_np, 1e-10, 1.0)
		probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)  # Renormalize
	
	# Step 2: Decode ranks
	from .codec import decompress_numeric_array
	try:
		ranks = decompress_numeric_array(encoded_data)
	except Exception:
		# Fallback: try OpenZL decompression
		if zl is not None:
			try:
				dctx = zl.DCtx()
				outs = dctx.decompress(encoded_data)
				if len(outs) != 1 or outs[0].type != zl.Type.Numeric:
					raise RuntimeError("Unexpected OpenZL output")
				ranks = outs[0].data
			except Exception:
				# Last resort: assume raw bytes
				ranks = np.frombuffer(encoded_data, dtype=np.int64)
		else:
			ranks = np.frombuffer(encoded_data, dtype=np.int64)
	
	# Ensure ranks match expected length
	if len(ranks) != n_rows:
		# Pad or truncate if needed (shouldn't happen, but be safe)
		if len(ranks) < n_rows:
			ranks = np.pad(ranks, (0, n_rows - len(ranks)), mode='constant')
		else:
			ranks = ranks[:n_rows]
	
	# Step 3: Reconstruct actual classes from ranks using probability distributions
	# Vectorized decoding: much faster
	# Sort probabilities for each row (descending) - same as encoding
	prob_sorted_indices = np.argsort(probs_np, axis=1)[:, ::-1]  # [n_rows, num_classes], sorted descending
	
	# For each row, get the child_value at position ranks[row_idx] in prob_sorted_indices[row_idx]
	# Clamp ranks to valid range to prevent index errors
	ranks_clamped = np.clip(ranks, 0, model.num_classes - 1).astype(np.int32)
	
	# Use advanced indexing to get child values (fully vectorized)
	row_indices = np.arange(n_rows)
	child_mapped = prob_sorted_indices[row_indices, ranks_clamped]
	
	# Ensure all decoded values are valid
	child_mapped = np.clip(child_mapped, 0, model.num_classes - 1)
	
	# Step 4: Map back to original class indices using inverse_child_map
	# Always build inverse map from child_map to ensure it's complete
	if hasattr(model, 'child_map') and model.child_map:
		# Build inverse map: mapped_value -> original_value
		# Check for issues in child_map first
		problematic_entries = [(orig, mapped) for orig, mapped in model.child_map.items() if orig == -1 or mapped < 0 or mapped >= model.num_classes]
		if problematic_entries:
			print(f"WARNING: Found {len(problematic_entries)} problematic entries in child_map:")
			for orig, mapped in problematic_entries[:10]:
				print(f"  Original: {orig}, Mapped: {mapped}")
		
		inverse_map = {}
		for orig, mapped in model.child_map.items():
			mapped_int = int(mapped)
			orig_int = int(orig)
			if mapped_int in inverse_map:
				# Duplicate mapped value - this shouldn't happen
				print(f"WARNING: Duplicate mapped value {mapped_int} in child_map!")
				print(f"  First original: {inverse_map[mapped_int]}, Second original: {orig_int}")
			inverse_map[mapped_int] = orig_int
		
		
		# Verify that all values in [0, num_classes-1] are in the map
		# If not, there's a bug in how child_map was created
		expected_range = set(range(model.num_classes))
		actual_range = set(inverse_map.keys())
		if expected_range != actual_range:
			missing = expected_range - actual_range
			raise RuntimeError(
				f"Child map is incomplete: missing {len(missing)} values. "
				f"Expected range: [0, {model.num_classes-1}], "
				f"Actual range: [{min(actual_range)}, {max(actual_range)}]"
			)
		
		# Build child_indices - ensure type consistency for lookup
		child_indices = np.array([
			inverse_map[int(v)] for v in child_mapped
		], dtype=np.int64)
		
		# Check for unmapped values (shouldn't happen if child_map is complete)
		unmapped_count = 0  # Should be 0 if inverse_map is complete
		
		# All values should be mapped (inverse_map is complete)
	else:
		# Fallback: if no inverse map, try to reconstruct from child_map
		if hasattr(model, 'child_map') and model.child_map:
			# Build inverse map on the fly
			inverse_map = {mapped: orig for orig, mapped in model.child_map.items()}
			child_indices = np.array([
				inverse_map.get(int(v), -1)
				for v in child_mapped
			], dtype=np.int64)
			if np.any(child_indices == -1):
				raise RuntimeError("Decoding error: could not reconstruct all values from child_map")
		else:
			# Last resort: assume direct mapping (shouldn't happen in practice)
			child_indices = child_mapped
	
	return child_indices


def compare_mlp_vs_histogram_mdl(
	child_indices: np.ndarray,
	parent_indices: np.ndarray,
	histogram_model_bits: float,
	histogram_data_bits: float,
	embedding_dim: int = 4,  # Smaller to reduce model cost
	hidden_dims: List[int] = [16],  # Smaller to reduce model cost
	num_epochs: int = 30,  # More epochs for better training
	max_samples: Optional[int] = 50000,
	margin: float = 0.0,
) -> Tuple[bool, Optional[TinyMLP], float]:
	"""
	Compare MLP vs histogram using MDL principle.
	
	Args:
		child_indices: Child column indices
		parent_indices: Parent column indices
		histogram_model_bits: Model cost for histogram approach
		histogram_data_bits: Data cost for histogram approach
		embedding_dim: MLP embedding dimension
		hidden_dims: MLP hidden layer dimensions
		num_epochs: Training epochs
		max_samples: Max samples for training
		margin: Additional margin required for MLP to win (bits)
	
	Returns:
		(use_mlp, model, mlp_total_bits)
		use_mlp: True if MLP should be used, False otherwise
		model: Trained MLP model (if use_mlp=True) or None
		mlp_total_bits: Total MLP cost (if use_mlp=True) or histogram cost
	"""
	if not TORCH_AVAILABLE:
		return False, None, histogram_model_bits + histogram_data_bits
	
	# Compute histogram total cost
	hist_total = histogram_model_bits + histogram_data_bits
	
	# Try MLP encoding
	encoded_data, model, mlp_total = encode_column_with_mlp(
		child_indices=child_indices,
		parent_indices=parent_indices,
		dicts=None,
		embedding_dim=embedding_dim,
		hidden_dims=hidden_dims,
		num_epochs=num_epochs,
		max_samples=max_samples,
	)
	
	# Debug: print comparison
	# print(f"MLP: {mlp_total:.1f} bits, Hist: {hist_total:.1f} bits, Margin: {margin:.1f}, Use MLP: {mlp_total < hist_total + margin}")
	
	if model is None or mlp_total >= hist_total + margin:
		return False, None, hist_total
	
	# MLP wins
	return True, model, mlp_total

