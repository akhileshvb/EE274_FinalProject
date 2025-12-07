"""
Tiny MLP used as an optional conditional model for some columns.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import struct
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
TORCH_AVAILABLE = True


class TinyMLP(nn.Module):
	"""Small MLP for conditional probability estimation."""

	def __init__(
		self,
		embedding_dims: List[int],  # Embedding dimension for each categorical feature
		embedding_vocab_sizes: List[int],  # Vocabulary size for each categorical feature
		hidden_dims: List[int] = [64],
		num_classes: int = 0,
		use_gelu: bool = True,
	):
		if not TORCH_AVAILABLE:
			raise ImportError("PyTorch is required for MLP extension. Install with: pip install torch")
		
		super().__init__()
		
		self.embedding_dims = embedding_dims
		self.embedding_vocab_sizes = embedding_vocab_sizes
		self.hidden_dims = hidden_dims
		self.num_classes = num_classes
		self.use_gelu = use_gelu
		
		# Embedding layers for categorical features
		self.embeddings = nn.ModuleList([
			nn.Embedding(vocab_size, emb_dim)
			for vocab_size, emb_dim in zip(embedding_vocab_sizes, embedding_dims)
		])
		
		# Compute input dimension: sum of all embedding dimensions
		input_dim = sum(embedding_dims)
		
		# Build hidden layers
		layers = []
		prev_dim = input_dim
		for hidden_dim in hidden_dims:
			layers.append(nn.Linear(prev_dim, hidden_dim))
			layers.append(nn.GELU() if use_gelu else nn.ReLU())
			prev_dim = hidden_dim
		
		# Output layer
		layers.append(nn.Linear(prev_dim, num_classes))
		# Note: We'll apply softmax during inference, not in the model
		
		self.mlp = nn.Sequential(*layers)
	
	def forward(self, feature_indices: List[torch.Tensor]) -> torch.Tensor:
		"""Forward pass. Returns logits of shape [batch_size, num_classes]."""
		# Embed each feature
		embedded = []
		for i, indices in enumerate(feature_indices):
			emb = self.embeddings[i](indices)
			embedded.append(emb)
		
		# Concatenate all embeddings
		x = torch.cat(embedded, dim=1)
		
		# Pass through MLP
		logits = self.mlp(x)
		
		return logits
	
	def predict_proba(self, feature_indices: List[torch.Tensor]) -> np.ndarray:
		"""
		Predict probability distribution over classes.
		
		Args:
			feature_indices: List of tensors with feature indices
		
		Returns:
			Probability array of shape [batch_size, num_classes]
		"""
		self.eval()
		with torch.no_grad():
			logits = self.forward(feature_indices)
			probs = torch.softmax(logits, dim=1)
			return probs.cpu().numpy()


def train_tiny_mlp(
	child_indices: np.ndarray,
	parent_indices: Optional[np.ndarray] = None,
	embedding_dim: int = 16,
	hidden_dims: List[int] = [64],
	num_epochs: int = 15,
	batch_size: int = 2048,
	learning_rate: float = 0.001,
	max_samples: Optional[int] = 50000,
	device: Optional[str] = None,
	all_unique_children: Optional[np.ndarray] = None,
	all_unique_parents: Optional[np.ndarray] = None,
	autoregressive_parents: Optional[List[np.ndarray]] = None,
	autoregressive_vocab_sizes: Optional[List[int]] = None,
) -> Optional[TinyMLP]:
	"""
	Train a TinyMLP to predict a child column from one or more parent columns.

	Returns the trained model, or None if training is skipped or fails.
	"""
	if not TORCH_AVAILABLE:
		return None
	
	if len(child_indices) < 100:
		# Too small to train
		return None
	
	# Determine mode
	is_autoregressive = autoregressive_parents is not None and len(autoregressive_parents) > 0
	
	if is_autoregressive:
		# Autoregressive mode: multiple parent columns
		n_rows = len(child_indices)
		child_sample = child_indices
		
		# Get unique children
		if all_unique_children is not None:
			unique_children_list = all_unique_children
		else:
			unique_children_list = np.unique(child_sample)
		unique_children_list = unique_children_list[unique_children_list != -1]
		unique_children = len(unique_children_list)
		
		if unique_children > 10000:
			return None
		
		# Build child map
		child_map = {v: i for i, v in enumerate(unique_children_list) if v != -1}
		
		# Build parent maps for each parent column
		parent_maps = []
		embedding_vocab_sizes = []
		parent_mapped_list = []
		
		for i, prev_col in enumerate(autoregressive_parents):
			if autoregressive_vocab_sizes and i < len(autoregressive_vocab_sizes):
				vocab_size = autoregressive_vocab_sizes[i]
				unique_parents_list = np.unique(prev_col)
				unique_parents_list = unique_parents_list[unique_parents_list != -1]
				# Use provided vocab size, but ensure we have enough unique values
				if len(unique_parents_list) > vocab_size:
					# Take most frequent values
					from collections import Counter
					counts = Counter(prev_col[prev_col != -1])
					most_common = [val for val, _ in counts.most_common(vocab_size)]
					unique_parents_list = np.array(most_common)
			else:
				unique_parents_list = np.unique(prev_col)
				unique_parents_list = unique_parents_list[unique_parents_list != -1]
			
			parent_map = {v: j for j, v in enumerate(unique_parents_list) if v != -1}
			parent_maps.append(parent_map)
			embedding_vocab_sizes.append(len(unique_parents_list))
			
			# Map parent column
			parent_mapped = np.array([
				parent_map.get(v, 0) if v != -1 else 0 for v in prev_col
			], dtype=np.int64)
			parent_mapped_list.append(parent_mapped)
		
		# Map child
		child_mapped = np.array([
			child_map.get(v, -1) if v != -1 else -1 for v in child_sample
		], dtype=np.int64)
		
		# Filter valid rows (all parents and child must be valid)
		valid_mask = (child_mapped != -1)
		for parent_mapped in parent_mapped_list:
			valid_mask = valid_mask & (parent_mapped >= 0)
		
		if not np.any(valid_mask) or np.sum(valid_mask) < 100:
			return None
		
		child_mapped = child_mapped[valid_mask]
		parent_mapped_list = [pm[valid_mask] for pm in parent_mapped_list]
		
		# Create model
		if device is None:
			device = 'cpu'
		
		model = TinyMLP(
			embedding_dims=[embedding_dim] * len(autoregressive_parents),
			embedding_vocab_sizes=embedding_vocab_sizes,
			hidden_dims=hidden_dims,
			num_classes=unique_children,
			use_gelu=True
		)
		if device != 'cpu':
			model = model.to(device)
		
		# Store parent maps in model for decoding
		for i, pm in enumerate(parent_maps):
			setattr(model, f'parent_map_{i}', pm)
		model.parent_map = parent_maps[0] if parent_maps else {}
		model.child_map = child_map
		
		# Convert to tensors
		parent_tensors = [torch.from_numpy(pm).long().to(device) for pm in parent_mapped_list]
		child_tensor = torch.from_numpy(child_mapped).long().to(device)
		
	else:
		# Conditional mode: single parent column
		if parent_indices is None:
			return None
		
		# Sample if needed
		n_rows = len(child_indices)
		if max_samples is not None and n_rows > max_samples:
			idx = np.random.choice(n_rows, size=max_samples, replace=False)
			child_sample = child_indices[idx]
			parent_sample = parent_indices[idx]
		else:
			child_sample = child_indices
			parent_sample = parent_indices
		
		# Get vocabulary sizes
		if all_unique_parents is not None:
			unique_parents_list = all_unique_parents
		else:
			unique_parents_list = np.unique(parent_sample)
		
		if all_unique_children is not None:
			unique_children_list = all_unique_children
		else:
			unique_children_list = np.unique(child_sample)
		
		unique_children_list = unique_children_list[unique_children_list != -1]
		unique_parents_list = unique_parents_list[unique_parents_list != -1]
		
		unique_parents = len(unique_parents_list)
		unique_children = len(unique_children_list)
		
		if unique_children > 10000:
			return None
		
		parent_map = {v: i for i, v in enumerate(unique_parents_list) if v != -1}
		child_map = {v: i for i, v in enumerate(unique_children_list) if v != -1}
		
		if -1 in parent_map or -1 in child_map:
			raise RuntimeError("ERROR: -1 (sentinel value) found in parent_map or child_map!")
		
		parent_mapped = np.array([
			parent_map.get(v, -1) if v != -1 else -1 for v in parent_sample
		], dtype=np.int64)
		child_mapped = np.array([
			child_map.get(v, -1) if v != -1 else -1 for v in child_sample
		], dtype=np.int64)
		
		valid_mask = (parent_mapped != -1) & (child_mapped != -1)
		if not np.any(valid_mask) or np.sum(valid_mask) < 100:
			return None
		
		parent_mapped = parent_mapped[valid_mask]
		child_mapped = child_mapped[valid_mask]
		
		if device is None:
			device = 'cpu'
		
		model = TinyMLP(
			embedding_dims=[embedding_dim],
			embedding_vocab_sizes=[unique_parents],
			hidden_dims=hidden_dims,
			num_classes=unique_children,
			use_gelu=True
		)
		if device != 'cpu':
			model = model.to(device)
		
		model.parent_map = parent_map
		model.child_map = child_map
		
		parent_tensors = [torch.from_numpy(parent_mapped).long().to(device)]
		child_tensor = torch.from_numpy(child_mapped).long().to(device)
	
	# Training with improved hyperparameters
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
	criterion = nn.CrossEntropyLoss()
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
	
	model.train()
	n_samples = len(child_tensor)
	best_loss = float('inf')
	patience_counter = 0
	max_patience = 5  # Stop if no improvement for 5 epochs
	
	for epoch in range(num_epochs):
		# Shuffle
		perm = torch.randperm(n_samples, device=device)
		parent_tensors_shuffled = [pt[perm] for pt in parent_tensors]
		child_shuffled = child_tensor[perm]
		
		epoch_loss = 0.0
		n_batches = 0
		
		# Mini-batch training
		for i in range(0, n_samples, batch_size):
			end_idx = min(i + batch_size, n_samples)
			parent_batches = [pt[i:end_idx] for pt in parent_tensors_shuffled]
			child_batch = child_shuffled[i:end_idx]
			
			optimizer.zero_grad()
			logits = model.forward(parent_batches)
			loss = criterion(logits, child_batch)
			loss.backward()
			# Gradient clipping to prevent instability
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()
			
			epoch_loss += loss.item()
			n_batches += 1
		
		avg_loss = epoch_loss / n_batches if n_batches > 0 else epoch_loss
		scheduler.step(avg_loss)
		
		# Early stopping with patience
		if avg_loss < best_loss * 0.999:  # Require at least 0.1% improvement
			best_loss = avg_loss
			patience_counter = 0
		else:
			patience_counter += 1
			if epoch > 5 and patience_counter >= max_patience:
				break
	
	# Store mapping for decoding
	model.parent_map = parent_map
	model.child_map = child_map
	model.inverse_child_map = {i: v for v, i in child_map.items()}
	
	return model


def compute_mlp_model_bits(model: TinyMLP) -> float:
	"""
	Compute MDL cost (in bits) for storing the MLP model parameters.
	
	Args:
		model: Trained TinyMLP model
	
	Returns:
		Model cost in bits
	"""
	if model is None:
		return float('inf')
	
	# Serialize model parameters
	params_bytes = serialize_mlp_params(model)
	
	# Model cost = 8 bits per byte
	return len(params_bytes) * 8.0


def serialize_mlp_params(model: TinyMLP) -> bytes:
	"""
	Serialize MLP model parameters to bytes with quantization to reduce size.
	
	Format:
	[architecture_metadata: JSON][embedding_params...][mlp_params...]
	
	Args:
		model: TinyMLP model
	
	Returns:
		Serialized parameters as bytes
	"""
	parts = []
	
	# Architecture metadata
	metadata = {
		"embedding_dims": model.embedding_dims,
		"embedding_vocab_sizes": model.embedding_vocab_sizes,
		"hidden_dims": model.hidden_dims,
		"num_classes": model.num_classes,
		"use_gelu": model.use_gelu,
		"parent_map": {str(k): int(v) for k, v in model.parent_map.items()},
		"child_map": {str(k): int(v) for k, v in model.child_map.items()},
	}
	metadata_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
	parts.append(struct.pack("<I", len(metadata_json)))
	parts.append(metadata_json)
	
	# Quantize parameters to int16 to reduce model size (4x reduction)
	# Scale factor: use 2^15-1 as max to preserve sign
	quant_scale = 32767.0
	
	# Embedding parameters
	for emb in model.embeddings:
		# Get weight matrix [vocab_size, emb_dim]
		weight = emb.weight.data.cpu().numpy().astype(np.float32)
		# Quantize to int16
		weight_max = np.abs(weight).max()
		if weight_max > 0:
			weight_quantized = np.clip(weight / weight_max * quant_scale, -quant_scale, quant_scale).astype(np.int16)
			parts.append(struct.pack("<f", weight_max))  # Store scale factor
		else:
			weight_quantized = np.zeros_like(weight, dtype=np.int16)
			parts.append(struct.pack("<f", 1.0))
		parts.append(struct.pack("<I", weight_quantized.size))
		parts.append(weight_quantized.tobytes())
	
	# MLP parameters (linear layers)
	for module in model.mlp:
		if isinstance(module, nn.Linear):
			# Weight matrix [out_features, in_features]
			weight = module.weight.data.cpu().numpy().astype(np.float32)
			# Quantize to int16
			weight_max = np.abs(weight).max()
			if weight_max > 0:
				weight_quantized = np.clip(weight / weight_max * quant_scale, -quant_scale, quant_scale).astype(np.int16)
				parts.append(struct.pack("<f", weight_max))  # Store scale factor
			else:
				weight_quantized = np.zeros_like(weight, dtype=np.int16)
				parts.append(struct.pack("<f", 1.0))
			parts.append(struct.pack("<I", weight_quantized.size))
			parts.append(weight_quantized.tobytes())
			
			# Bias [out_features]
			bias = module.bias.data.cpu().numpy().astype(np.float32)
			# Quantize to int16
			bias_max = np.abs(bias).max()
			if bias_max > 0:
				bias_quantized = np.clip(bias / bias_max * quant_scale, -quant_scale, quant_scale).astype(np.int16)
				parts.append(struct.pack("<f", bias_max))  # Store scale factor
			else:
				bias_quantized = np.zeros_like(bias, dtype=np.int16)
				parts.append(struct.pack("<f", 1.0))
			parts.append(struct.pack("<I", bias_quantized.size))
			parts.append(bias_quantized.tobytes())
	
	return b"".join(parts)


def deserialize_mlp_params(data: bytes) -> Tuple[Dict, TinyMLP]:
	"""
	Deserialize MLP model parameters from bytes.
	
	Args:
		data: Serialized parameters
	
	Returns:
		(metadata_dict, model)
	"""
	if not TORCH_AVAILABLE:
		raise ImportError("PyTorch is required for MLP extension")
	
	ptr = 0
	
	# Read metadata
	metadata_len = struct.unpack("<I", data[ptr:ptr+4])[0]
	ptr += 4
	metadata_json = data[ptr:ptr+metadata_len].decode('utf-8')
	ptr += metadata_len
	metadata = json.loads(metadata_json)
	
	# Reconstruct parent/child maps
	parent_map = {int(k): int(v) for k, v in metadata["parent_map"].items()}
	child_map = {int(k): int(v) for k, v in metadata["child_map"].items()}
	
	# Create model on CPU first to avoid MPS device issues
	model = TinyMLP(
		embedding_dims=metadata["embedding_dims"],
		embedding_vocab_sizes=metadata["embedding_vocab_sizes"],
		hidden_dims=metadata["hidden_dims"],
		num_classes=metadata["num_classes"],
		use_gelu=metadata["use_gelu"]
	)
	
	# Determine target device (use CPU for deserialization to avoid MPS issues)
	# We'll move to device after loading if needed
	target_device = 'cpu'  # Always load on CPU first
	
	# Load embedding parameters (with quantization support)
	for i, emb in enumerate(model.embeddings):
		# Read scale factor (for quantized params) or assume float32
		scale = struct.unpack("<f", data[ptr:ptr+4])[0]
		ptr += 4
		param_size = struct.unpack("<I", data[ptr:ptr+4])[0]
		ptr += 4
		param_bytes = data[ptr:ptr+param_size*2]  # int16 = 2 bytes
		ptr += param_size * 2
		weight_quantized = np.frombuffer(param_bytes, dtype=np.int16).reshape(
			metadata["embedding_vocab_sizes"][i], metadata["embedding_dims"][i]
		).copy()
		# Dequantize
		weight = (weight_quantized.astype(np.float32) / 32767.0) * scale
		emb.weight.data = torch.from_numpy(weight).to(target_device)
	
	# Load MLP parameters (with quantization support)
	for module in model.mlp:
		if isinstance(module, nn.Linear):
			# Weight
			scale = struct.unpack("<f", data[ptr:ptr+4])[0]
			ptr += 4
			param_size = struct.unpack("<I", data[ptr:ptr+4])[0]
			ptr += 4
			param_bytes = data[ptr:ptr+param_size*2]  # int16 = 2 bytes
			ptr += param_size * 2
			out_features, in_features = module.weight.shape
			weight_quantized = np.frombuffer(param_bytes, dtype=np.int16).reshape(out_features, in_features).copy()
			# Dequantize
			weight = (weight_quantized.astype(np.float32) / 32767.0) * scale
			module.weight.data = torch.from_numpy(weight).to(target_device)
			
			# Bias
			scale = struct.unpack("<f", data[ptr:ptr+4])[0]
			ptr += 4
			param_size = struct.unpack("<I", data[ptr:ptr+4])[0]
			ptr += 4
			param_bytes = data[ptr:ptr+param_size*2]  # int16 = 2 bytes
			ptr += param_size * 2
			bias_quantized = np.frombuffer(param_bytes, dtype=np.int16).copy()
			# Dequantize
			bias = (bias_quantized.astype(np.float32) / 32767.0) * scale
			module.bias.data = torch.from_numpy(bias).to(target_device)
	
	# Restore maps
	model.parent_map = parent_map
	model.child_map = child_map
	model.inverse_child_map = {i: v for v, i in child_map.items()}
	
	# Verify that child_map is complete after deserialization
	expected_mapped_values = set(range(metadata["num_classes"]))
	actual_mapped_values = set(child_map.values())
	if expected_mapped_values != actual_mapped_values:
		missing = expected_mapped_values - actual_mapped_values
		raise RuntimeError(
			f"Child map is incomplete after deserialization: missing mapped values {sorted(list(missing))[:20]}. "
			f"Expected {metadata['num_classes']} values, got {len(actual_mapped_values)}. "
			f"This indicates a bug in serialization/deserialization."
		)
	
	return metadata, model


def compute_mlp_data_bits(
	model: TinyMLP,
	parent_indices: np.ndarray,
	child_indices: np.ndarray,
	max_samples: Optional[int] = 100000,
) -> float:
	"""
	Compute data cost (in bits) for encoding child column using MLP probabilities.
	
	Uses negative log-likelihood as an approximation of arithmetic coding cost.
	
	Args:
		model: Trained TinyMLP model
		parent_indices: Parent column indices [n_rows]
		child_indices: Child column indices [n_rows]
		max_samples: Maximum samples to use for computation (None = all)
	
	Returns:
		Data cost in bits (negative log-likelihood)
	"""
	if model is None:
		return float('inf')
	
	# Sample if needed
	n_rows = len(parent_indices)
	if max_samples is not None and n_rows > max_samples:
		idx = np.random.choice(n_rows, size=max_samples, replace=False)
		parent_sample = parent_indices[idx]
		child_sample = child_indices[idx]
	else:
		parent_sample = parent_indices
		child_sample = child_indices
	
	# Map parent indices using model's parent_map
	parent_mapped = np.array([
		model.parent_map.get(v, 0) for v in parent_sample
	], dtype=np.int64)
	
	# Map child indices using model's child_map
	child_mapped = np.array([
		model.child_map.get(v, 0) for v in child_sample
	], dtype=np.int64)
	
	# Convert to tensors
	device = next(model.parameters()).device
	parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)  # Make writable and ensure long type
	
	# Get probabilities
	model.eval()
	with torch.no_grad():
		logits = model.forward([parent_tensor])
		probs = torch.softmax(logits, dim=1)
		probs_np = probs.cpu().numpy()
	
	# Compute negative log-likelihood
	nll = 0.0
	for i, true_class in enumerate(child_mapped):
		if true_class < probs_np.shape[1]:
			p = max(probs_np[i, true_class], 1e-10)  # Avoid log(0)
			nll -= np.log2(p)
	
	# Scale to full dataset size
	if max_samples is not None and n_rows > max_samples:
		nll = nll * (n_rows / max_samples)
	
	return nll

