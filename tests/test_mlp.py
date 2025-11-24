"""
Unit tests for Tiny MLP extension.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import tempfile
import os

try:
	import torch
	TORCH_AVAILABLE = True
except ImportError:
	TORCH_AVAILABLE = False
	pytestmark = pytest.mark.skip("PyTorch not available")

from src.tabcl.tiny_mlp import (
	TinyMLP,
	train_tiny_mlp,
	serialize_mlp_params,
	deserialize_mlp_params,
	compute_mlp_model_bits,
	compute_mlp_data_bits,
)
from src.tabcl.mlp_conditional import (
	encode_column_with_mlp,
	decode_column_with_mlp,
	compare_mlp_vs_histogram_mdl,
)
from src.tabcl.cli import compress_file, decompress_file


@pytest.fixture
def synthetic_data():
	"""Generate synthetic data with clear parent-child dependency."""
	np.random.seed(42)
	n_rows = 1000
	
	# Parent: 10 categories
	parent = np.random.randint(0, 10, size=n_rows)
	
	# Child: depends on parent with some noise
	# Each parent value maps to a preferred child value
	child = np.zeros(n_rows, dtype=np.int64)
	for i in range(n_rows):
		base_child = parent[i] % 5  # 5 child categories
		# Add some noise (20% chance of different value)
		if np.random.random() < 0.2:
			child[i] = np.random.randint(0, 5)
		else:
			child[i] = base_child
	
	return parent, child


def test_tiny_mlp_creation():
	"""Test creating a TinyMLP model."""
	model = TinyMLP(
		embedding_dims=[16],
		embedding_vocab_sizes=[10],
		hidden_dims=[64],
		num_classes=5,
		use_gelu=True
	)
	
	assert len(model.embeddings) == 1
	assert model.embeddings[0].num_embeddings == 10
	assert model.embeddings[0].embedding_dim == 16
	assert model.num_classes == 5


def test_tiny_mlp_forward(synthetic_data):
	"""Test forward pass of TinyMLP."""
	parent, child = synthetic_data
	
	model = TinyMLP(
		embedding_dims=[16],
		embedding_vocab_sizes=[10],
		hidden_dims=[64],
		num_classes=5,
		use_gelu=True
	)
	
	# Create feature indices
	parent_tensor = torch.from_numpy(parent[:10]).long()
	
	# Forward pass
	logits = model.forward([parent_tensor])
	
	assert logits.shape == (10, 5)


def test_train_tiny_mlp(synthetic_data):
	"""Test training a TinyMLP model."""
	parent, child = synthetic_data
	
	model = train_tiny_mlp(
		child_indices=child,
		parent_indices=parent,
		embedding_dim=16,
		hidden_dims=[64],
		num_epochs=5,
		batch_size=128,
	)
	
	assert model is not None
	assert hasattr(model, 'parent_map')
	assert hasattr(model, 'child_map')
	assert hasattr(model, 'inverse_child_map')


def test_serialize_deserialize_mlp(synthetic_data):
	"""Test serialization and deserialization of MLP models."""
	parent, child = synthetic_data
	
	# Train model
	model = train_tiny_mlp(
		child_indices=child,
		parent_indices=parent,
		embedding_dim=16,
		hidden_dims=[64],
		num_epochs=5,
	)
	
	assert model is not None
	
	# Serialize
	params_bytes = serialize_mlp_params(model)
	assert len(params_bytes) > 0
	
	# Deserialize
	metadata, model_restored = deserialize_mlp_params(params_bytes)
	
	assert model_restored is not None
	assert model_restored.num_classes == model.num_classes
	assert len(model_restored.embeddings) == len(model.embeddings)
	
	# Test that predictions match
	# Map parent values using model's parent_map
	parent_mapped = np.array([model.parent_map.get(v, 0) for v in parent[:10]], dtype=np.int64)
	device = next(model.parameters()).device
	parent_tensor = torch.from_numpy(parent_mapped.copy()).long().to(device)
	model.eval()
	model_restored.eval()
	
	with torch.no_grad():
		logits_orig = model.forward([parent_tensor])
		logits_restored = model_restored.forward([parent_tensor])
	
	np.testing.assert_allclose(logits_orig.cpu().numpy(), logits_restored.cpu().numpy(), rtol=1e-5)


def test_compute_mlp_model_bits(synthetic_data):
	"""Test computing model cost in bits."""
	parent, child = synthetic_data
	
	model = train_tiny_mlp(
		child_indices=child,
		parent_indices=parent,
		embedding_dim=16,
		hidden_dims=[64],
		num_epochs=5,
	)
	
	model_bits = compute_mlp_model_bits(model)
	assert model_bits > 0
	assert model_bits < 1e6  # Should be reasonable


def test_compute_mlp_data_bits(synthetic_data):
	"""Test computing data cost in bits."""
	parent, child = synthetic_data
	
	model = train_tiny_mlp(
		child_indices=child,
		parent_indices=parent,
		embedding_dim=16,
		hidden_dims=[64],
		num_epochs=5,
	)
	
	data_bits = compute_mlp_data_bits(model, parent, child)
	assert data_bits > 0
	assert data_bits < len(child) * 10  # Should be less than 10 bits per sample


def test_encode_decode_column_with_mlp(synthetic_data):
	"""Test encoding and decoding a column with MLP."""
	parent, child = synthetic_data
	
	# Encode
	encoded_data, model, total_bits = encode_column_with_mlp(
		child_indices=child,
		parent_indices=parent,
		dicts=None,
	)
	
	assert encoded_data is not None
	assert model is not None
	assert total_bits > 0
	
	# Decode
	decoded = decode_column_with_mlp(
		encoded_data=encoded_data,
		model=model,
		parent_indices=parent,
		n_rows=len(child),
	)
	
	# Check roundtrip
	np.testing.assert_array_equal(decoded, child)


def test_compare_mlp_vs_histogram_mdl(synthetic_data):
	"""Test MDL comparison between MLP and histogram."""
	parent, child = synthetic_data
	
	# Compare with histogram
	use_mlp, model, total_bits = compare_mlp_vs_histogram_mdl(
		child_indices=child,
		parent_indices=parent,
		histogram_model_bits=1000.0,
		histogram_data_bits=5000.0,
		margin=0.0,
	)
	
	# Should choose one or the other
	assert isinstance(use_mlp, bool)
	assert total_bits > 0


def test_roundtrip_compression():
	"""Test full compression/decompression roundtrip with MLP."""
	# Create a simple CSV file
	with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
		f.write("parent,child\n")
		for i in range(100):
			parent_val = i % 10
			child_val = parent_val % 5
			f.write(f"{parent_val},{child_val}\n")
		input_path = f.name
	
	try:
		output_path = input_path + ".tabcl"
		restored_path = input_path + ".restored.csv"
		
		# Compress with MLP
		compress_file(
			input_path=input_path,
			output_path=output_path,
			delimiter=",",
			use_mlp=True,
		)
		
		# Decompress
		decompress_file(
			input_path=output_path,
			output_path=restored_path,
		)
		
		# Read and compare
		with open(input_path, 'r') as f:
			original = f.read()
		with open(restored_path, 'r') as f:
			restored = f.read()
		
		# Should match (ignoring header)
		original_lines = original.strip().split('\n')[1:]
		restored_lines = restored.strip().split('\n')[1:]
		assert len(original_lines) == len(restored_lines)
		
		# Compare data rows
		for orig, rest in zip(original_lines, restored_lines):
			assert orig == rest
		
	finally:
		# Cleanup
		for path in [input_path, output_path, restored_path]:
			if os.path.exists(path):
				os.unlink(path)

