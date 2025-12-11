from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import struct

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from .codec import generic_bytes_compress, generic_bytes_decompress, compress_numeric_array_fast, decompress_numeric_array
from .tiny_mlp import TinyMLP, train_tiny_mlp, serialize_mlp_params, deserialize_mlp_params
from .conditional import orient_forest, _encode_varint, _decode_varint
from .mi import compute_empirical_mi
from .forest import build_mdl_weighted_forest_two_phase, build_maximum_weight_forest
from .codec import mdl_cost_fn_fast, mdl_cost_fn_openzl


class NeuralForestCompressor:
    """Compressor that replaces some histogram models with TinyMLP predictors."""
    def __init__(
        self,
        embedding_dim: int = 8,
        hidden_dims: List[int] = [32, 32],
        num_epochs: int = 20,  # Increased for better learning
        batch_size: int = 2048,
        max_samples: Optional[int] = 50000,
        learning_rate: float = 0.001,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.learning_rate = learning_rate
    
    def compress_columns(
        self,
        indices: List[np.ndarray],
        parents: List[int],
        dicts: List[Any],
    ) -> Tuple[List[bytes], List[Optional[bytes]], Dict[str, Any]]:
        """
        Compress columns using neural MLP-based approach.
        
        Args:
            indices: List of column arrays [n_rows] each
            parents: Parent array (from forest structure)
            dicts: List of dictionaries (for compatibility)
        
        Returns:
            (frames, models, metadata) where:
            - frames: List of compressed column data
            - models: List of serialized model bytes (one per column, None for roots)
            - metadata: Dict with parent structure and other info
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural compression. Install with: pip install torch")
        
        n_rows = len(indices[0]) if indices else 0
        n_cols = len(indices)
        
        if n_rows == 0 or n_cols == 0:
            return [], [], {}
        
        frames = [None] * n_cols
        models = [None] * n_cols
        
        # Process roots first, then children (for dependency order)
        root_ids = [j for j, p in enumerate(parents) if p == -1]
        child_ids = [j for j, p in enumerate(parents) if p != -1]
        
        # Encode roots (no parent, independent encoding)
        for col_idx in root_ids:
            col_data = indices[col_idx]
            frame_bytes, model_bytes = self._encode_independent(col_data)
            frames[col_idx] = frame_bytes
            models[col_idx] = model_bytes
        
        # Encode children (conditional on parent, using MLP)
        for col_idx in child_ids:
            parent_idx = parents[col_idx]
            col_data = indices[col_idx]
            parent_data = indices[parent_idx]
            
            frame_bytes, model_bytes = self._encode_conditional_mlp(
                child_col=col_data,
                parent_col=parent_data,
                col_idx=col_idx,
            )
            frames[col_idx] = frame_bytes
            models[col_idx] = model_bytes
        
        metadata = {"parents": parents}
        
        return frames, models, metadata
    
    def _encode_independent(self, col_data: np.ndarray) -> Tuple[bytes, Optional[bytes]]:
        """Encode a column independently (no parent)."""
        # Use same encoding as histogram approach for roots
        from .conditional import _try_all_compression_methods, _choose_best_compression
        
        if col_data.size > 10:
            frame_bytes = _try_all_compression_methods(col_data, None)
        elif col_data.size > 3:
            # Try multiple methods
            candidates = []
            try:
                candidates.append(compress_numeric_array_fast(col_data, use_ace=False, prefer_delta=True))
            except Exception:
                pass
            try:
                candidates.append(compress_numeric_array_fast(col_data, use_ace=False, prefer_delta=False))
            except Exception:
                pass
            if candidates:
                candidates.sort(key=len)
                frame_bytes = candidates[0]
            else:
                frame_bytes = compress_numeric_array_fast(col_data, use_ace=False, prefer_delta=False)
        else:
            use_ace, prefer_delta = _choose_best_compression(col_data, None, True)
            frame_bytes = compress_numeric_array_fast(col_data, use_ace, prefer_delta=prefer_delta)
        
        return frame_bytes, None
    
    def _encode_conditional_mlp(
        self,
        child_col: np.ndarray,
        parent_col: np.ndarray,
        col_idx: int,
    ) -> Tuple[bytes, Optional[bytes]]:
        """Encode a child column conditionally on parent using MLP predictions."""
        from .conditional import _encode_varint
        
        # Filter valid pairs
        valid_mask = (child_col != -1) & (parent_col != -1)
        if np.sum(valid_mask) == 0:
            return self._encode_independent(child_col)
        
        child_filtered = child_col[valid_mask]
        parent_filtered = parent_col[valid_mask]
        
        # Get unique values
        unique_children = np.unique(child_filtered)
        unique_parents = np.unique(parent_filtered)
        
        # Train MLP to predict child from parent
        model = train_tiny_mlp(
            child_indices=child_filtered,
            parent_indices=parent_filtered,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            max_samples=self.max_samples,
            all_unique_children=unique_children,
            all_unique_parents=unique_parents,
        )
        
        if model is None:
            # MLP training failed: fallback to histogram-based encoding
            return self._encode_conditional_histogram(child_col, parent_col)
        
        # Serialize model
        model_bytes = serialize_mlp_params(model)
        
        # Use MLP predictions to encode more efficiently
        # Strategy: For each parent value, use MLP to predict probabilities,
        # then encode child values using probability-weighted ordering
        
        frame_parts = []
        frame_parts.append(b"MLP\x00")  # MLP conditional marker
        frame_parts.append(struct.pack("<I", len(unique_parents)))
        
        # Group child values by parent value
        parent_to_children = {}
        for j in range(len(child_col)):
            if valid_mask[j]:
                p_val = int(parent_col[j])
                c_val = int(child_col[j])
                if p_val not in parent_to_children:
                    parent_to_children[p_val] = []
                parent_to_children[p_val].append(c_val)
        
        # Encode each parent bucket using MLP-informed encoding
        for p_val in sorted(unique_parents):
            frame_parts.append(_encode_varint(int(p_val), signed=True))
            
            child_vals = np.array(parent_to_children.get(p_val, []), dtype=np.int32)
            
            if len(child_vals) == 0:
                frame_parts.append(struct.pack("<I", 0))
                continue
            
            # Use MLP to predict probabilities for this parent
            # Map parent value to model's vocabulary using parent_map
            if hasattr(model, 'parent_map') and int(p_val) in model.parent_map:
                parent_mapped_idx = model.parent_map[int(p_val)]
            else:
                # Parent value not in model vocab: use fallback
                parent_mapped_idx = 0
            
            parent_tensor = torch.tensor([parent_mapped_idx], dtype=torch.long)
            model.eval()
            with torch.no_grad():
                logits = model.forward([parent_tensor])
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Map child values to model's vocabulary indices
            child_mapped = []
            for c_val in child_vals:
                if int(c_val) in model.child_map:
                    child_mapped.append(model.child_map[int(c_val)])
                else:
                    # Value not in model vocab: use 0 as fallback
                    # This shouldn't happen if model was trained correctly, but handle gracefully
                    child_mapped.append(0)
            
            child_mapped = np.array(child_mapped, dtype=np.int32)
            
            # Use probability-weighted encoding: order by MLP probability
            # Values with higher probability get smaller indices (better compression)
            prob_order = np.argsort(-probs)  # Descending order: prob_order[i] = vocab_idx with rank i
            # Create mapping: vocab_idx -> reordered_idx (where reordered_idx is the rank by probability)
            vocab_to_reordered = {int(prob_order[i]): i for i in range(len(prob_order))}
            
            # Reorder child values by probability (higher prob = smaller reordered index)
            # If a vocab_idx is not in vocab_to_reordered (shouldn't happen), use len(prob_order) as fallback
            child_reordered = np.array([
                vocab_to_reordered.get(int(c), len(prob_order)) for c in child_mapped
            ], dtype=np.int32)
            
            # Encode using the reordered values (more frequent = smaller = better compression)
            child_frame = compress_numeric_array_fast(child_reordered, use_ace=True, prefer_delta=False)
            
            frame_parts.append(struct.pack("<I", len(child_frame)))
            frame_parts.append(child_frame)
        
        frame_bytes = b"".join(frame_parts)
        return frame_bytes, model_bytes
    
    def _encode_conditional_histogram(
        self,
        child_col: np.ndarray,
        parent_col: np.ndarray,
    ) -> Tuple[bytes, Optional[bytes]]:
        """Fallback: encode using histogram-based conditional encoding."""
        from .conditional import _encode_varint
        
        valid_mask = (child_col != -1) & (parent_col != -1)
        if np.sum(valid_mask) == 0:
            return self._encode_independent(child_col)
        
        # Group child values by parent value
        parent_to_children = {}
        for j in range(len(child_col)):
            if valid_mask[j]:
                p_val = int(parent_col[j])
                c_val = int(child_col[j])
                if p_val not in parent_to_children:
                    parent_to_children[p_val] = []
                parent_to_children[p_val].append(c_val)
        
        # Encode conditional distribution
        frame_parts = []
        frame_parts.append(b"CND\x00")  # Conditional marker
        frame_parts.append(struct.pack("<I", len(parent_to_children)))
        
        for p_val in sorted(parent_to_children.keys()):
            frame_parts.append(_encode_varint(int(p_val), signed=True))
            child_arr = np.array(parent_to_children[p_val], dtype=np.int32)
            child_frame = compress_numeric_array_fast(child_arr, use_ace=True, prefer_delta=False)
            frame_parts.append(struct.pack("<I", len(child_frame)))
            frame_parts.append(child_frame)
        
        frame_bytes = b"".join(frame_parts)
        return frame_bytes, None


def encode_columns_neural(
    indices: List[np.ndarray],
    parents: List[int],
    dicts: List[Any],
    embedding_dim: int = 8,
    hidden_dims: List[int] = [32, 32],
    num_epochs: int = 20,
) -> Tuple[List[bytes], List[Optional[bytes]], Dict[str, Any]]:
    """
    Encode columns using neural MLP-based compression.
    
    Args:
        indices: List of column arrays [n_rows] each
        parents: Parent array (from forest structure)
        dicts: List of dictionaries (for compatibility)
        embedding_dim: Embedding dimension
        hidden_dims: Hidden layer dimensions
        num_epochs: Training epochs per column
    
    Returns:
        (frames, models, metadata)
    """
    compressor = NeuralForestCompressor(
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        num_epochs=num_epochs,
    )
    return compressor.compress_columns(indices, parents, dicts)


def decode_columns_neural(
    frames: List[bytes],
    models: List[Optional[bytes]],
    metadata: Dict[str, Any],
    n_rows: int,
) -> List[np.ndarray]:
    """
    Decode columns from neural MLP-based compression.
    
    Args:
        frames: List of compressed column data
        models: List of serialized model bytes
        metadata: Metadata dict with parent structure
        n_rows: Number of rows
    
    Returns:
        List of decoded column arrays
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural decompression")
    
    parents = metadata.get("parents", [-1] * len(frames))
    n_cols = len(frames)
    
    decoded_cols = [None] * n_cols
    
    # Decode roots first, then children (for dependency order)
    root_ids = [j for j, p in enumerate(parents) if p == -1]
    child_ids = [j for j, p in enumerate(parents) if p != -1]
    
    # Decode roots
    for col_idx in root_ids:
        if col_idx < len(frames):
            frame = frames[col_idx]
            decoded_cols[col_idx] = _decode_independent(frame, n_rows)
    
    # Decode children
    for col_idx in child_ids:
        parent_idx = parents[col_idx]
        if col_idx < len(frames):
            frame = frames[col_idx]
            model_bytes = models[col_idx] if col_idx < len(models) else None
            
            if decoded_cols[parent_idx] is None:
                # Parent not decoded yet: should not happen, but handle gracefully
                decoded_cols[col_idx] = _decode_independent(frame, n_rows)
            else:
                parent_col = decoded_cols[parent_idx]
                decoded_cols[col_idx] = _decode_conditional_mlp(
                    frame, model_bytes, parent_col, n_rows
                )
    
    return decoded_cols


def _decode_independent(frame: bytes, n_rows: int) -> np.ndarray:
    """Decode an independently encoded column (root)."""
    # Root frames are encoded with compress_numeric_array_fast
    return decompress_numeric_array(frame).astype(np.int32)


def _decode_conditional_mlp(
    frame: bytes,
    model_bytes: Optional[bytes],
    parent_col: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Decode a conditionally encoded column using MLP."""
    frame_decompressed = frame
    ptr = 0
    
    # Check encoding type
    if len(frame_decompressed) >= ptr + 4 and frame_decompressed[ptr:ptr+4] == b"MLP\x00":
        # MLP-based conditional encoding
        ptr += 4
        n_buckets = struct.unpack("<I", frame_decompressed[ptr:ptr+4])[0]
        ptr += 4
        
        # Deserialize model if available
        model = None
        if model_bytes:
            try:
                model = deserialize_mlp_params(model_bytes)
            except Exception:
                pass
        
        if model is None:
            # No model: fallback to histogram decoding
            return _decode_conditional_histogram(frame_decompressed, parent_col, n_rows)
        
        decoded_data = np.full(n_rows, -1, dtype=np.int32)
        
        for _ in range(n_buckets):
            p_val, consumed = _decode_varint(frame_decompressed, ptr, signed=True)
            ptr += consumed
            child_frame_len = struct.unpack("<I", frame_decompressed[ptr:ptr+4])[0]
            ptr += 4
            
            if child_frame_len > 0:
                child_frame = frame_decompressed[ptr:ptr+child_frame_len]
                ptr += child_frame_len
                
                # Decompress child values (these are probability-ordered indices)
                child_reordered = decompress_numeric_array(child_frame)
                
                # Get MLP probabilities for this parent to reverse the ordering
                # Map parent value to model's vocabulary using parent_map
                if hasattr(model, 'parent_map') and int(p_val) in model.parent_map:
                    parent_mapped_idx = model.parent_map[int(p_val)]
                else:
                    # Parent value not in model vocab: use fallback
                    parent_mapped_idx = 0
                
                parent_tensor = torch.tensor([parent_mapped_idx], dtype=torch.long)
                model.eval()
                with torch.no_grad():
                    logits = model.forward([parent_tensor])
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                
                # Reverse probability ordering: map reordered indices back to model vocab indices
                # During encoding: prob_order[i] = model_vocab_idx for probability rank i
                # So prob_order[0] = most likely model_vocab_idx, prob_order[1] = second most likely, etc.
                # During decoding: reordered_idx -> model_vocab_idx
                prob_order = np.argsort(-probs)  # Descending order (same as encoding)
                # Create reverse mapping: reordered_idx -> vocab_idx
                reordered_to_vocab = {i: int(prob_order[i]) for i in range(len(prob_order))}
                
                # Map reordered indices back to model vocabulary indices
                child_mapped = np.array([
                    reordered_to_vocab.get(int(idx), 0) if int(idx) < len(prob_order) else 0
                    for idx in child_reordered
                ], dtype=np.int32)
                
                # Map model vocabulary indices back to original values
                if hasattr(model, 'child_map'):
                    # Use inverse_child_map if available, otherwise create it
                    if hasattr(model, 'inverse_child_map'):
                        idx_to_val = model.inverse_child_map
                    else:
                        # Create reverse mapping from child_map
                        idx_to_val = {mapped_idx: orig_val for orig_val, mapped_idx in model.child_map.items()}
                    
                    child_vals = np.array([
                        idx_to_val.get(int(idx), 0) if int(idx) in idx_to_val else 0
                        for idx in child_mapped
                    ], dtype=np.int32)
                else:
                    # No mapping: assume direct
                    child_vals = child_mapped.astype(np.int32)
                
                # Fill in values where parent matches
                child_idx = 0
                for row_idx in range(n_rows):
                    if parent_col[row_idx] == p_val and child_idx < len(child_vals):
                        decoded_data[row_idx] = int(child_vals[child_idx])
                        child_idx += 1
        
        return decoded_data
    
    else:
        # Fallback to histogram decoding
        return _decode_conditional_histogram(frame_decompressed, parent_col, n_rows)


def _decode_conditional_histogram(
    frame_decompressed: bytes,
    parent_col: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Decode using histogram-based conditional encoding (fallback)."""
    ptr = 0
    
    if len(frame_decompressed) >= ptr + 4 and frame_decompressed[ptr:ptr+4] == b"CND\x00":
        ptr += 4
        n_buckets = struct.unpack("<I", frame_decompressed[ptr:ptr+4])[0]
        ptr += 4
        
        decoded_data = np.full(n_rows, -1, dtype=np.int32)
        
        for _ in range(n_buckets):
            p_val, consumed = _decode_varint(frame_decompressed, ptr, signed=True)
            ptr += consumed
            child_frame_len = struct.unpack("<I", frame_decompressed[ptr:ptr+4])[0]
            ptr += 4
            
            if child_frame_len > 0:
                child_frame = frame_decompressed[ptr:ptr+child_frame_len]
                ptr += child_frame_len
                
                child_vals = decompress_numeric_array(child_frame)
                
                # Fill in values where parent matches
                child_idx = 0
                for row_idx in range(n_rows):
                    if parent_col[row_idx] == p_val and child_idx < len(child_vals):
                        decoded_data[row_idx] = int(child_vals[child_idx])
                        child_idx += 1
        
        return decoded_data
    else:
        # Fallback to independent decoding
        return _decode_independent(frame_decompressed, n_rows)
