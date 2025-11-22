import argparse
import json
import sys
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .forest import build_mdl_weighted_forest
from .codec import mdl_cost_fn_openzl, mdl_cost_fn_fast, generic_bytes_compress, generic_bytes_decompress
from .conditional import orient_forest, encode_columns_with_parents, decode_columns_with_parents, _encode_varint, _decode_varint


@dataclass
class Model:
	columns: List[str]
	edges: List[List[Any]]  # [u, v, weight]
	dicts: List[Any]  # per-column tokens list or None for numeric columns
	parents: List[int]
	rare_meta: List[str]
	delimiter: str = ","  # CSV delimiter (default comma for backward compatibility)
	line_ending: str = "\n"  # Line ending style: "\n" (LF) or "\r\n" (CRLF)

	def to_bytes(self) -> bytes:
		# Optimize dictionary storage: store dictionaries as compact binary format
		# This is more efficient than JSON arrays of strings
		import struct
		
		# Encode dictionaries more efficiently
		dicts_encoded = []
		for d in self.dicts:
			if isinstance(d, tuple):
				# Special numeric format - store as tuple marker + data
				# Check if it's a timestamp format
				if len(d) == 2 and d[0] == "_timestamp":
					# Timestamp format: ("_timestamp", sentinel_value)
					dicts_encoded.append(("_timestamp", d[1]))
				else:
					# Other special formats (e.g., ("_scale", 10))
					dicts_encoded.append(("_tuple", list(d)))
			elif d is None:
				# Numeric column - store as None marker
				dicts_encoded.append(("_none", None))
			else:
				# Categorical column - store dictionary as binary, compress, then base64 for JSON
				# Format: [n:u32][for each: len:u16, string_bytes...]
				parts = []
				parts.append(struct.pack("<I", len(d)))  # 4 bytes for count
				for val in d:
					val_bytes = str(val).encode("utf-8")
					val_len = len(val_bytes)
					if val_len < 65535:
						parts.append(struct.pack("<H", val_len))  # 2 bytes for length
					else:
						parts.append(struct.pack("<H", 65535))  # Flag for long string
						parts.append(struct.pack("<I", val_len))  # 4 bytes for actual length
					parts.append(val_bytes)
				# Compress the binary data before base64 encoding (better compression)
				binary_data = b"".join(parts)
				compressed_dict = generic_bytes_compress(binary_data)
				# Base64 encode for JSON storage
				dicts_encoded.append(("_binary", base64.b64encode(compressed_dict).decode("ascii")))
		
		# Optimize column names: store as binary with length prefixes (more efficient than JSON strings)
		# Column names are often repetitive or have common prefixes
		import struct
		columns_binary = []
		columns_binary.append(struct.pack("<I", len(self.columns)))  # 4 bytes for count
		for col_name in self.columns:
			# Handle both string and integer column names (convert to string)
			col_str = str(col_name) if not isinstance(col_name, str) else col_name
			col_bytes = col_str.encode("utf-8")
			col_len = len(col_bytes)
			if col_len < 255:
				columns_binary.append(struct.pack("<B", col_len))  # 1 byte for length
			else:
				columns_binary.append(struct.pack("<B", 255))  # Flag
				columns_binary.append(struct.pack("<I", col_len))  # 4 bytes for actual length
			columns_binary.append(col_bytes)
		columns_compressed = generic_bytes_compress(b"".join(columns_binary))
		
		# Optimize edges: quantize weights to reduce precision (saves space, minimal impact)
		# Store edges as [u, v, weight_quantized] where weight is stored as int16 (scaled)
		edges_optimized = []
		for u, v, w in self.edges:
			# Quantize weight to int16 range: scale by 1000 and clamp
			weight_scaled = max(-32768, min(32767, int(w * 1000)))
			edges_optimized.append([int(u), int(v), weight_scaled])
		
		# For other fields, use JSON but optimize
		obj = {
			"columns": base64.b64encode(columns_compressed).decode("ascii"),  # Compressed binary
			"edges": edges_optimized,  # Quantized weights
			"dicts": dicts_encoded,  # Now stored as encoded format
			"parents": self.parents,
			"rare_meta": self.rare_meta,
			"delimiter": self.delimiter,
			"line_ending": self.line_ending,  # Preserve line ending style
		}
		json_bytes = json.dumps(obj, separators=(",", ":")).encode("utf-8")
		# Compress the model metadata to reduce overhead
		return generic_bytes_compress(json_bytes)

	@staticmethod
	def from_bytes(b: bytes) -> "Model":
		# Decompress the model metadata (if it was compressed)
		# Use fallback_to_uncompressed=True to handle cases where data is uncompressed
		decompressed_bytes = generic_bytes_decompress(b, fallback_to_uncompressed=True)
		obj = json.loads(decompressed_bytes.decode("utf-8"))
		
		import struct
		# Decode column names from binary format
		columns_data = obj.get("columns", [])
		if isinstance(columns_data, str):
			# New format: base64 encoded compressed binary
			columns_compressed = base64.b64decode(columns_data.encode("ascii"))
			# Use fallback_to_uncompressed=True to handle cases where data is uncompressed
			columns_binary = generic_bytes_decompress(columns_compressed, fallback_to_uncompressed=True)
			ptr = 0
			n_cols = struct.unpack("<I", columns_binary[ptr:ptr+4])[0]
			ptr += 4
			columns = []
			for _ in range(n_cols):
				col_len_byte = columns_binary[ptr]
				ptr += 1
				if col_len_byte == 255:
					col_len = struct.unpack("<I", columns_binary[ptr:ptr+4])[0]
					ptr += 4
				else:
					col_len = col_len_byte
				col_bytes = columns_binary[ptr:ptr+col_len]
				ptr += col_len
				columns.append(col_bytes.decode("utf-8"))
		else:
			# Old format: list of strings
			columns = list(columns_data) if isinstance(columns_data, list) else []
		
		# Decode edges: dequantize weights
		edges_data = obj.get("edges", [])
		edges = []
		for e in edges_data:
			if len(e) == 3:
				u, v, w_scaled = e
				# Dequantize: divide by 1000
				w = float(w_scaled) / 1000.0
				edges.append([u, v, w])
			else:
				# Old format: keep as-is
				edges.append(e)
		
		# Decode dictionaries from binary format
		dicts_deserialized = []
		for d in obj["dicts"]:
			# Handle both old format (list) and new format (list with type marker from JSON)
			# JSON converts tuples to lists, so ["_binary", "base64string"] is the new format
			if isinstance(d, list):
				if len(d) == 2 and isinstance(d[0], str) and d[0] in ["_tuple", "_none", "_binary", "_timestamp"]:
					# New format: encoded dictionary with type marker
					format_type, data = d
					if format_type == "_tuple":
						# Special numeric format (e.g., ["_scale", 10])
						dicts_deserialized.append(tuple(data))
					elif format_type == "_none":
						# Numeric column
						dicts_deserialized.append(None)
					elif format_type == "_timestamp":
						# Timestamp column: tuple ("_timestamp", sentinel_value)
						dicts_deserialized.append(("_timestamp", data))
					elif format_type == "_binary":
						# Decode binary dictionary format (base64 decoded, then decompressed)
						if isinstance(data, str):
							# New format: base64 encoded, then compressed
							compressed_data = base64.b64decode(data.encode("ascii"))
							# Use fallback_to_uncompressed=True to handle cases where data is uncompressed
							binary_data = generic_bytes_decompress(compressed_data, fallback_to_uncompressed=True)
						else:
							# Fallback: direct binary (shouldn't happen in JSON)
							binary_data = data
						ptr = 0
						n = struct.unpack("<I", binary_data[ptr:ptr+4])[0]
						ptr += 4
						vocab = []
						for _ in range(n):
							val_len = struct.unpack("<H", binary_data[ptr:ptr+2])[0]
							ptr += 2
							if val_len == 65535:  # Long string flag
								val_len = struct.unpack("<I", binary_data[ptr:ptr+4])[0]
								ptr += 4
							val_bytes = binary_data[ptr:ptr+val_len]
							ptr += val_len
							vocab.append(val_bytes.decode("utf-8"))
						dicts_deserialized.append(vocab)
					else:
						dicts_deserialized.append(d)
				elif len(d) == 2 and isinstance(d[0], str) and d[0].startswith("_") and d[0] not in ["_tuple", "_none", "_binary"]:
					# Old format: special numeric format tuple (e.g., ["_scale", 10])
					# This handles the old format where we stored tuples directly
					dicts_deserialized.append(tuple(d))
				else:
					# Old format: list of strings (categorical dictionary)
					dicts_deserialized.append(d)
			elif d is None:
				# Numeric column (None)
				dicts_deserialized.append(None)
			else:
				# Fallback: treat as-is (shouldn't happen)
				dicts_deserialized.append(d)
		
		return Model(
			columns=columns,
			edges=edges,
			dicts=dicts_deserialized,
			parents=list(obj["parents"]),
			rare_meta=list(obj.get("rare_meta", [""] * len(columns))),
			delimiter=obj.get("delimiter", ","),  # Default to comma for backward compatibility
			line_ending=obj.get("line_ending", "\n"),  # Default to LF for backward compatibility
		)


def _load_csv(path: Path, delimiter: str) -> pd.DataFrame:
	# Handle both tab-separated and comma-separated
	# Convert string representations of special characters
	# When passed from command line, "\t" comes as "\\t" (literal backslash-t)
	if delimiter == '\\t' or delimiter == '\t':
		sep = '\t'
	elif delimiter == '\\n':
		sep = '\n'
	else:
		sep = delimiter
	# Read with no header, keeping all values as strings to preserve exact format
	# Use dtype=str to prevent type inference that could change empty strings to NaN
	# Use engine='python' for tab delimiters to avoid regex warnings and handle malformed lines
	import warnings
	warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
	# Use on_bad_lines='skip' to handle lines with inconsistent field counts
	df = pd.read_csv(path, delimiter=sep, header=None, skipinitialspace=False, dtype=str, 
	                 keep_default_na=False, na_values=[], engine='python', on_bad_lines='skip')
	# Replace empty strings with None for processing, but we'll track original format
	# Empty strings become None for easier processing, but we preserve the fact they were empty
	return df


def _df_to_table(df: pd.DataFrame) -> np.ndarray:
	return df.astype(object).to_numpy()


def _encode_rare_overrides(overrides: List[Tuple[int, Any]]) -> bytes:
	"""
	Encode rare value overrides in a compact binary format.
	Format: [n:u32][sorted by row_idx, then for each: delta_row:u32, value_bytes...]
	Uses delta encoding for row indices and variable-length encoding for values.
	"""
	if not overrides:
		return b""
	# Sort by row index for delta encoding
	sorted_overrides = sorted(overrides, key=lambda x: x[0])
	parts = []
	parts.append(len(sorted_overrides).to_bytes(4, "little", signed=False))
	prev_row = 0
	for row_idx, value in sorted_overrides:
		# Delta encode row index
		delta = row_idx - prev_row
		prev_row = row_idx
		# Use variable-length encoding for delta (small deltas use fewer bytes)
		# Format: <128: single byte, 128-16383: two bytes (0x80-0xBF high byte), >=16384: 0xC0 flag + 4 bytes
		if delta < 128:
			parts.append(bytes([delta]))  # Single byte for small deltas
		elif delta < 16384:
			parts.append(bytes([0x80 | (delta >> 8), delta & 0xFF]))  # Two bytes: 0x80-0xBF + low byte
		else:
			parts.append(bytes([0xC0]) + delta.to_bytes(4, "little", signed=False))  # Flag + 4 bytes
		# Encode value as UTF-8 bytes with length prefix
		value_bytes = str(value).encode("utf-8")
		value_len = len(value_bytes)
		if value_len < 254:
			parts.append(bytes([value_len]))  # Single byte length (0-253)
		else:
			parts.append(bytes([0xFE]) + value_len.to_bytes(4, "little", signed=False))  # Flag + 4 bytes
		parts.append(value_bytes)
	return b"".join(parts)


def _decode_rare_overrides(data: bytes) -> List[Tuple[int, Any]]:
	"""
	Decode rare value overrides from compact binary format.
	"""
	if not data:
		return []
	ptr = 0
	n = int.from_bytes(data[ptr:ptr+4], "little", signed=False)
	ptr += 4
	overrides = []
	prev_row = 0
	for _ in range(n):
		# Decode delta-encoded row index
		first_byte = data[ptr]
		ptr += 1
		if first_byte < 0x80:
			delta = first_byte
		elif first_byte < 0xC0:
			second_byte = data[ptr]
			ptr += 1
			delta = ((first_byte & 0x3F) << 8) | second_byte
		else:  # 0xC0 flag
			delta = int.from_bytes(data[ptr:ptr+4], "little", signed=False)
			ptr += 4
		row_idx = prev_row + delta
		prev_row = row_idx
		# Decode value length
		len_byte = data[ptr]
		ptr += 1
		if len_byte < 254:
			value_len = len_byte
		else:  # 0xFE flag
			value_len = int.from_bytes(data[ptr:ptr+4], "little", signed=False)
			ptr += 4
		# Decode value
		value_bytes = data[ptr:ptr+value_len]
		ptr += value_len
		value = value_bytes.decode("utf-8")
		overrides.append((row_idx, value))
	return overrides


def _tokenize_columns(df: pd.DataFrame, rare_threshold: int) -> Tuple[List[np.ndarray], List[Any], List[bytes], List[bool]]:
	"""
	Tokenize columns and return indices, dictionaries, rare blobs, and is_numeric flags.
	Returns is_numeric list to track which columns should be output as numbers.
	"""
	indices_list: List[np.ndarray] = []
	dicts: List[Any] = []
	rare_blobs: List[bytes] = []
	is_numeric_list: List[bool] = []  # Track which columns are numeric
	MISSING_LITERAL = "nan"
	EMPTY_STRING = ""  # Empty string marker
	
	for col in df.columns:
		series = df[col].copy()
		# Convert empty strings to a marker for processing
		series = series.replace("", EMPTY_STRING)
		
		# Try to detect if this column should be numeric (int or float)
		# Check if all non-empty values can be converted to numbers
		can_be_numeric_int = True
		can_be_numeric_float = True
		has_empty = False
		numeric_int_values = []
		numeric_float_values = []
		is_float_column = False
		
		# Quick sample check first (for speed on large datasets)
		sample_size = min(1000, len(series))
		sample = series.sample(n=sample_size, random_state=42) if len(series) > sample_size else series
		
		for val in sample:
			val_str = str(val).strip() if val is not None else ""
			if val_str == EMPTY_STRING or val_str == "" or pd.isna(val):
				has_empty = True
				continue
			try:
				# Try to convert to int first
				numeric_int_values.append(int(float(val_str)))  # Convert via float to handle "1.0" -> 1
				numeric_float_values.append(float(val_str))
			except (ValueError, TypeError, OverflowError):
				can_be_numeric_int = False
				can_be_numeric_float = False
				break
		
		# If sample passes, check full column (but optimize for large datasets)
		if (can_be_numeric_int or can_be_numeric_float) and not has_empty:
			numeric_int_values = []
			numeric_float_values = []
			numeric_original_strings = []  # Store original string representations for format preservation
			has_decimal = False
			
			# For large datasets, use vectorized conversion for speed
			if len(series) > 10000:
				try:
					# Try to convert entire series at once
					series_float = pd.to_numeric(series, errors='coerce')
					if series_float.isna().sum() == 0:
						# All values are numeric
						numeric_float_values = series_float.tolist()
						# Store original strings for format preservation
						numeric_original_strings = [str(val).strip() if val is not None else "" for val in series]
						# Check if any are non-integer
						has_decimal = any(f != int(f) for f in numeric_float_values if not pd.isna(f))
						numeric_int_values = [int(f) for f in numeric_float_values]
					else:
						can_be_numeric_int = False
						can_be_numeric_float = False
				except (ValueError, TypeError, OverflowError):
					can_be_numeric_int = False
					can_be_numeric_float = False
			else:
				# For smaller datasets, check each value
				for val in series:
					val_str = str(val).strip() if val is not None else ""
					numeric_original_strings.append(val_str)  # Store original string
					try:
						f_val = float(val_str)
						numeric_float_values.append(f_val)
						if f_val != int(f_val):
							has_decimal = True
						numeric_int_values.append(int(f_val))
					except (ValueError, TypeError, OverflowError):
						can_be_numeric_int = False
						can_be_numeric_float = False
						break
		
		# Store as numeric if possible
		# Prefer integers (more efficient), but handle floats if needed
		# Only store as pure integers if there are NO decimal values
		if can_be_numeric_int and not has_empty and not has_decimal and len(numeric_int_values) == len(series):
			try:
				# Check if we can store as int64
				arr = np.array(numeric_int_values, dtype=np.int64)
				# Verify no overflow
				if np.all(arr == np.array(numeric_int_values, dtype=object)):
					indices_list.append(arr)
					dicts.append(None)
					rare_blobs.append(b"")
					is_numeric_list.append(True)
					continue
			except (ValueError, OverflowError):
				pass
		
		# Try float storage (quantize to fixed precision for compression)
		if can_be_numeric_float and not has_empty and len(numeric_float_values) == len(series):
			try:
				# For floats, we can quantize to a fixed number of decimal places
				# or scale to integers (multiply by a factor and store as int)
				# Strategy: find common denominator (e.g., if all are multiples of 0.1, scale by 10)
				# For now, store as floats but we'll compress them efficiently
				arr_float = np.array(numeric_float_values, dtype=np.float64)
				
				# Try to find a scaling factor that makes everything an integer
				# Check common scales: powers of 10, and also detect common denominators
				best_scale = 1
				best_int_array = None
				
				# First, try to detect the common decimal precision
				# Check if all values have the same number of decimal places
				decimal_places = []
				for val in numeric_float_values[:min(1000, len(numeric_float_values))]:  # Sample for speed
					# Handle both string representation and float directly
					if isinstance(val, (int, float)):
						# Check if it's effectively an integer
						if abs(val - round(val)) < 1e-10:
							decimal_places.append(0)
						else:
							# Count decimal places by converting to string (avoid scientific notation)
							val_str = f"{val:.15f}".rstrip('0').rstrip('.')
							if '.' in val_str:
								decimal_places.append(len(val_str.split('.')[1]))
							else:
								decimal_places.append(0)
				
				# If most values have the same decimal precision, use that scale
				# Also check for common patterns like .0, .5 (scale by 2), .25, .75 (scale by 4)
				if decimal_places:
					from collections import Counter
					most_common_decimals = Counter(decimal_places).most_common(1)[0]
					if most_common_decimals[1] > len(decimal_places) * 0.7:  # >70% have same precision
						suggested_scale = 10 ** most_common_decimals[0]
						# Remove duplicates and sort
						scales_to_try = list(dict.fromkeys([suggested_scale, 1, 10, 100, 1000, 10000]))
						scales_to_try.sort()
					else:
						# Check if values are multiples of 0.5 (common in currency/tax data)
						# Sample to check if scaling by 2 works - use larger sample for better detection
						sample_vals = numeric_float_values[:min(5000, len(numeric_float_values))]
						scaled_by_2 = [v * 2 for v in sample_vals]
						# Check if most values (>=90%) are multiples of 0.5
						multiples_of_0_5 = sum(1 for v in scaled_by_2 if abs(v - round(v)) < 1e-9)
						if multiples_of_0_5 >= len(sample_vals) * 0.9:
							# Most values are multiples of 0.5 - try scale 2 first
							scales_to_try = [2, 1, 10, 100, 1000, 10000, 100000]
						# Check if values are multiples of 0.25
						elif sum(1 for v in sample_vals if abs(v * 4 - round(v * 4)) < 1e-9) >= len(sample_vals) * 0.9:
							scales_to_try = [4, 2, 1, 10, 100, 1000, 10000, 100000]
						# Check if values are multiples of 0.1 (common in currency)
						elif sum(1 for v in sample_vals if abs(v * 10 - round(v * 10)) < 1e-9) >= len(sample_vals) * 0.9:
							scales_to_try = [10, 1, 2, 4, 100, 1000, 10000, 100000]
						else:
							scales_to_try = [1, 2, 4, 10, 100, 1000, 10000, 100000]
				else:
					scales_to_try = [1, 2, 4, 10, 100, 1000, 10000]
				
				for scale in scales_to_try:
					scaled = (arr_float * scale).round()
					# Check if scaling preserves accuracy (with tolerance for floating point errors)
					# Be more lenient: check if most values (>=99.5%) are preserved accurately
					close_mask = np.isclose(arr_float, scaled / scale, rtol=1e-9, atol=1e-10)
					if np.sum(close_mask) >= len(arr_float) * 0.995:
						# This scale works for most values - all values can be represented as integers
						try:
							scaled_int = scaled.astype(np.int64)
							# Verify no overflow and that conversion is exact for the close values
							# Allow small errors for values that weren't close (edge cases)
							if np.all(scaled_int == scaled) and np.sum(np.isclose(arr_float, scaled_int / scale, rtol=1e-9, atol=1e-10)) >= len(arr_float) * 0.995:
								best_scale = scale
								best_int_array = scaled_int
								break
						except (ValueError, OverflowError):
							continue
				
				if best_int_array is not None:
					# Store scaled integers with scale factor in metadata
					# Scale can be 1 (pure integers) or >1 (scaled floats)
					indices_list.append(best_int_array)
					# Store scale info: use a tuple ("_scale", scale) to indicate scaled float
					dicts.append(("_scale", best_scale))
					# Store original string formats in rare_meta for roundtrip accuracy
					# This preserves formats like "99.0" vs "99"
					if numeric_original_strings and len(numeric_original_strings) == len(series):
						# Store format map for ALL values to preserve exact string format
						# This handles cases like "99.0" vs "99", "99.10" vs "99.1", etc.
						# Optimize storage using delta encoding and varint for indices
						import struct
						format_entries = []
						for idx, (orig_str, f_val) in enumerate(zip(numeric_original_strings, numeric_float_values)):
							# Check if the reconstructed format would differ from original
							# For scale=1: check if orig_str ends with .0 but reconstructed would be int
							# For scale!=1: check if orig_str has trailing zeros that would be lost
							if best_scale == 1:
								# For integers, only store if original had .0
								if f_val == int(f_val) and orig_str.endswith('.0'):
									format_entries.append((idx, orig_str))
							else:
								# For scaled floats, check if format would differ
								reconstructed_str = str(f_val)
								if orig_str != reconstructed_str:
									# Store original format if it differs
									format_entries.append((idx, orig_str))
						
						if format_entries:
							# Optimize storage: use delta encoding for indices and varint encoding
							# Sort by index for better delta encoding
							format_entries.sort(key=lambda x: x[0])
							
							format_parts = []
							format_count = len(format_entries)
							format_parts.append(struct.pack("<I", format_count))
							
							prev_idx = -1
							for idx, orig_str in format_entries:
								# Delta encode index
								delta = idx - prev_idx
								prev_idx = idx
								# Use varint encoding for delta (more efficient for small deltas)
								format_parts.append(_encode_varint(delta, signed=False))
								
								# Store string with length prefix (use varint for length too)
								orig_bytes = orig_str.encode("utf-8")
								format_parts.append(_encode_varint(len(orig_bytes), signed=False))
								format_parts.append(orig_bytes)
							
							format_data = b"".join(format_parts)
							# Always compress format preservation data (it compresses well due to repetition)
							rare_blobs.append(generic_bytes_compress(format_data))
						else:
							rare_blobs.append(b"")
					else:
						rare_blobs.append(b"")
					is_numeric_list.append(True)
					continue
				else:
					# Can't scale to integers without precision loss
					# Try storing as float64 bit pattern (lossless but less compressible)
					# This is a fallback for values that can't be scaled
					arr_int64 = arr_float.view(np.int64)
					indices_list.append(arr_int64)
					dicts.append(("_float", None))
					rare_blobs.append(b"")
					is_numeric_list.append(True)
					continue
			except (ValueError, OverflowError, TypeError):
				pass
		
		# Check if this column contains timestamps/datetimes
		# Timestamps compress much better as numeric (Unix timestamps) than as strings
		is_timestamp = False
		timestamp_values = None
		if len(series) > 0:
			# Sample to check for timestamp patterns
			sample_size = min(1000, len(series))
			sample = series.head(sample_size).dropna()
			if len(sample) > sample_size * 0.8:  # Most values are non-empty
				# Check for common timestamp patterns
				from datetime import datetime
				timestamp_patterns = [
					"%Y-%m-%d %H:%M:%S",  # 2025-01-01 00:18:38
					"%Y-%m-%d %H:%M:%S.%f",  # With microseconds
					"%Y/%m/%d %H:%M:%S",
					"%Y-%m-%dT%H:%M:%S",
					"%Y-%m-%dT%H:%M:%S.%f",
					"%Y-%m-%dT%H:%M:%S%z",  # With timezone
				]
				
				# Try to parse as timestamp
				parsed_count = 0
				for val in sample:
					if isinstance(val, str) and len(val) > 10:  # Timestamps are usually >10 chars
						for pattern in timestamp_patterns:
							try:
								datetime.strptime(val, pattern)
								parsed_count += 1
								break
							except (ValueError, TypeError):
								continue
				
				# If >70% of sample can be parsed as timestamps, treat as timestamp column
				if parsed_count > len(sample) * 0.7:
					is_timestamp = True
					try:
						# Convert entire column to Unix timestamps (seconds since epoch)
						from datetime import datetime
						timestamp_values = []
						original_strings = []  # Store original strings for values that can't be parsed
						unparseable_indices = []  # Track which indices couldn't be parsed
						
						for idx, val in enumerate(series):
							if pd.isna(val) or val == '':
								timestamp_values.append(None)
								original_strings.append("")
								unparseable_indices.append(idx)
							else:
								val_str = str(val)
								parsed = None
								for pattern in timestamp_patterns:
									try:
										parsed = datetime.strptime(val_str, pattern)
										break
									except (ValueError, TypeError):
										continue
								if parsed is not None:
									# Convert to Unix timestamp (seconds since epoch)
									timestamp_values.append(int(parsed.timestamp()))
									original_strings.append(None)  # Mark as successfully parsed
								else:
									# Can't parse - preserve original string
									timestamp_values.append(None)
									original_strings.append(val_str)
									unparseable_indices.append(idx)
						
						# Check if conversion was successful for most values
						non_none_count = sum(1 for v in timestamp_values if v is not None)
						if non_none_count > len(series) * 0.7:
							# Convert to numpy array, handling None values
							# For None values, use a sentinel (e.g., -1 or min-1)
							valid_timestamps = [v for v in timestamp_values if v is not None]
							if valid_timestamps:
								min_ts = min(valid_timestamps)
								# Use min-1 as sentinel for None, but make sure it doesn't conflict
								# Check if min_ts - 1 could be a valid timestamp (unlikely but possible)
								sentinel = min_ts - 1
								# Double-check sentinel doesn't conflict (shouldn't happen in practice)
								while sentinel in valid_timestamps:
									sentinel -= 1
								
								timestamp_array = np.array([v if v is not None else sentinel for v in timestamp_values], dtype=np.int64)
								
								# Store unparseable original strings in rare_blobs
								# Format: [count:u32][for each: index:u32, len:u16, string_bytes...]
								import struct
								rare_parts = []
								unparseable_count = len(unparseable_indices)
								rare_parts.append(struct.pack("<I", unparseable_count))
								for idx in unparseable_indices:
									orig_str = original_strings[idx] if idx < len(original_strings) else ""
									if orig_str:
										orig_bytes = orig_str.encode("utf-8")
										rare_parts.append(struct.pack("<I", idx))  # Index
										rare_parts.append(struct.pack("<H", len(orig_bytes)))  # Length
										rare_parts.append(orig_bytes)
									else:
										# Empty string
										rare_parts.append(struct.pack("<I", idx))
										rare_parts.append(struct.pack("<H", 0))
								
								# Store as numeric with sentinel marker and unparseable strings
								# Compress the unparseable strings data (will be decompressed during decode)
								unparseable_data = b"".join(rare_parts)
								indices_list.append(timestamp_array)
								dicts.append(("_timestamp", sentinel))  # Store sentinel value for None
								rare_blobs.append(generic_bytes_compress(unparseable_data) if unparseable_data else b"")
								is_numeric_list.append(True)
								continue
					except (ValueError, OverflowError, TypeError, AttributeError) as e:
						# If conversion fails, fall through to categorical handling
						pass
		
		# Treat as categorical/string column
		is_numeric_list.append(False)
		series = series.astype(object)
		# Replace None/NaN with empty string for consistent handling
		series = series.fillna(EMPTY_STRING)
		
		vc = series.value_counts(dropna=False)
		# Optimize rare value handling for compression
		# Bucket rare values to reduce vocabulary size, which improves encoding efficiency
		# Adaptive threshold: for large datasets, be more aggressive about bucketing rare values
		total_count = len(series)
		rare_values = set()
		if rare_threshold > 0:
			# Calculate optimal threshold: bucket values that are truly rare
			# Strategy: bucket if frequency is low AND total occurrences are small
			# For large datasets, use more aggressive threshold to reduce vocabulary
			# More aggressive rare value bucketing for better compression
			# For large datasets, be very aggressive to reduce vocabulary size
			if total_count < 100_000:
				adaptive_freq_threshold = 0.01
				max_rare_count = rare_threshold
			elif total_count < 1_000_000:
				adaptive_freq_threshold = 0.002  # Even more aggressive (was 0.003)
				max_rare_count = max(rare_threshold, 4)  # Even more aggressive (was 3)
			else:
				adaptive_freq_threshold = 0.0003  # Extremely aggressive (was 0.0005)
				max_rare_count = max(rare_threshold, 15)  # Extremely aggressive for large datasets (was 10)
			
			# Also consider total vocabulary size - if vocab is very large, be more aggressive
			vocab_size = len(vc)
			if vocab_size > 10000 and total_count > 100_000:
				# For very high-cardinality columns, be aggressive but not too extreme
				max_rare_count = max(max_rare_count, min(50, vocab_size // 200))  # Aggressive
				adaptive_freq_threshold = min(adaptive_freq_threshold, 0.0002)  # Aggressive
			elif vocab_size > 5000 and total_count > 100_000:
				# For high-cardinality columns in large datasets, bucket more aggressively
				max_rare_count = max(max_rare_count, min(30, vocab_size // 300))  # More aggressive
				adaptive_freq_threshold = min(adaptive_freq_threshold, 0.0005)  # More aggressive
			elif vocab_size > 2000 and total_count > 50_000:
				# For medium-high cardinality, also be aggressive
				max_rare_count = max(max_rare_count, min(15, vocab_size // 250))
				adaptive_freq_threshold = min(adaptive_freq_threshold, 0.002)
			elif vocab_size > 1000:  # Even for medium vocab sizes, be more aggressive
				max_rare_count = max(max_rare_count, min(8, vocab_size // 150))
				adaptive_freq_threshold = min(adaptive_freq_threshold, 0.003)
			
			for val, count in vc.items():
				freq_ratio = count / total_count
				# Bucket rare values: low frequency and small overall presence
				# Use adaptive frequency threshold based on dataset size
				# Be more aggressive: also bucket if total occurrences are small relative to dataset
				if count <= max_rare_count and freq_ratio < adaptive_freq_threshold:
					rare_values.add(val)
				# Also bucket if frequency is very low even if count is slightly higher
				# This helps with high-cardinality columns where many values appear a few times
				elif freq_ratio < adaptive_freq_threshold * 0.5 and count <= max_rare_count * 2:
					rare_values.add(val)
				# For very high-cardinality columns, be even more aggressive
				# If vocab is huge (>10k) and value appears only a few times, bucket it
				elif vocab_size > 10000 and count <= 3 and freq_ratio < 0.0001:
					rare_values.add(val)
				# For extremely high-cardinality columns (>50k), be extremely aggressive
				# Bucket any value that appears <= 5 times
				elif vocab_size > 50000 and count <= 5:
					rare_values.add(val)
				# For very high-cardinality columns (20k-50k), also be very aggressive
				elif vocab_size > 20000 and vocab_size <= 50000 and count <= 4:
					rare_values.add(val)
				# For medium-high cardinality (5k-10k), also be more aggressive
				elif vocab_size > 5000 and vocab_size <= 10000 and count <= 4 and freq_ratio < 0.0002:
					rare_values.add(val)
				# For medium cardinality (2k-5k), also bucket more aggressively
				elif vocab_size > 2000 and vocab_size <= 5000 and count <= 3 and freq_ratio < 0.0005:
					rare_values.add(val)
		rare_values.discard(MISSING_LITERAL)
		rare_values.discard(EMPTY_STRING)
		
		codes, uniques = pd.factorize(series, sort=False)
		codes = codes.astype(np.int64, copy=False)
		uni_list = uniques.tolist()
		vocab: List[Any] = []
		value_to_newid: Dict[Any, int] = {}
		empty_id = None
		missing_id = None
		
		for v in uni_list:
			if v == EMPTY_STRING or (isinstance(v, float) and pd.isna(v)):
				if empty_id is None:
					empty_id = len(vocab)
					vocab.append(EMPTY_STRING)
				value_to_newid[EMPTY_STRING] = empty_id
				continue
			if v == MISSING_LITERAL:
				if missing_id is None:
					missing_id = len(vocab)
					vocab.append(MISSING_LITERAL)
				value_to_newid[MISSING_LITERAL] = missing_id
				continue
			if v in rare_values:
				continue
			value_to_newid[v] = len(vocab)
			vocab.append(str(v) if not pd.isna(v) else EMPTY_STRING)
		
		rare_placeholder_id = None
		if rare_values:
			rare_placeholder_id = len(vocab)
			vocab.append("<RARE>")
		
		overrides: List[Tuple[int, Any]] = []
		new_codes = np.empty_like(codes)
		for i, code in enumerate(codes.tolist()):
			if code == -1:
				# Unseen value - treat as empty
				if empty_id is None:
					empty_id = len(vocab)
					vocab.append(EMPTY_STRING)
					value_to_newid[EMPTY_STRING] = empty_id
				new_codes[i] = empty_id
				continue
			v = uni_list[code]
			if v == EMPTY_STRING or (isinstance(v, float) and pd.isna(v)):
				if empty_id is None:
					empty_id = len(vocab)
					vocab.append(EMPTY_STRING)
					value_to_newid[EMPTY_STRING] = empty_id
				new_codes[i] = empty_id
			elif v in rare_values:
				new_codes[i] = rare_placeholder_id  # type: ignore
				overrides.append((i, str(v)))
			else:
				new_codes[i] = value_to_newid[v]
		
		indices_list.append(new_codes)
		dicts.append(vocab)
		# Optimize rare value storage: use more efficient encoding
		if overrides:
			# Store as compact binary format: [n_overrides:u32][for each: delta_row:u32, value_len:u16, value_bytes...]
			# Use delta encoding for row indices to save space
			ov_bytes = _encode_rare_overrides(overrides)
		else:
			ov_bytes = b""
		rare_blobs.append(generic_bytes_compress(ov_bytes) if ov_bytes else b"")
	
	return indices_list, dicts, rare_blobs, is_numeric_list


def _prepare_raw_to_csv(input_path: Path, output_path: Path, header: bool | None = None) -> None:
	na_vals = ["?", "NA", "N/A", "null", "", " "]
	if header is None:
		try:
			df = pd.read_csv(input_path, sep=None, engine="python", na_values=na_vals, skipinitialspace=True)
		except Exception:
			df = pd.read_csv(input_path, sep=None, engine="python", header=None, na_values=na_vals, skipinitialspace=True)
	else:
		df = pd.read_csv(input_path, sep=None, engine="python", header=0 if header else None, na_values=na_vals, skipinitialspace=True)
	for c in df.columns:
		if pd.api.types.is_object_dtype(df[c]):
			df[c] = df[c].astype(str).str.strip()
	for c in df.columns:
		try:
			df[c] = pd.to_numeric(df[c])
		except Exception:
			pass
	output_path.write_text(df.to_csv(index=False))


def compress_file(input_path: str, output_path: str, delimiter: str, rare_threshold: int = 1, mi_mode: str = "exact", mi_buckets: int = 4096, mi_sample: int | None = None, mi_seed: int = 0, workers: int | None = None) -> None:
	import time
	profiler = {}  # Track time spent in each stage
	total_start = time.time()
	
	inp = Path(input_path)
	outp = Path(output_path)
	
	# Detect line ending style from original file
	line_ending = "\n"  # Default to LF
	try:
		with inp.open('rb') as f:
			first_chunk = f.read(8192)  # Read first 8KB to detect line endings
			if b'\r\n' in first_chunk:
				line_ending = "\r\n"  # CRLF (Windows)
			elif b'\n' in first_chunk:
				line_ending = "\n"  # LF (Unix)
	except Exception:
		pass  # Use default if detection fails
	
	stage_start = time.time()
	df = _load_csv(inp, delimiter)
	tab = _df_to_table(df)
	profiler["load_csv"] = time.time() - stage_start
	# Two-phase approach for speed + accuracy:
	# 1. Fast proxy MDL to rank ALL possible edges quickly (not just MST)
	# 2. OpenZL MDL on top candidates, then build MST
	n_rows = len(tab)
	n_cols = len(tab[0])
	
	# Skip high-cardinality columns (they compress better independently)
	stage_start = time.time()
	skip_high_card = set()
	rng = np.random.default_rng(mi_seed)
	card_check_sample = min(500, n_rows) if n_rows > 500 else n_rows
	if card_check_sample < n_rows:
		card_idx = rng.choice(n_rows, size=card_check_sample, replace=False)
		table_sample = tab[card_idx]
	else:
		table_sample = tab
	
	for i in range(n_cols):
		col_data = table_sample[:, i]
		try:
			col_unique = len(np.unique(col_data))
		except (TypeError, ValueError):
			sample_size = min(100, len(col_data))
			if sample_size < len(col_data):
				sample_idx = rng.choice(len(col_data), size=sample_size, replace=False)
				col_unique = len(set(col_data[sample_idx].tolist()))
			else:
				if col_data.dtype == object:
					col_unique = len(set(col_data.tolist()))
				else:
					col_unique = len(set(col_data.flatten().tolist()))
		threshold = int(0.5 * card_check_sample)
		if col_unique > threshold:
			skip_high_card.add(i)
	profiler["cardinality_check"] = time.time() - stage_start
	
	# Phase 1: Fast ranking with proxy MDL on all edge pairs
	# For datasets <= 10k rows, can afford to use exact MI throughout for better accuracy
	use_exact_mi_phase1 = (n_rows <= 10000)
	
	if mi_sample is None:
		# Use larger samples for better accuracy, especially for large datasets
		# Sample size should scale with dataset size but with diminishing returns
		if n_rows > 1_000_000:
			# For very large datasets (>1M rows), use 50k-100k samples for phase 1
			phase1_sample = min(100_000, n_rows // 35)  # ~2.8% of data, up to 100k
		elif n_rows > 100_000:
			# For large datasets (100k-1M rows), use 20k-50k samples
			phase1_sample = min(50_000, n_rows // 20)  # ~5% of data, up to 50k
		elif n_rows > 10_000:
			phase1_sample = min(10_000, n_rows // 2) if use_exact_mi_phase1 else min(5_000, n_rows // 4)
		elif n_rows > 1_000:
			phase1_sample = min(3_000, n_rows // 2) if use_exact_mi_phase1 else min(2_000, n_rows // 3)
		else:
			phase1_sample = None  # Use full data for small datasets
	else:
		phase1_sample = mi_sample
	
	# For small datasets, use exact MI throughout for accuracy
	# For larger datasets, use hashed MI in phase 1 with more buckets, exact in phase 2
	if mi_mode == "auto":
		if n_rows <= 10000:
			phase1_mi_mode = "exact"  # Use exact MI for better accuracy on smaller datasets
		else:
			phase1_mi_mode = "hashed"  # Use hashed for speed, but with more buckets
	elif mi_mode == "exact":
		phase1_mi_mode = "exact"
	else:
		phase1_mi_mode = "hashed"
	
	# Use more buckets for better accuracy in phase 1 if using hashed
	# Larger datasets benefit from more buckets to reduce hash collisions
	if phase1_mi_mode == "hashed":
		if n_rows > 1_000_000:
			phase1_buckets = min(mi_buckets * 2, 16384)  # Up to 16k buckets for very large datasets
		elif n_rows > 100_000:
			phase1_buckets = min(mi_buckets, 8192)  # Up to 8k buckets
		else:
			phase1_buckets = mi_buckets
	else:
		phase1_buckets = mi_buckets
	
	# Fast phase: compute weights for all edge pairs with proxy MDL
	from concurrent.futures import ThreadPoolExecutor
	import os
	from .mi import estimate_edge_weight_hashed, estimate_edge_weight
	
	def compute_fast_weight(pair):
		i, j = pair
		# Skip edges involving high-cardinality columns
		if i in skip_high_card or j in skip_high_card:
			return (i, j, -1.0)  # Negative weight to filter out
		x, y = tab[:, i], tab[:, j]
		mdl_bits = mdl_cost_fn_fast(x, y, row_sample=phase1_sample, seed=mi_seed)
		if phase1_mi_mode == "hashed":
			w = estimate_edge_weight_hashed(n_rows, x, y, mdl_bits, num_buckets=phase1_buckets, row_sample=phase1_sample, seed=mi_seed)
		else:
			# Use exact MI for phase 1 if dataset is small enough
			w = estimate_edge_weight(n_rows, x, y, mdl_bits)
		return (i, j, w)
	
	# Compute all edge pairs (skip high-cardinality columns)
	stage_start = time.time()
	all_pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols) if i not in skip_high_card and j not in skip_high_card]
	max_workers = min(len(all_pairs), int(os.cpu_count() or 4), 16)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		all_fast_edges = list(executor.map(compute_fast_weight, all_pairs))
	
	# Filter negative weights and sort by weight
	all_fast_edges = [(u, v, w) for u, v, w in all_fast_edges if w > 0.0]
	all_fast_edges.sort(key=lambda x: x[2], reverse=True)
	profiler["phase1_mi"] = time.time() - stage_start
	
	# Phase 2: Refine top candidates with OpenZL MDL and EXACT MI for accuracy
	# Use exact MI (not hashed) for final selection - critical for compression ratio
	# Take more candidates to ensure we don't miss good edges
	# Scale candidate count with dataset size and number of columns - be extremely aggressive
	# For large datasets, we can afford to evaluate more edges
	max_candidates = min(len(all_fast_edges), max(n_cols * 25, 1500), 3000)  # Top 1500-3000 or 25x columns (extremely aggressive)
	top_candidates = all_fast_edges[:max_candidates]
	
	# Re-compute weights with OpenZL MDL + EXACT MI for maximum accuracy
	# Use larger sample or full data for phase 2 to get accurate MDL cost
	# Phase 2 needs high accuracy for final edge selection
	if n_rows <= 10000:
		# For 10k rows, use full data for maximum accuracy
		phase2_sample = None
	elif n_rows > 1_000_000:
		# For very large datasets, use substantial sample (100k-200k rows)
		# This is critical for accurate MI estimation
		phase2_sample = min(200_000, n_rows // 20)  # ~5% of data, up to 200k
	elif n_rows > 100_000:
		# For large datasets, use 50k-100k sample
		phase2_sample = min(100_000, n_rows // 10)  # ~10% of data, up to 100k
	else:
		# For medium datasets, use larger sample than phase 1
		phase2_sample = min(n_rows, max(20_000, n_rows // 5))  # At least 20k or 20% of data
	
	def recompute_weight(edge):
		u, v, _ = edge
		x_full, y_full = tab[:, u], tab[:, v]
		# Use same sample for both MDL and MI to ensure consistency
		if phase2_sample is not None and phase2_sample < n_rows:
			rng = np.random.default_rng(mi_seed)
			sample_idx = rng.choice(n_rows, size=phase2_sample, replace=False)
			x_sample = x_full[sample_idx]
			y_sample = y_full[sample_idx]
			n_effective = phase2_sample
		else:
			x_sample = x_full
			y_sample = y_full
			n_effective = n_rows
		# Use OpenZL MDL for accurate model cost (on sample)
		# Model cost represents the cost to store the joint histogram, which is approximately
		# the same for sample vs full data (depends on unique pairs, not total rows)
		mdl_bits = mdl_cost_fn_openzl(x_sample, y_sample, row_sample=None, seed=mi_seed)
		# Use EXACT MI (not hashed) for accurate mutual information
		# Compute MI on sample - this estimates the true MI which scales with data
		from .mi import compute_empirical_mi
		mi_nats = compute_empirical_mi(x_sample, y_sample)
		mi_bits = mi_nats / np.log(2.0)
		# Edge weight: n_rows * I(X;Y) - model_cost
		# MI benefit scales with n_rows, but model cost is fixed (depends on unique pairs)
		w = n_rows * mi_bits - mdl_bits
		return (u, v, w)
	
	stage_start = time.time()
	max_workers = min(len(top_candidates), int(os.cpu_count() or 4), 8)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		refined_edges = list(executor.map(recompute_weight, top_candidates))
	
	# Filter negative weights
	refined_edges = [(u, v, w) for u, v, w in refined_edges if w > 0.0]
	profiler["phase2_mi"] = time.time() - stage_start
	
	# Build maximum-weight forest - keep all positive-weight edges that don't create cycles
	# The paper optimizes over all forests, so we use greedy algorithm (Kruskal's for forests)
	# This is optimal: we get maximum total weight while maintaining acyclic structure
	import networkx as nx
	G = nx.Graph()
	G.add_nodes_from(range(n_cols))
	# Sort edges by weight (descending) and add them greedily - this is optimal for forests
	refined_edges_sorted = sorted(refined_edges, key=lambda x: x[2], reverse=True)
	edges = []
	for u, v, w in refined_edges_sorted:
		# Check if adding this edge would create a cycle
		# If u and v are in different connected components, it's safe to add
		if not nx.has_path(G, u, v):
			G.add_edge(u, v, weight=w)
			edges.append((u, v, w))
	
	# The greedy algorithm above produces a maximum-weight forest
	# If no edges were added, we have isolated nodes (each is its own tree)
	# This is correct behavior for a forest
	stage_start = time.time()
	parents = orient_forest(len(df.columns), [(int(u), int(v), float(w)) for u, v, w in edges])
	profiler["build_forest"] = time.time() - stage_start
	
	stage_start = time.time()
	indices, dicts, rare_blobs, is_numeric = _tokenize_columns(df, rare_threshold=rare_threshold)
	profiler["tokenize_columns"] = time.time() - stage_start
	rare_b64 = [base64.b64encode(b).decode("ascii") for b in rare_blobs]
	# Store is_numeric information in the model (reuse rare_meta structure or add new field)
	# For now, encode it in the dicts structure: None = numeric, list = categorical
	# Store delimiter so we can output CSV with correct separator
	# Normalize delimiter before storing: convert "\\t" to '\t' (tab character)
	# This ensures it's stored as a single character and works correctly in to_csv()
	normalized_delimiter = delimiter
	if delimiter == '\\t':
		normalized_delimiter = '\t'
	elif delimiter == '\\n':
		normalized_delimiter = '\n'
	model = Model(columns=list(df.columns), edges=[[int(u), int(v), float(w)] for u, v, w in edges], dicts=dicts, parents=parents, rare_meta=rare_b64, delimiter=normalized_delimiter, line_ending=line_ending)
	# Store is_numeric as a separate metadata field by encoding in rare_meta
	# Actually, we can infer it from dicts (None = numeric), so we don't need to store it separately

	stage_start = time.time()
	model_bytes = model.to_bytes()
	profiler["serialize_model"] = time.time() - stage_start
	
	stage_start = time.time()
	frames: List[bytes] = encode_columns_with_parents(indices, parents, dicts, workers=workers or 1)
	profiler["encode_columns"] = time.time() - stage_start

	# Compress frames with an additional layer for better compression
	# This helps especially when frames have redundancy or patterns
	# NOTE: Frame compression is disabled due to OpenZL decompression issues with small frames
	# TODO: Re-enable once we can reliably compress/decompress all frame sizes
	compressed_frames = []
	for frame in frames:
		# For now, don't compress frames - just store them uncompressed
		# This avoids OpenZL "Internal buffer too small" errors on decompression
		compressed_frames.append((False, frame))

	stage_start = time.time()
	with outp.open("wb") as f:
		f.write(b"TABCL\x00")
		f.write((4).to_bytes(4, "little"))  # Bump version to 4 for compressed frames
		f.write(len(model_bytes).to_bytes(8, "little"))
		f.write(model_bytes)
		f.write(len(compressed_frames).to_bytes(4, "little"))
		for is_compressed, frame_data in compressed_frames:
			# Use varint encoding for frame length to save space
			from .conditional import _encode_varint
			# Encode length as varint, then add compression flag as separate byte if needed
			length_bytes = _encode_varint(len(frame_data), signed=False)
			if is_compressed:
				# Write flag byte (0xFF) before length to indicate compression
				f.write(b"\xFF")
			f.write(length_bytes)
			f.write(frame_data)
	profiler["write_file"] = time.time() - stage_start
	
	# Print profiling summary
	total_time = time.time() - total_start
	if total_time > 0.1:  # Only print if compression took more than 100ms
		print("\nCompression Profiling:")
		print(f"  Total time: {total_time:.3f}s")
		print("  Time breakdown:")
		for stage, stage_time in sorted(profiler.items(), key=lambda x: x[1], reverse=True):
			percentage = (stage_time / total_time) * 100
			print(f"    {stage:20s}: {stage_time:7.3f}s ({percentage:5.1f}%)")


def decompress_file(input_path: str, output_path: str) -> None:
	inp = Path(input_path)
	outp = Path(output_path)
	data = inp.read_bytes()
	p = 0
	if len(data) < 6:
		raise RuntimeError("Corrupt file: too short")
	if data[:6] != b"TABCL\x00":
		raise RuntimeError("Unrecognized magic")
	p += 6
	_vers = int.from_bytes(data[p:p+4], "little"); p += 4
	mlen = int.from_bytes(data[p:p+8], "little"); p += 8
	if _vers >= 3:
		model = Model.from_bytes(data[p:p+mlen])
	else:
		# Legacy version: uncompressed JSON
		obj = json.loads(data[p:p+mlen].decode("utf-8"))
		model = Model(
			columns=list(obj["columns"]),
			edges=[list(e) for e in obj["edges"]],
			dicts=list(obj["dicts"]),
			parents=list(obj["parents"]),
			rare_meta=list(obj.get("rare_meta", [""] * len(obj["columns"]))),
			delimiter=obj.get("delimiter", ","),  # Default to comma for legacy files
			line_ending=obj.get("line_ending", "\n"),  # Default to LF for legacy files
		)
	p += mlen
	ncols = int.from_bytes(data[p:p+4], "little"); p += 4
	frames: List[bytes] = []
	
	if _vers >= 4:
		# Version 4+: varint-encoded frame lengths with compression flag
		from .conditional import _decode_varint
		for frame_idx in range(ncols):
			# Check for compression flag (0xFF byte before length)
			is_compressed = False
			if p >= len(data):
				raise RuntimeError(f"Unexpected end of file while reading frame {frame_idx} of {ncols}")
			if data[p] == 0xFF:
				is_compressed = True
				p += 1
				if p >= len(data):
					raise RuntimeError(f"Unexpected end of file after compression flag for frame {frame_idx} of {ncols}")
			
			# Decode varint length
			if p >= len(data):
				raise RuntimeError(f"Unexpected end of file while reading varint length for frame {frame_idx} of {ncols}")
			flen, bytes_consumed = _decode_varint(data, p, signed=False)
			p += bytes_consumed
			
			# Check if we have enough data to read the frame
			if p + flen > len(data):
				raise RuntimeError(
					f"Frame {frame_idx} of {ncols} claims length {flen} bytes, "
					f"but only {len(data) - p} bytes remain in file"
				)
			
			frame_data = data[p:p+flen]
			p += flen
			
			# Decompress if needed
			if is_compressed:
				try:
					frame_data = generic_bytes_decompress(frame_data)
				except RuntimeError as e:
					# Frame compression is currently disabled, so this shouldn't happen with new files.
					# If we encounter a compressed frame, it's from an old file format.
					# The error might also be from OpenZL issues with small compressed data.
					error_msg = str(e)
					if "Internal buffer too small" in error_msg or "error code: 71" in error_msg:
						if flen < 50:
							raise RuntimeError(
								f"Failed to decompress frame {frame_idx} of {ncols} columns. "
								f"Compressed frame is too small ({flen} bytes) for reliable OpenZL decompression. "
								f"This may be from an old file format with frame compression, or a compression issue. "
								f"Please recompress the file with the current version. "
								f"Original error: {e}"
							) from e
						else:
							raise RuntimeError(
								f"Failed to decompress frame {frame_idx} of {ncols} columns. "
								f"This file was created with frame compression enabled, which has known issues. "
								f"Please recompress the file with the current version (frame compression is now disabled). "
								f"Compressed frame length: {flen} bytes. "
								f"Original error: {e}"
							) from e
					else:
						# Other errors should still be raised
						raise RuntimeError(
							f"Failed to decompress frame {frame_idx} of {ncols} columns. "
							f"Compressed frame length: {flen} bytes, "
							f"Data remaining after frame: {len(data) - p} bytes. "
							f"Original error: {e}"
						) from e
			frames.append(frame_data)
	else:
		# Legacy version: fixed 8-byte frame lengths
		for _ in range(ncols):
			flen = int.from_bytes(data[p:p+8], "little"); p += 8
			frames.append(data[p:p+flen]); p += flen

	n_rows = None
	for j, parent in enumerate(model.parents):
		if parent == -1:
			from .codec import decompress_numeric_array
			arr = decompress_numeric_array(frames[j])
			n_rows = int(arr.shape[0])
			break
	if n_rows is None:
		raise RuntimeError("Could not determine row count")

	indices = decode_columns_with_parents(frames, model.parents, n_rows)

	rec_cols: Dict[str, Any] = {}
	EMPTY_STRING = ""
	MISSING_TOKEN = "nan"
	
	for col_idx, (name, ids, vocab) in enumerate(zip(model.columns, indices, model.dicts)):
		if vocab is None:
			# Numeric column: output as strings to match original CSV format
			# Since we only store integers in numeric columns, output as integers
			# Use str() directly on the int64 values to preserve format
			rec_cols[name] = [str(x) for x in ids.tolist()]
			continue
		elif isinstance(vocab, tuple) and len(vocab) == 2:
			# Special numeric format: scaled float or raw float
			format_type, scale_info = vocab
			if format_type == "_scale":
				# Scaled float: divide by scale factor
				scale = scale_info
				# Reconstruct values
				reconstructed = []
				for x in ids.tolist():
					val = float(x) / scale
					if scale == 1:
						# Pure integers - check if we need to preserve .0 format
						reconstructed.append(str(int(val)))
					else:
						reconstructed.append(str(val))
				
				# Apply format preservation from rare_meta if available
				if col_idx < len(model.rare_meta) and model.rare_meta[col_idx]:
					try:
						# Decode format preservation data
						rare_b64 = model.rare_meta[col_idx]
						rare_compressed = base64.b64decode(rare_b64)
						rare_data = generic_bytes_decompress(rare_compressed, fallback_to_uncompressed=True)
						
						if len(rare_data) >= 4:
							import struct
							ptr = 0
							format_count = struct.unpack("<I", rare_data[ptr:ptr+4])[0]
							ptr += 4
							format_map = {}
							prev_idx = -1
							for _ in range(format_count):
								if ptr >= len(rare_data):
									break
								# Decode delta-encoded index using varint
								delta, bytes_read = _decode_varint(rare_data, ptr, signed=False)
								if bytes_read == 0:
									break
								ptr += bytes_read
								idx = prev_idx + delta
								prev_idx = idx
								
								# Decode string length using varint
								if ptr >= len(rare_data):
									break
								str_len, bytes_read = _decode_varint(rare_data, ptr, signed=False)
								if bytes_read == 0:
									break
								ptr += bytes_read
								
								# Read the string
								if ptr + str_len <= len(rare_data):
									orig_str = rare_data[ptr:ptr+str_len].decode("utf-8")
									ptr += str_len
									format_map[idx] = orig_str
							# Apply format map
							for idx, orig_str in format_map.items():
								if idx < len(reconstructed):
									reconstructed[idx] = orig_str
					except Exception as e:
						# If format preservation fails, use reconstructed values as-is
						# Silently continue - format preservation is optional for roundtrip accuracy
						pass
				
				rec_cols[name] = reconstructed
			elif format_type == "_float":
				# Raw float stored as int64 bit pattern
				float_arr = ids.view(np.float64)
				rec_cols[name] = [str(x) for x in float_arr.tolist()]
			elif format_type == "_timestamp":
				# Timestamp stored as Unix timestamp (int64), convert back to datetime string
				from datetime import datetime
				import struct
				sentinel = scale_info  # Sentinel value for None/empty
				
				# Reconstruct unparseable original strings from rare_meta (stored as base64)
				unparseable_map = {}
				if col_idx < len(model.rare_meta) and model.rare_meta[col_idx]:
					try:
						# Decode base64 and decompress
						rare_b64 = model.rare_meta[col_idx]
						rare_compressed = base64.b64decode(rare_b64)
						# Use fallback_to_uncompressed=True to handle cases where data is uncompressed
						rare_data = generic_bytes_decompress(rare_compressed, fallback_to_uncompressed=True)
						
						ptr = 0
						if len(rare_data) >= 4:
							unparseable_count = struct.unpack("<I", rare_data[ptr:ptr+4])[0]
							ptr += 4
							for _ in range(unparseable_count):
								if ptr + 6 <= len(rare_data):
									idx = struct.unpack("<I", rare_data[ptr:ptr+4])[0]
									ptr += 4
									str_len = struct.unpack("<H", rare_data[ptr:ptr+2])[0]
									ptr += 2
									if str_len > 0 and ptr + str_len <= len(rare_data):
										orig_str = rare_data[ptr:ptr+str_len].decode("utf-8")
										unparseable_map[idx] = orig_str
										ptr += str_len
									else:
										unparseable_map[idx] = ""
					except Exception:
						# If decoding fails, just continue without unparseable strings
						pass
				
				rec_cols[name] = []
				for idx, ts in enumerate(ids.tolist()):
					if ts == sentinel:
						# Check if we have an original string for this index
						if idx in unparseable_map:
							rec_cols[name].append(unparseable_map[idx])
						else:
							rec_cols[name].append("")
					else:
						# Convert Unix timestamp back to datetime string
						dt = datetime.fromtimestamp(ts)
						# Use the original format: "%Y-%m-%d %H:%M:%S"
						rec_cols[name].append(dt.strftime("%Y-%m-%d %H:%M:%S"))
			else:
				# Unknown format, treat as integer
				rec_cols[name] = [str(int(x)) for x in ids.tolist()]
			continue
		
		# Categorical column: reconstruct from vocabulary
		vals = []
		for idx in ids.tolist():
			if idx < 0 or idx >= len(vocab):
				vals.append(EMPTY_STRING)
			else:
				v = vocab[idx]
				if v == EMPTY_STRING or v == MISSING_TOKEN or v is None:
					vals.append(EMPTY_STRING)
				else:
					# Ensure we output as string
					vals.append(str(v) if v != EMPTY_STRING else EMPTY_STRING)
		
		# Apply rare overrides if present
		b64 = model.rare_meta[col_idx] if col_idx < len(model.rare_meta) else ""
		if b64:
			blob = base64.b64decode(b64)
			# Use fallback_to_uncompressed=True to handle cases where data is uncompressed
			override_bytes = generic_bytes_decompress(blob, fallback_to_uncompressed=True)
			if override_bytes:
				# Try new binary format first, fall back to JSON for backward compatibility
				try:
					overrides = _decode_rare_overrides(override_bytes)
				except Exception:
					# Fall back to JSON format for old files
					try:
						overrides = json.loads(override_bytes.decode("utf-8"))
					except Exception:
						overrides = []
				for row_idx, true_val in overrides:
					# Store override as string
					if true_val is None or true_val == "":
						vals[int(row_idx)] = EMPTY_STRING
					else:
						vals[int(row_idx)] = str(true_val)
		
		rec_cols[name] = vals
	
	# Create DataFrame - all values are already strings
	df = pd.DataFrame(rec_cols, dtype=str)
	
	# Replace any 'nan' strings (from pandas conversion) with empty string
	df = df.replace('nan', EMPTY_STRING)
	df = df.replace('None', EMPTY_STRING)
	
	# Write CSV exactly as input: use stored delimiter, no header, no index
	# Use stored line ending style to preserve original format (CRLF vs LF)
	# Get delimiter from model (always present, defaults to comma for backward compatibility)
	# Normalize delimiter: convert "\\t" to '\t' (tab character)
	delimiter = model.delimiter
	if delimiter == '\\t':
		delimiter = '\t'
	elif delimiter == '\\n':
		delimiter = '\n'
	# Use stored line ending style (defaults to '\n' for backward compatibility)
	line_ending = getattr(model, 'line_ending', '\n')
	csv_content = df.to_csv(index=False, header=False, sep=delimiter, na_rep=EMPTY_STRING, lineterminator=line_ending)
	
	# Remove trailing newline if present (to match some CSV formats)
	# Actually, keep it - most CSV files end with newline
	outp.write_bytes(csv_content.encode("utf-8"))


def main() -> None:
	parser = argparse.ArgumentParser(prog="tabcl", description="MDL-weighted Chow–Liu compressor using OpenZL")
	sub = parser.add_subparsers(dest="cmd", required=True)

	pp = sub.add_parser("prepare", help="Prepare raw delimited data into a clean CSV")
	pp.add_argument("--input", required=True, help="Raw file (csv/tsv/txt)")
	pp.add_argument("--output", required=True, help="Output cleaned CSV path")
	pp.add_argument("--has-header", action="store_true", help="Set if input already has a header row")

	pc = sub.add_parser("compress", help="Compress a CSV file")
	pc.add_argument("--input", required=True)
	pc.add_argument("--output", required=True)
	pc.add_argument("--delimiter", default=",")
	pc.add_argument("--rare-threshold", type=int, default=1, help="Bucket values with freq <= N as <RARE> and store sparse overrides (lossless)")
	pc.add_argument("--mi-mode", choices=["exact", "hashed"], default="exact", help="Mutual information estimation mode")
	pc.add_argument("--mi-buckets", type=int, default=4096, help="Buckets for hashed MI")
	pc.add_argument("--mi-sample", type=int, default=None, help="Row sample size for MI (optional)")
	pc.add_argument("--mi-seed", type=int, default=0, help="Random seed for MI sampling")
	pc.add_argument("--workers", type=int, default=None, help="Parallel workers for column/bucket compression")

	pdcp = sub.add_parser("decompress", help="Decompress to CSV")
	pdcp.add_argument("--input", required=True)
	pdcp.add_argument("--output", required=True)

	args = parser.parse_args()

	if args.cmd == "prepare":
		_prepare_raw_to_csv(Path(args.input), Path(args.output), header=args.has_header)
	elif args.cmd == "compress":
		compress_file(args.input, args.output, args.delimiter, rare_threshold=args.rare_threshold, mi_mode=args.mi_mode, mi_buckets=args.mi_buckets, mi_sample=args.mi_sample, mi_seed=args.mi_seed, workers=args.workers)
	elif args.cmd == "decompress":
		decompress_file(args.input, args.output)
	else:
		parser.print_help()
		sys.exit(2)

if __name__ == "__main__":
	main()
