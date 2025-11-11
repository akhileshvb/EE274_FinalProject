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
from .conditional import orient_forest, encode_columns_with_parents, decode_columns_with_parents


@dataclass
class Model:
	columns: List[str]
	edges: List[List[Any]]  # [u, v, weight]
	dicts: List[Any]  # per-column tokens list or None for numeric columns
	parents: List[int]
	rare_meta: List[str]
	delimiter: str = ","  # CSV delimiter (default comma for backward compatibility)

	def to_bytes(self) -> bytes:
		obj = {
			"columns": self.columns,
			"edges": self.edges,
			"dicts": self.dicts,
			"parents": self.parents,
			"rare_meta": self.rare_meta,
			"delimiter": self.delimiter,  # Store delimiter
		}
		json_bytes = json.dumps(obj, separators=(",", ":")).encode("utf-8")
		# Compress the model metadata to reduce overhead
		return generic_bytes_compress(json_bytes)

	@staticmethod
	def from_bytes(b: bytes) -> "Model":
		# Decompress the model metadata
		decompressed_bytes = generic_bytes_decompress(b)
		obj = json.loads(decompressed_bytes.decode("utf-8"))
		return Model(
			columns=list(obj["columns"]),
			edges=[list(e) for e in obj["edges"]],
			dicts=list(obj["dicts"]),
			parents=list(obj["parents"]),
			rare_meta=list(obj.get("rare_meta", [""] * len(obj.get("columns", [])))),
			delimiter=obj.get("delimiter", ","),  # Default to comma for backward compatibility
		)


def _load_csv(path: Path, delimiter: str) -> pd.DataFrame:
	# Handle both tab-separated and comma-separated
	sep = '\t' if delimiter == '\t' else delimiter
	# Read with no header, keeping all values as strings to preserve exact format
	# Use dtype=str to prevent type inference that could change empty strings to NaN
	df = pd.read_csv(path, delimiter=sep, header=None, skipinitialspace=False, dtype=str, keep_default_na=False, na_values=[])
	# Replace empty strings with None for processing, but we'll track original format
	# Empty strings become None for easier processing, but we preserve the fact they were empty
	return df


def _df_to_table(df: pd.DataFrame) -> np.ndarray:
	return df.astype(object).to_numpy()


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
		
		# Try to detect if this column should be numeric
		# Check if all non-empty values can be converted to integers
		can_be_numeric = True
		has_empty = False
		numeric_values = []
		
		for val in series:
			val_str = str(val).strip() if val is not None else ""
			if val_str == EMPTY_STRING or val_str == "" or pd.isna(val):
				has_empty = True
				continue
			try:
				# Try to convert to int
				numeric_values.append(int(val_str))
			except (ValueError, TypeError):
				can_be_numeric = False
				break
		
		# If all non-empty values are numeric integers, store as numeric column
		# Only treat as numeric if there are no empty values (empty values complicate reconstruction)
		if can_be_numeric and not has_empty:
			# All values are numeric integers, no missing values
			try:
				if len(numeric_values) == len(series):
					indices_list.append(np.array(numeric_values, dtype=np.int64))
					dicts.append(None)
					rare_blobs.append(b"")
					is_numeric_list.append(True)
					continue
			except (ValueError, OverflowError):
				# Can't convert to int64, treat as categorical
				can_be_numeric = False
		
		# Treat as categorical/string column
		is_numeric_list.append(False)
		series = series.astype(object)
		# Replace None/NaN with empty string for consistent handling
		series = series.fillna(EMPTY_STRING)
		
		vc = series.value_counts(dropna=False)
		# Optimize rare value handling for compression
		# Bucket rare values to reduce vocabulary size, which improves encoding efficiency
		total_count = len(series)
		rare_values = set()
		if rare_threshold > 0:
			# Calculate optimal threshold: bucket values that are truly rare
			# Strategy: bucket if frequency is low AND total occurrences are small
			for val, count in vc.items():
				freq_ratio = count / total_count
				# Bucket rare values: low frequency and small overall presence
				if count <= rare_threshold and freq_ratio < 0.01:  # Less than 1% of rows
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
		ov_bytes = json.dumps(overrides, separators=(",", ":")).encode("utf-8") if overrides else b"[]"
		rare_blobs.append(generic_bytes_compress(ov_bytes))
	
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
	inp = Path(input_path)
	outp = Path(output_path)
	df = _load_csv(inp, delimiter)
	tab = _df_to_table(df)
	# Two-phase approach for speed + accuracy:
	# 1. Fast proxy MDL to rank ALL possible edges quickly (not just MST)
	# 2. OpenZL MDL on top candidates, then build MST
	n_rows = len(tab)
	n_cols = len(tab[0])
	
	# Skip high-cardinality columns (they compress better independently)
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
	
	# Phase 1: Fast ranking with proxy MDL on all edge pairs
	# For datasets <= 10k rows, can afford to use exact MI throughout for better accuracy
	use_exact_mi_phase1 = (n_rows <= 10000)
	
	if mi_sample is None:
		# Use moderate samples for speed in phase 1
		if n_rows > 100_000:
			phase1_sample = 3000
		elif n_rows > 10_000:
			phase1_sample = 4000 if use_exact_mi_phase1 else 2000   # Larger sample if using exact MI
		elif n_rows > 1_000:
			phase1_sample = 2000 if use_exact_mi_phase1 else 1500
		else:
			phase1_sample = None
	else:
		phase1_sample = mi_sample
	
	# For small datasets, use exact MI throughout for accuracy
	# For larger datasets, use hashed MI in phase 1, exact in phase 2
	if mi_mode == "auto":
		if n_rows <= 10000:
			phase1_mi_mode = "exact"  # Use exact MI for better accuracy on smaller datasets
		else:
			phase1_mi_mode = "hashed"
	elif mi_mode == "exact":
		phase1_mi_mode = "exact"
	else:
		phase1_mi_mode = "hashed"
	
	# Use more buckets for better accuracy in phase 1 if using hashed
	phase1_buckets = mi_buckets if use_exact_mi_phase1 else min(mi_buckets, 2048)
	
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
	all_pairs = [(i, j) for i in range(n_cols) for j in range(i + 1, n_cols) if i not in skip_high_card and j not in skip_high_card]
	max_workers = min(len(all_pairs), int(os.cpu_count() or 4), 16)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		all_fast_edges = list(executor.map(compute_fast_weight, all_pairs))
	
	# Filter negative weights and sort by weight
	all_fast_edges = [(u, v, w) for u, v, w in all_fast_edges if w > 0.0]
	all_fast_edges.sort(key=lambda x: x[2], reverse=True)
	
	# Phase 2: Refine top candidates with OpenZL MDL and EXACT MI for accuracy
	# Use exact MI (not hashed) for final selection - critical for compression ratio
	# Take more candidates to ensure we don't miss good edges
	max_candidates = min(len(all_fast_edges), n_cols * 5, 200)  # Top 200 or 5x columns (increased)
	top_candidates = all_fast_edges[:max_candidates]
	
	# Re-compute weights with OpenZL MDL + EXACT MI for maximum accuracy
	# Use larger sample or full data for phase 2 to get accurate MDL cost
	if n_rows <= 10000:
		# For 10k rows, use full data for maximum accuracy
		phase2_sample = None
	else:
		# For larger datasets, use larger sample
		phase2_sample = min(n_rows, 8000)  # Larger sample for accuracy
	
	def recompute_weight(edge):
		u, v, _ = edge
		x, y = tab[:, u], tab[:, v]
		# Use OpenZL MDL for accurate model cost
		mdl_bits = mdl_cost_fn_openzl(x, y, row_sample=phase2_sample, seed=mi_seed)
		# Use EXACT MI (not hashed) for accurate mutual information
		# This is critical for high-cardinality data like hex strings
		w = estimate_edge_weight(n_rows, x, y, mdl_bits)
		return (u, v, w)
	
	max_workers = min(len(top_candidates), int(os.cpu_count() or 4), 8)
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		refined_edges = list(executor.map(recompute_weight, top_candidates))
	
	# Filter negative weights
	refined_edges = [(u, v, w) for u, v, w in refined_edges if w > 0.0]
	
	# Build MST on refined edges
	import networkx as nx
	G = nx.Graph()
	G.add_nodes_from(range(n_cols))
	for u, v, w in refined_edges:
		G.add_edge(u, v, weight=w)
	
	mst = nx.maximum_spanning_tree(G)
	edges = [(u, v, float(d.get("weight", 0.0))) for u, v, d in mst.edges(data=True)]
	edges = [(u, v, w) for u, v, w in edges if w > 0.0]
	parents = orient_forest(len(df.columns), [(int(u), int(v), float(w)) for u, v, w in edges])
	indices, dicts, rare_blobs, is_numeric = _tokenize_columns(df, rare_threshold=rare_threshold)
	rare_b64 = [base64.b64encode(b).decode("ascii") for b in rare_blobs]
	# Store is_numeric information in the model (reuse rare_meta structure or add new field)
	# For now, encode it in the dicts structure: None = numeric, list = categorical
	# Store delimiter so we can output CSV with correct separator
	model = Model(columns=list(df.columns), edges=[[int(u), int(v), float(w)] for u, v, w in edges], dicts=dicts, parents=parents, rare_meta=rare_b64, delimiter=delimiter)
	# Store is_numeric as a separate metadata field by encoding in rare_meta
	# Actually, we can infer it from dicts (None = numeric), so we don't need to store it separately

	model_bytes = model.to_bytes()
	frames: List[bytes] = encode_columns_with_parents(indices, parents, dicts, workers=workers or 1)

	with outp.open("wb") as f:
		f.write(b"TABCL\x00")
		f.write((3).to_bytes(4, "little"))  # Bump version to 3 for compressed model metadata
		f.write(len(model_bytes).to_bytes(8, "little"))
		f.write(model_bytes)
		f.write(len(frames).to_bytes(4, "little"))
		for frame in frames:
			f.write(len(frame).to_bytes(8, "little"))
			f.write(frame)


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
		)
	p += mlen
	ncols = int.from_bytes(data[p:p+4], "little"); p += 4
	frames: List[bytes] = []
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
			override_bytes = generic_bytes_decompress(blob)
			overrides = json.loads(override_bytes.decode("utf-8")) if override_bytes else []
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
	# Use lineterminator='\n' to ensure consistent line endings
	# Get delimiter from model (always present, defaults to comma for backward compatibility)
	csv_content = df.to_csv(index=False, header=False, sep=model.delimiter, na_rep=EMPTY_STRING, lineterminator='\n')
	
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
