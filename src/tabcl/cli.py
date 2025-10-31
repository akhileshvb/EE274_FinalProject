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
from .codec import mdl_cost_fn_openzl, generic_bytes_compress, generic_bytes_decompress
from .conditional import orient_forest, encode_columns_with_parents, decode_columns_with_parents


@dataclass
class Model:
	columns: List[str]
	edges: List[List[Any]]  # [u, v, weight]
	dicts: List[Any]  # per-column tokens list or None for numeric columns
	parents: List[int]
	rare_meta: List[str]

	def to_bytes(self) -> bytes:
		obj = {
			"columns": self.columns,
			"edges": self.edges,
			"dicts": self.dicts,
			"parents": self.parents,
			"rare_meta": self.rare_meta,
		}
		return json.dumps(obj, separators=(",", ":")).encode("utf-8")

	@staticmethod
	def from_bytes(b: bytes) -> "Model":
		obj = json.loads(b.decode("utf-8"))
		return Model(
			columns=list(obj["columns"]),
			edges=[list(e) for e in obj["edges"]],
			dicts=list(obj["dicts"]),
			parents=list(obj["parents"]),
			rare_meta=list(obj.get("rare_meta", [""] * len(obj["columns"]))),
		)


def _load_csv(path: Path, delimiter: str) -> pd.DataFrame:
	return pd.read_csv(path, delimiter=delimiter)


def _df_to_table(df: pd.DataFrame) -> np.ndarray:
	return df.astype(object).to_numpy()


def _tokenize_columns(df: pd.DataFrame, rare_threshold: int) -> Tuple[List[np.ndarray], List[Any], List[bytes]]:
	indices_list: List[np.ndarray] = []
	dicts: List[Any] = []
	rare_blobs: List[bytes] = []
	MISSING_LITERAL = "nan"
	for col in df.columns:
		series = df[col]
		# Numeric columns: keep raw int64 values, no dict, no rare overrides
		if pd.api.types.is_numeric_dtype(series):
			indices_list.append(series.astype(np.int64, copy=False).to_numpy())
			dicts.append(None)
			rare_blobs.append(b"")
			continue
		# Categorical/object: tokenization with rare bucketing
		series = series.astype(object)
		vc = series.value_counts(dropna=False)
		rare_values = set(vc[vc <= rare_threshold].index.tolist()) if rare_threshold > 0 else set()
		rare_values.discard(MISSING_LITERAL)
		rare_values.discard("")
		codes, uniques = pd.factorize(series, sort=False)
		codes = codes.astype(np.int64, copy=False)
		uni_list = uniques.tolist()
		vocab: List[Any] = []
		value_to_newid: Dict[Any, int] = {}
		missing_id = None
		for v in uni_list:
			if v == MISSING_LITERAL:
				missing_id = len(vocab)
				value_to_newid[v] = missing_id
				vocab.append(v)
				continue
			if v in rare_values:
				continue
			value_to_newid[v] = len(vocab)
			vocab.append(v if not pd.isna(v) else MISSING_LITERAL)
		rare_placeholder_id = None
		if rare_values:
			rare_placeholder_id = len(vocab)
			vocab.append("<RARE>")
		overrides: List[Tuple[int, Any]] = []
		new_codes = np.empty_like(codes)
		for i, code in enumerate(codes.tolist()):
			if code == -1:
				if missing_id is None:
					missing_id = len(vocab)
					value_to_newid[MISSING_LITERAL] = missing_id
					vocab.append(MISSING_LITERAL)
				new_codes[i] = missing_id
				continue
			v = uni_list[code]
			if v in rare_values:
				new_codes[i] = rare_placeholder_id  # type: ignore
				overrides.append((i, v))
			else:
				new_codes[i] = value_to_newid[v]
		indices_list.append(new_codes)
		dicts.append(vocab)
		ov_bytes = json.dumps(overrides, separators=(",", ":")).encode("utf-8") if overrides else b"[]"
		rare_blobs.append(generic_bytes_compress(ov_bytes))
	return indices_list, dicts, rare_blobs


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
	edges = build_mdl_weighted_forest(tab, mdl_cost_fn_openzl, mi_mode=mi_mode, num_buckets=mi_buckets, row_sample=mi_sample, seed=mi_seed)
	parents = orient_forest(len(df.columns), [(int(u), int(v), float(w)) for u, v, w in edges])
	indices, dicts, rare_blobs = _tokenize_columns(df, rare_threshold=rare_threshold)
	rare_b64 = [base64.b64encode(b).decode("ascii") for b in rare_blobs]
	model = Model(columns=list(df.columns), edges=[[int(u), int(v), float(w)] for u, v, w in edges], dicts=dicts, parents=parents, rare_meta=rare_b64)

	model_bytes = model.to_bytes()
	frames: List[bytes] = encode_columns_with_parents(indices, parents, dicts, workers=workers or 1)

	with outp.open("wb") as f:
		f.write(b"TABCL\x00")
		f.write((2).to_bytes(4, "little"))
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
	model = Model.from_bytes(data[p:p+mlen]); p += mlen
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
	missing_token = "nan"
	for col_idx, (name, ids, vocab) in enumerate(zip(model.columns, indices, model.dicts)):
		if vocab is None:
			# numeric column: take values directly
			rec_cols[name] = [int(x) for x in ids.tolist()]
			continue
		vals = []
		for idx in ids.tolist():
			if idx < 0 or idx >= len(vocab):
				vals.append(missing_token)
			else:
				v = vocab[idx]
				vals.append(missing_token if v is None else v)
		# Apply rare overrides if present
		b64 = model.rare_meta[col_idx] if col_idx < len(model.rare_meta) else ""
		if b64:
			blob = base64.b64decode(b64)
			override_bytes = generic_bytes_decompress(blob)
			overrides = json.loads(override_bytes.decode("utf-8")) if override_bytes else []
			for row_idx, true_val in overrides:
				vals[int(row_idx)] = true_val
		rec_cols[name] = vals
	df = pd.DataFrame(rec_cols)
	outp.write_bytes(df.to_csv(index=False).encode("utf-8"))


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
