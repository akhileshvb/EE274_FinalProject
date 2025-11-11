import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def human(n: int) -> str:
	for unit in ["B", "KB", "MB", "GB", "TB"]:
		if n < 1024:
			return f"{n:.1f} {unit}"
		n //= 1024
	return f"{n} PB"


def run(cmd: list[str]) -> float:
	start = time.perf_counter()
	subprocess.run(cmd, check=True)
	return time.perf_counter() - start


def main() -> None:
	ap = argparse.ArgumentParser(description="Benchmark tabcl vs gzip/zstd on a CSV")
	ap.add_argument("--input", required=True, help="Path to input CSV")
	ap.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
	ap.add_argument("--outdir", default=None, help="Output directory (default alongside input)")
	ap.add_argument("--zstd-level", type=int, default=19)
	ap.add_argument("--mi-mode", default="auto", choices=["exact", "hashed", "auto"], help="MI computation mode (auto detects)")
	ap.add_argument("--workers", type=int, default=None, help="Number of parallel workers for tabcl (default: auto-detect CPU count)")
	ap.add_argument("--rare-threshold", type=int, default=1, help="Rare value threshold for tabcl (default: 1)")
	args = ap.parse_args()

	inp = Path(args.input).resolve()
	if not inp.exists():
		print(f"Input not found: {inp}", file=sys.stderr)
		sys.exit(1)

	outdir = Path(args.outdir).resolve() if args.outdir else inp.parent
	outdir.mkdir(parents=True, exist_ok=True)

	orig_size = inp.stat().st_size

	# Auto-detect MI mode if needed
	mi_mode = args.mi_mode
	mi_sample = None
	if mi_mode == "auto":
		# Check file size - large files likely have high cardinality
		if orig_size > 1_000_000:  # >1MB
			mi_mode = "hashed"
			# Use more aggressive sampling for speed
			if orig_size > 100_000_000:  # >100MB
				mi_sample = 50000  # Sample 50k rows (reduced from 100k)
			elif orig_size > 10_000_000:  # >10MB
				mi_sample = 20000  # Sample 20k rows for medium files
			elif orig_size > 1_000_000:  # >1MB
				mi_sample = 10000  # Sample 10k rows for smaller large files

	# tabcl
	tabcl_out = outdir / (inp.name + ".tabcl")
	tabcl_cmd = ["tabcl", "compress", "--input", str(inp), "--output", str(tabcl_out), "--delimiter", args.delimiter, "--mi-mode", mi_mode, "--rare-threshold", str(args.rare_threshold)]
	if mi_sample:
		tabcl_cmd.extend(["--mi-sample", str(mi_sample)])
	# Auto-detect workers if not specified
	workers = args.workers
	if workers is None:
		workers = os.cpu_count() or 4
	if workers and workers > 0:
		tabcl_cmd.extend(["--workers", str(workers)])
	tabcl_time = run(tabcl_cmd)
	tabcl_size = tabcl_out.stat().st_size
	# Verify roundtrip (but don't time decompression)
	try:
		with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
			tmp_path = Path(tmp.name)
		subprocess.run(["tabcl", "decompress", "--input", str(tabcl_out), "--output", str(tmp_path)], check=True, capture_output=True)
		# Compare files (ignore whitespace differences)
		import hashlib
		orig_hash = hashlib.md5(open(inp, 'rb').read()).hexdigest()
		restored_hash = hashlib.md5(open(tmp_path, 'rb').read()).hexdigest()
		if orig_hash != restored_hash:
			print(f"WARNING: tabcl roundtrip verification failed! Files differ.", file=sys.stderr)
		tmp_path.unlink()
	except Exception as e:
		print(f"WARNING: Could not verify tabcl roundtrip: {e}", file=sys.stderr)

	# gzip
	gzip_out = outdir / (inp.name + ".gz")
	gzip_time = None
	if shutil.which("gzip"):
		# -k keep original, -f force overwrite
		cmd = ["gzip", "-kf", str(inp)]
		gzip_time = run(cmd)
		# Move produced file to outdir if needed
		produced = inp.parent / (inp.name + ".gz")
		if produced.exists() and produced != gzip_out:
			produced.replace(gzip_out)
		gzip_size = gzip_out.stat().st_size
	else:
		gzip_size = None

	# zstd
	zstd_out = outdir / (inp.name + ".zst")
	zstd_time = None
	if shutil.which("zstd"):
		cmd = ["zstd", "-q", f"-{args.zstd_level}", "-f", str(inp), "-o", str(zstd_out)]
		zstd_time = run(cmd)
		zstd_size = zstd_out.stat().st_size
	else:
		zstd_size = None
	
	# bzip2
	bzip2_out = outdir / (inp.name + ".bz2")
	bzip2_time = None
	if shutil.which("bzip2"):
		cmd = ["bzip2", "-kf", str(inp)]
		bzip2_time = run(cmd)
		produced = inp.parent / (inp.name + ".bz2")
		if produced.exists() and produced != bzip2_out:
			produced.replace(bzip2_out)
		bzip2_size = bzip2_out.stat().st_size if bzip2_out.exists() else None
	else:
		bzip2_size = None
	
	# Columnar gzip (compress each column separately)
	colgzip_size = None
	colgzip_time = None
	if shutil.which("gzip"):
		try:
			import pandas as pd
			df = pd.read_csv(inp, delimiter=args.delimiter, header=None)
			start = time.perf_counter()
			col_sizes = []
			with tempfile.TemporaryDirectory() as tmpdir:
				for col_idx, col_name in enumerate(df.columns):
					col_file = Path(tmpdir) / f"col_{col_idx}.txt"
					col_file.write_text("\n".join(str(v) for v in df[col_name].astype(str)))
					col_gz = Path(tmpdir) / f"col_{col_idx}.txt.gz"
					subprocess.run(["gzip", "-f", str(col_file)], check=True, capture_output=True)
					if col_gz.exists():
						col_sizes.append(col_gz.stat().st_size)
			colgzip_time = time.perf_counter() - start
			colgzip_size = sum(col_sizes)
		except Exception as e:
			print(f"Warning: Columnar gzip failed: {e}", file=sys.stderr)
			colgzip_size = None

	def ratio(size: int | None) -> str:
		if size is None:
			return "-"
		return f"x{orig_size / size:.2f}"

	print("Input:", inp)
	print(f"Original size: {human(orig_size)}")
	print()
	print("tabcl:")
	print(f"  size:  {human(tabcl_size)}  (ratio {ratio(tabcl_size)})")
	print(f"  time:  {tabcl_time:.3f}s")
	if gzip_size is not None:
		print("gzip:")
		print(f"  size:  {human(gzip_size)}  (ratio {ratio(gzip_size)})")
		print(f"  time:  {gzip_time:.3f}s")
	else:
		print("gzip: not found on PATH")
	if zstd_size is not None:
		print("zstd:")
		print(f"  size:  {human(zstd_size)}  (ratio {ratio(zstd_size)})")
		print(f"  time:  {zstd_time:.3f}s (level {args.zstd_level})")
	else:
		print("zstd: not found on PATH")
	if bzip2_size is not None:
		print("bzip2:")
		print(f"  size:  {human(bzip2_size)}  (ratio {ratio(bzip2_size)})")
		print(f"  time:  {bzip2_time:.3f}s")
	else:
		print("bzip2: not found on PATH")
	if colgzip_size is not None:
		print("columnar gzip:")
		print(f"  size:  {human(colgzip_size)}  (ratio {ratio(colgzip_size)})")
		print(f"  time:  {colgzip_time:.3f}s")
	else:
		print("columnar gzip: not available")

if __name__ == "__main__":
	main()
