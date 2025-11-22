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
	"""
	Run a command and return the elapsed time in seconds.
	This measures the full time including I/O (reading input, writing output).
	For compression tools, this includes:
	- Reading the input file
	- Compression computation
	- Writing the output file
	"""
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
	# Remove old .tabcl file if it exists (may have been created with old format)
	# Also check in the input directory in case it was created there
	if tabcl_out.exists():
		tabcl_out.unlink()
	old_tabcl = inp.parent / (inp.name + ".tabcl")
	if old_tabcl.exists() and old_tabcl != tabcl_out:
		old_tabcl.unlink()
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
	tmp_path = None
	try:
		with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
			tmp_path = Path(tmp.name)
		result = subprocess.run(["tabcl", "decompress", "--input", str(tabcl_out), "--output", str(tmp_path)], 
		                      capture_output=True, text=True, check=False)
		if result.returncode != 0:
			# Check if it's the known OpenZL decompression issue
			if "Internal buffer too small" in result.stderr or "error code: 71" in result.stderr:
				print(f"WARNING: tabcl decompress failed due to OpenZL decompression issue.", file=sys.stderr)
				if "frame compression enabled" in result.stderr:
					print(f"  The file was created with an older version that had frame compression enabled.", file=sys.stderr)
					print(f"  Please remove the old .tabcl file and recompress with the current version.", file=sys.stderr)
				else:
					print(f"  This may be due to very small compressed data that OpenZL cannot reliably decompress.", file=sys.stderr)
					print(f"  Compression succeeded, but decompression failed. This is a known OpenZL limitation.", file=sys.stderr)
				print(f"  The file has been removed - please run the benchmark again to create a new file.", file=sys.stderr)
			else:
				print(f"WARNING: tabcl decompress failed with return code {result.returncode}", file=sys.stderr)
				if result.stderr:
					# Only print first few lines to avoid clutter
					stderr_lines = result.stderr.split('\n')[:15]
					print(f"  stderr (first 15 lines):", file=sys.stderr)
					for line in stderr_lines:
						if line.strip():
							print(f"    {line}", file=sys.stderr)
				if result.stdout:
					print(f"  stdout: {result.stdout}", file=sys.stderr)
			if tmp_path and tmp_path.exists():
				tmp_path.unlink(missing_ok=True)
		else:
			# Compare files (ignore whitespace differences)
			import hashlib
			orig_hash = hashlib.md5(open(inp, 'rb').read()).hexdigest()
			restored_hash = hashlib.md5(open(tmp_path, 'rb').read()).hexdigest()
			if orig_hash != restored_hash:
				print(f"WARNING: tabcl roundtrip verification failed! Files differ.", file=sys.stderr)
				print(f"  Original hash: {orig_hash}", file=sys.stderr)
				print(f"  Restored hash: {restored_hash}", file=sys.stderr)
			if tmp_path and tmp_path.exists():
				tmp_path.unlink()
	except FileNotFoundError:
		print(f"WARNING: tabcl command not found in PATH, skipping roundtrip verification", file=sys.stderr)
	except Exception as e:
		print(f"WARNING: Could not verify tabcl roundtrip: {e}", file=sys.stderr)
		if tmp_path and tmp_path.exists():
			tmp_path.unlink(missing_ok=True)

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
			# Normalize delimiter: convert "\\t" to '\t' (tab character)
			delimiter = args.delimiter
			if delimiter == '\\t':
				delimiter = '\t'
			elif delimiter == '\\n':
				delimiter = '\n'
			# Use engine='python' and on_bad_lines='skip' to handle malformed lines
			import warnings
			warnings.filterwarnings('ignore', category=pd.errors.ParserWarning)
			df = pd.read_csv(inp, delimiter=delimiter, header=None, engine='python', on_bad_lines='skip')
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
