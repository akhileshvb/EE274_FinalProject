import argparse
import os
import shutil
import subprocess
import sys
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
	args = ap.parse_args()

	inp = Path(args.input).resolve()
	if not inp.exists():
		print(f"Input not found: {inp}", file=sys.stderr)
		sys.exit(1)

	outdir = Path(args.outdir).resolve() if args.outdir else inp.parent
	outdir.mkdir(parents=True, exist_ok=True)

	orig_size = inp.stat().st_size

	# tabcl
	tabcl_out = outdir / (inp.name + ".tabcl")
	tabcl_cmd = ["tabcl", "compress", "--input", str(inp), "--output", str(tabcl_out), "--delimiter", args.delimiter]
	tabcl_time = run(tabcl_cmd)
	tabcl_size = tabcl_out.stat().st_size

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

if __name__ == "__main__":
	main()
