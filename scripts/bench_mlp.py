#!/usr/bin/env python3
"""
Benchmark script comparing tabcl with and without MLP extension.
"""

import subprocess
import sys
import time
import csv
from pathlib import Path
import re


def get_file_size(path: Path) -> int:
	"""Get file size in bytes."""
	return path.stat().st_size


def run_benchmark(input_file: Path, use_mlp: bool = False) -> dict:
	"""Run tabcl compression benchmark."""
	output_file = input_file.with_suffix('.tabcl')
	if output_file.exists():
		output_file.unlink()
	
	# Build command
	cmd = [
		sys.executable, "-m", "src.tabcl.cli",
		"compress",
		"--input", str(input_file),
		"--output", str(output_file),
	]
	if use_mlp:
		cmd.append("--use-mlp")
	
	# Run compression
	start_time = time.time()
	try:
		result = subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			timeout=600,  # 10 minute timeout
		)
		compress_time = time.time() - start_time
		
		if result.returncode != 0:
			return {
				"success": False,
				"error": result.stderr,
				"time": compress_time,
			}
		
		# Get compressed size
		if not output_file.exists():
			return {
				"success": False,
				"error": "Output file not created",
				"time": compress_time,
			}
		
		compressed_size = get_file_size(output_file)
		original_size = get_file_size(input_file)
		compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
		
		# Test decompression
		restored_file = input_file.with_suffix('.restored.csv')
		if restored_file.exists():
			restored_file.unlink()
		
		decompress_cmd = [
			sys.executable, "-m", "src.tabcl.cli",
			"decompress",
			"--input", str(output_file),
			"--output", str(restored_file),
		]
		
		decompress_start = time.time()
		decompress_result = subprocess.run(
			decompress_cmd,
			capture_output=True,
			text=True,
			timeout=300,
		)
		decompress_time = time.time() - decompress_start
		
		roundtrip_ok = False
		if decompress_result.returncode == 0 and restored_file.exists():
			# Compare files
			import hashlib
			with open(input_file, 'rb') as f:
				original_hash = hashlib.md5(f.read()).hexdigest()
			with open(restored_file, 'rb') as f:
				restored_hash = hashlib.md5(f.read()).hexdigest()
			roundtrip_ok = (original_hash == restored_hash)
		
		# Cleanup
		if restored_file.exists():
			restored_file.unlink()
		
		return {
			"success": True,
			"original_size": original_size,
			"compressed_size": compressed_size,
			"compression_ratio": compression_ratio,
			"compress_time": compress_time,
			"decompress_time": decompress_time,
			"roundtrip_ok": roundtrip_ok,
		}
		
	except subprocess.TimeoutExpired:
		return {
			"success": False,
			"error": "Timeout",
			"time": 600.0,
		}
	except Exception as e:
		return {
			"success": False,
			"error": str(e),
			"time": time.time() - start_time,
		}
	finally:
		# Cleanup
		if output_file.exists():
			output_file.unlink()


def format_size(size_bytes: int) -> str:
	"""Format size in human-readable format."""
	for unit in ['B', 'KB', 'MB', 'GB']:
		if size_bytes < 1024.0:
			return f"{size_bytes:.1f} {unit}"
		size_bytes /= 1024.0
	return f"{size_bytes:.1f} TB"


def main():
	"""Main benchmark function."""
	if len(sys.argv) < 2:
		print("Usage: bench_mlp.py <input_file> [input_file2 ...]")
		sys.exit(1)
	
	input_files = [Path(f) for f in sys.argv[1:]]
	
	results = []
	
	for input_file in input_files:
		if not input_file.exists():
			print(f"Error: File not found: {input_file}")
			continue
		
		print(f"\n{'='*60}")
		print(f"Benchmarking: {input_file.name}")
		print(f"{'='*60}\n")
		
		# Test without MLP
		print("1. Testing without MLP (histogram-based)...")
		result_no_mlp = run_benchmark(input_file, use_mlp=False)
		
		if result_no_mlp["success"]:
			print(f"   ✓ Compression successful")
			print(f"   Size: {format_size(result_no_mlp['compressed_size'])} "
			      f"(ratio x{result_no_mlp['compression_ratio']:.2f})")
			print(f"   Time: {result_no_mlp['compress_time']:.3f}s")
			print(f"   Roundtrip: {'✓' if result_no_mlp['roundtrip_ok'] else '✗'}")
		else:
			print(f"   ✗ Compression failed: {result_no_mlp.get('error', 'Unknown error')}")
		
		# Test with MLP
		print("\n2. Testing with MLP extension...")
		result_mlp = run_benchmark(input_file, use_mlp=True)
		
		if result_mlp["success"]:
			print(f"   ✓ Compression successful")
			print(f"   Size: {format_size(result_mlp['compressed_size'])} "
			      f"(ratio x{result_mlp['compression_ratio']:.2f})")
			print(f"   Time: {result_mlp['compress_time']:.3f}s")
			print(f"   Roundtrip: {'✓' if result_mlp['roundtrip_ok'] else '✗'}")
			
			# Compare
			if result_no_mlp["success"]:
				size_diff = result_mlp['compressed_size'] - result_no_mlp['compressed_size']
				size_diff_pct = (size_diff / result_no_mlp['compressed_size']) * 100
				time_diff = result_mlp['compress_time'] - result_no_mlp['compress_time']
				time_diff_pct = (time_diff / result_no_mlp['compress_time']) * 100
				
				print(f"\n   Comparison:")
				print(f"   Size change: {size_diff:+.0f} bytes ({size_diff_pct:+.2f}%)")
				print(f"   Time change: {time_diff:+.3f}s ({time_diff_pct:+.2f}%)")
		else:
			print(f"   ✗ Compression failed: {result_mlp.get('error', 'Unknown error')}")
		
		# Store results
		results.append({
			"file": input_file.name,
			"no_mlp_size": result_no_mlp.get("compressed_size", 0),
			"no_mlp_ratio": result_no_mlp.get("compression_ratio", 0),
			"no_mlp_time": result_no_mlp.get("compress_time", 0),
			"no_mlp_roundtrip": result_no_mlp.get("roundtrip_ok", False),
			"mlp_size": result_mlp.get("compressed_size", 0),
			"mlp_ratio": result_mlp.get("compression_ratio", 0),
			"mlp_time": result_mlp.get("compress_time", 0),
			"mlp_roundtrip": result_mlp.get("roundtrip_ok", False),
			"size_diff": result_mlp.get("compressed_size", 0) - result_no_mlp.get("compressed_size", 0),
			"time_diff": result_mlp.get("compress_time", 0) - result_no_mlp.get("compress_time", 0),
		})
	
	# Write results to CSV
	output_csv = Path("mlp_benchmark_results.csv")
	with open(output_csv, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=[
			"file", "no_mlp_size", "no_mlp_ratio", "no_mlp_time", "no_mlp_roundtrip",
			"mlp_size", "mlp_ratio", "mlp_time", "mlp_roundtrip",
			"size_diff", "time_diff",
		])
		writer.writeheader()
		writer.writerows(results)
	
	print(f"\n{'='*60}")
	print(f"Results written to: {output_csv}")
	print(f"{'='*60}\n")


if __name__ == "__main__":
	main()

