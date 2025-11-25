#!/usr/bin/env python3
"""
Benchmark script comparing histogram-based compression with neural autoregressive compression.
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


def run_benchmark(input_file: Path, use_neural: bool = False) -> dict:
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
	if use_neural:
		cmd.append("--use-neural")
	
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
		print("Usage: bench_neural.py <input_file> [input_file2 ...]")
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
		
		# Test histogram-based (default)
		print("1. Testing histogram-based compression...")
		result_histogram = run_benchmark(input_file, use_neural=False)
		
		if result_histogram["success"]:
			print(f"   ✓ Compression successful")
			print(f"   Size: {format_size(result_histogram['compressed_size'])} "
			      f"(ratio x{result_histogram['compression_ratio']:.2f})")
			print(f"   Time: {result_histogram['compress_time']:.3f}s")
			print(f"   Roundtrip: {'✓' if result_histogram['roundtrip_ok'] else '✗'}")
		else:
			print(f"   ✗ Compression failed: {result_histogram.get('error', 'Unknown error')}")
		
		# Test neural compression
		print("\n2. Testing neural autoregressive compression...")
		result_neural = run_benchmark(input_file, use_neural=True)
		
		if result_neural["success"]:
			print(f"   ✓ Compression successful")
			print(f"   Size: {format_size(result_neural['compressed_size'])} "
			      f"(ratio x{result_neural['compression_ratio']:.2f})")
			print(f"   Time: {result_neural['compress_time']:.3f}s")
			print(f"   Roundtrip: {'✓' if result_neural['roundtrip_ok'] else '✗'}")
			
			# Compare
			if result_histogram["success"]:
				size_diff = result_neural['compressed_size'] - result_histogram['compressed_size']
				size_diff_pct = (size_diff / result_histogram['compressed_size']) * 100
				time_diff = result_neural['compress_time'] - result_histogram['compress_time']
				time_diff_pct = (time_diff / result_histogram['compress_time']) * 100
				
				print(f"\n   Comparison:")
				print(f"   Size change: {size_diff:+.0f} bytes ({size_diff_pct:+.2f}%)")
				print(f"   Time change: {time_diff:+.3f}s ({time_diff_pct:+.2f}%)")
		else:
			print(f"   ✗ Compression failed: {result_neural.get('error', 'Unknown error')}")
		
		# Store results
		results.append({
			"file": input_file.name,
			"histogram_size": result_histogram.get("compressed_size", 0),
			"histogram_ratio": result_histogram.get("compression_ratio", 0),
			"histogram_time": result_histogram.get("compress_time", 0),
			"histogram_roundtrip": result_histogram.get("roundtrip_ok", False),
			"neural_size": result_neural.get("compressed_size", 0),
			"neural_ratio": result_neural.get("compression_ratio", 0),
			"neural_time": result_neural.get("compress_time", 0),
			"neural_roundtrip": result_neural.get("roundtrip_ok", False),
			"size_diff": result_neural.get("compressed_size", 0) - result_histogram.get("compressed_size", 0),
			"time_diff": result_neural.get("compress_time", 0) - result_histogram.get("compress_time", 0),
		})
	
	# Write results to CSV
	output_csv = Path("neural_benchmark_results.csv")
	with open(output_csv, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=[
			"file", "histogram_size", "histogram_ratio", "histogram_time", "histogram_roundtrip",
			"neural_size", "neural_ratio", "neural_time", "neural_roundtrip",
			"size_diff", "time_diff",
		])
		writer.writeheader()
		writer.writerows(results)
	
	print(f"\n{'='*60}")
	print(f"Results written to: {output_csv}")
	print(f"{'='*60}\n")


if __name__ == "__main__":
	main()

