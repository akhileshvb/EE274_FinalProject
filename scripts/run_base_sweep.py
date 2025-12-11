import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def parse_size_to_bytes(size_str: str, unit: str) -> int:
	"""Convert size string to bytes."""
	size = float(size_str)
	if unit == 'GB':
		return int(size * 1024 * 1024 * 1024)
	elif unit == 'MB':
		return int(size * 1024 * 1024)
	elif unit == 'KB':
		return int(size * 1024)
	else:
		return int(size)


def parse_throughput_to_mbps(throughput_str: str, unit: str) -> float:
	"""Convert throughput to MB/s."""
	throughput = float(throughput_str)
	if unit == 'GB/s':
		return throughput * 1024
	elif unit == 'MB/s':
		return throughput
	elif unit == 'KB/s':
		return throughput / 1024
	else:
		return 0.0


def parse_method_results(output: str, method_name: str, field_prefix: str = None) -> dict:
	"""Parse results for a specific compression method."""
	results = {}
	if field_prefix is None:
		field_prefix = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
	
	# Pattern: method_name: (at start of line) followed by size, ratio, time, throughput
	# Use ^ to match start of line to avoid matching substrings
	escaped_name = re.escape(method_name)
	# Match method name at start of line, then capture the metrics
	pattern = rf'^{escaped_name}:\s*$\s+size:\s+([\d.]+)\s+(B|KB|MB|GB).*?ratio\s+x([\d.]+).*?time:\s+([\d.]+)s.*?throughput:\s+([\d.]+)\s+(KB/s|MB/s|GB/s)'
	match = re.search(pattern, output, re.DOTALL | re.MULTILINE)
	if match:
		size = match.group(1)
		unit = match.group(2)
		ratio = float(match.group(3))
		time_val = float(match.group(4))
		throughput = match.group(5)
		throughput_unit = match.group(6)
		
		results[f'{field_prefix}_size_bytes'] = parse_size_to_bytes(size, unit)
		results[f'{field_prefix}_ratio'] = ratio
		results[f'{field_prefix}_latency_seconds'] = time_val
		results[f'{field_prefix}_throughput_mbps'] = parse_throughput_to_mbps(throughput, throughput_unit)
	return results


def parse_bench_output(output: str) -> dict:
	"""Parse bench.py output to extract metrics for all methods."""
	results = {}
	
	# Parse original size
	orig_match = re.search(r'Original size: ([\d.]+) (B|KB|MB|GB)', output)
	if orig_match:
		results['original_size_bytes'] = parse_size_to_bytes(orig_match.group(1), orig_match.group(2))
	
	# Parse in order: most specific first to avoid overlapping matches
	# 1. Parse tabcl + zstd first (most specific, contains "tabcl")
	tabcl_zstd_results = parse_method_results(output, "tabcl + zstd", "tabcl_zstd")
	results.update(tabcl_zstd_results)
	
	# 2. Parse tabcl (line graph) - more specific than just "tabcl"
	tabcl_line_results = parse_method_results(output, "tabcl (line graph)", "tabcl_line_graph")
	results.update(tabcl_line_results)
	
	# 3. Parse tabcl (must come after tabcl + zstd and tabcl (line graph) to avoid matching those)
	# Use the same pattern format as parse_method_results but with explicit check
	tabcl_results = parse_method_results(output, "tabcl", "tabcl")
	results.update(tabcl_results)
	
	# Parse baseline methods
	baseline_mappings = {
		"gzip": "gzip",
		"zstd": "zstd",
		"bzip2": "bzip2",
		"columnar gzip": "columnar_gzip",
		"columnar zstd": "columnar_zstd"
	}
	for method_name, field_prefix in baseline_mappings.items():
		method_results = parse_method_results(output, method_name, field_prefix)
		results.update(method_results)
	
	return results


def find_datasets(datasets_dir: Path) -> list[Path]:
	"""Find one CSV file per subdirectory in the datasets directory, prefer files with 'sample' in their name."""
	csv_files = []
	for dataset_dir in datasets_dir.iterdir():
		if dataset_dir.is_dir():
			candidate_files = [
				csv_file for csv_file in dataset_dir.glob("*.csv")
				if not any(skip in csv_file.name for skip in ['.tabcl', '.gz', '.bz2', '.zst', 'restored'])
				and csv_file.name != "ad.csv"
			]
			if not candidate_files:
				continue
			# Prefer file with 'sample' in the name
			sample_files = [f for f in candidate_files if 'sample' in f.name.lower()]
			if sample_files:
				# If there are multiple, pick the first after sorting for reproducibility
				csv_files.append(sorted(sample_files)[0])
			else:
				# Else just pick the first sorted
				csv_files.append(sorted(candidate_files)[0])
	return sorted(csv_files)


def main():
	ap = argparse.ArgumentParser(description="Run bench.py on all datasets")
	ap.add_argument("--datasets-dir", type=Path, default=Path("datasets"),
	                help="Directory containing datasets (default: datasets)")
	ap.add_argument("--output", type=Path, default=Path("results/base_sweep_v3.csv"),
	                help="Output CSV file (default: results/base_sweep.csv)")
	ap.add_argument("--zstd-level", type=int, default=1, help="Zstd compression level")
	ap.add_argument("--mi-mode", default="auto", choices=["exact", "hashed", "auto"],
	                help="MI computation mode")
	ap.add_argument("--skip-existing", action="store_true",
	                help="Skip datasets already in results file")
	args = ap.parse_args()
	
	# Create output directory
	args.output.parent.mkdir(parents=True, exist_ok=True)
	
	# Find all datasets
	datasets = find_datasets(args.datasets_dir)
	print(f"Found {len(datasets)} datasets to process")
	
	# Load existing results if skipping
	existing_datasets = set()
	if args.skip_existing and args.output.exists():
		with open(args.output, 'r') as f:
			reader = csv.DictReader(f)
			for row in reader:
				existing_datasets.add(row['dataset'])
	
	# Prepare results
	all_results = []
	
	for i, dataset_path in enumerate(datasets, 1):
		dataset_name = f"{dataset_path.parent.name}/{dataset_path.name}"
		
		if args.skip_existing and dataset_name in existing_datasets:
			print(f"[{i}/{len(datasets)}] Skipping {dataset_name} (already in results)")
			continue
		
		print(f"[{i}/{len(datasets)}] Processing {dataset_name}...")
		
		# Run bench.py
		cmd = [
			"python3", "scripts/bench.py",
			"--input", str(dataset_path),
			"--delimiter", ",",
			"--zstd-level", str(args.zstd_level),
			"--mi-mode", args.mi_mode
		]
		
		try:
			result = subprocess.run(cmd, capture_output=True, text=True, check=True)
			output = result.stdout + result.stderr
			
			# Parse results
			parsed = parse_bench_output(output)
			if parsed:
				parsed['dataset'] = dataset_name
				parsed['dataset_path'] = str(dataset_path)
				all_results.append(parsed)
				# Show tabcl results in summary
				tabcl_ratio = parsed.get('tabcl_ratio', 'N/A')
				tabcl_time = parsed.get('tabcl_latency_seconds', 'N/A')
				tabcl_throughput = parsed.get('tabcl_throughput_mbps', 'N/A')
				print(f"  ✓ Success: tabcl ratio={tabcl_ratio:.2f}x, "
				      f"time={tabcl_time:.2f}s, "
				      f"throughput={tabcl_throughput:.2f} MB/s" if isinstance(tabcl_ratio, (int, float)) else f"  ✓ Success: parsed results")
			else:
				print(f"  ✗ Failed to parse output")
				
		except subprocess.CalledProcessError as e:
			print(f"  ✗ Error running bench.py: {e}")
			print(f"    stderr: {e.stderr[:200] if e.stderr else 'N/A'}")
			continue
		except Exception as e:
			print(f"  ✗ Unexpected error: {e}")
			continue
	
	# Write results to CSV
	if all_results:
		# Define all possible fieldnames (tabcl + all baselines)
		base_fields = ['dataset', 'dataset_path', 'original_size_bytes']
		
		# Tabcl fields
		tabcl_fields = [
			'tabcl_size_bytes', 'tabcl_ratio', 'tabcl_latency_seconds', 'tabcl_throughput_mbps',
			'tabcl_line_graph_size_bytes', 'tabcl_line_graph_ratio', 'tabcl_line_graph_latency_seconds', 'tabcl_line_graph_throughput_mbps',
			'tabcl_zstd_size_bytes', 'tabcl_zstd_ratio', 'tabcl_zstd_latency_seconds', 'tabcl_zstd_throughput_mbps'
		]
		
		# Baseline fields
		baseline_methods = ["gzip", "zstd", "bzip2", "columnar_gzip", "columnar_zstd"]
		baseline_fields = []
		for method in baseline_methods:
			baseline_fields.extend([
				f'{method}_size_bytes',
				f'{method}_ratio',
				f'{method}_latency_seconds',
				f'{method}_throughput_mbps'
			])
		
		fieldnames = base_fields + tabcl_fields + baseline_fields
		
		# Append to existing file or create new
		file_exists = args.output.exists()
		with open(args.output, 'a' if file_exists else 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			if not file_exists:
				writer.writeheader()
			writer.writerows(all_results)
		
		print(f"\n✓ Wrote {len(all_results)} results to {args.output}")
	else:
		print("\n✗ No results to write")


if __name__ == "__main__":
	main()
