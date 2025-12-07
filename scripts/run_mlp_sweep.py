import argparse
import csv
import sys
from pathlib import Path

# Import bench_mlp functions
sys.path.insert(0, str(Path(__file__).parent))
from bench_mlp import benchmark_file
from run_base_sweep import find_datasets


def main():
	ap = argparse.ArgumentParser(description="Run bench_mlp.py on all datasets")
	ap.add_argument("--datasets-dir", type=Path, default=Path("datasets"),
	                help="Directory containing datasets (default: datasets)")
	ap.add_argument("--output", type=Path, default=Path("results/mlp_sweep.csv"),
	                help="Output CSV file (default: results/mlp_sweep.csv)")
	ap.add_argument("--skip-existing", action="store_true",
	                help="Skip datasets already in results file")
	args = ap.parse_args()
	
	# Create output directory
	args.output.parent.mkdir(parents=True, exist_ok=True)
	
	# Find all datasets (reuse the same logic as run_base_sweep.py)
	datasets = [ds for ds in find_datasets(args.datasets_dir) if "yellow_tripdata" not in ds.name]
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
		
		# Run benchmark
		try:
			results = benchmark_file(dataset_path, verbose=False)
			
			# Extract results
			no_mlp = results.get("no_mlp", {})
			mlp = results.get("mlp", {})
			
			# Build result row
			row = {
				"dataset": dataset_name,
				"dataset_path": str(dataset_path),
				"original_size_bytes": no_mlp.get("original_size", 0),
				"no_mlp_size_bytes": no_mlp.get("compressed_size", 0) if no_mlp.get("success") else 0,
				"no_mlp_ratio": no_mlp.get("compression_ratio", 0.0) if no_mlp.get("success") else 0.0,
				"no_mlp_latency_seconds": no_mlp.get("compress_time", 0.0) if no_mlp.get("success") else 0.0,
				"no_mlp_roundtrip": no_mlp.get("roundtrip_ok", False) if no_mlp.get("success") else False,
				"mlp_size_bytes": mlp.get("compressed_size", 0) if mlp.get("success") else 0,
				"mlp_ratio": mlp.get("compression_ratio", 0.0) if mlp.get("success") else 0.0,
				"mlp_latency_seconds": mlp.get("compress_time", 0.0) if mlp.get("success") else 0.0,
				"mlp_roundtrip": mlp.get("roundtrip_ok", False) if mlp.get("success") else False,
				"size_improvement_bytes": (no_mlp.get("compressed_size", 0) - mlp.get("compressed_size", 0)) if (no_mlp.get("success") and mlp.get("success")) else 0,
				"size_improvement_pct": ((no_mlp.get("compressed_size", 0) - mlp.get("compressed_size", 0)) / no_mlp.get("compressed_size", 1) * 100) if (no_mlp.get("success") and mlp.get("success") and no_mlp.get("compressed_size", 0) > 0) else 0.0,
				"time_overhead_seconds": (mlp.get("compress_time", 0.0) - no_mlp.get("compress_time", 0.0)) if (no_mlp.get("success") and mlp.get("success")) else 0.0,
				"time_overhead_pct": ((mlp.get("compress_time", 0.0) - no_mlp.get("compress_time", 0.0)) / no_mlp.get("compress_time", 1.0) * 100) if (no_mlp.get("success") and mlp.get("success") and no_mlp.get("compress_time", 0.0) > 0) else 0.0,
				"no_mlp_success": no_mlp.get("success", False),
				"mlp_success": mlp.get("success", False),
				"no_mlp_error": no_mlp.get("error", "") if not no_mlp.get("success") else "",
				"mlp_error": mlp.get("error", "") if not mlp.get("success") else "",
			}
			
			all_results.append(row)
			
			# Print summary
			if no_mlp.get("success") and mlp.get("success"):
				size_improvement = row["size_improvement_pct"]
				time_overhead = row["time_overhead_pct"]
				print(f"  ✓ No MLP: {row['no_mlp_ratio']:.2f}x ratio, {row['no_mlp_latency_seconds']:.2f}s")
				print(f"  ✓ MLP: {row['mlp_ratio']:.2f}x ratio, {row['mlp_latency_seconds']:.2f}s")
				print(f"  Size improvement: {size_improvement:+.2f}%, Time overhead: {time_overhead:+.2f}%")
			elif not no_mlp.get("success"):
				print(f"  ✗ No MLP failed: {no_mlp.get('error', 'Unknown error')}")
			elif not mlp.get("success"):
				print(f"  ✗ MLP failed: {mlp.get('error', 'Unknown error')}")
		
		except Exception as e:
			print(f"  ✗ Error processing {dataset_name}: {e}")
			all_results.append({
				"dataset": dataset_name,
				"dataset_path": str(dataset_path),
				"no_mlp_success": False,
				"mlp_success": False,
				"no_mlp_error": str(e),
				"mlp_error": str(e),
			})
	
	# Write results to CSV
	fieldnames = [
		"dataset", "dataset_path", "original_size_bytes",
		"no_mlp_size_bytes", "no_mlp_ratio", "no_mlp_latency_seconds", "no_mlp_roundtrip", "no_mlp_success", "no_mlp_error",
		"mlp_size_bytes", "mlp_ratio", "mlp_latency_seconds", "mlp_roundtrip", "mlp_success", "mlp_error",
		"size_improvement_bytes", "size_improvement_pct", "time_overhead_seconds", "time_overhead_pct",
	]
	
	# Write or append to CSV
	file_exists = args.output.exists()
	with open(args.output, 'a' if file_exists and args.skip_existing else 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if not file_exists or not args.skip_existing:
			writer.writeheader()
		writer.writerows(all_results)
	
	print(f"\n{'='*60}")
	print(f"Results written to: {args.output}")
	print(f"Processed {len(all_results)} datasets")
	print(f"{'='*60}\n")


if __name__ == "__main__":
	main()
