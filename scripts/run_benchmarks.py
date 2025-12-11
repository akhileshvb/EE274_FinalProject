import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def parse_bench_output(output: str) -> Dict[str, Optional[float]]:
    """Parse the output from bench.py to extract metrics."""
    results = {}
    
    # Extract original size
    orig_match = re.search(r"Original size:\s+([\d.]+)\s+(\w+)", output)
    if orig_match:
        size_val = float(orig_match.group(1))
        unit = orig_match.group(2)
        # Convert to bytes
        multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        results["original_size_bytes"] = size_val * multipliers.get(unit, 1)
        print(f"  Parsed original_size_bytes: {results['original_size_bytes']}", file=sys.stderr)
    else:
        print(f"  WARNING: Could not parse original size", file=sys.stderr)
    
    # Extract metrics for each method
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    
    for method in methods:
        # Check if method was not found/available first
        if f"{method}: not found" in output or f"{method}: not available" in output:
            results[f"{method}_size_bytes"] = None
            results[f"{method}_ratio"] = None
            results[f"{method}_time"] = None
            continue
        
        # Find the section for this method - more flexible pattern
        # Pattern: method_name:\n  size:  X.XX UNIT  (ratio X.XX)\n  time:  X.XXs
        # Handle both "columnar gzip" (with space) and single-word methods
        method_pattern = method.replace(" ", r"\s+")
        pattern = rf"{method_pattern}:\s*\n\s*size:\s+([\d.]+)\s+(\w+)\s+\(ratio\s+([\d.]+|-)x?\)\s*\n\s*time:\s+([\d.]+)s"
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            print(f"  Matched {method} with primary pattern", file=sys.stderr)
            size_val = float(match.group(1))
            size_unit = match.group(2)
            ratio_str = match.group(3)
            time_val = float(match.group(4))
            
            # Convert size to bytes
            multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
            size_bytes = size_val * multipliers.get(size_unit, 1)
            
            results[f"{method}_size_bytes"] = size_bytes
            results[f"{method}_ratio"] = float(ratio_str) if ratio_str != "-" else None
            results[f"{method}_time"] = time_val
        else:
            # Try alternative pattern without ratio in parentheses
            alt_pattern = rf"{method_pattern}:\s*\n\s*size:\s+([\d.]+)\s+(\w+).*?ratio\s+([\d.]+|-)x?.*?\n\s*time:\s+([\d.]+)s"
            alt_match = re.search(alt_pattern, output, re.MULTILINE | re.DOTALL)
            if alt_match:
                print(f"  Matched {method} with alternative pattern", file=sys.stderr)
                size_val = float(alt_match.group(1))
                size_unit = alt_match.group(2)
                ratio_str = alt_match.group(3)
                time_val = float(alt_match.group(4))
                
                multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
                size_bytes = size_val * multipliers.get(size_unit, 1)
                
                results[f"{method}_size_bytes"] = size_bytes
                results[f"{method}_ratio"] = float(ratio_str) if ratio_str != "-" else None
                results[f"{method}_time"] = time_val
            else:
                # If we can't parse, set to None
                print(f"  WARNING: Could not parse {method} - no pattern matched", file=sys.stderr)
                results[f"{method}_size_bytes"] = None
                results[f"{method}_ratio"] = None
                results[f"{method}_time"] = None
    
    return results


def run_benchmark(input_file: Path, delimiter: str = ",", zstd_level: int = 19) -> Dict[str, Optional[float]]:
    """Run bench.py on a single file and return parsed results."""
    script_dir = Path(__file__).parent
    bench_script = script_dir / "bench.py"
    
    cmd = [
        sys.executable,
        str(bench_script),
        "--input", str(input_file),
        "--delimiter", delimiter,
        "--zstd-level", str(zstd_level),
        "--mi-mode", "auto",
    ]
    
    print(f"Running benchmark on {input_file.name}...", file=sys.stderr)
    print(f"Command: {' '.join(cmd)}", file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout + result.stderr
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Raw output from bench.py for {input_file.name}:", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(output, file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        results = parse_bench_output(output)
        print(f"Parsed results for {input_file.name}:", file=sys.stderr)
        for key, value in results.items():
            print(f"  {key}: {value}", file=sys.stderr)
        print(file=sys.stderr)
        results["dataset"] = input_file.stem
        results["input_file"] = str(input_file)
        return results
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark on {input_file}: {e}", file=sys.stderr)
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Error output from bench.py for {input_file.name}:", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"stdout:\n{e.stdout}", file=sys.stderr)
        print(f"stderr:\n{e.stderr}", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)
        return {"dataset": input_file.stem, "input_file": str(input_file)}


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on all datasets and write CSV")
    parser.add_argument("--datasets-dir", type=Path, default=Path(__file__).parent.parent / "datasets",
                       help="Directory containing datasets")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent.parent / "benchmark_results.csv",
                       help="Output CSV file")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default ',')")
    parser.add_argument("--zstd-level", type=int, default=19, help="zstd compression level")
    args = parser.parse_args()
    
    # Find all CSV files in dataset directories
    datasets_dir = args.datasets_dir
    csv_files = []
    
    # Define dataset files explicitly
    dataset_files = [
        datasets_dir / "census_income" / "adult_train.csv",
        datasets_dir / "kaggle-display-advertising-challenge-dataset" / "test_sample_10k.txt",
        datasets_dir / "yellow_cab" / "yellow_tripdata_2025-01_updated.csv",
    ]
    
    # Check which files exist
    for file_path in dataset_files:
        if file_path.exists():
            csv_files.append(file_path)
        else:
            print(f"Warning: {file_path} not found, skipping", file=sys.stderr)
    
    if not csv_files:
        print("No dataset files found!", file=sys.stderr)
        sys.exit(1)
    
    # Run benchmarks
    all_results = []
    for csv_file in csv_files:
        # Determine delimiter based on file extension
        delimiter = "\t" if csv_file.suffix == ".txt" else args.delimiter
        results = run_benchmark(csv_file, delimiter=delimiter, zstd_level=args.zstd_level)
        all_results.append(results)
    
    # Write CSV
    if not all_results:
        print("No results to write!", file=sys.stderr)
        sys.exit(1)
    
    # Get all unique keys from all results
    all_keys = set()
    for result in all_results:
        all_keys.update(result.keys())
    
    # Define column order
    methods = ["tabcl", "gzip", "zstd", "bzip2", "columnar gzip"]
    columns = ["dataset", "input_file", "original_size_bytes"] + \
              [f"{method}_{metric}" for method in methods
               for metric in ["size_bytes", "ratio", "time"]]
    
    # Filter to only columns that exist
    columns = [c for c in columns if c in all_keys]
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in all_results:
            # Only include columns that exist
            row = {k: result.get(k) for k in columns}
            writer.writerow(row)
    
    print(f"Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

