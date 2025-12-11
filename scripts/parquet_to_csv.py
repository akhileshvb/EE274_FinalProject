import argparse
import pandas as pd
from pathlib import Path


def convert_parquet_to_csv(input_path: str, output_path: str | None = None, delimiter: str = ",") -> None:
	"""
	Convert a parquet file to CSV.
	
	Args:
		input_path: Path to input parquet file
		output_path: Path to output CSV file (default: same as input but with .csv extension)
		delimiter: CSV delimiter (default: comma)
	"""
	input_file = Path(input_path)
	if not input_file.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")
	
	if output_path is None:
		output_file = input_file.with_suffix(".csv")
	else:
		output_file = Path(output_path)
	
	print(f"Reading parquet file: {input_file}")
	df = pd.read_parquet(input_file)
	
	print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
	print(f"Columns: {list(df.columns)}")
	
	print(f"Writing CSV file: {output_file}")
	df.to_csv(output_file, index=False, sep=delimiter)
	
	print(f"Done! Output file size: {output_file.stat().st_size / (1024*1024):.2f} MB")


def main():
	ap = argparse.ArgumentParser(description="Convert parquet file to CSV")
	ap.add_argument("--input", required=True, help="Input parquet file path")
	ap.add_argument("--output", default=None, help="Output CSV file path (default: same as input with .csv extension)")
	ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: comma)")
	
	args = ap.parse_args()
	convert_parquet_to_csv(args.input, args.output, args.delimiter)


if __name__ == "__main__":
	main()

