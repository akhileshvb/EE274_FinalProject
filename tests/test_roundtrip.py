import os
from pathlib import Path
import pandas as pd
import numpy as np

from tabcl.cli import compress_file, decompress_file


def test_roundtrip_tmp(tmp_path: Path):
	# synthetic categorical data with dependencies
	n = 200
	rng = np.random.default_rng(0)
	A = rng.integers(0, 5, size=n)
	B = (A + rng.integers(0, 2, size=n)) % 7
	C = rng.integers(0, 3, size=n)
	df = pd.DataFrame({"A": A, "B": B, "C": C})

	inp = tmp_path / "in.csv"
	outf = tmp_path / "out.tabcl"
	rest = tmp_path / "out.csv"

	inp.write_text(df.to_csv(index=False))
	compress_file(str(inp), str(outf), ",")
	decompress_file(str(outf), str(rest))
	df2 = pd.read_csv(rest)

	# categorical equality as strings/ints
	assert df.shape == df2.shape
	for c in df.columns:
		assert (df[c].astype(str).values == df2[c].astype(str).values).all()
