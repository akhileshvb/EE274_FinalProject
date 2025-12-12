# EE274 Final Project: Tabular Compression via MDL-weighted Chow–Liu using OpenZL

Modern applications generate massive amounts of tabular data (e.g., logs, telemetry, financial records). While general-purpose compressors like gzip or zstd are robust, they ignore statistical dependencies across columns, leaving compression potential untapped. In contrast, hand-crafted database encodings are often ad hoc and inflexible.

This repository provides structure-aware lossless compression for tabular data using the **Minimum Description Length (MDL) principle** and Chow–Liu trees. By modeling statistical relationships between columns, we learn a graphical model that captures high mutual-information dependencies, then compress root columns independently and encode others conditionally on their parents.

We implement two compression schemes: (1) a **histogram-based MDL-adjusted Chow–Liu compressor** building on Pavlichin et al. (DCC 2017), with algorithmic and systems-level optimizations including two-phase mutual information estimation with adaptive sampling, vectorized operations, parallelization, and specialized encoding schemes; and (2) a **lightweight MLP-based extension** that learns conditional distributions for high-cardinality dependencies via probability-weighted rank encoding. For most encoders, the implementation uses the OpenZL Python API.
<!-- 
- OpenZL repo: [facebook/openzl](https://github.com/facebook/openzl)  
- OpenZL docs: [openzl.org](https://openzl.org/) -->

## Quickstart

```bash
conda create -n tabcl python=3.11
conda activate tabcl
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

<!-- If you built OpenZL from source, make sure this works inside the `tabcl` environment:

```python
>>> import openzl.ext as zl
``` -->

## CLI

The `tabcl` entrypoint provides a simple command‑line interface.

```bash
# Compress CSV
tabcl compress --input data.csv --output data.tabcl --delimiter ,

# Decompress
tabcl decompress --input data.tabcl --output restored.csv
```

By default, `tabcl`:

1. Loads the CSV and tokenizes each column (numeric vs categorical).  
2. Estimates MDL‑adjusted mutual information for all column pairs and learns a maximum‑weight forest (Chow–Liu style).  
3. Encodes root columns independently and child columns conditionally on their parents using histogram- and OpenZL‑based codecs.  
4. Serializes the model (columns, forest, dictionaries) and per‑column frames into a single `.tabcl` file.

### Example: Compressing US Census 1990 Data

```bash
# Compress using histogram-based approach (default)
tabcl compress --input datasets/us_census_data_1990/USCensus1990_FULL.csv --output us_census.tabcl --delimiter ,

# Decompress
tabcl decompress --input us_census.tabcl --output us_census_restored.csv
```

You can control a few knobs via the CLI, e.g.:

- `--mi-mode {exact,hashed,auto}`: how mutual information is estimated.  
- `--rare-threshold`: how aggressively to factor out rare categorical values.  
- `--workers`: number of threads used during MI and encoding.

See `scripts/bench.py` for examples of running consistent benchmarks across multiple compressors.

## MLP Extensions

The repository includes two MLP-based extensions to the histogram approach:

### Conditional MLP (`--use-mlp`)

The conditional MLP extension (`src/tabcl/mlp_conditional.py`) trains a small neural network for each child column to learn conditional distributions `p(child | parent)` instead of using histograms. This can be beneficial when parent columns have high cardinality, as the MLP model cost is fixed while histogram cost scales with the number of parent values.

```bash
# Compress with conditional MLP
tabcl compress --input data.csv --output data.tabcl --use-mlp
```

The MLP extension uses probability-weighted rank encoding: for each row, it sorts child values by MLP-assigned probability and encodes the rank of the actual value. When the MLP assigns high probability to correct values, ranks are small (mostly 0s and 1s) and compress well. The system automatically selects MLP vs histogram for each column based on MDL comparison.

### Autoregressive MLP (`--use-mlp-autoregressive`)

The autoregressive MLP extension (`src/tabcl/mlp_autoregressive.py`) implements fully autoregressive compression: `p(x_j | x_<j)` for all previous columns. This extension is included in the repository but is not discussed in detail in the project report.

```bash
# Compress with autoregressive MLP
tabcl compress --input data.csv --output data.tabcl --use-mlp-autoregressive
```

Note: The autoregressive MLP can be significantly slower than the conditional MLP due to training models for each column with increasing context size.

## Testing

The repository includes a comprehensive test suite covering roundtrip compression, tree recovery, and MLP functionality. Run all tests with:

```bash
pytest tests/
```

Or run specific test files:

```bash
pytest tests/test_roundtrip.py
pytest tests/test_mlp.py
pytest tests/test_tree_recovery.py
```

Tests require PyTorch for MLP-related functionality; if PyTorch is not installed, MLP tests will be skipped automatically.
