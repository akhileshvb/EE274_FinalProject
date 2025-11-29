# EE274 Final Project: Tabular Compression via MDL-weighted Chow–Liu using OpenZL

This repository implements the core ideas of “Compressing Tabular Data via Pairwise Dependencies” (DCC 2017) using an MDL‑adjusted Chow–Liu forest, wired to the OpenZL Python API. The main focus is a practical implementation of the MDL criterion and forest learning for CSV files; the more experimental neural/MLP code is included as an optional aside.

- OpenZL repo: [facebook/openzl](https://github.com/facebook/openzl)  
- OpenZL docs: [openzl.org](https://openzl.org/)  
- Python quick start: [openzl.org/getting-started/examples/py/quick-start/](https://openzl.org/getting-started/examples/py/quick-start/)

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

If you built OpenZL from source, make sure this works inside your virtualenv:

```python
>>> import openzl.ext as zl
```

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

You can control a few knobs via the CLI, e.g.:

- `--mi-mode {exact,hashed,auto}`: how mutual information is estimated.  
- `--rare-threshold`: how aggressively to factor out rare categorical values.  
- `--workers`: number of threads used during MI and encoding.

See `scripts/bench.py` for examples of running consistent benchmarks across multiple compressors.

## Optional MLP / Neural Extensions

The `src/tabcl/tiny_mlp.py`, `src/tabcl/mlp_conditional.py`, and `src/tabcl/neural_compression.py` modules contain an experimental MLP‑based extension that tries to replace some histogram models with small neural networks (TinyMLP). In our experiments this did **not** consistently improve end‑to‑end compression compared to the baseline MDL‑weighted Chow–Liu + histogram approach, so it is not enabled by default and is best viewed as exploratory code.

We keep this code in the repository for completeness and as a starting point for future work on learned tabular compression, but the primary, well‑tested path is the MDL‑adjusted Chow–Liu compressor described above.
