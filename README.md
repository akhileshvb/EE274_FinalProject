# EE274 Final Project: Tabular Compression via MDL-weighted Chow–Liu using OpenZL

Implements “Compressing Tabular Data via Pairwise Dependencies” (DCC 2017) with an MDL-adjusted Chow–Liu forest, wired to the OpenZL Python API.

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

If you built OpenZL from source, ensure `import openzl.ext as zl` works in this venv.

## CLI

```bash
# Compress CSV
 tabcl compress --input data.csv --output data.tabcl --delimiter ,

# Decompress
 tabcl decompress --input data.tabcl --output restored.csv
```

Initially, compression uses a generic OpenZL bytes backend and stores a learned MDL-weighted forest alongside the payload. Subsequent iterations can refine column-wise coders using the forest.
