import pandas as pd
import sys

df = pd.read_csv(sys.argv[1])
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nColumn types:")
print(df.dtypes)
print(f"\nUnique value counts per column:")
for c in df.columns:
    nunique = df[c].nunique()
    nrows = len(df)
    pct = 100.0 * nunique / nrows if nrows > 0 else 0
    print(f"  {c:20s}: {nunique:6d} unique / {nrows:6d} rows ({pct:5.2f}%)")
    
print(f"\nValue frequency analysis (top values per column):")
for c in df.columns[:5]:  # First 5 columns
    print(f"\n{c}:")
    vc = df[c].value_counts()
    print(f"  Most common: {vc.head(3).to_dict()}")

print(f"\nNumeric column ranges:")
for c in df.columns:
    if pd.api.types.is_numeric_dtype(df[c]):
        print(f"  {c}: min={df[c].min()}, max={df[c].max()}, mean={df[c].mean():.2f}")

