"""
Example: Performance comparison between pandas+janitor and pyduck-janitor

This example benchmarks the performance of both approaches.
"""

import pandas as pd
import numpy as np
import time
from pyduck_janitor import DuckJanitor

# Try to import janitor for comparison
try:
    import janitor
    HAS_JANITOR = True
except ImportError:
    HAS_JANITOR = False
    print("pyjanitor not installed. Install with: pip install pyjanitor")

# Create test dataset
print("Creating test dataset...")
np.random.seed(42)
n = 100_000

data = pd.DataFrame({
    'Sales Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'] * (n // 6),
    'Company1': np.random.uniform(100, 500, n),
    'Company2': np.random.uniform(150, 600, n),
    'Company3': np.random.uniform(200, 700, n),
    '  Region  ': np.random.choice(['North', 'South', 'East', 'West'], n),
    'Category': np.random.choice(['A', 'B', 'C', 'D'], n),
})

# Add missing values
data.loc[np.random.choice(n, 5000), 'Company2'] = np.nan
data.loc[np.random.choice(n, 3000), 'Company3'] = np.nan

print(f"Dataset size: {n:,} rows")
print(f"Missing values: {data.isnull().sum().sum():,}")

# Define cleaning pipeline
def pandas_pipeline(df):
    """Standard pandas + pyjanitor pipeline."""
    if HAS_JANITOR:
        return (
            df
            .clean_names()
            .remove_empty()
            .dropna(subset=['company2', 'company3'])
            .rename_column('company1', 'acme_corp')
            .assign(total_sales=lambda x: x['acme_corp'] + x['company2'] + x['company3'])
            .query('total_sales > 500')
        )
    else:
        # Fallback without janitor
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip('_')
        df = df.dropna(subset=['company2', 'company3'])
        df = df.rename(columns={'company1': 'acme_corp'})
        df['total_sales'] = df['acme_corp'] + df['company2'] + df['company3']
        df = df[df['total_sales'] > 500]
        return df

def duck_janitor_pipeline(df):
    """pyduck-janitor pipeline."""
    return (
        DuckJanitor.from_pandas(df)
        .clean_names()
        .remove_empty()
        .dropna(subset=['company2', 'company3'])
        .rename_column('company1', 'acme_corp')
        .add_column('total_sales', 'company1 + company2 + company3')
        .filter_column('total_sales', 'total_sales > 500')
        .collect()
    )

# Benchmark pandas
print("\n" + "="*60)
print("Benchmarking pandas + pyjanitor")
print("="*60)

if HAS_JANITOR:
    start = time.time()
    pandas_result = pandas_pipeline(data)
    pandas_time = time.time() - start
    print(f"Time: {pandas_time:.3f} seconds")
    print(f"Result shape: {pandas_result.shape}")
else:
    print("Skipping (pyjanitor not installed)")
    pandas_time = None

# Benchmark pyduck-janitor
print("\n" + "="*60)
print("Benchmarking pyduck-janitor")
print("="*60)

start = time.time()
duck_result = duck_janitor_pipeline(data)
duck_time = time.time() - start
print(f"Time: {duck_time:.3f} seconds")
print(f"Result shape: {duck_result.shape}")

# Compare
print("\n" + "="*60)
print("Comparison")
print("="*60)

if pandas_time:
    speedup = pandas_time / duck_time
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time saved: {pandas_time - duck_time:.3f} seconds")
else:
    print("Cannot compare (pyjanitor not installed)")

# Verify results are equivalent
print("\n" + "="*60)
print("Result verification")
print("="*60)

# Compare shapes
print(f"Pandas shape: {pandas_result.shape if pandas_time else 'N/A'}")
print(f"DuckJanitor shape: {duck_result.shape}")

# Compare column names
if pandas_time:
    print(f"\nColumns match: {list(pandas_result.columns) == list(duck_result.columns)}")

print("\n" + "="*60)
print("Example completed!")
print("="*60)

# Additional benchmark: Out-of-core
print("\n" + "="*60)
print("Bonus: Out-of-core processing")
print("="*60)

import tempfile
import os

# Save to Parquet
temp_dir = tempfile.mkdtemp()
parquet_path = os.path.join(temp_dir, 'benchmark.parquet')
data.to_parquet(parquet_path, index=False)

print(f"Processing {n:,} rows from Parquet file...")

start = time.time()
result = (
    DuckJanitor.from_parquet(parquet_path)
    .clean_names()
    .dropna(subset=['company2', 'company3'])
    .filter_column('company1', 'company1 > 300')
    .collect()
)
parquet_time = time.time() - start

print(f"Time (from Parquet): {parquet_time:.3f} seconds")
print(f"Result shape: {result.shape}")

# Cleanup
os.remove(parquet_path)
os.rmdir(temp_dir)

print("\nNote: Out-of-core processing uses less memory and can handle")
print("datasets larger than available RAM!")
