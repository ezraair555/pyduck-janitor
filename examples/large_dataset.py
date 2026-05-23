"""
Example: Processing large datasets with out-of-core computation

This example demonstrates using pyduck-janitor to process datasets
larger than memory by working directly with Parquet files.
"""

import pandas as pd
import numpy as np
from pyduck_janitor import DuckJanitor
import tempfile
import os

# Create a large sample dataset and save as Parquet
print("Creating sample Parquet dataset...")

np.random.seed(42)
n = 1_000_000  # 1 million rows

# Generate in chunks to avoid memory issues
chunks = []
chunk_size = 100_000

for i in range(n // chunk_size):
    chunk = pd.DataFrame({
        'transaction_id': range(i * chunk_size, (i + 1) * chunk_size),
        'date': pd.date_range('2023-01-01', periods=chunk_size, freq='1min'),
        'amount': np.random.exponential(100, chunk_size),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Services'], chunk_size),
        'region': np.random.choice(['North', 'South', 'East', 'West'], chunk_size),
        'customer_id': np.random.randint(1, 10000, chunk_size),
    })
    # Add some missing values
    chunk.loc[np.random.choice(chunk_size, 1000), 'amount'] = np.nan
    chunks.append(chunk)

large_df = pd.concat(chunks, ignore_index=True)

# Save to Parquet
temp_dir = tempfile.mkdtemp()
parquet_path = os.path.join(temp_dir, 'large_transactions.parquet')
large_df.to_parquet(parquet_path, index=False)

print(f"Created dataset: {len(large_df):,} rows")
print(f"Saved to: {parquet_path}")
print(f"File size: {os.path.getsize(parquet_path) / 1024 / 1024:.2f} MB")

# Load with DuckJanitor (lazy, out-of-core)
print("\n" + "="*60)
print("Loading with DuckJanitor (out-of-core)")
print("="*60)

dj = DuckJanitor.from_parquet(parquet_path)
print(f"Loaded: {dj}")

# Build cleaning pipeline (still lazy)
print("\n" + "="*60)
print("Building cleaning pipeline")
print("="*60)

cleaned = (
    dj
    .clean_names()
    .dropna(subset=['amount'])
    .filter_column('amount', 'amount > 10')
    .add_column('amount_category', """
        CASE 
            WHEN amount < 50 THEN 'small'
            WHEN amount < 200 THEN 'medium'
            ELSE 'large'
        END
    """)
)

print(f"Pipeline built (no data loaded into memory yet)")

# Show query plan
print(f"\nQuery plan:\n{cleaned.explain()}")

# Peek at data without full materialization
print("\n" + "="*60)
print("Peeking at first 10 rows")
print("="*60)

preview = cleaned.head(10)
print(preview)

# Aggregate without loading all data
print("\n" + "="*60)
print("Aggregating data (still out-of-core)")
print("="*60)

# Use SQL for aggregation
aggregated = cleaned.sql("""
    SELECT 
        category,
        region,
        COUNT(*) as transaction_count,
        AVG(amount) as avg_amount,
        SUM(amount) as total_amount
    FROM self
    GROUP BY category, region
    ORDER BY total_amount DESC
""")

result = aggregated.collect()
print(f"\nAggregated results ({len(result)} rows):")
print(result)

# Collect filtered subset
print("\n" + "="*60)
print("Collecting filtered subset")
print("="*60)

# Only collect high-value transactions
high_value = (
    cleaned
    .filter_column('amount', 'amount > 500')
    .collect()
)

print(f"High-value transactions: {len(high_value):,} rows")
print(high_value.head())

# Cleanup
os.remove(parquet_path)
os.rmdir(temp_dir)

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
print(f"\nKey benefits demonstrated:")
print(f"- Out-of-core processing (1M+ rows)")
print(f"- Lazy evaluation (pipeline built without loading data)")
print(f"- Query optimization (DuckDB optimizes entire pipeline)")
print(f"- SQL interoperability (mix janitor methods with SQL)")
