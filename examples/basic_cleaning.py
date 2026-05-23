"""
Example: Basic data cleaning pipeline with pyduck-janitor

This example demonstrates a typical data cleaning workflow
using the DuckJanitor API.
"""

import pandas as pd
import numpy as np
from pyduck_janitor import DuckJanitor

# Create sample messy data
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'Sales Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'] * 20,
    'Company1': np.random.uniform(100, 500, n),
    'Company2': np.random.uniform(150, 600, n),
    'Company3': np.random.uniform(200, 700, n),
    '  Region  ': np.random.choice(['North', 'South', 'East', 'West'], n),
    'Invalid!@#': np.random.choice(['A', 'B', 'C'], n),
})

# Add some missing values
data.loc[np.random.choice(n, 10), 'Company2'] = np.nan
data.loc[np.random.choice(n, 5), 'Company3'] = np.nan

# Add an empty row
data.loc[len(data)] = [None] * len(data.columns)

print("Original data:")
print(data.head(10))
print(f"\nShape: {data.shape}")
print(f"\nColumn names: {list(data.columns)}")

# Create DuckJanitor instance
print("\n" + "="*60)
print("Creating DuckJanitor instance")
print("="*60)

dj = DuckJanitor.from_pandas(data)
print(f"DuckJanitor created: {dj}")

# Build cleaning pipeline
print("\n" + "="*60)
print("Building cleaning pipeline")
print("="*60)

cleaned = (
    dj
    .clean_names()  # Standardize column names
    .remove_empty()  # Remove empty rows/columns
    .dropna(subset=['company2', 'company3'])  # Drop rows with missing values
    .rename_column('company1', 'acme_corp')  # Rename for clarity
    .add_column('total_sales', 'company1 + company2 + company3')  # Add calculated column
    .filter_column('total_sales', 'total_sales > 500')  # Filter high-value rows
)

print(f"Pipeline built (lazy evaluation)")
print(f"Query plan:\n{cleaned.explain()}")

# Execute and collect results
print("\n" + "="*60)
print("Executing pipeline")
print("="*60)

result = cleaned.collect()

print(f"\nCleaned data shape: {result.shape}")
print(f"\nCleaned column names: {list(result.columns)}")
print(f"\nFirst 10 rows:")
print(result.head(10))

# Additional transformations
print("\n" + "="*60)
print("Additional transformations")
print("="*60)

# Add categorical encoding
encoded = (
    DuckJanitor.from_pandas(result)
    .encode_categorical('region')
    .collect()
)

print(f"\nWith categorical encoding:")
print(encoded.head())

# One-hot encode
dummies = (
    DuckJanitor.from_pandas(result)
    .get_dummies('region', prefix='reg')
    .collect()
)

print(f"\nWith dummy variables:")
print(dummies.head())

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
