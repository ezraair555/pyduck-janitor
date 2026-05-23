"""
Example: Mixing pyduck-janitor methods with SQL queries

This example demonstrates the interoperability between
janitor cleaning methods and custom SQL queries.
"""

import pandas as pd
import numpy as np
from pyduck_janitor import DuckJanitor

# Create sample sales data
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'SaleDate': pd.date_range('2023-01-01', periods=n, freq='D'),
    'Product': np.random.choice(['Widget A', 'Widget B', 'Widget C'], n),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'SalesRep': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana'], n),
    'Quantity': np.random.randint(1, 100, n),
    'UnitPrice': np.random.uniform(10, 100, n),
    'Discount': np.random.uniform(0, 0.3, n),
})

# Add some missing values
data.loc[np.random.choice(n, 50), 'Quantity'] = np.nan

print("Original data:")
print(data.head())
print(f"\nShape: {data.shape}")

# Create DuckJanitor instance
dj = DuckJanitor.from_pandas(data)

# Example 1: Clean then SQL
print("\n" + "="*60)
print("Example 1: Clean data, then run custom SQL")
print("="*60)

cleaned_then_sql = (
    dj
    .clean_names()
    .dropna(subset=['quantity'])
    .sql("""
        SELECT 
            product,
            region,
            SUM(quantity * unitprice * (1 - discount)) as revenue,
            COUNT(*) as transaction_count
        FROM self
        GROUP BY product, region
        ORDER BY revenue DESC
    """)
)

result1 = cleaned_then_sql.collect()
print(f"\nRevenue by product and region:")
print(result1)

# Example 2: SQL then clean
print("\n" + "="*60)
print("Example 2: Run SQL, then clean")
print("="*60)

sql_then_clean = (
    dj
    .sql("""
        SELECT 
            salesrep,
            DATE_TRUNC('month', saledate) as sale_month,
            SUM(quantity * unitprice * (1 - discount)) as revenue
        FROM self
        WHERE quantity IS NOT NULL
        GROUP BY salesrep, DATE_TRUNC('month', saledate)
    """)
    .clean_names()
    .add_column('performance_tier', """
        CASE 
            WHEN revenue > 10000 THEN 'high'
            WHEN revenue > 5000 THEN 'medium'
            ELSE 'low'
        END
    """)
)

result2 = sql_then_clean.collect()
print(f"\nSales rep performance by month:")
print(result2.head(15))

# Example 3: Multiple SQL passes with cleaning
print("\n" + "="*60)
print("Example 3: Multiple SQL passes with cleaning in between")
print("="*60)

multi_pass = (
    dj
    .clean_names()
    .sql("""
        SELECT *,
            quantity * unitprice * (1 - discount) as net_revenue
        FROM self
    """)
    .filter_column('net_revenue', 'net_revenue > 100')
    .sql("""
        SELECT 
            salesrep,
            product,
            COUNT(*) as deals,
            SUM(net_revenue) as total_revenue,
            AVG(net_revenue) as avg_deal_size
        FROM self
        GROUP BY salesrep, product
    """)
    .rename_column('deals', 'deal_count')
    .add_column('revenue_per_deal', 'total_revenue / deal_count')
)

result3 = multi_pass.collect()
print(f"\nSales rep by product analysis:")
print(result3)

# Example 4: Window functions with cleaning
print("\n" + "="*60)
print("Example 4: Window functions in SQL with janitor cleaning")
print("="*60)

window_analysis = (
    dj
    .clean_names()
    .dropna(subset=['quantity'])
    .sql("""
        SELECT 
            *,
            SUM(quantity * unitprice * (1 - discount)) OVER (
                PARTITION BY salesrep 
                ORDER BY saledate
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as rolling_7day_revenue,
            AVG(quantity * unitprice * (1 - discount)) OVER (
                PARTITION BY region
                ORDER BY saledate
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) as rolling_30day_avg_revenue
        FROM self
    """)
    .filter_column('rolling_7day_revenue', 'rolling_7day_revenue > 500')
)

result4 = window_analysis.head(20)
print(f"\nRolling revenue analysis:")
print(result4[['saledate', 'salesrep', 'region', 'rolling_7day_revenue', 'rolling_30day_avg_revenue']])

# Example 5: CTEs with cleaning
print("\n" + "="*60)
print("Example 5: Common Table Expressions (CTEs) with cleaning")
print("="*60)

cte_example = (
    dj
    .clean_names()
    .sql("""
        WITH monthly_sales AS (
            SELECT 
                salesrep,
                DATE_TRUNC('month', saledate) as month,
                SUM(quantity * unitprice * (1 - discount)) as monthly_revenue
            FROM self
            WHERE quantity IS NOT NULL
            GROUP BY salesrep, DATE_TRUNC('month', saledate)
        ),
        rep_averages AS (
            SELECT 
                salesrep,
                AVG(monthly_revenue) as avg_monthly_revenue
            FROM monthly_sales
            GROUP BY salesrep
        )
        SELECT 
            m.salesrep,
            m.month,
            m.monthly_revenue,
            r.avg_monthly_revenue,
            (m.monthly_revenue - r.avg_monthly_revenue) / r.avg_monthly_revenue * 100 as pct_deviation
        FROM monthly_sales m
        JOIN rep_averages r ON m.salesrep = r.salesrep
        ORDER BY ABS(pct_deviation) DESC
    """)
    .add_column('performance_flag', """
        CASE 
            WHEN pct_deviation > 20 THEN 'exceptional'
            WHEN pct_deviation < -20 THEN 'concerning'
            ELSE 'normal'
        END
    """)
)

result5 = cte_example.collect()
print(f"\nSales rep performance deviations:")
print(result5.head(15))

print("\n" + "="*60)
print("Example completed successfully!")
print("="*60)
print("\nKey takeaways:")
print("- Mix janitor methods with custom SQL freely")
print("- Use 'self' to reference the current relation in SQL")
print("- SQL enables complex operations not in janitor API")
print("- Cleaning methods provide readable preprocessing")
