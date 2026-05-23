# Hard To Do Functions - Implementation Plans

This document provides detailed implementation plans for the 15 "hard to do" pyjanitor functions that require hybrid approaches or have no direct SQL equivalent.

---

## 1. `also(func)` - Side Effects in Method Chain

**Challenge:** SQL is declarative and has no side effects. The `also()` function allows arbitrary Python functions to be applied mid-chain for side effects (logging, saving to disk, etc.).

**Current pyjanitor API:**
```python
df.also(lambda df: print(f"Shape: {df.shape}"))
df.also(lambda df: df.to_csv("midpoint.csv"))
```

**Proposed Implementation:**
```python
def also(self, func: Callable) -> 'DuckJanitor':
    """
    Apply a Python function with side effects.
    
    Note: This breaks lazy evaluation and materializes the data.
    """
    # Materialize current state
    df = self.collect()
    
    # Apply function (side effects happen here)
    result = func(df)
    
    # If function returns DataFrame, use it; otherwise use original
    if isinstance(result, pd.DataFrame):
        return DuckJanitor.from_pandas(result)
    return DuckJanitor.from_pandas(df)
```

**Trade-offs:**
- ✅ Works with any Python function
- ❌ Breaks lazy evaluation
- ❌ Materializes entire dataset

**Recommendation:** Implement as-is. Document that it breaks lazy evaluation.

---

## 2. `alias(alias)` - Series Renaming

**Challenge:** DuckDB doesn't have a Series concept like pandas. This is a Series-level operation.

**Current pyjanitor API:**
```python
series.alias("new_name")
series.alias(str.upper)  # callable
```

**Proposed Implementation:**
```python
# Not implemented as DuckJanitor method
# Users should use: df.rename_column() instead
```

**Trade-offs:**
- ✅ Column renaming already works via `rename_column()`
- ❌ Series operations not supported

**Recommendation:** Don't implement. Document that users should use `rename_column()` for DataFrames.

---

## 3. `compare_df_cols(df1, df2)` - Compare Two DataFrames

**Challenge:** Requires comparing column names and types across two separate DataFrames.

**Current pyjanitor API:**
```python
compare_df_cols(df1, df2)
```

**Proposed Implementation:**
```python
@staticmethod
def compare_df_cols(dj1: 'DuckJanitor', dj2: 'DuckJanitor') -> pd.DataFrame:
    """Compare columns between two DuckJanitor instances."""
    # Get column info from both
    cols1 = [(col, str(dtype)) for col, dtype in dj1._relation.dtypes.items()]
    cols2 = [(col, str(dtype)) for col, dtype in dj2._relation.dtypes.items()]
    
    # Compare in Python
    # Return comparison DataFrame
    ...
```

**Trade-offs:**
- ✅ Useful for debugging
- ❌ Static method, not part of method chain
- ❌ Requires accessing both relations

**Recommendation:** Implement as static method or utility function.

---

## 4. `conditional_join(right, on, how='inner')` - Complex Joins

**Challenge:** Supports non-equi joins (e.g., `df1.col1 > df2.col2`). DuckDB supports this, but API mapping is complex.

**Current pyjanitor API:**
```python
df.conditional_join(
    df2,
    ('col1', 'col2', '>='),  # non-equi condition
    ('col3', 'col4', '=='),
    how='inner'
)
```

**Proposed Implementation:**
```python
def conditional_join(self, other: 'DuckJanitor', on: List[tuple],
                     how: str = 'inner') -> 'DuckJanitor':
    """
    Perform conditional (non-equi) joins.
    
    Parameters
    ----------
    other : DuckJanitor
        Right DataFrame
    on : list of tuples
        Join conditions: [(left_col, right_col, operator), ...]
    """
    # Build WHERE clause from conditions
    conditions = []
    for left_col, right_col, op in on:
        conditions.append(f'self."{left_col}" {op} other."{right_col}"')
    
    where_clause = ' AND '.join(conditions)
    
    # Register both relations
    temp_self = f"_self_{id(self._relation)}"
    temp_other = f"_other_{id(other._relation)}"
    self._connection.register(temp_self, self._relation)
    self._connection.register(temp_other, other._relation)
    
    query = f"""
        SELECT * FROM {temp_self} self
        {how.upper()} JOIN {temp_other} other
        ON {where_clause}
    """
    
    new_relation = self._connection.execute(query)
    return DuckJanitor(new_relation, self._connection)
```

**Trade-offs:**
- ✅ DuckDB supports non-equi joins
- ✅ Pure SQL implementation possible
- ❌ Complex API

**Recommendation:** Implement. This is actually doable in pure SQL!

---

## 5. `convert_date()` - Matlab/Excel/Unix Epochs

**Challenge:** Converting from non-standard date formats (Matlab serial dates, Excel serial dates, Unix timestamps) requires custom epoch math.

**Current pyjanitor API:**
```python
df.convert_date('excel_date', format='excel')
df.convert_date('matlab_date', format='matlab')
df.convert_date('unix_timestamp', format='unix')
```

**Proposed Implementation:**
```python
def convert_date(self, column: str, date_format: Optional[str] = None) -> 'DuckJanitor':
    """
    Convert various date formats to Python datetime.
    
    Supported formats:
    - None: Auto-detect / standard date strings
    - 'excel': Excel serial dates (days since 1899-12-30)
    - 'matlab': Matlab serial dates (days since 0000-01-01)
    - 'unix': Unix timestamps (seconds since 1970-01-01)
    """
    conn = self._connection
    
    if date_format == 'excel':
        # Excel epoch: 1899-12-30
        convert_expr = f"""
            TIMESTAMP '1899-12-30' + INTERVAL '{column}' DAY
            AS "{column}"
        """
    elif date_format == 'matlab':
        # Matlab epoch: 0000-01-01
        convert_expr = f"""
            TIMESTAMP '0000-01-01' + INTERVAL '{column}' DAY
            AS "{column}"
        """
    elif date_format == 'unix':
        # Unix timestamp: seconds since 1970-01-01
        convert_expr = f"""
            TO_TIMESTAMP({column}) AS "{column}"
        """
    else:
        # Standard date conversion
        convert_expr = f'CAST("{column}" AS DATE) AS "{column}"'
    
    # Build SELECT
    old_columns = self._relation.columns
    select_parts = [f'"{col}"' for col in old_columns if col != column]
    select_parts.append(convert_expr)
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    new_relation = conn.execute(query)
    
    return DuckJanitor(new_relation, self._connection)
```

**Trade-offs:**
- ✅ DuckDB has excellent date/time support
- ✅ Pure SQL implementation
- ❌ Need to handle edge cases (leap years, etc.)

**Recommendation:** Implement! Move from "Hard" to "Can Do" category.

---

## 6. `drop_duplicate_columns()` - Remove Duplicate Content Columns

**Challenge:** Need to compare column contents (not just names) to find duplicates.

**Current pyjanitor API:**
```python
df.drop_duplicate_columns()
```

**Proposed Implementation:**
```python
def drop_duplicate_columns(self) -> 'DuckJanitor':
    """Remove columns that are exact duplicates of other columns."""
    # This requires comparing column contents
    # Best done in Python after materialization
    
    df = self.collect()
    
    # Find duplicate columns
    unique_cols = []
    seen_hashes = set()
    
    for col in df.columns:
        # Create hash of column values
        col_hash = hash(tuple(df[col].fillna('__NA__')))
        if col_hash not in seen_hashes:
            seen_hashes.add(col_hash)
            unique_cols.append(col)
    
    # Keep only unique columns
    return DuckJanitor.from_pandas(df[unique_cols])
```

**Trade-offs:**
- ✅ Removes redundancy
- ❌ Requires full materialization
- ❌ O(n*m) complexity for n columns, m rows

**Recommendation:** Implement with materialization. Document performance implications.

---

## 7. `dropnotnull()` - Inverse of dropna

**Challenge:** Keep rows where values ARE null (inverse of dropna).

**Current pyjanitor API:**
```python
df.dropnotnull(subset=['col1', 'col2'])
```

**Proposed Implementation:**
```python
def dropnotnull(self, subset: Optional[Union[str, List[str]]] = None,
                how: str = 'any') -> 'DuckJanitor':
    """
    Remove rows where values are NOT null (keep nulls).
    Inverse of dropna().
    """
    if subset is None:
        subset = self._relation.columns
    elif isinstance(subset, str):
        subset = [subset]
    
    # Invert the dropna logic
    if how == 'any':
        # Keep rows where ANY column IS NULL
        conditions = [f'"{col}" IS NULL' for col in subset]
        where_clause = ' OR '.join(conditions)
    elif how == 'all':
        # Keep rows where ALL columns IS NULL
        conditions = [f'"{col}" IS NULL' for col in subset]
        where_clause = ' AND '.join(conditions)
    
    query = f"SELECT * FROM relation WHERE {where_clause}"
    new_relation = self._connection.execute(query)
    
    return DuckJanitor(new_relation, self._connection)
```

**Trade-offs:**
- ✅ Simple SQL implementation
- ✅ Just invert dropna logic

**Recommendation:** Implement! This is actually easy - move to "Can Do" category.

---

## 8. `explode_index()` - MultiIndex to Columns

**Challenge:** DuckDB doesn't have MultiIndex concept.

**Current pyjanitor API:**
```python
df.explode_index(axis=1)  # or axis=0
```

**Proposed Implementation:**
```python
# Not applicable to DuckDB
# DuckDB relations don't have indexes like pandas
```

**Trade-offs:**
- ❌ No equivalent in DuckDB
- ❌ Pandas-specific concept

**Recommendation:** Don't implement. Document as pandas-specific feature.

---

## 9. `factorize_columns(columns)` - Numerical Encoding

**Challenge:** Similar to `encode_categorical()` but returns integer codes.

**Current pyjanitor API:**
```python
df.factorize_columns(['category1', 'category2'])
```

**Proposed Implementation:**
```python
def factorize_columns(self, columns: Union[str, List[str]],
                      suffix: str = '_factorized') -> 'DuckJanitor':
    """
    Encode categorical columns as numerical codes.
    Similar to encode_categorical() but with custom naming.
    """
    if isinstance(columns, str):
        columns = [columns]
    
    result = self
    for col in columns:
        # Use DENSE_RANK (same as encode_categorical)
        target_col = f"{col}{suffix}"
        result = result.encode_categorical(col, col_name=target_col)
    
    return result
```

**Trade-offs:**
- ✅ Already implemented via `encode_categorical()`
- ✅ Pure SQL

**Recommendation:** Implement as wrapper around `encode_categorical()`.

---

## 10. `get_dupes(columns=None)` - Find Duplicate Rows

**Challenge:** Identify and return duplicate rows.

**Current pyjanitor API:**
```python
df.get_dupes()  # all columns
df.get_dupes(columns=['col1', 'col2'])  # subset
```

**Proposed Implementation:**
```python
def get_dupes(self, columns: Optional[Union[str, List[str]]] = None) -> 'DuckJanitor':
    """
    Return duplicate rows.
    """
    if columns is None:
        columns = self._relation.columns
    elif isinstance(columns, str):
        columns = [columns]
    
    # Use window function to identify duplicates
    partition_cols = ', '.join(f'"{col}"' for col in columns)
    
    query = f"""
        SELECT * FROM (
            SELECT *,
                   COUNT(*) OVER (PARTITION BY {partition_cols}) as _dup_count
            FROM relation
        ) WHERE _dup_count > 1
    """
    
    new_relation = self._connection.execute(query)
    return DuckJanitor(new_relation, self._connection)
```

**Trade-offs:**
- ✅ Pure SQL with window functions
- ✅ Efficient

**Recommendation:** Implement! Move to "Can Do" category.

---

## 11. `join_apply(other, on, func, new_column_name)` - Join + Apply

**Challenge:** Apply arbitrary Python function after join.

**Current pyjanitor API:**
```python
df.join_apply(df2, on='key', func=lambda row: row['col1'] + row['col2'])
```

**Proposed Implementation:**
```python
def join_apply(self, other: 'DuckJanitor', on: Union[str, List[str]],
               func: Callable, new_column_name: str) -> 'DuckJanitor':
    """
    Join then apply Python function to each row.
    
    Note: This materializes the joined data to apply Python function.
    """
    # First, perform the join
    if isinstance(on, str):
        on = [on]
    
    # Register relations
    temp_self = f"_self_{id(self._relation)}"
    temp_other = f"_other_{id(other._relation)}"
    self._connection.register(temp_self, self._relation)
    self._connection.register(temp_other, other._relation)
    
    # Build join condition
    join_conditions = ' AND '.join(f'self."{col}" = other."{col}"' for col in on)
    
    join_query = f"""
        SELECT * FROM {temp_self} self
        INNER JOIN {temp_other} other
        ON {join_conditions}
    """
    
    joined = self._connection.execute(join_query)
    
    # Materialize and apply function
    df = joined.df()
    df[new_column_name] = df.apply(func, axis=1)
    
    return DuckJanitor.from_pandas(df)
```

**Trade-offs:**
- ✅ Flexible
- ❌ Materializes joined data
- ❌ row.apply() is slow

**Recommendation:** Implement with clear documentation about materialization.

---

## 12. `mutate(**kwargs)` - Create/Modify Columns

**Challenge:** This is essentially what `add_column()` already does.

**Current pyjanitor API:**
```python
df.mutate(new_col=df['col1'] * 2, another_col=df['col1'] + df['col2'])
```

**Proposed Implementation:**
```python
# Already covered by add_column() with SQL expressions
# Example:
# df.add_column('new_col', 'col1 * 2')
# df.add_column('another_col', 'col1 + col2')
```

**Trade-offs:**
- ✅ Redundant with add_column()
- ❌ No need to implement separately

**Recommendation:** Don't implement separately. Document that `add_column()` with SQL expressions provides the same functionality.

---

## 13. `process_text(column, func, new_column_name)` - Text Processing

**Challenge:** Arbitrary text processing functions may not have SQL equivalents.

**Current pyjanitor API:**
```python
df.process_text('text_col', lambda x: x.upper(), 'text_upper')
df.process_text('text_col', custom_func, 'processed')
```

**Proposed Implementation:**
```python
def process_text(self, column: str, func: Union[Callable, str],
                 new_column_name: str) -> 'DuckJanitor':
    """
    Apply text processing function to a column.
    
    If func is a string, it's treated as a SQL expression.
    If func is callable, data is materialized.
    """
    if isinstance(func, str):
        # SQL expression
        return self.add_column(new_column_name, func)
    elif callable(func):
        # Materialize and apply Python function
        df = self.collect()
        df[new_column_name] = df[column].apply(func)
        return DuckJanitor.from_pandas(df)
    else:
        raise ValueError("func must be a string (SQL) or callable")
```

**Trade-offs:**
- ✅ SQL expressions work without materialization
- ❌ Callables require materialization
- ✅ DuckDB has many built-in string functions

**Recommendation:** Implement hybrid approach. Document built-in DuckDB string functions as alternatives.

---

## 14. `move(column, position)` - Reorder Columns

**Challenge:** Moving columns to specific positions.

**Current pyjanitor API:**
```python
df.move('col1', 'first')
df.move('col2', 'after', 'col5')
df.move('col3', 'before', 'col1')
```

**Proposed Implementation:**
```python
def move(self, column: str, position: str = 'after',
         after: Optional[str] = None, before: Optional[str] = None) -> 'DuckJanitor':
    """
    Move a column to a new position.
    """
    all_columns = list(self._relation.columns)
    
    if column not in all_columns:
        raise ValueError(f"Column '{column}' not found")
    
    # Remove column from current position
    all_columns.remove(column)
    
    # Insert at new position
    if position == 'first':
        all_columns.insert(0, column)
    elif position == 'last':
        all_columns.append(column)
    elif position == 'after' and after:
        idx = all_columns.index(after) + 1
        all_columns.insert(idx, column)
    elif position == 'before' and before:
        idx = all_columns.index(before)
        all_columns.insert(idx, column)
    
    # Build SELECT with reordered columns
    select_parts = [f'"{col}"' for col in all_columns]
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    
    new_relation = self._connection.execute(query)
    return DuckJanitor(new_relation, self._connection)
```

**Trade-offs:**
- ✅ Pure SQL via SELECT ordering
- ✅ Already works!

**Recommendation:** Already implemented via SELECT column ordering. Document this.

---

## 15. `expand_column(column, sep, new_column_name)` - Delimited to Dummies

**Challenge:** Split delimited values and create dummy variables.

**Current pyjanitor API:**
```python
df.expand_column('categories', sep='|', new_column_name='cat_')
```

**Proposed Implementation:**
```python
# Already covered by get_dummies() after deconcatenate
# Or implement directly:

def expand_column(self, column: str, sep: str = '|',
                  prefix: Optional[str] = None) -> 'DuckJanitor':
    """
    Expand a delimited column into dummy variables.
    """
    if prefix is None:
        prefix = column
    
    # Get all unique values across all delimited entries
    # This requires unnesting the delimited values
    query = f"""
        SELECT DISTINCT UNNEST(str_split("{column}", '{sep}')) as value
        FROM relation
        WHERE "{column}" IS NOT NULL
    """
    
    unique_vals = [row[0] for row in self._connection.execute(query).fetchall()]
    
    # Create dummy variables
    dummy_exprs = []
    for val in unique_vals:
        dummy_name = f"{prefix}_{val}".replace(' ', '_').replace('-', '_')
        # Check if value is in the delimited list
        dummy_expr = f"""
            CASE WHEN list_contains(str_split("{column}", '{sep}'), '{val}')
            THEN 1 ELSE 0 END AS "{dummy_name}"
        """
        dummy_exprs.append(dummy_expr)
    
    # Keep other columns
    other_cols = [f'"{col}"' for col in self._relation.columns if col != column]
    select_parts = other_cols + dummy_exprs
    
    query = f"SELECT {', '.join(select_parts)} FROM relation"
    new_relation = self._connection.execute(query)
    
    return DuckJanitor(new_relation, self._connection)
```

**Trade-offs:**
- ✅ DuckDB has list functions (str_split, list_contains)
- ✅ Pure SQL implementation
- ❌ Slightly complex

**Recommendation:** Implement! Move to "Can Do" category.

---

## Summary: Recategorization

After detailed analysis, several functions can be moved from "Hard To Do" to "Can Do":

### Move to "Can Do" (implementable in pure SQL):
1. ✅ `conditional_join()` - DuckDB supports non-equi joins
2. ✅ `convert_date()` (all formats) - DuckDB has date arithmetic
3. ✅ `dropnotnull()` - Simple inversion of dropna
4. ✅ `factorize_columns()` - Wrapper around encode_categorical
5. ✅ `get_dupes()` - Window function solution
6. ✅ `move()` - Already works via SELECT ordering
7. ✅ `expand_column()` - DuckDB list functions

### Remain "Hard" (require hybrid approach):
1. ❌ `also()` - Side effects require materialization
2. ❌ `alias()` - Series concept doesn't exist in DuckDB
3. ❌ `compare_df_cols()` - Static utility, not method chain
4. ❌ `drop_duplicate_columns()` - Requires column content comparison
5. ❌ `explode_index()` - MultiIndex doesn't exist in DuckDB
6. ❌ `join_apply()` - Python function application
7. ❌ `mutate()` - Redundant with add_column()
8. ❌ `process_text()` - Arbitrary Python functions

**New Coverage:**
- Previously: 27/52 (52%)
- After moving 7 functions: 34/52 (65%)
- With hybrid implementations: 44/52 (85%)
