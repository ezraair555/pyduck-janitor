# pyjanitor Function Categorization for pyduck-janitor

## ✅ Implemented (27 functions)

### Core Functions (12 - Previously Complete)

| Function | SQL Translation | Status |
|----------|-----------------|--------|
| `clean_names()` | Column renaming | ✓ Complete |
| `remove_columns()` | SELECT exclusion | ✓ Complete |
| `add_column()` / `add_columns()` | Computed column | ✓ Complete |
| `rename_column()` / `rename_columns()` | AS alias | ✓ Complete |
| `dropna()` | WHERE IS NOT NULL | ✓ Complete |
| `remove_empty()` | WHERE + column check | ✓ Complete |
| `filter_column()` / `filter()` | WHERE clause | ✓ Complete |
| `coalesce()` | COALESCE() | ✓ Complete |
| `encode_categorical()` | DENSE_RANK() | ✓ Complete |
| `get_dummies()` / `expand_column()` | CASE WHEN | ✓ Complete |
| `sql()` | Pass-through | ✓ Complete |
| `collect()` / `head()` | LIMIT | ✓ Complete |

### Phase 1 High Priority (15 - Newly Implemented)

| Function | SQL Translation | Status |
|----------|-----------------|--------|
| `bin_numeric()` | NTILE() / WIDTH_BUCKET / CASE WHEN | ✓ Complete |
| `change_type()` | CAST() | ✓ Complete |
| `concatenate_columns()` | \|\| operator | ✓ Complete |
| `deconcatenate_column()` | str_split() | ✓ Complete |
| `drop_constant_columns()` | COUNT(DISTINCT) > 1 | ✓ Complete |
| `fill()` | Window functions (LAST_VALUE/FIRST_VALUE) | ✓ Complete |
| `flag_nulls()` | CASE WHEN IS NULL | ✓ Complete |
| `limit_column_characters()` | substr() \|\| | ✓ Complete |
| `min_max_scale()` | Normalization formula | ✓ Complete |
| `groupby_agg()` | GROUP BY | ✓ Complete |
| `groupby_topk()` | ROW_NUMBER() OVER | ✓ Complete |
| `case_when()` | CASE WHEN | ✓ Complete |
| `currency_column_to_numeric()` | regexp_replace + CAST | ✓ Complete |
| `convert_date()` | strptime / CAST | ✓ Complete |
| `move()` | Column reordering in SELECT | ✓ (via SELECT ordering) |

---

## 🟡 Can Do With Some Effort (25 functions)

These can be translated to SQL with moderate complexity:

| Function | SQL Approach | Priority |
|----------|--------------|----------|
| `bin_numeric()` | CASE WHEN or NTILE() | High |
| `change_type()` | CAST() | High |
| `concatenate_columns()` | \|\| operator | High |
| `convert_date()` | CAST/TRY_CAST | High |
| `currency_column_to_numeric()` | regexp_replace + CAST | High |
| `deconcatenate_column()` | str_split() | High |
| `drop_constant_columns()` | GROUP BY + HAVING | High |
| `fill()` | Window functions (LAG/LEAD) | High |
| `flag_nulls()` | CASE WHEN IS NULL | High |
| `limit_column_characters()` | substr() | High |
| `min_max_scale()` | Normalization formula | High |
| `move()` | Column reordering in SELECT | High |
| `impute()` | COALESCE + AVG/MODE | Medium |
| `jitter()` | + random() * scale | Medium |
| `label_encode()` | DENSE_RANK() | Medium |
| `find_replace()` | CASE WHEN or REPLACE() | Medium |
| `count_cumulative_unique()` | Window COUNT(DISTINCT) | Medium |
| `expand_grid()` | CROSS JOIN | Medium |
| `collapse_levels()` | MultiIndex flatten | Low |
| `complete()` | CROSS JOIN + COALESCE | Medium |
| `groupby_agg()` | GROUP BY | High |
| `groupby_topk()` | ROW_NUMBER() OVER | High |
| `case_when()` | CASE WHEN | High |
| `pivot()` | PIVOT (DuckDB supports) | Medium |
| `change_index_dtype()` | N/A (DuckDB has no index) | Low |

---

## 🔴 Hard To Do (15 functions)

These require Python processing, ML, or have no direct SQL equivalent:

| Function | Challenge | Proposed Solution |
|----------|-----------|-------------------|
| `also()` | Side effects in SQL | Materialize, apply Python func, re-wrap |
| `alias()` | Series-level operation | DuckDB doesn't have Series; materialize |
| `compare_df_cols()` | Two DataFrame comparison | Materialize both, compare in Python |
| `conditional_join()` | Complex join conditions | DuckDB supports, but complex API mapping |
| `convert_date()` (Matlab/Excel) | Custom epoch conversions | Python date math, then SQL |
| `drop_duplicate_columns()` | Column content comparison | Materialize, compare in Python |
| `dropnotnull()` | Inverse of dropna | Easy, just invert logic |
| `explode_index()` | MultiIndex operation | DuckDB has no MultiIndex; materialize |
| `factorize_columns()` | Python-specific encoding | Use DENSE_RANK() as alternative |
| `get_dupes()` | Duplicate detection | Window function ROW_NUMBER() |
| `join_apply()` | Apply Python func after join | Materialize join result, apply in Python |
| `mutate()` | Complex expressions | Already covered by add_column + SQL |
| `process_text()` | Arbitrary text functions | DuckDB has string funcs, but not all |
| `move()` | Column position | Already doable via SELECT ordering |
| `expand_column()` | Delimited to dummies | Already implemented as get_dummies |

---

## 📋 Implementation Plan

### Phase 1: High Priority "Can Do" (Implement Now)

1. `bin_numeric()` - NTILE() or CASE WHEN
2. `change_type()` - CAST()
3. `concatenate_columns()` - \|\| operator
4. `deconcatenate_column()` - str_split()
5. `drop_constant_columns()` - GROUP BY variance check
6. `fill()` - Window functions
7. `flag_nulls()` - CASE WHEN
8. `groupby_agg()` - GROUP BY
9. `groupby_topk()` - ROW_NUMBER() OVER
10. `case_when()` - CASE WHEN
11. `min_max_scale()` - Normalization formula
12. `currency_column_to_numeric()` - regexp_replace

### Phase 2: Medium Priority "Can Do"

13. `impute()` - COALESCE + stats
14. `jitter()` - random()
15. `label_encode()` - DENSE_RANK()
16. `find_replace()` - REPLACE()
17. `count_cumulative_unique()` - Window function
18. `expand_grid()` - CROSS JOIN
19. `complete()` - CROSS JOIN + fill
20. `pivot()` - DuckDB PIVOT
21. `limit_column_characters()` - substr()
22. `convert_date()` - CAST

### Phase 3: "Hard To Do" - Hybrid Approach

For functions that can't be pure SQL:

```python
def also(self, func: Callable) -> 'DuckJanitor':
    """Apply Python function with side effects."""
    # Materialize
    df = self.collect()
    # Apply function
    func(df)
    # Re-wrap
    return DuckJanitor.from_pandas(df)
```

---

## Notes

- DuckDB has excellent SQL support including: window functions, PIVOT, regex, string functions
- Some pandas-specific concepts (MultiIndex, Series operations) don't map to DuckDB
- For "hard" functions, we can use a hybrid: materialize → Python → re-wrap
- Priority should be given to functions that preserve lazy evaluation
