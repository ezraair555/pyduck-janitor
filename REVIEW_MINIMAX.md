# Code Review — Codex Hardening Pass (5f67d6c)

**Reviewer:** MiniMax (subagent, depth 1/1)
**Subject:** GPT-5.3 Codex hardening pass — `5f67d6c "Harden SQL-fragment APIs, add validations, docs, and edge-case tests"`
**Tests:** 152 pass (was 94); coverage 92% (was 93% — slight drop from added untested branches)

---

## Verdict: **REQUEST CHANGES**

The pass achieves its core security and validation goals (input validation, column-existence checks, SQL injection guard scaffolding, cross-connection join fallback). However, **the SQL-fragment validator is overly aggressive and breaks legitimate use of the API** (string-literal values and column names that happen to coincide with reserved words). This is a high-severity false-positive bug that must be fixed before merge, plus several smaller gaps in test coverage, doc accuracy, and one residual injection vector.

---

## Findings

### High

#### H1 — SQL-fragment validator blocks legitimate predicates with string literals
**File:** `pyduck_janitor/cleaning_ops.py:53-66` (`_validate_sql_fragment`)

The validator scans the raw user input for `;`, `--`, `/*`, `*/`, and a fixed keyword list (`attach|detach|copy|call|create|drop|alter|insert|update|delete|truncate`). It does **not** strip string literals or quoted identifiers before scanning, so any user predicate that contains a literal matching a blocked pattern is rejected — even when the literal is harmless.

Repro:
```python
dj.filter_on("name = 'a--b'")            # → ValueError: "cannot contain SQL comments"
dj.filter_on("name = 'drop'")            # → ValueError: "contains a disallowed SQL keyword"
dj.filter_on("comment = 'see /* note */'")  # → ValueError: "cannot contain SQL comments"
```

This is a **real false positive** for any dataset whose values contain dashes, slashes, or DDL-keyword-shaped strings — common in comment fields, ticket systems, log messages, product names, etc.

**Fix sketch:** Strip single-quoted string literals (and `'...' ESCAPE ...'`, `E'...'`) and double-quoted identifiers from the input before pattern-matching, or scope keyword matches to statement boundaries (e.g. `\bDROP\s+TABLE\b` rather than `\bDROP\b`).

---

#### H2 — SQL-fragment validator blocks column references whose names match reserved words
**File:** `pyduck_janitor/cleaning_ops.py:53-66` (used by `add_column`, `filter_on`, `case_when`, `groupby_agg`, `change_type`, `select_rows(criteria)`, `process_text`)

Real datasets sometimes have columns with names that look like SQL keywords (`drop`, `delete`, `update`, `insert`, `create`, `truncate`, `alter`, `call`, `copy`, `attach`, `detach`). Before this pass they worked fine via `filter_on("drop > 5")` or `add_column("y", "drop + 1")`. After the pass they fail:

```python
df = pd.DataFrame({"drop": [10, 20, 30]})
dj.filter_on("drop > 15")            # → ValueError, was: works
dj.add_column("y", "drop + 1")       # → ValueError, was: works
dj.groupby_agg("drop", {"x": "sum"}) # → still works (no condition), but adding any string agg breaks
```

`select` is the only keyword that escapes the block (it's not in the regex), confirming the blocklist is hand-picked rather than principled.

**Fix sketch:** Only block keywords when they are followed by whitespace + another SQL token (statement-form), e.g. `\bDROP\s+TABLE\b`, `\bDELETE\s+FROM\b`, `\bINSERT\s+INTO\b`, etc. Or whitelist column names against `relation.columns` before applying the regex.

This is also called out in the CHANGELOG as a security feature, so changing it will require a CHANGELOG note clarifying the new behavior.

---

#### H3 — `re.compile(search_string)` in `filter_string` is a no-op (defensive check) that surfaces an unhelpful error
**File:** `pyduck_janitor/cleaning_ops.py:818`

The new line `re.compile(search_string)` is intended to pre-validate the regex pattern, but its return value is discarded and any `re.error` bubbles up to the caller as an unhandled `re.error` (not a `ValueError`). The test suite never exercises this branch.

A user with an invalid regex sees `re.error: unterminated character set at position 0` instead of an actionable error like "invalid regex pattern for filter_string".

**Recommendation:** Either catch `re.error` and re-raise as `ValueError("invalid regex pattern: ...")` or drop the call entirely (DuckDB's `regexp_matches` will surface its own error, which is more contextual).

---

### Medium

#### M1 — Validator comment-marker checks use raw substring match, not SQL-aware
**File:** `pyduck_janitor/cleaning_ops.py:62-63`

`if "--" in text or "/*" in text or "*/" in text` is a substring search. It does not consider that `--` inside a string literal (`'--'`) is not a comment. This is the same root cause as H1 but specifically for the comment markers. The recommended fix (strip string literals first) covers this.

---

#### M2 — `find_replace` type-annotation is widened without bounds
**File:** `pyduck_janitor/cleaning_ops_final.py:375`

The signature changed from `value_pairs: Dict[str, str]` to `value_pairs: Dict[Any, Any]`. The reasoning (allow non-string keys/values for typed columns) is sound, but no test verifies that numeric keys/values are accepted by the underlying SQL construction. The SQL expression `_sql_literal(value)` already handles non-strings, so this works, but the lack of a regression test means a future SQL-construction refactor could silently break numeric mappings.

---

#### M3 — Inconsistent `target_column` / `new_column_name` validation
**Files:** `cleaning_ops.py:461` (coalesce), `cleaning_ops_extended.py` (deconcatenate_column `target_columns`), `cleaning_ops_final.py:91,127` (join_apply, process_text)

Some functions validate that the target column name is a non-empty stripped string, some don't:
- `coalesce`, `join_apply`, `process_text` — validate
- `min_max_scale` (`pyduck_janitor/cleaning_ops_extended.py:343-360`) — does not validate `target_column`; an empty string produces a `ParserException: zero-length delimited identifier`
- `pivot_wider` / `pivot_longer` — `name_col`, `value_col` not validated
- `add_column` — validates
- `transform_column` — does not validate `target_column` either; same opaque `ParserException` failure

Recommend applying the same `isinstance(..., str) and ....strip()` guard consistently.

---

#### M4 — `add_column(values=...)` SQL-fragment guard breaks the existing string-literal short-circuit
**File:** `pyduck_janitor/cleaning_ops.py:215-227`

`add_column` accepts a string for `values` and historically had a "try as SQL expression/column first" path. The new validator runs **before** that try, so any string that contains a blocked pattern (e.g. a user passing a literal value like `"abc; def"`) is rejected outright. This is consistent with H1 (false positives) but it's worth noting that this codepath now uniformly rejects literal strings containing `;` — a real user who wants `add_column("note", "abc; def")` (a literal, not an expression) can no longer do it.

The original code's fallback to `_sql_literal` for raw string scalars is the right pattern; the new validator undoes that flexibility.

---

#### M5 — Test coverage gaps for new validations
**File:** `tests/test_validation_and_edges.py`

The new tests cover ~14 of the ~50+ functions whose behavior changed. Untested validation paths include:
- `clean_names(case_type='invalid')` (line 92 of cleaning_ops.py)
- `remove_columns(columns=[])` (line 175)
- `add_column(column_name=None)` / `add_column(column_name='')` (line 216)
- `rename_column(old_name='x', new_name='')` (line 277)
- `coalesce(target_column='   ')` (line 461) — current coverage is 95%, missed
- `transform_columns(target_columns=[...wrong length...])` (line 737-739) — no test
- `pivot_wider(name_col=...)` with all-NULL values (line 528-529) — no test (the new branch is uncovered)
- `conditional_join(how='outer')` invalid value, `conditional_join(on=[])` (line 165) — no test
- `flag_nulls` (line 287) — no test
- `limit_column_characters(max_chars=-1)` (line 316) — no test
- `complete(columns=[])` (line 458) — no test
- `dropnotnull(subset=[])` (line 239) — no test
- `get_dupes(columns=[])` (line 207) — no test
- `label_encode(columns=[])` (line 376) — no test
- `expand_column(column='missing')` — no test (in the `test_methods_raise_for_missing_columns` list but quick check confirms it's present)
- `min_max_scale(target_column='')` (line 358) — no test for opaque error path
- `case_when(conditions=[])` (line 439) — no test
- `alias(callable returning '')` (line 526) — no test

Recommend at least one smoke test per new validation branch. The current 58 tests are a good foundation, but the test file's parameterised list `test_methods_raise_for_missing_columns` is the only place many of these branches get coverage — and not all of them are in that list.

---

#### M6 — Test for `test_filter_column_rejects_destructive_sql` is broader than it should be
**File:** `tests/test_validation_and_edges.py:42-56`

This test asserts `pytest.raises(ValueError)` for both *destructive* SQL (`create table x`, `truncate table t`, `call now()`) and the *legitimate* ones. That's fine for the rejection path, but the test name "rejects destructive SQL" is misleading — `call now()` is a function call, not DDL. The test should be split or renamed, and an additional positive control test should verify that a legitimate predicate like `"age > 30"` still passes through `filter_column`.

---

#### M7 — `re.compile` in `filter_string` lacks a regression test
**File:** `tests/test_validation_and_edges.py`

There is no test for `dj.filter_string('x', '[invalid', regex=True)`. This is the only new code path that has no test. Either drop the `re.compile` call (H3) or add a test asserting it raises a clean `ValueError`.

---

### Low

#### L1 — CHANGELOG "Added" entry for `__init__.py` import fix is misclassified
**File:** `CHANGELOG.md:18`

The "Removed duplicate import entry for `truncate_datetime`" is listed under "Fixed" but it's really a "Removed dead code" change. Trivial; cosmetic.

---

#### L2 — README "Supported Functions" count is inconsistent with actual count
**File:** `README.md:101-187`

The README claims "14 functions" in core, "17 functions" in extended, "24 functions" in final, but the lists under each heading contain different counts. The "Extended" list contains 21 items, not 17. The "Final/Hybrid" list contains 17 items, not 24. The number "51 functions" in the lead is the historical count from the v0.1.0 release. After the v0.1.1/0.1.2 additions, the actual function count is higher and should be verified.

---

#### L3 — `case_type` validation message lists 3 options but only 3 are accepted — fine, but docstring says 4
**File:** `pyduck_janitor/cleaning_ops.py:78-90`

The docstring still says `case_type : str — Case conversion ('lower', 'upper', 'original')` which is correct, but the docstring doesn't list `'preserve'` or other potential variants. This is fine; just noting that the docstring is now correct and matches the validator.

---

#### L4 — `target_columns` parameter inconsistency in `transform_columns`
**File:** `pyduck_janitor/cleaning_ops.py:728-741`

The new check is `if isinstance(target_columns, list) and len(target_columns) != len(columns)`. If the user passes `target_columns` as a `tuple` of correct length, it passes the check (because the second condition is False), but then the function later does `target_columns[i] if i < len(target_columns) else col` which works. However, the annotation says `Optional[Union[str, List[str]]]` and the check only covers `list`. Tuple is silently accepted; this is a minor type-vs-runtime mismatch.

---

#### L5 — `DuckJanitor.__init__` connection-handling: pre-existing, not regressed
**File:** `pyduck_janitor/duck_janitor.py:42`

The connection is typed as `Optional[DuckDBPyConnection]` but the constructor body (not in the diff but in the file) presumably defaults to `cls._shared_conn` when `None`. This is unchanged by the pass, but worth noting because the new tests in `test_join_apply_cross_connection_materialization` create explicit connections without testing the default path. Not a regression.

---

#### L6 — `head(n=0)` is allowed; `head(n=0.5)` is not
**File:** `pyduck_janitor/duck_janitor.py:202-207`

`n < 0` is rejected, but `n=0` is allowed and returns an empty DataFrame, which is fine. The test `test_head_rejects_negative_n` covers negatives. `n=0.5` is not tested; it would raise a DuckDB type error. Worth a single positive-and-zero test if completeness is desired.

---

#### L7 — `from_csv` `**kwargs: Any` is more accurate than the untyped `**kwargs` was, but `from_pandas` / `from_parquet` / `from_sql` remain untyped
**File:** `pyduck_janitor/duck_janitor.py:135`

A nice touch for `from_csv`, but the other `from_*` methods still have bare `**kwargs` (where applicable) or no kwargs at all. Inconsistent. Trivial.

---

#### L8 — `sql()` word-boundary fix is good, but the new test does not exercise it
**File:** `pyduck_janitor/duck_janitor.py:476`

The change from `query.replace('self', temp_name)` to `re.sub(r"\bself\b", temp_name, query)` is a clear improvement. No regression test in the new file exercises it. (Existing `test_duck_janitor.py` may cover it — to verify, but not part of this review's diff.)

---

#### L9 — Coverage drop: 93% → 92%
**Files:** all cleaning modules

Net coverage dropped from 93% to 92% even though 58 new tests were added. This is because the new validation branches (in `add_column`, `coalesce`, `filter_string`, `transform_columns`, `pivot_wider`, `conditional_join`, `case_when`, `complete`, `get_dupes`, `dropnotnull`, `label_encode`, `alias`, `limit_column_characters`, `min_max_scale`) added more uncovered lines than the new tests cover. Not a regression in quality — the *code* is more robust — but worth tracking.

---

### Residual Risks (from CODE_REVIEW.md, confirmed)

- **`DuckJanitor.sql()` accepts raw SQL** — intentional power-user surface, but the word-boundary fix in this pass is a small but real improvement. Document the trust boundary.
- **N+1 patterns in `remove_empty`, `drop_constant_columns`, `drop_duplicate_columns`** — listed in `CODE_REVIEW.md` as "acceptable for now"; still present. The CODE_REVIEW.md "Performance Notes" section is honest about this; no action required.
- **Defensive `try/except Exception` in `join_apply` cross-connection fallback** — broad catch is fine here (the fallback is well-defined and idempotent), but a narrower catch (`duckdb.Error`, `duckdb.InvalidInputException`) would be more defensive. Low priority.
- **Hybrid functions still materialize** — `also`, callable `transform_column`, `process_text`, `join_apply` are intentionally hybrid. The pass correctly preserves this contract; just confirming it's documented.

---

## What's Still Missing

1. **Real fix for H1/H2** — strip string literals from validator input, or scope keyword matches to statement form. Without this, the pass is a *net regression* for users with realistic column names or string values.
2. **Test for the new `pivot_wider` all-null name_col branch** (cleaning_ops_extended.py:528-529) — easy to add, currently untested.
3. **Tests for the `transform_columns` length-mismatch branch** — currently 0% covered for that line.
4. **`__init__.py` `__all__` consistency** — not in the diff but worth a glance: ensure the new public surface (`join_apply`, `process_text`, `mutate`, `also`, `alias`, `complete`, `drop_duplicate_columns`, `compare_df_cols`, `get_dupes`) is all in `__all__` if it isn't already.
5. **`from_csv` `**kwargs: Any` consistency with the other `from_*` constructors** — trivial but inconsistent.
6. **Function-by-function docstring expansion** — explicitly punted in the pass ("future incremental task"). Recommend at least the `head`, `filter_string`, `case_when`, `groupby_topk`, `bin_numeric`, `find_replace`, `alias`, `case_when` docstrings get a "Raises" section now that they have explicit validation.
7. **README "Supported Functions" counts** — wrong (see L2).
8. **`__version__` and the v0.1.3 bump** — consistent between `__init__.py` and `CHANGELOG.md`. ✓
9. **CHANGELOG note about H1/H2 behavior change** — if H1/H2 is fixed, the CHANGELOG entry should be updated to reflect the *actual* (more permissive) behavior; if H1/H2 is kept, document the limitation.
10. **A test for `sql()` word-boundary replacement** (e.g. column named `myself` or `selfish`) — L8.

---

## Recommendations for Next Steps

**Block merge** until H1 and H2 are addressed. Both are demonstrable false positives against legitimate, common user input.

Once H1/H2 are fixed:
- Re-run the full test suite (`pytest -q`).
- Add the missing tests in M5 / M7 to lock in the new behavior.
- Update README "Supported Functions" counts (L2).
- Decide whether the `re.compile` pre-check in `filter_string` (H3) stays; if yes, add a clean-error test and a test for legitimate regex.
- Run `black` and `flake8` (mentioned in CONTRIBUTING.md) to verify style.

If the false-positive behavior is *intentional* and considered acceptable for a hardening pass, then document it explicitly in the CHANGELOG and in a `SECURITY.md` or in the function docstrings (e.g. "Note: string literals matching keyword patterns will be rejected"). Without that, the next user who passes a column named `delete` will file a bug.

---

## Summary Table

| ID  | Severity | Area                | File:Line                          | Status |
|-----|----------|---------------------|------------------------------------|--------|
| H1  | High     | SQL fragment guard  | cleaning_ops.py:53-66              | Needs fix |
| H2  | High     | SQL fragment guard  | cleaning_ops.py:53-66              | Needs fix |
| H3  | High     | filter_string       | cleaning_ops.py:818                | Needs decision |
| M1  | Medium   | Comment markers     | cleaning_ops.py:62-63              | Same fix as H1 |
| M2  | Medium   | find_replace type   | cleaning_ops_final.py:375          | Add test |
| M3  | Medium   | target_column val.  | multiple                           | Apply uniformly |
| M4  | Medium   | add_column values   | cleaning_ops.py:215-227            | Same fix as H1 |
| M5  | Medium   | Test coverage       | tests/test_validation_and_edges.py | Add tests |
| M6  | Medium   | Test naming/scope   | tests/test_validation_and_edges.py | Rename/split |
| M7  | Medium   | filter_string test  | tests/test_validation_and_edges.py | Add test |
| L1  | Low      | CHANGELOG           | CHANGELOG.md:18                    | Cosmetic |
| L2  | Low      | README counts       | README.md                          | Verify counts |
| L3  | Low      | docstring parity    | cleaning_ops.py:78-90              | ✓ |
| L4  | Low      | type annotation     | cleaning_ops.py:728-741            | Cosmetic |
| L5  | Low      | connection default  | duck_janitor.py:42                 | Pre-existing |
| L6  | Low      | head n=0.5          | duck_janitor.py:202-207            | Add test |
| L7  | Low      | from_* consistency  | duck_janitor.py                    | Cosmetic |
| L8  | Low      | sql() word boundary | duck_janitor.py:476                | Add test |
| L9  | Low      | Coverage            | All                                | Tracking |

**Final verdict: REQUEST CHANGES — H1 and H2 must be resolved (or explicitly accepted with documentation) before merge.**
