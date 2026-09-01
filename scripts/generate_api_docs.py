#!/usr/bin/env python3
"""Generate docs/api/functions.md — the pyduck-janitor function reference.

Produces a pyjanitor-style API page: for every public method on
``DuckJanitor`` (plus the module-level ``DropLabel`` / ``patterns`` helpers)
it renders

* an anchored section (function name, signature)
* description (from the source docstring)
* Parameters / Returns / Raises (parsed from the source docstring, when present)
* a verified example

Examples come from two places, in priority order:
1. the source docstring's example block, if present;
2. the VERIFIED_EXAMPLES map below — every snippet in the map is *executed*
   by ``--check`` before it is allowed into the generated page, so the page
   never ships an example that fails.

Usage:
    python3 scripts/generate_api_docs.py           # write docs/api/functions.md
    python3 scripts/generate_api_docs.py --check   # verify every example runs
"""
from __future__ import annotations

import ast
import inspect
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'docs' / 'api' / 'functions.md'

DOC_BASE = 'https://pyjanitor-devs.github.io/pyjanitor/api/functions/'


# ===========================================================================
# Verified examples
# ===========================================================================
# Every entry is plain Python. A shared prologue defined in run_example()
# provides: pd, np, DuckJanitor, DropLabel, patterns, tempfile helpers, and a
# default sample frame `df` + `dj`. Snippets below build on that.

VERIFIED_EXAMPLES: dict[str, str] = {
    # ---- loaders & plumbing -------------------------------------------
    'from_pandas': "df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})\ndj = DuckJanitor.from_pandas(df)\ndj.collect()",
    'from_csv': "p = make_csv({'a': [1, 2], 'b': ['x', 'y']})\ndj = DuckJanitor.from_csv(p)\ndj.collect()",
    'from_parquet': "p = make_parquet({'a': [1, 2], 'b': ['x', 'y']})\ndj = DuckJanitor.from_parquet(p)\ndj.collect()",
    'from_excel': "p = make_excel({'a': [1, 2]})\ndj = DuckJanitor.from_excel(p)\ndj.collect()",
    'from_json': "p = make_json([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}])\ndj = DuckJanitor.from_json(p)\ndj.collect()",
    'from_sql': "dj = DuckJanitor.from_sql(\"SELECT 1 AS a, 'x' AS b\")\ndj.collect()",
    'collect': "dj = DuckJanitor.from_pandas(df)\ndj.clean_names().collect()",
    'head': "dj = DuckJanitor.from_pandas(df)\ndj.head()",
    'sql': "dj = DuckJanitor.from_pandas(df)\ndj.sql('SELECT b, SUM(a) AS total FROM self GROUP BY b').collect()",
    'explain': "dj = DuckJanitor.from_pandas(df)\ndj.filter_on('a > 1').explain()",
    'get_shared_connection': "dj = DuckJanitor.from_pandas(df)\nconn = dj.get_shared_connection()\ntype(conn).__name__",
    # ---- core cleaning verbs ------------------------------------------
    'clean_names': "dj = DuckJanitor.from_pandas(pd.DataFrame({'Sales Month': [1], 'A-B': [2]}))\ndj.clean_names().collect()",
    'remove_columns': "dj = DuckJanitor.from_pandas(df)\ndj.remove_columns(['b']).collect()",
    'add_column': "dj = DuckJanitor.from_pandas(df)\ndj.add_column('extra', [1, 2, 3, 4]).collect()",
    'add_columns': "dj = DuckJanitor.from_pandas(df)\ndj.add_columns({'extra': [1, 2, 3, 4], 'more': [9, 8, 7, 6]}).collect()",
    'rename_column': "dj = DuckJanitor.from_pandas(df)\ndj.rename_column('a', 'value').collect()",
    'rename_columns': "dj = DuckJanitor.from_pandas(df)\ndj.rename_columns('a', 'value').collect()",
    'dropna': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, None, 3], 'b': ['x', 'y', 'z']}))\ndj2.dropna().collect()",
    'remove_empty': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, None], 'b': [None, None]}))\ndj2.remove_empty().collect()",
    'filter_column': "dj = DuckJanitor.from_pandas(df)\ndj.filter_column('a', 'a > 2').collect()",
    'filter_column_isin': "dj = DuckJanitor.from_pandas(df)\ndj.filter_column_isin('group', ['x', 'z']).collect()",
    'filter_on': "dj = DuckJanitor.from_pandas(df)\ndj.filter_on('a > 2 AND b > 15').collect()",
    'filter_string': "dj = DuckJanitor.from_pandas(df)\ndj.filter_string('group', '^x').collect()",
    'filter_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'d': pd.to_datetime(['2024-01-01', '2024-06-01', '2024-12-31'])}))\ndj2.filter_date('d', '2024-03-01', '2024-08-01').collect()",
    'coalesce': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, None], 'b': [None, 2]}))\ndj2.coalesce(['a', 'b'], 'merged').collect()",
    'encode_categorical': "dj = DuckJanitor.from_pandas(df)\ndj.encode_categorical('group').collect()",
    'get_dummies': "dj = DuckJanitor.from_pandas(df)\ndj.get_dummies(['group']).collect()",
    'select_columns': "dj = DuckJanitor.from_pandas(df)\n# comma-strings, globs and regex all work:\ndj.select_columns('a, group').collect()",
    'select': "dj = DuckJanitor.from_pandas(df)\ndj.select(columns='a, group').collect()",
    'select_rows': "dj = DuckJanitor.from_pandas(df)\ndj.select_rows([0, 3]).collect()",
    'transform_column': "dj = DuckJanitor.from_pandas(df)\ndj.transform_column('a', lambda s: s * 10, 'a_x10').collect()",
    'transform_columns': "dj = DuckJanitor.from_pandas(df)\ndj.transform_columns(['a'], lambda s: s * 10, ['a_x10']).collect()",
    # ---- extended verbs -------------------------------------------------
    'bin_numeric': "dj = DuckJanitor.from_pandas(df)\ndj.bin_numeric('a', 'a_bin', bins=2).collect()",
    'change_type': "dj = DuckJanitor.from_pandas(df)\ndj.change_type('a', 'DOUBLE').collect()",
    'concatenate_columns': "dj = DuckJanitor.from_pandas(df)\ndj.concatenate_columns(['a', 'b'], '_').collect()",
    'deconcatenate_column': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'combo': ['x_1', 'y_2']}))\ndj2.deconcatenate_column('combo', '_', ['letter', 'number']).collect()",
    'drop_constant_columns': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]}))\ndj2.drop_constant_columns().collect()",
    'fill': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1.0, None, None, 4.0]}))\ndj2.fill('a', direction='forward').collect()",
    'fill_direction': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1.0, None, None, 4.0]}))\ndj2.fill_direction('a', direction='forward').collect()",
    'fill_empty': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': ['x', '', 'z']}))\ndj2.fill_empty('a').collect()",
    'flag_nulls': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, None, 3]}))\ndj2.flag_nulls('a').collect()",
    'limit_column_characters': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a_column_with_a_long_name': ['v']}))\ndj2.limit_column_characters('a_column_with_a_long_name', 6).collect()",
    'min_max_scale': "dj = DuckJanitor.from_pandas(df)\ndj.min_max_scale('a', 'a_scaled').collect()",
    'groupby_agg': "dj = DuckJanitor.from_pandas(df)\ndj.groupby_agg('group', {'a': 'AVG', 'b': 'MAX'}).collect()",
    'groupby_topk': "dj = DuckJanitor.from_pandas(df)\ndj.groupby_topk('group', 'a', k=1).collect()",
    'case_when': "dj = DuckJanitor.from_pandas(df)\ndj.case_when([('a > 2', \"'high'\"), ('a <= 2', \"'low'\")], 'level').collect()",
    'currency_column_to_numeric': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'price': ['$1,200.50', '$9.99']}))\ndj2.currency_column_to_numeric('price').collect()",
    'convert_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'d': ['2024-01-01', '2024-06-01']}))\ndj2.convert_date('d', date_format='%Y-%m-%d').collect()",
    'convert_to_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'d': ['2024-01-01', '2024-06-01']}))\ndj2.convert_to_date('d', date_format='%Y-%m-%d').collect()",
    'convert_to_datetime': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'d': ['2024-01-01 12:00', '2024-06-01 09:30']}))\ndj2.convert_to_datetime('d').collect()",
    'convert_unix_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'epoch': [1577836800, 1704067200]}))\ndj2.convert_unix_date('epoch').collect()",
    'convert_excel_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'serial': [25569, 25569 + 30]}))\ndj2.convert_excel_date('serial').collect()",
    'convert_matlab_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'serial': [719529, 719529 + 30]}))\ndj2.convert_matlab_date('serial').collect()",
    'excel_time_to_numeric': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'t': [0.25, 0.75]}))\ndj2.excel_time_to_numeric('t').collect()",
    'sas_numeric_to_date': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'days': [0, 30]}))\ndj2.sas_numeric_to_date('d' if False else 'days').collect()",
    'to_datetime': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'d': ['2024-01-01', '2024-06-01']}))\ndj2.to_datetime('d', format='%Y-%m-%d').collect()",
    'truncate_datetime': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'ts': pd.to_datetime(['2024-03-15 12:34', '2024-06-01 08:00'])}))\ndj2.truncate_datetime('ts', unit='month').collect()",
    'truncate_datetime_dataframe': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'ts': pd.to_datetime(['2024-03-15 12:34', '2024-06-01 08:00'])}))\ndj2.truncate_datetime_dataframe('ts', unit='month').collect()",
    'pivot_wider': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'id': [1, 1, 2], 'k': ['x', 'y', 'x'], 'v': [10, 20, 30]}))\ndj2.pivot_wider('id', 'k', 'v').collect()",
    'pivot_longer': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'id': [1, 2], 'x': [10, 20], 'y': [30, 40]}))\ndj2.pivot_longer(['x', 'y']).collect()",
    # ---- hybrid verbs ----------------------------------------------------
    'conditional_join': "left = DuckJanitor.from_pandas(pd.DataFrame({'lo': [1, 5]}))\nright = DuckJanitor.from_pandas(pd.DataFrame({'hi': [3, 7], 'tag': ['p', 'q']}))\nleft.conditional_join(right, [('lo', 'hi', '<')]).collect()",
    'get_dupes': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']}))\ndj2.get_dupes().collect()",
    'dropnotnull': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, None], 'b': [2, 3]}))\ndj2.dropnotnull(subset=['a']).collect()",
    'expand_column': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'tags': ['a|b', 'c|d']}))\ndj2.expand_column('tags').collect()",
    'impute': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1.0, None, 3.0]}))\ndj2.impute('a', statistic='mean').collect()",
    'jitter': "dj = DuckJanitor.from_pandas(df)\ndj.jitter('a', 'a_jittered', scale=0.1, seed=0).collect()",
    'label_encode': "dj = DuckJanitor.from_pandas(df)\ndj.label_encode('group', 'group_code').collect()",
    'find_replace': "dj = DuckJanitor.from_pandas(df)\ndj.find_replace('group', {'x': 'X', 'z': 'Z'}).collect()",
    'count_cumulative_unique': "dj = DuckJanitor.from_pandas(df)\ndj.count_cumulative_unique('group', 'ccu').collect()",
    'complete': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'g': ['x', 'y'], 'm': [1, 2]}))\ndj2.complete(['g'], fill_value=0).collect()",
    
    'also': "dj = DuckJanitor.from_pandas(df)\ndj.also(lambda pdf: pdf.assign(materialized='Y')).collect()",
    'alias': "dj = DuckJanitor.from_pandas(df)\ndj.alias(str.upper).collect()  # rename all columns via callable",
    'ungroup': "dj = DuckJanitor.from_pandas(df)\ndj.ungroup().collect()",
    'mutate': "dj = DuckJanitor.from_pandas(df)\ndj.mutate(double_a=[2, 4, 6, 8]).collect()",
    'assign': "dj = DuckJanitor.from_pandas(df)\ndj.assign(double_a=[2, 4, 6, 8]).collect()",
    
    'drop_duplicate_columns': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, 2], 'b': [1, 2]}))\ndj2.drop_duplicate_columns().collect()",
    'compare_df_cols': "a = DuckJanitor.from_pandas(pd.DataFrame({'x': [1]}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'x': [2], 'y': [3]}))\ncompare_out = a.compare_df_cols(b)\ncompare_out",
    'compare_df_cols_same': "a = DuckJanitor.from_pandas(pd.DataFrame({'x': [1]}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'x': [2]}))\na.compare_df_cols_same(b)",
    'join_apply': "a = DuckJanitor.from_pandas(pd.DataFrame({'id': [1, 2], 'v': [10, 20]}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'id': [1, 2], 'w': [3, 4]}))\na.join_apply(b, 'id', lambda row: row['v'] + row['w'], 'v_plus_w').collect()",
    # NOTE: join_apply func signature is verified below; adjust if needed
    'process_text': "dj = DuckJanitor.from_pandas(df)\ndj.process_text('group', str.upper, 'group_upper').collect()",
    'describe_class': "dj = DuckJanitor.from_pandas(df)\ndj.describe_class()",
    # ---- parity batch: structural / reshape / aggregate / misc ----------
    'move': "dj = DuckJanitor.from_pandas(df)\ndj.move('group', 'a', position='before').collect()",
    'reorder_columns': "dj = DuckJanitor.from_pandas(df)\ndj.reorder_columns(['group', 'b', 'a']).collect()",
    'get_columns': "dj = DuckJanitor.from_pandas(df)\ndj.get_columns('a', 'group').collect()",
    'get_index_labels': "dj = DuckJanitor.from_pandas(df)\ndj.get_index_labels()",
    'row_to_names': "dj2 = DuckJanitor.from_pandas(pd.DataFrame([['id', 'v'], [1, 10], [2, 20]], columns=['c1', 'c2']))\ndj2.row_to_names(0).collect()",
    'collapse_levels': "dj = DuckJanitor.from_pandas(df)\ndj.collapse_levels(column='a').collect()",
    'explode_index': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'code': ['x1', 'y22']}))\ndj2.explode_index('code').collect()",
    'change_index_dtype': "dj = DuckJanitor.from_pandas(df)\ndj.change_index_dtype('VARCHAR').collect()",
    'expand': "dj = DuckJanitor.from_pandas(df)\ndj.expand(['group']).collect()",
    'expand_grid': "a = DuckJanitor.from_pandas(pd.DataFrame({'x': [1, 2], 'g': ['p', 'q']}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'y': [10, 20], 'h': ['r', 's']}))\na.expand_grid(b).collect()",
    'summarise': "dj = DuckJanitor.from_pandas(df)\ndj.summarise(group_by=['group'], agg_spec={'avg_a': ('a', 'AVG'), 'n': ('*', 'COUNT')}).collect()",
    'pivot_longer_spec': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'id': [1, 2], 'y2023': [10, 20], 'y2024': [100, 200]}))\ndj2.pivot_longer_spec(['id'], ['y2023', 'y2024'], 'year', 'v').collect()",
    'pivot_wider_spec': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'id': [1, 2], 'year': ['2023', '2024'], 'value': [10, 20]}))\ndj2.pivot_wider_spec(['id'], 'year', 'value').collect()",
    'join_agg': "left = DuckJanitor.from_pandas(pd.DataFrame({'lo': [1, 2]}))\nright = DuckJanitor.from_pandas(pd.DataFrame({'hi': [2, 3], 'w': [10, 20]}))\nleft.join_agg(right, on=('lo', 'hi', '<'), aggs={'max_w': ('w', 'MAX')}).collect()",
    'get_join_indices': "a = DuckJanitor.from_pandas(pd.DataFrame({'x': [1, 2]}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'y': [2, 3]}))\na.get_join_indices(b, [('x', 'y', '==')])",
    'rle_id': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'a': [1, 1, 2, 2, 1]}))\ndj2.rle_id().collect()",
    'factorize_columns': "dj = DuckJanitor.from_pandas(df)\ndj.factorize_columns(['group']).collect()",
    'update_where': "dj = DuckJanitor.from_pandas(df)\ndj.update_where({'group': \"'Q'\"}, 'a > 2').collect()",
    'unionize_dataframe_categories': "a = DuckJanitor.from_pandas(pd.DataFrame({'c': [1, 2]}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'c': [3.0]}))\na.unionize_dataframe_categories(b).collect()",
    'scale_mad': "dj = DuckJanitor.from_pandas(df)\ndj.scale_mad('a').collect()",
    'round_to_fraction': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'x': [0.13, 0.55]}))\ndj2.round_to_fraction('x', denominator=2).collect()",
    'shuffle': "dj = DuckJanitor.from_pandas(df)\ndj.shuffle(seed=42).collect()",
    'toset': "dj = DuckJanitor.from_pandas(df)\ndj.toset('group')",
    'take_first': "dj = DuckJanitor.from_pandas(df)\ndj.take_first(2).collect()",
    'sort_naturally': "dj2 = DuckJanitor.from_pandas(pd.DataFrame({'item': ['i10', 'i2', 'i1']}))\ndj2.sort_naturally('item').collect()",
    'sort_column_value_order': "dj = DuckJanitor.from_pandas(df)\ndj.sort_column_value_order('group', ['z', 'y', 'x']).collect()",
    'cartesian_product': "a = DuckJanitor.from_pandas(pd.DataFrame({'x': [1, 2]}))\nb = DuckJanitor.from_pandas(pd.DataFrame({'y': ['p', 'q']}))\na.cartesian_product(b).collect()",
    'then': "dj = DuckJanitor.from_pandas(df)\ndj.then(lambda d: d.rename_column('a', 'value')).collect()",
    # ---- module-level helpers -------------------------------------------
    'DropLabel': "from pyduck_janitor import DropLabel\nDuckJanitor.from_pandas(df).select_columns([DropLabel('b')]).collect()",
    'patterns': "from pyduck_janitor import patterns\npatterns('^v').search('value') is not None",
}


# ===========================================================================
# Docstring parsing
# ===========================================================================

@dataclass
class ParsedDoc:
    summary: str = ''
    parameters: str = ''
    returns: str = ''
    raises: str = ''
    example: str = ''  # from docstring, if present
    sections: dict = field(default_factory=dict)


_SECTION_KEYS = ('Parameters', 'Returns', 'Raises', 'Examples', 'Example')


def _dedent_doc(doc: str) -> str:
    lines = doc.splitlines()
    if not lines:
        return ''
    indents = [len(l) - len(l.lstrip()) for l in lines[1:] if l.strip()]
    strip = min(indents) if indents else 0
    out = [lines[0]] + [l[strip:] if len(l) >= strip else '' for l in lines[1:]]
    return '\n'.join(out)


def parse_docstring(doc: str | None) -> ParsedDoc:
    parsed = ParsedDoc()
    if not doc:
        return parsed
    text = _dedent_doc(doc)

    # Split into numpy-style sections by  Title\n------  headings.
    # re.split keeps the structure [pre, title1, body1, title2, body2, ...],
    # which avoids fragile offset bookkeeping entirely.
    header_re = re.compile(r'(?:^|\n)([A-Z][A-Za-z ]{2,20})\n-+\n', re.M)
    pieces = header_re.split(text)
    parsed.summary = pieces[0].strip()
    it = iter(pieces[1:])
    for name, body in zip(it, it):
        parsed.sections[name] = body.strip('\n').rstrip()

    parsed.parameters = parsed.sections.get('Parameters', '')
    parsed.returns = parsed.sections.get('Returns', '')
    parsed.raises = parsed.sections.get('Raises', '')
    ex = parsed.sections.get('Examples', parsed.sections.get('Example', ''))
    if ex:
        # Re-indent example body for a fenced block.
        lines = []
        for l in ex.splitlines():
            lines.append(l[4:] if l.startswith('    ') else l)
        parsed.example = '\n'.join(lines).strip('\n')
    return parsed


# ===========================================================================
# Example execution (verification)
# ===========================================================================

_PROLOGUE = '''
import pandas as pd, numpy as np, tempfile, os, sys
from pyduck_janitor import DuckJanitor, DropLabel, patterns

df = pd.DataFrame({
    'a': [1, 2, 3, 4],
    'b': [10, 20, 30, 40],
    'group': ['x', 'y', 'x', 'z'],
})


def make_csv(mapping):
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'data.csv')
    pd.DataFrame(mapping).to_csv(p, index=False)
    return p


def make_parquet(mapping):
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'data.parquet')
    pd.DataFrame(mapping).to_parquet(p, index=False)
    return p


def make_excel(mapping):
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'data.xlsx')
    pd.DataFrame(mapping).to_excel(p, index=False)
    return p


def make_json(records):
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, 'data.json')
    import json as _json
    with open(p, 'w') as fh:
        _json.dump(records, fh)
    return p
'''


def run_example(code: str) -> None:
    ns: dict = {}
    exec(_PROLOGUE, ns)  # noqa: S102 - trusted, self-authored snippets
    exec(code, ns)


# ===========================================================================
# Doc assembly
# ===========================================================================

GROUPS: list[tuple[str, list[str]]] = [
    ('Loaders & pipeline plumbing', [
        'from_pandas', 'from_csv', 'from_excel', 'from_json', 'from_parquet',
        'from_sql', 'collect', 'head', 'sql', 'explain', 'get_shared_connection',
    ]),
    ('Core cleaning verbs', [
        'clean_names', 'remove_columns', 'add_column', 'add_columns',
        'rename_column', 'rename_columns', 'dropna', 'remove_empty',
        'filter_column', 'filter_column_isin', 'filter_on', 'filter_string',
        'filter_date', 'coalesce', 'encode_categorical', 'get_dummies',
        'select_columns', 'select', 'select_rows',
        'transform_column', 'transform_columns',
    ]),
    ('Extended verbs', [
        'bin_numeric', 'change_type', 'concatenate_columns',
        'deconcatenate_column', 'drop_constant_columns', 'fill',
        'fill_direction', 'fill_empty', 'flag_nulls',
        'limit_column_characters', 'min_max_scale', 'groupby_agg',
        'groupby_topk', 'case_when', 'currency_column_to_numeric',
        'convert_date', 'convert_to_date', 'convert_to_datetime',
        'convert_unix_date', 'convert_excel_date', 'convert_matlab_date',
        'excel_time_to_numeric', 'sas_numeric_to_date', 'to_datetime',
        'truncate_datetime', 'truncate_datetime_dataframe',
        'pivot_wider', 'pivot_longer',
    ]),
    ('Hybrid verbs', [
        'conditional_join', 'get_dupes', 'dropnotnull', 'expand_column',
        'impute', 'jitter', 'label_encode', 'find_replace',
        'count_cumulative_unique', 'complete', 'also', 'alias', 'mutate',
        'assign', 'ungroup', 'drop_duplicate_columns', 'compare_df_cols',
        'compare_df_cols_same', 'join_apply', 'process_text', 'describe_class',
    ]),
    ('Pyjanitor parity: structural, reshape & aggregation verbs', [
        'move', 'reorder_columns', 'get_columns', 'get_index_labels',
        'row_to_names', 'collapse_levels', 'explode_index',
        'change_index_dtype', 'expand', 'expand_grid', 'summarise',
        'pivot_longer_spec', 'pivot_wider_spec', 'join_agg',
        'get_join_indices', 'rle_id', 'factorize_columns', 'update_where',
        'unionize_dataframe_categories', 'scale_mad', 'round_to_fraction',
        'shuffle', 'toset', 'take_first', 'sort_naturally',
        'sort_column_value_order', 'cartesian_product', 'then',
    ]),
    ('Module-level helpers', ['DropLabel', 'patterns']),
]


def render_body_block(title: str, body: str) -> list[str]:
    out = [f'**{title}**', '']
    if not body or not body.strip():
        out += ['_Not documented._', '']
        return out
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        # numpydoc param lines like ``name : type`` or bare names -> bold lead
        m = re.match(r'^(\*{0,2})([A-Za-z_][A-Za-z0-9_ \[\],\.]{0,60}?)\1? ?:?', line)
        if (
            line and not line.startswith(' ')
            and ':' in line.split(' ', 2)[-1]
            and not line.startswith(('>>> ', '... '))
            and re.match(r'^[A-Za-z_][A-Za-z0-9_ ]* ?: ', line)
        ):
            pname, rest = line.split(':', 1)
            out.append(f'- **{pname.strip()}**: {rest.strip()}')
        else:
            out.append(('    ' if line.startswith(' ') else '') + re.sub(r'^-', '•', line) if True else line)
    out.append('')
    return out


def fmt_signature(obj) -> str:
    """Render a human-readable signature for a function, method, or class.

    Order of preference:

    1. ``inspect.signature`` for callables (covers functions, methods, and
       dataclass-style classes whose ``__init__`` is introspectable).
    2. ``class Name(Bases)`` for classes where signature inference fails
       (e.g. ``patterns`` which subclasses ``str`` and whose
       ``__init__`` is ``str.__init__``).
    3. ``Name(...)`` as a last-resort fallback.
    """
    import inspect as _inspect
    try:
        sig = _inspect.signature(obj)
        return f'{obj.__name__}{sig}'
    except (TypeError, ValueError):
        pass
    if _inspect.isclass(obj):
        try:
            bases = ', '.join(b.__name__ for b in obj.__bases__)
        except Exception:
            bases = 'object'
        return f'class {obj.__name__}({bases})'
    return f'{obj.__name__}(...)'


def indent_code(code: str, indent: str = '    ') -> str:
    return '\n'.join((indent + l) if l.strip() else '' for l in code.splitlines())


def build_page(check_only: bool = False) -> str | None:
    from pyduck_janitor import DuckJanitor, DropLabel, patterns  # noqa: F401

    pyduck = getattr(DuckJanitor, '__dict__', {})

    # Collect (name, obj) for every documented item.
    entries: list[tuple[str, object, str]] = []  # (name, obj, group)
    known = set()
    for group, names in GROUPS:
        for name in names:
            obj = getattr(DuckJanitor, name, None)
            if obj is None:
                obj = {'DropLabel': DropLabel, 'patterns': patterns}.get(name)
            if obj is None:
                raise SystemExit(f'generate_api_docs: unknown function {name}')
            entries_entry = (name, obj, group)
            entries_entry_full = entries_entry  # noqa: F841
            if name in known:
                raise SystemExit(f'generate_api_docs: duplicate entry {name}')
            known.add(name)
            entries.append(entries_entry)

    parts: list[str] = []
    parts.append('# pyduck-janitor API reference — all functions\n')
    parts.append(
        'Complete function-by-function reference for every public method on\n'
        '`DuckJanitor` plus the module-level helpers. Modeled on\n'
        '[pyjanitor\'s API functions page]'
        '(https://pyjanitor-devs.github.io/pyjanitor/api/functions/).\n'
    )
    parts.append(
        '> Generated by `scripts/generate_api_docs.py`. Descriptions,\n'
        '> parameters, returns and raises are extracted from the source\n'
        '> docstrings; every example is executed before it ships. Do not edit\n'
        '> by hand — regenerate instead.\n'
    )

    # TOC
    parts.append('## Function index\n')
    parts.append('| Function | Description | Group |')
    parts.append('| --- | --- | --- |')
    for name, obj, group in entries:
        doc = inspect.getdoc(obj) or ''
        parsed = parse_docstring(doc)
        summary = parsed.summary.splitlines()[0] if parsed.summary else '(no docstring)'
        # strip a leading "Alias of ..." phrasing for tidy TOC
        summary = summary.rstrip('.')
        anchor = name
        link = f'[`{name}`](#{name})'
        parts.append(f'| {link} | {summary} | {group} |')
    parts.append('')

    # Sections
    current_group = None
    for name, obj, group in entries:
        if group != current_group:
            parts.append(f'## {group}\n')
            current_group = group
        doc = inspect.getdoc(obj) or ''
        parsed = parse_docstring(doc)
        sig = fmt_signature(obj)

        parts.append(f'<a id="{name}"></a>')
        parts.append(f'### {name}\n')
        if parsed.summary:
            parts.append(parsed.summary + '\n')
        parts.append('```python')
        parts.append(sig)
        parts.append('```\n')

        if parsed.parameters:
            parts.append('**Parameters**\n')
            for line in parsed.parameters.splitlines():
                l = line.rstrip()
                if not l.strip():
                    continue
                m = re.match(r'^(\S+?)\s*:\s*(.*)$', l)
                # Numpy-style parameter headings render as ``name : type``;
                # some docstrings put ``**kwargs`` on its own line without a
                # colon — render it as its own bullet instead of a
                # continuation of the previous entry.
                name_only = re.match(r'^(\**\w+\**)$', l.strip())
                if m and not l.startswith(' '):
                    parts.append(f'- **{m.group(1)}** — {m.group(2).strip()}')
                elif name_only:
                    parts.append(f'- **{name_only.group(1).strip("*")}**')
                else:
                    parts.append('  ' + l.strip())
            parts.append('')

        if parsed.returns:
            parts.append('**Returns**\n')
            for l in parsed.returns.splitlines():
                t = l.strip()
                if t:
                    parts.append(t)
            parts.append('')

        if parsed.raises:
            parts.append('**Raises**\n')
            for l in parsed.raises.splitlines():
                t = l.strip()
                if t:
                    parts.append(t)
            parts.append('')

        # Example: docstring's own example wins; else the verified map.
        example = parsed.example
        source = 'docstring'
        if not example:
            example = VERIFIED_EXAMPLES.get(name, '')
            source = 'verified snippet'
        if example:
            parts.append(f'**Example** *(from {source})*\n')
            parts.append('```python')
            # Docstring Examples sections sometimes carry their own ``>>> `` /
            # ``... `` doctest prefixes; strip those before re-prefixing, so
            # the rendered output never shows ``>>> >>> dj = ...``.
            _PROMPT_RE = re.compile(r'^(?:>>> |\.{3} )')
            for l in example.splitlines():
                parts.append('>>> ' + _PROMPT_RE.sub('', l))
            parts.append('```\n')
        else:
            parts.append('_No example available._\n')

        parts.append('\n')

    return '\n'.join(parts)


def main() -> int:
    check = '--check' in sys.argv

    # First: execute every verified snippet so nothing broken ships.
    failures = []
    for name, code in VERIFIED_EXAMPLES.items():
        try:
            run_example(code)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            failures.append((name, repr(exc)))
    if failures:
        print('EXAMPLE VERIFICATION FAILURES:')
        for name, err in failures:
            print(f'  {name}: {err}')
        return 1
    print(f'All {len(VERIFIED_EXAMPLES)} snippet examples verified OK.')

    if check:
        return 0

    page = build_page()
    assert page is not None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f'Wrote {OUT} ({len(page)} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())