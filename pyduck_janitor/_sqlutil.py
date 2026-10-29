"""Shared SQL helpers for pyduck-janitor.

These utilities centralise the identifier quoting, literal rendering, and
SQL-fragment handling that every module needs. Keeping them in one place
removes the historical duplication between ``cleaning_ops``,
``cleaning_ops_extended``, and ``cleaning_ops_final``.

The helpers here are intentionally conservative: they never guess, and they
raise rather than silently degrade a user expression into a literal.
"""

from __future__ import annotations

import itertools
import re
import threading
from typing import Any

__all__ = [
    "quote_identifier",
    "sql_literal",
    "quote_identifiers",
    "substitute_self",
    "is_safe_identifier",
    "safe_identifier",
    "temp_name",
]

# Monotonic counter for temporary relation names. Using a counter instead of
# id(relation) avoids a real hazard: CPython reuses the memory address of a
# garbage-collected object, so a stale registration named "_temp_<id>" can
# collide with a newly created relation that happens to land on the same
# address. The counter is process-wide and never reused.
_temp_counter = itertools.count()
_temp_lock = threading.Lock()


def temp_name(prefix: str = "_pyduck") -> str:
    """Return a unique temporary SQL name.

    Names are process-unique and never reused, so two live registrations can
    never collide with each other or with a stale one.

    Parameters
    ----------
    prefix : str, optional
        Name prefix. Default is ``'_pyduck'``.

    Returns
    -------
    str
        A unique temporary name such as ``'_pyduck_17'``.
    """
    with _temp_lock:
        return f"{prefix}_{next(_temp_counter)}"


def quote_identifier(name: str) -> str:
    """Return a safely quoted SQL identifier.

    Embedded double quotes are doubled, per the SQL standard.

    Parameters
    ----------
    name : str
        The identifier to quote.

    Returns
    -------
    str
        The quoted identifier, ready to interpolate into SQL.
    """
    return '"' + str(name).replace('"', '""') + '"'


def quote_identifiers(names: list[str]) -> list[str]:
    """Quote every identifier in *names*.

    Parameters
    ----------
    names : list of str
        Identifiers to quote.

    Returns
    -------
    list of str
        The quoted identifiers.
    """
    return [quote_identifier(name) for name in names]


def sql_literal(value: Any) -> str:
    """Return a safely quoted SQL literal.

    Parameters
    ----------
    value : Any
        The value to render. ``None`` becomes ``NULL``, booleans become
        ``TRUE``/``FALSE``, numbers are rendered numerically, and everything
        else becomes a single-quoted string with embedded quotes doubled.

    Returns
    -------
    str
        The SQL literal.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def is_safe_identifier(name: str) -> bool:
    """Return True when *name* is a plain identifier safe to use unquoted.

    Parameters
    ----------
    name : str
        The candidate identifier.

    Returns
    -------
    bool
        True when the name matches ``[A-Za-z_][A-Za-z0-9_]*``.
    """
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(name)))


def safe_identifier(name: str, context: str = "identifier") -> str:
    """Return *name* when it is a plain identifier, else raise.

    Parameters
    ----------
    name : str
        The candidate identifier.
    context : str, optional
        Label used in the error message.

    Returns
    -------
    str
        The unchanged identifier.

    Raises
    ------
    ValueError
        If the name is not a plain identifier.
    """
    if not is_safe_identifier(name):
        raise ValueError(
            f"{context} must be a plain identifier "
            f"(letters, digits, underscore; not starting with a digit). Got {name!r}."
        )
    return str(name)


def substitute_self(query: str, replacement: str) -> str:
    """Replace the bare word ``self`` with *replacement* in *query*.

    The substitution is aware of SQL string literals, quoted identifiers, and
    comments, so ``'self'`` and ``"self"`` are left untouched. This replaces
    the historical regex substitution, which rewrote the word inside string
    literals and silently corrupted queries.

    Parameters
    ----------
    query : str
        The SQL query containing the ``self`` placeholder.
    replacement : str
        The SQL name that replaces ``self``.

    Returns
    -------
    str
        The rewritten query.
    """
    out: list[str] = []
    i = 0
    length = len(query)
    while i < length:
        char = query[i]

        # Single-quoted literal: copy verbatim, doubling '' escapes.
        if char == "'":
            out.append(char)
            i += 1
            while i < length:
                out.append(query[i])
                if query[i] == "'":
                    if i + 1 < length and query[i + 1] == "'":
                        out.append("'")
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue

        # Double-quoted identifier: copy verbatim.
        if char == '"':
            out.append(char)
            i += 1
            while i < length:
                out.append(query[i])
                if query[i] == '"':
                    if i + 1 < length and query[i + 1] == '"':
                        out.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue

        # Line comment.
        if query.startswith("--", i):
            end = query.find("\n", i)
            end = length if end == -1 else end
            out.append(query[i:end])
            i = end
            continue

        # Block comment.
        if query.startswith("/*", i):
            end = query.find("*/", i + 2)
            end = length if end == -1 else end + 2
            out.append(query[i:end])
            i = end
            continue

        # The placeholder word itself.
        if char == "s" and query.startswith("self", i):
            before_ok = i == 0 or not (query[i - 1].isalnum() or query[i - 1] == "_")
            after = i + 4
            after_ok = after >= length or not (query[after].isalnum() or query[after] == "_")
            if before_ok and after_ok:
                out.append(replacement)
                i = after
                continue

        out.append(char)
        i += 1

    return "".join(out)
