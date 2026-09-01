#!/usr/bin/env python3
"""
ci_review.py — Posts a structured agent-friendly review comment on the active PR.

Outputs three sections, each designed to give an AI agent (or a human) the
minimum context needed to orient to a change without reading every file:

  1. Module map — every public symbol in pyduck_janitor/ with its first-line
     docstring. Re-exported names (from __init__.py) are followed back to their
     defining module so docstrings render correctly.

  2. API surface diff — for PRs, the set of public symbols added/removed
     between the PR base and HEAD. Catches the class of breakage where a
     refactor renames a verb and breaks downstream users without any test
     failing.

  3. Docstring coverage — count of public symbols with vs. without full
     Parameters/Returns sections. Guards against the docstring work from
     2026-08-31 regressing in future PRs.

Posts via `gh api` so no Python GitHub SDK is required; the only runtime
dependency is Python 3.10+ stdlib + git + the gh CLI.

Designed to be run from .github/workflows/ci.yml::review-agent. Runs locally
too: `python scripts/ci_review.py --dry-run` prints the comment body to stdout.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = REPO_ROOT / "pyduck_janitor"
PUBLIC_API_FILES = [
    PACKAGE_DIR / "__init__.py",
    PACKAGE_DIR / "cleaning_ops.py",
    PACKAGE_DIR / "cleaning_ops_extended.py",
    PACKAGE_DIR / "cleaning_ops_final.py",
    PACKAGE_DIR / "duck_janitor.py",
]


# ---------- AST helpers -------------------------------------------------------


def _defined_symbols(py_file: Path) -> dict[str, ast.AST]:
    """Parse a file and return {name: node} for every top-level def/class.
    Used to look up the *defining* file for re-exported symbols in __init__.py.
    Returns empty dict on parse error."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _resolve_symbol(name: str) -> Path | None:
    """Find the file where `name` is actually defined (not re-exported)."""
    for py_file in sorted(PUBLIC_API_FILES):
        if py_file.name == "__init__.py":
            continue
        if name in _defined_symbols(py_file):
            return py_file
    return None


def _public_symbols_from_all(py_file: Path) -> set[str]:
    """Parse a file and return the set of names in its __all__ (or top-level
    defs if __all__ is missing). Falls back to empty set on parse error."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    names: set[str] = set()
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.add(elt.value)
                    return names
        if isinstance(node, ast.AnnAssign) and (
            isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            names = set()
            for elt in node.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
            return names

    # No __all__: fall back to top-level defs + classes
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    }


def _first_docstring_line(name: str) -> str | None:
    """Look up a name's one-line docstring. For re-exports in __init__.py,
    follow to the defining module so the docstring renders."""
    for py_file in PUBLIC_API_FILES:
        node = _defined_symbols(py_file).get(name)
        if node is None:
            continue
        doc = ast.get_docstring(node)
        if not doc:
            return None
        return doc.splitlines()[0].strip()
    return None


def _has_full_docstring(name: str) -> bool:
    """True iff the symbol's docstring (in its defining module) contains both
    'Parameters' and 'Returns' sections."""
    defining = _resolve_symbol(name)
    if defining is None:
        return False
    node = _defined_symbols(defining).get(name)
    if node is None:
        return False
    doc = ast.get_docstring(node) or ""
    return ("Parameters" in doc) and ("Returns" in doc)


# ---------- Section builders --------------------------------------------------


def build_module_map() -> str:
    """Module map: every public symbol with its one-line docstring."""
    lines: list[str] = []
    for py_file in sorted(PUBLIC_API_FILES):
        if not py_file.exists():
            continue
        rel = py_file.relative_to(REPO_ROOT)
        names = sorted(_public_symbols_from_all(py_file))
        lines.append(f"### `{rel}`")
        for name in names:
            doc = _first_docstring_line(name)
            bullet = f"- `{name}`"
            if doc:
                bullet += f" — {doc}"
            lines.append(bullet)
        lines.append("")
    return "\n".join(lines).rstrip()


def build_api_diff(base_ref: str) -> str:
    """Public symbol set before vs. after the PR. Uses git show to read each
    file at base_ref, then compares to the working copy."""
    base_symbols: dict[Path, set[str]] = {}
    for py_file in PUBLIC_API_FILES:
        if not py_file.exists():
            continue
        rel = py_file.relative_to(REPO_ROOT)
        result = subprocess.run(
            ["git", "show", f"{base_ref}:{rel.as_posix()}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # File didn't exist on base_ref (new file in this PR)
            base_symbols[py_file] = set()
            continue
        tmp = REPO_ROOT / f".tmp_{py_file.name}"
        tmp.write_text(result.stdout, encoding="utf-8")
        try:
            base_symbols[py_file] = _public_symbols_from_all(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    added: list[tuple[str, str]] = []
    removed: list[tuple[str, str]] = []
    for py_file in PUBLIC_API_FILES:
        if not py_file.exists():
            continue
        rel = py_file.relative_to(REPO_ROOT)
        before = base_symbols.get(py_file, set())
        after = _public_symbols_from_all(py_file)
        for sym in sorted(after - before):
            added.append((rel.as_posix(), sym))
        for sym in sorted(before - after):
            removed.append((rel.as_posix(), sym))

    if not added and not removed:
        return "_No public-API changes in this PR._"

    parts: list[str] = []
    if added:
        parts.append("**Added:**")
        for rel, sym in added:
            parts.append(f"- `{rel}` → `{sym}`")
    if removed:
        parts.append("**Removed:**")
        for rel, sym in removed:
            parts.append(f"- `{rel}` → `{sym}`")
    return "\n".join(parts)


def build_docstring_coverage() -> str:
    """Count public functions/classes with full Parameters+Returns docstrings.
    Walks the four source modules (not __init__.py) since __init__.py just
    re-exports names — docstrings live on the defining module."""
    total = 0
    covered = 0
    missing: list[str] = []
    source_modules = [f for f in sorted(PUBLIC_API_FILES) if f.name != "__init__.py"]
    seen: set[str] = set()
    for py_file in source_modules:
        if not py_file.exists():
            continue
        rel = py_file.relative_to(REPO_ROOT)
        for name in sorted(_defined_symbols(py_file).keys()):
            if name in seen or name.startswith("_"):
                continue
            seen.add(name)
            total += 1
            if _has_full_docstring(name):
                covered += 1
            else:
                missing.append(f"`{rel}` → `{name}`")
    pct = (covered / total * 100) if total else 0.0
    lines = [f"**Coverage:** {covered}/{total} symbols ({pct:.0f}%) have Parameters + Returns."]
    if missing:
        lines.append("")
        lines.append("Missing or incomplete docstrings:")
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append("")
        lines.append("✅ All public symbols have full docstrings.")
    return "\n".join(lines)


# ---------- Posting -----------------------------------------------------------


def render_comment(base_ref: str | None) -> str:
    parts: list[str] = []
    parts.append("## 🤖 Agent Review Summary\n")
    parts.append("<!-- ci_review.py: do not edit; regenerated on every PR refresh. -->\n")

    parts.append("### 📚 Module Map")
    parts.append(build_module_map())
    parts.append("")

    if base_ref:
        parts.append("### 🔁 API Surface Diff")
        parts.append(f"_Comparing `{base_ref}` → `HEAD`._\n")
        parts.append(build_api_diff(base_ref))
        parts.append("")

    parts.append("### 📝 Docstring Coverage")
    parts.append(build_docstring_coverage())
    parts.append("")

    return "\n".join(parts)


def post_comment(body: str, pr_number: str) -> None:
    """Post or update the bot's own comment on the PR. Looks for an existing
    marker so we update in place rather than spamming on every push."""
    marker = "<!-- ci_review.py: do not edit"
    list_result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{{owner}}/{{repo}}/issues/{pr_number}/comments",
            "--paginate",
            "--jq",
            f'.[] | select(.body | startswith("{marker}")) | .id',
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    existing_id = (
        list_result.stdout.strip().splitlines()[-1] if list_result.stdout.strip() else None
    )

    if existing_id:
        subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "PATCH",
                f"repos/{{owner}}/{{repo}}/issues/comments/{existing_id}",
                "-f",
                f"body={body}",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(f"Updated existing comment {existing_id}", file=sys.stderr)
    else:
        subprocess.run(
            [
                "gh",
                "api",
                "-X",
                "POST",
                f"repos/{{owner}}/{{repo}}/issues/{pr_number}/comments",
                "-f",
                f"body={body}",
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(f"Posted new comment on PR #{pr_number}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the comment body to stdout instead of posting.",
    )
    parser.add_argument(
        "--base-ref",
        default=os.environ.get("BASE_REF"),
        help="Git ref to diff against (e.g. main). Defaults to $BASE_REF.",
    )
    parser.add_argument(
        "--pr-number",
        default=os.environ.get("PR_NUMBER"),
        help="PR number to comment on. Defaults to $PR_NUMBER.",
    )
    args = parser.parse_args(argv)

    body = render_comment(base_ref=args.base_ref)

    if args.dry_run:
        print(body)
        return 0

    if not args.pr_number:
        print("ERROR: --pr-number (or $PR_NUMBER) is required to post.", file=sys.stderr)
        return 2

    post_comment(body, args.pr_number)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
