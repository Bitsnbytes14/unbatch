"""CLAUDE.md invariant 7: ground_truth.json is read ONLY by metrics.py. This
scans every module under src/unbatch/ (not just stages/) for any import or
string literal referencing ground_truth — a scoring leak could just as
easily creep into cli.py's cascade runner as into a stage.

generate.py is exempt: it is the documented *writer* of ground_truth.json
(ARCHITECTURE.md's data flow), never reads it back for any decision, and
the invariant this guards is specifically about reading it back into the
pipeline — CLAUDE.md's own wording is "ground truth is never read by the
pipeline. Only by metrics.py.\""""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "unbatch"
_ALLOWED_FILENAMES = {"metrics.py", "generate.py"}


def _python_files() -> list[Path]:
    return [path for path in _PACKAGE_ROOT.rglob("*.py") if path.name not in _ALLOWED_FILENAMES]


def _import_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return ([node.module] if node.module else []) + [alias.name for alias in node.names]


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """Identity of every Constant node that is a docstring (the first
    statement of a module/function/class body) — legitimate prose mentions
    of "ground_truth.json" belong here, e.g. "writes data/ground_truth.json"
    describing what generate.py produces, and must not trip this check."""
    ids: set[int] = set()
    doc_holders = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, doc_holders) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def test_no_module_outside_metrics_imports_ground_truth() -> None:
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                if any("ground_truth" in name for name in _import_names(node)):
                    offenders.append(path)
    assert offenders == [], f"modules importing ground_truth outside metrics.py: {offenders}"


def test_no_module_outside_metrics_references_the_ground_truth_filename_in_code() -> None:
    """Catches the leak an import-only check would miss: opening
    "data/ground_truth.json" (or any string literal containing
    "ground_truth") directly in actual code, with no import at all.
    Docstrings are exempt — see _docstring_node_ids — since describing what
    generate.py writes or what metrics.py alone is allowed to read is
    exactly the kind of prose this project's docstrings are full of."""
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        skip = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if id(node) in skip:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "ground_truth" in node.value:
                    offenders.append((path, node.value))
    assert offenders == [], f"modules referencing ground_truth by name: {offenders}"


def test_metrics_module_itself_does_reference_ground_truth_in_code() -> None:
    """Sanity check on the check above: if metrics.py stopped referencing
    ground_truth in actual code (not just its docstring), that test would
    pass vacuously and stop meaning anything."""
    metrics_path = _PACKAGE_ROOT / "metrics.py"
    tree = ast.parse(metrics_path.read_text(encoding="utf-8"), filename=str(metrics_path))
    skip = _docstring_node_ids(tree)
    found = any(
        id(node) not in skip
        and isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "ground_truth" in node.value
        for node in ast.walk(tree)
    )
    assert found, "metrics.py no longer references ground_truth in code — is this check still live?"
