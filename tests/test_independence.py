"""Independence guard: the package must not depend on the deeptutor monolith.

This is the load-bearing regression test for the extraction. It parses every
``.py`` file in the package with the ``ast`` module and asserts that none of
them import anything under the ``deeptutor`` namespace (whether top-level or
nested inside a function). If this test ever fails, the package has been
re-coupled to the monolith and can no longer be dropped into a fresh repo.
"""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _package_py_files() -> list[Path]:
    files = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        # Skip the test tree itself and any virtualenv that happens to live here.
        parts = set(path.relative_to(PACKAGE_ROOT).parts)
        if "tests" in parts or ".venv" in parts or "venv" in parts:
            continue
        files.append(path)
    return files


def _deeptutor_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "deeptutor" or alias.name.startswith("deeptutor."):
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "deeptutor" or module.startswith("deeptutor."):
                offenders.append(f"from {module} import ...")
    return offenders


def test_package_has_python_files():
    # Guard against the walker silently finding nothing and passing vacuously.
    assert _package_py_files(), "no package .py files discovered"


def test_no_deeptutor_imports_anywhere():
    violations: dict[str, list[str]] = {}
    for path in _package_py_files():
        offenders = _deeptutor_imports(path)
        if offenders:
            violations[str(path.relative_to(PACKAGE_ROOT))] = offenders
    assert not violations, f"deeptutor imports must not exist in the package: {violations}"
