"""The 3.10 TOML backfill is a two-sided contract, and nothing read either side.

`pyproject.toml` declares `requires-python = ">=3.10"` and classifiers for 3.10
through 3.14. `tomllib` arrived in **3.11**, so on the oldest interpreter this
package claims to support it does not exist -- and the package reads TOML config
files. Three things make that work, and all three were unchecked:

1. `pyproject.toml` ships the backfill: `"tomli>=2.0; python_version < '3.11'"`.
   The `3.11` in that marker and the version `tomllib` entered the stdlib are the
   same number, and nothing tied them together. Bump the marker to `< '3.12'`,
   or drop the line, and a 3.10 install silently loses its parser.
2. Every `tomllib` import is inside a `try:` with a `tomli` fallback --
   `system_utils.py:4689` and `test_system_utils.py:671`, the latter commenting
   the fallback as "3.10 (declared dependency)", which is what names the
   contract. An unguarded one raises `ModuleNotFoundError` on 3.10.
3. When neither is importable the loader falls back to `_parse_config_naive`
   rather than failing, which is the third tier and is what
   `TestNaiveConfigParserParity` exists to keep honest.

Nothing here changes behaviour: measured first, all three hold today. This is
the check, and slurmate was the last of the five packages without one --
nodetop has `test_readme.py`, rapidu `test_py36_compat.py` (floor 3.6),
slurmpast `test_portability.py`, slurmwatch `test_python_floor_is_enforced.py`.

`sys.stdlib_module_names` cannot answer "was this in 3.10", being the *running*
interpreter's, so the table below is the alternative and is deliberately small.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: module -> the version it entered the stdlib.
ADDED_IN = {
    "tomllib": (3, 11),
    "dbm.sqlite3": (3, 13),
    "annotationlib": (3, 14),
    "compression": (3, 14),
}


def _pyproject() -> str:
    """Read as text, deliberately -- `tomllib` is the very thing under test."""
    return (ROOT / "pyproject.toml").read_text()


def _floor() -> tuple:
    spec = re.search(r'requires-python\s*=\s*"[>=~^]*([\d.]+)"', _pyproject())
    assert spec, "pyproject no longer declares requires-python"
    return tuple(int(part) for part in spec.group(1).split("."))


def _sources() -> list:
    """Every module the sweep reads -- excluding THIS file.

    It quotes `import tomllib` in its own docstring and plants one in a vacuity
    guard, so a scanner that does not exclude itself finds its own explanation.
    Fourth time that shape has bitten in this campaign; apply it when writing
    the scanner, not after the failure.
    """
    here = pathlib.Path(__file__).resolve()
    return [
        path
        for path in sorted(
            list((ROOT / "src" / "slurmate").glob("*.py"))
            + list((ROOT / "tests").glob("*.py"))
        )
        if path.resolve() != here
    ]


def _guarded(tree) -> set:
    """Names imported inside a `try:` -- a backfill, not an unguarded import."""
    safe = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import):
                safe.update(alias.name for alias in inner.names)
            elif isinstance(inner, ast.ImportFrom) and inner.module:
                safe.add(inner.module)
    return safe


def _tomllib_guards(tree) -> list:
    """`(lineno, falls_back_to_tomli)` for each `try:` importing `tomllib`."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = set()
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Import):
                    body.update(alias.name for alias in sub.names)
        if "tomllib" not in body:
            continue
        handled = set()
        for handler in node.handlers:
            for sub in ast.walk(handler):
                if isinstance(sub, ast.Import):
                    handled.update(alias.name for alias in sub.names)
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    handled.add(sub.module)
        out.append((node.lineno, "tomli" in handled))
    return out


def _imports(tree) -> list:
    """`(name, lineno)` for every absolute import in the tree."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.module, node.lineno))
    return found


class TestTheBackfillIsDeclared:
    def test_the_marker_boundary_is_where_tomllib_entered_the_stdlib(self):
        """The two numbers nothing tied together."""
        marker = re.search(
            r'"tomli>=[\d.]+;\s*python_version\s*<\s*[\'"](\d+\.\d+)[\'"]"', _pyproject()
        )
        assert marker, "the tomli backfill marker is gone from pyproject.toml"
        boundary = tuple(int(part) for part in marker.group(1).split("."))
        assert boundary == ADDED_IN["tomllib"], (boundary, ADDED_IN["tomllib"])

    def test_the_backfill_is_needed_at_the_declared_floor(self):
        """If the floor ever reaches 3.11 the marker, the guards and this file
        all become dead weight -- so say which way round it is."""
        assert _floor() < ADDED_IN["tomllib"], _floor()

    def test_the_floor_is_also_a_declared_classifier(self):
        """A floor no classifier mentions is a claim only half made."""
        floor = ".".join(str(part) for part in _floor())
        assert f'"Programming Language :: Python :: {floor}"' in _pyproject()


class TestEveryLateImportIsGuarded:
    def test_no_module_newer_than_the_floor_is_imported_unguarded(self):
        floor = _floor()
        late = {name: added for name, added in ADDED_IN.items() if added > floor}
        assert late, "the floor has moved past every module in the table; prune it"
        offenders = []
        for path in _sources():
            tree = ast.parse(path.read_text())
            guarded = _guarded(tree)
            for name, lineno in _imports(tree):
                if name in late and name not in guarded:
                    offenders.append(f"{path.name}:{lineno} imports {name}")
        assert offenders == []

    def test_every_tomllib_guard_falls_back_to_tomli(self):
        """A guard that catches the error and does nothing is worse than none:
        it turns a loud `ModuleNotFoundError` into a silent loss of the parser.
        Both sites name `tomli`, which is the dependency the marker ships.

        Checked on the AST, not on the file's text. The first version of this
        asserted `"tomli" in source.replace("tomllib", "")` and **survived a
        neuter that deleted both real lines**, because two docstrings two
        hundred lines up say "``tomllib`` on 3.11+, ``tomli`` on older Pythons"
        and "real TOML (tomllib/tomli)". A file-wide substring check measures
        the prose, not the code.
        """
        sites = {}
        for path in _sources():
            for lineno, has_fallback in _tomllib_guards(ast.parse(path.read_text())):
                sites[f"{path.name}:{lineno}"] = has_fallback
        assert sites, "no guarded tomllib import found at all"
        assert sorted(name for name, ok in sites.items() if not ok) == [], sites
        assert sorted(n.split(":")[0] for n in sites) == [
            "system_utils.py",
            "test_system_utils.py",
        ], sites

    def test_the_detector_would_catch_a_planted_import(self):
        """Vacuity guard: the sweep above passes on a clean tree either way."""
        planted = ast.parse("import tomllib\n")
        assert _guarded(planted) == set()
        assert [n for n, _ in _imports(planted)] == ["tomllib"]


class TestControls:
    """Each passes with this whole file's assertions removed as well as with
    them -- this round adds a check rather than changing behaviour, so they
    cover the facts it rests on and must survive any neuter of the sweep.
    """

    def test_the_guarded_import_still_works_on_this_interpreter(self):
        """The chain's first tier, exercised rather than asserted about."""
        from slurmate.system_utils import _parse_config_naive

        assert callable(_parse_config_naive)

    def test_the_naive_parser_is_the_third_tier(self):
        """What answers when neither module is importable. Its parity with
        tomllib is `TestNaiveConfigParserParity`'s subject, not this file's."""
        from slurmate.system_utils import _parse_config_naive

        assert _parse_config_naive('cpus = 4\nname = "x"\n') == {"cpus": 4, "name": "x"}

    def test_the_source_list_is_not_empty(self):
        """Vacuity guard: an empty list makes every sweep above pass trivially."""
        paths = _sources()
        assert len(paths) > 20, len(paths)
        assert any(p.name == "system_utils.py" for p in paths)
        assert any(p.name.startswith("test_") for p in paths)

    def test_the_floor_reader_finds_the_declaration(self):
        assert "requires-python" in _pyproject()
        assert _floor() >= (3, 0)
