"""Every ``--ntasks-per-node`` directive the script carries has a summary row.

``job_summary_rows``' Nodes row states the rule this file pins: reading the raw
answer "omitted the row, leaving a directive in the script that nothing in the
summary accounted for -- SM-15's shape in miniature ... the summary is what the
user checks the script by."  The row below it had drifted from that rule in one
direction the comment does not cover: ``build_sbatch_script`` emits an
``--ntasks-per-node=1`` *fallback* for a multi-node job that asked for no task
count, and no summary row accounted for it.  A ``--nodes 4`` job -- the whole
point of the fallback -- showed four nodes and said nothing about the four tasks
slurmate had decided to put on them.

The two surfaces cannot disagree with each other here, because ``main.py``'s
summary panel and the wizard's Review step both read ``job_summary_rows``; they
could only both be silent, and were.
"""

from __future__ import annotations

from typing import Any

import pytest

from slurmate.builder import build_from_answers, job_summary_rows

BASE: dict[str, Any] = {"job_name": "j", "partition": "amd"}


def _directive(answers: dict[str, Any]) -> str | None:
    """The value of the script's ``--ntasks-per-node``, or None if absent."""
    hits = [
        line.split("=", 1)[1]
        for line in build_from_answers(answers).splitlines()
        if line.startswith("#SBATCH --ntasks-per-node")
    ]
    assert len(hits) <= 1, f"more than one directive: {hits}"
    return hits[0] if hits else None


def _row(answers: dict[str, Any]) -> str | None:
    """The summary's "Tasks per node" value, or None if there is no such row."""
    hits = [v for k, v in job_summary_rows(answers) if k == "Tasks per node"]
    assert len(hits) <= 1, f"more than one row: {hits}"
    return hits[0] if hits else None


# Both halves of the builder's condition, and both spellings of `nodes` the
# builder tolerates (the CLI coerces to int; a config file or the wizard can
# leave it a string, and build_sbatch_script int()s it defensively).
SHAPES: list[tuple[str, dict[str, Any]]] = [
    ("multi-node, no task count", {"nodes": 4}),
    ("multi-node, nodes a string", {"nodes": "4"}),
    ("many nodes", {"nodes": 128}),
    ("single node", {"nodes": 1}),
    ("single node, string", {"nodes": "1"}),
    ("no nodes answer at all", {}),
    ("explicit task count", {"nodes": 4, "ntasks_per_node": 2}),
    ("explicit task count of 1", {"nodes": 4, "ntasks_per_node": 1}),
    ("explicit count on one node", {"nodes": 1, "ntasks_per_node": 3}),
    ("custom --ntasks suppresses it", {"nodes": 4, "custom_sbatch": "--ntasks=8"}),
    ("custom --ntasks, explicit count", {"nodes": 4, "ntasks_per_node": 2,
                                         "custom_sbatch": "--ntasks=8"}),
]


class TestTheSummaryAccountsForEveryTaskDirective:
    @pytest.mark.parametrize("name,extra", SHAPES, ids=[s[0] for s in SHAPES])
    def test_a_directive_and_a_row_appear_together(
        self, name: str, extra: dict[str, Any]
    ) -> None:
        """The stated rule, both ways: no unaccounted directive, no phantom row."""
        answers = {**BASE, **extra}
        directive, row = _directive(answers), _row(answers)
        assert (directive is None) == (row is None), (
            f"{name}: script says {directive!r} but the summary row is {row!r}"
        )

    def test_the_fallback_directive_is_named_in_the_summary(self) -> None:
        """The case the rule was drifting on: a multi-node job with no count."""
        answers = {**BASE, "nodes": 4}
        assert _directive(answers) == "1"
        row = _row(answers)
        assert row is not None, "the fallback directive has no summary row"
        assert row.startswith("1")

    def test_the_automatic_value_says_it_is_automatic(self) -> None:
        """A bare "1" would read as a value the user typed; this one is not."""
        row = _row({**BASE, "nodes": 4})
        assert row is not None and row != "1", (
            "the fallback row must distinguish itself from a typed 1"
        )
        assert "automatic" in row


class TestControls:
    """Each of these passes with the fallback row removed as well as with it.

    They are about the rows the fix did not touch, so a neuter of the new
    ``else`` branch must leave all three green -- verified by running it.
    """

    def test_a_typed_task_count_is_shown_verbatim(self) -> None:
        """An answered value is the user's own and is never annotated."""
        assert _row({**BASE, "nodes": 4, "ntasks_per_node": 2}) == "2"
        assert _row({**BASE, "nodes": 4, "ntasks_per_node": 1}) == "1"

    def test_a_single_node_job_has_neither(self) -> None:
        """The fallback needs `nodes > 1`; one node emits no directive."""
        answers = {**BASE, "nodes": 1}
        assert _directive(answers) is None
        assert _row(answers) is None

    def test_a_custom_ntasks_suppresses_the_fallback(self) -> None:
        """`--custom-sbatch=--ntasks=N` wins, so there is nothing to disclose."""
        answers = {**BASE, "nodes": 4, "custom_sbatch": "--ntasks=8"}
        assert _directive(answers) is None
        assert _row(answers) is None
