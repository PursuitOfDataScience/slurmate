"""Every partition query is wide or narrow on purpose, and nothing checked which.

`system_utils` states the rule itself, in `_fetch_all_partition_names_uncached`:

    Every partition name the controller knows, hidden ones included.
    Deliberately wider than :func:`fetch_partitions`, which runs a plain
    ``sinfo`` so the picker offers what the user can see. Rejecting a
    user-supplied ``--partition`` needs the widest list available, or a
    hidden-but-submittable partition (Slurm's ``Hidden=YES`` is a display
    setting, not an ACL) gets reported as "no such partition on this cluster".

So the width follows the QUESTION, not the shape of the call:

* **wide** (`-a` / `sinfo -a`) -- rejecting or describing a name the *user*
  supplied. The controller does not describe a `Hidden=YES` partition to an
  ordinary caller, so a narrow query here answers "not found" and the caller
  cannot tell that from "no scheduler".
* **narrow** -- offering a list. `Hidden=YES` means do not advertise, and a
  picker that lists hidden partitions is advertising them.

This has been got wrong twice, once per ACL fetcher, each time found only by
re-applying the controller's non-privileged filter against a real cluster
(`tests/test_hidden_partition.py` carries both). Those pins are behavioural and
name their fetcher; a *third* site would be born unchecked. This freezes the
classification instead, so a new partition query has to be classified on purpose.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "slurmate" / "system_utils.py"

# function name -> (wide?, why). Both halves are load-bearing: adding `-a` to a
# narrow query is as much a defect as dropping it from a wide one.
EXPECTED = {
    "fetch_partitions": (False, "offers the picker's list -- what the user can see"),
    "_fetch_all_partition_names_uncached": (True, "rejects a user-supplied --partition"),
    "fetch_public_partitions": (False, "offers only publicly-usable partitions"),
    "fetch_account_acl": (True, "describes a user-supplied partition name"),
    "fetch_qos_acl": (True, "describes a user-supplied partition name"),
    "fetch_system_partitions": (False, "advisory list, de-prioritised not hidden"),
}


def partition_queries(source: str):
    """``{function: (wide, argv)}`` for every partition query in `source`.

    Only literal argv elements are read, which is enough: the width flag is
    always a literal, while the partition name is a variable at the two `-o`
    ACL sites and so simply does not appear.
    """
    found = {}

    def walk(node, fn=None):
        for child in ast.iter_child_nodes(node):
            here = child.name if isinstance(child, ast.FunctionDef) else fn
            if isinstance(child, ast.Call):
                func = child.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "_run_command" and child.args:
                    argv = child.args[0]
                    if isinstance(argv, ast.List):
                        lits = [e.value for e in argv.elts if isinstance(e, ast.Constant)]
                        blob = " ".join(str(x) for x in lits)
                        if lits and lits[0] in ("scontrol", "sinfo") and (
                            "partition" in blob or "%P" in blob
                        ):
                            found[here] = ("-a" in lits, lits)
            walk(child, here)

    walk(ast.parse(source))
    return found


class TestEveryPartitionQueryHasTheWidthItsQuestionNeeds:
    def test_the_classification_is_exactly_what_is_expected(self):
        found = partition_queries(SRC.read_text())
        wrong = {
            fn: ("wide" if wide else "narrow", EXPECTED[fn][1])
            for fn, (wide, _argv) in found.items()
            if fn in EXPECTED and wide != EXPECTED[fn][0]
        }
        assert wrong == {}, (
            f"partition query width disagrees with the question it asks: {wrong} -- "
            f"see `_fetch_all_partition_names_uncached`'s docstring for the rule"
        )

    def test_no_query_is_unclassified(self):
        """A new partition query is a decision; it must not arrive silently."""
        found = set(partition_queries(SRC.read_text()))
        assert found == set(EXPECTED), {
            "unclassified": sorted(found - set(EXPECTED)),
            "vanished": sorted(set(EXPECTED) - found),
        }

    def test_both_widths_are_actually_represented(self):
        # Guards against a rule that passes because every site drifted the same way.
        widths = {wide for wide, _ in partition_queries(SRC.read_text()).values()}
        assert widths == {True, False}, widths


class TestControls:
    """Measure the extractor, on planted source, so no edit to system_utils moves them."""

    def test_the_extractor_reads_the_width_off_a_planted_pair(self):
        planted = (
            "def offers():\n"
            "    return _run_command(['scontrol', 'show', 'partition', '-o'])\n"
            "def describes(part):\n"
            "    return _run_command(['scontrol', '-a', 'show', 'partition', part, '-o'])\n"
        )
        found = partition_queries(planted)
        assert found["offers"][0] is False
        assert found["describes"][0] is True

    def test_a_command_that_is_not_a_partition_query_is_ignored(self):
        planted = (
            "def other():\n"
            "    return _run_command(['sacctmgr', '-a', 'show', 'assoc'])\n"
            "def nodes():\n"
            "    return _run_command(['scontrol', '-a', 'show', 'node', '-o'])\n"
        )
        assert partition_queries(planted) == {}

    def test_sinfo_is_found_by_its_format_string_not_the_word_partition(self):
        planted = "def parts():\n    return _run_command(['sinfo', '-a', '-h', '-o', '%P'])\n"
        assert partition_queries(planted) == {"parts": (True, ["sinfo", "-a", "-h", "-o", "%P"])}
