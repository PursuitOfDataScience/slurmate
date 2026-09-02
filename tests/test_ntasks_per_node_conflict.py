"""The auto ``--ntasks-per-node=1`` versus a custom task count.

On a multi-node request with no tasks-per-node of its own, the builder adds
``--ntasks-per-node=1`` so the allocation shape is predictable. slurmate has no
``--ntasks`` option, so ``--custom-sbatch=--ntasks=N`` is the only way to express
an MPI job with it -- :func:`builder.custom_ntasks` says as much, calling it "a
likely path rather than an exotic one" -- and the two directives contradict each
other. Measured on Slurm 20.11.8 with ``sbatch --test-only``, which validates
without submitting:

    --nodes=4 --ntasks=8                          rc=0, accepted
    --nodes=4 --ntasks=8 --ntasks-per-node=1      rc=1, "Requested node
                                                  configuration is not available"

One task per node over eight tasks asks for eight nodes, and ``--nodes`` caps it
at four. So ``slurmate --nodes 4 --custom-sbatch=--ntasks=8`` built a script that
its own pre-submit check then refused, and printed the refusal instead of the
script.

``_custom_mem_override`` already states the rule this restores, for the three
memory directives: "whichever one a custom flag carries, the auto directive must
give way or the controller refuses the script." ``--ntasks-per-node`` was the one
auto directive that did not follow it.
"""

from slurmate.builder import build_sbatch_script


def _script(**kwargs):
    base = dict(
        job_name="t",
        partition="amd",
        cpus=1,
        memory="4G",
        time_limit="1",
        account="rcc-staff",
    )
    base.update(kwargs)
    return build_sbatch_script(**base)


def _directives(script, flag):
    return [ln for ln in script.splitlines() if ln.startswith("#SBATCH %s" % flag)]


def test_the_auto_directive_gives_way_to_a_custom_ntasks():
    """The conflict, in the shape a user reaches it: an MPI job over four nodes."""
    script = _script(nodes=4, custom_sbatch=["--ntasks=8"])
    assert _directives(script, "--ntasks-per-node") == [], script
    assert "#SBATCH --ntasks=8" in script, "the user's own flag is untouched"
    assert "#SBATCH --nodes=4" in script


def test_the_short_form_counts_too():
    """``-n 8`` is the same request; ``custom_ntasks`` already reads both."""
    script = _script(nodes=4, custom_sbatch=["-n 8"])
    assert _directives(script, "--ntasks-per-node") == [], script


def test_control_a_the_multi_node_default_survives():
    """CONTROL, passing with the fix present or absent.

    No custom task count, so nothing contradicts the fallback and the reason it
    was added still holds. A fix that dropped it unconditionally would pass the
    two tests above and fail this one.
    """
    script = _script(nodes=4)
    assert _directives(script, "--ntasks-per-node") == ["#SBATCH --ntasks-per-node=1"], script


def test_control_b_an_explicit_tasks_per_node_is_the_users_instruction():
    """CONTROL, passing in both states. Only the *fallback* gives way.

    Someone who asks for two tasks per node alongside eight tasks has asked for
    four nodes' worth of work and is entitled to the directive they named, even
    though it is the same flag the fallback would have emitted.
    """
    script = _script(nodes=4, ntasks_per_node=2, custom_sbatch=["--ntasks=8"])
    assert _directives(script, "--ntasks-per-node") == ["#SBATCH --ntasks-per-node=2"], script


def test_control_c_a_single_node_never_had_the_directive():
    """CONTROL, passing in both states. The fallback is multi-node only.

    Pins that the fix did not widen the fallback's reach while changing when it
    yields -- a single-node request emits nothing here and emitted nothing before.
    """
    assert _directives(_script(nodes=1), "--ntasks-per-node") == []
    assert _directives(_script(nodes=1, custom_sbatch=["--ntasks=8"]), "--ntasks-per-node") == []


def test_control_d_a_custom_ntasks_per_node_is_a_duplicate_not_a_conflict():
    """CONTROL, passing in both states.

    A custom ``--ntasks-per-node`` repeats a directive slurmate owns rather than
    contradicting one, so it keeps its existing ``_MANAGED_CUSTOM_FLAGS``
    treatment: both lines are emitted and Slurm honours the last. Not this fix's
    business, and pinned here so it does not become so by accident.
    """
    from slurmate.builder import managed_custom_flags

    script = _script(nodes=4, custom_sbatch=["--ntasks-per-node=3"])
    assert "#SBATCH --ntasks-per-node=3" in script
    assert managed_custom_flags(["--ntasks-per-node=3"]) == [
        ("--ntasks-per-node", "--ntasks-per-node")
    ]
