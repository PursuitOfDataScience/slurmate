"""A partition that exists, is submittable, and is described by nothing.

`fetch_partitions()` runs a plain ``sinfo`` (so the picker offers what the user
can see) while `_fetch_all_partition_names_uncached` runs ``sinfo -a`` (so a
user-supplied ``--partition`` is validated against the widest list the
controller will give).  Both widths are deliberate and documented.  The gap
between them was not: ``slurmctld``'s ``pack_all_part`` drops a ``Hidden=YES``
partition from a plain query unless the caller is an Operator, so for an
ordinary user a name can be **in** the wide list and **absent** from the narrow
one.  Existence validation passes; the partition object is the synthetic blank.

Measured on midway3 (Slurm 20.11.8, ``PrivateData=none``, 88 partitions, three
of them ``Hidden=YES``: ``test``, ``climate``, ``climate-build``).  ``test`` is
``State=DOWN`` with ``AllowAccounts=none``, and ``sbatch --test-only -A
rcc-staff -p test`` refuses it: *"Invalid account or account/partition
combination specified"*.  With the hidden partitions filtered back out of the
plain ``sinfo``, ``slurmate --print -p test`` exited **0** and said exactly one
thing:

    Capacity limits NOT checked: partition 'test' is not on this cluster ...

— which contradicts the existence check that had just passed in the same run,
and is the false half of the contradiction.  Two things were wrong:

* the synthetic record had only two reasons, ``absent`` and ``unreadable``, so
  "exists but this view does not describe it" was reported as "not on this
  cluster" — the SM-4 false rejection, in the one place that had been fixed for
  it twice already;
* ``fetch_account_acl`` ran ``scontrol show partition <name>`` without ``-a``,
  which answers "Partition test not found" (rc 1) for an ordinary user.  That
  is indistinguishable from a broken ``scontrol``, so the ACL came back empty
  and the ``AllowAccounts=none`` refusal — the accurate, actionable message —
  went unsaid.
"""

from __future__ import annotations

import pytest
from rich.console import Console

import slurmate.main as m
from slurmate import system_utils as su

#: The plain-``sinfo`` rows an ordinary user gets: ``visible`` only.  Shaped like
#: midway3's real output (``%P|%l|%D|%a|%c|%m|%G|%T``).
_VISIBLE_ROWS = "visible|1-00:00:00|24|up|128|250000|(null)|idle\n"

#: ``sinfo -a -h -o %P`` — the controller's widest list, hidden ones included.
_ALL_NAMES = "visible\nhidden-part\n"

#: What ``scontrol -a show partition hidden-part -o`` returns.  ``AllowAccounts``
#: is midway3's ``test``: the sentinel no account can satisfy.
_HIDDEN_SCONTROL = (
    "PartitionName=hidden-part AllowGroups=ALL AllowAccounts=none AllowQos=ALL "
    "Default=NO Hidden=YES MaxTime=UNLIMITED State=DOWN TotalCPUs=36504 "
    "TotalNodes=611\n"
)
_VISIBLE_SCONTROL = (
    "PartitionName=visible AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL "
    "Default=YES Hidden=NO MaxTime=UNLIMITED State=UP TotalCPUs=3072 "
    "TotalNodes=24\n"
)

#: A second hidden partition, shaped like midway3's ``climate``: hidden, **up**,
#: an account this user does hold (``AllowAccounts=rcc-staff,lenovo``) and one
#: real QoS name in ``AllowQos`` rather than the ``none`` sentinel. ``test``
#: cannot show what the QoS ACL costs — its ``AllowQos=none`` is dropped as a
#: sentinel either way — so the partition that can is fixtured separately.
_HIDDEN_QOS_SCONTROL = (
    "PartitionName=hidden-qos AllowGroups=ALL AllowAccounts=rcc-staff,lenovo "
    "AllowQos=lab-qos DenyQos=(null) Default=NO Hidden=YES MaxTime=1-00:00:00 "
    "State=UP TotalCPUs=1280 TotalNodes=32\n"
)

#: The names and ``scontrol`` lines a test needs on top of the two above.  Keys
#: join the ``sinfo -a`` list; the default is empty, so the fixtures the earlier
#: fixes pin are untouched.
_QOS_CASE = {"hidden-qos": _HIDDEN_QOS_SCONTROL}


def _ordinary_user(mocker, *, hidden=("hidden-part",), described=None):
    """Wire the Slurm clients to an ordinary (non-Operator) user's answers.

    The asymmetry is reproduced at the subprocess boundary rather than by
    hand-building a partition dict, so the whole path is under test: the plain
    ``sinfo`` parse, the ``sinfo -a`` name list, and both ``scontrol`` widths.

    ``described`` maps a partition name to the ``scontrol -o`` line it answers
    with, for a test that needs a hidden partition shaped differently from
    ``hidden-part``; its keys are appended to the ``sinfo -a`` name list and are
    hidden from the plain query exactly as ``hidden`` is.
    """
    mocker.patch.object(su, "is_tool_available", return_value=True)
    described = dict(described or {})
    hidden = tuple(hidden) + tuple(n for n in described if n not in hidden)
    all_names = _ALL_NAMES + "".join(f"{n}\n" for n in described)

    def run(cmd, timeout=30, **kwargs):
        cmd = list(cmd)
        wide = "-a" in cmd
        if cmd[0] == "sinfo":
            return (all_names if wide else _VISIBLE_ROWS), "", 0
        if cmd[0] == "scontrol" and "partition" in cmd:
            name = cmd[cmd.index("partition") + 1] if cmd[-1] != "partition" else ""
            if name in hidden and not wide:
                # slurmctld does not return a Hidden=YES partition without
                # SHOW_ALL, and scontrol reports that as an outright miss.
                return "", f"Partition {name} not found", 1
            if name in described:
                return described[name], "", 0
            if name in hidden:
                return _HIDDEN_SCONTROL, "", 0
            return _VISIBLE_SCONTROL, "", 0
        return "", "", 1

    mocker.patch.object(su, "_run_command", side_effect=run)


def _answers(partition, **over):
    a = {
        "partition": partition, "account": "rcc-staff", "cpus": 4, "nodes": 1,
        "memory": "4G", "time_limit": "01:00:00", "command": "true",
        "job_name": "probe",
    }
    a.update(over)
    a["_partition_obj"] = m._get_partition(su.fetch_partitions(), partition)
    return a


def _capacity_note(answers):
    return [
        msg for _lvl, msg in su.validate_job_config(answers) if "NOT checked" in msg
    ]


def _summary_text(answers):
    """Every string the CLI summary surface renders for this record."""
    console = Console(no_color=True, highlight=False, width=300, record=True)
    m._show_script_and_summary(
        console, "#!/bin/bash\n", answers, "1.0",
        {"running": 0, "pending": 0, "eta_seconds": 0, "eta_label": "now",
         "source": "unknown", "feasible": True, "reason": "", "queue_known": True},
    )
    return console.export_text()


# ── Fix A: a partition the narrow view cannot describe is not denied ──────


class TestAHiddenPartitionIsNotCalledAbsent:
    def test_the_two_sinfo_widths_really_do_disagree(self, mocker):
        """The premise, asserted rather than assumed."""
        _ordinary_user(mocker)
        assert [p["name"] for p in su.fetch_partitions()] == ["visible"]
        assert su.fetch_all_partition_names() == {"visible", "hidden-part"}

    def test_the_record_says_undescribed_not_absent(self, mocker):
        _ordinary_user(mocker)
        part = m._get_partition(su.fetch_partitions(), "hidden-part")
        assert part["_unknown"] is True
        assert part["_unknown_reason"] == "undescribed"

    def test_the_capacity_note_does_not_deny_the_partition(self, mocker):
        _ordinary_user(mocker)
        note = _capacity_note(_answers("hidden-part", cpus=999))
        assert note, "a blank partition must still say the limits went unchecked"
        assert "is not on this cluster" not in note[0], note[0]
        assert "exists" in note[0], note[0]

    def test_the_run_does_not_contradict_itself(self, mocker):
        """The finding in one assertion.

        Existence validation consults ``sinfo -a`` and accepts the name; the
        capacity note is built from the plain-``sinfo`` object and denied it.
        Both halves are in the same run's output, and they cannot both be true.
        """
        _ordinary_user(mocker)
        answers = _answers("hidden-part", cpus=999)
        existence = su.validate_cluster_targets(
            "hidden-part", "rcc-staff",
            known_partitions=su.fetch_all_partition_names(),
            known_accounts=["rcc-staff"],
        )
        assert not [msg for lvl, msg in existence if lvl == "error"], existence
        assert "not on this cluster" not in "\n".join(_capacity_note(answers))

    def test_the_summary_surface_does_not_deny_it_either(self, mocker):
        """Three rows read the reason, not one — see the pinned sweep."""
        _ordinary_user(mocker)
        text = _summary_text(_answers("hidden-part", cpus=999))
        assert "not on this cluster" not in text, text
        assert "not described by this cluster's sinfo" in text, text

    @pytest.mark.parametrize(
        "field,value", [("cpus", 999), ("memory", "9000G"), ("nodes", 99),
                        ("gpus", 8), ("time_limit", "99:00:00")],
    )
    def test_every_gated_request_gets_the_honest_reason(self, mocker, field, value):
        # The note is gated on the user having asked for *something*; whichever
        # field opens the gate, the reason must be the same one.
        _ordinary_user(mocker)
        note = _capacity_note(_answers("hidden-part", **{field: value}))
        assert note and "is not on this cluster" not in note[0], (field, note)


# ── Fix B: the ACL of a partition the user named is readable ──────────────


class TestAHiddenPartitionsAccountAclIsRead:
    def test_the_allowaccounts_none_refusal_is_said(self, mocker):
        _ordinary_user(mocker)
        acl = su.fetch_account_acl("hidden-part")
        assert acl["nobody"] is True, acl
        refusal = su.partition_account_refusal("hidden-part", "rcc-staff", acl)
        assert refusal and "AllowAccounts=none" in refusal, refusal

    def test_the_query_asks_for_hidden_partitions(self, mocker):
        # Pinned as a command, not only as an outcome: the outcome can be
        # reproduced by a mock that is too generous, the flag cannot.
        _ordinary_user(mocker)
        su.fetch_account_acl("hidden-part")
        calls = [list(c.args[0]) for c in su._run_command.call_args_list]
        acl_calls = [c for c in calls if c[0] == "scontrol" and "hidden-part" in c]
        assert acl_calls, calls
        assert all("-a" in c for c in acl_calls), acl_calls

    def test_the_wizard_and_cli_both_report_it(self, mocker):
        # site_check_issues is the shared surface; it must carry the refusal for
        # a hidden partition exactly as it does for a visible one.
        _ordinary_user(mocker)
        mocker.patch.object(m, "check_modules", return_value=[])
        issues = m.site_check_issues(_answers("hidden-part"))
        errors = [msg for lvl, msg in issues if lvl == "error"]
        assert any("AllowAccounts=none" in e for e in errors), issues


# ── Fix C: the QoS list of a partition the user named is readable ─────────


class TestAHiddenPartitionsQosListIsRead:
    """``fetch_qos_acl`` had the identical missing ``-a``, one function over.

    Same query, same rc-1 "not found", same empty-ACL fallback — but a different
    consequence, because this value is not an entitlement gate. It feeds exactly
    one thing: the QoS step's choice list (``Wizard._resolve_choices``). So the
    cost is not a missing refusal, it is a **missing choice**.

    Measured on midway3 by re-applying the controller's non-privileged filter in
    front of the real Slurm binaries (this account is a Slurm Operator, so the
    plain and the ``-a`` views are identical for it and the narrow view has to be
    simulated).  ``climate`` and ``climate-build`` are ``Hidden=YES State=UP``
    with ``AllowQos=climate``, and ``climate`` allows ``rcc-staff``, so they are
    partitions this user may actually submit to.  The narrow query answered
    ``{'allow': [], 'deny': []}`` where the wide one answers ``{'allow':
    ['climate'], 'deny': []}``, and the picker's rows went from ``['Default
    (none)', 'climate']`` to ``[]`` — which ``_setup_select`` renders as its lone
    ``s.choices`` fallback, "Default (none)".  The one QoS those partitions
    permit was unpickable.  No privilege is needed to reach it: the partition
    step's "Enter partition name manually…" row accepts a hidden name.

    **Not a gate, and deliberately still not one.**  ``AllowQos`` as a *refusal*
    check has been measured and withdrawn on this repo twice (0 partitions
    excluded across 25 sampled users).  This cluster's QoS refusals come from a
    site submit plugin that rewrites ``QOS=<partition>`` before any ACL is
    consulted: ``sbatch --test-only -A rcc-staff -p climate --qos=normal``
    reports ``QOS-Flag: climate`` and ``Verification: ***PASSED***``, i.e. the
    controller accepts the exact combination an ``AllowQos`` check would refuse.
    The controls below pin that the widened query did not become a gate.
    """

    #: What ``sacctmgr`` knows.  The picker filters ``AllowQos`` against this
    #: set, and mock mode's ``MOCK_QOS`` does not contain the fixture's name — so
    #: without this the filter, not the missing ``-a``, would empty the list.
    KNOWN = ["normal", "debug", "lab-qos"]

    def test_the_narrow_query_really_does_refuse_the_name(self, mocker):
        """The premise, asserted at the boundary rather than assumed."""
        _ordinary_user(mocker, described=_QOS_CASE)
        narrow = su._run_command(["scontrol", "show", "partition", "hidden-qos", "-o"])
        wide = su._run_command(["scontrol", "-a", "show", "partition", "hidden-qos", "-o"])
        assert narrow[2] == 1 and "not found" in narrow[1], narrow
        assert "AllowQos=lab-qos" in wide[0], wide

    def test_the_qos_acl_is_read(self, mocker):
        _ordinary_user(mocker, described=_QOS_CASE)
        assert su.fetch_qos_acl("hidden-qos") == {"allow": ["lab-qos"], "deny": []}
        assert su.fetch_qos_for_partition("hidden-qos") == ["lab-qos"]
        # Both hidden shapes, not just the interesting one: ``hidden-part``'s
        # ``AllowQos=ALL`` was equally unreadable and is equally recovered.
        assert su.fetch_qos_acl("hidden-part") == {"allow": ["ALL"], "deny": []}

    def test_the_query_asks_for_hidden_partitions(self, mocker):
        # Pinned as a command, not only as an outcome — the same pin Fix B has,
        # for the same reason: a too-generous mock can fake the outcome, not the
        # flag.
        _ordinary_user(mocker, described=_QOS_CASE)
        su.fetch_qos_acl("hidden-qos")
        calls = [list(c.args[0]) for c in su._run_command.call_args_list]
        qos_calls = [c for c in calls if c[0] == "scontrol" and "hidden-qos" in c]
        assert qos_calls, calls
        assert all("-a" in c for c in qos_calls), qos_calls

    def test_the_picker_offers_the_qos_the_partition_permits(self, mocker):
        """End to end through the wizard's own choice resolution."""
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard

        _ordinary_user(mocker, described=_QOS_CASE)
        mocker.patch.object(t, "fetch_known_qos", return_value=self.KNOWN)
        wizard = Wizard()
        wizard.answers["partition"] = "hidden-qos"
        step = next(s for s in STEPS if s.key == "qos")
        assert wizard._resolve_choices(step) == ["Default (none)", "lab-qos"]

    def test_the_manual_entry_row_reaches_that_step(self, mocker):
        """Reachability without privilege, through the real confirm handler.

        The hidden partition is not in the plain-``sinfo`` picker at all, so the
        only way an ordinary user selects it in the wizard is by typing it — and
        that row sets ``answers["partition"]`` unconditionally, so the QoS step
        that follows is reached with a name the narrow ``scontrol`` denies.
        """
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard

        _ordinary_user(mocker, described=_QOS_CASE)
        mocker.patch.object(t, "fetch_known_qos", return_value=self.KNOWN)
        wizard = Wizard()
        wizard.idx = next(i for i, s in enumerate(STEPS) if s.key == "partition")
        wizard.transient["all_parts"] = su.fetch_partitions()
        wizard.step_cache["partition_sub"] = "text"
        wizard.text_area.text = "hidden-qos"
        wizard._handle_partition_confirm()
        assert wizard.answers["partition"] == "hidden-qos"
        step = next(s for s in STEPS if s.key == "qos")
        assert "lab-qos" in wizard._resolve_choices(step)

    # ── controls: the widened query must change nothing else ──────────

    def test_a_visible_partitions_qos_list_is_unchanged_item_for_item(self, mocker):
        """The control that matters most: ``amd``'s analogue, row for row.

        A partition the narrow query already described answers identically to
        both widths, so its ACL and every picker row it produces — including the
        ``ALL``-expands-to-the-cluster-list behaviour and the order — must be
        byte-identical before and after.
        """
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard

        _ordinary_user(mocker, described=_QOS_CASE)
        assert su.fetch_qos_acl("visible") == {"allow": ["ALL"], "deny": []}
        assert su.fetch_qos_for_partition("visible") == ["ALL"]
        mocker.patch.object(t, "fetch_known_qos", return_value=self.KNOWN)
        wizard = Wizard()
        wizard.answers["partition"] = "visible"
        step = next(s for s in STEPS if s.key == "qos")
        assert wizard._resolve_choices(step) == [
            "Default (none)", "normal", "debug", "lab-qos",
        ]

    def test_the_none_sentinel_is_still_nobody_not_a_name(self, mocker):
        """And midway3's ``test`` shape is untouched: whichever width reaches
        ``AllowQos=none``, it is still dropped rather than offered as a QoS row
        under the picker's own "Default (none)"."""
        import slurmate.tui as t
        from slurmate.tui import STEPS, Wizard

        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(t, "fetch_known_qos", return_value=[])
        mocker.patch.object(
            su, "_run_command",
            return_value=("PartitionName=p AllowQos=none AllowAccounts=none\n", "", 0),
        )
        wizard = Wizard()
        wizard.answers["partition"] = "p"
        step = next(s for s in STEPS if s.key == "qos")
        assert wizard._resolve_choices(step) == []

    def test_allowqos_is_not_a_refusal(self, mocker):
        """The twice-withdrawn gating must NOT be back.

        ``--qos normal`` against ``AllowQos=lab-qos`` is precisely the
        combination a resurrected check would refuse, and precisely the one the
        controller passes (``QOS-Flag: climate``, ``Verification: PASSED``, from
        the site submit plugin).  Every surface that could carry a refusal is
        checked, and the module is checked for the function itself.
        """
        _ordinary_user(mocker, described=_QOS_CASE)
        mocker.patch.object(m, "check_modules", return_value=[])
        answers = _answers("hidden-qos", qos="normal")
        surfaces = (
            su.validate_job_config(answers)
            + m.site_check_issues(answers)
            + su.validate_cluster_targets(
                "hidden-qos", "rcc-staff", qos="normal",
                known_partitions=su.fetch_all_partition_names(),
                known_accounts=["rcc-staff"], known_qos=self.KNOWN,
            )
        )
        assert not [msg for lvl, msg in surfaces if lvl == "error"], surfaces
        assert not [
            msg for _lvl, msg in surfaces
            if "AllowQos" in msg or "DenyQos" in msg
        ], surfaces
        # Structural: no partition-QoS refusal helper exists to be wired in.
        assert not [
            n for n in dir(su) if "qos" in n.lower() and "refusal" in n.lower()
        ], "a partition-QoS refusal was reintroduced"


# ── Controls: neither fix may touch a described or a nonexistent partition ─


class TestControls:
    def test_a_fully_described_partition_is_unaffected(self, mocker):
        """The control that matters: message for message, ``amd``'s analogue.

        A partition the plain ``sinfo`` describes must produce the same object,
        the same (empty) capacity note, the same silent ACL and the same summary
        rows as it always did — the fixes are for the record that has no shape,
        and this one has one.
        """
        _ordinary_user(mocker)
        part = m._get_partition(su.fetch_partitions(), "visible")
        assert "_unknown" not in part and "_unknown_reason" not in part
        assert part["cpus_per_node"] == 128 and part["state"] == "up"
        answers = _answers("visible")
        assert _capacity_note(answers) == []
        assert su.validate_job_config(answers) == []
        acl = su.fetch_account_acl("visible")
        assert acl == {"allow": ["ALL"], "deny": [], "nobody": False}
        assert su.partition_account_refusal("visible", "rcc-staff", acl) is None
        text = _summary_text(answers)
        assert "NOT checked" not in text
        assert "not on this cluster" not in text
        assert "not described" not in text

    def test_a_nonexistent_name_still_gets_the_old_error(self, mocker):
        """The other control: the honest denial must survive.

        ``absent`` is now reserved for a name the *widest* list does not have —
        which is exactly the name that deserves it. Trading a false denial for
        silence would be no fix at all.
        """
        _ordinary_user(mocker)
        part = m._get_partition(su.fetch_partitions(), "no-such-part")
        assert part["_unknown_reason"] == "absent"
        note = _capacity_note(_answers("no-such-part", cpus=999))
        assert note and "'no-such-part' is not on this cluster" in note[0], note
        existence = su.validate_cluster_targets(
            "no-such-part", None, known_partitions=su.fetch_all_partition_names()
        )
        assert any(
            lvl == "error" and "no partition 'no-such-part'" in msg
            for lvl, msg in existence
        ), existence
        assert "not on this cluster" in _summary_text(_answers("no-such-part", cpus=999))

    def test_an_unreadable_list_still_blames_the_list(self, mocker):
        """And the third reason keeps its meaning.

        With no partition list at all, ``sinfo -a`` has nothing to add either, so
        the reason must stay ``unreadable`` rather than becoming ``absent``.
        """
        _ordinary_user(mocker)
        part = m._get_partition([], "hidden-part")
        assert part["_unknown_reason"] == "unreadable"
        note = _capacity_note({**_answers("visible"), "cpus": 999,
                               "_partition_obj": part, "partition": "hidden-part"})
        assert note and "could not be read" in note[0], note
        assert "is not on this cluster" not in note[0], note[0]

    def test_the_wizards_record_matches_the_clis(self, mocker):
        """Both copies of ``_get_partition`` must agree — they always have.

        The wizard is the default interface and the only one with a "type a
        partition name" row, and it is the surface that has been left behind by
        each of the previous fixes to this record.
        """
        from slurmate.tui import _get_partition as tui_get

        _ordinary_user(mocker)
        parts = su.fetch_partitions()
        for name in ("hidden-part", "no-such-part"):
            assert tui_get(parts, name)["_unknown_reason"] == \
                m._get_partition(parts, name)["_unknown_reason"]
        assert tui_get([], "hidden-part")["_unknown_reason"] == "unreadable"
