"""The wizard's queue strip, against the synthetic blank partition object.

`main._get_partition` / `tui._get_partition` answer a name the plain ``sinfo``
does not list with a synthetic record: every capacity field 0/None, flagged
``_unknown`` with an ``_unknown_reason`` of ``absent`` / ``undescribed`` /
``unreadable``.  Every partition-dependent rule in `validate_job_config` then
has nothing to compare against and stays **silent** — which is the right answer
for a check that could not run — and one warning says so:

    Capacity limits NOT checked: ... the request above has been validated for
    shape only

Measured against a request that trips every rule (cpus 999, nodes 20, mem 500G,
time 30-00:00:00, gpus 99, gpu_type h100), a fully-described partition yields
**eight** findings and the blank yields **one**, the note.  All eight silences
are silences, not passes, so the note covers them; `TestWhatTheBlankCosts`
pins that as the measurement, and it holds both before and after the fix below.

The one surface that did **not** stay silent was this strip.  `_render_queue_text`
read `queue_info` unconditionally, so for an unresolved partition it printed

    Queue status (typo): 0 running / 0 pending   ETA: now

in the under-an-hour GREEN — ``squeue -p <name>``'s empty answer (or a failed
query) rendered as a measurement, and the queue-depth heuristic's flat constant
rendered as a wait time.  The CLI summary settled exactly this and says
``unknown — <reason>`` for both rows; the wizard is the surface that *owns* the
"Enter partition name manually..." row, so it is the one that can reach the
blank record, and it was the one still making the claim.  Separately, a request
Slurm had already refused got its "never" in the same green, because the colour
keys on ``eta_seconds < 3600`` and a refusal carries ``eta_seconds=0``.
"""

from __future__ import annotations

import pytest
from rich.console import Console

import slurmate.main as m
from slurmate import system_utils as su
from slurmate.tui import STEPS, Wizard

# The three colours the strip can choose between, spelled as the renderer spells
# them.  Pinned by value, not by name: the point of the fix is *which* colour a
# refusal and an unknown get, and a test that imported the same constant the
# renderer uses could not tell green from red.
_GREEN = "fg:#54c99a bold"
_AMBER = "fg:#e0b661 bold"
_RED = "fg:#ef6f7e bold"


def _idx(key: str) -> int:
    return next(i for i, s in enumerate(STEPS) if s.key == key)


def _blank(name: str = "typo", reason: str = "absent") -> dict:
    """The synthetic record, field for field as both `_get_partition`s build it."""
    return {"name": name, "nodes": 0, "nodes_up": None, "cpus_per_node": 0,
            "mem_per_node_mb": 0, "gpu_types": [], "timelimit": None,
            "is_public": True, "is_default": False, "_unknown": True,
            "_unknown_reason": reason}


def _described(name: str = "amd") -> dict:
    """midway3's ``amd`` as `fetch_partitions` records it: 40 nodes, no GPUs."""
    return {"name": name, "nodes": 40, "nodes_up": 37, "state": "up",
            "cpus_per_node": 128, "mem_per_node_mb": 250000,
            "heterogeneous": False, "gpu_types": [], "has_gpu": False,
            "gpus_per_node": 0, "timelimit": "infinite", "is_default": False}


def _qinfo(**over) -> dict:
    q = {"running": 0, "pending": 0, "queue_known": True, "eta_seconds": 0,
         "eta_label": "now", "source": "pressure", "feasible": True,
         "reason": "", "refusal_is_permanent": True, "refusal_is_transient": False}
    q.update(over)
    return q


#: A live reading: squeue answered, the scheduler placed the job.
_LIVE = _qinfo(running=7, pending=3, eta_seconds=120, eta_label="~2min",
               source="scheduler")
#: What `fetch_queue_eta` returns when the controller refused the request.
_REFUSED = _qinfo(eta_label="never", source="scheduler", feasible=False,
                  reason="Invalid account or account/partition combination specified")
#: A refusal that clears on its own (a submit-count cap).
_TRANSIENT = _qinfo(eta_label="never", source="scheduler", feasible=False,
                    reason="MaxSubmitJobsPerAccount", refusal_is_permanent=False,
                    refusal_is_transient=True)
#: squeue failed or timed out: 0/0 is not a reading of an idle queue.
_UNREADABLE = _qinfo(queue_known=False)


def _strip(part_obj, qinfo, *, name="typo", idx_key="modules"):
    """The strip's fragments, rendered from a wizard parked past the hardware steps."""
    w = Wizard()
    w.idx = _idx(idx_key)
    w.answers["partition"] = name
    w.answers["_partition_obj"] = part_obj
    w.transient["queue_info"] = qinfo
    w.transient["queue_info_part"] = name
    return w._render_queue_text()


def _row(frags, header):
    """The text of the fragment following the ``header`` label, and its style."""
    for i, (_style, text) in enumerate(frags):
        if text == header:
            return frags[i + 1][1], frags[i + 1][0]
    raise AssertionError(f"no {header!r} row in {frags!r}")


def _queue(frags):
    return _row(frags, "Queue status (typo): ")[0].strip()


def _eta(frags):
    text, style = _row(frags, "ETA: ")
    return text.strip(), style


# ── What the blank costs: measured, and covered by the note ───────────────


class TestWhatTheBlankCosts:
    """The withdrawal, pinned. Holds in both states — the fix is elsewhere."""

    #: cpus 999 / nodes 20 / 500G / 30 days / 99 GPUs / a model the partition
    #: does not have, against a partition that is itself DOWN with every node
    #: drained: one request that trips every partition-dependent rule at once.
    HOSTILE = {"cpus": 999, "nodes": 20, "memory": "500G",
               "time_limit": "30-00:00:00", "gpus": 99, "gpu_type": "h100",
               "command": "true", "job_name": "probe"}
    FULL = {"name": "p", "nodes": 4, "nodes_up": 0, "state": "down",
            "cpus_per_node": 48, "mem_per_node_mb": 192000,
            "heterogeneous": False, "gpu_types": ["a100"], "has_gpu": True,
            "gpus_per_node": 4, "timelimit": "1-00:00:00", "is_default": False}

    def _issues(self, part):
        return su.validate_job_config({**self.HOSTILE, "_partition_obj": part})

    def test_a_described_partition_answers_eight_ways(self):
        assert len(self._issues(self.FULL)) == 8

    def test_the_blank_answers_once_and_it_is_the_note(self):
        issues = self._issues(_blank("p", "undescribed"))
        assert len(issues) == 1
        level, msg = issues[0]
        assert level == "warning"
        assert msg.startswith("Capacity limits NOT checked:")
        assert "validated for shape only" in msg

    @pytest.mark.parametrize("field,value", [
        # Every field the blank lacks or zeroes, and the check it disables.
        ("cpus_per_node", 48),          # CPUs vs the node
        ("mem_per_node_mb", 192000),    # memory vs the node
        ("timelimit", "1-00:00:00"),    # --time vs the partition limit
        ("gpus_per_node", 4),           # --gpus vs what a node advertises
        ("nodes", 4),                   # --nodes vs how many exist
        ("state", "down"),              # the partition's own state
        ("has_gpu", False),             # GPUs on a partition that has none
        ("gpu_types", ["a100"]),        # a GPU model it does not offer
    ])
    def test_each_absent_field_costs_exactly_one_check(self, field, value):
        """Restore one field and exactly one finding wakes up: nothing else moved."""
        before = {msg for _l, msg in self._issues(_blank("p", "undescribed"))}
        after = [msg for _l, msg in self._issues({**_blank("p", "undescribed"),
                                                  field: value})]
        assert len(set(after) - before) == 1
        assert before <= set(after)     # the note is still said

    @pytest.mark.parametrize("field,value", [
        ("cpus_per_node", 0), ("mem_per_node_mb", 0), ("timelimit", None),
        ("nodes", 0), ("nodes_up", None),
    ])
    def test_a_zero_ceiling_is_read_as_unknown_not_as_a_refusal(self, field, value):
        """A 0/None limit must silence its check, never refuse everything.

        The alternative reading — 0 cores is a ceiling of zero — would refuse a
        1-core job on a partition slurmate simply could not describe, which is
        the SM-4 false rejection this module is most careful about.
        """
        part = {**self.FULL, field: value}
        msgs = [msg for _l, msg in su.validate_job_config(
            {"cpus": 1, "nodes": 1, "memory": "1G", "time_limit": "00:10:00",
             "_partition_obj": part}
        )]
        assert not any("exceeds" in m or "does not support" in m for m in msgs)

    def test_the_blank_is_not_reported_as_a_refusal(self):
        """`capacity_refusal` stays empty: no figures means no verdict.

        "" is read by the caller as "nothing to add", not as "this fits" — the
        summary's ETA row is decided by ``_unknown`` before it is consulted.
        """
        assert su.capacity_refusal(_blank("p", "undescribed"), self.HOSTILE) == ""
        assert su.capacity_refusal(self.FULL, self.HOSTILE) != ""


# ── The fix: the queue strip stops presenting a non-reading as a reading ──


class TestTheQueueStripDoesNotClaimAReading:
    @pytest.mark.parametrize("reason", ["absent", "undescribed", "unreadable"])
    def test_an_unresolved_partitions_queue_row_is_not_a_count(self, reason):
        # All three reasons differ in what to *say*; none of them is a reading.
        assert _queue(_strip(_blank(reason=reason), _qinfo())) == "unknown"

    @pytest.mark.parametrize("reason", ["absent", "undescribed", "unreadable"])
    def test_an_unresolved_partitions_eta_is_not_a_wait_time(self, reason):
        text, style = _eta(_strip(_blank(reason=reason), _qinfo()))
        assert text == "unknown"
        assert style == _AMBER

    def test_the_heuristics_flat_constant_is_not_shown_for_a_blank(self):
        """The pressure tier answers even for a partition that is not there."""
        text, _style = _eta(_strip(_blank(), _qinfo(eta_seconds=420,
                                                    eta_label="~7min")))
        assert text == "unknown"

    def test_a_failed_squeue_is_not_an_idle_queue(self):
        assert _queue(_strip(_described(), _UNREADABLE)) == "unknown"

    def test_a_refused_request_is_not_the_under_an_hour_green(self):
        text, style = _eta(_strip(_described(), _REFUSED))
        assert text == "never"
        assert style == _RED

    def test_a_refusal_that_clears_on_its_own_is_amber_not_red(self):
        # Same split the summary's ETA row makes: a submit-count cap is a
        # statement about the moment, not about the request.
        _text, style = _eta(_strip(_described(), _TRANSIENT))
        assert style == _AMBER

    def test_a_refusal_beats_the_unknown_partition_wording(self):
        # Slurm's own verdict is the more specific fact, so it wins the row.
        text, style = _eta(_strip(_blank(), _REFUSED))
        assert (text, style) == ("never", _RED)


# ── Controls ─────────────────────────────────────────────────────────────


class TestControls:
    def test_a_described_partition_with_a_live_reading_is_unchanged(self):
        frags = _strip(_described(), _LIVE)
        assert _queue(frags) == "7 running / 3 pending"
        assert _eta(frags) == ("~2min", _GREEN)

    def test_a_described_partition_over_an_hour_is_still_amber(self):
        _text, style = _eta(_strip(_described(), _qinfo(eta_seconds=7200,
                                                        eta_label="~2h")))
        assert style == _AMBER

    def test_the_strips_shape_is_unchanged(self):
        """Five fragments, one blank line then one content line: still 2 rows.

        The strip's window is a fixed ``height=2``; a row that wrapped would
        push the ETA off the bottom, which is why the fix says "unknown" rather
        than the summary's fuller "unknown — <reason>".
        """
        for part, q in ((_described(), _LIVE), (_blank(), _qinfo()),
                        (_described(), _REFUSED)):
            frags = _strip(part, q)
            assert len(frags) == 5
            assert "".join(t for _s, t in frags).count("\n") == 2
            assert len("".join(t for _s, t in frags).strip()) < 80

    def test_the_strip_stays_empty_before_the_hardware_steps_are_done(self):
        assert _strip(_blank(), _LIVE, idx_key="cpus") == []
        assert _strip(_described(), _LIVE, idx_key="cpus") == []

    def test_no_queue_info_renders_nothing(self):
        w = Wizard()
        w.idx = _idx("modules")
        w.answers["_partition_obj"] = _blank()
        assert w._render_queue_text() == []

    def test_a_missing_partition_obj_is_treated_as_described(self):
        """A blank partition (the legitimate "site default") is not _unknown.

        `_partition_obj` is None there, and the queue really is a reading of the
        site default's queue, so the row must keep reporting it.
        """
        frags = _strip(None, _LIVE)
        assert _queue(frags) == "7 running / 3 pending"
        assert _eta(frags) == ("~2min", _GREEN)


class TestTheCliSummaryIsUntouched:
    """The other surface, message for message. Passes in both states."""

    def _summary(self, answers, queue_info):
        console = Console(no_color=True, highlight=False, width=300, record=True)
        m._show_script_and_summary(console, "#!/bin/bash\n", answers, "1.0",
                                   queue_info)
        return console.export_text()

    def test_a_described_partition_still_prints_its_reading(self):
        text = self._summary(
            {"partition": "amd", "job_name": "probe", "command": "true",
             "_partition_obj": _described()}, _LIVE)
        assert "7 running / 3 pending" in text
        assert "~2min" in text

    @pytest.mark.parametrize("reason,phrase", [
        ("absent", "partition not on this cluster"),
        ("undescribed", "partition not described by this cluster's sinfo"),
        ("unreadable", "the partition list could not be read"),
    ])
    def test_the_summary_keeps_its_fuller_wording(self, reason, phrase):
        text = self._summary(
            {"partition": "typo", "job_name": "probe", "command": "true",
             "_partition_obj": _blank(reason=reason)}, _qinfo())
        assert f"unknown — {phrase}" in text
        assert "0 running / 0 pending" not in text

    def test_a_nonexistent_name_still_gets_its_own_error(self, mocker):
        """The existing "no partition 'x' on this cluster." error, unchanged.

        The blank record is what a name gets *after* this check has passed (a
        hidden partition) or alongside it (a typo); neither the note nor the
        strip may stand in for it.
        """
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"amd"})
        mocker.patch.object(m, "fetch_account_acl", return_value={})
        mocker.patch.object(m, "check_conda_env", return_value=[])
        issues = m.site_check_issues({"partition": "typo"})
        assert [lvl for lvl, _m in issues] == ["error"]
        assert issues[0][1].startswith("no partition 'typo' on this cluster.")
        assert "This cluster's partitions: amd" in issues[0][1]

    def test_a_real_name_gets_no_such_error(self, mocker):
        mocker.patch.object(m, "fetch_all_partition_names", return_value={"amd"})
        mocker.patch.object(m, "fetch_account_acl", return_value={})
        mocker.patch.object(m, "check_conda_env", return_value=[])
        assert m.site_check_issues({"partition": "amd"}) == []
