"""The partition picker's GPU capacity: parsed, summed, and shown to nobody.

``fetch_partitions`` reads ``sinfo %G`` and records two GPU facts per partition:
``gpus_per_node`` (``_parse_gpu_count``, ``max()``-merged across the partition's
sinfo rows) and the boolean ``has_gpu``.  Both were read by exactly two capacity
*checks* — ``capacity_refusal`` and ``validate_job_config``, which fire only when
the request already exceeds the limit — and by no surface at all.  The one place a
partition is actually chosen, ``tui._fmt_partition``, read neither: its GPU
segment was keyed on ``gpu_types``, the model list.

Two figures were therefore lost, both measured on midway3 (Slurm 20.11.8, 88
partitions):

* **A count-only-GRES GPU partition was invisible.**  ``Gres=gpu:4`` with no model
  populates ``gpus_per_node=4`` and ``has_gpu=True`` but leaves ``gpu_types``
  empty, so the row carried no GPU marker whatsoever and was byte-identical to a
  CPU-only partition of the same shape.  18 of this cluster's 88 partitions are
  count-only — ``gpu``, ``beagle3``, ``kicp-gpu``, ``ssd-gpu`` and 14 more — and
  the partition step runs *before* every GPU step, so nothing later in the wizard
  recovers the choice.  This is H2's cluster shape reappearing on a different
  surface, and mock mode cannot show it: every ``MOCK_PARTITIONS`` GPU partition
  carries typed GRES.

* **The count never appeared even when the models did.**  A 1-GPU-per-node and an
  8-GPU-per-node ``a100`` partition both rendered ``GPU:[a100]`` — the picker's
  one per-node figure with no number, next to ``48 CPU`` and ``180G``, which are
  the same kind of figure and do carry theirs.

Proved before fixing, with inputs differing only in the missing dimension::

    {..., "gpu_types": [], "has_gpu": True,  "gpus_per_node": 4}
    {..., "gpu_types": [], "has_gpu": False, "gpus_per_node": 0}
    both -> 'p            8 nodes · 48 CPU · 180G'

    {..., "gpu_types": ["a100"], "gpus_per_node": 1}
    {..., "gpu_types": ["a100"], "gpus_per_node": 8}
    both -> 'p            8 nodes · 48 CPU · 180G · GPU:[a100]'

The tests below split into two groups.  Every member of
``TestGpuCapacityReachesThePicker`` reddens when the fix is neutered — 5 of 7 on
either neuter site alone, all 7 with both.  Every member of
``TestPickerRowUnchangedElsewhere`` passes with the fix in and with it neutered,
including the pre-existing ``GPU:[a100]`` spelling, which is still what a
partition with models but no resolvable count renders.
"""

from __future__ import annotations

from slurmate import system_utils as su
from slurmate.tui import _fmt_partition

# A partition record shaped like fetch_partitions' output, minus the GPU fields —
# every test varies only those, so a difference in the row can only come from them.
_BASE = {
    "name": "p",
    "nodes": 8,
    "nodes_up": 8,
    "state": "up",
    "cpus_per_node": 48,
    "mem_per_node_mb": 184320,
    "heterogeneous": False,
    "is_default": False,
    "timelimit": None,
}


def _row(**gpu_fields):
    return _fmt_partition({**_BASE, **gpu_fields})


class TestGpuCapacityReachesThePicker:
    """Teeth. Every assertion here turns on the count/has_gpu the fix reads."""

    def test_count_only_gpu_partition_differs_from_cpu_only(self):
        # The measured pair. midway3's `gpu` is Gres=gpu:4; `build` is (null).
        gpu = _row(gpu_types=[], has_gpu=True, gpus_per_node=4)
        cpu = _row(gpu_types=[], has_gpu=False, gpus_per_node=0)
        assert gpu != cpu

    def test_count_only_gpu_partition_names_its_gpu_count(self):
        assert "GPU:4" in _row(gpu_types=[], has_gpu=True, gpus_per_node=4)

    def test_count_distinguishes_two_partitions_with_the_same_model(self):
        one = _row(gpu_types=["a100"], has_gpu=True, gpus_per_node=1)
        eight = _row(gpu_types=["a100"], has_gpu=True, gpus_per_node=8)
        assert one != eight
        assert "GPU:1 " in one and "GPU:8 " in eight

    def test_count_and_models_are_both_shown_when_both_are_known(self):
        row = _row(gpu_types=["a100", "v100"], has_gpu=True, gpus_per_node=4)
        assert "GPU:4 [a100,v100]" in row

    def test_known_gpus_of_unknown_count_and_model_are_still_marked(self):
        # has_gpu without either figure: admitted as "?", the same thing this row
        # already does for an absent cpus_per_node, not silence.
        assert "GPU:?" in _row(has_gpu=True)

    def test_real_count_only_sinfo_row_reaches_the_picker(self, mocker):
        # midway3's actual `gpu` row, end to end through fetch_partitions.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=("gpu|infinite|11|up|48|184320+|gpu:4|mixed\n", "", 0),
        )
        part = su.fetch_partitions()[0]
        assert (part["gpu_types"], part["has_gpu"], part["gpus_per_node"]) == (
            [], True, 4,
        )
        assert "GPU:4" in _fmt_partition(part)

    def test_two_sinfo_rows_show_the_merged_maximum(self, mocker):
        # fetch_partitions max()-merges the count across a partition's rows; the
        # picker must show the merged figure, not the first row's.
        mocker.patch.object(su, "is_tool_available", return_value=True)
        mocker.patch.object(
            su, "_run_command",
            return_value=(
                "mix|infinite|3|up|48|184320|gpu:2|idle\n"
                "mix|infinite|5|up|48|184320|gpu:8|mixed\n",
                "", 0,
            ),
        )
        part = su.fetch_partitions()[0]
        assert part["gpus_per_node"] == 8
        assert "GPU:8" in _fmt_partition(part)


class TestPickerRowUnchangedElsewhere:
    """Controls — every one of these passes with the fix in and with it neutered.

    Their verdicts all predate the fix.  ``test_cpu_only_partition_says_nothing``
    is the one that traverses the widened condition, and it is here rather than
    above precisely because its answer does not change: a partition with
    ``has_gpu is False`` and no models had no GPU segment before and must still
    have none, or the fix would have made all 68 of this cluster's CPU-only
    partitions claim GPUs.
    """

    def test_cpu_only_partition_says_nothing_about_gpus(self):
        assert "GPU" not in _row(gpu_types=[], has_gpu=False, gpus_per_node=0)

    def test_partially_drained_still_shows_both_node_counts(self):
        assert "13 of 17 nodes" in _fmt_partition(
            {**_BASE, "nodes": 17, "nodes_up": 13}
        )

    def test_fully_drained_is_still_marked_unavailable(self):
        assert "unavailable" in _fmt_partition({**_BASE, "nodes": 8, "nodes_up": 0})

    def test_healthy_partition_still_has_a_bare_node_count(self):
        row = _fmt_partition({**_BASE, "nodes": 8, "nodes_up": 8})
        assert "8 nodes" in row
        assert "of" not in row and "unavailable" not in row

    def test_per_node_cpu_and_memory_figures_are_untouched(self):
        assert "48 CPU" in _row() and "180G" in _row()

    def test_default_partition_is_still_labelled(self):
        assert "default" in _fmt_partition({**_BASE, "is_default": True})

    def test_models_without_a_count_keep_the_original_spelling(self):
        # The form this row has always used, and the one a hand-built partition
        # record (no gpus_per_node key) still gets.
        assert "GPU:[a100]" in _row(gpu_types=["a100"])

    def test_label_still_round_trips_back_to_its_partition(self):
        # The label is the picker's identity key: _set_partition_from_select
        # matches the chosen row against _fmt_partition of each record. Widening
        # the row must not break that, or a pick resolves to a raw label.
        from slurmate.tui import Wizard

        part = {**_BASE, "name": "gpu", "gpu_types": [], "has_gpu": True,
                "gpus_per_node": 4}
        w = Wizard.__new__(Wizard)
        w.answers = {}
        w.transient = {"public_parts": [part], "all_parts": [part]}
        w._set_partition_from_select(_fmt_partition(part))
        assert w.answers["partition"] == "gpu"
        assert w.answers["_partition_obj"] is part
