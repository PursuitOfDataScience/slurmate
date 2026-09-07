"""A model reported as a GRES *type* must keep the spelling the GRES used.

`fetch_gpu_type_sources` splits a partition's GPU models by how they can be
requested, and its own docstring binds the first bucket:

    ``typed`` -- seen in a real ``gpu:MODEL:N`` GRES, so requestable as a GRES
    type (``--gres=gpu:MODEL:N``, ``--gpus=MODEL:N``, ...)

while warning that a model from the *other* bucket "makes Slurm reject the job
outright (Requested node configuration is not available)". Both passes folded
`_` to `-` while collecting, so on a site whose configured GRES type carries an
underscore slurmate put a model in `typed` under a spelling that is not a GRES
type on that cluster -- manufacturing the exact rejection the docstring
attributes to the other bucket.

Measured end to end, `gpu:rtx_6000:2` with no node features at all:

    typed: ["rtx-6000"]                        # before
    validate_job_config(gpu_type="rtx_6000")   # -> ERROR, "not in partition
                                               #    list (rtx-6000)"

so the tool rejected the cluster's real type name and accepted only the one
Slurm will refuse at submit. `validate_job_config` already treats a **case**-only
difference as "a real, and otherwise invisible, way for a validated job to be
rejected at submit" and warns about it; the separator difference was slurmate's
own doing, so nothing warned.

The fold itself is load-bearing and stays: the two sources genuinely disagree on
separators ("rtx_6000" GRES vs. an "rtx-6000" feature), so every *comparison*
still runs on the folded key -- corroboration, `feature -= typed`, and `_norm`
for `constraint`. Only the reported `typed` strings changed. `constraint` keeps
the folded key on purpose (it names a node feature, and the fold is what matches
the two spellings), as does `fetch_partitions`' `gpu_types`, which is a matching
key for `validate_job_config` rather than a request string. Both are pinned in
`TestControls`.
"""

from unittest import mock

import pytest

from slurmate import system_utils as su


def _sources(sinfo_out, partition="gpuq"):
    """Drive `fetch_gpu_type_sources` against canned `sinfo -N -o '%f|%G'`."""
    with mock.patch.object(su, "is_tool_available", return_value=True), \
         mock.patch.object(su, "_run_command", return_value=(sinfo_out, "", 0)):
        return su.fetch_gpu_type_sources(partition)


def _gpu_complaints(chosen, typed):
    part = {
        "name": "gpuq", "gpu_types": [], "has_gpu": True,
        "cpus_per_node": 8, "mem_per_node_mb": 64000, "gpus_per_node": 2,
    }
    msgs = su.validate_job_config(
        {"partition": "gpuq", "gpus": 2, "gpu_type": chosen, "_partition_obj": part},
        extra_gpu_types=list(typed),
    )
    return [(lvl, text) for lvl, text in msgs if "GPU type" in text]


class TestTheGresSpellingIsReportedVerbatim:
    def test_an_underscored_type_is_not_reported_with_a_hyphen(self):
        # The decisive shape: no features anywhere, so the GRES is the only
        # source of the name and there is nothing to corroborate against.
        assert _sources("(null)|gpu:rtx_6000:2\n")["typed"] == ["rtx_6000"]

    @pytest.mark.parametrize(
        "model", ["rtx_6000", "rtx-6000", "a100_80gb", "a100.80gb", "mi300x"]
    )
    def test_the_reported_name_always_occurs_in_the_source(self, model):
        """The invariant behind the fix: `typed` never synthesises a spelling.

        Stated as a property rather than per-model, so a future normalisation
        of any kind trips it, not just the underscore one.
        """
        text = f"(null)|gpu:{model}:2\n"
        for reported in _sources(text)["typed"]:
            assert reported in text, (reported, text)

    def test_the_two_surfaces_each_name_what_their_request_form_needs(self):
        # One card, two spellings, two request forms: `--gres=gpu:rtx_6000:2`
        # needs the GRES type, `--constraint=rtx-6000` needs the feature.
        got = _sources("rack5,rtx-6000|gpu:rtx_6000:2\n")
        assert got["typed"] == ["rtx_6000"]
        assert got["constraint"] == ["rtx-6000"]

    def test_corroboration_still_crosses_the_separator(self):
        # A typed node and a count-only node that spells the card differently:
        # the fold is what lets the second corroborate the first, so this must
        # still collapse to ONE model rather than listing both spellings.
        got = _sources("(null)|gpu:rtx_6000:2\nrack6,rtx-6000|gpu:2\n")
        assert got["typed"] == ["rtx_6000"]
        assert got["feature"] == []

    def test_the_reported_type_is_the_one_validation_accepts(self):
        """The round trip, which is where the defect was actually felt."""
        typed = _sources("(null)|gpu:rtx_6000:2\n")["typed"]
        # What the cluster really calls the card validates clean...
        assert _gpu_complaints("rtx_6000", typed) == []
        # ...and a wrong spelling is refused, naming the real list.
        complaints = _gpu_complaints("rtx-6000", typed)
        assert [lvl for lvl, _ in complaints] == ["error"]
        assert "rtx_6000" in complaints[0][1], complaints


class TestControls:
    """Behaviour that must not change. Each passes in BOTH states."""

    def test_the_answer_does_not_depend_on_node_order(self):
        """Stability, which held before the fix too -- verified by neutering.

        Two nodes spelling the same card differently still collapse to one
        model, and to the same one on every run. `setdefault` is what keeps
        that true now that a spelling is remembered rather than derived.
        """
        text = "(null)|gpu:rtx-6000:2\n(null)|gpu:rtx_6000:2\n"
        first = _sources(text)["typed"]
        assert first == _sources(text)["typed"]
        assert len(first) == 1 and first[0] in ("rtx-6000", "rtx_6000")

    @pytest.mark.parametrize(
        ("sinfo_out", "expected"),
        [
            ("rack5,a100|gpu:a100:4\n",
             {"typed": ["a100"], "feature": [], "constraint": ["a100"]}),
            ("rack6,a100|gpu:4\n",
             {"typed": [], "feature": ["a100"], "constraint": ["a100"]}),
            ("(null)|gpu:a100:2,gpu:v100:2\n",
             {"typed": ["a100", "v100"], "feature": [], "constraint": []}),
            ("(null)|gpu:mps:100,gpu:a100:2\n",
             {"typed": ["a100"], "feature": [], "constraint": []}),
            ("intel,avx512|(null)\n",
             {"typed": [], "feature": [], "constraint": []}),
        ],
    )
    def test_a_partition_with_no_underscore_is_untouched(self, sinfo_out, expected):
        assert _sources(sinfo_out) == expected

    def test_the_detect_helper_still_folds_on_purpose(self):
        # Scope control: `_detect_gpu_type`'s own return is unchanged. Its value
        # reaches `feature`, which is compared against feature tokens, so the
        # fix deliberately stopped at the `typed` reporting surface.
        assert su._detect_gpu_type("", "gpu:rtx_6000:2") == "rtx-6000"

    def test_the_partition_listing_still_folds_on_purpose(self):
        # The other scope control: `fetch_partitions`' `gpu_types` is a matching
        # key for `validate_job_config` and `main.py`, both of which compare it
        # case-insensitively but NOT separator-insensitively. Re-spelling it
        # would change what validates, which this fix does not touch.
        # `sinfo -h -o "%P|%l|%D|%a|%c|%m|%G|%T"` -- the GRES is field 7.
        row = "gpuq|1-00:00:00|1|up|8|64000|gpu:rtx_6000:2|idle\n"
        with mock.patch.object(su, "is_tool_available", return_value=True), \
             mock.patch.object(su, "_force_mock", return_value=False), \
             mock.patch.object(su, "_run_command", return_value=(row, "", 0)):
            parts = su.fetch_partitions()
        # Non-vacuous on purpose: the row must actually have parsed into a
        # model, and that model must still be the folded spelling.
        assert [p["gpu_types"] for p in parts] == [["rtx-6000"]], parts

    def test_a_count_only_partition_still_reports_no_typed_model(self):
        # The distinction the whole function exists for.
        got = _sources("rack1,a100|gpu:4\nrack2,a100|gpu:4\n")
        assert got["typed"] == [] and got["feature"] == ["a100"]
