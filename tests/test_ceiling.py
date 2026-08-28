"""The context-ceiling search, proved without loading a model.

Every probe in a real search is a full model load -- tens of seconds of GPU and
gigabytes off the page cache. A strategy that needed a live rig to test would be
tested once and then trusted forever, so the decision logic is a pure function
and this exercises it against a simulated machine.

The simulator is calibrated against the real one. With a base of 6,400 MiB and
64 KiB per token it predicts 2,304 MiB free at 64K where the development box
measured 2,281, and 1,280 MiB at 80K where the operator's hand-written note
recorded "about 1,222". Close enough that a strategy which behaves well here
will behave well there.
"""

from __future__ import annotations

import pytest

from headroom.ceiling import (
    MAX_PROBES,
    MIN_CTX,
    SETTLE_TOKENS,
    CeilingSearch,
    CeilingStatus,
    Probe,
    measured_mib_per_token,
    plan_next_context,
    summarise,
)

MARGIN = 1200
MAX_CTX = 262144
MIB_PER_TOKEN = 64 / 1024  # 64 KiB of KV per token, as on the real model


def machine(ctx: int, *, base_mib: int = 6400, margin: int = MARGIN) -> Probe:
    """What this context would do on a simulated rig.

    Below zero free the allocation fails outright, which is what an oversized
    context actually does -- llama-server exits during startup rather than
    loading into a negative number.
    """
    free = base_mib - ctx * MIB_PER_TOKEN
    if free <= 0:
        return Probe(ctx=ctx, loaded=False, error="exited during startup")
    return Probe(
        ctx=ctx,
        loaded=True,
        free_mib=int(free),
        within_margin=free >= margin,
    )


def run_search(
    start: int,
    *,
    base_mib: int = 6400,
    margin: int = MARGIN,
    max_ctx: int = MAX_CTX,
    hint: float | None = None,
):
    """Drive the planner to completion, returning every probe it asked for."""
    probes = [machine(start, base_mib=base_mib, margin=margin)]
    while True:
        nxt = plan_next_context(probes, margin_mib=margin, max_ctx=max_ctx, hint_mib_per_token=hint)
        if nxt is None:
            return probes
        probes.append(machine(nxt, base_mib=base_mib, margin=margin))


# ------------------------------------------------------------------ the slope


def test_the_slope_needs_two_successes_before_it_claims_anything() -> None:
    assert measured_mib_per_token([]) is None
    assert measured_mib_per_token([machine(65536)]) is None
    assert measured_mib_per_token([Probe(ctx=98304, loaded=False)]) is None


def test_the_slope_is_measured_not_assumed() -> None:
    """And it should land on the machine's real cost per token."""
    probes = [machine(16384), machine(65536)]

    slope = measured_mib_per_token(probes)

    assert slope == pytest.approx(MIB_PER_TOKEN, rel=0.02)


def test_the_slope_uses_the_widest_pair_available() -> None:
    """Two nearby contexts differ by rounding as much as by cost."""
    probes = [machine(65536), machine(66560), machine(8192)]

    assert measured_mib_per_token(probes) == pytest.approx(MIB_PER_TOKEN, rel=0.02)


# ----------------------------------------------------------------- converging


def test_it_finds_the_ceiling_the_margin_implies() -> None:
    """Free falls 0.0625 MiB per token from 6,400, so 1,200 MiB of margin is
    reached at about 83,200 tokens -- 81,920 once rounded onto the grid."""
    probes = run_search(65536)

    best = max(p.ctx for p in probes if p.within_margin)
    assert 79000 <= best <= 83200, f"settled at {best}"
    # And it never claims a context that would breach the margin.
    assert all(p.free_mib >= MARGIN for p in probes if p.within_margin)


def test_it_gets_there_in_a_handful_of_loads() -> None:
    """Each probe is a real model load, so the count IS the cost.

    Without a seed the first upward step has nothing to predict from and has to
    double, which spends a load discovering that twice was far too much.
    """
    probes = run_search(65536)

    assert len(probes) <= 5, f"took {len(probes)} loads: {[p.ctx for p in probes]}"
    assert len(probes) <= MAX_PROBES


def test_the_recorded_cost_per_token_pays_for_itself_immediately() -> None:
    """This is what `kv_bytes_per_token` in the registry is for.

    Seeded, the first step is a prediction rather than a guess and the search
    lands on the answer in two loads instead of four -- minutes of GPU, on a
    figure the registry already had.
    """
    seeded = run_search(65536, hint=MIB_PER_TOKEN)
    blind = run_search(65536)

    assert len(seeded) <= 2, f"took {len(seeded)}: {[p.ctx for p in seeded]}"
    assert len(seeded) < len(blind)


def test_the_seed_changes_the_cost_and_never_the_answer() -> None:
    """It steers the search; it does not decide the result."""
    seeded = run_search(65536, hint=MIB_PER_TOKEN)
    blind = run_search(65536)

    assert max(p.ctx for p in seeded if p.within_margin) == max(
        p.ctx for p in blind if p.within_margin
    )


@pytest.mark.parametrize("wrongness", [0.4, 3.0])
def test_a_stale_recorded_figure_still_converges(wrongness: float) -> None:
    """A registry value can be wrong -- copied from a sibling build, or measured
    before a change to the KV cache type. The measured slope replaces it as soon
    as two real probes exist, so a bad seed costs loads and not correctness."""
    probes = run_search(65536, hint=MIB_PER_TOKEN * wrongness)

    best = max(p.ctx for p in probes if p.within_margin)
    assert 79000 <= best <= 83200, f"a wrong seed moved the answer to {best}"
    assert all(p.free_mib >= MARGIN for p in probes if p.within_margin)


def test_starting_far_too_high_walks_back_down() -> None:
    """A registry value that no longer fits -- a bigger model, or a card now
    driving a display -- must not leave the search stuck at a failure."""
    probes = run_search(262144)

    assert probes[0].loaded is False
    assert any(p.within_margin for p in probes), "never recovered from a failing start"


def test_starting_far_too_low_reaches_upward() -> None:
    probes = run_search(4096)

    best = max(p.ctx for p in probes if p.within_margin)
    assert best > 60000, f"only reached {best}"


def test_it_never_proposes_beyond_the_models_own_ceiling() -> None:
    """A context past what the model was trained for is not a win."""
    probes = run_search(4096, base_mib=200000, max_ctx=32768)

    assert all(p.ctx <= 32768 for p in probes), [p.ctx for p in probes]


# --------------------------------------------------------------- termination


def test_it_stops_once_the_bracket_is_not_worth_another_load() -> None:
    good = Probe(ctx=81920, loaded=True, free_mib=1300, within_margin=True)
    bad = Probe(ctx=81920 + SETTLE_TOKENS, loaded=True, free_mib=1100, within_margin=False)

    assert plan_next_context([good, bad], margin_mib=MARGIN, max_ctx=MAX_CTX) is None


def test_it_stops_at_the_probe_budget() -> None:
    """Bounded regardless of what the numbers do. Each one costs a load."""
    probes = [machine(1024 * (i + 1)) for i in range(MAX_PROBES)]

    assert plan_next_context(probes, margin_mib=MARGIN, max_ctx=MAX_CTX) is None


def test_a_proposal_always_lands_strictly_inside_the_bracket() -> None:
    """Otherwise the search re-tries a known answer and never converges."""
    good = Probe(ctx=65536, loaded=True, free_mib=2304, within_margin=True)
    bad = Probe(ctx=98304, loaded=False)

    nxt = plan_next_context([good, bad], margin_mib=MARGIN, max_ctx=MAX_CTX)

    assert nxt is not None
    assert 65536 < nxt < 98304


def test_it_will_not_propose_a_uselessly_small_context() -> None:
    """If nothing fits, saying so beats proposing 512 tokens."""
    probes = [Probe(ctx=MIN_CTX, loaded=False)]

    nxt = plan_next_context(probes, margin_mib=MARGIN, max_ctx=MAX_CTX)

    assert nxt is None or nxt >= MIN_CTX


def test_nothing_tried_yet_is_the_callers_decision() -> None:
    """The starting point comes from the registry, not from this function."""
    assert plan_next_context([], margin_mib=MARGIN, max_ctx=MAX_CTX) is None


# ------------------------------------------------------------------ reporting


def test_a_machine_where_nothing_fits_says_so_rather_than_guessing() -> None:
    search = CeilingSearch(
        id="x", model_key="m", port=8080, margin_mib=MARGIN, max_ctx=MAX_CTX, start_ctx=65536
    )
    search.probes = [Probe(ctx=65536, loaded=False), Probe(ctx=32768, loaded=False)]

    out = summarise(search)

    assert out["found"] is False
    assert "may simply not fit" in out["note"]


def test_the_summary_reports_what_it_cost_and_what_it_means() -> None:
    search = CeilingSearch(
        id="x", model_key="m", port=8080, margin_mib=MARGIN, max_ctx=MAX_CTX, start_ctx=65536
    )
    search.probes = run_search(65536)
    search.status = CeilingStatus.COMPLETE

    out = summarise(search)

    assert out["found"] is True
    assert out["ctx"] == search.best.ctx
    # The translation that makes the number portable to another model.
    assert out["tokens_per_gib"] and out["tokens_per_gib"] > 0
    assert "written to the registry" in out["note"], "must say it did not save anything"
    assert "as it was at the time" in out["note"], "must say the answer is situational"
