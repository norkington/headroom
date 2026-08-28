"""Portable unit tests.

These run anywhere — no GPU, no llama.cpp, no local registry — which is the
point. `test_argv_parity.py` covers the contract that matters most, but it can
only run on a machine with the real shell launcher installed, so in CI it skips.
A pipeline whose only green signal is "lint passed and everything skipped" is
decoration, not evidence.

So these exercise the actual decision logic against synthetic inputs: the
headroom grading, the CUDA-order reconciliation, the server state machine, and
the argv builder's guard rails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.gguf import GgufAnalysis, GgufError, normalise_repo
from headroom.gpu import (
    _DEVICE_LINE,
    HEADROOM_CRITICAL_MIB,
    HEADROOM_TIGHT_MIB,
    THROTTLE_APP_CLOCKS,
    THROTTLE_GPU_IDLE,
    THROTTLE_HW_THERMAL,
    THROTTLE_SW_POWER_CAP,
    THROTTLE_SW_THERMAL,
    CudaMapping,
    Gpu,
    devices_in_use,
    mark_vision_residency,
    order_differs,
    resolve_cuda_mapping,
)
from headroom.registry import RegistryError, build_argv, load
from headroom.server import ServerState


def make_gpu(**kw) -> Gpu:
    base = {
        "nvml_index": 0,
        "name": "NVIDIA GeForce RTX 4070 SUPER",
        "memory_total_mib": 12282,
        "memory_used_mib": 1000,
        "memory_free_mib": 11282,
    }
    base.update(kw)
    return Gpu(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- headroom


@pytest.mark.parametrize(
    ("free", "expected"),
    [
        (0, "critical"),
        (HEADROOM_CRITICAL_MIB - 1, "critical"),
        (HEADROOM_CRITICAL_MIB, "tight"),
        (HEADROOM_TIGHT_MIB - 1, "tight"),
        (HEADROOM_TIGHT_MIB, "ok"),
        (11282, "ok"),
    ],
)
def test_headroom_grading_boundaries(free: int, expected: str) -> None:
    assert make_gpu(memory_free_mib=free).headroom_state == expected


def test_grading_is_per_card_not_aggregate() -> None:
    """The premise of the project in one assertion.

    Two cards can total plenty of free memory while one of them is nearly out.
    Grading on the total would report this pair as healthy.
    """
    roomy = make_gpu(nvml_index=0, memory_free_mib=2200)
    starved = make_gpu(nvml_index=1, name="NVIDIA GeForce RTX 3060", memory_free_mib=400)

    total_free = roomy.memory_free_mib + starved.memory_free_mib
    assert total_free > HEADROOM_TIGHT_MIB, "the aggregate looks fine"
    assert roomy.headroom_state == "ok"
    assert starved.headroom_state == "critical", "but one card is nearly out"


def test_label_never_invents_a_cuda_index() -> None:
    """An unresolved mapping must not be presented as CUDA0.

    Guessing here would be worse than admitting ignorance: the whole reason this
    mapping exists is that the obvious assumption is often wrong.
    """
    unmapped = make_gpu()
    assert "CUDA" not in unmapped.label
    assert "nvml 0" in unmapped.label

    unmapped.cuda_index = 1
    assert "CUDA1" in unmapped.label


# ----------------------------------------------------------------- thermals
#
# Graded against the card's OWN slowdown threshold, never a fixed temperature.
# The two cards on the development box slow down at 95 C and 96 C, with GPU_MAX
# at 93 and 90 -- any single hardcoded limit is wrong on at least one of them,
# and would be wrong in a different way on a laptop or a datacentre card.


def test_an_idle_card_is_not_reported_as_throttling() -> None:
    """NVML sets GPU_IDLE on a card doing nothing.

    A naive "throttle reasons != 0" check lights the panel up on a machine that
    is perfectly healthy and merely idle -- which teaches the user to ignore the
    warning, making it worse than no warning at all.
    """
    idle = make_gpu(temperature_c=31, temp_slowdown_c=95, throttle_reasons=THROTTLE_GPU_IDLE)

    assert idle.throttle_labels == ()
    assert idle.throttling_thermally is False
    assert idle.thermal_state == "ok"


def test_user_set_clocks_are_not_a_fault_either() -> None:
    card = make_gpu(temperature_c=40, temp_slowdown_c=95, throttle_reasons=THROTTLE_APP_CLOCKS)
    assert card.throttle_labels == ()


@pytest.mark.parametrize(
    ("temp", "slowdown", "expected"),
    [
        (31, 95, "ok"),
        (79, 95, "ok"),
        (80, 95, "warm"),  # within 15 of slowdown
        (89, 95, "warm"),
        (90, 95, "hot"),  # within 5 of slowdown
        (95, 95, "hot"),
        # The same temperature on a card with a lower limit is further along.
        (85, 90, "hot"),
        (85, 105, "ok"),
    ],
)
def test_thermal_state_is_relative_to_the_cards_own_limit(temp, slowdown, expected) -> None:
    card = make_gpu(temperature_c=temp, temp_slowdown_c=slowdown)
    assert card.thermal_state == expected


def test_throttling_outranks_the_temperature() -> None:
    """A card already clamping itself is past the point where the reading leads.

    Reported temperature can even fall while throttling -- that is the throttle
    working -- so grading on temperature alone would show a cooling card as
    healthy at the exact moment its throughput collapsed.
    """
    cool_but_clamped = make_gpu(
        temperature_c=62, temp_slowdown_c=95, throttle_reasons=THROTTLE_HW_THERMAL
    )

    assert cool_but_clamped.thermal_state == "throttling"
    assert cool_but_clamped.throttling_thermally is True
    assert "hardware thermal slowdown" in cool_but_clamped.throttle_labels


def test_power_capping_is_reported_but_is_not_thermal() -> None:
    """Normal on a stock card under sustained load. It explains a low throughput
    figure without being a fault to fix."""
    card = make_gpu(temperature_c=70, temp_slowdown_c=95, throttle_reasons=THROTTLE_SW_POWER_CAP)

    assert card.throttling_for_power is True
    assert card.throttling_thermally is False
    assert card.thermal_state == "ok"
    assert card.throttle_labels == ("power cap",)


def test_thermal_headroom_is_degrees_not_a_grade() -> None:
    card = make_gpu(temperature_c=71, temp_slowdown_c=96)
    assert card.thermal_headroom_c == 25


def test_a_card_that_reports_no_threshold_still_grades() -> None:
    """Some virtualised and WSL setups omit the thresholds. Falling back to a
    conservative constant beats showing nothing on the one panel that exists to
    warn people."""
    card = make_gpu(temperature_c=88, temp_slowdown_c=None)
    assert card.thermal_state == "hot"


def test_no_temperature_at_all_is_unknown_rather_than_ok() -> None:
    """Absence of a reading is not evidence of a cool card."""
    card = make_gpu(temperature_c=None, temp_slowdown_c=None)
    assert card.thermal_state == "unknown"


def test_several_throttle_reasons_are_all_named() -> None:
    card = make_gpu(
        temperature_c=94,
        temp_slowdown_c=95,
        throttle_reasons=THROTTLE_SW_THERMAL | THROTTLE_SW_POWER_CAP | THROTTLE_GPU_IDLE,
    )
    assert card.throttle_labels == ("software thermal slowdown", "power cap")


# ------------------------------------------------------- vision residency
#
# A resident projector makes a card's free figure an upper bound rather than a
# reading: llama.cpp's image buffer is a retained high-water mark, so it grows
# with the first large image and is never given back. Measured on the
# development box, one 4K image took a card from 578 MiB free to 170 MiB and
# left it there. A card sitting comfortably above the tight line while holding a
# projector has therefore not finished falling, and reading its headroom as
# spare capacity is how the server gets OOMed by something that looked
# affordable at the time.


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["-m", "x.gguf", "-dev", "CUDA0,CUDA1"], [0, 1]),
        (["-m", "x.gguf", "--device", "CUDA1"], [1]),
        (["-m", "x.gguf", "--device=CUDA1"], [1]),
        (["-m", "x.gguf", "-dev", "CUDA1, CUDA0"], [1, 0]),
    ],
)
def test_the_device_list_is_read_off_the_command_line(argv, expected) -> None:
    assert devices_in_use(argv) == expected


def test_naming_no_devices_is_not_the_same_as_naming_none() -> None:
    """`None` and `[]` are opposite claims and must not share a spelling.

    A server started without the flag uses every visible device. Returning an
    empty list would say the opposite -- that it uses nothing -- and every card
    would go unmarked on exactly the command line that puts a projector on all
    of them.
    """
    assert devices_in_use(["-m", "x.gguf"]) is None
    assert devices_in_use([]) is None


def test_a_projector_is_pinned_to_the_cards_it_is_actually_on() -> None:
    """And on this hardware that is not the card the index suggests.

    CUDA0 is the second NVML device here, which is the reversed order the whole
    mapping exists for. Marking by NVML index would put the warning on the wrong
    card -- the one thing worse than not warning.
    """
    cards = [
        make_gpu(nvml_index=0, name="NVIDIA GeForce RTX 3060", cuda_index=1),
        make_gpu(nvml_index=1, name="NVIDIA GeForce RTX 4070 SUPER", cuda_index=0),
    ]

    mark_vision_residency(cards, vision=True, command_line=["-dev", "CUDA0"])

    assert [c.vision_resident for c in cards] == [False, True]


def test_a_server_that_names_no_devices_is_holding_all_of_them() -> None:
    cards = [make_gpu(nvml_index=0, cuda_index=0), make_gpu(nvml_index=1, cuda_index=1)]

    mark_vision_residency(cards, vision=True, command_line=["-m", "x.gguf"])

    assert all(c.vision_resident for c in cards)


def test_an_unresolved_mapping_warns_about_every_card() -> None:
    """Under-warning costs a server mid-generation; over-warning costs a label.

    Without a CUDA mapping Headroom cannot rule any card out, and the same is
    true when the command line names a device that does not resolve to one. Both
    resolve towards the warning.
    """
    unmapped = [make_gpu(nvml_index=0), make_gpu(nvml_index=1)]
    mark_vision_residency(unmapped, vision=True, command_line=["-dev", "CUDA0"])
    assert all(c.vision_resident for c in unmapped)

    stranger = [make_gpu(nvml_index=0, cuda_index=0)]
    mark_vision_residency(stranger, vision=True, command_line=["-dev", "CUDA0,CUDA3"])
    assert stranger[0].vision_resident


def test_stopping_the_server_clears_the_mark() -> None:
    """The UI stays open across a start and a stop, so this is not a fresh list.

    A stale mark would keep telling someone their headroom is still falling on a
    box where nothing is loaded, which spends the warning's credibility on a
    card that is genuinely free.
    """
    cards = [make_gpu(nvml_index=0, cuda_index=0)]
    mark_vision_residency(cards, vision=True, command_line=[])
    assert cards[0].vision_resident

    mark_vision_residency(cards, vision=False, command_line=[])
    assert not cards[0].vision_resident


@pytest.mark.parametrize(
    ("free", "expected"),
    [(11282, True), (900, True), (400, False)],
)
def test_a_critical_card_is_not_also_called_provisional(free: int, expected: bool) -> None:
    """There is no worse grade to warn about, and the warning would dilute one
    that already says everything."""
    card = make_gpu(memory_free_mib=free, vision_resident=True)
    assert card.headroom_provisional is expected


def test_nothing_is_provisional_without_a_projector() -> None:
    assert make_gpu(memory_free_mib=1300).headroom_provisional is False


def test_the_grade_itself_is_not_demoted_by_a_projector() -> None:
    """Deliberate. `ok` still means what it measures.

    Collapsing a 1.3 GiB card into the same bucket as a 600 MiB one would throw
    away the distinction that decides whether the first large image is
    survivable at all -- which is precisely the question the label is there to
    raise.
    """
    roomy = make_gpu(memory_free_mib=1300, vision_resident=True)
    tight = make_gpu(memory_free_mib=600, vision_resident=True)

    assert roomy.headroom_state == "ok"
    assert tight.headroom_state == "tight"
    assert roomy.headroom_provisional and tight.headroom_provisional


# ------------------------------------------------- what people actually paste
#
# Nobody types a repository identifier; they copy the address bar. Every one of
# these was a real failure, and none of them blamed the input:
#
#   a full URL          -> "repository not found"      (blamed the repo)
#   .../tree/main       -> "could not reach the hub"   (blamed the network; it
#                                                       was an AttributeError,
#                                                       the hub returns a list
#                                                       for that path)
#   a trailing space    -> "is gated; accept its terms" (blamed the user, and
#                                                        sent them to a terms
#                                                        page that did not exist)


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("  unsloth/Qwen3-8B-GGUF  ", "unsloth/Qwen3-8B-GGUF"),
        ("unsloth/Qwen3-8B-GGUF/", "unsloth/Qwen3-8B-GGUF"),
        ("https://huggingface.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("http://huggingface.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("huggingface.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("www.huggingface.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("hf.co/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("https://huggingface.co/models/unsloth/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF"),
        ("https://huggingface.co/unsloth/Qwen3-8B-GGUF/tree/main", "unsloth/Qwen3-8B-GGUF"),
        ("unsloth/Qwen3-8B-GGUF/blob/main/x.gguf", "unsloth/Qwen3-8B-GGUF"),
        ("unsloth/Qwen3-8B-GGUF?library=true", "unsloth/Qwen3-8B-GGUF"),
        ("unsloth/Qwen3-8B-GGUF#files", "unsloth/Qwen3-8B-GGUF"),
    ],
)
def test_a_pasted_model_page_url_is_a_repository(typed: str, expected: str) -> None:
    assert normalise_repo(typed) == expected


@pytest.mark.parametrize("typed", ["gpt2", "bert-base-uncased", "gpt2/tree/main"])
def test_the_hubs_bare_canonical_names_are_still_repositories(typed: str) -> None:
    """`gpt2` predates namespaces and is served without an owner.

    An owner/name rule rejected these as "not a HuggingFace repository", which
    is false — and false in the confident way that makes someone stop looking.
    """
    assert normalise_repo(typed) == typed.split("/")[0]


@pytest.mark.parametrize("typed", ["", "   ", "/", "//", "https://huggingface.co/", "-bad/name"])
def test_input_that_is_not_a_repository_says_so_and_shows_the_shape(typed: str) -> None:
    """The error names the expected form. "Invalid" alone leaves someone
    guessing at which of several plausible things was wanted."""
    with pytest.raises(GgufError, match="owner/name"):
        normalise_repo(typed)


def test_a_repo_name_is_not_truncated_at_an_unknown_segment() -> None:
    """Cutting at any third segment would silently turn a wrong identifier into
    a plausible one, and report success for a repository nobody asked for."""
    with pytest.raises(GgufError):
        normalise_repo("owner/name/something-else")


# ---------------------------------------------------------------- cuda order


def test_device_line_parses_llama_cpp_output() -> None:
    sample = """
Available devices:
  CUDA0: NVIDIA GeForce RTX 4070 SUPER (12281 MiB, 11069 MiB free)
  CUDA1: NVIDIA GeForce RTX 3060 (12287 MiB, 11253 MiB free)
"""
    matches = list(_DEVICE_LINE.finditer(sample))
    assert [m.group("idx") for m in matches] == ["0", "1"]
    assert matches[0].group("name") == "NVIDIA GeForce RTX 4070 SUPER"
    assert matches[1].group("total") == "12287"


def test_order_differs_detects_the_reversal() -> None:
    """The condition that makes `-dev CUDA0` and `nvidia-smi -i 0` disagree."""
    same = CudaMapping(cuda_to_nvml={0: 0, 1: 1})
    reversed_ = CudaMapping(cuda_to_nvml={0: 1, 1: 0})

    assert order_differs(same) is False
    assert order_differs(reversed_) is True


def test_unresolved_mapping_is_not_treated_as_agreement() -> None:
    """An empty mapping means "unknown", which must not read as "they agree"."""
    empty = CudaMapping()
    assert empty.resolved is False
    assert order_differs(empty) is False  # nothing known, so nothing claimed


# ------------------------------------------------- rigs that are not this one
#
# The reconciliation used to match on (name, total VRAM) alone and take the
# first still-unclaimed card. On a rig of IDENTICAL cards every device matched
# the first candidate, so the mapping came out as the identity -- whatever
# llama.cpp had actually said -- and reported itself resolved, with
# order_differs False and no warning.
#
# That is the worst outcome on precisely the machines this exists for. Four
# 3090s enumerated differently by the two libraries would have had every
# per-card figure attributed to the wrong physical card, silently.


def _fake_llama(tmp_path: Path, lines: list[str]) -> Path:
    """A stand-in for llama-server that prints the given device lines."""
    import sys

    if sys.platform == "win32":
        exe = tmp_path / "llama-server.bat"
        exe.write_text("@echo off\n" + "".join(f"echo {ln}\n" for ln in lines), encoding="ascii")
    else:
        exe = tmp_path / "llama-server"
        exe.write_text("#!/bin/sh\n" + "".join(f"echo '{ln}'\n" for ln in lines), encoding="ascii")
        exe.chmod(0o755)
    return exe


def _card(idx: int, name: str, total: int, free: int) -> Gpu:
    return Gpu(
        nvml_index=idx,
        name=name,
        memory_total_mib=total,
        memory_used_mib=total - free,
        memory_free_mib=free,
    )


def test_distinct_cards_reconcile_at_any_count(tmp_path: Path) -> None:
    """Four cards, llama.cpp enumerating them in the reverse of NVML order."""
    gpus = [
        _card(0, "NVIDIA GeForce RTX 3060", 12288, 11000),
        _card(1, "NVIDIA GeForce RTX 4070 SUPER", 12282, 10000),
        _card(2, "NVIDIA GeForce RTX 4080", 16376, 15000),
        _card(3, "NVIDIA GeForce RTX 4090", 24564, 24000),
    ]
    exe = _fake_llama(
        tmp_path,
        [
            "  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 24000 MiB free)",
            "  CUDA1: NVIDIA GeForce RTX 4080 (16376 MiB, 15000 MiB free)",
            "  CUDA2: NVIDIA GeForce RTX 4070 SUPER (12282 MiB, 10000 MiB free)",
            "  CUDA3: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)",
        ],
    )

    m = resolve_cuda_mapping(exe, gpus)

    assert m.cuda_to_nvml == {0: 3, 1: 2, 2: 1, 3: 0}
    assert order_differs(m) is True
    assert m.ambiguous == ()
    assert m.trustworthy is True


def test_identical_cards_are_pinned_by_free_vram(tmp_path: Path) -> None:
    """The card that matters is the constrained one, and free VRAM finds it.

    Four 3090s where NVML0 drives the display, and llama.cpp calls that card
    CUDA3. Getting this wrong points every headroom figure at the wrong card.
    """
    gpus = [
        _card(0, "NVIDIA GeForce RTX 3090", 24576, 21000),  # drives the display
        _card(1, "NVIDIA GeForce RTX 3090", 24576, 24000),
        _card(2, "NVIDIA GeForce RTX 3090", 24576, 24000),
        _card(3, "NVIDIA GeForce RTX 3090", 24576, 24000),
    ]
    exe = _fake_llama(
        tmp_path,
        [
            "  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 24000 MiB free)",
            "  CUDA1: NVIDIA GeForce RTX 3090 (24576 MiB, 24000 MiB free)",
            "  CUDA2: NVIDIA GeForce RTX 3090 (24576 MiB, 24000 MiB free)",
            "  CUDA3: NVIDIA GeForce RTX 3090 (24576 MiB, 21000 MiB free)",
        ],
    )

    m = resolve_cuda_mapping(exe, gpus)

    # The one that is distinguishable is resolved correctly, and the identity
    # mapping the old code invented would have put it at CUDA0.
    assert m.cuda_to_nvml[3] == 0
    assert order_differs(m) is True


def test_cards_that_cannot_be_told_apart_are_reported_as_guesses(tmp_path: Path) -> None:
    """Honest ambiguity beats a confident identity mapping.

    Three of the four 3090s above are genuinely indistinguishable -- same model,
    same free VRAM -- so those CUDA indices are guesses and must say so.
    """
    gpus = [_card(i, "NVIDIA GeForce RTX 3090", 24576, 24000) for i in range(4)]
    exe = _fake_llama(
        tmp_path,
        [f"  CUDA{i}: NVIDIA GeForce RTX 3090 (24576 MiB, 24000 MiB free)" for i in range(4)],
    )

    m = resolve_cuda_mapping(exe, gpus)

    assert m.resolved is True, "a partial mapping would be worse than a flagged one"
    assert m.trustworthy is False
    assert len(m.ambiguous) >= 2
    assert m.warning and "could not be pinned" in m.warning


def test_free_vram_within_noise_does_not_count_as_distinguishing(tmp_path: Path) -> None:
    """The two readings are taken moments apart and drift on their own.

    Desktop compositing alone moves free VRAM by tens of MiB, so a near-tie is a
    tie -- treating a 12 MiB gap as identification would be false precision.
    """
    gpus = [
        _card(0, "NVIDIA GeForce RTX 3090", 24576, 24000),
        _card(1, "NVIDIA GeForce RTX 3090", 24576, 23988),
    ]
    exe = _fake_llama(
        tmp_path,
        [
            "  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB, 23994 MiB free)",
            "  CUDA1: NVIDIA GeForce RTX 3090 (24576 MiB, 23994 MiB free)",
        ],
    )

    m = resolve_cuda_mapping(exe, gpus)

    assert m.trustworthy is False


def test_a_listing_with_no_free_column_still_maps_and_admits_it(tmp_path: Path) -> None:
    """Older llama.cpp builds print only the total. Nothing left to separate
    identical cards by, so the mapping is a guess and says so rather than
    silently reverting to the old behaviour."""
    gpus = [_card(i, "NVIDIA GeForce RTX 3090", 24576, 24000 - i) for i in range(2)]
    exe = _fake_llama(
        tmp_path,
        [
            "  CUDA0: NVIDIA GeForce RTX 3090 (24576 MiB)",
            "  CUDA1: NVIDIA GeForce RTX 3090 (24576 MiB)",
        ],
    )

    m = resolve_cuda_mapping(exe, gpus)

    assert m.resolved is True
    assert m.trustworthy is False


def test_a_single_card_needs_no_disambiguation(tmp_path: Path) -> None:
    gpus = [_card(0, "NVIDIA GeForce RTX 4090", 24564, 20000)]
    exe = _fake_llama(tmp_path, ["  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 20000 MiB free)"])

    m = resolve_cuda_mapping(exe, gpus)

    assert m.cuda_to_nvml == {0: 0}
    assert m.trustworthy is True
    assert order_differs(m) is False


# ---------------------------------------------------------------- server state


def test_status_distinguishes_loading_from_stopped() -> None:
    """Loading must not read as stopped, or the user starts a second server."""
    assert ServerState().status == "stopped"
    assert ServerState(running=True, pid=123).status == "loading"
    assert ServerState(running=True, pid=123, reachable=True).status == "running"


def test_status_orphaned_when_reachable_process_is_unknown() -> None:
    """Something answers the port but no matching process was found."""
    assert ServerState(running=True, pid=None).status == "orphaned"


# ---------------------------------------------------------------- argv


@pytest.fixture
def fake_registry(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "fake.gguf").write_bytes(b"GGUF")
    (weights / "mmproj.gguf").write_bytes(b"GGUF")

    doc = {
        "default": "demo",
        "models": {
            "_template": {"label": "ignored"},
            "demo": {
                "label": "Demo Model",
                "repo": "example/demo-GGUF",
                "file": "fake.gguf",
                "mmproj": "mmproj.gguf",
                "dir": str(weights).replace("\\", "/"),
                "size_gib": 1.0,
                "arch": "demo-arch",
                "serve": {
                    "ctx": 8192,
                    "ubatch": 512,
                    "batch": 2048,
                    "ngl": 99,
                    "devices": "CUDA0,CUDA1",
                    "split": "",
                    "flash_attn": "on",
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                    "cache_ram": 32768,
                    "parallel": 1,
                    "mtp": True,
                    "jinja": True,
                    "chat_template_file": None,
                    "sampling": {"temp": 1.0, "top_p": 0.95},
                },
                "vision": {
                    "supported": True,
                    "ctx": 4096,
                    "split": "0.4,0.6",
                    "image_min_tokens": 1024,
                },
                "measured": {"status": "MEASURED on this file"},
                "verified": {},
            },
        },
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_template_entries_are_not_runnable(fake_registry: Path) -> None:
    reg = load(fake_registry)
    assert "_template" not in reg.models
    assert reg.default == "demo"


def test_argv_carries_the_registry_values(fake_registry: Path) -> None:
    reg = load(fake_registry)
    argv = build_argv(reg.get(), "llama-server")

    def value_after(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert value_after("--ctx-size") == "8192"
    assert value_after("-ub") == "512"
    assert value_after("-ctk") == "q8_0"
    assert value_after("-dev") == "CUDA0,CUDA1"
    assert value_after("--temp") == "1.0"
    assert "--jinja" in argv
    assert "--spec-type" in argv, "speculative decoding was enabled in the registry"
    assert "--mmproj" not in argv, "vision was not requested"


def test_vision_applies_its_own_operating_point(fake_registry: Path) -> None:
    """Vision is a different profile, not a flag: it carries its own ctx and split."""
    reg = load(fake_registry)
    argv = build_argv(reg.get(), "llama-server", vision=True)

    assert argv[argv.index("--ctx-size") + 1] == "4096", "vision ctx overrides the text default"
    assert argv[argv.index("-ts") + 1] == "0.4,0.6"
    assert "--mmproj" in argv
    assert argv[argv.index("--image-min-tokens") + 1] == "1024"


def test_explicit_override_beats_the_vision_profile(fake_registry: Path) -> None:
    reg = load(fake_registry)
    argv = build_argv(reg.get(), "llama-server", vision=True, overrides={"ctx": 16384})
    assert argv[argv.index("--ctx-size") + 1] == "16384"


def test_large_micro_batch_with_mtp_is_refused(fake_registry: Path) -> None:
    reg = load(fake_registry)
    with pytest.raises(RegistryError, match="micro-batch"):
        build_argv(reg.get(), "llama-server", overrides={"ubatch": 2048})


def test_split_device_count_mismatch_is_refused(fake_registry: Path) -> None:
    reg = load(fake_registry)
    with pytest.raises(RegistryError, match="ratio"):
        build_argv(reg.get(), "llama-server", overrides={"split": "0.5,0.3,0.2"})


def test_missing_model_file_is_reported_clearly(fake_registry: Path, tmp_path: Path) -> None:
    reg = load(fake_registry)
    entry = reg.get()
    entry.file = "does-not-exist.gguf"
    with pytest.raises(RegistryError, match="missing"):
        build_argv(entry, "llama-server")


def test_measured_provenance_is_distinguished(fake_registry: Path) -> None:
    """Inherited numbers must not be indistinguishable from measured ones."""
    reg = load(fake_registry)
    entry = reg.get()
    assert entry.measured_on_this_file is True

    entry.measured = {"status": "INHERITED from a sibling build"}
    assert entry.measured_on_this_file is False


# ---------------------------------------------------------------- gguf analysis


def _analysis(**kw) -> GgufAnalysis:
    base = {"source": "test/model.gguf", "architecture": "demo", "tensor_count": 100}
    base.update(kw)
    return GgufAnalysis(**base)  # type: ignore[arg-type]


def _titles(a) -> list[str]:
    return [f.title for f in a.findings]


def _level_of(a, needle: str) -> str:
    from headroom.gguf import Finding

    match: Finding = next(f for f in a.findings if needle.lower() in f.title.lower())
    return match.level


def test_missing_speculative_head_is_flagged() -> None:
    from headroom.gguf import interpret

    a = _analysis(mtp_tensors=[])
    interpret(a)
    assert _level_of(a, "speculative") == "caution"

    b = _analysis(mtp_tensors=["blk.64.nextn.eh_proj"])
    interpret(b)
    assert _level_of(b, "speculative") == "good"


def test_protected_recurrent_layers_read_as_good() -> None:
    """The distinction the whole probe exists to draw."""
    from headroom.gguf import interpret

    protected = _analysis(families={"recurrent": {"F32": 192, "Q8_0": 96, "Q5_K": 33}})
    interpret(protected)
    assert _level_of(protected, "recurrent") == "good"

    uniform = _analysis(families={"recurrent": {"F32": 192, "Q4_K": 144}})
    interpret(uniform)
    assert _level_of(uniform, "recurrent") == "caution"


def test_f32_is_excluded_from_the_quantization_description() -> None:
    """F32 tensors are norms and biases, and the ratios already exclude them.

    Listing them in the prose would describe a distribution the adjacent numbers
    do not refer to.
    """
    from headroom.gguf import interpret

    a = _analysis(families={"recurrent": {"F32": 192, "Q4_K": 144}})
    interpret(a)
    detail = next(f.detail for f in a.findings if "recurrent" in f.title.lower())
    assert "144xQ4_K" in detail
    assert "F32" not in detail


def test_aggressively_quantized_attention_is_flagged() -> None:
    from headroom.gguf import interpret

    a = _analysis(families={"attention": {"F32": 99, "IQ3_S": 96, "Q8_0": 34}})
    interpret(a)
    assert _level_of(a, "attention") == "caution"


def test_fit_is_judged_against_real_free_vram() -> None:
    """A size in gibibytes means nothing without the machine it has to fit on."""
    from headroom.gguf import interpret

    size = 15 * 1024**3

    roomy = _analysis(file_size_bytes=size)
    interpret(roomy, free_vram_mib=22000)
    assert _level_of(roomy, "fit") == "good"

    snug = _analysis(file_size_bytes=size)
    interpret(snug, free_vram_mib=16000)
    assert _level_of(snug, "leaves little") == "caution"

    too_big = _analysis(file_size_bytes=size)
    interpret(too_big, free_vram_mib=8000)
    assert _level_of(too_big, "not fit") == "caution"


def test_fit_is_silent_without_telemetry() -> None:
    """With no GPU data, saying nothing beats guessing."""
    from headroom.gguf import interpret

    a = _analysis(file_size_bytes=15 * 1024**3)
    interpret(a, free_vram_mib=None)
    assert not any("fit" in t.lower() for t in _titles(a))


def test_non_gguf_input_explains_the_likely_cause() -> None:
    from io import BytesIO

    from headroom.gguf import GgufError, parse

    with pytest.raises(GgufError, match="gated"):
        parse(BytesIO(b"<html>401 Unauthorized</html>"), source="x")


# ---------------------------------------------------------------- registry writes


def test_adding_an_entry_preserves_everything_else(fake_registry: Path) -> None:
    """models.json belongs to the user and is shared with their shell scripts.

    Comment blocks, the template, and unrelated entries must survive a write
    untouched -- this app is a guest in that file.
    """
    from headroom.registry import add_entry

    original = json.loads(fake_registry.read_text(encoding="utf-8"))
    original["_comment"] = ["a comment the user wrote"]
    fake_registry.write_text(json.dumps(original, indent=2), encoding="utf-8")

    add_entry(fake_registry, "another", {"label": "Another", "serve": {}})

    after = json.loads(fake_registry.read_text(encoding="utf-8"))
    assert after["_comment"] == ["a comment the user wrote"]
    assert "_template" in after["models"], "the template must survive"
    assert after["models"]["demo"] == original["models"]["demo"], "existing entry changed"
    assert "another" in after["models"]


def test_a_backup_is_written_before_the_edit(fake_registry: Path) -> None:
    from headroom.registry import add_entry

    before = fake_registry.read_text(encoding="utf-8")
    add_entry(fake_registry, "another", {"label": "Another"})

    backup = fake_registry.with_suffix(fake_registry.suffix + ".bak")
    assert backup.exists(), "no backup was written"
    assert backup.read_text(encoding="utf-8") == before, "backup does not match the pre-edit file"


def test_existing_keys_are_refused(fake_registry: Path) -> None:
    """Silently replacing an entry would discard measurements someone earned."""
    from headroom.registry import RegistryError, add_entry

    with pytest.raises(RegistryError, match="already in the registry"):
        add_entry(fake_registry, "demo", {"label": "clobbered"})


def test_private_keys_are_refused(fake_registry: Path) -> None:
    from headroom.registry import RegistryError, add_entry

    with pytest.raises(RegistryError, match="private"):
        add_entry(fake_registry, "_sneaky", {"label": "x"})


def test_serve_block_is_not_inherited_across_architectures(fake_registry: Path) -> None:
    """The rule the whole registry design exists to enforce.

    A micro-batch or context size tuned for one architecture can be actively
    wrong for another, because the bottleneck moves. Copying it over would
    produce a config that looks authoritative and is not.
    """
    from headroom.registry import derive_entry, load

    reg = load(fake_registry)
    donor = reg.get("demo")
    assert donor.serve["ctx"] == 8192

    entry = derive_entry(
        key="different",
        label="Different Architecture",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture="some-other-arch",
        has_mtp=False,
        template={"serve": {"ctx": 4096}},
        inherit_from=donor,
    )

    assert entry["serve"]["ctx"] == 4096, "template default should win, not the donor's 8192"
    assert "does not transfer" in entry["measured"]["status"]


def test_serve_block_is_inherited_within_an_architecture_but_marked(fake_registry: Path) -> None:
    from headroom.registry import derive_entry, load

    reg = load(fake_registry)
    donor = reg.get("demo")

    entry = derive_entry(
        key="sibling",
        label="Same Architecture",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture=donor.arch,
        has_mtp=True,
        template={"serve": {"ctx": 4096}},
        inherit_from=donor,
    )

    assert entry["serve"]["ctx"] == 8192, "same architecture should inherit"
    assert "INHERITED" in entry["measured"]["status"]
    assert "NOT measured" in entry["measured"]["status"]
    assert entry["verified"]["benched"] is False, "inherited numbers are not verification"


def test_mtp_comes_from_the_probe_not_from_the_donor(fake_registry: Path) -> None:
    """Whether a speculative head exists is a fact about the file, not a guess."""
    from headroom.registry import derive_entry, load

    reg = load(fake_registry)
    donor = reg.get("demo")
    assert donor.serve["mtp"] is True

    entry = derive_entry(
        key="nomtp",
        label="No Speculative Head",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture=donor.arch,
        has_mtp=False,
        inherit_from=donor,
    )
    assert entry["serve"]["mtp"] is False, "the probe found no head; the donor's true must not win"


def test_speculative_decoding_forces_a_small_micro_batch(fake_registry: Path) -> None:
    """The draft context's buffers scale with -ub, so the two are coupled."""
    from headroom.registry import derive_entry

    entry = derive_entry(
        key="spec",
        label="Speculative",
        repo="x/y",
        filename="y.gguf",
        directory="/tmp/y",
        size_gib=5.0,
        architecture="a",
        has_mtp=True,
        template={"serve": {"ubatch": 4096}},
    )
    assert entry["serve"]["ubatch"] == 512


def test_env_paths_tolerate_trailing_whitespace(monkeypatch) -> None:
    """A trailing space in an env var must not leak into derived paths.

    `set VAR=value && cmd` in cmd.exe captures the space before the `&&`.
    Windows still opens the file, so nothing looks wrong -- but every derived
    path inherits the space, and a backup lands as "models.json .bak".
    """
    from headroom.app import Settings

    monkeypatch.setenv("HEADROOM_REGISTRY", "C:/models/models.json   ")
    settings = Settings.resolve(create_registry=False)

    assert str(settings.registry_path).endswith("models.json")
    derived = settings.registry_path.with_suffix(settings.registry_path.suffix + ".bak")
    assert derived.name == "models.json.bak"


# ---------------------------------------------------------------- discovery


def test_explicit_argument_beats_the_environment(monkeypatch, tmp_path: Path) -> None:
    from headroom.config import resolve_registry

    monkeypatch.setenv("HEADROOM_REGISTRY", str(tmp_path / "from-env.json"))
    r = resolve_registry(str(tmp_path / "explicit.json"))
    assert r.path is not None and r.path.name == "explicit.json"
    assert r.source == "argument"


def test_environment_beats_discovery(monkeypatch, tmp_path: Path) -> None:
    from headroom.config import resolve_registry

    target = tmp_path / "from-env.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HEADROOM_REGISTRY", str(target))
    r = resolve_registry()
    assert r.path == target
    assert r.source == "HEADROOM_REGISTRY"
    assert r.exists


def test_a_fresh_machine_gets_a_usable_registry(monkeypatch, tmp_path: Path) -> None:
    """The whole point of this module.

    With nothing configured and nothing installed, Headroom must still come up
    with somewhere to put a model. An app that errors until the user reads the
    source is not a working app.
    """
    from headroom.config import resolve_registry
    from headroom.registry import load

    monkeypatch.delenv("HEADROOM_REGISTRY", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    r = resolve_registry(create=True)
    assert r.exists, "no registry was created"
    assert r.path is not None

    # And it must be loadable, not merely present.
    reg = load(r.path)
    assert reg.models == {}, "a starter registry has no real models, only a template"


def test_a_missing_llama_server_is_reported_not_guessed(monkeypatch, tmp_path: Path) -> None:
    """Reporting nothing beats inventing a path that does not exist."""
    from headroom.config import resolve_llama_server

    monkeypatch.delenv("HEADROOM_LLAMA_SERVER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    r = resolve_llama_server()
    if r.exists:
        pytest.skip("a real llama-server is installed in a conventional location")
    assert r.path is None, "a not-found result must not carry a fabricated path"
    assert r.source == "not found"
    assert r.searched, "the search locations should be reported so the user can fix it"


def test_an_empty_registry_says_so_rather_than_reporting_a_typo(tmp_path: Path) -> None:
    """A fresh install is not a mistyped model name."""
    from headroom.config import write_starter_registry
    from headroom.registry import RegistryError, load

    path = tmp_path / "models.json"
    write_starter_registry(path)
    reg = load(path)

    with pytest.raises(RegistryError, match="no models in the registry yet"):
        reg.get()


# ---------------------------------------------------------------- degraded environments


def test_the_app_serves_without_a_gpu_or_a_registry(monkeypatch, tmp_path: Path) -> None:
    """Headroom must come up on a machine with none of its dependencies.

    This lived as an inline Python snippet inside the CI workflow, where it was
    unlinted, untested locally, and free to drift -- which it promptly did,
    calling a constructor signature that had changed. YAML is a poor place to
    keep code. Here it runs on every machine, including the developer's.
    """
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("HEADROOM_REGISTRY", raising=False)
    monkeypatch.delenv("HEADROOM_LLAMA_SERVER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    # Pinned, because "degraded install" is the subject here and "nothing is
    # serving" is only the backdrop. Left to the real probe this asserted a fact
    # about the developer's machine instead: on a box with a model up -- which
    # is the normal state while working on the benchmark -- it failed for a
    # reason unrelated to anything it tests.
    async def nothing_serving(_port, timeout: float = 3.0):
        return ServerState()

    monkeypatch.setattr("headroom.server.probe", nothing_serving)

    settings = Settings.resolve()

    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        assert health["ok"] is True, health
        assert "problems" in health, "a degraded install must say what it is missing"

        # Whatever the hardware, this must be a list rather than an error.
        gpus = client.get("/api/gpus").json()
        assert isinstance(gpus["gpus"], list)

        # No server running, and no llama.cpp to start one with.
        assert client.get("/api/server").json()["status"] == "stopped"

        # The registry was created, so this is 200 with nothing in it -- not a 404.
        models = client.get("/api/models")
        assert models.status_code == 200, models.text
        assert models.json()["models"] == []


def test_starting_without_llama_cpp_explains_itself(monkeypatch, tmp_path: Path) -> None:
    """503 with a reason beats a traceback or a silent no-op."""
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("HEADROOM_LLAMA_SERVER", raising=False)
    monkeypatch.delenv("HEADROOM_REGISTRY", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    settings = Settings.resolve()
    if settings.llama_server is not None:
        pytest.skip("llama-server is installed in a conventional location here")

    with TestClient(create_app(settings)) as client:
        resp = client.post("/api/server/start")
        assert resp.status_code == 503
        assert "llama-server" in resp.json()["detail"]


# ---------------------------------------------------------------- projectors


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("mmproj-F16.gguf", True),
        ("mmproj-Unleashed-f16.gguf", True),
        ("Qwen3-VL-mmproj-f16.gguf", True),
        ("MMPROJ-MODEL-F16.GGUF", True),
        ("Qwen3.8-27B-UD-Q4_K_M.gguf", False),
        # A model that merely mentions projection is not a projector. Matching
        # loosely here would silently swap the weights for a few hundred MiB.
        ("some-projection-model-Q4.gguf", False),
    ],
)
def test_projectors_are_recognised_by_name(filename: str, expected: bool) -> None:
    from headroom.gguf import is_projector

    assert is_projector(filename) is expected


def test_the_highest_precision_projector_is_suggested() -> None:
    from headroom.gguf import choose_projector

    files = [
        {"filename": "model-Q4_K_M.gguf", "size_bytes": 16_000_000_000, "kind": "model"},
        {"filename": "mmproj-q8_0.gguf", "size_bytes": 500_000_000, "kind": "projector"},
        {"filename": "mmproj-f16.gguf", "size_bytes": 900_000_000, "kind": "projector"},
    ]
    # f16 over q8_0: the projector is small next to the weights, so quantizing it
    # saves little and image fidelity is what pays for it.
    assert choose_projector(files) == "mmproj-f16.gguf"


def test_no_projector_in_the_repo_suggests_nothing() -> None:
    from headroom.gguf import choose_projector

    assert choose_projector([{"filename": "m.gguf", "size_bytes": 1, "kind": "model"}]) is None


def test_an_unrecognised_precision_falls_back_to_the_largest() -> None:
    from headroom.gguf import choose_projector

    files = [
        {"filename": "mmproj-weird.gguf", "size_bytes": 100, "kind": "projector"},
        {"filename": "mmproj-odd.gguf", "size_bytes": 900, "kind": "projector"},
    ]
    assert choose_projector(files) == "mmproj-odd.gguf"


def test_an_added_entry_carries_its_projector(fake_registry: Path) -> None:
    from headroom.registry import derive_entry

    entry = derive_entry(
        key="vlm",
        label="A vision model",
        repo="owner/vlm-GGUF",
        filename="vlm-Q4_K_M.gguf",
        directory="/w/vlm",
        size_gib=8.0,
        architecture="new-arch",
        has_mtp=False,
        mmproj="mmproj-f16.gguf",
    )

    assert entry["mmproj"] == "mmproj-f16.gguf"
    assert entry["vision"]["supported"] is True
    # A projector existing is not a tuned operating point, and the entry has to
    # say which of the two it has.
    assert entry["vision"]["tuned"] is False
    assert any("NOT tuned" in line for line in entry["why_this_build"])


def test_a_model_without_a_projector_is_not_marked_vision_capable(fake_registry: Path) -> None:
    from headroom.registry import derive_entry

    entry = derive_entry(
        key="plain",
        label="Text only",
        repo="owner/plain-GGUF",
        filename="plain-Q4_K_M.gguf",
        directory="/w/plain",
        size_gib=8.0,
        architecture="new-arch",
        has_mtp=False,
    )

    assert entry["mmproj"] is None
    assert entry["vision"] == {"supported": False}


def test_an_inherited_vision_profile_counts_as_tuned(fake_registry: Path) -> None:
    from headroom.registry import derive_entry, load

    donor = load(fake_registry).models["demo"]
    entry = derive_entry(
        key="sibling",
        label="Same architecture",
        repo="owner/sibling-GGUF",
        filename="sibling-Q4_K_M.gguf",
        directory="/w/sibling",
        size_gib=8.0,
        architecture=donor.arch,
        has_mtp=False,
        inherit_from=donor,
        mmproj="mmproj-f16.gguf",
    )

    # The donor's profile carries a real ctx and split, measured on that build.
    assert entry["vision"]["ctx"] == 4096
    assert entry["vision"]["tuned"] is True


def test_adding_a_vision_model_attaches_and_fetches_its_projector(
    monkeypatch, tmp_path: Path, fake_registry: Path
) -> None:
    """The whole point of the feature, through the API.

    Not exercised against a live repository on purpose: the real path downloads
    tens of gigabytes, and what needs proving here is the wiring, not the hub.
    """
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    async def fake_probe(repo, file, free_vram_mib=None):
        return _analysis(
            architecture="demo-arch",
            name="A vision model",
            file_size_bytes=8 * 1024**3,
        )

    monkeypatch.setattr("headroom.gguf.probe_remote", fake_probe)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HEADROOM_REGISTRY", str(fake_registry))
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app(Settings.resolve())) as client:
        resp = client.post(
            "/api/registry/add",
            params={
                "key": "vlm",
                "repo": "owner/vlm-GGUF",
                "file": "vlm-Q4_K_M.gguf",
                "mmproj": "mmproj-f16.gguf",
                "download": "false",
            },
        )
        assert resp.status_code == 200, resp.text
        entry = resp.json()["entry"]

    assert entry["mmproj"] == "mmproj-f16.gguf"
    assert entry["vision"]["supported"] is True
    assert entry["vision"]["tuned"] is False

    # And it round-trips: the registry on disk now describes a vision model.
    written = load(fake_registry).models["vlm"]
    assert written.mmproj == "mmproj-f16.gguf"
    assert written.vision_tuned is False


def test_adding_without_a_projector_leaves_the_entry_text_only(
    monkeypatch, tmp_path: Path, fake_registry: Path
) -> None:
    from fastapi.testclient import TestClient

    from headroom.app import Settings, create_app

    async def fake_probe(repo, file, free_vram_mib=None):
        return _analysis(architecture="demo-arch", file_size_bytes=8 * 1024**3)

    monkeypatch.setattr("headroom.gguf.probe_remote", fake_probe)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("HEADROOM_REGISTRY", str(fake_registry))
    monkeypatch.chdir(tmp_path)

    with TestClient(create_app(Settings.resolve())) as client:
        resp = client.post(
            "/api/registry/add",
            params={
                "key": "plain",
                "repo": "owner/plain-GGUF",
                "file": "plain-Q4_K_M.gguf",
                "download": "false",
            },
        )
        assert resp.status_code == 200, resp.text

    written = load(fake_registry).models["plain"]
    assert written.mmproj is None
    assert written.vision_tuned is False
    # Claiming vision without a projector is the failure this guards: build_argv
    # would accept the request and then die on a missing file.
    assert written.vision.get("supported") is not True
