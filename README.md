# Headroom

[![CI](https://github.com/norkington/headroom/actions/workflows/ci.yml/badge.svg)](https://github.com/norkington/headroom/actions/workflows/ci.yml)

**An operations console for running large language models on constrained consumer GPUs.**

Not another chat UI. Headroom is for the part that actually goes wrong when you
run a 27B on two 12 GB cards: knowing how much VRAM you have left, which card is
which, whether a quant is worth downloading, and what has to stop before you can
start something else.

> **Status: early.** Working end to end on real hardware, but the API is not
> stable yet. NVIDIA-only for now.

---

## Why this exists

There are plenty of good local-LLM chat interfaces — Open WebUI, LM Studio, Jan,
KoboldCpp. This is not one of them, and if you want chat you should use one of
those.

What none of them do well is treat **the hardware as the thing you are
operating**. On a constrained machine the interesting questions are not "what did
the model say" but:

- **How much VRAM is actually free, on the card that matters?** A rig can have
  3 GB free "in total" while the card also driving your desktop sits at 400 MB
  and is one browser tab away from OOMing the server mid-generation. Totals lie;
  Headroom grades each card on its own.

- **Which card does llama.cpp call `CUDA0`?** NVML and `nvidia-smi` enumerate by
  PCI bus. llama.cpp uses the CUDA runtime's FASTEST_FIRST order. When those
  disagree — a fast card in a later slot — `-dev CUDA0` and `nvidia-smi -i 0`
  refer to **different GPUs**, and every tuning decision made from the wrong one
  is wasted. Headroom reconciles them against llama.cpp's own `--list-devices`
  and says so loudly when they differ.

- **Is this quant worth 15 GiB of download?** Two builds of the same model at the
  same file size can differ a lot in quality, and the difference is visible only
  in the tensor table. On hybrid architectures (Qwen3.5+ and friends) most blocks
  are recurrent; quantization error in those **accumulates with context depth**,
  while feed-forward error does not compound. A build keeping `ssm_*` tensors at
  Q8_0 will hold up at long context where an all-Q4_K build of identical size
  degrades. Headroom reads a remote GGUF's tensor table over a ranged request —
  tens of megabytes, not the whole file — so you can see that before committing.

- **What has to stop before I can start this?** Inference, diffusion, LoRA
  training and games all want the same cards. Headroom knows what is holding
  them and can hand them over.

## Design principles

**Nothing leaves your machine.** Headroom binds to loopback only. It has no
accounts, no telemetry, and no cloud dependency of any kind. The only outbound
traffic is downloading model weights you explicitly asked for.

**Headroom never owns your inference server.** `llama-server` is spawned
*detached* and Headroom *attaches* to it, discovering state from the server's own
`/props` endpoint and its command line. Closing the UI leaves your model
resident. This is deliberate: reloading a 15 GiB model costs minutes, and no
dashboard should be able to cost you that by being closed.

**Your config file is the source of truth.** Headroom reads and writes the same
registry your scripts use. It does not keep a private copy of your settings,
because two sources of truth is how a CLI and a GUI drift into disagreeing about
what is running.

**Measurements are reported honestly.** Throughput figures come with a standard
deviation and the sample size. Where a number is inherited rather than measured
on the exact artifact in front of you, it says so.

## Requirements

- Python 3.11+
- An NVIDIA GPU with a working driver (NVML)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) built with `llama-server`

## Running it

```bash
git clone https://github.com/norkington/headroom
cd headroom
uv venv
uv pip install -e ".[dev]"

cd web && npm install && npm run build && cd ..

headroom
```

That serves the UI and the API from a **single process** on
<http://127.0.0.1:7315>. The frontend builds into `src/headroom/static/`, which
is what makes it ship inside the wheel rather than being left behind by `pip`.

If you skip the frontend build, the API still works and the root route says so
rather than returning a bare 404.

### Working on the frontend

```bash
npm run dev    # in web/ -- port 7316, proxies /api to the backend on 7315
```

Run `headroom` alongside it for the API.

### Tests

```bash
uv run pytest
uv run ruff check src tests
```

The suite is in two halves. `test_units.py` is portable and runs anywhere.
`test_argv_parity.py` checks Headroom's generated command line against a real
shell launcher's own dry-run output, so it needs a local llama.cpp install and
skips cleanly without one.

## Licence

MIT — see [LICENSE](LICENSE).
