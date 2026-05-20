# MGMemory

This baseline is provided by the upstream **memengine** meta-repository
(a single package shipping multiple memory algorithms).  The source
lives at:

- Main class: `../memengine/memory/MGMemory.py`
- Config:     `(uses default memengine config)`
- Shared library: `../memengine/{function,operation,utils}`

The eval adapter (`eval_framework/memory_adapters/memengine_native.py`)
loads this baseline via `registry.get_memory_class("MGMemory")` — it adds
`baselines/memengine/` to `sys.path` and imports directly.

**Why a thin wrapper dir?**  The project convention is that every
baseline has an independent `baselines/<Name>/` entry so the inventory
is uniform, even when several baselines share one upstream package.  We
do **not** duplicate the memengine source tree here (~1 MB × 4 would
waste disk and break `git status` on upstream changes) — we link via
the symlinked `upstream` below plus this README pointer.

## Dependencies

See `../memengine/environment.yml` or `../memengine/setup.py` for the
full dep list.  OpenAI endpoint + embedding model come from the top-level
project `.env` (`OPENAI_BASE_URL`, `OPENAI_MODEL`,
`OPENAI_EMBEDDING_MODEL`).

## Upstream modifications

- None currently.

## Entry point

Run via:
```bash
python -m eval_framework.cli --baseline MGMemory --dataset <dataset> --output-dir <out> --smoke
```
