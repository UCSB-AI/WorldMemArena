# OpenClaw General Runner

Standalone Python wrapper that drives the [OpenClaw](https://github.com/openclaw/openclaw) CLI in an isolated config/state directory, used by the `Harness-OpenClaw-*` baselines.

`run_openclaw_general.py` is invoked as a subprocess by [`eval_framework/baselines/_clients/harness_api.py`](../eval_framework/baselines/_clients/harness_api.py); the path is set via the `runner:` field in [`eval_framework/baselines/_clients/harness_config.yaml`](../eval_framework/baselines/_clients/harness_config.yaml).

## Prerequisites

- `openclaw` CLI on `PATH`: `npm install -g openclaw`
- Python 3.10+
- A model provider configured in `eval_framework/.env` (e.g. `OPENAI_API_KEY` + `OPENAI_BASE_URL`)

## Standalone usage

```bash
python OpenClaw_General/run_openclaw_general.py \
  --prompt "Summarize the file at /tmp/notes.md" \
  --model gpt-4o
```

See `run_openclaw_general.py --help` for all flags. The script does not depend on any other module in this repository.
