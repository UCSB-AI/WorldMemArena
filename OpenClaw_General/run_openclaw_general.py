#!/usr/bin/env python3
"""Run a general OpenClaw local prompt with isolated config/state.

This script is standalone. It does not use benchmark task YAMLs and does not
start benchmark MCP servers by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

KEEP_OPENCLAW_ISOLATION_ENV = "OPENCLAW_GENERAL_KEEP_ISOLATION"
OPENCLAW_ISOLATION_ROOT_ENV = "OPENCLAW_GENERAL_ISOLATION_ROOT"
OPENCLAW_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class OpenClawLocalSession:
    root_dir: Path
    workspace: Path
    config_path: Path
    state_dir: Path
    session_path: Path
    agent_id: str
    session_id: str
    command: tuple[str, ...]
    env: dict[str, str]


class OpenClawBootstrapManager:
    """Temporarily hide OpenClaw's global BOOTSTRAP.md during a run."""

    def __init__(self) -> None:
        self._backup_content: str | None = None

    @staticmethod
    def _bootstrap_path() -> Path:
        return Path.home() / ".openclaw" / "workspace" / "BOOTSTRAP.md"

    def suppress(self) -> None:
        if self._backup_content is not None:
            return

        bootstrap = self._bootstrap_path()
        if not bootstrap.exists():
            return

        self._backup_content = bootstrap.read_text(encoding="utf-8")
        bootstrap.unlink()

    def restore(self) -> None:
        if self._backup_content is None:
            return

        bootstrap = self._bootstrap_path()
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.write_text(self._backup_content, encoding="utf-8")
        self._backup_content = None


class OpenClawLocalRuntime:
    """Build and clean up an isolated OpenClaw local-mode runtime."""

    def __init__(self) -> None:
        self._root_dir: Path | None = None
        self._bootstrap_manager = OpenClawBootstrapManager()

    def prepare_session(
        self,
        *,
        run_id: str,
        model: str,
        workspace: Path | None,
        timeout_seconds: int,
        agent_id: str = "general",
        agent_name: str = "General OpenClaw",
        inherit_mcp_servers: bool = False,
    ) -> OpenClawLocalSession:
        self._bootstrap_manager.suppress()
        root_dir = _mk_isolation_root(
            prefix="openclaw-general-",
            root_env=OPENCLAW_ISOLATION_ROOT_ENV,
        )
        self._root_dir = root_dir

        workspace_path = workspace or (root_dir / "workspace")
        workspace_path.mkdir(parents=True, exist_ok=True)
        if workspace is None:
            _mark_workspace_setup_completed(workspace_path)

        state_dir = root_dir / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        config_path = root_dir / "openclaw.json"

        session_id = run_id
        session_path = state_dir / "agents" / agent_id / "sessions" / f"{session_id}.jsonl"

        config = _load_json_object(Path.home() / ".openclaw" / "openclaw.json")
        _configure_agent(
            config,
            agent_id=agent_id,
            agent_name=agent_name,
            workspace_path=workspace_path,
            timeout_seconds=timeout_seconds,
            model=model,
        )
        _configure_mcp(config, inherit_mcp_servers=inherit_mcp_servers)
        _configure_openai_model(config, model=model)

        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        env = dict(os.environ)
        env["OPENCLAW_CONFIG_PATH"] = str(config_path)
        env["OPENCLAW_STATE_DIR"] = str(state_dir)

        command = (
            "openclaw",
            "agent",
            "--local",
            "--agent",
            agent_id,
            "--session-id",
            session_id,
            "--json",
        )

        return OpenClawLocalSession(
            root_dir=root_dir,
            workspace=workspace_path,
            config_path=config_path,
            state_dir=state_dir,
            session_path=session_path,
            agent_id=agent_id,
            session_id=session_id,
            command=command,
            env=env,
        )

    def cleanup(self) -> None:
        try:
            if self._root_dir is not None:
                if os.environ.get(KEEP_OPENCLAW_ISOLATION_ENV) != "1":
                    shutil.rmtree(self._root_dir, ignore_errors=True)
                self._root_dir = None
        finally:
            self._bootstrap_manager.restore()


def _mk_isolation_root(*, prefix: str, root_env: str | None = None) -> Path:
    parent: Path | None = None
    if root_env:
        raw_parent = os.environ.get(root_env, "").strip()
        if raw_parent:
            parent = Path(raw_parent).expanduser()
            parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent) if parent else None))


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _mark_workspace_setup_completed(workspace_path: Path) -> None:
    state_path = workspace_path / ".openclaw" / "workspace-state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "setupCompletedAt": datetime.now(timezone.utc).isoformat(),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _configure_agent(
    config: dict[str, object],
    *,
    agent_id: str,
    agent_name: str,
    workspace_path: Path,
    timeout_seconds: int,
    model: str,
) -> None:
    agents_config = config.get("agents")
    if not isinstance(agents_config, dict):
        agents_config = {}
    else:
        agents_config = dict(agents_config)

    defaults = agents_config.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {}
    else:
        defaults = dict(defaults)

    defaults.update(
        {
            "workspace": str(workspace_path),
            "timeoutSeconds": timeout_seconds,
            "model": {"primary": model},
        }
    )
    agents_config["defaults"] = defaults
    agents_config["list"] = [
        {
            "id": agent_id,
            "name": agent_name,
            "workspace": str(workspace_path),
            "model": {"primary": model},
        }
    ]
    config["agents"] = agents_config


def _configure_mcp(config: dict[str, object], *, inherit_mcp_servers: bool) -> None:
    if inherit_mcp_servers:
        return

    mcp_config = config.get("mcp")
    if not isinstance(mcp_config, dict):
        mcp_config = {}
    else:
        mcp_config = dict(mcp_config)
    mcp_config["servers"] = {}
    config["mcp"] = mcp_config


def _openai_model_id(model: str) -> str | None:
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    if "/" not in model:
        return model
    return None


def _configure_openai_model(config: dict[str, object], *, model: str) -> None:
    model_id = _openai_model_id(model)
    if not model_id:
        return

    models_cfg = config.setdefault("models", {})
    if not isinstance(models_cfg, dict):
        models_cfg = {}
        config["models"] = models_cfg

    providers = models_cfg.setdefault("providers", {})
    if not isinstance(providers, dict):
        providers = {}
        models_cfg["providers"] = providers

    openai_cfg = providers.setdefault("openai", {})
    if not isinstance(openai_cfg, dict):
        openai_cfg = {}
        providers["openai"] = openai_cfg

    openai_cfg["baseUrl"] = (
        os.environ.get("OPENCLAW_OPENAI_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or openai_cfg.get("baseUrl")
        or OPENCLAW_DEFAULT_OPENAI_BASE_URL
    )
    model_api = os.environ.get("OPENCLAW_MODEL_API", "").strip()
    if model_api:
        openai_cfg["api"] = model_api
        # Force the embedded PI runtime instead of the default Codex
        # app-server harness for `openai/*` model refs; the Codex harness
        # plugin is not installed in evaluation environments.
        openai_cfg.setdefault("agentRuntime", {"id": "pi"})
    openai_cfg["models"] = [{"id": model_id, "name": model_id}]


def _load_dotenv_file(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    if args.message:
        return " ".join(args.message)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide a prompt, a prompt file, a message, or stdin.")


def _stage_files(files: list[str], workspace: Path) -> list[Path]:
    if not files:
        return []

    asset_dir = workspace / "input_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    used_names: set[str] = set()

    for raw in files:
        source = Path(raw).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Attachment is not a file: {source}")

        target_name = source.name
        if target_name in used_names:
            stem = source.stem or "file"
            suffix = source.suffix
            index = 2
            while f"{stem}_{index}{suffix}" in used_names:
                index += 1
            target_name = f"{stem}_{index}{suffix}"
        used_names.add(target_name)

        target = asset_dir / target_name
        shutil.copy2(source, target)
        staged.append(target)

    return staged


def _prompt_with_attachments(prompt: str, staged: list[Path], workspace: Path) -> str:
    if not staged:
        return prompt

    lines = [
        prompt.rstrip(),
        "",
        "Attached files are available in the current workspace:",
    ]
    for path in staged:
        lines.append(f"- {path.relative_to(workspace).as_posix()}")
    lines.extend(
        [
            "",
            "Use these files directly when answering the user's request.",
        ]
    )
    return "\n".join(lines)


def _read_final_output(session_path: Path, *, created_after: float) -> tuple[str, str | None]:
    if not session_path.exists():
        return "", f"OpenClaw session file not found: {session_path}"
    if created_after > 0.0 and session_path.stat().st_mtime <= created_after:
        return "", f"OpenClaw session is older than this run: {session_path}"

    final_output = ""
    try:
        with session_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") != "message":
                    continue
                message = record.get("message", {})
                if message.get("role") != "assistant":
                    continue
                content = message.get("content", [])
                if isinstance(content, str):
                    final_output = content
                    continue
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and item.get("text"):
                        final_output = str(item["text"])
    except json.JSONDecodeError as exc:
        return final_output, f"Failed to parse OpenClaw session {session_path}: {exc}"

    if not final_output:
        return "", f"OpenClaw session contained no assistant text: {session_path}"
    return final_output, None


def _load_env_files(args: argparse.Namespace, *, script_dir: Path) -> None:
    _load_dotenv_file(script_dir / ".env")
    cwd_env = Path.cwd() / ".env"
    if cwd_env.resolve() != (script_dir / ".env").resolve():
        _load_dotenv_file(cwd_env)
    if args.env_file:
        _load_dotenv_file(Path(args.env_file).expanduser(), override=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a general OpenClaw local prompt with isolated config/state.",
    )
    parser.add_argument("message", nargs="*", help="Prompt text if --prompt is not used.")
    parser.add_argument("--prompt", "-p", help="Prompt text.")
    parser.add_argument("--prompt-file", help="Read prompt text from a file.")
    parser.add_argument("--file", action="append", default=[], help="Attach a file. Repeatable.")
    parser.add_argument("--workspace", help="Existing project directory to use as OpenClaw cwd.")
    parser.add_argument("--model", default="gpt-5.4", help="OpenClaw model id.")
    parser.add_argument("--timeout", type=int, default=360, help="Agent timeout in seconds.")
    parser.add_argument("--env-file", help="Load an extra .env file, overriding existing values.")
    parser.add_argument(
        "--keep-isolation",
        action="store_true",
        help="Keep the temporary OpenClaw config/state/workspace after the run.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print isolation/session paths after the final answer.",
    )
    parser.add_argument(
        "--inherit-mcp",
        action="store_true",
        help="Keep MCP servers from ~/.openclaw/openclaw.json. Off by default for speed.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    _load_env_files(args, script_dir=script_dir)

    if args.keep_isolation:
        os.environ[KEEP_OPENCLAW_ISOLATION_ENV] = "1"

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else None
    if workspace is not None:
        workspace.mkdir(parents=True, exist_ok=True)

    run_id = f"general-openclaw-{uuid.uuid4().hex[:8]}"
    runtime = OpenClawLocalRuntime()
    session = runtime.prepare_session(
        run_id=run_id,
        model=args.model,
        workspace=workspace,
        timeout_seconds=args.timeout,
        inherit_mcp_servers=args.inherit_mcp,
    )

    try:
        prompt = _read_prompt(args)
        staged = _stage_files(args.file, session.workspace)
        prompt = _prompt_with_attachments(prompt, staged, session.workspace)

        started_at = time.time()
        completed = subprocess.run(
            [*session.command, "--message", prompt],
            cwd=str(session.workspace),
            env=session.env,
            capture_output=True,
            text=True,
            timeout=args.timeout + 30,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            print(detail, file=sys.stderr)
            return completed.returncode

        final_output, warning = _read_final_output(session.session_path, created_after=started_at)
        if final_output:
            print(final_output)
        elif completed.stdout.strip():
            print(completed.stdout.strip())
        if warning:
            print(f"WARNING: {warning}", file=sys.stderr)

        if args.show_paths or args.keep_isolation:
            print(f"\nworkspace: {session.workspace}", file=sys.stderr)
            print(f"config:    {session.config_path}", file=sys.stderr)
            print(f"state:     {session.state_dir}", file=sys.stderr)
            print(f"session:   {session.session_path}", file=sys.stderr)
        return 0
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
