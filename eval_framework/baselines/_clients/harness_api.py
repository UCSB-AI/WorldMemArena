"""Runtime helpers for terminal harness baselines."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_framework.openai_compat import empty_token_usage


@dataclass(frozen=True)
class HarnessTarget:
    provider: str
    baseline_name: str
    kind: str
    runner: str
    api_key: str
    base_url: str
    codex_home: Path | None
    workdir: Path
    model: str
    timeout: int
    mm_mode: str
    keep_isolation: bool
    show_paths: bool
    sandbox: str
    approval_policy: str
    ephemeral: bool
    model_context_window: int | None
    model_auto_compact_token_limit: int | None
    model_api: str


def _normalize_mm_mode(raw: Any) -> str:
    mode = str(raw or "text").strip().lower()
    if mode in {"image", "mm", "multimodal"}:
        return "image"
    return "text"


def build_harness_target(baseline_name: str) -> HarnessTarget | None:
    from eval_framework.config import (
        harness_baseline_name,
        resolve_harness_config_for_baseline,
    )

    cfg = resolve_harness_config_for_baseline(baseline_name)
    if cfg is None:
        return None
    provider = str(cfg.get("key") or baseline_name)
    kind = str(cfg.get("kind") or "openclaw_general").strip().lower()
    runner_raw = str(cfg.get("runner") or ("codex" if kind == "codex_exec" else "")).strip()
    if not runner_raw:
        raise RuntimeError(f"{baseline_name}: missing harness runner")
    api_key = str(cfg.get("api_key") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    if kind == "openclaw_general":
        if not api_key:
            raise RuntimeError(f"{baseline_name}: missing api_key in harness_config.yaml")
        if not base_url:
            raise RuntimeError(f"{baseline_name}: missing base_url in harness_config.yaml")
    codex_home_raw = str(cfg.get("codex_home") or "").strip()
    workdir_raw = str(cfg.get("workdir") or "/tmp").strip()
    return HarnessTarget(
        provider=provider,
        baseline_name=str(cfg.get("baseline") or harness_baseline_name(provider)),
        kind=kind,
        runner=runner_raw,
        api_key=api_key,
        base_url=base_url,
        codex_home=Path(codex_home_raw).expanduser().resolve() if codex_home_raw else None,
        workdir=Path(workdir_raw).expanduser().resolve(),
        model=str(cfg.get("model") or ""),
        timeout=int(cfg.get("timeout", 360)),
        mm_mode=_normalize_mm_mode(cfg.get("mm_mode")),
        keep_isolation=bool(cfg.get("keep_isolation", False)),
        show_paths=bool(cfg.get("show_paths", False)),
        sandbox=str(cfg.get("sandbox") or "read-only"),
        approval_policy=str(cfg.get("approval_policy") or "never"),
        ephemeral=bool(cfg.get("ephemeral", True)),
        model_context_window=(
            int(cfg["model_context_window"])
            if cfg.get("model_context_window") is not None
            else None
        ),
        model_auto_compact_token_limit=(
            int(cfg["model_auto_compact_token_limit"])
            if cfg.get("model_auto_compact_token_limit") is not None
            else None
        ),
        model_api=str(cfg.get("model_api") or "").strip(),
    )


def _limited_files(
    image_paths: list[str],
    *,
    max_images: int,
    max_image_bytes: int,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    for raw in image_paths:
        if not raw or raw in seen:
            continue
        seen.add(raw)
        path = Path(raw).expanduser()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if max_image_bytes > 0 and total_bytes + size > max_image_bytes:
            continue
        out.append(str(path.resolve()))
        total_bytes += size
        if max_images > 0 and len(out) >= max_images:
            break
    return out


def _compose_prompt(*, system_prompt: str, user_prompt: str) -> str:
    return (
        "You are being invoked through a terminal harness for long-context QA.\n"
        "Do not use external tools, shell commands, or workspace files unless the "
        "prompt explicitly lists attached files.\n\n"
        "System instructions:\n"
        f"{system_prompt.strip()}\n\n"
        "User request:\n"
        f"{user_prompt.strip()}\n"
    )


def _resolve_command(runner: str) -> str:
    if "/" in runner:
        path = Path(runner).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"harness runner not found: {path}")
        return str(path)
    found = shutil.which(runner)
    if not found:
        raise RuntimeError(f"harness command not found on PATH: {runner}")
    return found


def harness_env_overrides(target: HarnessTarget) -> dict[str, str]:
    """Environment values sourced directly from ``harness_config.yaml``."""
    env: dict[str, str] = {}
    if target.api_key:
        env["OPENAI_API_KEY"] = target.api_key
        env["CODEX_API_KEY"] = target.api_key
    if target.base_url:
        env["OPENAI_BASE_URL"] = target.base_url
        env["OPENAI_API_BASE"] = target.base_url
        env["OPENCLAW_OPENAI_BASE_URL"] = target.base_url
    if target.model_api:
        env["OPENCLAW_MODEL_API"] = target.model_api
    return env


def build_harness_env(target: HarnessTarget) -> dict[str, str]:
    env = dict(os.environ)
    env.update(harness_env_overrides(target))
    if target.codex_home is not None:
        env["CODEX_HOME"] = str(target.codex_home)
    return env


def call_harness_target(
    target: HarnessTarget,
    *,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[str] | None = None,
    max_images: int = 0,
    max_image_bytes: int = 0,
) -> tuple[str, dict[str, int]]:
    """Call a terminal harness and return ``(raw_text, token_usage)``."""
    prompt = _compose_prompt(system_prompt=system_prompt, user_prompt=user_prompt)
    return call_harness_prompt(
        target,
        prompt=prompt,
        image_paths=image_paths,
        max_images=max_images,
        max_image_bytes=max_image_bytes,
    )


def call_harness_prompt(
    target: HarnessTarget,
    *,
    prompt: str,
    image_paths: list[str] | None = None,
    max_images: int = 0,
    max_image_bytes: int = 0,
    use_workdir: bool = False,
) -> tuple[str, dict[str, int]]:
    """Call a terminal harness with an already-composed prompt.

    ``use_workdir`` is mainly for native-memory harness runs: OpenClaw normally
    receives a fresh isolated workspace, but native-memory mode needs the same
    workspace to survive across ingest and QA calls.
    """
    if target.kind == "codex_exec":
        return _call_codex_exec_prompt(
            target,
            prompt=prompt,
            image_paths=image_paths,
            max_images=max_images,
            max_image_bytes=max_image_bytes,
        )
    return _call_openclaw_general_prompt(
        target,
        prompt=prompt,
        image_paths=image_paths,
        max_images=max_images,
        max_image_bytes=max_image_bytes,
        use_workdir=use_workdir,
    )


def _call_openclaw_general_prompt(
    target: HarnessTarget,
    *,
    prompt: str,
    image_paths: list[str] | None,
    max_images: int,
    max_image_bytes: int,
    use_workdir: bool,
) -> tuple[str, dict[str, int]]:
    runner = _resolve_command(target.runner)
    if not target.model:
        raise RuntimeError(f"{target.provider}: missing model")

    cmd = [
        sys.executable,
        runner,
        "--model",
        target.model,
        "--timeout",
        str(target.timeout),
        "--prompt",
        prompt,
    ]
    if use_workdir:
        target.workdir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--workspace", str(target.workdir)])
    if target.keep_isolation:
        cmd.append("--keep-isolation")
    if target.show_paths:
        cmd.append("--show-paths")
    if target.mm_mode == "image":
        for path in _limited_files(
            image_paths or [],
            max_images=max_images,
            max_image_bytes=max_image_bytes,
        ):
            cmd.extend(["--file", path])

    completed = subprocess.run(
        cmd,
        cwd=str(Path(runner).resolve().parent),
        env=build_harness_env(target),
        capture_output=True,
        text=True,
        input="",
        timeout=target.timeout + 60,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{target.provider}: harness command failed with exit "
            f"{completed.returncode}: {detail}"
        )
    raw = completed.stdout.strip()
    if not raw:
        raise RuntimeError(f"{target.provider}: harness produced no stdout")
    return raw, empty_token_usage()


def _call_codex_exec_prompt(
    target: HarnessTarget,
    *,
    prompt: str,
    image_paths: list[str] | None,
    max_images: int,
    max_image_bytes: int,
) -> tuple[str, dict[str, int]]:
    runner = _resolve_command(target.runner)
    target.workdir.mkdir(parents=True, exist_ok=True)
    env = build_harness_env(target)

    with tempfile.NamedTemporaryFile(
        prefix="eval-harness-codex-",
        suffix=".txt",
        dir=str(target.workdir),
        delete=False,
    ) as fh:
        output_path = Path(fh.name)

    cmd = [runner]
    if target.approval_policy:
        cmd.extend(["--ask-for-approval", target.approval_policy])
    cmd.append("exec")
    if target.model:
        cmd.extend(["--model", target.model])
    if target.sandbox:
        cmd.extend(["--sandbox", target.sandbox])
    if target.model_context_window is not None:
        cmd.extend(["--config", f"model_context_window={target.model_context_window}"])
    if target.model_auto_compact_token_limit is not None:
        cmd.extend([
            "--config",
            f"model_auto_compact_token_limit={target.model_auto_compact_token_limit}",
        ])
    if target.mm_mode == "image":
        for path in _limited_files(
            image_paths or [],
            max_images=max_images,
            max_image_bytes=max_image_bytes,
        ):
            cmd.extend(["--image", path])
    cmd.extend([
        "--skip-git-repo-check",
        "--output-last-message",
        str(output_path),
    ])
    if target.ephemeral:
        cmd.append("--ephemeral")
    cmd.append(prompt)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(target.workdir),
            env=env,
            capture_output=True,
            text=True,
            input="",
            timeout=target.timeout + 60,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"{target.provider}: codex harness failed with exit "
                f"{completed.returncode}: {detail}"
            )
        try:
            raw = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            raw = ""
        if not raw:
            raw = completed.stdout.strip()
        if not raw:
            raise RuntimeError(f"{target.provider}: codex harness produced no output")
        return raw, empty_token_usage()
    finally:
        try:
            output_path.unlink()
        except OSError:
            pass
