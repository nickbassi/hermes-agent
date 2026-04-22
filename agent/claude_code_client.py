"""OpenAI-compatible shim that forwards Hermes requests to `claude -p`.

This adapter mirrors the pattern OpenClaw uses for Claude Code CLI: spawn a
short-lived `claude -p` process with `--output-format stream-json`, parse the
JSONL stream, and convert the result back into the small chat.completions-like
shape Hermes expects.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

CLAUDE_CLI_MARKER_BASE_URL = "claude-cli://local"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}", re.DOTALL)

_CLAUDE_CLI_CLEAR_ENV = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY_OLD",
    "ANTHROPIC_API_TOKEN",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "ANTHROPIC_OAUTH_TOKEN",
    "ANTHROPIC_UNIX_SOCKET",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
    "CLAUDE_CODE_OAUTH_SCOPES",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
    "CLAUDE_CODE_PLUGIN_CACHE_DIR",
    "CLAUDE_CODE_PLUGIN_SEED_DIR",
    "CLAUDE_CODE_REMOTE",
    "CLAUDE_CODE_USE_COWORK_PLUGINS",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_VERTEX",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_LOGS_HEADERS",
    "OTEL_EXPORTER_OTLP_LOGS_PROTOCOL",
    "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    "OTEL_EXPORTER_OTLP_METRICS_HEADERS",
    "OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_LOGS_EXPORTER",
    "OTEL_METRICS_EXPORTER",
    "OTEL_SDK_DISABLED",
    "OTEL_TRACES_EXPORTER",
]

_CLAUDE_CLI_SESSION_ID_FIELDS = (
    "session_id",
    "sessionId",
    "conversation_id",
    "conversationId",
)

_CLAUDE_CLI_MODEL_ALIASES = {
    "opus": "opus",
    "sonnet": "sonnet",
    "haiku": "haiku",
    "opus-4.7": "claude-opus-4-7",
    "opus-4.6": "claude-opus-4-6",
    "opus-4.5": "claude-opus-4-5",
    "opus-4": "claude-opus-4",
    "sonnet-4.6": "claude-sonnet-4-6",
    "sonnet-4.5": "claude-sonnet-4-5",
    "sonnet-4.1": "claude-sonnet-4-1",
    "sonnet-4.0": "claude-sonnet-4-0",
    "haiku-3.5": "claude-haiku-3-5",
    "claude-opus-4.7": "claude-opus-4-7",
    "claude-opus-4.6": "claude-opus-4-6",
    "claude-opus-4.5": "claude-opus-4-5",
    "claude-opus-4": "claude-opus-4",
    "claude-opus-4-7": "claude-opus-4-7",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-opus-4-5": "claude-opus-4-5",
    "claude-opus-4": "claude-opus-4",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-sonnet-4.5": "claude-sonnet-4-5",
    "claude-sonnet-4.1": "claude-sonnet-4-1",
    "claude-sonnet-4.0": "claude-sonnet-4-0",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-sonnet-4-1": "claude-sonnet-4-1",
    "claude-sonnet-4-0": "claude-sonnet-4-0",
    "claude-haiku-3.5": "claude-haiku-3-5",
    "claude-haiku-3-5": "claude-haiku-3-5",
}


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_CLAUDE_CLI_COMMAND", "").strip()
        or os.getenv("CLAUDE_CODE_CLI_PATH", "").strip()
        or "claude"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_CLAUDE_CLI_ARGS", "").strip()
    if raw:
        return _normalize_args(shlex.split(raw))
    return _normalize_args([
        "-p",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--setting-sources",
        "user",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "",
    ])


def _normalize_model_hint(model: str | None) -> str:
    raw = str(model or "").strip().lower()
    if not raw:
        return ""
    if "/" in raw:
        raw = raw.split("/", 1)[1].strip()
    return _CLAUDE_CLI_MODEL_ALIASES.get(raw, raw.replace(".", "-") if raw.startswith("claude-") else raw)


def _normalize_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    has_print = False
    has_output_format = False
    has_partials = False
    has_verbose = False
    has_setting_sources = False
    has_permission_mode = False
    has_tools = False
    while i < len(args):
        arg = args[i]
        if arg == "--dangerously-skip-permissions":
            i += 1
            continue
        if arg == "-p":
            has_print = True
            normalized.append(arg)
            i += 1
            continue
        if arg == "--output-format":
            has_output_format = True
            normalized.extend([arg, "stream-json"])
            i += 2
            continue
        if arg.startswith("--output-format="):
            has_output_format = True
            normalized.extend(["--output-format", "stream-json"])
            i += 1
            continue
        if arg == "--include-partial-messages":
            has_partials = True
            normalized.append(arg)
            i += 1
            continue
        if arg == "--verbose":
            has_verbose = True
            normalized.append(arg)
            i += 1
            continue
        if arg == "--setting-sources":
            has_setting_sources = True
            normalized.extend([arg, "user"])
            i += 2
            continue
        if arg.startswith("--setting-sources="):
            has_setting_sources = True
            normalized.extend(["--setting-sources", "user"])
            i += 1
            continue
        if arg == "--permission-mode":
            has_permission_mode = True
            normalized.extend([arg, "bypassPermissions"])
            i += 2
            continue
        if arg.startswith("--permission-mode="):
            has_permission_mode = True
            normalized.extend(["--permission-mode", "bypassPermissions"])
            i += 1
            continue
        if arg == "--tools":
            has_tools = True
            normalized.extend([arg, args[i + 1] if i + 1 < len(args) else ""])
            i += 2
            continue
        if arg.startswith("--tools="):
            has_tools = True
            normalized.append(arg)
            i += 1
            continue
        normalized.append(arg)
        i += 1
    if not has_print:
        normalized.insert(0, "-p")
    if not has_output_format:
        normalized.extend(["--output-format", "stream-json"])
    if not has_partials:
        normalized.append("--include-partial-messages")
    if not has_verbose:
        normalized.append("--verbose")
    if not has_setting_sources:
        normalized.extend(["--setting-sources", "user"])
    if not has_permission_mode:
        normalized.extend(["--permission-mode", "bypassPermissions"])
    if not has_tools:
        normalized.extend(["--tools", ""])
    return normalized


def _normalize_model_hint(model: str | None) -> str:
    raw = str(model or "").strip().lower()
    if not raw:
        return ""
    if "/" in raw:
        raw = raw.split("/", 1)[1].strip()
    if raw.startswith("claude-"):
        raw = raw.replace(".", "-")
    return _CLAUDE_CLI_MODEL_ALIASES.get(raw, raw)


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        inner = content.get("content")
        if isinstance(inner, str) and inner.strip():
            return inner.strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            if item_type in {"input_text", "output_text", "text"}:
                maybe_text = item.get("text") or item.get("content")
                if isinstance(maybe_text, str) and maybe_text.strip():
                    parts.append(maybe_text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _render_message_for_transcript(message: dict[str, Any]) -> str:
    role = str(message.get("role") or "context").strip().lower()
    rendered = _render_message_content(message.get("content"))
    extra: list[str] = []

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        extra.append("Tool calls:\n" + json.dumps(tool_calls, ensure_ascii=False))

    if role == "tool":
        tool_name = str(message.get("name") or message.get("tool_name") or "tool").strip()
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        prefix_bits = [f"tool={tool_name}"]
        if tool_call_id:
            prefix_bits.append(f"tool_call_id={tool_call_id}")
        prefix = "Tool result (" + ", ".join(prefix_bits) + ")"
    else:
        prefix = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }.get(role, role.title() or "Context")

    payload_parts = [part for part in [rendered, *extra] if part]
    if not payload_parts:
        return ""
    return prefix + ":\n" + "\n\n".join(payload_parts)


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are being used as the Claude Code CLI model backend inside Hermes.",
        "Hermes executes tools itself. Do not rely on Claude Code built-in tools.",
        "IMPORTANT: If you need a Hermes tool, emit tool calls using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            tool_specs.append(
                {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript = [
        rendered for rendered in (_render_message_for_transcript(message) for message in messages)
        if rendered
    ]
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"claude_cli_call_{len(extracted)+1}"
        extracted.append(
            SimpleNamespace(
                id=call_id,
                call_id=call_id,
                response_item_id=None,
                type="function",
                function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
            )
        )

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add_tool_call(match.group(1))
        consumed_spans.append((match.start(), match.end()))

    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add_tool_call(match.group(0))
            consumed_spans.append((match.start(), match.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return extracted, cleaned


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _extract_session_id(payload: Any) -> str:
    if isinstance(payload, dict):
        for field in _CLAUDE_CLI_SESSION_ID_FIELDS:
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            found = _extract_session_id(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_session_id(item)
            if found:
                return found
    return ""


def _extract_text_segments(payload: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(payload, str):
        if payload.strip():
            parts.append(payload.strip())
        return parts
    if isinstance(payload, dict):
        block_type = str(payload.get("type") or payload.get("event") or "").strip().lower()
        text = payload.get("text")
        if isinstance(text, str) and text.strip() and block_type in {"text", "text_delta", "input_text", "output_text", ""}:
            parts.append(text.strip())
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            parts.append(result.strip())
        message = payload.get("message")
        if message is not None:
            parts.extend(_extract_text_segments(message))
        delta = payload.get("delta")
        if delta is not None:
            parts.extend(_extract_text_segments(delta))
        content = payload.get("content")
        if content is not None:
            parts.extend(_extract_text_segments(content))
        return parts
    if isinstance(payload, list):
        for item in payload:
            parts.extend(_extract_text_segments(item))
    return parts


def _normalize_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    numeric = [float(value) for value in candidates if isinstance(value, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS


def _message_fingerprint(message: dict[str, Any]) -> str:
    return json.dumps(message, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class _ClaudeChatCompletions:
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeChatNamespace:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _ClaudeChatCompletions(client)


class ClaudeCodeClient:
    """Minimal OpenAI-client-compatible facade for Claude Code CLI."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        claude_cwd: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "claude-cli"
        self.base_url = base_url or CLAUDE_CLI_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._command = command or _resolve_command()
        self._args = list(args or _resolve_args())
        self._claude_cwd = str(Path(claude_cwd or os.getcwd()).resolve())
        self.chat = _ClaudeChatNamespace(self)
        self.is_closed = False
        self._active_process: subprocess.Popen[str] | None = None
        self._active_process_lock = threading.Lock()
        self._claude_session_id: str | None = None
        self._last_message_fingerprint: list[str] = []
        self._system_prompt_sent = False

    def close(self) -> None:
        proc: subprocess.Popen[str] | None
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        messages = list(messages or [])
        message_fingerprint = [_message_fingerprint(message) for message in messages if isinstance(message, dict)]
        use_resume = bool(
            self._claude_session_id
            and self._last_message_fingerprint
            and len(message_fingerprint) >= len(self._last_message_fingerprint)
            and message_fingerprint[: len(self._last_message_fingerprint)] == self._last_message_fingerprint
        )
        prompt_messages = messages[len(self._last_message_fingerprint):] if use_resume else messages
        prompt_text = _format_messages_as_prompt(
            prompt_messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        system_prompt = None if use_resume or self._system_prompt_sent else (
            "You are Claude Code running inside Hermes. Hermes owns the tool loop. "
            "When you need a Hermes tool, emit <tool_call>{...}</tool_call> blocks only."
        )
        parsed = self._run_prompt(
            prompt_text,
            model_hint=model,
            timeout_seconds=_normalize_timeout(timeout),
            resume_session_id=self._claude_session_id if use_resume else None,
            system_prompt=system_prompt,
        )
        if parsed.get("session_id"):
            self._claude_session_id = str(parsed["session_id"])
        self._last_message_fingerprint = message_fingerprint
        if system_prompt is not None:
            self._system_prompt_sent = True

        response_text = str(parsed.get("text") or "").strip()
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage_info = parsed.get("usage") or {}
        prompt_tokens = int(usage_info.get("input_tokens") or usage_info.get("prompt_tokens") or 0)
        completion_tokens = int(usage_info.get("output_tokens") or usage_info.get("completion_tokens") or 0)
        total_tokens = int(usage_info.get("total_tokens") or (prompt_tokens + completion_tokens))
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=int(usage_info.get("cache_read_input_tokens") or 0)),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=str(model or "claude-cli"),
            claude_session_id=self._claude_session_id,
        )

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        for key in _CLAUDE_CLI_CLEAR_ENV:
            env.pop(key, None)
        token = str(self.api_key or "").strip()
        if token and token not in {"claude-cli", "***"}:
            env["ANTHROPIC_API_KEY"] = token
        return env

    def _run_prompt(
        self,
        prompt_text: str,
        *,
        model_hint: str | None,
        timeout_seconds: float,
        resume_session_id: str | None,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        command = [self._command] + _normalize_args(list(self._args))
        normalized_model = _normalize_model_hint(model_hint)
        if normalized_model:
            command.extend(["--model", normalized_model])
        if resume_session_id:
            command.extend(["--resume", resume_session_id])
        if system_prompt:
            command.extend(["--append-system-prompt", system_prompt])

        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._claude_cwd,
                env=self._build_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Claude Code CLI command '{self._command}'. "
                "Install Claude Code CLI or set HERMES_CLAUDE_CLI_COMMAND/CLAUDE_CODE_CLI_PATH."
            ) from exc

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc
        try:
            stdout, stderr = proc.communicate(prompt_text, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise TimeoutError(
                f"Claude Code CLI timed out after {timeout_seconds:.0f}s."
            ) from exc
        finally:
            with self._active_process_lock:
                if self._active_process is proc:
                    self._active_process = None

        parsed = self._parse_stream_json(stdout or "", stderr or "")
        if proc.returncode not in (0, None):
            detail = (parsed.get("text") or "").strip() or (stderr or "").strip()
            raise RuntimeError(detail or f"Claude Code CLI exited with status {proc.returncode}.")
        if not (parsed.get("text") or "").strip() and (stderr or "").strip():
            raise RuntimeError((stderr or "").strip())
        return parsed

    def _parse_stream_json(self, stdout: str, stderr: str) -> dict[str, Any]:
        session_id = ""
        usage: dict[str, Any] = {}
        assistant_chunks: list[str] = []
        partial_chunks: list[str] = []
        result_text = ""
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            record = _safe_json_loads(line)
            if not record:
                continue
            if not session_id:
                session_id = _extract_session_id(record)
            record_type = str(record.get("type") or "").strip().lower()
            if record_type == "assistant":
                assistant_chunks.extend(_extract_text_segments(record.get("message") or record))
                continue
            if record_type == "result":
                if isinstance(record.get("result"), str) and record.get("result", "").strip():
                    result_text = str(record.get("result") or "").strip()
                if isinstance(record.get("usage"), dict):
                    usage.update(record.get("usage") or {})
                continue
            if record_type == "stream_event":
                event = record.get("event") or {}
                if not session_id:
                    session_id = _extract_session_id(event)
                partial_chunks.extend(_extract_text_segments(event))
                continue
            assistant_chunks.extend(_extract_text_segments(record))

        text = "\n".join(chunk for chunk in assistant_chunks if chunk).strip()
        if not text:
            text = result_text.strip()
        if not text:
            text = "\n".join(chunk for chunk in partial_chunks if chunk).strip()
        return {
            "text": text,
            "usage": usage,
            "session_id": session_id,
            "stderr": stderr.strip(),
            "raw_stdout": stdout,
        }
