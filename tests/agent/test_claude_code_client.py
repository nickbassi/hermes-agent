from __future__ import annotations

import json
from pathlib import Path


def _stream_json_lines(*records: dict) -> str:
    return "\n".join(json.dumps(record) for record in records) + "\n"


class _FakeProcess:
    def __init__(self, stdout_text: str, stderr_text: str = "") -> None:
        self._stdout_text = stdout_text
        self._stderr_text = stderr_text
        self.returncode = 0
        self.pid = 4321
        self.killed = False
        self.terminated = False
        self.last_input = None
        self.last_timeout = None

    def communicate(self, input=None, timeout=None):
        self.last_input = input
        self.last_timeout = timeout
        return self._stdout_text, self._stderr_text

    def kill(self):
        self.killed = True

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.returncode


class TestClaudeCodeClient:
    def test_chat_completion_parses_stream_json_and_tracks_session(self, monkeypatch):
        from agent.claude_code_client import ClaudeCodeClient

        calls: list[dict] = []
        stdout_text = _stream_json_lines(
            {
                "type": "system",
                "subtype": "init",
                "session_id": "11111111-1111-1111-1111-111111111111",
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "hello from claude"},
                    ],
                },
                "session_id": "11111111-1111-1111-1111-111111111111",
            },
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "hello from claude",
                "session_id": "11111111-1111-1111-1111-111111111111",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                },
            },
        )

        def _fake_popen(cmd, **kwargs):
            proc = _FakeProcess(stdout_text)
            calls.append({"cmd": cmd, "kwargs": kwargs, "proc": proc})
            return proc

        monkeypatch.setattr("agent.claude_code_client.subprocess.Popen", _fake_popen)

        client = ClaudeCodeClient(
            command="/usr/local/bin/claude",
            claude_cwd="/tmp",
        )
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Say hello"}],
        )

        assert response.choices[0].message.content == "hello from claude"
        assert response.choices[0].finish_reason == "stop"
        assert response.model == "claude-sonnet-4-6"
        assert client._claude_session_id == "11111111-1111-1111-1111-111111111111"
        assert calls[0]["cmd"][0] == "/usr/local/bin/claude"
        assert "--append-system-prompt" in calls[0]["cmd"]
        assert calls[0]["kwargs"]["cwd"] == "/tmp"
        assert calls[0]["kwargs"]["stdin"] is not None
        assert "Conversation transcript" in calls[0]["proc"].last_input

    def test_second_call_reuses_prior_session_with_resume(self, monkeypatch):
        from agent.claude_code_client import ClaudeCodeClient

        outputs = [
            _stream_json_lines(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "22222222-2222-2222-2222-222222222222",
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "first",
                    "session_id": "22222222-2222-2222-2222-222222222222",
                },
            ),
            _stream_json_lines(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "second",
                    "session_id": "22222222-2222-2222-2222-222222222222",
                },
            ),
        ]
        calls: list[dict] = []

        def _fake_popen(cmd, **kwargs):
            proc = _FakeProcess(outputs[len(calls)])
            calls.append({"cmd": cmd, "kwargs": kwargs, "proc": proc})
            return proc

        monkeypatch.setattr("agent.claude_code_client.subprocess.Popen", _fake_popen)

        client = ClaudeCodeClient(command="claude", claude_cwd=str(Path("/tmp").resolve()))

        first_messages = [{"role": "user", "content": "First prompt"}]
        second_messages = [
            {"role": "user", "content": "First prompt"},
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "Second prompt"},
        ]

        client.chat.completions.create(model="claude-sonnet-4-6", messages=first_messages)
        response = client.chat.completions.create(model="claude-sonnet-4-6", messages=second_messages)

        assert response.choices[0].message.content == "second"
        assert "--resume" in calls[1]["cmd"]
        assert "22222222-2222-2222-2222-222222222222" in calls[1]["cmd"]
        assert "--append-system-prompt" not in calls[1]["cmd"]
        stdin_text = calls[1]["proc"].last_input
        assert "Second prompt" in stdin_text
        assert "First prompt" not in stdin_text
