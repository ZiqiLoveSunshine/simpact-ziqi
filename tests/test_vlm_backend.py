"""The pluggable VLM backend seam: Gemini (default) vs any OpenAI-compatible server.

``openai_generate`` is the local-model path (llama.cpp ``llama-server``, vLLM,
Ollama). These tests stand up a real HTTP server on localhost and assert the wire
format simpact sends, so the contract is pinned without needing a model or a key.
"""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from simpact.generator.vlm import default_generate, openai_generate

PIL = pytest.importorskip("PIL.Image")


class _Handler(BaseHTTPRequestHandler):
    """Records the last request body on the server object; replies with canned JSON."""

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.last_request = json.loads(body)
        self.server.last_path = self.path
        self.server.last_auth = self.headers.get("Authorization")
        payload = json.dumps({
            "choices": [{"message": {"content": self.server.reply}}]
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):  # keep pytest output clean
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.reply = '{"ok": true}'
    srv.last_request = srv.last_path = srv.last_auth = None
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    srv.base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield srv
    srv.shutdown()


def _img(color=(255, 0, 0)):
    from PIL import Image
    return Image.new("RGB", (8, 8), color)


def test_text_and_images_flatten_to_openai_parts(server, monkeypatch):
    monkeypatch.setenv("SIMPACT_VLM_MODEL", "test-model")
    out = openai_generate(["prompt text", _img(), "trailing text"],
                          base_url=server.base_url)
    assert out == '{"ok": true}'
    req = server.last_request
    assert server.last_path == "/v1/chat/completions"
    assert req["model"] == "test-model"
    assert req["stream"] is False
    parts = req["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text"]
    assert parts[0]["text"] == "prompt text"
    assert parts[2]["text"] == "trailing text"


def test_image_is_a_decodable_base64_png_data_uri(server):
    openai_generate([_img()], base_url=server.base_url)
    url = server.last_request["messages"][0]["content"][0]["image_url"]["url"]
    prefix = "data:image/png;base64,"
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix):])[:8] == b"\x89PNG\r\n\x1a\n"


def test_empty_strings_are_dropped(server):
    # the callers interleave separator strings; blank ones must not become parts
    openai_generate(["real", "", "   ", "also real"], base_url=server.base_url)
    parts = server.last_request["messages"][0]["content"]
    assert [p["text"] for p in parts] == ["real", "also real"]


def test_json_mode_on_by_default_and_disablable(server, monkeypatch):
    openai_generate(["x"], base_url=server.base_url)
    assert server.last_request["response_format"] == {"type": "json_object"}

    monkeypatch.setenv("SIMPACT_VLM_JSON_MODE", "0")
    openai_generate(["x"], base_url=server.base_url)
    assert "response_format" not in server.last_request


def test_sampling_and_auth_come_from_env(server, monkeypatch):
    monkeypatch.setenv("SIMPACT_VLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("SIMPACT_VLM_MAX_TOKENS", "512")
    monkeypatch.setenv("SIMPACT_VLM_API_KEY", "sk-local")
    openai_generate(["x"], base_url=server.base_url)
    assert server.last_request["temperature"] == 0.25
    assert server.last_request["max_tokens"] == 512
    assert server.last_auth == "Bearer sk-local"


def test_base_url_from_env_when_not_passed(server, monkeypatch):
    monkeypatch.setenv("SIMPACT_VLM_BASE_URL", server.base_url)
    openai_generate(["x"])
    assert server.last_path == "/v1/chat/completions"


def test_default_generate_dispatches_to_openai(server, monkeypatch):
    monkeypatch.setenv("SIMPACT_VLM_BACKEND", "openai")
    monkeypatch.setenv("SIMPACT_VLM_BASE_URL", server.base_url)
    server.reply = '{"success": true}'
    assert default_generate(["x"]) == '{"success": true}'


def test_default_generate_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("SIMPACT_VLM_BACKEND", "nope")
    with pytest.raises(ValueError, match="unknown SIMPACT_VLM_BACKEND"):
        default_generate(["x"])


def test_unreachable_server_raises_actionable_error(monkeypatch):
    # port 1 is never a VLM server; the message must name the URL and the fix
    with pytest.raises(RuntimeError, match="local VLM request to"):
        openai_generate(["x"], base_url="http://127.0.0.1:1/v1", timeout_s=5)


def test_backend_reaches_the_real_call_sites(server, monkeypatch, tmp_path):
    """The proposer's default generate_fn must honour SIMPACT_VLM_BACKEND."""
    from simpact.generator.propose import VLMProposer

    monkeypatch.setenv("SIMPACT_VLM_BACKEND", "openai")
    monkeypatch.setenv("SIMPACT_VLM_BASE_URL", server.base_url)
    server.reply = json.dumps({"action_proposals": [
        {"description": "nudge", "action_sequence": [
            {"type": "PUSH", "delta_x": 0.01, "delta_y": 0.0, "reasoning": "r"}]}]})
    photo = tmp_path / "camera1_rgb.png"
    _img().save(photo)
    ps = VLMProposer(prompt_template="primitive").propose(
        "push it", photo, context="ctx", allowed_types={"PUSH"})
    assert len(ps.action_proposals) == 1
    assert ps.action_proposals[0].action_sequence[0].TYPE == "PUSH"


def test_schema_restricts_primitive_types():
    from simpact.generator.vlm import proposalset_schema

    schema = proposalset_schema({"PUSH", "LIFT"})
    seq = schema["properties"]["action_proposals"]["items"]["properties"]["action_sequence"]
    assert [v["properties"]["type"]["const"] for v in seq["items"]["anyOf"]] == ["LIFT", "PUSH"]
    push = next(v for v in seq["items"]["anyOf"] if v["properties"]["type"]["const"] == "PUSH")
    assert set(push["required"]) == {"type", "delta_x", "delta_y", "reasoning"}


def test_schema_bounds_arrays_so_output_cannot_run_past_max_tokens():
    from simpact.generator.vlm import proposalset_schema

    ap = proposalset_schema({"PUSH"}, max_proposals=3, max_actions=10)["properties"]["action_proposals"]
    assert (ap["minItems"], ap["maxItems"]) == (1, 3)
    seq = ap["items"]["properties"]["action_sequence"]
    assert (seq["minItems"], seq["maxItems"]) == (1, 10)


def test_single_allowed_type_needs_no_anyof():
    from simpact.generator.vlm import proposalset_schema

    seq = (proposalset_schema({"RELEASE"})["properties"]["action_proposals"]
           ["items"]["properties"]["action_sequence"])
    assert seq["items"]["properties"]["type"]["const"] == "RELEASE"


def test_allowed_types_reach_the_server_as_a_json_schema(server, monkeypatch):
    """The whitelist must arrive as a decode-time grammar, not just post-hoc validation."""
    from simpact.generator.propose import VLMProposer

    monkeypatch.setenv("SIMPACT_VLM_BACKEND", "openai")
    monkeypatch.setenv("SIMPACT_VLM_BASE_URL", server.base_url)
    server.reply = json.dumps({"action_proposals": [
        {"description": "d", "action_sequence": [
            {"type": "PUSH", "delta_x": 0.0, "delta_y": 0.1, "reasoning": "r"}]}]})
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        _img().save(f.name)
        VLMProposer(prompt_template="primitive").propose(
            "push", f.name, context="c", allowed_types={"PUSH", "LIFT"})
    rf = server.last_request["response_format"]
    assert rf["type"] == "json_schema"
    variants = (rf["json_schema"]["schema"]["properties"]["action_proposals"]["items"]
                ["properties"]["action_sequence"]["items"]["anyOf"])
    assert {v["properties"]["type"]["const"] for v in variants} == {"PUSH", "LIFT"}


def test_no_allowed_types_falls_back_to_plain_json_mode(server, monkeypatch):
    # the regress step passes allowed_types=None; it must not get a propose schema
    from simpact.generator.regress import RegressOptimizer

    monkeypatch.setenv("SIMPACT_VLM_BACKEND", "openai")
    monkeypatch.setenv("SIMPACT_VLM_BASE_URL", server.base_url)
    fn = RegressOptimizer(prompt_template="push").generate_fn
    from simpact.generator.vlm import generate_proposalset
    server.reply = json.dumps({"action_proposals": [
        {"description": "d", "action_sequence": [
            {"type": "gripper_control", "width": 0.04, "reasoning": "r"}]}]})
    generate_proposalset(fn, ["text"], allowed_types=None)
    assert server.last_request["response_format"] == {"type": "json_object"}
