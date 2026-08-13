from __future__ import annotations

import json

from research_tree.providers import openrouter
from research_tree.providers.openrouter import OpenRouterClient, resolve_openrouter_key
from research_tree.research import ANSWER_SCHEMA


def test_pi_credential_precedes_legacy_fluff_config(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(openrouter, "_key_from_research_config", lambda: None)
    monkeypatch.setattr(openrouter, "_key_from_pi", lambda model: "active-pi-key")
    monkeypatch.setattr(openrouter, "_key_from_fluff_cutter", lambda: "stale-fluff-key")
    assert resolve_openrouter_key("test/model") == "active-pi-key"


def test_chat_sends_reasoning_web_tools_and_json_schema(monkeypatch):
    captured = {}

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "id": "response-1",
                    "model": "resolved/model",
                    "choices": [{"message": {"content": "{}", "annotations": []}}],
                    "usage": {"cost": 0.01},
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeHTTPResponse()

    monkeypatch.setattr(openrouter.urllib.request, "urlopen", fake_urlopen)
    client = OpenRouterClient(api_key="not-a-real-key", base_url="https://router.test/v1")
    response = client.chat(
        model="requested/model",
        messages=[{"role": "user", "content": "Research this"}],
        web=True,
        reasoning_effort="high",
        response_schema=ANSWER_SCHEMA,
    )
    assert captured["url"] == "https://router.test/v1/chat/completions"
    assert captured["payload"]["reasoning"]["effort"] == "high"
    assert captured["payload"]["tools"][0]["type"] == "openrouter:web_search"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert response.resolved_model == "resolved/model"
