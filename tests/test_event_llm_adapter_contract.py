from __future__ import annotations

import sys
import types

import llm.extraction.event_llm_adapter as adapter_mod


def test_llm_adapter_returns_none_when_disabled():
    out = adapter_mod.llm_extract_event_payload(text="SEBI action on XYZ", llm_enabled=False)
    assert out is None


def test_llm_adapter_returns_none_for_unsupported_provider(monkeypatch):
    monkeypatch.setattr(adapter_mod, "EVENT_INTELLIGENCE_LLM_PROVIDER", "OTHER")
    out = adapter_mod.llm_extract_event_payload(text="SEBI action on XYZ", llm_enabled=True)
    assert out is None


def test_openai_contract_uses_strict_json_schema(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                output_text='{"event_type":"regulatory_action","instrument_scope":"single_stock","expected_direction":"bearish","directional_confidence":0.8,"vol_impact":"expansion","vol_confidence":0.75,"event_strength":0.7,"uncertainty_score":0.85,"gap_risk_score":0.8,"time_horizon":"1_3_sessions","catalyst_quality":"high","risk_flag":true,"summary":"Regulatory probe headline"}'
            )

    class _FakeOpenAI:
        def __init__(self, timeout=None):
            captured["timeout"] = timeout
            self.responses = _FakeResponses()

    fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setattr(adapter_mod, "EVENT_INTELLIGENCE_LLM_PROVIDER", "OPENAI")

    out = adapter_mod.llm_extract_event_payload(text="SEBI action on XYZ", llm_enabled=True)

    assert isinstance(out, dict)
    assert out is not None
    assert out["event_type"] == "regulatory_action"
    assert out["expected_direction"] == "bearish"

    assert captured["model"] == adapter_mod.EVENT_INTELLIGENCE_LLM_MODEL
    assert captured["temperature"] == adapter_mod.EVENT_INTELLIGENCE_LLM_TEMPERATURE

    response_format = captured["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["additionalProperties"] is False
    required = set(json_schema["schema"]["required"])
    assert "event_type" in required
    assert "instrument_scope" in required
    assert "expected_direction" in required
    assert "summary" in required


def test_openai_contract_accepts_nested_output_blocks(monkeypatch):
    class _FakeResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                output_text="",
                output=[
                    types.SimpleNamespace(
                        content=[
                            types.SimpleNamespace(
                                text='{"event_type":"guidance_revision","instrument_scope":"single_stock","expected_direction":"bullish","directional_confidence":0.7,"vol_impact":"neutral","vol_confidence":0.6,"event_strength":0.65,"uncertainty_score":0.4,"gap_risk_score":0.35,"time_horizon":"1_3_sessions","catalyst_quality":"medium","risk_flag":false,"summary":"Guidance raised"}'
                            )
                        ]
                    )
                ],
            )

    class _FakeOpenAI:
        def __init__(self, timeout=None):
            self.responses = _FakeResponses()

    fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setattr(adapter_mod, "EVENT_INTELLIGENCE_LLM_PROVIDER", "OPENAI")

    out = adapter_mod.llm_extract_event_payload(text="Guidance raised", llm_enabled=True)
    assert isinstance(out, dict)
    assert out is not None
    assert out["event_type"] == "guidance_revision"


def test_openai_contract_returns_none_on_invalid_json(monkeypatch):
    class _FakeResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(output_text="not-json")

    class _FakeOpenAI:
        def __init__(self, timeout=None):
            self.responses = _FakeResponses()

    fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setattr(adapter_mod, "EVENT_INTELLIGENCE_LLM_PROVIDER", "OPENAI")

    out = adapter_mod.llm_extract_event_payload(text="Bad payload", llm_enabled=True)
    assert out is None


def test_openai_client_initialization_propagates_timeout(monkeypatch):
    captured_timeout: list[object] = []

    class _FakeResponses:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                output_text='{"event_type":"rating_action","instrument_scope":"single_stock","expected_direction":"neutral","directional_confidence":0.6,"vol_impact":"neutral","vol_confidence":0.6,"event_strength":0.6,"uncertainty_score":0.3,"gap_risk_score":0.3,"time_horizon":"1_3_sessions","catalyst_quality":"medium","risk_flag":false,"summary":"Rating action update"}'
            )

    class _FakeOpenAI:
        def __init__(self, timeout=None):
            captured_timeout.append(timeout)
            self.responses = _FakeResponses()

    fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setattr(adapter_mod, "EVENT_INTELLIGENCE_LLM_PROVIDER", "OPENAI")

    out = adapter_mod.llm_extract_event_payload(text="Rating update", llm_enabled=True)

    assert isinstance(out, dict)
    assert out is not None
    assert len(captured_timeout) == 1
    assert captured_timeout[0] == adapter_mod.EVENT_INTELLIGENCE_LLM_TIMEOUT_SECONDS