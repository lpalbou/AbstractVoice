"""The remote discovery memo AGES OUT instead of freezing (operator ruling 2026-08-03).

The adapter's model/voice discovery used to be memoised by two once-per-process
booleans, set BEFORE the fetch — so the first answer (including an EMPTY one
from a failed probe) was served for the life of the process, and the only
refresh mechanism in the whole system was a gateway restart. Loading a new
model into LM Studio, or the machine regaining internet access, changed
nothing on screen.

These tests lock the replacement semantics:

  * a fresh answer is served without re-probing (the probe stays saved),
  * an answer older than the TTL triggers exactly one re-probe,
  * the TTL is FIVE MINUTES, FLOORED — env values below 300, non-numeric,
    or non-finite (`inf` would silently re-create the frozen-for-life bug)
    all clamp back,
  * `refresh_profiles()` remains the immediate override,
  * and the operator's real scenario repairs itself: a catalog emptied by a
    failed first probe recovers one TTL after the endpoint comes back,
    with no restart.

Everything here was first verified by hand against the live gateway
(adversary round, 2026-08-04); this file is what keeps it verified.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import requests

from abstractvoice.adapters.tts_openai_compatible import (
    OpenAICompatibleTTSAdapter,
    _remote_discovery_ttl_s,
)


class _Response:
    def __init__(self, payload: dict):
        self.content = json.dumps(payload).encode("utf-8")
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.text = self.content.decode("utf-8")

    def json(self):
        return json.loads(self.content.decode("utf-8"))


class _CountingSession:
    """Serves every request from a script; counts what the adapter really asked."""

    def __init__(self):
        self.calls = 0
        self.refuse = False  # True = behave like a host that is down

    def request(self, method: str, url: str, **kwargs: Any):
        self.calls += 1
        if self.refuse:
            raise requests.ConnectionError(f"refused: {url}")
        if "/models" in url:
            return _Response({"data": [{"id": "tts-alpha"}, {"id": "tts-beta"}]})
        return _Response({"voices": [{"id": "v1", "name": "Voice One"}]})


def _adapter(session: _CountingSession) -> OpenAICompatibleTTSAdapter:
    return OpenAICompatibleTTSAdapter(
        provider="openai-compatible",
        base_url="http://127.0.0.1:9/v1",  # never dialled: the session is fake
        api_key="k",
        session=session,
    )


def _expire(adapter: OpenAICompatibleTTSAdapter, by_s: float = 301.0) -> None:
    for attr in ("_remote_tts_models_loaded_at", "_remote_profiles_loaded_at"):
        stamp = getattr(adapter, attr)
        if stamp is not None:
            setattr(adapter, attr, stamp - by_s)


def test_ttl_lifecycle_fetch_save_expire_refetch():
    session = _CountingSession()
    adapter = _adapter(session)

    models = adapter._get_tts_models()
    assert "tts-alpha" in models
    cold = session.calls
    assert cold > 0, "first listing must actually probe"

    # Fresh: the saved answer is served, the provider is left alone.
    assert adapter._get_tts_models() == models
    assert session.calls == cold

    # Just short of the TTL is still fresh — the boundary is the contract.
    _expire(adapter, by_s=_remote_discovery_ttl_s() - 1.0)
    adapter._get_tts_models()
    assert session.calls == cold

    # Past the TTL: exactly one re-probe, then saved again.
    _expire(adapter, by_s=2.0)  # tips the stamp over the line
    adapter._get_tts_models()
    refetched = session.calls
    assert refetched > cold, "an expired answer must be re-fetched"
    adapter._get_tts_models()
    assert session.calls == refetched


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 300.0),  # unset -> the ruling's default
        ("10", 300.0),  # below the floor -> clamped, never less
        ("0", 300.0),
        ("-5", 300.0),
        ("abc", 300.0),  # unparseable -> default
        ("inf", 300.0),  # non-finite would re-freeze for process life
        ("nan", 300.0),
        ("900", 900.0),  # above the floor -> honoured
    ],
)
def test_ttl_env_is_floored_and_finite(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("ABSTRACTVOICE_REMOTE_DISCOVERY_TTL_S", raising=False)
    else:
        monkeypatch.setenv("ABSTRACTVOICE_REMOTE_DISCOVERY_TTL_S", raw)
    assert _remote_discovery_ttl_s() == expected


def test_refresh_profiles_is_the_immediate_override():
    session = _CountingSession()
    adapter = _adapter(session)
    adapter._get_tts_models()
    adapter._get_remote_profiles()
    before = session.calls

    assert adapter.refresh_profiles() is True
    assert adapter._remote_tts_models_loaded_at is None
    assert adapter._remote_profiles_loaded_at is None

    adapter._get_tts_models()
    assert session.calls > before, "refresh_profiles must force a real re-probe"


def test_failed_first_probe_repairs_after_ttl_without_restart():
    """The operator's actual scenario: internet access comes back.

    Under the old booleans, step 3 stayed empty until a gateway restart —
    the stamp was set before the fetch, so the failure froze for process life.
    """
    session = _CountingSession()
    session.refuse = True
    adapter = _adapter(session)

    # 1. Host down: the probe fails, an EMPTY answer lands and is saved.
    assert adapter._get_tts_models() == []
    failed = session.calls
    assert failed > 0

    # 2. Host comes back, but the (empty) answer is still fresh: served as-is.
    #    A dead host is re-asked once per TTL window, not once per keystroke.
    session.refuse = False
    assert adapter._get_tts_models() == []
    assert session.calls == failed

    # 3. One TTL later: the provider is re-asked and the catalog RECOVERS.
    _expire(adapter)
    assert "tts-alpha" in adapter._get_tts_models()
    assert session.calls > failed
