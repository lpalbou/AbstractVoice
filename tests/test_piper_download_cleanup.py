from __future__ import annotations

from abstractvoice.adapters.tts_piper import PiperTTSAdapter


def test_piper_download_failure_preserves_existing_cached_companion(tmp_path, monkeypatch) -> None:
    adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)
    adapter._piper_available = True
    adapter._allow_downloads = True
    adapter._model_dir = tmp_path

    model_path, config_path = adapter._get_model_path("en")
    model_path.write_bytes(b"already valid")

    import requests

    def fail_get(*_args, **_kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(requests, "get", fail_get)

    assert adapter._download_model("en") is False
    assert model_path.read_bytes() == b"already valid"
    assert not config_path.exists()
