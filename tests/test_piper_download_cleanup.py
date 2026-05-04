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


def test_piper_synthesize_to_file_defaults_extensionless_path_to_wav(tmp_path) -> None:
    adapter = PiperTTSAdapter.__new__(PiperTTSAdapter)
    adapter.synthesize_to_bytes = lambda _text, format="wav": b"wav-bytes"

    out_path = tmp_path / "speech"

    assert adapter.synthesize_to_file("hello", str(out_path)) == str(out_path)
    assert out_path.read_bytes() == b"wav-bytes"
