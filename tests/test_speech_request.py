from __future__ import annotations

from abstractvoice.speech_request import (
    SpeechCapabilities,
    SpeechCapability,
    build_speech_request,
)
from abstractvoice.vm.tts_mixin import TtsMixin


def test_build_speech_request_normalizes_optional_fields() -> None:
    request = build_speech_request(
        "Hello",
        language=" en ",
        provider=" openai ",
        model=" gpt-4o-mini-tts ",
        profile=" narrator ",
        voice=" clone-123 ",
        instructions=" warm and calm ",
        speed="1.25",
        pace="0.9",
        target_duration_s="12.5",
        quality_preset=" high ",
        scene_context=" rainy street ",
        actions=[" take a breath ", "", "smiles"],
        ambient_audio=" soft rain ",
        background_sfx=True,
        output_format=" wav ",
        output_channels="2",
        sanitize_syntax=False,
        metadata={"x": 1},
    )

    assert request.text == "Hello"
    assert request.language == "en"
    assert request.provider == "openai"
    assert request.model == "gpt-4o-mini-tts"
    assert request.profile == "narrator"
    assert request.voice == "clone-123"
    assert request.instructions == "warm and calm"
    assert request.speed == 1.25
    assert request.pace == 0.9
    assert request.target_duration_s == 12.5
    assert request.quality_preset == "high"
    assert request.scene_context == "rainy street"
    assert request.actions == ("take a breath", "smiles")
    assert request.ambient_audio == "soft rain"
    assert request.background_sfx is True
    assert request.output_format == "wav"
    assert request.output_channels == 2
    assert request.sanitize_syntax is False
    assert request.metadata == {"x": 1}


def test_speech_capabilities_to_dict() -> None:
    caps = SpeechCapabilities(
        fields={
            "pace": SpeechCapability(name="pace", support="native"),
            "scene_context": SpeechCapability(
                name="scene_context",
                support="unsupported",
                reason="engine has no scene-aware prompting",
            ),
        }
    )

    assert caps.support_for("pace") is not None
    assert caps.to_dict() == {
        "pace": {"support": "native", "reason": None},
        "scene_context": {
            "support": "unsupported",
            "reason": "engine has no scene-aware prompting",
        },
    }


def test_tts_capabilities_mark_audiodit_speed_as_unsupported() -> None:
    class _Adapter:
        engine_id = "audiodit"

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()

    caps = _VM().get_tts_capabilities().to_dict()

    assert caps["speed"]["support"] == "unsupported"
    assert "degrades output quality" in str(caps["speed"]["reason"] or "")


def test_tts_capabilities_mark_openai_compatible_speed_as_conditional() -> None:
    class _Adapter:
        engine_id = "openai-compatible"

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()
            self.tts_model = "gpt-4o-mini-tts"
            self.stt_model = None

    caps = _VM().get_tts_capabilities().to_dict()

    assert caps["speed"]["support"] == "conditional"
    assert "provider support varies" in str(caps["speed"]["reason"] or "")
    assert caps["instructions"]["support"] == "conditional"


def test_speak_to_bytes_forwards_speed_to_voice_aware_adapter() -> None:
    seen: dict[str, object] = {}

    class _Adapter:
        engine_id = "openai-compatible"

        def is_available(self) -> bool:
            return True

        def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
            raise AssertionError("voice-aware path should be used for non-default speed")

        def synthesize_to_bytes_with_voice(
            self,
            text: str,
            *,
            format: str = "wav",
            voice: str | None = None,
            speed: float | None = None,
            instructions: str | None = None,
        ) -> bytes:
            seen.update(
                {
                    "text": text,
                    "format": format,
                    "voice": voice,
                    "speed": speed,
                    "instructions": instructions,
                }
            )
            return b"RIFF....WAVE"

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()
            self.speed = 1.25
            self.language = "en"
            self._last_tts_metrics = None

        def get_tts_quality_preset(self):
            return None

        def _set_last_tts_metrics(self, metrics):
            self._last_tts_metrics = metrics

    out = _VM().speak_to_bytes("Hello", format="wav")

    assert out == b"RIFF....WAVE"
    assert seen == {
        "text": "Hello",
        "format": "wav",
        "voice": None,
        "speed": 1.25,
        "instructions": None,
    }


def test_speak_to_bytes_does_not_forward_unsupported_controls() -> None:
    seen: dict[str, object] = {}

    class _Adapter:
        engine_id = "audiodit"

        def is_available(self) -> bool:
            return True

        def synthesize_to_bytes(self, text: str, format: str = "wav") -> bytes:
            seen.update({"text": text, "format": format, "path": "plain"})
            return b"RIFF....WAVE"

        def synthesize_to_bytes_with_voice(
            self,
            text: str,
            *,
            format: str = "wav",
            voice: str | None = None,
            speed: float | None = None,
            instructions: str | None = None,
        ) -> bytes:
            seen.update(
                {
                    "text": text,
                    "format": format,
                    "voice": voice,
                    "speed": speed,
                    "instructions": instructions,
                    "path": "voice-aware",
                }
            )
            return b"RIFF....WAVE"

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()
            self.speed = 1.25
            self.language = "en"
            self._last_tts_metrics = None

        def get_tts_quality_preset(self):
            return None

        def _set_last_tts_metrics(self, metrics):
            self._last_tts_metrics = metrics

    out = _VM().speak_to_bytes("Hello", format="wav", instructions="dramatic")

    assert out == b"RIFF....WAVE"
    assert seen == {
        "text": "Hello",
        "format": "wav",
        "path": "plain",
    }


def test_set_tts_quality_preset_respects_central_compatibility_when_unsupported() -> None:
    seen: list[str] = []

    class _Adapter:
        engine_id = "piper"

        def set_quality_preset(self, preset: str) -> bool:
            seen.append(preset)
            return True

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()

    assert _VM().set_tts_quality_preset("high") is False
    assert seen == []


def test_set_tts_quality_preset_uses_central_compatibility_when_supported() -> None:
    seen: list[str] = []

    class _Adapter:
        engine_id = "supertonic"
        model_id = "supertonic-3"

        def set_quality_preset(self, preset: str) -> bool:
            seen.append(preset)
            return True

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.tts_adapter = _Adapter()
            self.tts_model = "supertonic-3"

    assert _VM().set_tts_quality_preset("balanced") is True
    assert seen == ["standard"]


def test_clone_speak_to_audio_chunks_suppresses_unsupported_speed() -> None:
    seen: dict[str, object] = {}

    class _Cloner:
        def get_cloned_voice(self, voice_id: str) -> dict[str, str]:
            return {"engine": "audiodit", "voice_id": voice_id}

        def speak_to_audio_chunks(
            self,
            text: str,
            *,
            voice_id: str,
            speed: float | None = None,
            max_chars: int = 120,
            language: str | None = None,
        ):
            seen.update(
                {
                    "text": text,
                    "voice_id": voice_id,
                    "speed": speed,
                    "max_chars": max_chars,
                    "language": language,
                }
            )
            yield [0.0, 0.0], 24000

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.speed = 1.4
            self.language = "en"
            self._last_tts_metrics = None

        def _get_voice_cloner(self):
            return _Cloner()

        def _set_last_tts_metrics(self, metrics):
            self._last_tts_metrics = metrics

    chunks = list(_VM().speak_to_audio_chunks("Hello", voice="clone-1"))

    assert len(chunks) == 1
    assert seen == {
        "text": "Hello",
        "voice_id": "clone-1",
        "speed": 1.0,
        "max_chars": 120,
        "language": "en",
    }


def test_clone_speak_buffered_preserves_native_speed() -> None:
    import io
    import threading
    import wave

    seen: dict[str, object] = {}
    done = threading.Event()

    class _Cloner:
        def get_cloned_voice(self, voice_id: str) -> dict[str, str]:
            return {"engine": "f5_tts", "voice_id": voice_id}

        def speak_to_bytes(
            self,
            text: str,
            *,
            voice_id: str,
            format: str = "wav",
            speed: float | None = None,
            language: str | None = None,
        ) -> bytes:
            seen.update(
                {
                    "text": text,
                    "voice_id": voice_id,
                    "format": format,
                    "speed": speed,
                    "language": language,
                }
            )
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x00\x00" * 16)
            done.set()
            return buf.getvalue()

    class _Engine:
        def begin_playback(self, callback=None, sample_rate=None):
            _ = (callback, sample_rate)

        def enqueue_audio(self, audio, sample_rate=None):
            _ = (audio, sample_rate)

        def stop(self, close_stream=False):
            _ = close_stream
            return True

        def is_active(self):
            return False

    class _VM(TtsMixin):
        def __init__(self) -> None:
            self.speed = 1.3
            self.language = "en"
            self.tts_engine = _Engine()
            self.cloned_tts_streaming = False
            self._last_tts_metrics = None

        def _get_voice_cloner(self):
            return _Cloner()

        def _set_last_tts_metrics(self, metrics):
            self._last_tts_metrics = metrics

    vm = _VM()
    assert vm.speak("Hello", voice="clone-2") is True
    assert done.wait(timeout=1.0) is True
    assert seen == {
        "text": "Hello",
        "voice_id": "clone-2",
        "format": "wav",
        "speed": 1.3,
        "language": "en",
    }
