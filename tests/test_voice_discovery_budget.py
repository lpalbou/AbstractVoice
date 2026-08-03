"""Listing voices must not load models, and must not wait on dead hosts."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

from abstractvoice.integrations.abstractcore_plugin import (
    _REMOTE_DISCOVERY_TIMEOUT_S,
    _VoiceCapability,
    _probe_in_parallel,
)


class _Owner:
    config: dict = {}


@pytest.fixture(autouse=True)
def _isolated_discovery(monkeypatch):
    """A clean slate: no ambient provider configuration, no shared manager cache.

    Discovery reads the environment, so an `OPENAI_BASE_URL` in the developer's
    shell would otherwise change which providers get probed.
    """
    import abstractvoice.integrations.abstractcore_plugin as plugin

    for key in list(os.environ):
        if key.startswith("ABSTRACTVOICE_") or key.startswith("OPENAI_"):
            monkeypatch.delenv(key, raising=False)

    saved = dict(plugin._VM_CACHE)
    plugin._VM_CACHE.clear()
    try:
        yield
    finally:
        plugin._VM_CACHE.clear()
        plugin._VM_CACHE.update(saved)


def _run_isolated(code: str, **extra_env: str) -> subprocess.CompletedProcess:
    """Run `code` in a fresh interpreter with no voice provider configured."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ABSTRACTVOICE_", "OPENAI_"))}
    env.update(extra_env)
    # macOS + native deps make vfork()-based launching flaky inside long pytest
    # sessions; the same escape hatch the other import-boundary test uses.
    old_vfork = getattr(subprocess, "_USE_VFORK", None)
    try:
        if hasattr(subprocess, "_USE_VFORK"):
            subprocess._USE_VFORK = False  # type: ignore[attr-defined]
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            check=False,
            text=True,
            capture_output=True,
            env=env,
        )
    finally:
        if old_vfork is not None and hasattr(subprocess, "_USE_VFORK"):
            subprocess._USE_VFORK = old_vfork  # type: ignore[attr-defined]


def test_listing_voice_models_does_not_import_a_machine_learning_framework():
    """The regression this guards: discovery used to build one adapter per local
    engine, and the AudioDiT adapter imports torch, transformers and diffusers --
    seconds of cold start to report a catalog the model cache already knew.

    Import-light engine modules may still be touched: an engine owns the layout of
    its own cache directory, and reading it is what replaced loading the engine."""

    result = _run_isolated(
        """
        import sys
        from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

        class _Owner:
            config = {}

        cap = _VoiceCapability(_Owner())
        cap.available_providers()
        cap.list_tts_models()
        cap.voice_catalog(providers_only=True)

        heavy = sorted(
            module
            for module in ("torch", "transformers", "diffusers", "onnxruntime", "piper")
            if module in sys.modules
        )
        assert not heavy, f"discovery imported {heavy}"
        # These two packages import their modeling code, and with it torch.
        assert "abstractvoice.audiodit" not in sys.modules
        assert "abstractvoice.omnivoice" not in sys.modules
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"


def test_listing_voice_models_does_not_load_a_local_active_engine():
    """A local engine configured as active must still answer from disk. Building
    its VoiceManager imports the engine AND loads its weights (`auto_load=True`)
    -- measured at 28s for audiodit, worse than the cold start being fixed."""

    result = _run_isolated(
        """
        import sys
        from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

        class _Owner:
            config = {}

        _VoiceCapability(_Owner()).list_tts_models()

        heavy = sorted(m for m in ("torch", "transformers", "diffusers") if m in sys.modules)
        assert not heavy, f"discovery imported {heavy}"
        assert "abstractvoice.audiodit" not in sys.modules
        print("ok")
        """,
        ABSTRACTVOICE_TTS_ENGINE="audiodit",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"


def test_an_abandoned_probe_leaves_the_process_free_to_exit():
    """An abandoned probe must not be joined at interpreter shutdown: a pool of
    ordinary workers turned a 5s bounded call into a 20s uninterruptible hang
    *after* the caller already had its answer."""

    started = time.monotonic()
    result = _run_isolated(
        """
        import time
        from abstractvoice.integrations.abstractcore_plugin import _probe_in_parallel

        results = _probe_in_parallel(
            {"hangs": lambda: time.sleep(30), "answers": lambda: "catalog"},
            budget_s=0.3,
        )
        assert results == {"answers": "catalog"}, results
        print("ok")
        """
    )
    elapsed = time.monotonic() - started

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"
    assert elapsed < 10.0, f"the straggler held the process open for {elapsed:.1f}s"


def _slow_owner(*, profiles_after: float = 0.0, catalog_after: float = 30.0):
    """An owner whose active manager answers profiles quickly and models slowly."""
    from abstractvoice.voice_profiles import VoiceProfile

    class _SlowVM:
        tts_engine = "openai"
        stt_engine = "openai"

        def get_profiles(self, kind="tts"):
            time.sleep(profiles_after)
            return [
                VoiceProfile(
                    engine_id="openai",
                    profile_id="alloy",
                    label="Alloy",
                    params={"provider": "openai", "voice": "alloy"},
                    tags={"provider": "openai", "engine_id": "openai", "kind": "voice"},
                )
            ]

        def get_active_profile(self, kind="tts"):
            return None

        def list_available_models(self):
            time.sleep(catalog_after)
            return {"openai": {"alloy": {"model": "tts-1", "remote": True}}}

    class _SlowOwner:
        config = {"voice_manager_instance": _SlowVM()}

    return _SlowOwner()


def test_an_unreachable_provider_is_marked_not_reported_as_empty(monkeypatch):
    """Absent must mean "no information", never "this provider has no models": a
    live host just past the budget would otherwise render as an empty selector.

    Asserted on `tts_catalog_by_provider`, which is what `list_tts_models(provider)`
    and `list_tts_voices(provider)` actually read.
    """
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setattr(plugin, "_REMOTE_DISCOVERY_TIMEOUT_S", 0.3)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    started = time.monotonic()
    catalog = plugin._VoiceCapability(_slow_owner()).voice_catalog()
    elapsed = time.monotonic() - started

    # The budget is honoured, and patching it is not silently ignored -- a default
    # argument would have been bound at def time and kept the full 5s.
    assert elapsed < 2.0, f"the discovery budget was not applied: {elapsed:.2f}s"
    assert catalog["catalogs"] == {}
    assert catalog["tts_catalog_by_provider"]["openai"]["unreachable"] is True


def test_a_reachable_provider_is_never_marked_unreachable(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    catalog = plugin._VoiceCapability(_slow_owner(catalog_after=0.0)).voice_catalog()

    by_provider = catalog["tts_catalog_by_provider"]
    assert "unreachable" not in by_provider["openai"]
    assert "openai" in catalog["catalogs"]
    # Local providers were never probed, so there is no claim to make either way.
    assert all("unreachable" not in entry for entry in by_provider.values())


def test_a_non_active_provider_keeps_what_it_already_paid_for(monkeypatch):
    """Every provider reports through the same slot, so partial retention is not a
    privilege of the active one. Realistic shape: the profiles endpoint answers, the
    models endpoint hangs -- `list_available_models` fetches both, so a slow models
    endpoint is exactly how a landed profile fetch gets orphaned."""
    import abstractvoice.integrations.abstractcore_plugin as plugin
    from abstractvoice.voice_profiles import VoiceProfile

    class _ProfilesThenHang:
        def get_profiles(self, kind="tts"):
            return [
                VoiceProfile(
                    engine_id="openai-compatible",
                    profile_id="served-voice",
                    label="Served",
                    params={"provider": "openai-compatible", "voice": "served-voice"},
                    tags={"provider": "openai-compatible", "engine_id": "openai-compatible", "kind": "voice"},
                )
            ]

        def list_available_models(self):
            time.sleep(30)
            return {}

        def get_active_profile(self, kind="tts"):
            return None

    class _ActiveVM:
        tts_engine = "openai"
        stt_engine = "openai"

        def list_available_models(self):
            return {"openai": {"alloy": {"model": "tts-1", "remote": True}}}

        def get_profiles(self, kind="tts"):
            return []

        def get_active_profile(self, kind="tts"):
            return None

    class _Owner3:
        config = {"voice_manager_instance": _ActiveVM()}

    monkeypatch.setattr(plugin, "_REMOTE_DISCOVERY_TIMEOUT_S", 0.5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9/v1")

    cap = plugin._VoiceCapability(_Owner3())
    cap._remote_discovery_vm = lambda engine: _ProfilesThenHang()
    catalog = cap.voice_catalog()

    entry = catalog["tts_catalog_by_provider"]["openai-compatible"]
    assert entry["unreachable"] is True, "its catalog never landed"
    assert "served-voice" in {voice.get("profile_id") for voice in entry["voices"]}
    assert "openai-compatible" in catalog["unreachable_tts_providers"]


def test_a_fetch_that_landed_inside_the_budget_is_kept(monkeypatch):
    """The active manager's fetches share one HTTP adapter and land one at a time.
    Work already paid for must not be discarded because a later fetch overran."""
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setattr(plugin, "_REMOTE_DISCOVERY_TIMEOUT_S", 1.5)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    catalog = plugin._VoiceCapability(_slow_owner(profiles_after=0.2)).voice_catalog()

    entry = catalog["tts_catalog_by_provider"]["openai"]
    assert entry["unreachable"] is True, "the catalog fetch was supposed to overrun"
    assert "alloy" in {voice.get("profile_id") for voice in entry["profiles"]}


def test_a_local_fetch_is_not_sequenced_behind_the_one_that_can_hang(monkeypatch):
    """THE RULE applies to every field, not only the catalog. `get_active_profile`
    reads the adapter's own `voice` attribute, so publishing `active_profile: None`
    when the catalog overran would report "no voice is selected" as fact."""
    import abstractvoice.integrations.abstractcore_plugin as plugin
    from abstractvoice.voice_profiles import VoiceProfile

    nova = VoiceProfile(
        engine_id="openai",
        profile_id="nova",
        label="Nova",
        params={"provider": "openai", "voice": "nova"},
        tags={"provider": "openai", "engine_id": "openai", "kind": "voice"},
    )

    class _HangsOnCatalog:
        tts_engine = "openai"
        stt_engine = "openai"

        def get_profiles(self, kind="tts"):
            return [nova]

        def get_active_profile(self, kind="tts"):
            return nova

        def list_available_models(self):
            time.sleep(30)
            return {}

    class _Owner4:
        config = {"voice_manager_instance": _HangsOnCatalog()}

    monkeypatch.setattr(plugin, "_REMOTE_DISCOVERY_TIMEOUT_S", 0.4)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    catalog = plugin._VoiceCapability(_Owner4()).voice_catalog()

    assert catalog["unreachable_tts_providers"] == ["openai"], "the catalog was supposed to overrun"
    assert (catalog["active_profile"] or {}).get("profile_id") == "nova"
    assert "nova" in {voice.get("profile_id") for voice in catalog["profiles"]}


def test_the_light_path_makes_no_reachability_claim(monkeypatch):
    """It contacts nobody, so an empty list would assert that every provider it
    happens to list -- remote ones included -- is reachable."""
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://10.255.255.1:8080/v1")

    class _Owner5:
        config: dict = {}

    light = plugin._VoiceCapability(_Owner5()).voice_catalog(providers_only=True)

    assert "openai" in light["tts_providers"]
    assert "unreachable_tts_providers" not in light


def test_the_discovery_budget_is_tunable_by_environment(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setenv("ABSTRACTVOICE_DISCOVERY_TIMEOUT_S", "0.4")
    started = time.monotonic()
    assert _probe_in_parallel({"hangs": lambda: time.sleep(30)}) == {}
    assert time.monotonic() - started < 2.0

    monkeypatch.delenv("ABSTRACTVOICE_DISCOVERY_TIMEOUT_S")
    assert plugin._remote_discovery_timeout_s() == _REMOTE_DISCOVERY_TIMEOUT_S


def test_every_listing_surface_answers_without_loading_the_active_engine():
    """The listing family must all be engine-free, not just the ones first fixed.
    `list_cloning_models` reached the packaged compatibility catalog through the
    active manager and cost 49s with a local engine configured, for static data."""

    result = _run_isolated(
        """
        import sys
        from abstractvoice.integrations.abstractcore_plugin import _VoiceCapability

        class _Owner:
            config = {}

        cap = _VoiceCapability(_Owner())
        answers = {
            "tts": cap.list_tts_models(),
            "tts_local_filter": cap.list_tts_models(provider="piper"),
            "tts_remote_filter": cap.list_tts_models(provider="openai"),
            "stt": cap.list_stt_models(),
            "stt_remote_filter": cap.list_stt_models(provider="openai"),
            "cloning": cap.list_cloning_models(),
            "voices": cap.list_tts_voices(provider="piper"),
            "compatibility": cap.compatibility_catalog(),
            "providers": cap.available_providers(),
            "light_catalog": cap.voice_catalog(providers_only=True),
            "local_catalog": cap.voice_catalog(provider="piper"),
        }
        assert answers["cloning"], "the compatibility catalog came back empty"
        assert answers["compatibility"].get("providers"), "no compatibility providers"
        assert answers["stt_remote_filter"], "remote STT models came back empty"

        heavy = sorted(m for m in ("torch", "transformers", "diffusers") if m in sys.modules)
        assert not heavy, f"discovery imported {heavy}"
        assert "abstractvoice.audiodit" not in sys.modules
        print("ok")
        """,
        ABSTRACTVOICE_TTS_ENGINE="audiodit",
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "ok"


def test_compatibility_catalog_is_built_from_config_without_a_manager(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    class _Local:
        config = {"voice_tts_engine": "audiodit", "voice_cloning_engine": "f5_tts"}

    cap = plugin._VoiceCapability(_Local())
    cap._get_vm = lambda: (_ for _ in ()).throw(AssertionError("must not build a manager"))

    catalog = cap.compatibility_catalog()

    # Real packaged data, hinted with the configured selection -- not an empty dict.
    assert "audiodit" in catalog["providers"]["tts"]
    assert "f5_tts" in catalog["providers"]["cloning"]
    assert cap.list_cloning_models(provider="f5_tts") == ["mrfakename/OpenF5-TTS-Base"]


def test_probe_in_parallel_costs_the_slowest_probe_not_the_sum():
    def slow(seconds: float):
        def probe():
            time.sleep(seconds)
            return seconds
        return probe

    probes = {f"engine-{index}": slow(0.3) for index in range(5)}

    started = time.monotonic()
    results = _probe_in_parallel(probes, budget_s=5.0)
    elapsed = time.monotonic() - started

    assert results == {f"engine-{index}": 0.3 for index in range(5)}
    assert elapsed < 1.0, f"probes ran serially: {elapsed:.2f}s"


def test_probe_in_parallel_drops_probes_that_overrun_the_budget():
    def hang():
        time.sleep(30)
        return "never"

    probes = {
        "reachable": lambda: "catalog",
        "broken": lambda: (_ for _ in ()).throw(RuntimeError("connection refused")),
        "unreachable": hang,
    }

    started = time.monotonic()
    results = _probe_in_parallel(probes, budget_s=0.4)
    elapsed = time.monotonic() - started

    assert results == {"reachable": "catalog"}
    assert elapsed < 2.0, f"waited on the straggler: {elapsed:.2f}s"


def test_probe_in_parallel_handles_an_empty_batch():
    assert _probe_in_parallel({}) == {}


def test_remote_discovery_manager_uses_the_discovery_budget(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    seen: dict = {}

    class _VM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ABSTRACTVOICE_REMOTE_TIMEOUT_S", raising=False)

    _VoiceCapability(_Owner())._remote_discovery_vm("openai-compatible")

    assert seen["remote_timeout_s"] == _REMOTE_DISCOVERY_TIMEOUT_S


def test_remote_discovery_never_lengthens_a_configured_timeout(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    seen: dict = {}

    class _VM:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr("abstractvoice.voice_manager.VoiceManager", _VM)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ABSTRACTVOICE_REMOTE_TIMEOUT_S", "1.5")

    _VoiceCapability(_Owner())._remote_discovery_vm("openai-compatible")

    assert seen["remote_timeout_s"] == 1.5


def test_local_tts_engine_availability_follows_the_filesystem(monkeypatch):
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setattr(plugin, "_engine_runtime_available", lambda engine, *_a, **_k: True)

    monkeypatch.setattr("abstractvoice.local_models.cached_tts_model_ids", lambda engine, **_kw: [])
    assert plugin._local_tts_engine_available("audiodit") is False

    monkeypatch.setattr(
        "abstractvoice.local_models.cached_tts_model_ids",
        lambda engine, **_kw: ["meituan-longcat/LongCat-AudioDiT-1B"],
    )
    assert plugin._local_tts_engine_available("audiodit") is True
    assert plugin._local_tts_engine_available("openai") is False
    # An engine is available exactly when it has a selectable model, so the two
    # answers cannot drift apart.
    assert plugin._selectable_tts_model_ids_for_provider("audiodit") == [
        "meituan-longcat/LongCat-AudioDiT-1B"
    ]


def test_a_configured_checkpoint_is_discoverable_through_config_and_env(monkeypatch, tmp_path):
    """Integrators pass a config dict, CLI users set the environment. Both must
    reach discovery, and neither may hand one engine's checkpoint to its siblings."""
    import abstractvoice.integrations.abstractcore_plugin as plugin

    hub = tmp_path / "hub"
    snapshot = hub / "models--myorg--my-finetune" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(plugin, "_engine_runtime_available", lambda engine, *_a, **_k: True)
    for key in ("ABSTRACTVOICE_TTS_ENGINE", "ABSTRACTVOICE_TTS_MODEL"):
        monkeypatch.delenv(key, raising=False)

    class _Unconfigured:
        config: dict = {}

    class _Configured:
        config = {"voice_tts_engine": "audiodit", "voice_tts_model": "myorg/my-finetune"}

    assert plugin._VoiceCapability(_Unconfigured())._selectable_local_tts_models("audiodit") == []

    configured = plugin._VoiceCapability(_Configured())
    assert configured._selectable_local_tts_models("audiodit") == ["myorg/my-finetune"]
    assert "audiodit" in configured._catalog_safe_local_engines()
    # The checkpoint belongs to the engine it was configured for.
    assert configured._selectable_local_tts_models("omnivoice") == []

    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "audiodit")
    monkeypatch.setenv("ABSTRACTVOICE_TTS_MODEL", "myorg/my-finetune")
    from_env = plugin._VoiceCapability(_Unconfigured())
    assert from_env._selectable_local_tts_models("audiodit") == ["myorg/my-finetune"]
    assert from_env._selectable_local_tts_models("omnivoice") == []

    # The mixed case: engine from config, model from the environment. Resolving the
    # two sources against separate views of "which engine" dropped this one.
    monkeypatch.delenv("ABSTRACTVOICE_TTS_ENGINE")

    class _EngineInConfig:
        config = {"voice_tts_engine": "audiodit"}

    assert plugin._VoiceCapability(_EngineInConfig())._selectable_local_tts_models("audiodit") == [
        "myorg/my-finetune"
    ]


def test_provider_filtered_voice_listing_keeps_the_light_path(monkeypatch):
    """`list_tts_voices(provider=...)` must forward its filter: asking for the
    unfiltered catalog instead builds the active engine, which for a local one is
    seconds and a torch import for an identical answer."""
    import abstractvoice.integrations.abstractcore_plugin as plugin

    monkeypatch.setenv("ABSTRACTVOICE_TTS_ENGINE", "audiodit")
    seen: list = []

    class _Owner2:
        config: dict = {}

    cap = plugin._VoiceCapability(_Owner2())
    cap._get_vm = lambda: seen.append("built") or (_ for _ in ()).throw(
        AssertionError("provider-filtered voice listing must not build the manager")
    )
    monkeypatch.setattr(plugin, "_catalog_safe_local_tts_engines", lambda *_a: ["piper"])
    monkeypatch.setattr(plugin, "_local_tts_engine_available", lambda engine, *_a: engine == "piper")

    cap.list_tts_voices(provider="piper")

    assert seen == []
