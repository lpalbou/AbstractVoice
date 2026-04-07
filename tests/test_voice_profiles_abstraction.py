from abstractvoice.voice_profiles import VoiceProfile, find_voice_profile, get_builtin_voice_profiles


def test_voice_profile_normalizes_engine_id_and_label() -> None:
    p = VoiceProfile(engine_id="OmNiVoIcE", profile_id="female_01", label="")
    assert p.engine_id == "omnivoice"
    assert p.profile_id == "female_01"
    assert p.label == "female_01"


def test_builtin_omnivoice_profiles_load_and_are_findable() -> None:
    profiles = get_builtin_voice_profiles("omnivoice")
    assert isinstance(profiles, list)
    assert any(p.profile_id == "default" for p in profiles)
    assert find_voice_profile(profiles, "FEMALE_01") is not None

