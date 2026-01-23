"""Test suite for Piper TTS Adapter.

This test suite validates:
1. Adapter initialization
2. Language support
3. Audio synthesis
4. Network methods (bytes/file generation)
5. Cross-platform compatibility
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


def test_piper_adapter_import():
    """Test that Piper adapter can be imported."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
        assert PiperTTSAdapter is not None
    except ImportError as e:
        pytest.skip(f"Piper TTS not installed: {e}")


def test_piper_adapter_initialization():
    """Test Piper adapter initializes correctly."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter(language='en')
    
    # Check availability
    # Note: May be False if piper-tts package not installed
    # We just check it doesn't crash
    is_available = adapter.is_available()
    assert isinstance(is_available, bool)


def test_piper_supported_languages():
    """Test that Piper supports required languages."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter()
    supported_langs = adapter.get_supported_languages()
    
    # Must support at least: EN, FR, DE, ES, RU, ZH-CN
    required_langs = ['en', 'fr', 'de', 'es', 'ru', 'zh']
    
    for lang in required_langs:
        assert lang in supported_langs, f"Language {lang} not supported by Piper"


def test_piper_synthesize():
    """Test basic synthesis functionality."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter(language='en')
    
    if not adapter.is_available():
        pytest.skip("Piper TTS not functional (package/models not available)")
    
    # Synthesize simple text
    text = "Hello world"
    audio_array = adapter.synthesize(text)
    
    # Verify output
    assert isinstance(audio_array, np.ndarray)
    assert audio_array.dtype == np.float32
    assert len(audio_array) > 0
    assert audio_array.min() >= -1.0
    assert audio_array.max() <= 1.0


def test_piper_synthesize_to_bytes():
    """Test synthesis to bytes for network transmission."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter(language='en')
    
    if not adapter.is_available():
        pytest.skip("Piper TTS not functional")
    
    # Synthesize to bytes
    text = "Testing network transmission"
    audio_bytes = adapter.synthesize_to_bytes(text, format='wav')
    
    # Verify output
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0
    # Check WAV header
    assert audio_bytes[:4] == b'RIFF'
    assert audio_bytes[8:12] == b'WAVE'


def test_piper_synthesize_to_file():
    """Test synthesis to file."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter(language='en')
    
    if not adapter.is_available():
        pytest.skip("Piper TTS not functional")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        # Synthesize to file
        text = "Testing file generation"
        result_path = adapter.synthesize_to_file(text, output_path)
        
        # Verify file exists
        assert os.path.exists(result_path)
        assert Path(result_path).stat().st_size > 0
        
        # Verify WAV format
        with open(result_path, 'rb') as f:
            header = f.read(12)
            assert header[:4] == b'RIFF'
            assert header[8:12] == b'WAVE'
    
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_piper_language_switching():
    """Test switching between languages."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter(language='en')
    
    if not adapter.is_available():
        pytest.skip("Piper TTS not functional")
    
    # Test switching to French
    success = adapter.set_language('fr')
    # May fail if model not downloaded yet
    # Just verify it doesn't crash and returns bool
    assert isinstance(success, bool)


def test_piper_get_info():
    """Test getting adapter metadata."""
    try:
        from abstractvoice.adapters.tts_piper import PiperTTSAdapter
    except ImportError:
        pytest.skip("Piper TTS not installed")
    
    adapter = PiperTTSAdapter(language='en')
    info = adapter.get_info()
    
    # Verify metadata structure
    assert 'name' in info
    assert 'languages' in info
    assert 'sample_rate' in info
    assert 'available' in info
    assert 'engine' in info
    assert info['engine'] == 'Piper TTS'
    assert 'requires_system_deps' in info
    assert info['requires_system_deps'] is False


def test_voice_manager_integration():
    """Test that VoiceManager can use Piper adapter."""
    try:
        from abstractvoice import VoiceManager
    except ImportError as e:
        pytest.skip(f"AbstractVoice not installed: {e}")
    
    # Initialize with auto engine selection (should try Piper first)
    vm = VoiceManager(language='en', tts_engine='auto', debug_mode=False)
    
    # Check that we have a TTS engine (Piper or VITS)
    assert vm._tts_engine_name in ['piper', 'vits', None]


def test_voice_manager_network_methods():
    """Test VoiceManager network methods with Piper."""
    try:
        from abstractvoice import VoiceManager
    except ImportError as e:
        pytest.skip(f"AbstractVoice not installed: {e}")
    
    vm = VoiceManager(language='en', tts_engine='piper', debug_mode=False)
    
    # Check if Piper is available
    if not vm.tts_adapter or not vm.tts_adapter.is_available():
        pytest.skip("Piper TTS not functional")
    
    # Test speak_to_bytes
    text = "Network test"
    audio_bytes = vm.speak_to_bytes(text)
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0
    
    # Test speak_to_file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        result_path = vm.speak_to_file(text, output_path)
        assert os.path.exists(result_path)
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
