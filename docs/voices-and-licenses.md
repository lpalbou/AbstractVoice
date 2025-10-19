# Voice Models and Licensing Information

AbstractVoice provides access to multiple TTS models with different characteristics. **All models used have permissive licenses** that allow commercial use.

## 🔍 Testing Voices Online

Before choosing voices, you can test them online:

- **Official Coqui TTS Demo**: [coquitts.com](https://coquitts.com/)
  - 3 free credits to test voices
  - Supports 8 languages including French, German, Italian
  - Voice cloning capabilities

- **Vocloner Demo**: [vocloner.com/voicecloning2.php](https://vocloner.com/voicecloning2.php)
  - Free voice cloning demo
  - Supports 17 languages including all AbstractVoice languages
  - Real-time generation

## 🎭 Available Voices by Language

### 🇺🇸 English (3 voices)

| Voice ID | Quality | Gender | Accent | License | Requirements |
|----------|---------|--------|--------|---------|--------------|
| `vits_premium` | ✨ Premium | Female | US English | Open source (LJSpeech) | espeak-ng |
| `fast_pitch_reliable` | 🔧 Good | Female | US English | Open source (LJSpeech) | none |
| `vctk_multi` | ✨ Premium | Multiple | British English | Open source (VCTK) | espeak-ng |

### 🇫🇷 French (2 voices)

| Voice ID | Quality | Gender | Accent | License | Requirements |
|----------|---------|--------|--------|---------|--------------|
| `css10_vits` | ✨ Premium | Male | **France French** | Apache 2.0 (CSS10/LibriVox) | espeak-ng |
| `mai_tacotron` | 🔧 Good | Female | **France French** | Permissive (M-AILABS/LibriVox) | none |

**⚠️ Important**: Both French voices are **France French** accent, not Canadian French. If you need Canadian French TTS, consider using cloud services like Google Cloud TTS or Azure Speech which offer `fr-CA` variants.

### 🇪🇸 Spanish (2 voices)

| Voice ID | Quality | Gender | Accent | License | Requirements |
|----------|---------|--------|--------|---------|--------------|
| `mai_tacotron` | 🔧 Good | Female | Spain Spanish | Permissive (M-AILABS) | none |
| `css10_vits` | ✨ Premium | Female | Spain Spanish | Apache 2.0 (CSS10) | espeak-ng |

### 🇩🇪 German (2 voices)

| Voice ID | Quality | Gender | Accent | License | Requirements |
|----------|---------|--------|--------|---------|--------------|
| `thorsten_vits` | ✨ Premium | Male | Standard German | Open source (Thorsten) | espeak-ng |
| `thorsten_tacotron` | 🔧 Good | Male | Standard German | Open source (Thorsten) | none |

### 🇮🇹 Italian (3 voices)

| Voice ID | Quality | Gender | Accent | License | Requirements |
|----------|---------|--------|--------|---------|--------------|
| `mai_male_vits` | ✨ Premium | Male | Standard Italian | Permissive (M-AILABS) | espeak-ng |
| `mai_female_vits` | ✨ Premium | Female | Standard Italian | Permissive (M-AILABS) | espeak-ng |
| `mai_female_glow` | 🔧 Good | Female | Standard Italian | Permissive (M-AILABS) | none |

## 📜 License Details

### ✅ Commercial Use Allowed

All TTS models in AbstractVoice use **permissive licenses** that allow commercial use:

1. **Apache License 2.0** (CSS10 dataset)
   - Commercial use: ✅ Allowed
   - Modification: ✅ Allowed
   - Distribution: ✅ Allowed
   - Source: LibriVox audiobooks (public domain)

2. **M-AILABS Permissive License**
   - Commercial use: ✅ Explicitly allowed
   - Very permissive: "even more free than before"
   - Source: LibriVox + Project Gutenberg (public domain texts 1884-1964)

3. **Open Source (LJSpeech, VCTK, Thorsten)**
   - Commercial use: ✅ Allowed
   - Research and commercial applications permitted

### 📚 Dataset Sources

- **LibriVox**: Public domain audiobooks
- **Project Gutenberg**: Public domain texts
- **LJSpeech**: Open research dataset
- **CSS10**: Multi-language speech synthesis corpus
- **VCTK**: CSTR VCTK Corpus (Edinburgh)

## 🎯 Choosing the Right Voice

### For Maximum Compatibility
Use voices that require **"none"** - they work everywhere:
```python
vm.set_voice('fr', 'mai_tacotron')      # French, always works
vm.set_voice('it', 'mai_female_glow')   # Italian, always works
```

### For Best Quality (requires espeak-ng)
Use **premium** voices with espeak-ng installed:
```python
vm.set_voice('fr', 'css10_vits')        # Premium French
vm.set_voice('it', 'mai_male_vits')     # Premium Italian male
```

### Voice Selection Examples

```python
from abstractvoice import VoiceManager

# Browse all available voices
vm = VoiceManager()
vm.list_voices()

# Browse French voices only
vm.list_voices('fr')

# Set specific voice
vm.set_voice('fr', 'css10_vits')        # Premium France French (male)
vm.set_voice('it', 'mai_female_vits')   # Premium Italian (female)
vm.set_voice('de', 'thorsten_vits')     # Premium German (male)

# Filter voices by criteria
premium_voices = vm.browse_voices(quality='premium')
female_voices = vm.browse_voices(gender='female')
compatible_voices = vm.browse_voices()  # Shows compatibility status
```

## ⚖️ Legal Compliance

AbstractVoice is designed for legal compliance:

- ✅ **All models**: Permissive licenses allowing commercial use
- ✅ **Source data**: Public domain or openly licensed
- ✅ **No restrictions**: Safe for commercial and research applications
- ✅ **Attribution**: Model sources documented for transparency

For the most current licensing information, consult the original dataset documentation:
- CSS10: [GitHub repository](https://github.com/Kyubyong/CSS10)
- M-AILABS: [Official website](https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/)
- Coqui TTS: [GitHub repository](https://github.com/coqui-ai/TTS)