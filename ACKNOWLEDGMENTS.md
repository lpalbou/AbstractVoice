# Acknowledgments

AbstractVoice uses several open-source libraries and models. We would like to acknowledge and thank the developers and contributors of these projects.

## Key Dependencies

### Speech Recognition
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Licensed under MIT
- [CTranslate2](https://github.com/OpenNMT/CTranslate2) - Licensed under MIT
- (Optional) [OpenAI Whisper](https://github.com/openai/whisper) - Licensed under MIT
- [WebRTCVAD](https://github.com/wiseman/py-webrtcvad) - Licensed under MIT

### Text-to-Speech
- [Piper](https://github.com/rhasspy/piper) - Licensed under MIT
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - Licensed under MIT
- (Optional) [Coqui TTS](https://github.com/coqui-ai/TTS) - Licensed under MPL-2.0

### Machine Learning
- [PyTorch](https://github.com/pytorch/pytorch) - Licensed under BSD-3-Clause
- [TorchAudio](https://github.com/pytorch/audio) - Licensed under BSD-3-Clause
- [NumPy](https://github.com/numpy/numpy) - Licensed under BSD-3-Clause
- [SciPy](https://github.com/scipy/scipy) - Licensed under BSD-3-Clause

### Web and API
- [Flask](https://github.com/pallets/flask) - Licensed under BSD-3-Clause
- [Requests](https://github.com/psf/requests) - Licensed under Apache-2.0

### Audio Processing
- [SoundFile](https://github.com/bastibe/python-soundfile) - Licensed under BSD-3-Clause
- [SoundDevice](https://github.com/spatialaudio/python-sounddevice) - Licensed under BSD-3-Clause
- [PortAudio](http://www.portaudio.com/) - Cross-platform audio I/O library (used via SoundDevice)
- [Librosa](https://github.com/librosa/librosa) - Licensed under ISC

### Utilities
- [tiktoken](https://github.com/openai/tiktoken) - Licensed under MIT

## Optional System Dependencies

AbstractVoice can optionally use the following system-level software for enhanced functionality:

### Text-to-Speech Phonemization
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) - Licensed under GPL-3.0
  - Optional dependency for VITS TTS model (provides best voice quality)
  - Used for phoneme conversion in advanced TTS synthesis
  - AbstractVoice automatically falls back to other models if not installed
  - Installation: `brew install espeak-ng` (macOS), `apt-get install espeak-ng` (Linux), or conda/chocolatey (Windows)

## Models

AbstractVoice may download and use pre-trained models as part of its operation:

- faster-whisper / Whisper model weights - Licensing varies by model; verify upstream terms.
- Piper voice models - Licensing varies by voice; verify upstream terms.
- (Optional) Coqui TTS models - Licensing varies by model/dataset; verify upstream terms.

## Notice

This acknowledgment file aims to properly credit the open-source projects used in AbstractVoice. If you believe a project is missing or improperly attributed, please open an issue or contact the maintainers.

When using AbstractVoice, be aware that some of the underlying dependencies may have different licensing terms than AbstractVoice itself. While AbstractVoice is licensed under MIT, users should review the licenses of the dependencies if they intend to use them independently or in ways that may conflict with their respective licenses.

For example, some TTS models may have non-commercial use restrictions. If you plan to use AbstractVoice in a commercial application, you should ensure you are using models that permit commercial use or obtain appropriate licenses. 