# Agentic Voice Assistant Stack Deep Research

## Executive summary

High-quality agentic voice assistants in 2026 are still best built as a *streaming cascade*—**STT → LLM/agent orchestration → TTS**—with real-time pipelining and careful turn-taking, rather than relying solely on “native” speech-to-speech foundation models. A 2026 technical tutorial on enterprise realtime voice agents reports that (a) native speech-to-speech models can be too slow for realtime (example: ~13 s time-to-first-audio for one such model), and (b) the industry-standard approach is the cascaded streaming pipeline, achieving measured sub‑second P50 time-to-first-audio in a well-engineered setup. citeturn14academia38

On **STT**, the quality frontier is characterised by (i) large-scale weakly supervised models such as Whisper (trained at internet scale, strong robustness and multilingual generalisation), citeturn19search0turn19search7 and (ii) massively multilingual families like Meta’s MMS (claims >1,100 languages for ASR and “more than halves WER vs Whisper” on 54 FLEURS languages in the paper’s comparison). citeturn15search1turn17search6turn18search4 In production, however, cloud STT vendors differentiate on *streaming UX* (partial stability, endpointing), *enterprise features* (language ID constraints, custom vocabularies, diarisation, SLAs), and *integration simplicity*.

On **TTS**, “naturalness” is now dominated less by classic concatenative/neural vocoders and more by **expressive generative TTS**, plus strong **voice customisation/branding**. For research/open stacks, StyleTTS2 claims human-level or human-matching subjective results on LJSpeech and VCTK and strong zero-shot speaker adaptation when trained on LibriTTS. citeturn17search4turn17search31turn20search6turn20search1turn20search3 For commercial systems, differentiation often comes from *voice catalogue breadth*, *streaming audio synthesis*, and *safe custom voice programmes* (especially due to deepfake risk, hence access controls). citeturn5search2turn5search16turn6search4turn10search3

Within the entity["company","Amazon Web Services","cloud provider"] ecosystem, a modern “agentic voice” architecture can be composed from:
- **Amazon Transcribe (Streaming)** for bidirectional HTTP/2/WebSocket realtime STT and language identification, citeturn19search6turn15search4
- **Amazon Bedrock** for foundation models (plus Agents/knowledge bases/guardrails), citeturn6search3turn1search10turn1search11turn1search12
- **Amazon Polly** (Neural/Generative voices + recent bidirectional streaming API for generative voices) for lower-latency conversational TTS and branded voices (Brand Voice), citeturn10search10turn10search3turn6search1turn10search1
- **AgentCore** (Runtime, Gateway, Memory, Policy) to operationalise tool use, memory, security enforcement, and observability at scale, citeturn12view0turn13view3turn13view1turn13view0turn13view2
- **AWS Lambda**, **Kinesis Video Streams (WebRTC)**, **Chime SDK**, and **IoT Core** for event-driven tools, realtime audio transport, and device connectivity. citeturn7search0turn7search1turn7search2turn7search23

## Speech-to-text options and trade-offs

### How to interpret “STT quality” in 2026

Accuracy for STT is still most commonly reported as **word error rate (WER)**, but production voice assistants care equally about: (a) **streaming behaviour** (time-to-first-partial, stability of partials, endpointing/turn detection), (b) **robustness** (far-field, overlapped speech, accents, code-switching), and (c) **language coverage and adaptation** (language ID, domain terms). Distant multi-speaker conditions remain challenging enough that dedicated benchmarks like CHiME-6 exist for diarisation + recognition in everyday home environments (multi-mic, conversation, overlap). citeturn21search0turn21search9turn21search1

Multilingual evaluation increasingly uses datasets like **FLEURS** (102 languages, parallel speech, designed to benchmark ASR, speech language ID, translation, retrieval). citeturn15search0turn15search10 For real-world multilingual political speech and accents, datasets like **VoxPopuli** provide large-scale multilingual speech resources and are commonly used for representation learning and downstream ASR benchmarking. citeturn21search3turn21search7turn21search15

### Comparative table: open-source / self-host STT

| Option | Deployment | Streaming support | Multilingual / accents | Customisation (domain, diarisation) | Published accuracy signals | Licensing | Integration complexity | Notes / best fit |
|---|---|---|---|---|---|---|---|---|
| Whisper (incl. large‑v3 and large‑v3‑turbo) | Self-host (GPU/CPU), on-device variants exist via ports | Not “native streaming” in the paper; commonly implemented via chunking; turbo variants target speed | Trained with large-scale weak supervision; strong multilingual generalisation reported; commonly used baseline in multilingual ASR research citeturn19search0turn19search7turn19search4 | No built-in enterprise diarisation; domain adaptation typically via prompting, custom decoding constraints, or external rescoring | Paper reports strong zero-shot robustness and competitiveness with supervised baselines at scale citeturn19search0turn19search7 | MIT (code + weights) citeturn19search5turn19search2 | Medium–High | Best “generalist” open model; streaming UX requires engineering (VAD, buffering, incremental decoding). |
| whisper.cpp | On-device / edge CPU (and some GPU via backends) | Includes realtime microphone “stream” example (sampling and continuous transcription) citeturn11search0 | Follows Whisper multilingual capability (depends on chosen Whisper weights) citeturn19search5 | Same as Whisper | Depends on Whisper model used; focus is efficient inference | MIT (via Whisper + project conventions; commonly distributed as MIT) citeturn19search5turn19search2 | Medium | Strong choice for privacy/on-device prototypes; best with quantisation and careful threading. |
| faster-whisper (CTranslate2) | Self-host GPU/CPU | Typically used in chunked/streaming loops; optimised for speed | Same language coverage as Whisper model selected | Same as Whisper | Claims “up to 4× faster than openai/whisper for same accuracy” and memory savings; supports 8-bit quantisation citeturn11search1turn11search21 | MIT citeturn11search21turn11search13 | Medium | Practical when you want Whisper-level quality with better latency/throughput on modest hardware. |
| MMS (Meta) multilingual ASR (e.g., mms‑1b‑all) | Self-host (GPU), research/community deployments | Not primarily positioned as a streaming UX product; can be engineered into streaming | Paper claims pretrain across 1,406 languages; ASR model for 1,107 languages; language ID for 4,017 languages citeturn15search1turn18search8 | Domain customisation not productised like enterprise cloud; diarisation external | Paper reports multilingual ASR that “more than halves” Whisper WER on 54 languages of FLEURS in their experiment citeturn15search1turn15search0 | Weights under CC‑BY‑NC 4.0 (non‑commercial) on HF model cards citeturn18search4turn18search30turn18search0 | High | Outstanding language coverage, but non‑commercial licensing is a blocker for many production uses. |
| Vosk | On-device / edge CPU (mobile, Raspberry Pi), server | Provides streaming API for “best UX” (per project docs) citeturn11search22 | 20+ languages/dialects listed; multilingual but narrower than Whisper/MMS citeturn11search22turn11search2 | Supports vocabulary reconfiguration; diarisation/speaker ID varies by model/tooling citeturn11search22 | Publishes per-model WER on various test sets on model page (varies widely) citeturn11search6 | Apache‑2.0 for many models/toolkit (some models differ) citeturn11search6turn11search36 | Low–Medium | Good for lightweight offline STT; accuracy/robustness often behind modern large models, but latency can be excellent. |
| NVIDIA Riva ASR | On-prem / private cloud (GPU) | Designed for real-time speech AI; deployed as services/skills | Multilingual depends on Riva model pack; enterprise focus | Emphasises customisation and real-time performance | Vendor performance/latency characteristics are documented per release; targeted at production | Proprietary/commercial (platform); clients repo MIT but service is not citeturn17search26turn17search30 | Medium–High | Strong choice when you standardise on NVIDIA GPUs and want enterprise control over hosting and performance. |

### Comparative table: commercial STT APIs

| Option | Deployment | Streaming support | Multilingual / accents | Customisation (domain, diarisation) | Latency signals | Licensing / access | Integration complexity | Notes / best fit |
|---|---|---|---|---|---|---|---|---|
| Amazon Transcribe (Streaming) | Cloud | Bidirectional HTTP/2 or WebSocket streaming STT citeturn19search6turn19search10 | Supports dominant language ID and multi-language ID in streaming; multi-language ID transcribes supported languages as speakers switch citeturn15search4turn15search7 | Custom vocabularies; custom language models; speaker diarisation; note: streaming language ID has feature-combination constraints citeturn1search15turn10search23turn1search18turn15search4 | Real-time design, but exact “ms” latency not standardised publicly; chunking and partial results are supported via streaming interfaces citeturn19search6turn19search10 | Proprietary cloud service | Low | Best if you want deep AWS integration, compliance controls, and straightforward scale. |
| Google Cloud Speech-to-Text | Cloud | Supports streaming in APIs/SDKs; phrase “hints”/adaptation is supported in interfaces citeturn4search16turn4search17 | Broad language coverage (varies by model/version) | Adaptation/phrase sets; diarisation supported in some modes (feature varies by version) | Vendor promotes realtime transcription; measure yourself per domain | Proprietary cloud service | Low | Strong general-purpose STT; model/version selection is crucial for latency+accuracy. |
| Azure Speech to Text | Cloud | Real-time transcription supported in SDKs citeturn4search21turn5search12 | 100+ languages claimed in platform overview; verify per locale/model citeturn5search12turn4search21 | Speaker diarisation is available; custom speech features exist (training/adaptation) citeturn4search21 | Designed for realtime; measure TT-first-partial and endpointing in your conditions | Proprietary cloud service | Low | Good enterprise fit, especially if you already use Microsoft identity/compliance tooling. |
| Deepgram | Cloud (and some on-prem offerings) | WebSocket streaming with interim results; official docs define streaming finalisation behaviour and latency measurement guidance citeturn4search15turn4search18turn4search21 | Broad language support (varies by model); marketed for conversational AI | Custom dictionaries/vocab features exist; diarisation depends on plan/model | Claims “<300 ms transcription latency” in a product sheet; interpret as vendor claim, validate in situ citeturn4search2 | Proprietary | Low–Medium | Popular choice for low-latency streaming STT; best evaluated with your audio transport + VAD/endpointing. |
| Speechmatics | Cloud and on-prem | Real-time APIs; also shipping broader “Flow” voice-interaction concept (ASR+LLM+TTS) | Markets inclusive speech recognition across demographics/accents; verify with your evaluation set | Enterprise features vary; on-prem possible | Vendor messaging focuses on low latency; quantify yourself | Proprietary | Medium | Strong for organisations wanting on-prem or “speech-first” vendor focus; validate language/locale fit in trials. citeturn1search4turn4search8 |
| AssemblyAI | Cloud | Realtime streaming products; vendor docs/blogs describe ~300 ms class latency for streaming (vendor claim) citeturn4search35turn10search19 | Multilingual support varies by model | Diarisation, custom vocabulary features exist | Vendor describes low-latency realtime; measure end-to-end with your stack | Proprietary | Low | Good developer UX and modern agent integrations; treat published latency numbers as directional until benchmarked. |

## Text-to-speech options and trade-offs

### What “TTS quality” means for agentic assistants

For conversational assistants, TTS quality is less about long-form audiobook fidelity and more about:
- **time-to-first-audio** and stable streaming synthesis,
- **prosody control** (pauses, emphasis, pronunciation),
- **interruptibility** (barge‑in: stopping audio mid-utterance),
- and **voice identity and safety** (custom voices/voice cloning with consent controls).

Because TTS can be abused for impersonation, some enterprise custom voice programmes explicitly gate access. For example, Azure’s Custom Voice documentation states that access is limited and must be requested based on eligibility and usage criteria. citeturn5search2

### Comparative table: open-source / self-host TTS

| Option | Deployment | Streaming support | Voice cloning / custom voices | Multilingual | Published quality signals | Licensing | Integration complexity | Notes / best fit |
|---|---|---|---|---|---|---|---|---|
| Piper | On-device / edge CPU, server | Typically used as low-latency local synthesis; ONNX-based; real-time factor examples are community-reported (hardware-dependent) citeturn5search0turn5search10 | Voice training/fine-tuning exists via community tooling; quality depends on voice | Many voices/languages via community ecosystem | Known for speed; example voice discussion reports ~0.04× real-time factor on RTX 4080 for a voice model citeturn5search10 | MIT in original repo; note project direction changes (repository notes “development has moved”) citeturn5search0 | Medium | Best for privacy-focused local TTS with modest compute; voice quality varies by voice dataset/model. |
| Coqui XTTS‑v2 | Self-host GPU | Coqui docs explicitly claim “streaming inference with <200 ms latency” (validate in your environment) citeturn16view1 | Strong capability: cross-language voice cloning from short reference clip (model card describes ~6 s) citeturn16view0 | Model card lists 17 languages; docs list 16 in that revision (expect drift by version) citeturn16view0turn16view1 | Positioned as voice generation/clone model; quality varies by language and reference audio | Coqui Public Model License (CPML) citeturn16view0turn16view1 | High | Excellent for prototype voice cloning; licensing and responsible-use constraints must be reviewed carefully for production. |
| StyleTTS2 | Self-host GPU | Not primarily marketed as “streaming-first”; can be engineered | Supports style control; research shows strong zero-shot speaker adaptation when trained on LibriTTS citeturn17search4turn20search3 | Typically English-focused benchmarks | Repo claims MIT; paper/repo claims human-level or human-matching subjective results on LJSpeech and VCTK; and strong LibriTTS zero-shot adaptation citeturn17search4turn17search31 | MIT code (but pay attention to any checkpoint/voice use terms) citeturn17search4turn17search0 | High | Strong research-grade naturalness; productionisation requires careful latency work and a clear voice IP/consent story. |
| MMS TTS (Meta) | Self-host | Not positioned as a streaming UX system | Provides speech synthesis models at scale; voice cloning not the focus | “Text-to-speech models for over 1,100 languages” in MMS materials citeturn15search1turn17search6turn17search17 | Breadth is the key claim; quality varies by language/data | Often CC‑BY‑NC 4.0 for models on HF (non‑commercial) citeturn18search4turn17search17 | High | Best when language coverage is the overriding constraint and licensing allows your use. |
| NVIDIA Riva TTS | On-prem / private cloud (GPU) | Supports streaming and batch; streaming returns audio chunks early to reduce time-to-first-audio citeturn17search14turn17search7 | Vendor claims ability to create a “natural custom voice” with ~30 min of actor data (vendor statement) citeturn17search26 | Language coverage depends on Riva packs | Enterprise-oriented; performance evaluation guidance exists by release | Proprietary/commercial | Medium–High | Strong option for enterprise/on-prem with NVIDIA GPUs and operational control. |

### Comparative table: commercial TTS APIs

| Option | Deployment | Streaming support | Voice cloning / custom voice | Multilingual / voice catalogue | Latency / quality signals | Licensing / access | Integration complexity | Notes / best fit |
|---|---|---|---|---|---|---|---|---|
| Amazon Polly (Standard/Neural/Generative + Brand Voice) | Cloud | API returns an audio stream for immediate playback; recent “What’s New” announces bidirectional streaming API for Generative voices citeturn10search12turn10search10 | Brand Voice is a custom engagement to build exclusive NTTS voice for an organisation citeturn10search3turn10search1 | “Available voices” table enumerates voices/locales; supports many languages/variants (see docs) citeturn6search4turn6search8 | Generative voices positioned as highly expressive; SSML support varies by voice tier (see SSML tags table) citeturn10search20turn6search0 | Proprietary | Low | Best for AWS-native deployments; serious custom voice requires enterprise engagement (Brand Voice). |
| Google Cloud Text-to-Speech | Cloud | Marketing page promotes “streaming audio synthesis” for agents citeturn5search16 | “Neural2” voices are based on the same tech as Custom Voice; implies custom voice tech availability without training your own (per docs) citeturn5search3 | Claims 380+ voices across 75+ languages/variants (marketing; verify per region/model) citeturn5search16 | Strong catalogue breadth; evaluate per language and prosody needs | Proprietary | Low | Strong multilingual TTS; ideal when voice variety and language breadth matter. |
| Azure Text to Speech + Custom Voice | Cloud | Part of Azure Speech platform; supports TTS APIs and custom voice tooling citeturn5search12turn5search30 | Custom Voice allows creating a customised synthetic voice from speech samples; access is limited/controlled citeturn5search2turn5search30 | Broad language coverage (verify per locale/voice) | Enterprise governance and consent controls are a major differentiator | Proprietary | Medium | Best for enterprises needing controlled custom voices and governance/auditability. |
| ElevenLabs | Cloud | API-first voice infra; positioned for realtime high-quality output; model release notes highlight “Multilingual v2” citeturn5search8turn5search5turn5search1 | Provides voice cloning and conversational infrastructure (vendor capability statement) citeturn5search8 | Blog states Multilingual v2 supports 29 languages citeturn5search1 | “Expressive” naturalness is key vendor positioning; benchmark in your UX pipeline | Proprietary | Low | Best when you want top-tier expressiveness + fast iteration and accept external dependency. |
| Speechmatics TTS (preview) | Cloud | Docs describe “low latency TTS API” in preview citeturn4search30 | Custom voice details vary; verify programme | Language coverage evolving | Preview/free-to-use status suggests change risk citeturn4search30 | Proprietary | Medium | Consider for experimentation; avoid hard dependencies until stable GA + pricing + SLA. |

### AWS-specific notes for TTS pricing and tiers

As of the published pricing page, Amazon Polly bills per characters processed. Public list prices include Standard ($4 / 1M chars), Neural ($16 / 1M chars), Long‑Form ($100 / 1M chars), and Generative ($30 / 1M chars), outside free tier conditions. citeturn6search1

## Agentic orchestration best practices for realtime voice agents

### Core principle: pipelining beats monolithic “giant models”

The 2026 enterprise realtime voice-agent tutorial concludes that realtime performance is primarily achieved by **streaming and pipelining across STT, LLM, and TTS**, not merely by picking a single model. It reports measurable sub‑second time-to-first-audio in a cascaded pipeline and highlights that some native speech-to-speech models (in their investigation) can be too slow for realtime interaction. citeturn14academia38

In practice, achieving “feels instant” UX typically requires:
- streaming STT partials into the agent loop,
- beginning LLM generation before the user fully finishes (carefully gated by turn detection),
- and streaming TTS audio frames as soon as the first text chunk is available.

Frameworks such as **LiveKit Agents** and **Pipecat** explicitly productise these concerns (turn detection, interruptions, orchestrating a streaming STT→LLM→TTS pipeline). citeturn14search0turn14search1turn14search11

### Tool use and planning patterns that work in production

Agentic “tool use” is most reliable when the model’s free-form generation is constrained by structured actions, bounded policies, and observable retries.

Research patterns that influenced modern agent design include:
- **ReAct**: interleaving reasoning traces and actions, improving reliability by grounding the agent in external tools/knowledge sources. citeturn8search0turn8search4
- **Toolformer**: self-supervised training to decide *when* and *how* to call tools (APIs) and integrate results. citeturn8search1turn8search5

For evaluation of tool-using agents, benchmarks such as **AgentBench**, **WebArena**, **ToolBench / StableToolBench**, and **GAIA** are widely cited as ways to measure multi-turn decision making, tool selection, and real-world task completion under controlled environments. citeturn8search3turn9search1turn9search12turn9search0turn9search3

### Memory: separate short-term context from long-term personalisation

A reliable voice assistant usually needs both:
- **short-term memory**: turn-by-turn context within a session (pronoun resolution, ellipsis, “tomorrow” follow‑ups);
- **long-term memory**: durable preferences and summaries extracted across sessions.

This distinction is explicit in AgentCore Memory, which defines short-term session memory and long-term memory that extracts and stores key insights (preferences, facts, summaries) for future interactions. citeturn13view0 A similar separation is common in agent research (episodic memory streams, reflection, planning) such as Generative Agents. citeturn8search2turn8search6

### Safety and governance: treat STT transcripts as untrusted input

For voice assistants, *audio is an input channel for adversarial prompting*. Spoken prompt injection (“ignore previous instructions”, “call this tool with my account”) becomes text after STT, and must be governed like any other untrusted user input.

Production controls that consistently improve safety outcomes:
- **Boundary enforcement outside the model** (policy checks before tool execution),
- **Least-privilege tool design** (narrow tool schemas, scoped credentials),
- **Audit logs** of tool calls and policy decisions,
- **PII/secret handling** (redaction policies; avoid storing raw audio by default).

AgentCore Policy is designed specifically to intercept agent-to-tool traffic through gateways and evaluate requests against deterministic policies written in Cedar (with optional natural-language authoring that generates and validates candidate policies). citeturn13view2turn12view0 Amazon Bedrock Guardrails and agent monitoring/evaluation features provide additional safety scaffolding at the model layer (content/PII/prompt-attack controls vary by configuration). citeturn1search12turn12view0

## AWS reference architecture with AgentCore and related services

### Architectural building blocks in AWS

AgentCore is positioned by AWS as an “agentic platform” with intelligent memory and a gateway for controlled tool/data access, plus operational monitoring and evaluation (token usage, latency, goal success rate, safety). citeturn12view0 Key components relevant to voice assistants include:
- **AgentCore Runtime** for hosting agents with session isolation (microVM per user session, sanitised on termination) and up to 8‑hour workloads, citeturn13view3
- **AgentCore Gateway** to convert APIs/Lambda/OpenAPI/Smithy into MCP-compatible tools with semantic tool selection and authentication handling, citeturn13view1
- **AgentCore Policy** for tool-call interception and enforcement, citeturn13view2
- **AgentCore Memory** for short-term and long-term memory. citeturn13view0

For realtime audio transport and device connectivity:
- **Kinesis Video Streams with WebRTC** supports realtime audio/video streaming to the cloud via WebRTC, citeturn7search0turn7search4
- **Chime SDK** provides WebRTC media for sending/receiving audio with device/browser sample rates up to 48 kHz, citeturn7search1turn7search5
- **AWS IoT Core** supports MQTT / MQTT over WSS device protocols, useful for device commands/state and low-bandwidth signalling. citeturn7search2turn7search6

For speech services:
- **Amazon Transcribe Streaming** provides bidirectional HTTP/2 or WebSocket sessions where audio is streamed in and transcripts streamed out, citeturn19search6 including streaming language identification/multi-language ID with documented constraints. citeturn15search4
- **Amazon Polly** provides text-to-speech returning audio streams; recent updates announce a bidirectional streaming API for generative voices. citeturn10search12turn10search10
- **Amazon Lex** is a managed service that combines ASR + NLU for voice/text bots and integrates natively with Lambda for business logic—useful for tightly scoped dialog flows, especially when you **don’t** want an open-domain LLM loop for everything. citeturn1search0turn7search23

### Mermaid diagram: cloud-native realtime voice agent on AWS

```mermaid
flowchart LR
  U[User device\n(mic + speaker)] -->|WebRTC audio| RTC[WebRTC Transport\n(Chime SDK or Kinesis Video Streams WebRTC)]
  RTC -->|PCM/Opus frames| VAD[VAD + Turn detection\n(edge or server)]

  VAD -->|streaming audio| STT[Amazon Transcribe Streaming\n(WebSocket/HTTP2)]
  STT -->|partial + final transcripts| ORCH[Agent Orchestrator\n(AgentCore Runtime)]

  ORCH --> MEM[AgentCore Memory\n(short-term + long-term)]
  ORCH -->|tool calls| GW[AgentCore Gateway]
  GW -->|policy-enforced| POL[AgentCore Policy]
  POL -->|invoke| TOOLS[AWS Lambda tools\n+ enterprise APIs]

  ORCH -->|response text chunks| TTS[Amazon Polly\n(Neural/Generative TTS)]
  TTS -->|streaming audio| RTC
  RTC -->|audio| U

  U <--> IOT[AWS IoT Core\n(MQTT/WSS control plane)]
  ORCH --> OBS[CloudWatch + OpenTelemetry\n(AgentCore Observability)]
```

This diagram captures the operational separation that tends to make voice agents robust: streaming audio transport → specialised STT/TTS services → an agent runtime that owns session state/tools/memory. AgentCore explicitly supports this separation: Runtime for isolated sessions and long workloads, Gateway for tool conversion/discovery, Policy for boundary enforcement, and Memory for context across turns/sessions. citeturn13view3turn13view1turn13view2turn13view0

### Mermaid diagram: AgentCore WebRTC voice agent with Bedrock speech-to-speech

AWS also documents a pattern where a browser establishes a WebRTC voice connection directly to an agent running on AgentCore Runtime, using Kinesis Video Streams managed TURN for relaying, and the agent streams audio to/from an Amazon Bedrock foundation model for speech-to-speech conversation. citeturn12view1

```mermaid
flowchart LR
  B[Browser client] -->|WebRTC offer/ICE| AG[AgentCore Runtime\n(agent endpoint)]
  AG -->|GetIceServerConfig| KVS[Kinesis Video Streams TURN]
  B <--> KVS
  KVS <--> AG

  B -->|mic audio| AG
  AG -->|audio stream| FM[Amazon Bedrock\nspeech-to-speech FM]
  FM -->|spoken response stream| AG
  AG -->|agent audio| B
```

This pattern can reduce pipeline complexity (no explicit STT/TTS), but *latency and controllability* still must be benchmarked against cascaded STT→LLM→TTS approaches, which remain standard for sub‑second conversational UX. citeturn12view1turn14academia38

## Recommended stacks by scenario

### Realtime consumer device assistant

A consumer assistant (smartphone app, smart speaker, wearable) typically prioritises **barge‑in**, **low round-trip latency**, and **multilingual support**. A pragmatic high-quality stack in AWS is:

STT:
- Start with **Amazon Transcribe Streaming** for operational simplicity and scale, using multi-language identification if you need code-switching (with awareness of documented feature constraints). citeturn19search6turn15search4  
- If you need best-in-class multilingual accuracy for niche languages and licensing permits, evaluate MMS-family models; the MMS paper claims very large language coverage and strong FLEURS results. citeturn15search1turn18search4

TTS:
- Use **Amazon Polly Generative voices** (and bidirectional streaming where available) for low-latency conversational synthesis; consider **Brand Voice** if a unique voice identity is central to product differentiation. citeturn10search10turn10search3turn10search1

Orchestration:
- Run the agent on **AgentCore Runtime** to get session isolation and the ability to scale down to zero while supporting realtime plus longer workflows. citeturn13view3  
- Use **AgentCore Gateway + Policy** to expose a small set of safe device/app tools (e.g., account lookup, order status, smart-home controls) with deterministic pre-execution enforcement. citeturn13view1turn13view2  
- Use **AgentCore Memory** for preferences (language, tone, frequently used devices) to reduce friction across sessions. citeturn13view0

Transport:
- Use **Chime SDK** or **Kinesis Video Streams WebRTC** for realtime bidirectional audio; use **IoT Core** for device control-plane events and state sync. citeturn7search1turn7search4turn7search2

### Enterprise cloud assistant

Enterprise assistants usually prioritise **governance**, **auditability**, **VPC-enclosed connectivity**, and safe integration with internal systems.

STT:
- **Amazon Transcribe** for scalable ingestion plus language identification, custom vocabularies, custom language models, and diarisation where needed. citeturn1search15turn10search23turn1search18turn15search7  
- If you need on‑prem STT, evaluate enterprise offerings such as **NVIDIA Riva** or speech vendors offering on‑prem deployments; benchmark against your accented/noisy corpora. citeturn17search26turn5search12

TTS:
- **Amazon Polly** for AWS-native scale; use voice tiering (Standard vs Neural vs Generative) based on use case. citeturn6search1turn10search12turn10search20  
- If strict “brand voice” identity is required, use Polly **Brand Voice** or an equivalent governed custom voice programme. citeturn10search3turn5search2

Orchestration and safety:
- Use **Amazon Bedrock Agents** when you want managed agent orchestration and integration with company systems and knowledge bases (action groups / KB patterns). citeturn1search10turn1search11  
- Use **AgentCore** when you need *stronger operational primitives*—session isolation microVMs, boundary policy enforcement, memory services, and production evaluation/observability as first-class features. citeturn12view0turn13view3turn13view2  
- Use **Bedrock Guardrails** plus tool-boundary policies (defence in depth). citeturn1search12turn13view2

Transport:
- Chime SDK for web/mobile embedded assistants; Kinesis WebRTC for media ingestion and relaying patterns; integrate with enterprise auth. citeturn7search1turn7search0turn12view1

### Privacy-focused on-device assistant

Privacy-first assistants aim to keep audio and transcripts **entirely local by default**, optionally syncing *only minimal intents*.

STT (local):
- whisper.cpp / faster‑whisper for high-quality local recognition, with quantisation and streaming loops. citeturn11search0turn11search1turn11search21  
- Vosk for ultra-lightweight offline STT when compute is limited and supported languages fit. citeturn11search22turn11search2

TTS (local):
- Piper for fast local TTS with a broad community voice ecosystem. citeturn5search0turn5search10  
- For high-fidelity research-grade synthesis, evaluate StyleTTS2-class systems (likely GPU-leaning). citeturn17search4turn17search31

Orchestration (local-first):
- Use a local pipeline framework (e.g., LiveKit Agents or Pipecat) to handle turn detection, interruptions, and realtime streaming between local STT/LLM/TTS components. citeturn14search0turn14search1  
- If you still need cloud connectivity for tools, send *structured intents* and *minimal context* rather than raw audio or full transcripts.

AWS fit (privacy-preserving):
- Use **IoT Core** for device fleet management, state sync, and command routing (MQTT/WSS), without shipping raw audio. citeturn7search2  
- If selective cloud “skills” are needed, route only explicit tool calls to **Lambda** (or an AgentCore Gateway endpoint) with strict policies and redaction. citeturn13view1turn13view2turn7search23

## Benchmarks, evaluation methodology, latency and cost engineering, safety and privacy controls

### Benchmarks and datasets

A rigorous evaluation plan should include (at minimum) **in-domain** data plus respected public benchmarks for regression testing:

STT benchmarks:
- **LibriSpeech** (∼1000 hours read English speech, 16 kHz) as a classic WER reference point. citeturn20search0turn20search4  
- **FLEURS** (102 languages) for multilingual ASR and language ID comparisons. citeturn15search0turn15search10  
- **Common Voice** (CC0 speech datasets, community-led multilingual data releases) for accent/dialect diversity and long-tail language coverage in evaluation and fine-tuning. citeturn11search31turn11search15turn11search19  
- **CHiME‑6** for far-field, multi-speaker conversational recognition and diarisation stress testing. citeturn21search0turn21search1turn21search9  
- **AMI Meeting Corpus** (100 hours meeting recordings) for speech in interactive meeting conditions. citeturn21search2turn21search14  
- **Switchboard** (telephone conversations) for conversational telephone speech. citeturn21search8turn21search16  
- **VoxPopuli** for multilingual political speech and semi-supervised settings. citeturn21search3turn21search7turn21search15

TTS benchmarks:
- **LJ Speech** (single-speaker, ~24 hours, public domain) for single-speaker naturalness testing. citeturn20search1  
- **VCTK** (multi-speaker English with diverse accents) for multi-speaker quality and speaker similarity. citeturn20search6turn20search34  
- **LibriTTS** (585 hours, 2,456 speakers; designed for TTS) for multi-speaker and adaptation evaluation. citeturn20search3turn20search11

### Evaluation metrics that matter for agentic voice assistants

STT metrics:
- **WER/CER** overall and per‑condition slice (noise type, SNR, accent group, far-field vs close mic).
- **Streaming latency**: time to first partial, time to stable partial, time to final transcript after end-of-speech.
- **Endpointing errors**: cut‑offs (finalising too early) vs lag (waiting too long).
- **Semantic accuracy** (task success impact): sometimes a small WER change causes a large tool-call argument error; measure downstream.

TTS metrics:
- **Time-to-first-audio** and **real-time factor** under streaming synthesis.
- **MOS** (human ratings) for naturalness and speaker similarity; StyleTTS2’s claims are anchored in MOS-style evaluations on LJSpeech/VCTK and adaptation experiments. citeturn17search4turn20search1turn20search6  
- **Barge-in behaviour**: ability to interrupt synthesis cleanly without audio artefacts.

Agentic metrics:
- **Goal success rate** (task completion), **tool success rate** (valid tool calls), and **safety violation rate** (policy/guardrail hits).
- **Cost per successful task** and **latency percentiles** (P50/P95 overall; plus TTFA).
- AgentCore explicitly frames production monitoring as token usage, latency, session duration, error rates, and continuous quality scoring (correctness/helpfulness/safety/goal success). citeturn12view0

### Latency engineering considerations

A practical end-to-end latency budget for “conversational” feel often targets “sub‑second to first audio” in common cases, but you should engineer to **P95** and treat network variability as a first-class problem. The 2026 enterprise tutorial provides a concrete example: with streaming STT + streaming LLM generation + streaming TTS, they report P50 time-to-first-audio under 1 second (best-case ~729 ms), reinforcing that careful pipelining is decisive. citeturn14academia38

Key engineering tactics:
- Run **VAD/turn detection** as close to the microphone as possible to reduce upstream buffering.
- Use **partial transcript gating**: allow the LLM to start thinking, but delay tool execution until stable user intent is confirmed.
- Prefer **short, incremental responses** (and progressively refine) for conversational UX.
- Make TTS **interruptible** by design: treat audio playback as a cancellable stream and stop synthesis immediately on user barge‑in.
- Cache frequent phrases and “boilerplate acknowledgements” at the TTS layer (some frameworks support caching patterns; implement carefully to avoid stale or unsafe responses). citeturn14search13

### Cost considerations

Even with “no budget constraints”, cost still matters because it interacts with latency (bigger models → higher latency) and determines what you can afford at scale.

In AWS-centric architectures, the main recurring cost drivers are:
- **STT minutes** (Amazon Transcribe pricing varies by usage tiers and features; consult the pricing page for the latest rates and add-ons). citeturn6search2turn10search2  
- **TTS characters** (Polly per‑million character pricing differs by voice tier, with Generative voices priced separately). citeturn6search1  
- **LLM tokens** (Bedrock per‑token pricing varies widely by model family and throughput mode). citeturn6search3turn6search19  
- **Realtime transport** (WebRTC relays/TURN, media egress), especially at high concurrency.

A useful financial control is *model routing*:
- cheaper/faster models for boilerplate turns,
- larger models for high-stakes tool decisions,
- and deferred batch reasoning for long-running tasks (AgentCore Runtime supports long workloads up to 8 hours when needed). citeturn13view3turn12view0

### Safety and privacy controls

Voice assistants should be designed with explicit controls for:
- **Data minimisation**: store transcripts only when needed; store summaries/preferences rather than raw audio; give users control over memory.
- **Isolation**: AgentCore Runtime’s per-session microVM isolation and memory sanitisation after termination provides a strong primitive against cross-session leakage. citeturn13view3
- **Network containment**: AgentCore advertises VPC connectivity and PrivateLink support for controlled network access paths. citeturn12view0
- **Tool boundary enforcement**: enforce tool access with deterministic policies outside the model (AgentCore Policy intercepts tool calls via Gateway). citeturn13view2turn13view1
- **Consent and anti-impersonation for voice cloning**: gated custom voice programmes (e.g., Azure Custom Voice access limitations) reflect the risk profile; adopt comparable consent verification, disclosure, and watermarking/detection strategies where legally appropriate. citeturn5search2
- **Device/security hygiene**: for device fleets, IoT Core security best practices (TLS, per-device identity, secure auth patterns) reduce risk of device impersonation in voice control scenarios. citeturn7search14turn7search2

For governance and evaluation loops, invest in:
- red-team datasets (“prompt injection spoken aloud”, overlapping speakers, accented speech, code-switching),
- policy simulation tests (tool calls under adversarial prompts),
- and continuous monitoring (AgentCore evaluation/observability concepts map naturally to this). citeturn12view0turn13view2