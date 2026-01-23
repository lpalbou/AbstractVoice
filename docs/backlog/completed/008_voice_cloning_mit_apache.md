## Task 008: Near‑Realtime Voice Cloning (MIT/Apache) Investigation + Plan

**Status**: ✅ Completed (investigation + plan)  
**Completed**: 2026-01-23  
**Priority**: P1

---

## Result

Produced a consolidated feasibility report:

- `docs/voice_cloning_2026.md`

Key findings:
- **XTTS‑v2** is CPML (non‑permissive) → rejected.
- There are permissive candidates (**Apache 2.0 / MIT**) but most are **GPU/torch-heavy**,
  so cloning should be an **optional extra** (`abstractvoice[cloning]`).
- **OpenVoice** is license-ambiguous in the wild (MIT label vs non‑commercial claims) → do not adopt unless clarified.

---

## Next step

Implementation work is tracked separately:
- `docs/backlog/planned/009_voice_cloning_adapter_impl.md`

