## Task 010: Voice cloning candidate selection (license + feasibility)

**Status**: Planned  
**Priority**: P0 (blocks Task 009 implementation)  

---

## Goal

Pick a single candidate for `abstractvoice[cloning]` that satisfies:

- **Permissive license** (MIT/Apache 2.0) for *code and model weights*
- **Practical install** story (documented; ideally wheels; accept “GPU required” as optional extra)
- **Near‑realtime** behavior (or honest constraints)
- Clear language scope (start English-only if needed; don’t overpromise)

---

## Work items

1. **License verification**
   - Confirm code license and model/weights license (HF card, repo license, usage terms)
   - Record evidence links in the completion report

2. **Install + runtime validation**
   - Minimal install steps
   - CPU vs GPU baseline requirements
   - Approx latency / real-time factor

3. **Decision**
   - Select one candidate and justify tradeoffs
   - Update `docs/voice_cloning_2026.md` with the choice

