## Planned Scenema track

This folder groups the accepted Scenema-originated backlog track.

Promotion history:

- Promoted from `docs/backlog/proposed/scenema/` to
  `docs/backlog/planned/scenema/` on 2026-05-20.

Current branch state:

- `044` is no longer greenfield contract work. The internal
  `SpeechRequest`/capability seam and truthful package-owned TTS capability
  export now exist on this branch.
- `045` is no longer “create the foundation” work. Shared torch runtime
  resolution and explicit fallback reporting now exist and have already been
  applied to multiple advanced runtimes.
- `046` remains the Scenema-specific item, but it is now a bounded
  Linux/CUDA proof-of-load spike rather than a catch-all rewrite program.
- `047` stays as the Apple Silicon validation and explicit fallback follow-on
  after `046`.
- `048` was folded into `044` once ADR 0007 and the package-owned request
  layer landed, so it is no longer tracked as a standalone planned item.

Recommended implementation order:

1. `044_package_owned_directed_speech_request_and_capabilities.md`
2. `045_shared_runtime_foundation_for_advanced_speech_engines.md`
3. `046_scenema_class_python_runtime_spike.md`
4. `047_device_agnostic_runtime_and_apple_silicon_path.md`

Why this folder exists:

- keep the Scenema-originated planned work readable as one track;
- preserve the track's proposal-to-planned history without duplicating files;
- avoid spreading closely related architecture notes across the top-level
  `planned/` directory;
- make it explicit that only `046` is Scenema-specific while the rest of the
  track remains reusable architecture work.
