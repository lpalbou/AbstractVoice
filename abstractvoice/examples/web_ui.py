#!/usr/bin/env python3
"""Local FastAPI example that embeds AbstractVoice.

This is a readable browser smoke-test around `VoiceManager`, not the production
HTTP surface for AbstractFramework. AbstractCore owns production
OpenAI-compatible audio endpoints; the `/v1/audio/*` routes here are local
compatibility aliases that map to the same `VoiceManager` calls.

The LLM panel is equally example-only: it forwards one non-streaming
OpenAI-compatible chat request to a local provider such as Ollama or LM Studio.
It is intentionally not an agent runtime or a production chat server.
"""

import argparse
import atexit
import importlib.metadata
import importlib.util
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from abstractvoice.examples.llm_provider import DEFAULT_MODEL, DEFAULT_PROVIDER, resolve_provider


LOCAL_ROUTES = [
    {
        "method": "GET",
        "path": "/api/status",
        "maps_to": "ExampleState status; does not initialize VoiceManager",
    },
    {
        "method": "GET",
        "path": "/api/voices",
        "maps_to": "VoiceManager.get_profiles(), list_available_models(), list_cloned_voices()",
    },
    {
        "method": "GET",
        "path": "/v1/audio/voices",
        "maps_to": "OpenAI-compatible extension -> VoiceManager.get_profiles() + list_cloned_voices()",
    },
    {
        "method": "POST",
        "path": "/api/voices/select",
        "maps_to": "VoiceManager.set_profile() or local base/cloned voice selection; optional role=assistant|user",
    },
    {
        "method": "POST",
        "path": "/api/voices/clone",
        "maps_to": "Example-only cloned-voice creation from uploaded browser audio; stores via VoiceManager.clone_voice() and validates with a short synthesis by default",
    },
    {
        "method": "POST",
        "path": "/v1/voice/clone",
        "maps_to": "OpenAI-compatible extension -> VoiceManager.clone_voice(); returns a voice_id for later /v1/audio/speech voice use",
    },
    {
        "method": "POST",
        "path": "/api/tts",
        "maps_to": "VoiceManager.speak_to_bytes(text, format, voice); optional local role chooses default voice",
    },
    {
        "method": "POST",
        "path": "/api/stt/transcriptions",
        "maps_to": "VoiceManager.transcribe_file()",
    },
    {
        "method": "POST",
        "path": "/api/stt/transcribe",
        "maps_to": "Compatibility alias for /api/stt/transcriptions",
    },
    {
        "method": "GET",
        "path": "/api/llm/models",
        "maps_to": "Example-only OpenAI-compatible model listing; does not initialize VoiceManager",
    },
    {
        "method": "POST",
        "path": "/api/chat",
        "maps_to": "Example-only OpenAI-compatible chat completion; browser appends response to the discussion",
    },
    {
        "method": "POST",
        "path": "/v1/audio/speech",
        "maps_to": "OpenAI-compatible local alias -> VoiceManager.speak_to_bytes()",
    },
    {
        "method": "POST",
        "path": "/v1/audio/transcriptions",
        "maps_to": "OpenAI-compatible local alias -> VoiceManager.transcribe_file()",
    },
]


PAGE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AbstractVoice Local</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #19232d;
      --muted: #637083;
      --line: #d6dde6;
      --accent: #2457a6;
      --accent-ink: #ffffff;
      --warm: #b65f2a;
      --ok: #26734d;
      --danger: #a64242;
      --shadow: 0 12px 30px rgba(25, 35, 45, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    .wrap {
      width: min(1180px, calc(100vw - 28px));
      margin: 0 auto;
    }
    .topbar {
      min-height: 62px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      font-weight: 760;
    }
    h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.2;
      font-weight: 760;
    }
    main {
      padding: 18px 0 26px;
      display: grid;
      grid-template-columns: minmax(0, 1.55fr) minmax(320px, 0.85fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel-head {
      min-height: 48px;
      padding: 13px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .panel-body { padding: 14px; }
    .stack { display: grid; gap: 12px; }
    .grid2 {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label {
      display: grid;
      gap: 5px;
      min-width: 0;
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
    }
    textarea,
    input,
    select {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      color: var(--ink);
      padding: 8px 10px;
      font: inherit;
      letter-spacing: 0;
    }
    textarea {
      min-height: 104px;
      resize: vertical;
    }
    textarea:focus,
    input:focus,
    select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(36, 87, 166, 0.14);
      outline: none;
    }
    button,
    .download {
      min-height: 38px;
      border: 1px solid var(--accent);
      border-radius: 8px;
      background: var(--accent);
      color: var(--accent-ink);
      padding: 8px 12px;
      font: inherit;
      font-weight: 720;
      text-decoration: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
    }
    button.secondary,
    .download {
      background: #fff;
      color: var(--accent);
    }
    button.warm { border-color: var(--warm); background: var(--warm); }
    button:disabled,
    .download[aria-disabled="true"] {
      opacity: 0.55;
      cursor: not-allowed;
    }
    button.loading::before {
      content: "";
      width: 14px;
      height: 14px;
      margin-right: 7px;
      border: 2px solid rgba(255, 255, 255, 0.48);
      border-top-color: currentColor;
      border-radius: 999px;
      animation: spin 0.75s linear infinite;
    }
    button.secondary.loading::before {
      border-color: #d7e0ea;
      border-top-color: currentColor;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    button.icon-button,
    .play-one {
      width: 38px;
      min-width: 38px;
      height: 38px;
      padding: 0;
      border-color: #8aa0bd;
      background: #fff;
      color: var(--accent);
    }
    button.icon-button.warm {
      border-color: var(--warm);
      background: var(--warm);
      color: var(--accent-ink);
    }
    .icon {
      font-size: 17px;
      line-height: 1;
    }
    .status {
      color: var(--muted);
      overflow-wrap: anywhere;
      text-align: right;
    }
    .status.ok,
    .message.ok { color: var(--ok); }
    .message.error { color: var(--danger); }
    .message {
      min-height: 20px;
      margin: 0;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    .thread {
      height: min(56vh, 540px);
      min-height: 360px;
      overflow: auto;
      padding: 14px;
      background: #eef2f5;
      border-bottom: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .bubble-row {
      display: grid;
      grid-template-columns: 40px minmax(0, max-content);
      gap: 8px;
      align-items: end;
      max-width: 82%;
    }
    .bubble-row.user {
      grid-template-columns: minmax(0, max-content) 40px;
      margin-left: auto;
    }
    .bubble {
      max-width: min(100%, 560px);
      min-width: 0;
      padding: 9px 11px;
      border: 1px solid #cfd8e3;
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .message-spinner {
      display: none;
      width: 15px;
      height: 15px;
      margin-left: 8px;
      border: 2px solid #d7e0ea;
      border-top-color: var(--warm);
      border-radius: 999px;
      animation: spin 0.75s linear infinite;
      vertical-align: -2px;
    }
    .bubble-row.synthesizing .message-spinner {
      display: inline-block;
    }
    .bubble-row.user .bubble {
      background: #dbe8ff;
      border-color: #bfd2f3;
    }
    .bubble-row.active .bubble {
      border-color: var(--warm);
      box-shadow: 0 0 0 3px rgba(182, 95, 42, 0.16);
    }
    .bubble-row.user .play-one { order: 2; }
    .composer {
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    textarea.compact { min-height: 76px; }
    .checkbox-line {
      min-height: 38px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
    }
    .checkbox-line input {
      width: auto;
      min-height: 0;
      margin: 0;
    }
    .hidden {
      display: none !important;
    }
    audio {
      width: 100%;
      min-height: 42px;
    }
    .transcript {
      min-height: 92px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
      padding: 10px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .busy-overlay {
      position: fixed;
      inset: 0;
      z-index: 30;
      display: grid;
      place-items: center;
      padding: 18px;
      background: rgba(244, 246, 248, 0.68);
      backdrop-filter: blur(6px);
    }
    .busy-overlay[hidden] { display: none; }
    body.busy main,
    body.busy header {
      filter: blur(1.4px);
      pointer-events: none;
      user-select: none;
    }
    .busy-card {
      width: min(360px, calc(100vw - 32px));
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 18px;
      display: grid;
      gap: 10px;
      text-align: center;
    }
    .spinner {
      width: 42px;
      height: 42px;
      margin: 0 auto 2px;
      border: 4px solid #d7e0ea;
      border-top-color: var(--accent);
      border-radius: 999px;
      animation: spin 0.8s linear infinite;
    }
    .busy-title {
      font-weight: 760;
      font-size: 15px;
    }
    .busy-detail {
      margin: 0;
      color: var(--muted);
      overflow-wrap: anywhere;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @media (max-width: 880px) {
      .topbar,
      main { display: block; }
      .topbar { padding: 12px 0; }
      .status { text-align: left; margin-top: 8px; }
      .panel + .panel { margin-top: 14px; }
      .thread { height: 48vh; min-height: 310px; }
      .bubble-row { max-width: 92%; }
      .grid2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <h1>AbstractVoice Local</h1>
      <div id="status" class="status">Starting</div>
    </div>
  </header>

  <main class="wrap">
    <section class="panel">
      <div class="panel-head">
        <h2>Discussion</h2>
        <div class="actions">
          <button id="conversation-toggle" type="button" class="icon-button" title="Play conversation" aria-label="Play conversation"><span class="icon">&#9654;</span></button>
          <button id="stop-read" type="button" class="icon-button warm" title="Stop playback" aria-label="Stop playback"><span class="icon">&#9632;</span></button>
          <button id="clear-chat" type="button" class="secondary">Clear</button>
        </div>
      </div>
      <div id="thread" class="thread"></div>
      <div class="composer">
        <div class="grid2">
          <label>Speaker
            <select id="speaker">
              <option value="assistant">Assistant</option>
              <option value="user">User</option>
            </select>
          </label>
          <label>Language
            <input id="language" value="en" autocomplete="off">
          </label>
        </div>
        <label>Message
          <textarea id="message-text">First item: keep the local API small and mapped to VoiceManager.</textarea>
        </label>
        <div class="actions">
          <button id="add-message" type="button">Add Message</button>
          <button id="ask-assistant" type="button">Ask Assistant</button>
          <p id="discussion-message" class="message">Ready.</p>
        </div>
      </div>
    </section>

    <aside class="stack">
      <section class="panel">
        <div class="panel-head"><h2>LLM</h2></div>
        <div class="panel-body stack">
          <div class="grid2">
            <label>Provider
              <input id="llm-provider" value="ollama" autocomplete="off">
            </label>
            <label>Model
              <input id="llm-model" value="gemma3:1b" list="llm-models" autocomplete="off">
              <datalist id="llm-models"></datalist>
            </label>
          </div>
          <div class="grid2">
            <label>Temperature
              <input id="llm-temperature" type="number" min="0" max="2" step="0.05" value="0.4">
            </label>
            <label>Max Tokens
              <input id="llm-max-tokens" type="number" min="32" max="8192" step="32" value="1024">
            </label>
          </div>
          <label>System Prompt
            <textarea id="system-prompt" class="compact">You are a helpful voice assistant. Keep replies short and conversational unless asked for detail.</textarea>
          </label>
          <div class="actions">
            <button id="refresh-models" type="button" class="secondary">Models</button>
            <label class="checkbox-line"><input id="chat-speak" type="checkbox" checked> Speak Reply</label>
            <p id="llm-message" class="message">Ready.</p>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Voice</h2></div>
        <div class="panel-body stack">
          <div class="grid2">
            <label>Assistant Voice
              <select id="assistant-voice-choice">
                <option value="base">Base TTS</option>
              </select>
            </label>
            <label>User Voice
              <select id="user-voice-choice">
                <option value="base">Base TTS</option>
              </select>
            </label>
          </div>
          <label>Base TTS Profile
            <select id="profile-choice">
              <option value="">Default</option>
            </select>
          </label>
          <label>Speed
            <input id="speed" type="number" min="0.5" max="2" step="0.05" value="1">
          </label>
          <div class="actions">
            <button id="refresh-voices" type="button" class="secondary">Refresh</button>
            <p id="voice-message" class="message">Ready.</p>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Voice Cloning</h2></div>
        <div class="panel-body stack">
          <div class="grid2">
            <label>Name
              <input id="clone-name" value="my_voice" autocomplete="off">
            </label>
            <label>Engine
              <select id="clone-engine">
                <option value="f5_tts">OpenF5</option>
                <option value="chroma">Chroma</option>
                <option value="audiodit">AudioDiT</option>
                <option value="omnivoice">OmniVoice</option>
                <option value="openai-compatible">OpenAI-compatible Remote</option>
                <option value="openai">OpenAI Remote</option>
              </select>
            </label>
          </div>
          <label>Reference Audio
            <input id="clone-file" type="file" accept=".wav,.flac,.ogg,.mp3,.m4a,.webm,.aac,audio/*">
          </label>
          <label>Reference Text
            <textarea id="clone-reference-text" class="compact"></textarea>
          </label>
          <audio id="clone-preview" class="hidden" controls></audio>
          <div class="actions">
            <button id="record-clone" type="button" class="secondary">Record</button>
            <button id="stop-clone-recording" type="button" class="secondary" disabled>Stop</button>
            <button id="clone-voice" type="button">Clone Voice</button>
            <p id="clone-message" class="message">Ready.</p>
          </div>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Text To Speech</h2></div>
        <div class="panel-body stack">
          <label>Speaker Voice
            <select id="tts-role">
              <option value="assistant">Assistant</option>
              <option value="user">User</option>
            </select>
          </label>
          <label>Text
            <textarea id="tts-text">Hello from AbstractVoice.</textarea>
          </label>
          <div class="actions">
            <button id="speak" type="button">Speak</button>
            <a id="download" class="download" aria-disabled="true">Download WAV</a>
          </div>
          <audio id="audio" controls></audio>
          <p id="tts-message" class="message">Ready.</p>
        </div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2>Speech To Text</h2></div>
        <div class="panel-body stack">
          <label>Audio File
            <input id="file" type="file" accept="audio/*">
          </label>
          <div class="actions">
            <button id="transcribe" type="button" class="secondary">Transcribe</button>
          </div>
          <div id="transcript" class="transcript"></div>
          <p id="stt-message" class="message">Ready.</p>
        </div>
      </section>
    </aside>
  </main>

  <div id="busy-overlay" class="busy-overlay" hidden aria-live="polite" aria-busy="true">
    <div class="busy-card" role="status">
      <div class="spinner"></div>
      <div id="busy-title" class="busy-title">Working</div>
      <p id="busy-detail" class="busy-detail">Preparing audio.</p>
    </div>
  </div>

  <script>
    const statusEl = document.getElementById("status");
    const thread = document.getElementById("thread");
    const discussionMessage = document.getElementById("discussion-message");
    const voiceMessage = document.getElementById("voice-message");
    const cloneMessage = document.getElementById("clone-message");
    const llmMessage = document.getElementById("llm-message");
    const ttsMessage = document.getElementById("tts-message");
    const sttMessage = document.getElementById("stt-message");
    const audio = document.getElementById("audio");
    const download = document.getElementById("download");
    const conversationToggle = document.getElementById("conversation-toggle");
    const assistantVoiceChoice = document.getElementById("assistant-voice-choice");
    const userVoiceChoice = document.getElementById("user-voice-choice");
    const profileChoice = document.getElementById("profile-choice");
    const languageInput = document.getElementById("language");
    const speedInput = document.getElementById("speed");
    const providerInput = document.getElementById("llm-provider");
    const modelInput = document.getElementById("llm-model");
    const modelOptions = document.getElementById("llm-models");
    const temperatureInput = document.getElementById("llm-temperature");
    const maxTokensInput = document.getElementById("llm-max-tokens");
    const systemPromptInput = document.getElementById("system-prompt");
    const chatSpeakInput = document.getElementById("chat-speak");
    const cloneNameInput = document.getElementById("clone-name");
    const cloneEngineInput = document.getElementById("clone-engine");
    const cloneFileInput = document.getElementById("clone-file");
    const cloneReferenceTextInput = document.getElementById("clone-reference-text");
    const clonePreview = document.getElementById("clone-preview");
    const cloneRecordButton = document.getElementById("record-clone");
    const cloneStopButton = document.getElementById("stop-clone-recording");
    const cloneVoiceButton = document.getElementById("clone-voice");
    const transcript = document.getElementById("transcript");
    const busyOverlay = document.getElementById("busy-overlay");
    const busyTitle = document.getElementById("busy-title");
    const busyDetail = document.getElementById("busy-detail");

    const ICON_PLAY = "&#9654;";
    const ICON_PAUSE = "&#10073;&#10073;";
    // Browser-only playback state. Message buttons play one item; the
    // conversation button loops through the thread and pauses the active audio.
    let busyDepth = 0;
    let lastAudioUrl = null;
    let reading = false;
    let activeIndex = -1;
    let playbackMode = null;
    let playbackToken = 0;
    let aborter = null;
    let playResolve = null;
    let cloneRecorder = null;
    let cloneRecorderStream = null;
    let cloneRecordingChunks = [];
    let recordedCloneBlob = null;
    let recordedCloneUrl = null;
    const messages = [
      {role: "assistant", text: "Morning. I saved the notes from the design review."},
      {role: "user", text: "Great. Can you read them back in order?"},
      {role: "assistant", text: "First item: keep voice selection clear. Second item: keep the HTTP example mapped to VoiceManager."}
    ];
    let optionalDependencies = {};

    function setMessage(el, text, kind) {
      el.textContent = text;
      el.classList.remove("ok", "error");
      if (kind) el.classList.add(kind);
    }

    function roleLabel(role) {
      return role === "user" ? "user" : "assistant";
    }

    function setIcon(button, html) {
      if (!button) return;
      button.innerHTML = '<span class="icon">' + html + '</span>';
    }

    function setButtonLoading(button, loading, label) {
      if (!button) return;
      if (loading) {
        if (!button.dataset.idleText) button.dataset.idleText = button.textContent;
        if (label) button.textContent = label;
        button.classList.add("loading");
        return;
      }
      button.classList.remove("loading");
      if (button.dataset.idleText) {
        button.textContent = button.dataset.idleText;
        delete button.dataset.idleText;
      }
    }

    function beginButtonLoading(button, label, delayMs) {
      let active = true;
      const timer = window.setTimeout(() => {
        if (active) setButtonLoading(button, true, label);
      }, Number.isFinite(delayMs) ? delayMs : 140);
      return () => {
        active = false;
        window.clearTimeout(timer);
        setButtonLoading(button, false);
      };
    }

    function updatePlaybackControls() {
      const conversationPlaying = playbackMode === "conversation" && reading && !audio.paused;
      setIcon(conversationToggle, conversationPlaying ? ICON_PAUSE : ICON_PLAY);
      conversationToggle.title = conversationPlaying ? "Pause conversation" : "Play conversation";
      conversationToggle.setAttribute("aria-label", conversationToggle.title);

      for (const button of thread.querySelectorAll("[data-play-index]")) {
        const index = Number.parseInt(button.dataset.playIndex || "-1", 10);
        const active = index === activeIndex && (playbackMode === "message" || playbackMode === "conversation") && !audio.paused;
        setIcon(button, active ? ICON_PAUSE : ICON_PLAY);
        button.title = active ? "Pause message" : "Play message";
        button.setAttribute("aria-label", button.title + " " + (index + 1));
      }
    }

    function toggleCurrentAudio(label) {
      if (!audio.src) return false;
      if (audio.paused) {
        audio.play().catch(() => {});
        setMessage(discussionMessage, "Resumed " + label + ".", "");
      } else {
        audio.pause();
        setMessage(discussionMessage, "Paused " + label + ".", "");
      }
      updatePlaybackControls();
      return true;
    }

    function showBusy(title, detail) {
      busyDepth += 1;
      busyTitle.textContent = title || "Working";
      busyDetail.textContent = detail || "Preparing audio.";
      busyOverlay.hidden = false;
      document.body.classList.add("busy");
    }

    function hideBusy() {
      busyDepth = Math.max(0, busyDepth - 1);
      if (busyDepth === 0) {
        busyOverlay.hidden = true;
        document.body.classList.remove("busy");
      }
    }

    async function withBusy(title, detail, task, delayMs) {
      let shown = false;
      const timer = window.setTimeout(() => {
        shown = true;
        showBusy(title, detail);
      }, Number.isFinite(delayMs) ? delayMs : 220);
      try {
        return await task();
      } finally {
        window.clearTimeout(timer);
        if (shown) hideBusy();
      }
    }

    function errorText(data, fallback) {
      if (!data) return fallback;
      if (typeof data.detail === "string") return data.detail;
      if (typeof data.error === "string") return data.error;
      return fallback;
    }

    async function fetchJson(url, options) {
      const res = await fetch(url, options);
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(errorText(data, "Request failed."));
      return data;
    }

    function voiceSelectForRole(role) {
      return role === "user" ? userVoiceChoice : assistantVoiceChoice;
    }

    function selectedVoiceForRole(role) {
      const select = voiceSelectForRole(role);
      return select ? select.value : "base";
    }

    function optionEngineInstalled(engine) {
      const key = String(engine || "").trim().toLowerCase();
      if (!key || !optionalDependencies[key]) return true;
      return optionalDependencies[key].installed !== false;
    }

    function updateOptionalEngineUi() {
      for (const opt of cloneEngineInput.options) {
        const installed = optionEngineInstalled(opt.value);
        opt.disabled = !installed;
        const base = opt.dataset.baseLabel || opt.textContent.replace(" (not installed)", "");
        opt.dataset.baseLabel = base;
        opt.textContent = installed ? base : base + " (not installed)";
      }
      if (cloneEngineInput.selectedOptions[0] && cloneEngineInput.selectedOptions[0].disabled) {
        const firstReady = Array.from(cloneEngineInput.options).find((opt) => !opt.disabled);
        if (firstReady) cloneEngineInput.value = firstReady.value;
      }
    }

    function updateProfileEngineUi() {
      for (const opt of profileChoice.options) {
        const engine = opt.dataset.engine || "";
        if (!engine) continue;
        const installed = optionEngineInstalled(engine);
        opt.disabled = !installed;
        const base = opt.dataset.baseLabel || opt.textContent.replace(" (not installed)", "");
        opt.dataset.baseLabel = base;
        opt.textContent = installed ? base : base + " (not installed)";
      }
      if (profileChoice.selectedOptions[0] && profileChoice.selectedOptions[0].disabled) profileChoice.value = "";
    }

    function audioPayload(text, role) {
      const speakerRole = roleLabel(role);
      const voice = selectedVoiceForRole(speakerRole);
      const speed = Number.parseFloat(speedInput.value || "1");
      const payload = {
        input: text,
        response_format: "wav",
        language: languageInput.value.trim() || "en",
        role: speakerRole
      };
      if (Number.isFinite(speed)) payload.speed = speed;
      if (voice && voice !== "base") payload.voice = voice;
      return payload;
    }

    async function synthesize(text, signal, role) {
      const res = await fetch("/api/tts", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(audioPayload(text, role)),
        signal
      });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(errorText(data, "TTS failed."));
      }
      return res.blob();
    }

    function clearActive() {
      activeIndex = -1;
      for (const row of thread.querySelectorAll(".bubble-row")) row.classList.remove("active");
    }

    function setActive(index) {
      clearActive();
      activeIndex = index;
      const row = thread.querySelector(`[data-index="${index}"]`);
      if (row) {
        row.classList.add("active");
        row.scrollIntoView({block: "nearest", behavior: "smooth"});
      }
    }

    function setMessageSynthesizing(index, loading) {
      const row = thread.querySelector(`[data-index="${index}"]`);
      if (!row) return;
      row.classList.toggle("synthesizing", Boolean(loading));
      if (loading) {
        row.setAttribute("aria-busy", "true");
      } else {
        row.removeAttribute("aria-busy");
      }
    }

    function beginMessageSynthesizing(index) {
      let active = true;
      const timer = window.setTimeout(() => {
        if (active) setMessageSynthesizing(index, true);
      }, 140);
      return () => {
        active = false;
        window.clearTimeout(timer);
        setMessageSynthesizing(index, false);
      };
    }

    function clearMessageSynthesizing() {
      for (const row of thread.querySelectorAll(".bubble-row.synthesizing")) {
        row.classList.remove("synthesizing");
        row.removeAttribute("aria-busy");
      }
    }

    function stopPlayback(message) {
      if (aborter) aborter.abort();
      playbackToken += 1;
      aborter = null;
      reading = false;
      playbackMode = null;
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      if (playResolve) {
        const resolve = playResolve;
        playResolve = null;
        resolve();
      }
      clearMessageSynthesizing();
      clearActive();
      updatePlaybackControls();
      if (message !== false) setMessage(discussionMessage, message || "Stopped.", "");
    }

    function playBlob(blob) {
      return new Promise((resolve) => {
        if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
        lastAudioUrl = URL.createObjectURL(blob);
        audio.src = lastAudioUrl;
        download.href = lastAudioUrl;
        download.download = "abstractvoice.wav";
        download.removeAttribute("aria-disabled");

        const done = () => {
          audio.removeEventListener("ended", done);
          audio.removeEventListener("error", done);
          if (playResolve === done) playResolve = null;
          updatePlaybackControls();
          resolve();
        };
        playResolve = done;
        audio.addEventListener("ended", done, {once: true});
        audio.addEventListener("error", done, {once: true});
        audio.play().then(updatePlaybackControls).catch(done);
      });
    }

    async function playMessage(index, mode) {
      const item = messages[index];
      if (!item) return;
      const speakerRole = roleLabel(item.role);
      aborter = new AbortController();
      playbackMode = mode || "message";
      setActive(index);
      updatePlaybackControls();
      setMessage(discussionMessage, "Preparing " + speakerRole + " message " + (index + 1) + ".", "");
      const finishMessageLoading = beginMessageSynthesizing(index);
      let blob;
      try {
        blob = await synthesize(item.text, aborter.signal, speakerRole);
      } finally {
        finishMessageLoading();
      }
      setMessage(discussionMessage, "Reading " + speakerRole + " message " + (index + 1) + ".", "");
      await playBlob(blob);
      aborter = null;
    }

    async function playSingleMessage(index) {
      if (activeIndex === index && (playbackMode === "message" || playbackMode === "conversation") && audio.src) {
        toggleCurrentAudio("message");
        return;
      }
      if (reading || playbackMode) stopPlayback(false);
      const token = ++playbackToken;
      reading = true;
      playbackMode = "message";
      try {
        await playMessage(index, "message");
        if (reading && token === playbackToken) setMessage(discussionMessage, "Done.", "ok");
      } catch (err) {
        if (token === playbackToken && err.name !== "AbortError") setMessage(discussionMessage, err.message || String(err), "error");
      } finally {
        if (token === playbackToken) {
          reading = false;
          aborter = null;
          playbackMode = null;
          clearActive();
          updatePlaybackControls();
        }
      }
    }

    async function readConversation(start) {
      if (playbackMode === "conversation" && reading) {
        toggleCurrentAudio("conversation");
        return;
      }
      if (reading || playbackMode) stopPlayback(false);
      const token = ++playbackToken;
      reading = true;
      playbackMode = "conversation";
      updatePlaybackControls();
      try {
        for (let i = start; i < messages.length; i += 1) {
          if (!reading || token !== playbackToken) break;
          await playMessage(i, "conversation");
        }
        if (reading && token === playbackToken) setMessage(discussionMessage, "Done.", "ok");
      } catch (err) {
        if (token === playbackToken && err.name !== "AbortError") setMessage(discussionMessage, err.message || String(err), "error");
      } finally {
        if (token === playbackToken) {
          reading = false;
          aborter = null;
          playbackMode = null;
          clearActive();
          updatePlaybackControls();
        }
      }
    }

    function renderMessages() {
      thread.innerHTML = "";
      messages.forEach((item, index) => {
        const row = document.createElement("div");
        row.className = "bubble-row " + item.role;
        row.dataset.index = String(index);

        const button = document.createElement("button");
        button.type = "button";
        button.className = "play-one";
        button.dataset.playIndex = String(index);
        button.title = "Play message";
        button.setAttribute("aria-label", "Play message " + (index + 1));
        setIcon(button, ICON_PLAY);
        button.addEventListener("click", () => playSingleMessage(index));

        const bubble = document.createElement("div");
        bubble.className = "bubble";
        const bubbleText = document.createElement("span");
        bubbleText.textContent = item.text;
        const spinner = document.createElement("span");
        spinner.className = "message-spinner";
        spinner.setAttribute("aria-hidden", "true");
        bubble.appendChild(bubbleText);
        bubble.appendChild(spinner);

        if (item.role === "user") {
          row.appendChild(bubble);
          row.appendChild(button);
        } else {
          row.appendChild(button);
          row.appendChild(bubble);
        }
        thread.appendChild(row);
      });
      if (activeIndex >= 0) setActive(activeIndex);
      updatePlaybackControls();
    }

    function clearChat() {
      if (reading || playbackMode) stopPlayback(false);
      messages.length = 0;
      activeIndex = -1;
      renderMessages();
      setMessage(discussionMessage, "Chat cleared.", "ok");
    }

    async function refreshStatus() {
      try {
        const data = await fetchJson("/api/status");
        statusEl.textContent = data.voice_manager_initialized ? "VoiceManager loaded" : "Ready";
        statusEl.classList.add("ok");
        if (data.defaults && data.defaults.language) languageInput.value = data.defaults.language;
        optionalDependencies = data.optional_dependencies || {};
        updateOptionalEngineUi();
        updateProfileEngineUi();
      } catch (_) {
        statusEl.textContent = "Server unavailable";
        statusEl.classList.remove("ok");
      }
    }

    function buildVoiceOptions(select, selectedVoice, clonedVoices) {
      select.innerHTML = "";
      const base = document.createElement("option");
      base.value = "base";
      base.textContent = "Base TTS";
      select.appendChild(base);
      for (const item of clonedVoices || []) {
        const opt = document.createElement("option");
        opt.value = item.id || item.voice_id;
        opt.textContent = (item.name || item.id || item.voice_id) + (item.engine ? " (" + item.engine + ")" : "");
        select.appendChild(opt);
      }
      select.value = Array.from(select.options).some((o) => o.value === selectedVoice) ? selectedVoice : "base";
    }

    async function refreshVoices() {
      try {
        const data = await fetchJson("/api/voices");
        const roleVoices = (data.current && data.current.role_voices) || {};
        const assistantSelected = roleVoices.assistant || assistantVoiceChoice.value || "base";
        const userSelected = roleVoices.user || userVoiceChoice.value || "base";
        buildVoiceOptions(assistantVoiceChoice, assistantSelected, data.cloned_voices || []);
        buildVoiceOptions(userVoiceChoice, userSelected, data.cloned_voices || []);

        const selectedProfile = profileChoice.value;
        profileChoice.innerHTML = "";
        const def = document.createElement("option");
        def.value = "";
        def.textContent = "Default";
        profileChoice.appendChild(def);
        for (const item of data.profiles || []) {
          const opt = document.createElement("option");
          opt.value = item.id;
          opt.textContent = item.label || item.id;
          opt.dataset.baseLabel = opt.textContent;
          if (item.engine) {
            opt.dataset.engine = item.engine;
            if (!optionEngineInstalled(item.engine)) {
              opt.disabled = true;
              opt.textContent += " (not installed)";
            }
          }
          if (item.active) opt.selected = true;
          profileChoice.appendChild(opt);
        }
        if (selectedProfile && Array.from(profileChoice.options).some((o) => o.value === selectedProfile && !o.disabled)) {
          profileChoice.value = selectedProfile;
        }
        if (profileChoice.selectedOptions[0] && profileChoice.selectedOptions[0].disabled) profileChoice.value = "";
        profileChoice.disabled = profileChoice.options.length <= 1;
        setMessage(voiceMessage, "Voices refreshed.", "ok");
        refreshStatus();
      } catch (err) {
        setMessage(voiceMessage, err.message || String(err), "error");
      }
    }

    async function selectRoleVoice(role) {
      const speakerRole = roleLabel(role);
      const select = voiceSelectForRole(speakerRole);
      const value = select.value;
      const isBase = value === "base";
      const title = isBase ? "Switching voice" : "Loading cloned voice";
      const detail = isBase
        ? "Using base TTS for the " + speakerRole + " role."
        : "Loading the selected " + speakerRole + " cloned voice. First load can take a while.";
      try {
        setMessage(voiceMessage, detail, "");
        const request = () => fetchJson("/api/voices/select", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            role: speakerRole,
            kind: isBase ? "base" : "clone",
            voice: isBase ? null : value,
            preload: !isBase
          })
        });
        const data = isBase ? await request() : await withBusy(title, detail, request);
        const extra = data.preload && data.preload.seconds ? " Preloaded in " + data.preload.seconds.toFixed(1) + "s." : "";
        setMessage(voiceMessage, (speakerRole === "user" ? "User" : "Assistant") + " voice updated." + extra, "ok");
        refreshStatus();
      } catch (err) {
        await refreshVoices();
        const text = voiceSelectionErrorText(err.message || String(err), speakerRole);
        setMessage(voiceMessage, text, "error");
      }
    }

    function voiceSelectionErrorText(message, role) {
      let text = (role === "user" ? "User" : "Assistant") + " voice was not changed; the previous selection is still active. " + (message || "Selection failed.");
      const lower = text.toLowerCase();
      if (text.includes("OpenF5 artifacts")) text += " Prefetch: abstractvoice-prefetch --openf5, or REPL: /cloning_download f5_tts.";
      if (lower.includes("f5_tts") && lower.includes("not installed")) text += ' Install: pip install "abstractvoice[cloning]".';
      if (lower.includes("audiodit") && (lower.includes("not installed") || lower.includes("prefetch"))) text += ' Install: pip install "abstractvoice[audiodit]"; prefetch: abstractvoice-prefetch --audiodit.';
      if (lower.includes("omnivoice") || text.includes("No module named 'omnivoice'")) text += ' Install: pip install "abstractvoice[web-omnivoice]"; prefetch: abstractvoice-prefetch --omnivoice.';
      if (lower.includes("chroma") && lower.includes("artifacts")) text += " Prefetch: abstractvoice-prefetch --chroma.";
      return text;
    }

    async function selectProfile() {
      const value = profileChoice.value;
      if (!value) return;
      try {
        const detail = "Preparing the selected base TTS profile.";
        setMessage(voiceMessage, detail, "");
        await withBusy("Applying TTS profile", detail, () => fetchJson("/api/voices/select", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({kind: "profile", profile: value})
        }));
        setMessage(voiceMessage, "Profile applied.", "ok");
      } catch (err) {
        setMessage(voiceMessage, err.message || String(err), "error");
      }
    }

    async function speakSingle() {
      const text = document.getElementById("tts-text").value.trim();
      if (!text) {
        setMessage(ttsMessage, "Enter text first.", "error");
        return;
      }
      const role = document.getElementById("tts-role").value;
      if (reading || playbackMode) stopPlayback(false);
      const button = document.getElementById("speak");
      const finishButtonLoading = beginButtonLoading(button, "Preparing");
      button.disabled = true;
      setMessage(ttsMessage, "Synthesizing.", "");
      try {
        const blob = await synthesize(text, undefined, role);
        await playBlob(blob);
        setMessage(ttsMessage, "Audio ready.", "ok");
        refreshStatus();
      } catch (err) {
        setMessage(ttsMessage, err.message || String(err), "error");
      } finally {
        finishButtonLoading();
        button.disabled = false;
      }
    }

    async function transcribe() {
      const file = document.getElementById("file").files[0];
      if (!file) {
        setMessage(sttMessage, "Choose an audio file first.", "error");
        return;
      }
      const button = document.getElementById("transcribe");
      button.disabled = true;
      const finishButtonLoading = beginButtonLoading(button, "Transcribing");
      setMessage(sttMessage, "Transcribing.", "");
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("language", languageInput.value.trim() || "en");
      try {
        const data = await fetchJson("/api/stt/transcriptions", {method: "POST", body: form});
        transcript.textContent = data.text || "";
        setMessage(sttMessage, "Transcript ready.", "ok");
        refreshStatus();
      } catch (err) {
        setMessage(sttMessage, err.message || String(err), "error");
      } finally {
        finishButtonLoading();
        button.disabled = false;
      }
    }

    // The browser owns chat history for this example. The server validates the
    // payload and forwards one OpenAI-compatible chat completion request.
    function llmPayload(userText) {
      const temperature = Number.parseFloat(temperatureInput.value || "0.4");
      const maxTokens = Number.parseInt(maxTokensInput.value || "1024", 10);
      const history = messages.map((item) => ({role: roleLabel(item.role), content: item.text}));
      history.push({role: "user", content: userText});
      return {
        provider: providerInput.value.trim() || "ollama",
        model: modelInput.value.trim() || "gemma3:1b",
        system_prompt: systemPromptInput.value.trim(),
        messages: history,
        temperature: Number.isFinite(temperature) ? temperature : 0.4,
        max_tokens: Number.isFinite(maxTokens) ? maxTokens : 1024
      };
    }

    async function askAssistant() {
      const input = document.getElementById("message-text");
      const text = input.value.trim();
      if (!text) {
        setMessage(discussionMessage, "Enter a message first.", "error");
        return;
      }
      if (reading || playbackMode) stopPlayback(false);

      const payload = llmPayload(text);
      messages.push({role: "user", text});
      input.value = "";
      renderMessages();
      setMessage(llmMessage, "Thinking.", "");

      const button = document.getElementById("ask-assistant");
      button.disabled = true;
      const finishButtonLoading = beginButtonLoading(button, "Thinking");
      try {
        const data = await fetchJson("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const reply = String(data.text || "").trim();
        if (!reply) throw new Error("LLM returned an empty response.");
        messages.push({role: "assistant", text: reply});
        renderMessages();
        const seconds = data.seconds ? " in " + data.seconds.toFixed(1) + "s" : "";
        setMessage(llmMessage, "Reply ready" + seconds + ".", "ok");
        setMessage(discussionMessage, "Reply added.", "ok");
        if (chatSpeakInput.checked) await playSingleMessage(messages.length - 1);
      } catch (err) {
        setMessage(llmMessage, err.message || String(err), "error");
      } finally {
        finishButtonLoading();
        button.disabled = false;
      }
    }

    async function refreshModels() {
      const button = document.getElementById("refresh-models");
      button.disabled = true;
      const finishButtonLoading = beginButtonLoading(button, "Loading");
      try {
        const provider = encodeURIComponent(providerInput.value.trim() || "ollama");
        setMessage(llmMessage, "Loading models.", "");
        const data = await fetchJson("/api/llm/models?provider=" + provider);
        modelOptions.innerHTML = "";
        for (const id of data.models || []) {
          const opt = document.createElement("option");
          opt.value = id;
          modelOptions.appendChild(opt);
        }
        setMessage(llmMessage, (data.models || []).length + " model(s) found.", "ok");
      } catch (err) {
        setMessage(llmMessage, err.message || String(err), "error");
      } finally {
        finishButtonLoading();
        button.disabled = false;
      }
    }

    function writeAscii(view, offset, text) {
      for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    }

    function wavBlobFromAudioBuffer(buffer) {
      const channels = Math.max(1, buffer.numberOfChannels);
      const sampleRate = buffer.sampleRate;
      const frames = buffer.length;
      const samples = new Float32Array(frames);
      for (let channel = 0; channel < channels; channel += 1) {
        const data = buffer.getChannelData(channel);
        for (let i = 0; i < frames; i += 1) samples[i] += data[i] / channels;
      }

      const output = new ArrayBuffer(44 + frames * 2);
      const view = new DataView(output);
      writeAscii(view, 0, "RIFF");
      view.setUint32(4, 36 + frames * 2, true);
      writeAscii(view, 8, "WAVE");
      writeAscii(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeAscii(view, 36, "data");
      view.setUint32(40, frames * 2, true);
      let offset = 44;
      for (let i = 0; i < frames; i += 1) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        offset += 2;
      }
      return new Blob([output], {type: "audio/wav"});
    }

    async function wavBlobFromRecording(blob) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextClass) throw new Error("This browser cannot convert microphone audio to WAV.");
      const context = new AudioContextClass();
      try {
        const buffer = await blob.arrayBuffer();
        const audioBuffer = await context.decodeAudioData(buffer.slice(0));
        return wavBlobFromAudioBuffer(audioBuffer);
      } finally {
        if (context.close) context.close().catch(() => {});
      }
    }

    function recorderMimeType() {
      if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
      for (const type of ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm", "audio/ogg"]) {
        if (MediaRecorder.isTypeSupported(type)) return type;
      }
      return "";
    }

    function stopCloneTracks() {
      if (!cloneRecorderStream) return;
      for (const track of cloneRecorderStream.getTracks()) track.stop();
      cloneRecorderStream = null;
    }

    async function startCloneRecording() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
        setMessage(cloneMessage, "Microphone recording is not available in this browser.", "error");
        return;
      }
      try {
        cloneRecordingChunks = [];
        recordedCloneBlob = null;
        cloneRecorderStream = await navigator.mediaDevices.getUserMedia({audio: true});
        const mimeType = recorderMimeType();
        cloneRecorder = new MediaRecorder(cloneRecorderStream, mimeType ? {mimeType} : undefined);
        cloneRecorder.addEventListener("dataavailable", (event) => {
          if (event.data && event.data.size) cloneRecordingChunks.push(event.data);
        });
        cloneRecorder.addEventListener("stop", async () => {
          const raw = new Blob(cloneRecordingChunks, {type: cloneRecorder.mimeType || "audio/webm"});
          stopCloneTracks();
          cloneRecordButton.disabled = false;
          cloneStopButton.disabled = true;
          try {
            recordedCloneBlob = await wavBlobFromRecording(raw);
            if (recordedCloneUrl) URL.revokeObjectURL(recordedCloneUrl);
            recordedCloneUrl = URL.createObjectURL(recordedCloneBlob);
            clonePreview.src = recordedCloneUrl;
            clonePreview.classList.remove("hidden");
            cloneFileInput.value = "";
            setMessage(cloneMessage, "Recording ready.", "ok");
          } catch (err) {
            setMessage(cloneMessage, err.message || String(err), "error");
          } finally {
            cloneRecorder = null;
            cloneRecordingChunks = [];
          }
        });
        cloneRecorder.start();
        cloneRecordButton.disabled = true;
        cloneStopButton.disabled = false;
        setMessage(cloneMessage, "Recording.", "");
      } catch (err) {
        stopCloneTracks();
        cloneRecordButton.disabled = false;
        cloneStopButton.disabled = true;
        setMessage(cloneMessage, err.message || String(err), "error");
      }
    }

    function stopCloneRecording() {
      if (cloneRecorder && cloneRecorder.state !== "inactive") {
        cloneRecorder.stop();
      } else {
        stopCloneTracks();
        cloneRecordButton.disabled = false;
        cloneStopButton.disabled = true;
      }
    }

    function cloneUploadFile() {
      const selected = cloneFileInput.files && cloneFileInput.files[0];
      if (selected) return selected;
      if (recordedCloneBlob) {
        return new File([recordedCloneBlob], "microphone-reference.wav", {type: "audio/wav"});
      }
      throw new Error("Choose reference audio or record a microphone sample first.");
    }

    async function cloneVoice() {
      let file;
      try {
        file = cloneUploadFile();
      } catch (err) {
        setMessage(cloneMessage, err.message || String(err), "error");
        return;
      }

      const form = new FormData();
      form.append("file", file, file.name || "reference.wav");
      form.append("name", cloneNameInput.value.trim() || "web_voice");
      form.append("engine", cloneEngineInput.value || "f5_tts");
      form.append("reference_text", cloneReferenceTextInput.value.trim());
      form.append("validate", "true");

      cloneVoiceButton.disabled = true;
      const finishButtonLoading = beginButtonLoading(cloneVoiceButton, "Cloning");
      try {
        const data = await withBusy(
          "Cloning voice",
          "Storing and validating the reference audio with " + (cloneEngineInput.value || "f5_tts") + ".",
          () => fetchJson("/api/voices/clone", {method: "POST", body: form})
        );
        await refreshVoices();
        setMessage(cloneMessage, "Cloned and validated voice ready: " + (data.name || data.voice_id || "voice") + ".", "ok");
      } catch (err) {
        setMessage(cloneMessage, err.message || String(err), "error");
      } finally {
        finishButtonLoading();
        cloneVoiceButton.disabled = false;
      }
    }

    document.getElementById("add-message").addEventListener("click", () => {
      const text = document.getElementById("message-text").value.trim();
      if (!text) {
        setMessage(discussionMessage, "Enter a message first.", "error");
        return;
      }
      messages.push({role: document.getElementById("speaker").value, text});
      renderMessages();
      setMessage(discussionMessage, "Message added.", "ok");
    });
    document.getElementById("clear-chat").addEventListener("click", clearChat);
    conversationToggle.addEventListener("click", () => readConversation(0));
    document.getElementById("stop-read").addEventListener("click", () => stopPlayback());
    document.getElementById("ask-assistant").addEventListener("click", askAssistant);
    cloneRecordButton.addEventListener("click", startCloneRecording);
    cloneStopButton.addEventListener("click", stopCloneRecording);
    cloneVoiceButton.addEventListener("click", cloneVoice);
    cloneFileInput.addEventListener("change", () => {
      recordedCloneBlob = null;
      if (recordedCloneUrl) {
        URL.revokeObjectURL(recordedCloneUrl);
        recordedCloneUrl = null;
      }
      const selected = cloneFileInput.files && cloneFileInput.files[0];
      if (selected) {
        recordedCloneUrl = URL.createObjectURL(selected);
        clonePreview.src = recordedCloneUrl;
        clonePreview.classList.remove("hidden");
        setMessage(cloneMessage, "Reference file ready.", "ok");
      } else {
        clonePreview.removeAttribute("src");
        clonePreview.classList.add("hidden");
      }
    });
    document.getElementById("speak").addEventListener("click", speakSingle);
    document.getElementById("transcribe").addEventListener("click", transcribe);
    document.getElementById("refresh-models").addEventListener("click", refreshModels);
    document.getElementById("refresh-voices").addEventListener("click", () => withBusy("Loading TTS voices", "Reading local profiles and cloned voices; the first run may initialize the TTS engine.", refreshVoices));
    assistantVoiceChoice.addEventListener("change", () => selectRoleVoice("assistant"));
    userVoiceChoice.addEventListener("change", () => selectRoleVoice("user"));
    profileChoice.addEventListener("change", selectProfile);
    audio.addEventListener("play", updatePlaybackControls);
    audio.addEventListener("pause", updatePlaybackControls);
    audio.addEventListener("ended", updatePlaybackControls);

    renderMessages();
    refreshStatus();
    withBusy("Loading TTS voices", "Reading local profiles and cloned voices; the first run may initialize the TTS engine.", refreshVoices);
  </script>
</body>
</html>
"""


class ExampleState:
    def __init__(
        self,
        *,
        language: str,
        tts_engine: str,
        stt_engine: str,
        whisper_model: str,
        tts_model: Optional[str] = None,
        stt_model: Optional[str] = None,
        cloning_engine: str = "f5_tts",
        remote_base_url: Optional[str] = None,
        remote_api_key: Optional[str] = None,
        remote_timeout_s: Optional[float] = None,
        allow_downloads: bool = False,
        debug_mode: bool = False,
        voice_manager_factory: Optional[Callable[["ExampleState"], Any]] = None,
    ) -> None:
        self.language = str(language or "en").strip().lower() or "en"
        self.tts_engine = str(tts_engine or "auto").strip().lower().replace("_", "-") or "auto"
        self.stt_engine = str(stt_engine or "auto").strip().lower().replace("_", "-") or "auto"
        self.whisper_model = str(whisper_model or "base").strip() or "base"
        self.tts_model = str(tts_model).strip() if isinstance(tts_model, str) and tts_model.strip() else None
        self.stt_model = str(stt_model).strip() if isinstance(stt_model, str) and stt_model.strip() else None
        self.cloning_engine = str(cloning_engine or "f5_tts").strip().lower().replace("_", "-") or "f5_tts"
        self.remote_base_url = (
            str(remote_base_url).strip() if isinstance(remote_base_url, str) and remote_base_url.strip() else None
        )
        self.remote_api_key = (
            str(remote_api_key).strip() if isinstance(remote_api_key, str) and remote_api_key.strip() else None
        )
        self.remote_timeout_s = remote_timeout_s
        self.allow_downloads = bool(allow_downloads)
        self.debug_mode = bool(debug_mode)
        self.current_voice: Optional[str] = None
        self.role_voices: dict[str, Optional[str]] = {"assistant": None, "user": None}
        self.lock = threading.RLock()
        self.voice_manager = None
        self._voice_manager_factory = voice_manager_factory

    def get_voice_manager(self) -> Any:
        with self.lock:
            if self.voice_manager is not None:
                return self.voice_manager

            if self._voice_manager_factory is not None:
                self.voice_manager = self._voice_manager_factory(self)
                return self.voice_manager

            from abstractvoice import VoiceManager

            self.voice_manager = VoiceManager(
                language=self.language,
                whisper_model=self.whisper_model,
                debug_mode=self.debug_mode,
                tts_engine=self.tts_engine,
                stt_engine=self.stt_engine,
                tts_model=self.tts_model,
                stt_model=self.stt_model,
                allow_downloads=self.allow_downloads,
                cloned_tts_streaming=False,
                cloning_engine=self.cloning_engine,
                remote_base_url=self.remote_base_url,
                remote_api_key=self.remote_api_key,
                remote_timeout_s=self.remote_timeout_s,
            )
            return self.voice_manager

    def cleanup(self) -> None:
        with self.lock:
            if self.voice_manager is not None:
                try:
                    self.voice_manager.cleanup()
                except Exception:
                    pass
            self.voice_manager = None

    def status_dict(self) -> dict[str, Any]:
        with self.lock:
            current: dict[str, Any] = {
                "language": self.language,
                "voice": self.current_voice,
                "voice_kind": "clone" if self.current_voice else "base",
                "role_voices": dict(self.role_voices),
            }
            if self.voice_manager is not None:
                current.update(self._current_loaded_state(self.voice_manager))

            return {
                "ok": True,
                "status": "ok",
                "voice_manager_initialized": self.voice_manager is not None,
                "allow_downloads": self.allow_downloads,
                "defaults": {
                    "language": self.language,
                    "tts_engine": self.tts_engine,
                    "stt_engine": self.stt_engine,
                    "tts_model": self.tts_model,
                    "stt_model": self.stt_model,
                    "cloning_engine": self.cloning_engine,
                    "remote_base_url": self.remote_base_url,
                    "remote_api_key_configured": bool(self.remote_api_key),
                    "remote_timeout_s": self.remote_timeout_s,
                    "whisper_model": self.whisper_model,
                    "llm_provider": DEFAULT_PROVIDER,
                    "llm_model": DEFAULT_MODEL,
                },
                "current": current,
                "optional_dependencies": optional_dependency_status(),
                "routes": LOCAL_ROUTES,
            }

    def _current_loaded_state(self, vm: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        try:
            out["language"] = str(vm.get_language())
        except Exception:
            pass
        try:
            out["speed"] = float(vm.get_speed())
        except Exception:
            pass
        try:
            adapter = getattr(vm, "tts_adapter", None)
            out["tts_engine"] = str(getattr(adapter, "engine_id", "") or getattr(vm, "_tts_engine_name", "") or "")
        except Exception:
            pass
        try:
            profile = vm.get_active_profile(kind="tts")
            if profile is not None:
                out["profile"] = self._profile_dict(profile, active=True)
        except Exception:
            pass
        return out

    def _profile_dict(self, profile: Any, *, active: bool = False) -> dict[str, Any]:
        pid = str(getattr(profile, "profile_id", "") or "")
        engine_id = str(getattr(profile, "engine_id", "") or "").strip().lower()
        return {
            "id": pid,
            "profile_id": pid,
            "engine": engine_id,
            "engine_id": engine_id,
            "label": str(getattr(profile, "label", "") or pid),
            "description": str(getattr(profile, "description", "") or ""),
            "params": dict(getattr(profile, "params", {}) or {}),
            "tags": dict(getattr(profile, "tags", {}) or {}),
            "active": bool(active),
        }

    def _clone_dict(self, item: dict[str, Any]) -> dict[str, Any]:
        voice_id = str(item.get("voice_id") or "")
        name = str(item.get("name") or "").strip()
        roles = [role for role, selected in self.role_voices.items() if voice_id and selected == voice_id]
        return {
            "id": voice_id,
            "voice_id": voice_id,
            "name": name,
            "label": name or voice_id,
            "engine": str(item.get("engine") or "").strip(),
            "active": bool((voice_id and voice_id == self.current_voice) or roles),
            "roles": roles,
        }

    def list_voices(self) -> dict[str, Any]:
        with self.lock:
            vm = self.get_voice_manager()
            profiles: list[dict[str, Any]] = []
            active_profile_id = ""
            try:
                active = vm.get_active_profile(kind="tts")
                if active is not None:
                    active_profile_id = str(getattr(active, "profile_id", "") or "")
            except Exception:
                active_profile_id = ""
            try:
                for profile in list(vm.get_profiles(kind="tts") or []):
                    pid = str(getattr(profile, "profile_id", "") or "")
                    profiles.append(self._profile_dict(profile, active=bool(pid and pid == active_profile_id)))
            except Exception:
                profiles = []

            cloned_voices: list[dict[str, Any]] = []
            clones_error = None
            try:
                cloned_voices = [self._clone_dict(v) for v in list(vm.list_cloned_voices() or [])]
            except Exception as e:
                clones_error = str(e)

            base_models: dict[str, Any] = {}
            try:
                base_models = dict(vm.list_available_models(language=self.language) or {})
            except Exception:
                base_models = {}

            languages: list[str] = []
            try:
                languages = list(vm.get_supported_languages() or [])
            except Exception:
                languages = []

            return {
                "ok": True,
                "current": self.status_dict()["current"],
                "languages": languages,
                "profiles": profiles,
                "cloned_voices": cloned_voices,
                "clones_error": clones_error,
                "base_models": base_models,
            }

    def voice_profile_payload(self) -> dict[str, Any]:
        voices = self.list_voices()
        data: list[dict[str, Any]] = []
        for profile in list(voices.get("profiles") or []):
            pid = str(profile.get("profile_id") or profile.get("id") or "").strip()
            if not pid:
                continue
            item = dict(profile)
            item.setdefault("id", pid)
            item.setdefault("voice", pid)
            item.setdefault("object", "voice.profile")
            item.setdefault("kind", "profile")
            data.append(item)
        for clone in list(voices.get("cloned_voices") or []):
            vid = str(clone.get("voice_id") or clone.get("id") or "").strip()
            if not vid:
                continue
            item = dict(clone)
            item.setdefault("id", vid)
            item.setdefault("voice_id", vid)
            item.setdefault("voice", vid)
            item.setdefault("object", "voice")
            item.setdefault("kind", "clone")
            data.append(item)

        out = dict(voices)
        out["object"] = "list"
        out["data"] = data
        return out

    def _resolve_cloned_voice_id(self, wanted: str) -> str:
        raw = str(wanted or "").strip()
        if not raw:
            raise ValueError("Missing cloned voice id.")
        vm = self.get_voice_manager()
        voices = list(vm.list_cloned_voices() or [])
        for item in voices:
            voice_id = str(item.get("voice_id") or "")
            name = str(item.get("name") or "")
            if raw == voice_id or raw == name or (voice_id and voice_id.startswith(raw)):
                return voice_id
        raise ValueError(f"Unknown cloned voice: {raw}")

    def _set_selected_voice(self, voice_id: Optional[str], *, role: Optional[str]) -> None:
        if role:
            self.role_voices[role] = voice_id
        else:
            self.current_voice = voice_id

    def _preload_voice(self, vm: Any, voice_id: str) -> dict[str, Any]:
        warm = {
            "en": "Hello.",
            "fr": "Bonjour.",
            "de": "Hallo.",
            "es": "Hola.",
            "ru": "Hello.",
            "zh": "Hello.",
        }.get(str(self.language or "en").strip().lower(), "Hello.")
        t0 = time.monotonic()
        audio = vm.speak_to_bytes(str(warm), format="wav", voice=str(voice_id), sanitize_syntax=False)
        metrics = None
        try:
            metrics = vm.pop_last_tts_metrics()
        except Exception:
            metrics = None
        return {
            "ok": True,
            "seconds": float(time.monotonic() - t0),
            "bytes": len(bytes(audio)),
            "metrics": metrics if isinstance(metrics, dict) else None,
        }

    def select_voice(
        self,
        *,
        kind: str,
        voice: Optional[str] = None,
        profile: Optional[str] = None,
        role: Optional[str] = None,
        preload: bool = False,
    ) -> dict[str, Any]:
        mode = str(kind or "").strip().lower()
        target_role = normalize_voice_role(role)
        with self.lock:
            vm = self.get_voice_manager()
            preload_result: Optional[dict[str, Any]] = None
            if mode in ("", "base", "tts", "piper"):
                self._set_selected_voice(None, role=target_role)
                return {"ok": True, "current": self.status_dict()["current"]}
            if mode in ("clone", "cloned"):
                resolved = self._resolve_cloned_voice_id(str(voice or ""))
                if preload:
                    # Commit the selection only after preload succeeds so missing
                    # optional engines/artifacts leave the previous role voice intact.
                    preload_result = self._preload_voice(vm, resolved)
                self._set_selected_voice(resolved, role=target_role)
                return {"ok": True, "current": self.status_dict()["current"], "preload": preload_result}
            if mode == "profile":
                pid = str(profile or voice or "").strip()
                if not pid:
                    raise ValueError("Missing profile id.")
                if not bool(vm.set_profile(pid, kind="tts")):
                    raise ValueError(f"Profile is not supported by the active TTS engine: {pid}")
                self._set_selected_voice(None, role=target_role)
                return {"ok": True, "current": self.status_dict()["current"]}
        raise ValueError("Voice kind must be base, clone, or profile.")

    def synthesize(
        self,
        *,
        text: str,
        fmt: str = "wav",
        voice: Optional[str] = None,
        role: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        sanitize_syntax: bool = True,
    ) -> tuple[bytes, Optional[dict[str, Any]]]:
        speak_text = str(text or "").strip()
        if not speak_text:
            raise ValueError("Missing input text.")

        audio_format = normalize_audio_format(fmt)
        requested_language = str(language or "").strip().lower()
        requested_voice = normalize_voice_id(voice)
        requested_role = normalize_voice_role(role)

        with self.lock:
            vm = self.get_voice_manager()
            if requested_language and requested_language != self.language:
                if not bool(vm.set_language(requested_language)):
                    raise ValueError(f"Unsupported or unavailable language: {requested_language}")
                self.language = requested_language

            selected_voice = requested_voice
            profile_restore_id: Optional[str] = None
            profile_changed = False
            if selected_voice is None:
                if requested_role:
                    selected_voice = self.role_voices.get(requested_role)
                if selected_voice is None:
                    selected_voice = self.current_voice
            elif self._matches_tts_profile(vm, selected_voice):
                try:
                    active = vm.get_active_profile(kind="tts")
                    profile_restore_id = str(getattr(active, "profile_id", "") or "").strip() if active else None
                except Exception:
                    profile_restore_id = None
                if not bool(vm.set_profile(str(selected_voice), kind="tts")):
                    raise ValueError(f"Unknown TTS profile: {selected_voice}")
                profile_changed = True
                selected_voice = None

            old_speed = None
            speed_changed = False
            if speed is not None:
                try:
                    old_speed = float(vm.get_speed())
                except Exception:
                    old_speed = None
                if not bool(vm.set_speed(float(speed))):
                    raise ValueError("Speed must be between 0.5 and 2.0 and supported by the active TTS engine.")
                speed_changed = True

            try:
                audio = vm.speak_to_bytes(
                    speak_text,
                    format=audio_format,
                    voice=selected_voice,
                    sanitize_syntax=bool(sanitize_syntax),
                )
                metrics = None
                try:
                    metrics = vm.pop_last_tts_metrics()
                except Exception:
                    metrics = None
                return bytes(audio), metrics if isinstance(metrics, dict) else None
            finally:
                if profile_changed and profile_restore_id:
                    try:
                        vm.set_profile(str(profile_restore_id), kind="tts")
                    except Exception:
                        pass
                if speed_changed and old_speed is not None:
                    try:
                        vm.set_speed(old_speed)
                    except Exception:
                        pass

    def _matches_tts_profile(self, vm: Any, profile_id: str) -> bool:
        wanted = str(profile_id or "").strip().lower()
        if not wanted:
            return False
        try:
            for item in list(vm.list_cloned_voices() or []):
                vid = str(item.get("voice_id") or "").strip().lower()
                name = str(item.get("name") or "").strip().lower()
                if wanted in {vid, name}:
                    return False
        except Exception:
            pass
        try:
            for profile in list(vm.get_profiles(kind="tts") or []):
                pid = str(getattr(profile, "profile_id", "") or "").strip().lower()
                if pid and pid == wanted:
                    return True
        except Exception:
            return False
        return False

    def transcribe_bytes(self, audio_bytes: bytes, *, filename: str = "audio.wav", language: Optional[str] = None) -> str:
        if not audio_bytes:
            raise ValueError("Missing audio file.")
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with self.lock:
            vm = self.get_voice_manager()
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = Path(tmp.name)
            try:
                tmp.write(bytes(audio_bytes))
                tmp.flush()
            finally:
                try:
                    tmp.close()
                except Exception:
                    pass
            try:
                return str(vm.transcribe_file(str(tmp_path), language=(language or self.language)) or "")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def clone_voice_from_upload(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "reference.wav",
        name: Optional[str] = None,
        engine: Optional[str] = None,
        reference_text: Optional[str] = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        if not audio_bytes:
            raise ValueError("Missing reference audio.")
        suffix = Path(filename or "reference.wav").suffix.lower() or ".wav"
        engine_name = str(engine or self.cloning_engine or "").strip().lower().replace("_", "-")
        allowed_suffixes = {".wav", ".flac", ".ogg"}
        if engine_name in {"openai", "openai-compatible", "remote"}:
            allowed_suffixes |= {".mp3", ".mpeg", ".mpga", ".m4a", ".webm", ".aac"}
        if suffix not in allowed_suffixes:
            if engine_name in {"openai", "openai-compatible", "remote"}:
                raise ValueError("Remote reference audio must be WAV, FLAC, OGG, MP3, M4A, WEBM, or AAC.")
            raise ValueError("Reference audio must be WAV, FLAC, or OGG. Browser microphone recordings are converted to WAV.")

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        try:
            tmp.write(bytes(audio_bytes))
            tmp.flush()
        finally:
            try:
                tmp.close()
            except Exception:
                pass

        try:
            with self.lock:
                vm = self.get_voice_manager()
                voice_id = vm.clone_voice(
                    str(tmp_path),
                    name=(str(name or "").strip() or None),
                    reference_text=(str(reference_text or "").strip() or None),
                    engine=(str(engine or "").strip().lower() or None),
                )
                info: dict[str, Any] = {}
                try:
                    info = dict(vm.get_cloned_voice(str(voice_id)) or {})
                except Exception:
                    info = {}
                validation: Optional[dict[str, Any]] = None
                if bool(validate):
                    try:
                        validation = self._preload_voice(vm, str(voice_id))
                    except Exception as e:
                        try:
                            vm.delete_cloned_voice(str(voice_id))
                        except Exception:
                            pass
                        raise RuntimeError(
                            "Cloned voice was stored but failed validation and was removed. "
                            + describe_exception(e)
                        ) from e
                return {
                    "ok": True,
                    "voice_id": str(voice_id),
                    "name": info.get("name") or str(name or "").strip() or str(voice_id),
                    "engine": info.get("engine") or str(engine or "").strip().lower() or None,
                    "voice": info,
                    "validation": validation,
                    "current": self.status_dict()["current"],
                }
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def normalize_audio_format(value: Optional[str]) -> str:
    fmt = str(value or "wav").strip().lower().lstrip(".") or "wav"
    if fmt == "wave":
        fmt = "wav"
    allowed = {"wav", "mp3", "flac", "ogg", "opus", "pcm"}
    if fmt not in allowed:
        raise ValueError(f"Unsupported audio format: {fmt}")
    return fmt


def normalize_voice_id(value: Optional[str]) -> Optional[str]:
    voice = str(value or "").strip()
    if not voice or voice.lower() in {"base", "default", "piper", "tts"}:
        return None
    return voice


def normalize_voice_role(value: Optional[str]) -> Optional[str]:
    role = str(value or "").strip().lower()
    if not role:
        return None
    if role not in {"assistant", "user"}:
        raise ValueError("Voice role must be assistant or user.")
    return role


def describe_exception(exc: Exception) -> str:
    """Return the visible error plus useful chained causes."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = str(cur).strip()
        if text and text not in parts:
            parts.append(text)
        cur = cur.__cause__ or cur.__context__
    return " Caused by: ".join(parts) if parts else exc.__class__.__name__


def optional_dependency_status() -> dict[str, dict[str, Any]]:
    """Report optional runtime imports without importing heavy engines."""
    out: dict[str, dict[str, Any]] = {}
    modules = {
        "omnivoice": "omnivoice",
        "f5_tts": "f5_tts",
        "chroma": "transformers",
        "audiodit": "torch",
    }
    packages = {
        "omnivoice": "omnivoice",
        "f5_tts": "f5-tts",
        "chroma": "transformers",
        "audiodit": "torch",
    }
    for key, module in modules.items():
        installed = importlib.util.find_spec(module) is not None
        version = None
        if installed:
            try:
                version = importlib.metadata.version(packages.get(key, module))
            except Exception:
                version = None
        out[key] = {"installed": bool(installed), "version": version}
    return out


def normalize_chat_messages(messages: Any, *, system_prompt: Optional[str] = None) -> list[dict[str, str]]:
    """Validate the tiny browser-chat payload before proxying to the LLM."""
    out: list[dict[str, str]] = []
    prompt = str(system_prompt or "").strip()
    if prompt:
        out.append({"role": "system", "content": prompt})

    raw_messages = messages or []
    if not isinstance(raw_messages, list):
        raise ValueError("Chat messages must be a list.")

    saw_user = False
    for item in raw_messages:
        if not isinstance(item, dict):
            raise ValueError("Each chat message must be an object.")
        role = str(item.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Chat message role must be system, user, or assistant.")
        if role == "system" and prompt:
            continue
        content = str(item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        if role == "user":
            saw_user = True
        out.append({"role": role, "content": content})

    if not saw_user:
        raise ValueError("Chat messages must include at least one user message.")
    return out


def clamp_float(value: Optional[float], *, default: float, minimum: float, maximum: float) -> float:
    try:
        x = float(default if value is None else value)
    except Exception:
        x = float(default)
    return max(float(minimum), min(float(maximum), x))


def clamp_int(value: Optional[int], *, default: int, minimum: int, maximum: int) -> int:
    try:
        x = int(default if value is None else value)
    except Exception:
        x = int(default)
    return max(int(minimum), min(int(maximum), x))


def media_type_for_format(fmt: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "pcm": "application/octet-stream",
    }.get(str(fmt or "wav").lower(), "application/octet-stream")


def _load_fastapi():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
        from fastapi.responses import HTMLResponse, Response
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise RuntimeError(
            'The web example requires FastAPI. Install it with: pip install "abstractvoice[web]"'
        ) from e
    return FastAPI, File, Form, HTTPException, Query, UploadFile, HTMLResponse, Response, BaseModel, Field


def _load_uvicorn():
    try:
        import uvicorn
    except ImportError as e:
        raise RuntimeError(
            'Running the web example requires Uvicorn. Install it with: pip install "abstractvoice[web]"'
        ) from e
    return uvicorn


def create_app(
    *,
    language: str = "en",
    tts_engine: str = "auto",
    stt_engine: str = "auto",
    whisper_model: str = "base",
    tts_model: Optional[str] = None,
    stt_model: Optional[str] = None,
    cloning_engine: str = "f5_tts",
    remote_base_url: Optional[str] = None,
    remote_api_key: Optional[str] = None,
    remote_timeout_s: Optional[float] = None,
    allow_downloads: bool = False,
    debug_mode: bool = False,
    voice_manager_factory: Optional[Callable[[ExampleState], Any]] = None,
):
    FastAPI, File, Form, HTTPException, Query, UploadFile, HTMLResponse, Response, BaseModel, Field = _load_fastapi()
    state = ExampleState(
        language=language,
        tts_engine=tts_engine,
        stt_engine=stt_engine,
        whisper_model=whisper_model,
        tts_model=tts_model,
        stt_model=stt_model,
        cloning_engine=cloning_engine,
        remote_base_url=remote_base_url,
        remote_api_key=remote_api_key,
        remote_timeout_s=remote_timeout_s,
        allow_downloads=allow_downloads,
        debug_mode=debug_mode,
        voice_manager_factory=voice_manager_factory,
    )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            state.cleanup()

    class SpeechRequest(BaseModel):
        input: Optional[str] = Field(None, description="Text to synthesize.", examples=["Hello from AbstractVoice."])
        text: Optional[str] = Field(None, description="Alias for input.")
        voice: Optional[str] = Field(None, description="Optional cloned voice id/name or active-engine profile id. Omit for base TTS.")
        format: Optional[str] = Field(None, description="Audio output format alias.", examples=["wav"])
        response_format: Optional[str] = Field("wav", description="Audio output format.", examples=["wav"])
        language: Optional[str] = Field(None, description="Language code to use for this request.", examples=["en"])
        role: Optional[str] = Field(None, description="Browser example role default: assistant or user.", examples=["assistant"])
        speed: Optional[float] = Field(None, description="Temporary TTS speed for this request, 0.5 to 2.0.", examples=[1.0])
        sanitize_syntax: bool = Field(True, description="Sanitize Markdown/code-like syntax before speech.")

    class VoiceSelectRequest(BaseModel):
        kind: str = Field("base", description="base, clone, or profile.", examples=["clone"])
        voice: Optional[str] = Field(None, description="Cloned voice id/name, or profile alias.", examples=["my_voice"])
        profile: Optional[str] = Field(None, description="Base TTS profile id.", examples=["female_01"])
        role: Optional[str] = Field(None, description="Browser example role default: assistant or user.", examples=["assistant"])
        preload: bool = Field(False, description="Warm the cloned voice before committing the selection.")

    class ChatRequest(BaseModel):
        provider: Optional[str] = Field(None, description="Provider preset or base URL.", examples=["ollama"])
        model: Optional[str] = Field(None, description="OpenAI-compatible model id.", examples=["gemma3:1b"])
        system_prompt: Optional[str] = Field(None, description="Optional system prompt prepended by the server.")
        messages: Optional[list[dict[str, Any]]] = Field(
            None,
            description="Short chat history owned by the browser.",
            examples=[[{"role": "user", "content": "Say hi in one sentence."}]],
        )
        temperature: Optional[float] = Field(0.4, description="Sampling temperature clamped to 0.0-2.0.")
        max_tokens: Optional[int] = Field(1024, description="Maximum output tokens clamped to 1-32768.")

    app = FastAPI(
        title="AbstractVoice Local Example",
        version="0.1",
        lifespan=lifespan,
    )
    app.state.abstractvoice_state = state
    atexit.register(state.cleanup)

    def http_error(exc: Exception, *, status_code: int = 500):
        if isinstance(exc, ValueError):
            status_code = 400
        raise HTTPException(status_code=status_code, detail=describe_exception(exc))

    audio_responses = {
        200: {
            "description": "Synthesized audio bytes.",
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
                "audio/flac": {},
                "audio/ogg": {},
                "application/octet-stream": {},
            },
        }
    }

    def speech_response(payload: SpeechRequest):
        text = str(payload.input or payload.text or "").strip()
        fmt = normalize_audio_format(payload.response_format or payload.format or "wav")
        try:
            audio, _metrics = state.synthesize(
                text=text,
                fmt=fmt,
                voice=payload.voice,
                role=payload.role,
                language=payload.language,
                speed=payload.speed,
                sanitize_syntax=payload.sanitize_syntax,
            )
        except Exception as e:
            http_error(e)
        return Response(
            content=audio,
            media_type=media_type_for_format(fmt),
            headers={"Content-Disposition": f'inline; filename="abstractvoice.{fmt}"'},
        )

    async def transcription_response(file: UploadFile, language: Optional[str]):
        try:
            audio_bytes = await file.read()
            text = state.transcribe_bytes(
                audio_bytes,
                filename=str(file.filename or "audio.wav"),
                language=language,
            )
            return {"ok": True, "text": text}
        except Exception as e:
            http_error(e)

    @app.get("/", response_class=HTMLResponse)
    async def home():
        return PAGE

    @app.get("/api/status")
    async def status():
        return state.status_dict()

    @app.get("/api/routes")
    async def routes():
        return {"ok": True, "routes": LOCAL_ROUTES}

    @app.get("/api/voices")
    async def voices():
        try:
            return state.list_voices()
        except Exception as e:
            http_error(e)

    @app.get("/v1/audio/voices", summary="List TTS profiles and cloned voices")
    async def openai_voice_profiles():
        try:
            return state.voice_profile_payload()
        except Exception as e:
            http_error(e)

    @app.post(
        "/api/voices/clone",
        summary="Clone and validate a browser voice",
        description=(
            "Upload or record a WAV/FLAC/OGG reference sample. By default the example validates "
            "the clone by synthesizing a short sample; if validation fails, the stored clone is removed."
        ),
    )
    async def clone_voice(
        file: UploadFile = File(..., description="Reference audio file. Browser microphone recordings are sent as WAV."),
        name: Optional[str] = Form(None, description="Friendly cloned voice name.", examples=["my_voice"]),
        engine: Optional[str] = Form(
            None,
            description="Optional clone engine id, for example f5_tts, omnivoice, audiodit, chroma, openai, or openai-compatible.",
            examples=["openai-compatible"],
        ),
        reference_text: Optional[str] = Form(
            None,
            description="Transcript of the reference audio when the selected engine can use it.",
        ),
        validate_clone: bool = Form(
            True,
            alias="validate",
            description="When true, synthesize a short sample before reporting the clone as ready.",
        ),
    ):
        try:
            audio_bytes = await file.read()
            return state.clone_voice_from_upload(
                audio_bytes,
                filename=str(file.filename or "reference.wav"),
                name=name,
                engine=engine,
                reference_text=reference_text,
                validate=bool(validate_clone),
            )
        except Exception as e:
            http_error(e)

    @app.post(
        "/v1/voice/clone",
        summary="Create a remote-compatible cloned voice",
        description=(
            "AbstractVoice-compatible extension endpoint. It stores a cloned voice "
            "through VoiceManager.clone_voice() and returns a voice_id for later "
            "/v1/audio/speech requests."
        ),
    )
    async def openai_compatible_clone_voice(
        file: UploadFile = File(..., description="Reference audio file."),
        name: Optional[str] = Form(None, description="Friendly cloned voice name.", examples=["my_voice"]),
        reference_text: Optional[str] = Form(None, description="Transcript of the reference audio when available."),
        engine: Optional[str] = Form(None, description="Optional clone engine id."),
        validate_clone: bool = Form(False, alias="validate", description="When true, validate by synthesizing a short sample."),
    ):
        try:
            audio_bytes = await file.read()
            out = state.clone_voice_from_upload(
                audio_bytes,
                filename=str(file.filename or "reference.wav"),
                name=name,
                engine=engine,
                reference_text=reference_text,
                validate=bool(validate_clone),
            )
            return {
                "ok": True,
                "id": out["voice_id"],
                "voice_id": out["voice_id"],
                "voice": out.get("voice") or {},
                "name": out.get("name"),
                "engine": out.get("engine"),
                "validation": out.get("validation"),
            }
        except Exception as e:
            http_error(e)

    @app.get("/api/llm/models", summary="List models from an OpenAI-compatible provider")
    async def llm_models(provider: Optional[str] = Query(None, description="Provider preset or base URL.", examples=["ollama"])):
        llm = resolve_provider(provider or DEFAULT_PROVIDER)
        return {
            "ok": True,
            "provider": llm.name,
            "base_url": llm.base_url,
            "models": llm.list_models(),
        }

    # Example-only LLM bridge: no server-side memory, auth, streaming, or agent loop.
    @app.post("/api/chat")
    def chat(payload: ChatRequest):
        try:
            llm = resolve_provider(payload.provider or DEFAULT_PROVIDER)
            model = str(payload.model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
            messages = normalize_chat_messages(payload.messages, system_prompt=payload.system_prompt)
            t0 = time.monotonic()
            result = llm.chat(
                model=model,
                messages=messages,
                temperature=clamp_float(payload.temperature, default=0.4, minimum=0.0, maximum=2.0),
                max_tokens=clamp_int(payload.max_tokens, default=1024, minimum=1, maximum=32768),
            )
            text = str(result.get("text") or "").strip()
            return {
                "ok": True,
                "provider": llm.name,
                "base_url": llm.base_url,
                "model": model,
                "text": text,
                "message": {"role": "assistant", "content": text},
                "usage": result.get("usage") if isinstance(result.get("usage"), dict) else {},
                "seconds": float(time.monotonic() - t0),
            }
        except Exception as e:
            http_error(e, status_code=502)

    @app.post("/api/voices/select")
    async def select_voice(payload: VoiceSelectRequest):
        try:
            return state.select_voice(
                kind=payload.kind,
                voice=payload.voice,
                profile=payload.profile,
                role=payload.role,
                preload=payload.preload,
            )
        except Exception as e:
            http_error(e)

    @app.post("/api/tts", response_class=Response, responses=audio_responses)
    async def local_tts(payload: SpeechRequest):
        return speech_response(payload)

    @app.post("/v1/audio/speech", response_class=Response, responses=audio_responses)
    async def openai_speech(payload: SpeechRequest):
        return speech_response(payload)

    @app.post("/api/stt/transcriptions", summary="Transcribe uploaded audio")
    async def local_transcriptions(
        file: UploadFile = File(..., description="Audio file to transcribe."),
        language: Optional[str] = Form(None, description="Optional language code.", examples=["en"]),
    ):
        return await transcription_response(file, language)

    @app.post("/api/stt/transcribe", summary="Compatibility alias for local transcription")
    async def local_transcribe_compat(
        file: UploadFile = File(..., description="Audio file to transcribe."),
        language: Optional[str] = Form(None, description="Optional language code.", examples=["en"]),
    ):
        return await transcription_response(file, language)

    @app.post("/v1/audio/transcriptions", summary="OpenAI-compatible local transcription alias")
    async def openai_transcriptions(
        file: UploadFile = File(..., description="Audio file to transcribe."),
        language: Optional[str] = Form(None, description="Optional language code.", examples=["en"]),
    ):
        return await transcription_response(file, language)

    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 5000,
    language: str = "en",
    tts_engine: str = "auto",
    stt_engine: str = "auto",
    whisper_model: str = "base",
    tts_model: Optional[str] = None,
    stt_model: Optional[str] = None,
    cloning_engine: str = "f5_tts",
    remote_base_url: Optional[str] = None,
    remote_api_key: Optional[str] = None,
    remote_timeout_s: Optional[float] = None,
    allow_downloads: bool = False,
    debug_mode: bool = False,
) -> None:
    app = create_app(
        language=language,
        tts_engine=tts_engine,
        stt_engine=stt_engine,
        whisper_model=whisper_model,
        tts_model=tts_model,
        stt_model=stt_model,
        cloning_engine=cloning_engine,
        remote_base_url=remote_base_url,
        remote_api_key=remote_api_key,
        remote_timeout_s=remote_timeout_s,
        allow_downloads=allow_downloads,
        debug_mode=debug_mode,
    )
    print(f"Starting AbstractVoice local FastAPI UI on http://{host}:{int(port)}")
    print(f"Model downloads from web requests: {'allowed' if allow_downloads else 'off'}")
    print("Press CTRL+C to quit.")
    uvicorn = _load_uvicorn()
    uvicorn.run(app, host=host, port=int(port), log_level="debug" if debug_mode else "info")


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="AbstractVoice local FastAPI web example")
    parser.add_argument("--host", default="127.0.0.1", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--language", "--lang", default="en", help="Default language code")
    parser.add_argument("--tts-engine", default="auto", help="Default TTS engine")
    parser.add_argument("--stt-engine", default="auto", help="Default STT engine")
    parser.add_argument("--tts-model", default=None, help="Model id for remote TTS engines")
    parser.add_argument("--stt-model", default=None, help="Model id for remote STT engines")
    parser.add_argument(
        "--cloning-engine",
        default="f5_tts",
        choices=["f5_tts", "chroma", "audiodit", "omnivoice", "openai", "openai-compatible"],
        help="Default cloning backend for new voices",
    )
    parser.add_argument("--remote-base-url", default=None, help="Base URL for OpenAI-compatible remote voice endpoints")
    parser.add_argument("--remote-api-key", default=None, help="Bearer API key for remote voice endpoints")
    parser.add_argument("--remote-timeout", type=float, default=None, help="Remote voice request timeout in seconds")
    parser.add_argument("--whisper", default="base", help="Default faster-whisper model")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow model downloads from web requests")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run_server(
            host=args.host,
            port=args.port,
            language=args.language,
            tts_engine=args.tts_engine,
            stt_engine=args.stt_engine,
            whisper_model=args.whisper,
            tts_model=args.tts_model,
            stt_model=args.stt_model,
            cloning_engine=args.cloning_engine,
            remote_base_url=args.remote_base_url,
            remote_api_key=args.remote_api_key,
            remote_timeout_s=args.remote_timeout,
            allow_downloads=args.allow_downloads,
            debug_mode=args.debug,
        )
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
