const state = {
  userId: "",
  conversationId: "",
  selectedAvatarId: "",
  selectedVoiceId: "alloy",
  selectedAvatarName: "Avatar",
  muted: false,
  callActive: false,
  callStarting: false,
  mediaStream: null,
  processingTurn: false,
  listeningActive: false,
  speechRecognition: null,
  pendingUserText: "",
  lastAssistantText: "",
  useBrowserStt: Boolean(window.SpeechRecognition || window.webkitSpeechRecognition),
  useSseStreaming: false,
  preferBrowserTts: false,
  currentMood: "neutral",
  browserTtsRate: 0.84,
  audioPlaybackRate: 0.92,
  lipSyncRaf: null,
  audioContext: null,
  audioAnalyser: null,
  audioData: null,
  audioSourceNode: null,
  browserLipTimer: null,
  mouthOpen: 0.06,
  currentTurnAbortController: null,
  userStopRequested: false,
  lastStopHandledAt: 0,
  speechBuffer: "",
  studentFinalTranscript: "",
  studentInterimTranscript: "",
  conversationTranscript: [],
  speechResponseTimer: null,
  responseDelayMs: 5000,
  aiReady: false,
  aiLastCheckedAt: 0,
  aiCheckInFlight: null,
};
const API_BASE = "/avatar-service";
const TEACHER_NAME = "SP Sir";
const AI_HEALTH_CACHE_MS = 180000;

const ui = {
  signupForm: document.getElementById("signup-form"),
  emailInput: document.getElementById("email-input"),
  userId: document.getElementById("user-id"),
  conversationId: document.getElementById("conversation-id"),
  refreshAvatars: document.getElementById("refresh-avatars"),
  avatarGrid: document.getElementById("avatar-grid"),
  avatarTitle: document.getElementById("avatar-title"),
  callStatus: document.getElementById("call-status"),
  startCall: document.getElementById("start-call"),
  endCall: document.getElementById("end-call"),
  muteMic: document.getElementById("mute-mic"),
  testVoice: document.getElementById("test-voice"),
  holdToTalk: document.getElementById("hold-to-talk"),
  talkHint: document.getElementById("talk-hint"),
  studentTranscript: document.getElementById("student-transcript"),
  conversationTranscript: document.getElementById("conversation-transcript"),
  liveCaption: document.getElementById("live-caption"),
  audioPlayer: document.getElementById("audio-player"),
  avatarRig: document.getElementById("avatar-rig"),
  expressionChip: document.getElementById("expression-chip"),
  toast: document.getElementById("toast"),
  trainingForm: document.getElementById("training-form"),
  trainingFile: document.getElementById("training-file"),
  uploadTraining: document.getElementById("upload-training"),
  refreshTraining: document.getElementById("refresh-training"),
  clearTraining: document.getElementById("clear-training"),
  trainingDocCount: document.getElementById("training-doc-count"),
  trainingChunkCount: document.getElementById("training-chunk-count"),
  trainingCharCount: document.getElementById("training-char-count"),
  trainingDocList: document.getElementById("training-doc-list"),
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setMouthShape(openness, width = 1, roundness = 0.55) {
  const open = clamp(openness, 0.04, 1);
  const w = clamp(width, 0.82, 1.5);
  const round = clamp(roundness, 0.2, 1);
  state.mouthOpen = open;
  ui.avatarRig.style.setProperty("--mouth-open", open.toFixed(3));
  ui.avatarRig.style.setProperty("--mouth-width", w.toFixed(3));
  ui.avatarRig.style.setProperty("--mouth-round", round.toFixed(3));
}

function resetMouthShape() {
  setMouthShape(0.06, 1, 0.55);
}

function ensureAudioAnalyser() {
  if (state.audioAnalyser) return true;
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return false;
  try {
    state.audioContext = state.audioContext || new AudioCtx();
    if (state.audioContext.state === "suspended") {
      state.audioContext.resume().catch(() => {});
    }
    state.audioAnalyser = state.audioContext.createAnalyser();
    state.audioAnalyser.fftSize = 256;
    state.audioAnalyser.smoothingTimeConstant = 0.82;
    state.audioData = new Uint8Array(state.audioAnalyser.frequencyBinCount);
    state.audioSourceNode = state.audioContext.createMediaElementSource(ui.audioPlayer);
    state.audioSourceNode.connect(state.audioAnalyser);
    state.audioAnalyser.connect(state.audioContext.destination);
    return true;
  } catch (_err) {
    return false;
  }
}

function stopLipSyncLoop() {
  if (state.lipSyncRaf) {
    cancelAnimationFrame(state.lipSyncRaf);
    state.lipSyncRaf = null;
  }
  if (state.browserLipTimer) {
    window.clearInterval(state.browserLipTimer);
    state.browserLipTimer = null;
  }
}

function normalizeVoiceCommand(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isAvatarBusy() {
  return state.processingTurn || !ui.audioPlayer.paused || Boolean(window.speechSynthesis?.speaking);
}

function isStopCommand(text) {
  const cmd = normalizeVoiceCommand(text);
  if (!cmd) return false;
  const words = cmd.split(" ").filter(Boolean);
  if (cmd === "stop" || cmd === "stop now" || cmd === "pause" || cmd === "mute" || cmd === "silence") {
    return true;
  }
  if (cmd.startsWith("stop ") && words.length <= 5) return true;
  if (cmd === "stop avatar" || cmd === "stop talking" || cmd === "be quiet") return true;
  if (words.length <= 7 && (words.includes("stop") || words.includes("pause") || words.includes("quiet"))) {
    return true;
  }
  return false;
}

function shouldTriggerStop(text) {
  const cmd = normalizeVoiceCommand(text);
  if (!cmd) return false;
  if (isStopCommand(cmd)) return true;
  if (isAvatarBusy() && /\b(stop|pause|quiet|silence)\b/.test(cmd)) return true;
  return false;
}

function clearSpeechResponseTimer() {
  if (state.speechResponseTimer) {
    window.clearTimeout(state.speechResponseTimer);
    state.speechResponseTimer = null;
  }
}

function clearSpeechBuffer() {
  state.speechBuffer = "";
  state.studentInterimTranscript = "";
  updateStudentTranscriptUI();
}

function updateStudentTranscriptUI() {
  const finalText = state.studentFinalTranscript.trim();
  const interimText = state.studentInterimTranscript.trim();
  const combined = `${finalText} ${interimText}`.trim();
  if (ui.studentTranscript) {
    ui.studentTranscript.textContent = combined || "Listening transcript will appear here...";
  }
}

function resetStudentTranscriptUI() {
  state.studentFinalTranscript = "";
  state.studentInterimTranscript = "";
  updateStudentTranscriptUI();
}

function renderConversationTranscriptUI() {
  if (!ui.conversationTranscript) return;
  ui.conversationTranscript.innerHTML = "";
  if (!state.conversationTranscript.length) {
    const empty = document.createElement("div");
    empty.className = "conversation-line muted";
    empty.textContent = "No conversation yet.";
    ui.conversationTranscript.appendChild(empty);
    return;
  }
  state.conversationTranscript.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "conversation-line";
    const speaker = document.createElement("span");
    speaker.className = "speaker";
    speaker.textContent = `${entry.speaker}:`;
    row.appendChild(speaker);
    row.appendChild(document.createTextNode(` ${entry.text}`));
    ui.conversationTranscript.appendChild(row);
  });
}

function appendConversationTranscript(speaker, text) {
  const clean = String(text || "").trim();
  if (!clean) return;
  const prev = state.conversationTranscript[state.conversationTranscript.length - 1];
  if (prev && prev.speaker === speaker && prev.text === clean) {
    return;
  }
  state.conversationTranscript.push({ speaker, text: clean });
  if (state.conversationTranscript.length > 80) {
    state.conversationTranscript = state.conversationTranscript.slice(-80);
  }
  renderConversationTranscriptUI();
}

function clearConversationTranscript() {
  state.conversationTranscript = [];
  renderConversationTranscriptUI();
}

function scheduleBufferedSpeechResponse() {
  clearSpeechResponseTimer();
  const text = state.speechBuffer.trim();
  if (!text) return;
  ui.talkHint.textContent = "Waiting 5s silence, then SP Sir will answer...";
  state.speechResponseTimer = window.setTimeout(async () => {
    state.speechResponseTimer = null;
    const finalText = state.speechBuffer.trim();
    clearSpeechBuffer();
    if (!finalText || !state.callActive || state.muted) return;
    await runTurn(finalText);
  }, state.responseDelayMs);
}

function requestImmediateStop(source = "voice") {
  const now = Date.now();
  if (now - state.lastStopHandledAt < 350) return;
  state.lastStopHandledAt = now;
  state.userStopRequested = true;
  state.pendingUserText = "";
  clearSpeechResponseTimer();
  clearSpeechBuffer();
  state.studentInterimTranscript = "";
  updateStudentTranscriptUI();
  if (state.currentTurnAbortController) {
    state.currentTurnAbortController.abort();
    state.currentTurnAbortController = null;
  }
  interruptAvatarPlayback();
  ui.liveCaption.textContent = "Stopped. You can continue speaking.";
  ui.talkHint.textContent = "stopped immediately";
  if (source !== "system") {
    showToast("Stopped");
  }
}

function startAudioLipSyncFromElement() {
  stopLipSyncLoop();
  if (state.audioContext && state.audioContext.state === "suspended") {
    state.audioContext.resume().catch(() => {});
  }
  const hasAnalyser = ensureAudioAnalyser();
  if (!hasAnalyser || !state.audioAnalyser || !state.audioData) {
    setMouthShape(0.28, 1.12, 0.8);
    return;
  }

  const tick = () => {
    if (ui.audioPlayer.paused || ui.audioPlayer.ended) {
      setMouthShape(0.08, 1.02, 0.56);
      return;
    }

    state.audioAnalyser.getByteFrequencyData(state.audioData);
    let energy = 0;
    let bins = 0;
    for (let i = 2; i < 42; i += 1) {
      energy += state.audioData[i];
      bins += 1;
    }
    const normalized = bins ? energy / (bins * 255) : 0;
    const targetOpen = clamp(0.08 + normalized * 2.6, 0.08, 1);
    const smoothOpen = state.mouthOpen * 0.64 + targetOpen * 0.36;
    const width = clamp(0.96 + normalized * 0.6, 0.92, 1.4);
    const round = clamp(0.44 + normalized * 0.72, 0.44, 0.98);
    setMouthShape(smoothOpen, width, round);
    state.lipSyncRaf = requestAnimationFrame(tick);
  };
  state.lipSyncRaf = requestAnimationFrame(tick);
}

function getWordAround(text, charIndex) {
  if (!text || charIndex < 0 || charIndex >= text.length) return "";
  const start = text.slice(0, charIndex).search(/[^\s]+$/);
  const endMatch = text.slice(charIndex).match(/\s/);
  const end = endMatch ? charIndex + endMatch.index : text.length;
  if (start < 0 || end <= start) return "";
  return text.slice(start, end).trim();
}

function visemeForWord(word) {
  const clean = String(word || "").toLowerCase();
  if (!clean) return { open: 0.2, width: 1.02, round: 0.55 };
  const vowelCount = (clean.match(/[aeiou]/g) || []).length;
  const hardStops = (clean.match(/[bmp]/g) || []).length;
  const rounded = (clean.match(/[owu]/g) || []).length;
  const lengthBoost = clamp(clean.length / 9, 0, 1);
  const open = clamp(0.18 + vowelCount * 0.11 + lengthBoost * 0.12 - hardStops * 0.05, 0.12, 0.96);
  const width = clamp(0.94 + (vowelCount - rounded) * 0.07 + hardStops * 0.05, 0.86, 1.34);
  const round = clamp(0.45 + rounded * 0.12, 0.35, 0.98);
  return { open, width, round };
}

function startBrowserLipSync(text) {
  stopLipSyncLoop();
  let phase = 0;
  state.browserLipTimer = window.setInterval(() => {
    phase += 1;
    const pulse = 0.22 + Math.abs(Math.sin(phase * 0.75)) * 0.28;
    setMouthShape(pulse, 1.03, 0.62);
  }, 95);
  // Keep this so onboundary can override with word-specific visemes.
  return (charIndex) => {
    const word = getWordAround(text, charIndex);
    const v = visemeForWord(word);
    setMouthShape(v.open, v.width, v.round);
  };
}

function pickMaleBrowserVoice() {
  const synth = window.speechSynthesis;
  if (!synth) return null;
  const voices = synth.getVoices() || [];
  if (!voices.length) return null;

  const maleHints = [
    /alex/i,
    /daniel/i,
    /david/i,
    /thomas/i,
    /james/i,
    /male/i,
  ];
  const english = voices.filter((v) => String(v.lang || "").toLowerCase().startsWith("en"));
  const pool = english.length ? english : voices;
  return pool.find((v) => maleHints.some((hint) => hint.test(v.name))) || pool[0] || null;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.userId) headers["X-User-Id"] = state.userId;

  const response = await fetch(path, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function showToast(message) {
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  window.setTimeout(() => ui.toast.classList.remove("show"), 2200);
}

function setCallStatus(text) {
  ui.callStatus.textContent = text;
}

function updateCallButtons() {
  ui.startCall.disabled = state.callActive || state.callStarting;
  ui.endCall.disabled = !state.callActive;
}

function setAvatarSpeaking(speaking) {
  ui.avatarRig.classList.toggle("speaking", speaking);
}

function setAvatarMood(mood) {
  const allowed = ["neutral", "thinking", "empathetic", "curious", "confident"];
  const next = allowed.includes(mood) ? mood : "neutral";
  state.currentMood = next;
  ui.avatarRig.classList.remove("mood-neutral", "mood-thinking", "mood-empathetic", "mood-curious", "mood-confident");
  ui.avatarRig.classList.add(`mood-${next}`);
  ui.expressionChip.textContent = next;
}

function inferMoodFromReply(text) {
  const t = text.toLowerCase();
  if (t.includes("i hear you") || t.includes("together") || t.includes("feel")) return "empathetic";
  if (t.includes("?") || t.includes("question")) return "curious";
  if (t.includes("let's") || t.includes("step") || t.includes("plan")) return "confident";
  return "neutral";
}

function pulseAvatarForAudio() {
  if (state.currentMood === "thinking") setAvatarMood("confident");
  setAvatarSpeaking(true);
  setMouthShape(0.2, 1.06, 0.72);
}

function stopAvatarPulse() {
  stopLipSyncLoop();
  setAvatarSpeaking(false);
  resetMouthShape();
  if (state.currentMood !== "thinking") setAvatarMood("neutral");
}

function interruptAvatarPlayback() {
  window.speechSynthesis?.cancel();
  if (!ui.audioPlayer.paused) {
    ui.audioPlayer.pause();
    ui.audioPlayer.currentTime = 0;
  }
  stopAvatarPulse();
  setAvatarMood("curious");
}

function speakWithBrowserVoice(text, { signal } = {}) {
  return new Promise((resolve, reject) => {
    const synth = window.speechSynthesis;
    if (!synth) {
      reject(new Error("browser_tts_unavailable"));
      return;
    }
    if (signal?.aborted) {
      resolve();
      return;
    }

    let finished = false;
    const finish = (fn) => {
      if (finished) return;
      finished = true;
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
      fn();
    };
    const onAbort = () => {
      try {
        synth.cancel();
      } catch (_err) {
        // no-op
      }
      stopAvatarPulse();
      finish(() => resolve());
    };
    if (signal) {
      signal.addEventListener("abort", onAbort, { once: true });
    }

    synth.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    const maleVoice = pickMaleBrowserVoice();
    if (maleVoice) utter.voice = maleVoice;
    utter.rate = state.browserTtsRate;
    utter.pitch = 0.94;
    utter.volume = 1;
    const onVisemeBoundary = startBrowserLipSync(text);
    utter.onstart = () => {
      if (signal?.aborted) {
        onAbort();
        return;
      }
      pulseAvatarForAudio();
    };
    utter.onboundary = (event) => {
      if (typeof event.charIndex === "number") {
        onVisemeBoundary(event.charIndex);
      }
    };
    utter.onend = () => {
      stopAvatarPulse();
      finish(() => resolve());
    };
    utter.onerror = () => {
      stopAvatarPulse();
      if (signal?.aborted) {
        finish(() => resolve());
        return;
      }
      finish(() => reject(new Error("browser_tts_failed")));
    };
    synth.speak(utter);
  });
}

function getSpeechRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

async function signup(email) {
  const user = await api(`${API_BASE}/auth/signup`, {
    method: "POST",
    body: JSON.stringify({ email }),
  });
  state.userId = user.id;
  ui.userId.textContent = state.userId;
}

async function loadAvatars() {
  const avatars = await api(`${API_BASE}/avatars`, { method: "GET" });
  if (!avatars.length) return;

  if (!state.selectedAvatarId) {
    state.selectedAvatarId = avatars[0].id;
    state.selectedVoiceId = avatars[0].voiceId;
    state.selectedAvatarName = avatars[0].name;
    ui.avatarTitle.textContent = TEACHER_NAME;
    setAvatarMood("neutral");
  }

  ui.avatarGrid.innerHTML = "";
  avatars.forEach((avatar) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `avatar-card ${state.selectedAvatarId === avatar.id ? "active" : ""}`;
    card.innerHTML = `<div class="avatar-name">${avatar.name}</div><div class="avatar-meta">Voice: ${avatar.voiceId}</div>`;
    card.addEventListener("click", async () => {
      state.selectedAvatarId = avatar.id;
      state.selectedVoiceId = avatar.voiceId;
      state.selectedAvatarName = avatar.name;
      ui.avatarTitle.textContent = TEACHER_NAME;
      setAvatarMood("neutral");
      if (state.userId) await createConversation();
      await loadAvatars();
      await refreshTrainingPanel();
    });
    ui.avatarGrid.appendChild(card);
  });
}

async function createConversation() {
  const convo = await api(`${API_BASE}/conversations`, {
    method: "POST",
    body: JSON.stringify({ avatarId: state.selectedAvatarId }),
  });
  state.conversationId = convo.id;
  ui.conversationId.textContent = convo.id;
}

function renderTrainingDocuments(docs) {
  ui.trainingDocList.innerHTML = "";
  if (!docs.length) {
    ui.trainingDocList.textContent = "No books uploaded yet.";
    return;
  }
  docs.forEach((doc) => {
    const row = document.createElement("div");
    row.className = "training-doc-item";
    row.textContent = `${doc.filename} (${doc.characters} chars)`;
    ui.trainingDocList.appendChild(row);
  });
}

function applyTrainingStatus(status) {
  ui.trainingDocCount.textContent = String(status.documents || 0);
  ui.trainingChunkCount.textContent = String(status.chunks || 0);
  ui.trainingCharCount.textContent = String(status.totalChars || 0);
}

async function refreshTrainingPanel() {
  if (!state.selectedAvatarId) return;
  const avatarId = encodeURIComponent(state.selectedAvatarId);
  const [status, docs] = await Promise.all([
    api(`${API_BASE}/training/status?avatarId=${avatarId}`, { method: "GET" }),
    api(`${API_BASE}/training/documents?avatarId=${avatarId}`, { method: "GET" }),
  ]);
  applyTrainingStatus(status);
  renderTrainingDocuments(docs);
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      if (!result.includes(",")) {
        reject(new Error("file_read_failed"));
        return;
      }
      resolve(result.split(",", 2)[1]);
    };
    reader.onerror = () => reject(new Error("file_read_failed"));
    reader.readAsDataURL(file);
  });
}

async function uploadTrainingMaterial(file) {
  const fileBase64 = await fileToBase64(file);
  return api(`${API_BASE}/training/upload`, {
    method: "POST",
    body: JSON.stringify({
      avatarId: state.selectedAvatarId,
      filename: file.name,
      fileBase64,
    }),
  });
}

async function ensureCallContext() {
  if (!state.conversationId) await createConversation();
}

async function ensureGuestSession() {
  if (state.userId) return;
  const guestEmail = `guest-${Date.now()}@local.dev`;
  await signup(guestEmail);
}

async function ensureAiReady({ force = false } = {}) {
  const now = Date.now();
  if (!force && state.aiReady && now - state.aiLastCheckedAt < AI_HEALTH_CACHE_MS) {
    setCallStatus("ready");
    return;
  }

  if (state.aiCheckInFlight) {
    await state.aiCheckInFlight;
    return;
  }

  state.aiCheckInFlight = (async () => {
    const aiHealth = await api(`${API_BASE}/ai/health`, { method: "GET" });
    if (!aiHealth.ok) {
      state.aiReady = false;
      throw new Error(`AI unavailable: ${aiHealth.error || "unknown"}`);
    }
    state.aiReady = true;
    state.aiLastCheckedAt = Date.now();
    setCallStatus("ready");
  })();

  try {
    await state.aiCheckInFlight;
  } finally {
    state.aiCheckInFlight = null;
  }
}

async function startCall() {
  if (state.callActive || state.callStarting) {
    updateCallButtons();
    return;
  }
  state.callStarting = true;
  updateCallButtons();
  setCallStatus("starting...");

  let stream = null;
  try {
    const mediaPromise = navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    const prepPromise = Promise.all([ensureCallContext(), ensureAiReady()]);
    [stream] = await Promise.all([mediaPromise, prepPromise]);
    state.mediaStream = stream;
    state.callActive = true;
    updateCallButtons();
    clearConversationTranscript();
    setCallStatus("in call");
    setAvatarMood("neutral");
    showToast("Call started");
  } catch (err) {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
    setCallStatus("ready");
    throw err;
  } finally {
    state.callStarting = false;
    updateCallButtons();
  }

  if (!state.useBrowserStt) {
    ui.talkHint.textContent = "SpeechRecognition not supported in this browser";
    return;
  }

  await startContinuousListening();
}

function endCall() {
  state.callActive = false;
  updateCallButtons();
  stopContinuousListening();
  clearSpeechResponseTimer();
  clearSpeechBuffer();
  interruptAvatarPlayback();
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
  }
  setCallStatus("ended");
  ui.talkHint.textContent = "Continuous listening is off";
}

async function synthesizeAndPlay(text, { signal } = {}) {
  if (state.muted || !text.trim() || signal?.aborted || state.userStopRequested) {
    setAvatarSpeaking(false);
    return;
  }

  state.lastAssistantText = text;
  if (state.preferBrowserTts) {
    try {
      await speakWithBrowserVoice(text, { signal });
      return;
    } catch (err) {
      // continue
    }
  }

  try {
    if (signal?.aborted || state.userStopRequested) return;
    const tts = await api(`${API_BASE}/voice/tts`, {
      method: "POST",
      signal,
      body: JSON.stringify({ text, voiceId: state.selectedVoiceId || "alloy" }),
    });
    if (signal?.aborted || state.userStopRequested) return;
    ui.audioPlayer.src = `data:${tts.mimeType};base64,${tts.audioBase64}`;
    ui.audioPlayer.playbackRate = state.audioPlaybackRate;
    if ("preservesPitch" in ui.audioPlayer) ui.audioPlayer.preservesPitch = true;
    if ("webkitPreservesPitch" in ui.audioPlayer) ui.audioPlayer.webkitPreservesPitch = true;
    if (signal?.aborted || state.userStopRequested) return;
    pulseAvatarForAudio();
    await ui.audioPlayer.play();
    startAudioLipSyncFromElement();
  } catch (err) {
    if (err?.name === "AbortError" || signal?.aborted || state.userStopRequested) {
      return;
    }
    await speakWithBrowserVoice(text, { signal });
  }
}

function isConversationOrUserError(errorMessage) {
  return errorMessage.includes("conversation_not_found") || errorMessage.includes("user_not_found");
}

async function sendTurnFallback(userText) {
  try {
    const signal = state.currentTurnAbortController?.signal;
    return await api(`${API_BASE}/conversations/${state.conversationId}/messages`, {
      method: "POST",
      signal,
      body: JSON.stringify({ text: userText }),
    });
  } catch (err) {
    if (err?.name === "AbortError") {
      throw err;
    }
    if (!isConversationOrUserError(String(err.message || err))) {
      throw err;
    }
    await ensureGuestSession();
    await createConversation();
    const signal = state.currentTurnAbortController?.signal;
    return api(`${API_BASE}/conversations/${state.conversationId}/messages`, {
      method: "POST",
      signal,
      body: JSON.stringify({ text: userText }),
    });
  }
}

async function runTurn(userText) {
  const clean = userText.trim();
  if (!clean) return;
  if (shouldTriggerStop(clean)) {
    requestImmediateStop("voice");
    return;
  }

  if (state.processingTurn) {
    state.pendingUserText = clean;
    return;
  }

  appendConversationTranscript("Student", clean);

  const requestController = new AbortController();
  state.currentTurnAbortController = requestController;
  state.userStopRequested = false;
  state.processingTurn = true;
  setAvatarMood("thinking");
  ui.liveCaption.textContent = "Avatar is thinking...";

  try {
    const turn = await sendTurnFallback(clean);
    if (requestController.signal.aborted || state.userStopRequested || !state.callActive) {
      return;
    }
    const finalText = turn.assistantMessage?.text || "";
    appendConversationTranscript(TEACHER_NAME, finalText);
    setAvatarMood(inferMoodFromReply(finalText));
    ui.liveCaption.textContent = "Avatar is speaking...";
    await synthesizeAndPlay(finalText, { signal: requestController.signal });
    if (requestController.signal.aborted || state.userStopRequested || !state.callActive) {
      return;
    }
    ui.liveCaption.textContent = "Voice-only mode: avatar responses are spoken.";
  } catch (err) {
    if (err?.name !== "AbortError") {
      showToast(err.message);
    }
  } finally {
    if (state.currentTurnAbortController === requestController) {
      state.currentTurnAbortController = null;
    }
    state.processingTurn = false;
    if (state.userStopRequested) {
      state.userStopRequested = false;
      return;
    }
    if (state.pendingUserText) {
      const queued = state.pendingUserText;
      state.pendingUserText = "";
      await runTurn(queued);
    }
  }
}

function isLikelyEcho(text) {
  const userText = text.toLowerCase().trim();
  const assistant = state.lastAssistantText.toLowerCase().trim();
  if (!assistant || userText.length < 8) return false;
  return assistant.includes(userText) || userText.includes(assistant.slice(0, 30));
}

function stopContinuousListening() {
  state.listeningActive = false;
  clearSpeechResponseTimer();
  clearSpeechBuffer();
  resetStudentTranscriptUI();
  if (state.speechRecognition) {
    state.speechRecognition.onend = null;
    state.speechRecognition.onerror = null;
    state.speechRecognition.onresult = null;
    state.speechRecognition.onspeechstart = null;
    try {
      state.speechRecognition.stop();
    } catch (_err) {
      // no-op
    }
    state.speechRecognition = null;
  }
  if (ui.holdToTalk) {
    ui.holdToTalk.classList.remove("recording");
    ui.holdToTalk.textContent = "Start Listening";
  }
}

function startRecognitionSession() {
  const Ctor = getSpeechRecognitionCtor();
  if (!Ctor) {
    throw new Error("SpeechRecognition not supported");
  }

  const recognition = new Ctor();
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.continuous = true;

  recognition.onresult = (event) => {
    let finalText = "";
    let interimText = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      if (result.isFinal) {
        finalText += result[0].transcript;
      } else {
        interimText += result[0].transcript;
      }
    }
    const combined = `${interimText} ${finalText}`.trim();
    const interim = interimText.trim();
    const interimEcho = interim ? isLikelyEcho(interim) : false;
    const combinedEcho = combined ? isLikelyEcho(combined) : false;
    if (
      (interim && shouldTriggerStop(interim) && !interimEcho) ||
      (combined && shouldTriggerStop(combined) && !combinedEcho)
    ) {
      requestImmediateStop("voice");
      return;
    }
    const avatarBusy = isAvatarBusy();
    if (avatarBusy) {
      if (interim || finalText.trim()) {
        ui.talkHint.textContent = "SP Sir is speaking. Say STOP to interrupt.";
      }
      return;
    }
    if (interim) {
      state.studentInterimTranscript = interim;
      updateStudentTranscriptUI();
      // User is still speaking: keep extending silence timer window.
      if (state.speechBuffer.trim()) {
        scheduleBufferedSpeechResponse();
      }
      ui.talkHint.textContent = "listening...";
    }
    const text = finalText.trim();
    if (!text || state.muted || !state.callActive) return;
    const echoText = isLikelyEcho(text);
    if (shouldTriggerStop(text) && !echoText) {
      requestImmediateStop("voice");
      return;
    }
    if (echoText) {
      return;
    }

    const existingTranscript = state.studentFinalTranscript.trim();
    state.studentFinalTranscript = existingTranscript ? `${existingTranscript} ${text}` : text;
    state.studentInterimTranscript = "";
    updateStudentTranscriptUI();

    const existing = state.speechBuffer.trim();
    state.speechBuffer = existing ? `${existing} ${text}` : text;
    scheduleBufferedSpeechResponse();
  };

  recognition.onspeechstart = () => {
    if (!state.callActive || !state.listeningActive) return;
    const avatarSpeaking = isAvatarBusy();
    if (!avatarSpeaking && !state.speechBuffer.trim()) {
      resetStudentTranscriptUI();
    }
    ui.talkHint.textContent = avatarSpeaking ? "SP Sir is speaking. Say STOP to interrupt." : "you are speaking...";
  };

  recognition.onerror = () => {
    if (!state.callActive || !state.listeningActive) return;
    ui.talkHint.textContent = "listening recovered";
  };

  recognition.onend = () => {
    if (!state.callActive || !state.listeningActive) return;
    // Keep continuous listening alive.
    startRecognitionSession();
  };

  state.speechRecognition = recognition;
  recognition.start();
}

async function startContinuousListening() {
  if (!state.callActive) {
    showToast("Start call first");
    return;
  }
  if (!state.useBrowserStt) {
    showToast("SpeechRecognition unavailable");
    return;
  }
  if (state.listeningActive) return;

  state.listeningActive = true;
  resetStudentTranscriptUI();
  if (ui.holdToTalk) {
    ui.holdToTalk.classList.add("recording");
    ui.holdToTalk.textContent = "Stop Listening";
  }
  ui.talkHint.textContent = "continuous listening on";
  startRecognitionSession();
}

function toggleMute() {
  state.muted = !state.muted;
  ui.muteMic.textContent = state.muted ? "Unmute Mic" : "Mute Mic";
  if (state.mediaStream) {
    state.mediaStream.getAudioTracks().forEach((track) => {
      track.enabled = !state.muted;
    });
  }
}

ui.signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await signup(ui.emailInput.value.trim());
    await createConversation();
    await refreshTrainingPanel();
    showToast("Session ready");
  } catch (err) {
    showToast(err.message);
  }
});

ui.refreshAvatars.addEventListener("click", async () => {
  try {
    await loadAvatars();
    await refreshTrainingPanel();
  } catch (err) {
    showToast(err.message);
  }
});

ui.trainingForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    if (!state.selectedAvatarId) {
      showToast("Choose avatar first");
      return;
    }
    const file = ui.trainingFile.files?.[0];
    if (!file) {
      showToast("Select a PDF/book file");
      return;
    }
    ui.uploadTraining.disabled = true;
    const result = await uploadTrainingMaterial(file);
    ui.trainingFile.value = "";
    await refreshTrainingPanel();
    showToast(`Uploaded ${result.filename}`);
  } catch (err) {
    showToast(err.message);
  } finally {
    ui.uploadTraining.disabled = false;
  }
});

ui.refreshTraining.addEventListener("click", async () => {
  try {
    await refreshTrainingPanel();
  } catch (err) {
    showToast(err.message);
  }
});

ui.clearTraining.addEventListener("click", async () => {
  try {
    if (!state.selectedAvatarId) {
      showToast("Choose avatar first");
      return;
    }
    const avatarId = encodeURIComponent(state.selectedAvatarId);
    const result = await api(`${API_BASE}/training/documents?avatarId=${avatarId}`, { method: "DELETE" });
    await refreshTrainingPanel();
    showToast(`Cleared ${result.deleted} document(s)`);
  } catch (err) {
    showToast(err.message);
  }
});

ui.startCall.addEventListener("click", async () => {
  try {
    await startCall();
  } catch (err) {
    updateCallButtons();
    showToast(err.message);
  }
});

ui.endCall.addEventListener("click", endCall);
ui.muteMic.addEventListener("click", toggleMute);
ui.testVoice.addEventListener("click", async () => {
  try {
    await synthesizeAndPlay("Hello. Full duplex voice test is active.");
    showToast("Voice test played");
  } catch (err) {
    showToast(`Voice failed: ${err.message}`);
  }
});

ui.audioPlayer.addEventListener("play", () => {
  pulseAvatarForAudio();
  startAudioLipSyncFromElement();
});
ui.audioPlayer.addEventListener("pause", stopAvatarPulse);
ui.audioPlayer.addEventListener("ended", stopAvatarPulse);
ui.audioPlayer.addEventListener("error", stopAvatarPulse);

updateCallButtons();
updateStudentTranscriptUI();
renderConversationTranscriptUI();

(async function boot() {
  try {
    await ensureGuestSession();
    await loadAvatars();
    if (!state.conversationId && state.selectedAvatarId) {
      await createConversation();
    }
    await refreshTrainingPanel();
    await ensureAiReady();
    setAvatarMood("neutral");
    resetMouthShape();
    updateCallButtons();
  } catch (err) {
    updateCallButtons();
    showToast(`Boot failed: ${err.message}`);
  }
})();
