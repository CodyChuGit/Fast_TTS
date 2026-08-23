<script lang="ts">
  import { onDestroy, onMount } from 'svelte';
  import {
    browserDecodeToWav,
    closePcm16Playback,
    encodePcm16BytesWav,
    Pcm16StreamPlayer,
    primePcm16Playback
  } from '$lib/audio';
  import {
    activateCharacter,
    base64ToBytes,
    getLlmSettings,
    resetLlmSettings,
    setLlmSettings,
    type LlmSettings,
    character as fetchCharacter,
    chatSpeak,
    prewarmChat,
    speculateChat,
    deleteCharacter,
    health,
    listCharacters,
    mcpEndpoint,
    setCharacterCustom,
    setCharacterPreset,
    speakStream,
    updateCharacter,
    type Character,
    type ChatMessage,
    type ChatStats,
    type SavedCharacterEntry,
    type ServerHealth,
    type SpeakStats
  } from '$lib/client';
  import {
    CaptionTimeline,
    SentenceCaptionTrack,
    type CaptionWord,
    type SentenceCaption
  } from '$lib/karaoke';
  import '../app.css';

  let view: 'speak' | 'chat' | 'settings' | 'live' | 'voice' = 'speak';
  let server: ServerHealth | null = null;
  let active: Character | null = null;

  // Speak view.
  let text = '';
  let speaking = false;
  let status = '';
  let errorStatus = '';
  let stats: SpeakStats | null = null;
  let lastWavUrl = '';
  let aborter: AbortController | null = null;
  let player: Pcm16StreamPlayer | null = null;

  // Voice view: live streaming STT. The mic is captured at 16 kHz through an
  // AudioWorklet, shipped to the server as small sequential PCM posts (a
  // browser cannot stream a request body over HTTP/1.1), and the transcript
  // events come back over one SSE connection. Solid text has stabilized
  // across ASR re-decodes; dimmed text is the still-revising tail.
  type VoiceTurn = {
    key: string;
    kind: 'user' | 'assistant';
    stable: string;
    tentative: string;
    final?: string;
    text?: string; // assistant reply accumulator
    timing?: { first_partial_ms: number; eot_to_final_ms: number };
  };
  let voiceActive = false;
  let voiceSpeechActive = false;
  let voiceConverse = true;
  let voiceError = '';
  let voiceTurns: VoiceTurn[] = [];
  let voiceSessionId = '';
  let voiceEvents: EventSource | null = null;
  let voiceStream: MediaStream | null = null;
  let voiceContext: AudioContext | null = null;
  let voiceNode: AudioWorkletNode | null = null;
  let voicePending: Float32Array[] = [];
  let voiceFlushTimer: ReturnType<typeof setInterval> | null = null;
  let voicePostChain: Promise<unknown> = Promise.resolve();
  // Conversation loop: her reply plays while the mic stays hot; speaking
  // over her cuts the reply (barge-in), and every stable-prefix update
  // speculates her reply on the server so it is often ready before the
  // final transcript exists.
  let voiceChat: ChatMessage[] = [];
  let voiceReplyPlayer: Pcm16StreamPlayer | null = null;
  let voiceReplyAborter: AbortController | null = null;
  let voiceSpeakingReply = false;
  let voiceReplyFinal = true;
  let voiceReplyCounter = 0;
  let lastVoiceSpeculated = '';

  function voiceTurn(id: number): VoiceTurn {
    const key = `u${id}`;
    let turn = voiceTurns.find((t) => t.key === key);
    if (!turn) {
      turn = { key, kind: 'user', stable: '', tentative: '' };
      voiceTurns = [...voiceTurns, turn];
      if (voiceTurns.length > 24) voiceTurns = voiceTurns.slice(-24);
    }
    return turn;
  }

  function voiceMessages(draft: string): ChatMessage[] {
    return [...voiceChat, { role: 'user' as const, content: draft }].slice(-24);
  }

  function stopVoiceReply() {
    voiceReplyAborter?.abort();
    voiceReplyAborter = null;
    void voiceReplyPlayer?.stop();
    voiceReplyPlayer = null;
    voiceSpeakingReply = false;
  }

  async function sendVoiceTurn(content: string) {
    stopVoiceReply();
    voiceChat = [...voiceChat, { role: 'user', content }];
    const reply: VoiceTurn = {
      key: `a${voiceReplyCounter++}`,
      kind: 'assistant',
      stable: '',
      tentative: '',
      text: ''
    };
    voiceTurns = [...voiceTurns, reply];
    voiceSpeakingReply = true;
    voiceReplyFinal = false;
    voiceReplyAborter = new AbortController();
    try {
      await primePcm16Playback();
      voiceReplyPlayer = new Pcm16StreamPlayer(24000, 1, {
        startBufferSeconds: 0.12,
        startMaxWaitMs: 250
      });
      await voiceReplyPlayer.start();
      const history = voiceChat.slice(-24).map(({ role, content: c }) => ({ role, content: c }));
      await chatSpeak(
        history,
        (event) => {
          if (event.type === 'token') {
            reply.text = (reply.text ?? '') + event.text;
            voiceTurns = voiceTurns;
          } else if (event.type === 'audio') {
            voiceReplyPlayer?.push(base64ToBytes(event.audio));
          } else if (event.type === 'llm_done') {
            voiceReplyFinal = true;
          } else if (event.type === 'error') {
            voiceError = event.message;
          }
        },
        voiceReplyAborter.signal
      );
      await voiceReplyPlayer?.finish();
      if (reply.text?.trim()) {
        voiceChat = [...voiceChat, { role: 'assistant', content: reply.text }];
      }
    } catch (error) {
      if ((error as Error)?.name !== 'AbortError') {
        voiceError = error instanceof Error ? error.message : String(error);
      } else if (reply.text?.trim()) {
        // Barged-in mid-reply: keep what she actually said.
        voiceChat = [...voiceChat, { role: 'assistant', content: reply.text }];
      }
    } finally {
      void voiceReplyPlayer?.stop();
      voiceReplyPlayer = null;
      voiceSpeakingReply = false;
      voiceReplyFinal = true;
    }
  }

  async function startVoice() {
    voiceError = '';
    voiceTurns = [];
    try {
      const created = await fetch('/v1/voice/sessions', { method: 'POST' });
      if (!created.ok) throw new Error(`voice session: ${created.status} ${await created.text()}`);
      voiceSessionId = (await created.json()).id;

      voiceEvents = new EventSource(`/v1/voice/sessions/events?id=${voiceSessionId}`);
      voiceEvents.onmessage = (message) => {
        if (message.data === '[DONE]') return;
        try {
          const event = JSON.parse(message.data);
          if (event.type === 'speech_started') {
            voiceSpeechActive = true;
            voiceTurn(event.utterance_id);
            // Barge-in: the user talking over her wins instantly.
            if (voiceSpeakingReply) stopVoiceReply();
          } else if (event.type === 'stable_transcript') {
            voiceTurn(event.utterance_id).stable = event.text;
            voiceTurns = voiceTurns;
            // Speculate her reply on the stabilized prefix while the user
            // is still speaking; the punctuation-tolerant attach means the
            // final send usually finds it ready.
            if (
              voiceConverse &&
              voiceReplyFinal &&
              event.text &&
              event.text !== lastVoiceSpeculated
            ) {
              lastVoiceSpeculated = event.text;
              speculateChat(voiceMessages(event.text));
            }
          } else if (event.type === 'partial_transcript') {
            voiceTurn(event.utterance_id).tentative = event.text;
            voiceTurns = voiceTurns;
          } else if (event.type === 'speech_ended') {
            voiceSpeechActive = false;
          } else if (event.type === 'final_transcript') {
            const turn = voiceTurn(event.utterance_id);
            turn.final = event.text;
            turn.tentative = '';
            turn.timing = event.timings;
            voiceTurns = voiceTurns;
            lastVoiceSpeculated = '';
            if (voiceConverse && event.text?.trim()) {
              void sendVoiceTurn(event.text);
            }
          } else if (event.type === 'error') {
            voiceError = event.message;
          }
        } catch {
          // Malformed frame; keep listening.
        }
      };

      voiceStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
      });
      voiceContext = new AudioContext({ sampleRate: 16000 });
      const workletUrl = URL.createObjectURL(
        new Blob(
          [
            `class VoiceCapture extends AudioWorkletProcessor {
               process(inputs) {
                 const ch = inputs[0] && inputs[0][0];
                 if (ch) this.port.postMessage(ch.slice(0));
                 return true;
               }
             }
             registerProcessor('voice-capture', VoiceCapture);`
          ],
          { type: 'application/javascript' }
        )
      );
      await voiceContext.audioWorklet.addModule(workletUrl);
      URL.revokeObjectURL(workletUrl);
      const source = voiceContext.createMediaStreamSource(voiceStream);
      voiceNode = new AudioWorkletNode(voiceContext, 'voice-capture');
      voiceNode.port.onmessage = (message) => {
        voicePending.push(message.data as Float32Array);
      };
      source.connect(voiceNode);

      // Ship ~128 ms batches, strictly in order.
      voiceFlushTimer = setInterval(() => {
        if (voicePending.length === 0 || !voiceSessionId) return;
        const chunks = voicePending;
        voicePending = [];
        const total = chunks.reduce((n, c) => n + c.length, 0);
        const pcm = new Int16Array(total);
        let offset = 0;
        for (const chunk of chunks) {
          for (let i = 0; i < chunk.length; i += 1) {
            const v = Math.max(-1, Math.min(1, chunk[i]));
            pcm[offset + i] = v < 0 ? v * 32768 : v * 32767;
          }
          offset += chunk.length;
        }
        const id = voiceSessionId;
        voicePostChain = voicePostChain.then(() =>
          fetch(`/v1/voice/sessions/audio?id=${id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: pcm.buffer
          }).catch(() => {})
        );
      }, 128);
      voiceActive = true;
    } catch (error) {
      voiceError = error instanceof Error ? error.message : String(error);
      await stopVoice();
    }
  }

  async function stopVoice() {
    voiceActive = false;
    voiceSpeechActive = false;
    stopVoiceReply();
    if (voiceFlushTimer !== null) {
      clearInterval(voiceFlushTimer);
      voiceFlushTimer = null;
    }
    voiceNode?.disconnect();
    voiceNode = null;
    voiceStream?.getTracks().forEach((track) => track.stop());
    voiceStream = null;
    if (voiceContext) {
      try {
        await voiceContext.close();
      } catch {
        // Already closed.
      }
      voiceContext = null;
    }
    if (voiceSessionId) {
      const id = voiceSessionId;
      voiceSessionId = '';
      // Let queued audio land, then close the turn server-side.
      voicePostChain.then(() => fetch(`/v1/voice/sessions/stop?id=${id}`, { method: 'POST' }).catch(() => {}));
    }
    // Keep the SSE open briefly so the last final_transcript arrives.
    const events = voiceEvents;
    voiceEvents = null;
    if (events) setTimeout(() => events.close(), 4000);
  }

  // Live view: the minimalist sing-along stage. Words of the input light up
  // exactly as the voice reaches them, driven by the player's playback clock
  // against a caption timeline that snaps to the true clip length at the end.
  let liveText = '';
  let liveSpeaking = false;
  let liveDone = false;
  let liveError = '';
  let liveWavUrl = '';
  let liveAborter: AbortController | null = null;
  let livePlayer: Pcm16StreamPlayer | null = null;
  let liveTimeline: CaptionTimeline | null = null;
  let liveWords: CaptionWord[] = [];
  // Counts syllables, not words -- the reveal advances syllable by syllable.
  let liveVisible = 0;
  let liveRaf = 0;
  let liveStage: HTMLElement | null = null;
  let liveScrollQueued = false;

  // Settings view. The form is populated from the server once (and again after
  // each save), never on navigation -- switching to Speak and back must not
  // discard what the user was editing.
  let formReady = false;
  let editName = '';
  let voiceSource: 'preset' | 'custom' = 'preset';
  let editPreset = '';
  let editPersona = '';
  let customFile: File | null = null;
  let customTranscript = '';
  let saving = false;
  let settingsStatus = '';
  let settingsError = '';
  let recorder: MediaRecorder | null = null;
  let recordingStream: MediaStream | null = null;
  let recording = false;
  let library: SavedCharacterEntry[] = [];
  let switching = '';
  let sampleSeconds: number | null = null;
  let voiceRev = 0;
  let testing = false;

  // Chat view.
  let chatMessages: Array<ChatMessage & { speaking?: boolean }> = [];
  let chatInput = '';
  let chatBusy = false;
  let chatError = '';
  let chatStats: ChatStats | null = null;
  let chatAborter: AbortController | null = null;
  let chatPlayer: Pcm16StreamPlayer | null = null;
  let chatLog: HTMLElement | null = null;
  let chatScrollQueued = false;
  // Spoken-sync subtitles for the reply: the server sends each sentence's text
  // before its audio, so the caption bar tracks the voice exactly.
  let chatTrack: SentenceCaptionTrack | null = null;
  let chatCaption: SentenceCaption | null = null;
  let chatRaf = 0;

  // Roleplay engine settings.
  let llm: LlmSettings | null = null;
  let llmSaving = false;
  let llmStatus = '';
  let llmError = '';
  // The dropdown's selection, kept apart from llm.model so picking a model
  // does nothing until "Switch model" — a switch restarts the sidecar and can
  // take minutes for a big file.
  let llmModelChoice = '';
  let llmSwitching = false;

  // MCP panel.
  let endpoint = '';
  let copied = false;

  $: sampleNote = sampleSeconds === null
    ? ''
    : sampleSeconds > 20
      ? `This sample is ${Math.round(sampleSeconds)} s long. Every second of reference adds roughly 11 ms to EVERY reply's first audio, and cloning copies everything — length, pauses, background noise. 5–12 s of clean, music-free speech clones better and speaks sooner.`
      : sampleSeconds < 3
        ? 'This sample is very short; 5–15 s of clean speech clones better.'
        : '';

  function populateForm(from: Character | null) {
    editName = from?.name || '';
    voiceSource = from?.source === 'custom' ? 'custom' : 'preset';
    editPreset = from?.preset || from?.available_presets?.[0] || '';
    editPersona = from?.persona || '';
    customTranscript = from?.transcript || '';
    customFile = null;
    formReady = true;
  }

  async function refresh() {
    try {
      server = await health();
    } catch {
      server = null;
    }
    try {
      active = await fetchCharacter();
      if (!formReady) populateForm(active);
    } catch {
      active = null;
    }
    try {
      library = (await listCharacters()).characters;
    } catch {
      library = [];
    }
    if (!llm && server?.llm) {
      try {
        llm = await getLlmSettings();
        llmModelChoice = llm.model;
      } catch {
        llm = null;
      }
    }
  }

  async function saveLlmSettings() {
    if (!llm || llmSaving) return;
    llmSaving = true;
    llmError = '';
    llmStatus = 'Saving…';
    try {
      // Send only the editable fields: including `model` here would restart
      // the sidecar as a side effect of saving a prompt tweak.
      const { model, models, ...editable } = llm;
      llm = await setLlmSettings(editable);
      llmModelChoice = llm.model;
      llmStatus = 'Roleplay settings saved.';
    } catch (error) {
      llmError = error instanceof Error ? error.message : String(error);
      llmStatus = '';
    } finally {
      llmSaving = false;
    }
  }

  async function restoreLlmDefaults() {
    if (llmSaving) return;
    llmSaving = true;
    llmError = '';
    try {
      llm = await resetLlmSettings();
      llmModelChoice = llm.model;
      llmStatus = 'Roleplay defaults restored.';
    } catch (error) {
      llmError = error instanceof Error ? error.message : String(error);
    } finally {
      llmSaving = false;
    }
  }

  async function switchLlmModel() {
    if (!llm || llmSwitching || !llmModelChoice || llmModelChoice === llm.model) return;
    llmSwitching = true;
    llmError = '';
    llmStatus = '';
    try {
      const next = await setLlmSettings({ model: llmModelChoice });
      llm = next;
      llmModelChoice = next.model;
      const name = next.models.find((option) => option.id === next.model)?.name ?? next.model;
      llmStatus = `${name} is loaded and ready to chat.`;
    } catch (error) {
      llmError = error instanceof Error ? error.message : String(error);
    } finally {
      llmSwitching = false;
    }
  }

  async function useCharacter(entry: SavedCharacterEntry) {
    if (switching) return;
    switching = entry.id;
    settingsError = '';
    try {
      active = await activateCharacter(entry.id);
      populateForm(active);
      voiceRev += 1;
      library = (await listCharacters()).characters;
      settingsStatus = `Now speaking as ${entry.name}.`;
    } catch (error) {
      settingsError = error instanceof Error ? error.message : String(error);
    } finally {
      switching = '';
    }
  }

  async function removeCharacter(entry: SavedCharacterEntry) {
    if (switching) return;
    switching = entry.id;
    settingsError = '';
    try {
      library = (await deleteCharacter(entry.id)).characters;
      settingsStatus = `Deleted ${entry.name}.`;
    } catch (error) {
      settingsError = error instanceof Error ? error.message : String(error);
    } finally {
      switching = '';
    }
  }

  async function speak() {
    const input = text.trim();
    if (!input || speaking) return;
    speaking = true;
    errorStatus = '';
    stats = null;
    status = 'Generating…';
    aborter = new AbortController();
    try {
      await primePcm16Playback();
      player = new Pcm16StreamPlayer(24000, 1);
      await player.start();
      const result = await speakStream(input, null, (chunk) => {
        player?.push(chunk);
        if (status !== 'Speaking…') status = 'Speaking…';
      }, aborter.signal);
      status = 'Finishing…';
      await player.finish();
      stats = result.stats;
      if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
      lastWavUrl = URL.createObjectURL(encodePcm16BytesWav(result.chunks, 24000, 1));
      status = '';
    } catch (error) {
      if ((error as Error)?.name === 'AbortError') {
        status = 'Stopped.';
      } else {
        errorStatus = error instanceof Error ? error.message : String(error);
        status = '';
      }
    } finally {
      await player?.stop();
      player = null;
      speaking = false;
      aborter = null;
    }
  }

  function stop() {
    aborter?.abort();
  }

  function stopLiveLoop() {
    if (liveRaf) cancelAnimationFrame(liveRaf);
    liveRaf = 0;
  }

  // Keeps the word being spoken in the middle of the stage without ever
  // scrolling the page itself.
  function queueLiveScroll() {
    if (liveScrollQueued) return;
    liveScrollQueued = true;
    requestAnimationFrame(() => {
      liveScrollQueued = false;
      const stage = liveStage;
      if (!stage) return;
      let current = stage.querySelector<HTMLElement>('.syl.current');
      if (!current) {
        const shown = stage.querySelectorAll<HTMLElement>('.syl.shown');
        current = shown.length ? shown[shown.length - 1] : null;
      }
      if (!current) return;
      const target = current.offsetTop - stage.clientHeight * 0.45;
      if (Math.abs(stage.scrollTop - target) > 8) {
        stage.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
      }
    });
  }

  function startLiveLoop() {
    stopLiveLoop();
    const step = () => {
      if (livePlayer && liveTimeline) {
        const count = liveTimeline.update(livePlayer.playedSeconds());
        if (count !== liveVisible) {
          liveVisible = count;
          queueLiveScroll();
        }
      }
      liveRaf = requestAnimationFrame(step);
    };
    liveRaf = requestAnimationFrame(step);
  }

  async function liveSpeak() {
    const input = liveText.trim();
    if (!input || liveSpeaking) return;
    liveSpeaking = true;
    liveDone = false;
    liveError = '';
    if (liveWavUrl) {
      URL.revokeObjectURL(liveWavUrl);
      liveWavUrl = '';
    }
    liveTimeline = new CaptionTimeline(input);
    liveWords = liveTimeline.words;
    liveVisible = 0;
    liveStage?.scrollTo({ top: 0 });
    liveAborter = new AbortController();
    try {
      await primePcm16Playback();
      livePlayer = new Pcm16StreamPlayer(24000, 1);
      await livePlayer.start();
      startLiveLoop();
      const result = await speakStream(input, null, (chunk) => livePlayer?.push(chunk), liveAborter.signal);
      // The true clip length is now known; the words still ahead re-pace to it.
      liveTimeline.finish(result.stats.audioSeconds);
      await livePlayer.finish();
      liveVisible = liveTimeline.revealAll();
      liveWavUrl = URL.createObjectURL(encodePcm16BytesWav(result.chunks, 24000, 1));
      liveDone = true;
    } catch (error) {
      if ((error as Error)?.name !== 'AbortError') {
        liveError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      stopLiveLoop();
      await livePlayer?.stop();
      livePlayer = null;
      liveSpeaking = false;
      liveAborter = null;
    }
  }

  function liveStop() {
    liveAborter?.abort();
  }

  function liveClear() {
    if (liveSpeaking) return;
    liveWords = [];
    liveVisible = 0;
    liveDone = false;
    liveError = '';
    liveTimeline = null;
  }

  function liveKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void liveSpeak();
    }
  }

  function recordingMimeType() {
    if (typeof MediaRecorder === 'undefined') return '';
    for (const type of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4']) {
      if (MediaRecorder.isTypeSupported?.(type)) return type;
    }
    return '';
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      settingsError = 'Microphone recording is not supported by this browser.';
      return;
    }
    if (recorder) return;
    try {
      recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const mimeType = recordingMimeType();
      recorder = new MediaRecorder(recordingStream, mimeType ? { mimeType } : undefined);
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder?.mimeType || mimeType || 'audio/webm' });
        probeSample(new File([blob], 'recording.webm', { type: blob.type }));
        recordingStream?.getTracks().forEach((track) => track.stop());
        recordingStream = null;
        recorder = null;
        recording = false;
        settingsStatus = 'Recording captured.';
      };
      recorder.start();
      recording = true;
      settingsError = '';
      settingsStatus = 'Recording… speak 5–15 seconds, then stop.';
    } catch (error) {
      settingsError = error instanceof Error ? error.message : String(error);
      recordingStream?.getTracks().forEach((track) => track.stop());
      recordingStream = null;
      recorder = null;
      recording = false;
    }
  }

  function stopRecording() {
    if (recorder?.state === 'recording') recorder.stop();
  }

  // Measures a chosen or recorded sample so quality problems are visible
  // before saving: cloning copies everything in the reference, so length and
  // background noise matter more than people expect.
  function probeSample(file: File | null) {
    customFile = file;
    sampleSeconds = null;
    if (!file) return;
    const url = URL.createObjectURL(file);
    const probe = new Audio();
    probe.preload = 'metadata';
    probe.onloadedmetadata = () => {
      sampleSeconds = Number.isFinite(probe.duration) ? probe.duration : null;
      URL.revokeObjectURL(url);
    };
    probe.onerror = () => URL.revokeObjectURL(url);
    probe.src = url;
  }

  async function testVoice() {
    if (speaking || testing) return;
    testing = true;
    settingsError = '';
    settingsStatus = 'Speaking a test line with the active voice…';
    aborter = new AbortController();
    try {
      await primePcm16Playback();
      player = new Pcm16StreamPlayer(24000, 1);
      await player.start();
      await speakStream(
        'Here is how the active character sounds right now. If this does not match the reference, try a shorter, cleaner sample.',
        null,
        (chunk) => player?.push(chunk),
        aborter.signal
      );
      await player.finish();
      settingsStatus = '';
    } catch (error) {
      settingsError = error instanceof Error ? error.message : String(error);
      settingsStatus = '';
    } finally {
      await player?.stop();
      player = null;
      testing = false;
      aborter = null;
    }
  }

  async function saveCharacter() {
    if (saving) return;
    saving = true;
    settingsError = '';
    settingsStatus = 'Saving…';
    try {
      if (voiceSource === 'preset') {
        if (!editPreset) throw new Error('Choose a bundled voice.');
        active = await setCharacterPreset(editName, editPreset, editPersona.trim());
      } else if (customFile) {
        // A new recording replaces the voice.
        if (!customTranscript.trim()) {
          throw new Error('Enter the transcript of what the sample says — cloning quality depends on it.');
        }
        const wav = await browserDecodeToWav(customFile);
        active = await setCharacterCustom(editName, customTranscript.trim(), wav, editPersona.trim());
      } else if (active?.source === 'custom') {
        // No new recording: keep the saved voice, update name and transcript.
        active = await updateCharacter(editName, customTranscript.trim() || undefined, editPersona.trim());
      } else {
        throw new Error('Choose or record a voice sample.');
      }
      populateForm(active);
      voiceRev += 1;
      try {
        library = (await listCharacters()).characters;
      } catch {
        // The save succeeded; a stale list corrects on the next poll.
      }
      settingsStatus = 'Character saved.';
    } catch (error) {
      settingsError = error instanceof Error ? error.message : String(error);
      settingsStatus = '';
    } finally {
      saving = false;
    }
  }

  async function copyEndpoint() {
    try {
      await navigator.clipboard.writeText(endpoint);
      copied = true;
      setTimeout(() => (copied = false), 1500);
    } catch {
      // Clipboard can be unavailable; the URL is visible and selectable anyway.
    }
  }

  // While the user types, the server prefills the draft into the LLM's
  // prompt cache on a debounce, so send finds the prompt already resident
  // and the first token is one decode step away.
  let chatPrewarmTimer: ReturnType<typeof setTimeout> | null = null;
  let chatSpeculateTimer: ReturnType<typeof setTimeout> | null = null;
  let lastPrewarmedDraft = '';
  let lastSpeculatedDraft = '';
  // The current reply's text is complete (its audio may still be playing).
  // From this point the conversation is final enough for the NEXT draft to
  // prewarm and speculate against — the user can type while she talks.
  let chatTextFinal = false;

  function draftMessages(draft: string) {
    // Mirrors sendChat's history exactly; any divergence would break the
    // cached prefix instead of extending it.
    return [...chatMessages, { role: 'user' as const, content: draft }]
      .slice(-24)
      .map(({ role, content }) => ({ role, content }));
  }

  // During a reply, drafting is allowed but speculation must wait until the
  // reply's text is final (llm_done) — before that, the history the server
  // would speculate against does not exist yet.
  function chatDraftReady(): boolean {
    return !chatBusy || chatTextFinal;
  }

  // Send-intent detection: the speculation delay shrinks with how finished
  // the draft looks. Sentence-final punctuation, trailing discourse closers
  // ("lol", "tho", "thanks"), Chinese sentence-final particles (吧/呢/吗…),
  // and trailing emoji are near-certain end signals; a question-shaped or
  // very short draft is likely complete the moment typing pauses. A wrong
  // guess only costs one cheaply aborted background generation, so the bias
  // is toward firing early.
  const SENTENCE_END = /[.!?…~。！？～][)\]"'’”』」]*$/;
  const TRAILING_CLOSER =
    /(?:\b(?:lol|lmao|haha+|hehe+|tho|though|right|pls|please|thanks|thank you|ty|ok|okay|kk|fr|ngl|tbh|imo|btw|you know|i guess|for real|bye|goodnight|night|see ya|cya)|[吧呢吗啊呀嘛啦哦哟咯了]|哈哈+|嘿嘿+|[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}])$/iu;
  const QUESTION_START =
    /^(?:what|where|when|why|how|who|which|whose|can|could|do|does|did|is|are|was|were|should|would|will|shall|am|have|has|any|got)\b/i;

  function speculateDelay(raw: string): number {
    // Shape-gated: a speculation costs one to two seconds of uninterruptible
    // sidecar prefill, so it fires fast ONLY when the draft looks sendable
    // (the pre-enter pause is exactly such a moment) and the server attaches
    // it even if trailing punctuation is typed afterwards. A mid-thought
    // draft only speculates on a long deliberate hover -- real-user testing
    // showed hot tiers on stall-heavy or Chinese typists (no spaces, so a
    // word count is meaningless) keep the slot permanently busy and push the
    // real send's first token past five seconds.
    const draft = raw.trimEnd();
    if (SENTENCE_END.test(draft) || TRAILING_CLOSER.test(draft)) return 120;
    const cjk = /[㐀-鿿]/.test(draft);
    if (!cjk && QUESTION_START.test(draft)) return 300;
    if (!cjk && draft.split(/\s+/).length <= 3) return 500;
    if (cjk && Array.from(draft).length <= 6) return 500;
    return 1500;
  }

  function queueChatPrewarm() {
    if (chatPrewarmTimer !== null) clearTimeout(chatPrewarmTimer);
    if (chatSpeculateTimer !== null) clearTimeout(chatSpeculateTimer);
    chatPrewarmTimer = setTimeout(() => {
      chatPrewarmTimer = null;
      const draft = chatInput.trim();
      if (!draft || !chatDraftReady() || draft === lastPrewarmedDraft) return;
      if (draft === lastSpeculatedDraft) return; // speculation already covers it
      lastPrewarmedDraft = draft;
      prewarmChat(draftMessages(draft));
    }, 200);
    chatSpeculateTimer = setTimeout(() => {
      chatSpeculateTimer = null;
      const draft = chatInput.trim();
      if (!draft || !chatDraftReady() || draft === lastSpeculatedDraft) return;
      lastSpeculatedDraft = draft;
      speculateChat(draftMessages(draft));
    }, speculateDelay(chatInput));
  }

  function stopChatCaptionLoop() {
    if (chatRaf) cancelAnimationFrame(chatRaf);
    chatRaf = 0;
  }

  function startChatCaptionLoop() {
    stopChatCaptionLoop();
    const step = () => {
      if (chatPlayer && chatTrack) {
        const next = chatTrack.captionAt(chatPlayer.playedSeconds());
        if (!next) {
          if (chatCaption) chatCaption = null;
        } else if (
          !chatCaption ||
          chatCaption.index !== next.index ||
          chatCaption.visible !== next.visible
        ) {
          chatCaption = next;
        }
      }
      chatRaf = requestAnimationFrame(step);
    };
    chatRaf = requestAnimationFrame(step);
  }

  async function sendChat() {
    const content = chatInput.trim();
    if (!content || chatBusy) return;
    if (chatPrewarmTimer !== null) {
      clearTimeout(chatPrewarmTimer);
      chatPrewarmTimer = null;
    }
    if (chatSpeculateTimer !== null) {
      clearTimeout(chatSpeculateTimer);
      chatSpeculateTimer = null;
    }
    lastPrewarmedDraft = '';
    lastSpeculatedDraft = '';
    chatTextFinal = false;
    chatBusy = true;
    chatError = '';
    chatStats = null;
    chatInput = '';
    chatMessages = [...chatMessages, { role: 'user', content }, { role: 'assistant', content: '', speaking: true }];
    chatAborter = new AbortController();
    try {
      await primePcm16Playback();
      // Measured on the current engine stack, TTS production never falls
      // behind playback any more -- the LLM finishes writing long before the
      // audio catches up -- so the chat reserve only needs to ride out
      // browser-side jitter. A shallow reserve starts replies sooner; the
      // player's rebuffer-on-underrun logic absorbs the rare mid-reply
      // graph-capture stall with a single clean pause.
      chatPlayer = new Pcm16StreamPlayer(24000, 1, {
        startBufferSeconds: 0.12,
        startMaxWaitMs: 250
      });
      await chatPlayer.start();
      chatTrack = new SentenceCaptionTrack();
      startChatCaptionLoop();
      // The full visible history goes up each turn; the sidecar's prompt cache
      // makes re-sending the prefix nearly free.
      const history = chatMessages
        .slice(0, -1)
        .slice(-24)
        .map(({ role, content: text }) => ({ role, content: text }));
      await chatSpeak(history, (event) => {
        if (event.type === 'token') {
          const last = chatMessages[chatMessages.length - 1];
          last.content += event.text;
          chatMessages = chatMessages;
          if (!chatScrollQueued) {
            chatScrollQueued = true;
            requestAnimationFrame(() => {
              chatScrollQueued = false;
              chatLog?.scrollTo({ top: chatLog.scrollHeight });
            });
          }
        } else if (event.type === 'sentence') {
          chatTrack?.addSentence(event.text);
        } else if (event.type === 'audio') {
          const pcm = base64ToBytes(event.audio);
          chatPlayer?.push(pcm);
          // 24 kHz mono s16le: 48000 bytes per second of speech.
          chatTrack?.addAudioSeconds(pcm.byteLength / 48000);
        } else if (event.type === 'error') {
          chatError = event.message;
        } else if (event.type === 'llm_done') {
          // Her text is final even though she is still speaking: anything
          // typed into the box so far can start prewarming and speculating
          // against the finished conversation right now.
          chatTextFinal = true;
          if (chatInput.trim()) queueChatPrewarm();
        } else if (event.type === 'done') {
          chatStats = event.stats;
        }
      }, chatAborter.signal);
      await chatPlayer.finish();
    } catch (error) {
      if ((error as Error)?.name !== 'AbortError') {
        chatError = error instanceof Error ? error.message : String(error);
      }
    } finally {
      stopChatCaptionLoop();
      chatCaption = null;
      chatTrack = null;
      await chatPlayer?.stop();
      chatPlayer = null;
      const last = chatMessages[chatMessages.length - 1];
      if (last) {
        last.speaking = false;
        if (!last.content) {
          chatMessages = chatMessages.slice(0, -1);
        } else {
          chatMessages = chatMessages;
        }
      }
      chatBusy = false;
      chatAborter = null;
    }
  }

  function stopChat() {
    chatAborter?.abort();
  }

  function clearChat() {
    if (chatBusy) return;
    chatMessages = [];
    chatStats = null;
    chatError = '';
  }

  onMount(() => {
    endpoint = mcpEndpoint(window.location.origin);
    void refresh();
    const poll = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(poll);
  });

  onDestroy(() => {
    aborter?.abort();
    liveAborter?.abort();
    void stopVoice();
    stopLiveLoop();
    stopChatCaptionLoop();
    if (recorder?.state === 'recording') recorder.stop();
    recordingStream?.getTracks().forEach((track) => track.stop());
    if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
    if (liveWavUrl) URL.revokeObjectURL(liveWavUrl);
    void closePcm16Playback();
  });
</script>

<svelte:head>
  <title>Super Fast TTS MCP Server</title>
</svelte:head>

<header class="topbar">
  <div class="brand">
    <div class="mark">⚡</div>
    <div>
      <strong>Super Fast TTS</strong>
      <span>MCP Server</span>
    </div>
  </div>
  <nav>
    <button class:active={view === 'speak'} on:click={() => (view = 'speak')}>Speak</button>
    <button class:active={view === 'live'} on:click={() => (view = 'live')}>Live</button>
    <button class:active={view === 'voice'} on:click={() => (view = 'voice')}>Voice</button>
    {#if server?.llm}
      <button class:active={view === 'chat'} on:click={() => (view = 'chat')}>Chat</button>
    {/if}
    <button class:active={view === 'settings'} on:click={() => (view = 'settings')}>Settings</button>
  </nav>
  <div class="server-pill" class:online={server?.status === 'ok'}>
    <i></i>{server ? server.backend : 'offline'}
  </div>
</header>

<main>
  {#if view === 'speak'}
    <section class="hero">
      <p class="eyebrow">MCP AUDIO SERVER</p>
      <h1>Super Fast TTS MCP Server</h1>
      <p>
        Text in, speech out — streamed while it is still being generated, with first audio in
        well under a second once warm. Agents connect over MCP and call the
        <code>speak</code> tool; this page drives the same engine by hand.
      </p>
    </section>

    <section class="panel speak-panel">
      <textarea rows="5" bind:value={text} disabled={speaking}
        placeholder="Type something to say…"></textarea>
      <div class="speak-actions">
        <button class="primary" on:click={speak} disabled={speaking || !text.trim()}>
          {speaking ? 'Speaking…' : 'Speak'}
        </button>
        {#if speaking}
          <button class="danger" on:click={stop}>Stop</button>
        {/if}
        {#if lastWavUrl && !speaking}
          <a class="ghost" href={lastWavUrl} download="speech.wav">Download WAV</a>
        {/if}
      </div>
      {#if errorStatus}
        <p class="status bad">{errorStatus}</p>
      {:else if status}
        <p class="status">{status}</p>
      {:else if stats}
        <p class="status">
          First audio in {stats.firstPcmMs === null ? '—' : `${stats.firstPcmMs.toFixed(0)} ms`}
          · {stats.audioSeconds.toFixed(1)} s spoken in {(stats.wallMs / 1000).toFixed(1)} s
        </p>
      {/if}
    </section>

    <section class="panel mcp-panel">
      <h2>Connect an agent</h2>
      <p class="mcp-hint">MCP endpoint (streamable HTTP) — one tool, <code>speak</code>, returns WAV audio:</p>
      <div class="endpoint-row">
        <code class="endpoint">{endpoint}</code>
        <button class="ghost" on:click={copyEndpoint}>{copied ? 'Copied' : 'Copy'}</button>
      </div>
      <details>
        <summary>Use from Open WebUI</summary>
        <ol>
          <li>
            <strong>As an MCP tool server</strong> — Admin Settings → External Tools → add this
            endpoint as a streamable-HTTP MCP server. Models can then call
            <code>speak</code> and attach the audio to their replies. If your Open WebUI only
            accepts OpenAPI tool servers, bridge it with
            <code>uvx mcpo --port 8600 --server-type streamable-http -- {endpoint}</code> and add
            <code>http://localhost:8600</code> instead.
          </li>
          <li>
            <strong>As the TTS engine</strong> — Admin Settings → Audio → Text-to-Speech →
            OpenAI, API base <code>{endpoint.replace('/mcp', '/v1')}</code> (any API key), any
            voice name. Every read-aloud then uses this server's voice.
          </li>
        </ol>
        <p class="mcp-note">
          Audio is generated faster than real time and streamed by this server, but MCP tool calls
          and Open WebUI's read-aloud both send finished text and collect a complete clip. For
          spoken replies that begin before the model finishes writing, the client would need to
          stream sentences to this server as they are generated — an Open WebUI-side change.
        </p>
      </details>
    </section>
  {:else if view === 'live'}
    <section class="live-stage">
      <div class="live-caption" bind:this={liveStage}>
        {#if liveWords.length}
          <p class="live-line">
            {#each liveWords as word (word.offset)}
              <span class="live-word">
                {#each word.syllables as syllable, part}
                  <span
                    class="syl"
                    class:shown={word.offset + part < liveVisible}
                    class:current={liveSpeaking && word.offset + part === liveVisible - 1}
                  >{syllable.text}</span>
                {/each}
              </span>
            {/each}
          </p>
        {:else}
          <p class="live-empty">Words appear here the instant they're spoken.</p>
        {/if}
      </div>
      {#if liveError}
        <p class="live-error">{liveError}</p>
      {/if}
      <div class="live-bar">
        <textarea
          rows="1"
          bind:value={liveText}
          disabled={liveSpeaking}
          on:keydown={liveKeydown}
          placeholder="Type something and press Enter…"
        ></textarea>
        {#if liveSpeaking}
          <button class="danger" on:click={liveStop}>Stop</button>
        {:else}
          <button class="primary" on:click={liveSpeak} disabled={!liveText.trim()}>Speak</button>
        {/if}
        {#if liveWords.length && !liveSpeaking}
          {#if liveWavUrl && liveDone}
            <a class="ghost" href={liveWavUrl} download="speech.wav">WAV</a>
          {/if}
          <button class="ghost" on:click={liveClear}>Clear</button>
        {/if}
      </div>
    </section>
  {:else if view === 'voice'}
    <section class="panel" style="display:flex;flex-direction:column;gap:14px;">
      <h2 style="margin:0;">Voice — talk with her</h2>
      <p style="margin:0;opacity:.75;">
        Speak; your words appear as you say them (solid = settled, dimmed = still revising).
        The moment you pause, she answers out loud — her reply is speculated from the settled
        part of your sentence while you're still talking. Speak over her to cut her off.
      </p>
      <div class="speak-actions">
        {#if !voiceActive}
          <button class="primary" on:click={startVoice}>Start listening</button>
        {:else}
          <button class="danger" on:click={stopVoice}>Stop</button>
          <span class="status" style="align-self:center;">
            {voiceSpeakingReply
              ? '🗣 she’s speaking — talk to interrupt'
              : voiceSpeechActive
                ? '● listening — speech detected'
                : '○ listening'}
          </span>
        {/if}
        <label class="status" style="align-self:center;display:flex;gap:6px;align-items:center;cursor:pointer;">
          <input type="checkbox" bind:checked={voiceConverse} />
          conversation mode
        </label>
      </div>
      {#if voiceError}
        <p class="status bad">{voiceError}</p>
      {/if}
      <div style="display:flex;flex-direction:column;gap:10px;min-height:120px;">
        {#each voiceTurns as turn (turn.key)}
          {#if turn.kind === 'user'}
            <div style="padding:10px 12px;border-radius:10px;background:rgba(127,127,127,.08);max-width:85%;">
              <span>{turn.final ?? turn.stable}</span>{#if !turn.final && turn.tentative}<span style="opacity:.45;font-style:italic;"> {turn.tentative}</span>{/if}
              {#if turn.final && turn.timing}
                <div style="margin-top:4px;font-size:.78em;opacity:.55;">
                  stop → final {turn.timing.eot_to_final_ms} ms
                </div>
              {/if}
            </div>
          {:else}
            <div style="padding:10px 12px;border-radius:10px;background:rgba(100,160,255,.10);align-self:flex-end;max-width:85%;">
              <span>{turn.text || '…'}</span>
            </div>
          {/if}
        {/each}
        {#if voiceActive && voiceTurns.length === 0}
          <p class="status" style="opacity:.6;">Say something…</p>
        {/if}
      </div>
    </section>
  {:else if view === 'chat'}
    <section class="hero">
      <p class="eyebrow">TALK TO THE CHARACTER</p>
      <h1>Chat</h1>
      <p>
        The reply streams as text while completed sentences are spoken in the character's voice —
        the model keeps writing while earlier sentences play, and the audio catches up.
      </p>
    </section>

    <section class="panel chat-panel">
      <div class="chat-log" bind:this={chatLog}>
        {#if !chatMessages.length}
          <p class="status">Say something to start. The active character replies in voice.</p>
        {/if}
        {#each chatMessages as message, index (index)}
          <div class="chat-message {message.role}">
            <div class="bubble">
              {message.content}{#if message.speaking}<span class="caret"></span>{/if}
            </div>
          </div>
        {/each}
      </div>
      {#if chatCaption && chatCaption.words.length}
        <div class="chat-caption">
          {#key chatCaption.index}
            {#each chatCaption.words as word (word.offset)}
              <span class="live-word">
                {#each word.syllables as syllable, part}
                  <span
                    class="syl"
                    class:shown={word.offset + part < chatCaption.visible}
                    class:current={word.offset + part === chatCaption.visible - 1}
                  >{syllable.text}</span>
                {/each}
              </span>
            {/each}
          {/key}
        </div>
      {/if}
      <form class="chat-input" on:submit|preventDefault={sendChat}>
        <input
          bind:value={chatInput}
          on:input={queueChatPrewarm}
          placeholder={chatBusy ? 'Type your next message while she talks…' : 'Say something…'}
        />
        {#if chatBusy}
          <button class="danger" type="button" on:click={stopChat}>Stop</button>
        {:else}
          <button class="primary" type="submit" disabled={!chatInput.trim()}>Send</button>
        {/if}
        <button class="ghost" type="button" on:click={clearChat} disabled={chatBusy}>Clear</button>
      </form>
      {#if chatError}
        <p class="status bad">{chatError}</p>
      {:else if chatStats}
        <p class="status">
          {#if chatStats.speculative_hit}⚡ instant (reply was ready before send) · {/if}
          First token {chatStats.first_token_ms < 0 ? '—' : `${chatStats.first_token_ms.toFixed(0)} ms`}
          · first audio {chatStats.first_audio_ms < 0 ? '—' : `${chatStats.first_audio_ms.toFixed(0)} ms`}
          · {chatStats.audio_seconds.toFixed(1)} s spoken in {(chatStats.wall_ms / 1000).toFixed(1)} s
        </p>
      {/if}
    </section>
  {:else}
    <section class="hero">
      <p class="eyebrow">SETTINGS</p>
      <h1>Voice settings</h1>
      <p>
        The character is server state: the name and voice chosen here apply to this page and to
        every MCP call, and survive restarts.
      </p>
    </section>

    {#if library.length}
      <section class="panel library-panel">
        <h2>Saved characters</h2>
        <p class="mcp-hint">One click switches the whole server — this page and every MCP call.</p>
        <ul class="library">
          {#each library as entry (entry.id)}
            <li>
              <div class="library-name">
                <strong>{entry.name}</strong>
                <small>{entry.source === 'custom' ? 'custom recording' : entry.preset}</small>
              </div>
              {#if entry.active}
                <span class="active-badge">Active</span>
              {:else}
                <button class="ghost" disabled={Boolean(switching)}
                  on:click={() => useCharacter(entry)}>
                  {switching === entry.id ? 'Switching…' : 'Use'}
                </button>
              {/if}
              <button class="danger slim" disabled={Boolean(switching)} title="Delete"
                on:click={() => removeCharacter(entry)}>✕</button>
            </li>
          {/each}
        </ul>
      </section>
    {/if}

    <section class="panel settings-panel">
      <h2>{active ? 'Edit or add a character' : 'Add a character'}</h2>
      <p class="mcp-hint">Saving stores it above and makes it the active voice. Saving under a new
        name adds a new character; the same name replaces it.</p>
      <label for="char-name">Character name</label>
      <input id="char-name" bind:value={editName} maxlength="64" placeholder="Name this character" />

      <label for="char-persona">Persona <span style="font-weight:400">— who they are in chat</span></label>
      <textarea id="char-persona" rows="3" bind:value={editPersona}
        placeholder="Describe the character for the roleplay model: personality, backstory, how they talk…"></textarea>

      <fieldset>
        <legend>Voice</legend>
        <label class="choice">
          <input type="radio" bind:group={voiceSource} value="preset" />
          <span>
            <strong>Bundled voice</strong>
            <small>One of the demo voices shipped with the app; warmed at startup, lowest latency.</small>
          </span>
        </label>
        {#if voiceSource === 'preset'}
          <select bind:value={editPreset}>
            {#each active?.available_presets || [] as preset}
              <option value={preset}>{preset}</option>
            {/each}
          </select>
        {/if}

        <label class="choice">
          <input type="radio" bind:group={voiceSource} value="custom" />
          <span>
            <strong>Custom recording</strong>
            <small>Clone from 5–15 seconds of clean speech. The first request after saving pays a
            one-time warmup.</small>
          </span>
        </label>
        {#if voiceSource === 'custom'}
          <input id="char-voice" class="file-native" type="file" accept="audio/*"
            on:change={(event) => probeSample(event.currentTarget.files?.[0] || null)} />
          <label class="file-picker" for="char-voice">
            <strong>Choose file</strong>
            <span>{customFile?.name || (active?.source === 'custom' ? 'Keeping the saved recording' : 'No file selected')}</span>
          </label>
          <div class="record-row">
            {#if recording}
              <button class="danger" type="button" on:click={stopRecording}>Stop recording</button>
            {:else}
              <button class="ghost" type="button" on:click={startRecording}>Record microphone</button>
            {/if}
          </div>
          {#if sampleNote}
            <p class="status warn">{sampleNote}</p>
          {/if}
          <label for="char-transcript">Transcript of the recording</label>
          <textarea id="char-transcript" rows="2" bind:value={customTranscript}
            placeholder="Exactly what the sample says…"></textarea>
        {/if}
      </fieldset>

      {#if active?.source === 'custom'}
        <label for="active-reference">Active character's reference recording</label>
        <audio id="active-reference" class="reference-audio" controls
          src={`/v1/character/voice?v=${voiceRev}`}></audio>
        <p class="status">This is what the clone is conditioned on — if it has background music or
          noise, the spoken output will drift from the voice.</p>
      {/if}

      <div class="speak-actions">
        <button class="primary" on:click={saveCharacter} disabled={saving}>
          {saving ? 'Saving…' : 'Save character'}
        </button>
        <button class="ghost" on:click={testVoice} disabled={testing || speaking}>
          {testing ? 'Speaking…' : 'Test voice'}
        </button>
        <button class="ghost" on:click={() => (view = 'speak')}>Back</button>
      </div>
      {#if settingsError}
        <p class="status bad">{settingsError}</p>
      {:else if settingsStatus}
        <p class="status">{settingsStatus}</p>
      {/if}
    </section>

    {#if llm}
      <section class="panel settings-panel">
        <h2>Roleplay engine</h2>
        <p class="mcp-hint">How every character is played. The master prompt wraps the persona;
          <code>{'{name}'}</code> and <code>{'{persona}'}</code> are filled from the active character.</p>

        {#if llm.models.length > 0}
          <div class="llm-model-row">
            <div>
              <label for="llm-model">Chat model</label>
              <select id="llm-model" bind:value={llmModelChoice} disabled={llmSwitching}>
                {#each llm.models as option (option.id)}
                  <option value={option.id} disabled={!option.installed}>
                    {option.name}{option.installed ? '' : ' — not installed'}
                  </option>
                {/each}
              </select>
            </div>
            <button
              class="primary"
              on:click={switchLlmModel}
              disabled={llmSwitching || llmModelChoice === llm.model}
            >
              {llmSwitching ? 'Loading…' : 'Switch model'}
            </button>
          </div>
          {#if llmSwitching}
            <p class="status">Loading the model — a large one can take a minute or two. Chat stays
              offline until it finishes.</p>
          {/if}
        {/if}

        <label for="llm-master">Master prompt</label>
        <textarea id="llm-master" rows="6" bind:value={llm.master_prompt}></textarea>

        <div class="llm-grid">
          <div>
            <label for="llm-temp">Temperature</label>
            <input id="llm-temp" type="number" min="0" max="2" step="0.05" bind:value={llm.temperature} />
            <small>Lower is steadier; roleplay models drift above ~0.8.</small>
          </div>
          <div>
            <label for="llm-topp">Top-p</label>
            <input id="llm-topp" type="number" min="0.05" max="1" step="0.05" bind:value={llm.top_p} />
            <small>Nucleus width; lower keeps replies on-script.</small>
          </div>
          <div>
            <label for="llm-rep">Repetition penalty</label>
            <input id="llm-rep" type="number" min="1" max="2" step="0.01" bind:value={llm.repeat_penalty} />
            <small>Raise slightly if the character loops phrases.</small>
          </div>
          <div>
            <label for="llm-max">Max tokens per reply</label>
            <input id="llm-max" type="number" min="16" max="1024" step="8" bind:value={llm.max_tokens} />
            <small>Reply length is the wall-clock of a voice chat.</small>
          </div>
        </div>

        <label class="choice ramp-choice">
          <input type="checkbox" bind:checked={llm.length_ramp} />
          <span>
            <strong>Grow replies over the conversation</strong>
            <small>The opener stays snappy — first audio waits on it — then each turn earns more
            room, up to the max tokens ceiling.</small>
          </span>
        </label>

        <div class="speak-actions">
          <button class="primary" on:click={saveLlmSettings} disabled={llmSaving}>
            {llmSaving ? 'Saving…' : 'Save roleplay settings'}
          </button>
          <button class="ghost" on:click={restoreLlmDefaults} disabled={llmSaving}>Reset to defaults</button>
        </div>
        {#if llmError}
          <p class="status bad">{llmError}</p>
        {:else if llmStatus}
          <p class="status">{llmStatus}</p>
        {/if}
      </section>
    {/if}
  {/if}
</main>

<footer>
  <span>Super Fast TTS · Qwen3 streaming CUDA inference · MCP streamable HTTP</span>
  <span>voice settings persist on the server</span>
</footer>
