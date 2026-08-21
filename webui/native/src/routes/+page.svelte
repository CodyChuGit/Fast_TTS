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
  import '../app.css';

  let view: 'speak' | 'chat' | 'settings' = 'speak';
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

  // Roleplay engine settings.
  let llm: LlmSettings | null = null;
  let llmSaving = false;
  let llmStatus = '';
  let llmError = '';

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
      llm = await setLlmSettings(llm);
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
      llmStatus = 'Roleplay defaults restored.';
    } catch (error) {
      llmError = error instanceof Error ? error.message : String(error);
    } finally {
      llmSaving = false;
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

  async function sendChat() {
    const content = chatInput.trim();
    if (!content || chatBusy) return;
    chatBusy = true;
    chatError = '';
    chatStats = null;
    chatInput = '';
    chatMessages = [...chatMessages, { role: 'user', content }, { role: 'assistant', content: '', speaking: true }];
    chatAborter = new AbortController();
    try {
      await primePcm16Playback();
      chatPlayer = new Pcm16StreamPlayer(24000, 1);
      await chatPlayer.start();
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
        } else if (event.type === 'audio') {
          chatPlayer?.push(base64ToBytes(event.audio));
        } else if (event.type === 'error') {
          chatError = event.message;
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
    if (recorder?.state === 'recording') recorder.stop();
    recordingStream?.getTracks().forEach((track) => track.stop());
    if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
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
      <form class="chat-input" on:submit|preventDefault={sendChat}>
        <input bind:value={chatInput} disabled={chatBusy} placeholder="Say something…" />
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

        <label for="llm-master">Master prompt</label>
        <textarea id="llm-master" rows="6" bind:value={llm.master_prompt}></textarea>

        <div class="llm-grid">
          <div>
            <label for="llm-temp">Temperature</label>
            <input id="llm-temp" type="number" min="0" max="2" step="0.05" bind:value={llm.temperature} />
            <small>Lower is steadier; Peach hallucinates above ~0.8.</small>
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
