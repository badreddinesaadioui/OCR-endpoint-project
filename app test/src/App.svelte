<script>
  import { onDestroy } from 'svelte';

  const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';
  const API_TOKEN = import.meta.env.VITE_API_AUTH_TOKEN || '';

  let file = null;
  let dragging = false;
  let status = 'idle'; // 'idle' | 'loading' | 'success' | 'error'
  let result = null;
  let errorMessage = '';
  let loadingStartTime = 0;
  let elapsedSeconds = 0;
  let timerInterval = null;

  function handleDrop(e) {
    e.preventDefault();
    dragging = false;
    const f = e.dataTransfer?.files?.[0];
    if (f && isPdf(f)) setFile(f);
  }

  function handleDragOver(e) {
    e.preventDefault();
    dragging = true;
  }

  function handleDragLeave() {
    dragging = false;
  }

  function handleFileInput(e) {
    const f = e.target.files?.[0];
    if (f && isPdf(f)) setFile(f);
    e.target.value = '';
  }

  function isPdf(f) {
    return f && (f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'));
  }

  function setFile(f) {
    file = f;
    status = 'idle';
    result = null;
    errorMessage = '';
  }

  function clearFile() {
    file = null;
    status = 'idle';
    result = null;
    errorMessage = '';
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  function startLoadingTimer() {
    loadingStartTime = Date.now();
    elapsedSeconds = 0;
    if (timerInterval) clearInterval(timerInterval);
    timerInterval = setInterval(() => {
      elapsedSeconds = Math.floor((Date.now() - loadingStartTime) / 1000);
    }, 1000);
  }

  function stopLoadingTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
    elapsedSeconds = (Date.now() - loadingStartTime) / 1000;
  }

  async function parseCv() {
    if (!file) return;
    status = 'loading';
    result = null;
    errorMessage = '';
    startLoadingTimer();

    const formData = new FormData();
    formData.append('file', file);

    const headers = {};
    if (API_TOKEN) headers['Authorization'] = `Bearer ${API_TOKEN}`;

    try {
      const res = await fetch(`${API_BASE}/v1/parse-cv`, {
        method: 'POST',
        headers,
        body: formData,
      });
      const data = await res.json().catch(() => ({}));

      stopLoadingTimer();

      if (!res.ok) {
        status = 'error';
        errorMessage = data?.error?.message || res.statusText || 'Request failed';
        return;
      }
      result = data;
      status = 'success';
    } catch (err) {
      stopLoadingTimer();
      status = 'error';
      errorMessage = err.message || 'Network error. Is the API running at ' + API_BASE + '?';
    }
  }

  function copyJson() {
    if (!result?.result) return;
    const text = JSON.stringify(result.result, null, 2);
    navigator.clipboard.writeText(text);
  }

  function resultJson() {
    if (!result?.result) return '';
    return JSON.stringify(result.result, null, 2);
  }

  onDestroy(() => {
    if (timerInterval) clearInterval(timerInterval);
  });
</script>

<svelte:head>
  <link rel="stylesheet" href="/src/app.css" />
</svelte:head>

<header class="header">
  <div class="logo">
    <img src="/logo.png" alt="forvis mazars" on:error={(e) => { e.target.style.display = 'none'; e.target.nextElementSibling?.classList.add('show'); }} />
    <span class="logo-text"><span class="forvis">forvis</span><span class="mazars">mazars</span></span>
  </div>
  <span class="header-title">CV Parser</span>
</header>

<main class="main">
  <h1 class="title">CV Parser</h1>
  <p class="subtitle">Upload a CV to extract structured data.</p>

  <!-- Drop zone or file card -->
  {#if !file}
    <div
      class="dropzone"
      class:dragging
      role="button"
      tabindex="0"
      on:drop={handleDrop}
      on:dragover={handleDragOver}
      on:dragleave={handleDragLeave}
      on:click|self={() => document.getElementById('file-input').click()}
      on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); document.getElementById('file-input').click(); } }}
    >
      <svg class="dropzone-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
      <p class="dropzone-text">Drop your CV here</p>
      <p class="dropzone-hint">or click to browse · PDF only</p>
    </div>
  {:else}
    <div class="file-card">
      <svg class="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
      </svg>
      <div class="file-info">
        <span class="file-name">{file.name}</span>
        <span class="file-size">{formatSize(file.size)}</span>
      </div>
      <button type="button" class="file-remove" on:click={clearFile} aria-label="Remove file">×</button>
    </div>
  {/if}

  <input id="file-input" type="file" accept=".pdf,application/pdf" on:change={handleFileInput} style="display: none" />

  <button type="button" class="btn-parse" disabled={!file || status === 'loading'} on:click={parseCv}>
    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
    Parse CV
  </button>

  <!-- Results area -->
  <div class="results">
    {#if status === 'idle'}
      <div class="results-placeholder">
        <svg class="results-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
          <line x1="10" y1="9" x2="8" y2="9" />
        </svg>
        <p>Your parsed result will appear here</p>
      </div>
    {:else if status === 'loading'}
      <div class="results-loading">
        <div class="spinner"></div>
        <p>Parsing your CV... <strong>{elapsedSeconds}s</strong></p>
        <div class="skeleton">
          <div class="skeleton-line short"></div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line medium"></div>
          <div class="skeleton-line"></div>
        </div>
      </div>
    {:else if status === 'error'}
      <div class="results-error">
        <p>{errorMessage}</p>
      </div>
    {:else if status === 'success' && result}
      <div class="results-success">
        <div class="results-header">
          <span class="results-label">Parsed Result <span class="results-time">(in {elapsedSeconds.toFixed(1)}s)</span></span>
          <button type="button" class="btn-copy" on:click={copyJson}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        Copy JSON
      </button>
        </div>
        <pre class="results-json">{resultJson()}</pre>
      </div>
    {/if}
  </div>
</main>

<style>
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: var(--card);
    border-bottom: 1px solid var(--border);
  }
  .logo img {
    height: 36px;
    width: auto;
    display: block;
    object-fit: contain;
  }
  .logo-text {
    display: none;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }
  .logo-text.show {
    display: block;
  }
  .logo-text .forvis { color: #0066b3; display: block; }
  .logo-text .mazars { color: #004c8c; display: block; }
  .header-title {
    font-weight: 600;
    font-size: 1rem;
    color: var(--text);
  }

  .main {
    max-width: 640px;
    margin: 0 auto;
    padding: 2rem 1rem 3rem;
  }
  .title {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.25rem;
    color: var(--text);
  }
  .subtitle {
    color: var(--text-muted);
    margin: 0 0 1.5rem;
    font-size: 0.95rem;
  }

  .dropzone {
    border: 2px dashed var(--border-dashed);
    border-radius: var(--radius);
    padding: 2.5rem;
    text-align: center;
    cursor: pointer;
    background: var(--card);
    transition: border-color 0.2s, background 0.2s;
  }
  .dropzone:hover,
  .dropzone.dragging {
    border-color: var(--primary);
    background: #f0f7ff;
  }
  .dropzone-icon {
    width: 48px;
    height: 48px;
    margin: 0 auto 1rem;
    color: var(--text-muted);
  }
  .dropzone.dragging .dropzone-icon {
    color: var(--primary);
  }
  .dropzone-text {
    font-weight: 600;
    margin: 0 0 0.25rem;
    color: var(--text);
  }
  .dropzone-hint {
    font-size: 0.875rem;
    color: var(--text-muted);
    margin: 0;
  }

  .file-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 1.25rem;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 1rem;
  }
  .file-icon {
    width: 40px;
    height: 40px;
    flex-shrink: 0;
    color: var(--primary);
  }
  .file-info {
    flex: 1;
    min-width: 0;
  }
  .file-name {
    display: block;
    font-weight: 500;
    color: var(--text);
  }
  .file-size {
    font-size: 0.875rem;
    color: var(--text-muted);
  }
  .file-remove {
    width: 32px;
    height: 32px;
    border: none;
    background: transparent;
    color: var(--text-muted);
    font-size: 1.25rem;
    cursor: pointer;
    border-radius: 6px;
    line-height: 1;
  }
  .file-remove:hover {
    background: #f1f5f9;
    color: var(--text);
  }

  .btn-parse {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    padding: 0.875rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    font-family: inherit;
    color: white;
    background: var(--primary);
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    margin-top: 1rem;
    transition: background 0.2s;
  }
  .btn-parse:hover:not(:disabled) {
    background: var(--primary-hover);
  }
  .btn-parse:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .btn-icon {
    width: 20px;
    height: 20px;
  }

  .results {
    margin-top: 1.5rem;
    min-height: 200px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--card);
    overflow: hidden;
  }
  .results-placeholder,
  .results-loading,
  .results-error {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2.5rem;
    color: var(--text-muted);
    text-align: center;
  }
  .results-icon {
    width: 48px;
    height: 48px;
    margin-bottom: 1rem;
    opacity: 0.6;
  }
  .results-loading p,
  .results-placeholder p {
    margin: 0 0 1rem;
  }
  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--border);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-bottom: 1rem;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  .skeleton {
    width: 100%;
    max-width: 320px;
  }
  .skeleton-line {
    height: 12px;
    background: var(--border);
    border-radius: 6px;
    margin-bottom: 0.5rem;
  }
  .skeleton-line.short { width: 40%; }
  .skeleton-line.medium { width: 70%; }
  .skeleton-line:not(.short):not(.medium) { width: 100%; }

  .results-error {
    color: var(--error);
  }
  .results-success {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 280px;
  }
  .results-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
  }
  .results-label {
    font-weight: 600;
    font-size: 0.9rem;
  }
  .results-time {
    font-weight: 500;
    font-size: 0.8rem;
    color: var(--text-muted);
  }
  .btn-copy {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
    font-family: inherit;
    font-weight: 500;
    color: var(--primary);
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 6px;
    cursor: pointer;
  }
  .btn-copy:hover {
    background: #f0f7ff;
  }
  .results-json {
    flex: 1;
    margin: 0;
    padding: 1rem;
    font-size: 0.8rem;
    line-height: 1.5;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
</style>
