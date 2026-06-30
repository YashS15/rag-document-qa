let documents = [];

async function init() {
  await Promise.all([loadDocuments(), loadModels()]);
  setupDragDrop();
  setupTextarea();
}

// ── Document list ──

async function loadDocuments() {
  try {
    const resp = await fetch('/documents');
    documents = await resp.json();
    renderDocuments();
  } catch (e) {
    console.error('Failed to load documents:', e);
  }
}

function renderDocuments() {
  const list = document.getElementById('documentList');
  const badge = document.getElementById('docCount');
  badge.textContent = `${documents.length} doc${documents.length !== 1 ? 's' : ''} indexed`;

  if (documents.length === 0) {
    list.innerHTML = '<p class="empty-state">No documents uploaded yet</p>';
    return;
  }

  list.innerHTML = documents.map(doc => `
    <div class="doc-item" data-id="${doc.doc_id}">
      <span class="doc-icon">📄</span>
      <div class="doc-info">
        <div class="doc-name" title="${esc(doc.name)}">${esc(doc.name)}</div>
        <div class="doc-meta">${doc.page_count} pages · ${doc.chunk_count} chunks</div>
      </div>
      <button class="doc-delete" onclick="deleteDocument('${doc.doc_id}')" title="Remove document">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');
}

async function deleteDocument(docId) {
  if (!confirm('Remove this document from the index?')) return;
  try {
    const resp = await fetch(`/documents/${docId}`, { method: 'DELETE' });
    if (resp.ok) await loadDocuments();
  } catch (e) {
    alert('Failed to remove document.');
  }
}

// ── Models ──

async function loadModels() {
  try {
    const resp = await fetch('/models');
    const data = await resp.json();
    if (data.models && data.models.length) {
      const select = document.getElementById('modelSelect');
      select.innerHTML = data.models.map(m => `<option value="${m}">${m}</option>`).join('');
    }
  } catch (e) { /* keep defaults */ }
}

// ── Upload ──

function setupDragDrop() {
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => { if (e.target.files[0]) uploadFile(e.target.files[0]); });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
      uploadFile(file);
    } else {
      alert('Please drop a PDF file.');
    }
  });
}

async function uploadFile(file) {
  const dropZone = document.getElementById('dropZone');
  const progress = document.getElementById('uploadProgress');
  const fill = document.getElementById('progressFill');
  const status = document.getElementById('uploadStatus');

  dropZone.classList.add('hidden');
  progress.classList.remove('hidden');

  const steps = [
    [15, 'Reading PDF…'],
    [45, 'Extracting text…'],
    [70, 'Generating embeddings…'],
    [90, 'Indexing chunks…'],
  ];

  let stepIdx = 0;
  const ticker = setInterval(() => {
    if (stepIdx < steps.length) {
      const [pct, msg] = steps[stepIdx++];
      fill.style.width = pct + '%';
      status.textContent = msg;
    }
  }, 600);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();

    clearInterval(ticker);

    if (!resp.ok) throw new Error(data.error || 'Upload failed');

    fill.style.width = '100%';
    status.textContent = `Indexed ${data.page_count} pages · ${data.chunk_count} chunks`;

    await loadDocuments();
    setTimeout(() => resetUploadUI(dropZone, progress, fill), 2000);

  } catch (err) {
    clearInterval(ticker);
    fill.style.width = '0';
    status.textContent = `Error: ${err.message}`;
    setTimeout(() => resetUploadUI(dropZone, progress, fill), 3500);
  }
}

function resetUploadUI(dropZone, progress, fill) {
  progress.classList.add('hidden');
  dropZone.classList.remove('hidden');
  fill.style.width = '0';
  document.getElementById('fileInput').value = '';
}

// ── Chat ──

function setupTextarea() {
  const input = document.getElementById('questionInput');
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); askQuestion(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 130) + 'px';
  });
}

async function askQuestion() {
  const input = document.getElementById('questionInput');
  const btn = document.getElementById('askBtn');
  const question = input.value.trim();

  if (!question) return;
  if (documents.length === 0) { alert('Please upload at least one PDF document first.'); return; }

  const model = document.getElementById('modelSelect').value;

  document.querySelector('.welcome-message')?.remove();

  appendMessage('user', question);
  input.value = '';
  input.style.height = 'auto';
  btn.disabled = true;

  const thinkingId = 'think-' + Date.now();
  appendThinking(thinkingId);

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model, k: 5 }),
    });
    const data = await resp.json();
    removeElement(thinkingId);

    if (!resp.ok) {
      appendMessage('assistant', data.error || 'Something went wrong.', null, true);
    } else {
      appendMessage('assistant', data.answer, data.sources);
    }
  } catch (err) {
    removeElement(thinkingId);
    appendMessage('assistant', 'Network error — could not reach the server.', null, true);
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

function appendMessage(role, text, sources, isError = false) {
  const messages = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = `message ${role}`;

  const avatar = role === 'user' ? '👤' : '🤖';
  const sourcesHtml = sources && sources.length ? buildSourcesHtml(sources) : '';

  el.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-content">
      <div class="msg-bubble${isError ? ' error-bubble' : ''}">${esc(text)}</div>
      ${sourcesHtml}
    </div>
  `;

  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function buildSourcesHtml(sources) {
  const id = 'src-' + Date.now();
  const items = sources.map(s => `
    <div class="source-item">
      <div class="source-header">📄 ${esc(s.doc_name)} — Page ${s.page}</div>
      <div class="source-text">${esc(s.text)}</div>
    </div>
  `).join('');

  return `
    <div class="sources">
      <span class="sources-toggle" onclick="toggleSources('${id}')">
        ▸ ${sources.length} source${sources.length !== 1 ? 's' : ''} used
      </span>
      <div id="${id}" class="sources-list hidden">${items}</div>
    </div>
  `;
}

function toggleSources(id) {
  const el = document.getElementById(id);
  const toggle = el.previousElementSibling;
  const hidden = el.classList.toggle('hidden');
  toggle.textContent = (hidden ? '▸' : '▾') + toggle.textContent.slice(1);
}

function appendThinking(id) {
  const messages = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'message assistant';
  el.id = id;
  el.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble thinking-bubble">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      </div>
    </div>
  `;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function removeElement(id) {
  document.getElementById(id)?.remove();
}

// ── Helpers ──

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

init();
