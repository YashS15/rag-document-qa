let documents = [];
let conversationHistory = [];  // [{question, answer}]  max 6 turns kept

async function init() {
  await Promise.all([loadDocuments(), loadModels()]);
  setupDragDrop();
  setupTextarea();
}

// ── Models ──────────────────────────────────────────────────────────────────

async function loadModels() {
  try {
    const resp = await fetch('/models');
    const data = await resp.json();
    const select = document.getElementById('modelSelect');
    select.innerHTML = data.models
      .map(m => `<option value="${m.id}">${esc(m.name)}</option>`)
      .join('');
  } catch (e) { /* keep defaults */ }
}

// ── Documents ────────────────────────────────────────────────────────────────

async function loadDocuments() {
  try {
    const resp = await fetch('/documents');
    documents = await resp.json();
    renderDocuments();
    renderDocFilter();
  } catch (e) { console.error('Failed to load documents:', e); }
}

function renderDocuments() {
  const list = document.getElementById('documentList');
  const badge = document.getElementById('docCount');
  badge.textContent = `${documents.length} doc${documents.length !== 1 ? 's' : ''} indexed`;

  if (!documents.length) {
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
      <button class="doc-delete" onclick="deleteDocument('${doc.doc_id}')" title="Remove">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');
}

function renderDocFilter() {
  const select = document.getElementById('docFilter');
  const current = select.value;
  select.innerHTML = '<option value="">All documents</option>' +
    documents.map(d => `<option value="${d.doc_id}">${esc(d.name)}</option>`).join('');
  if (documents.find(d => d.doc_id === current)) select.value = current;
}

async function deleteDocument(docId) {
  if (!confirm('Remove this document from the index?')) return;
  try {
    const resp = await fetch(`/documents/${docId}`, { method: 'DELETE' });
    if (resp.ok) await loadDocuments();
  } catch (e) { alert('Failed to remove document.'); }
}

// ── Upload ───────────────────────────────────────────────────────────────────

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
    if (file?.name.toLowerCase().endsWith('.pdf')) uploadFile(file);
    else alert('Please drop a PDF file.');
  });
}

async function uploadFile(file) {
  const dropZone = document.getElementById('dropZone');
  const progress = document.getElementById('uploadProgress');
  const fill = document.getElementById('progressFill');
  const status = document.getElementById('uploadStatus');
  const summaryBox = document.getElementById('uploadSummary');
  const summaryText = document.getElementById('summaryText');

  summaryBox.classList.add('hidden');
  dropZone.classList.add('hidden');
  progress.classList.remove('hidden');

  const steps = [[15,'Reading PDF…'],[45,'Extracting text…'],[70,'Generating embeddings…'],[90,'Summarising…']];
  let i = 0;
  const ticker = setInterval(() => {
    if (i < steps.length) { fill.style.width = steps[i][0]+'%'; status.textContent = steps[i][1]; i++; }
  }, 700);

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

    // Close sidebar drawer on mobile after upload so user lands on chat
    if (window.innerWidth <= 640) toggleSidebar();

    if (data.summary) {
      summaryText.textContent = data.summary;
      setTimeout(() => summaryBox.classList.remove('hidden'), 400);
    }

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

// ── Chat ─────────────────────────────────────────────────────────────────────

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
  if (!documents.length) { alert('Please upload at least one PDF first.'); return; }

  const model = document.getElementById('modelSelect').value;
  const docId = document.getElementById('docFilter').value;

  document.querySelector('.welcome-message')?.remove();

  appendMessage('user', question);
  input.value = '';
  input.style.height = 'auto';
  btn.disabled = true;

  // Create a placeholder assistant message for streaming into
  const { msgEl, bubbleEl, sourcesSlot } = appendStreamingMessage();
  let fullAnswer = '';
  let streamError = false;

  try {
    const resp = await fetch('/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model, k: 5, history: conversationHistory, doc_id: docId || null }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      bubbleEl.textContent = err.error || 'Something went wrong.';
      bubbleEl.classList.add('error-bubble');
      streamError = true;
    } else {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const raw = line.slice(6);
          if (raw === '[DONE]') break;
          try {
            const parsed = JSON.parse(raw);
            if (parsed.type === 'sources') {
              renderSources(sourcesSlot, parsed.sources);
            } else if (parsed.type === 'token') {
              fullAnswer += parsed.token;
              bubbleEl.textContent = fullAnswer;
              scrollToBottom();
            } else if (parsed.type === 'error') {
              bubbleEl.textContent = parsed.message;
              bubbleEl.classList.add('error-bubble');
              streamError = true;
            }
          } catch (e) { /* malformed SSE line */ }
        }
      }
    }
  } catch (err) {
    bubbleEl.textContent = 'Network error — could not reach the server.';
    bubbleEl.classList.add('error-bubble');
    streamError = true;
  } finally {
    btn.disabled = false;
    input.focus();
  }

  if (!streamError && fullAnswer) {
    // Attach copy button now that answer is complete
    addCopyButton(msgEl, fullAnswer);
    // Keep last 6 turns in history
    conversationHistory.push({ question, answer: fullAnswer });
    if (conversationHistory.length > 6) conversationHistory.shift();
  }
}

function appendMessage(role, text) {
  const messages = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div class="msg-content">
      <div class="msg-bubble">${esc(text)}</div>
    </div>`;
  messages.appendChild(el);
  scrollToBottom();
}

function appendStreamingMessage() {
  const messages = document.getElementById('messages');
  const msgEl = document.createElement('div');
  msgEl.className = 'message assistant';
  msgEl.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-content">
      <div class="msg-bubble streaming-bubble">
        <span class="cursor">▌</span>
      </div>
      <div class="sources-slot"></div>
    </div>`;
  messages.appendChild(msgEl);
  scrollToBottom();
  return {
    msgEl,
    bubbleEl: msgEl.querySelector('.msg-bubble'),
    sourcesSlot: msgEl.querySelector('.sources-slot'),
  };
}

function renderSources(slot, sources) {
  if (!sources?.length) return;
  const id = 'src-' + Date.now();
  slot.innerHTML = `
    <div class="sources">
      <span class="sources-toggle" onclick="toggleSources('${id}')">
        ▸ ${sources.length} source${sources.length !== 1 ? 's' : ''} used
      </span>
      <div id="${id}" class="sources-list hidden">
        ${sources.map(s => `
          <div class="source-item">
            <div class="source-header">📄 ${esc(s.doc_name)} — Page ${s.page}</div>
            <div class="source-text">${esc(s.text)}</div>
          </div>`).join('')}
      </div>
    </div>`;
}

function addCopyButton(msgEl, text) {
  const content = msgEl.querySelector('.msg-content');
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.title = 'Copy answer';
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
  </svg>`;
  btn.onclick = () => {
    navigator.clipboard.writeText(text);
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
    setTimeout(() => {
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
      </svg>`;
    }, 1500);
  };
  content.appendChild(btn);
}

function toggleSources(id) {
  const el = document.getElementById(id);
  const toggle = el.previousElementSibling;
  const hidden = el.classList.toggle('hidden');
  toggle.textContent = (hidden ? '▸' : '▾') + toggle.textContent.slice(1);
}

function scrollToBottom() {
  const messages = document.getElementById('messages');
  messages.scrollTop = messages.scrollHeight;
}

// ── Sidebar toggle (mobile) ───────────────────────────────────────────────────

function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('open');
  document.getElementById('sidebarOverlay').classList.toggle('active');
}

// ── New Conversation ──────────────────────────────────────────────────────────

function newConversation() {
  conversationHistory = [];
  const messages = document.getElementById('messages');
  messages.innerHTML = `
    <div class="welcome-message">
      <div class="welcome-icon">🔍</div>
      <h3>Ready to answer questions</h3>
      <p>Upload PDF documents on the left, then ask anything about their content.</p>
    </div>`;
}

// ── Export Chat ───────────────────────────────────────────────────────────────

function exportChat() {
  const msgs = document.querySelectorAll('.message');
  if (!msgs.length) { alert('No conversation to export.'); return; }

  let md = `# RAG Document Q&A — Chat Export\n\n`;
  md += `*Exported ${new Date().toLocaleString()}*\n\n---\n\n`;

  msgs.forEach(msg => {
    const isUser = msg.classList.contains('user');
    const text = msg.querySelector('.msg-bubble')?.textContent?.trim();
    if (!text) return;
    md += isUser ? `**You:** ${text}\n\n` : `**Assistant:** ${text}\n\n---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `chat-export-${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

init();
