/**
 * Agentic Adaptive RAG — Frontend Logic
 * Handles query submission, response rendering, and UI state management.
 */

const API_BASE = '';

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    
    // Submit on Enter (Shift+Enter for newline)
    document.getElementById('query-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuery();
        }
    });
});

// ── Fetch Stats ──
async function fetchStats() {
    try {
        const res = await fetch(`${API_BASE}/api/stats`);
        const data = await res.json();
        document.querySelector('#stat-docs .stat-value').textContent = data.document_count ?? '—';
    } catch (e) {
        console.error('Stats fetch failed:', e);
    }
}

// ── Set Example Query ──
function setQuery(text) {
    const input = document.getElementById('query-input');
    input.value = text;
    input.focus();
}

// ── Submit Query ──
async function submitQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query) return;

    const btn = document.getElementById('submit-btn');
    const responseArea = document.getElementById('response-area');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');

    // Show loading
    btn.disabled = true;
    responseArea.classList.remove('hidden');
    loading.classList.remove('hidden');
    result.classList.add('hidden');

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await res.json();
        renderResult(data);

    } catch (e) {
        renderError(e.message);
    } finally {
        btn.disabled = false;
        loading.classList.add('hidden');
    }
}

// ── Render Result ──
function renderResult(data) {
    const result = document.getElementById('result');
    result.classList.remove('hidden');

    // Agent path steps
    const pathSteps = document.getElementById('path-steps');
    pathSteps.innerHTML = '';
    (data.steps_taken || []).forEach(step => {
        const div = document.createElement('div');
        div.className = `path-step ${getStepClass(step)}`;
        
        const icon = document.createElement('span');
        icon.className = 'step-icon';
        icon.textContent = getStepIcon(step);
        
        const text = document.createElement('span');
        text.textContent = step;
        
        div.appendChild(icon);
        div.appendChild(text);
        pathSteps.appendChild(div);
    });

    // Answer
    document.getElementById('answer-content').textContent = data.answer || 'No answer generated';

    // Metrics
    const strategyEl = document.getElementById('m-strategy');
    strategyEl.textContent = data.query_type || '—';
    strategyEl.className = `metric-value strategy-${data.query_type}`;

    const groundedEl = document.getElementById('m-grounded');
    groundedEl.textContent = data.is_grounded ? '✅ Yes' : '❌ No';
    groundedEl.className = `metric-value grounded-${data.is_grounded}`;

    document.getElementById('m-relevance').textContent = 
        typeof data.relevance_score === 'number' ? data.relevance_score.toFixed(2) : '—';

    document.getElementById('m-retries').textContent = data.retry_count ?? '—';
    document.getElementById('m-time').textContent = 
        typeof data.processing_time_seconds === 'number' ? `${data.processing_time_seconds}s` : '—';
}

// ── Render Error ──
function renderError(message) {
    const result = document.getElementById('result');
    result.classList.remove('hidden');

    document.getElementById('path-steps').innerHTML = 
        `<div class="path-step" style="border-left-color: var(--error);">
            <span class="step-icon">❌</span>
            <span>Error: ${message}</span>
        </div>`;

    document.getElementById('answer-content').textContent = `Error: ${message}`;
}

// ── Helpers ──
function getStepClass(step) {
    if (step.includes('[Router]')) return 'step-router';
    if (step.includes('[Retrieval]')) return 'step-retrieval';
    if (step.includes('[Multi-Retrieval]')) return 'step-retrieval';
    if (step.includes('[Decomposition]')) return 'step-decomp';
    if (step.includes('[Generation]')) return 'step-generation';
    if (step.includes('[Grader]')) return 'step-grader';
    if (step.includes('[Transform]')) return 'step-transform';
    if (step.includes('[Web Search]')) return 'step-web';
    return '';
}

function getStepIcon(step) {
    if (step.includes('[Router]')) return '🔀';
    if (step.includes('[Retrieval]')) return '📚';
    if (step.includes('[Multi-Retrieval]')) return '📚';
    if (step.includes('[Decomposition]')) return '🧩';
    if (step.includes('[Generation]')) return '✍️';
    if (step.includes('[Grader]')) return '🔎';
    if (step.includes('[Transform]')) return '🔄';
    if (step.includes('[Web Search]')) return '🌐';
    return '•';
}
