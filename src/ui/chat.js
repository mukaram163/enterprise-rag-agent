(function () {
    'use strict';

    const chatContainer = document.getElementById('chat-container');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function formatPageMarkers(text) {
        return text
            .replace(/\[[^\]\[]+?\.pdf\s+p\.\s*(\d+)\]/g, '<span class="page-badge">p.$1</span>')
            .replace(/\[p\.\s*(\d+)\]/g, '<span class="page-badge">p.$1</span>');
    }

    // FIX: model output is sanitized before it touches innerHTML.
    // If the libraries failed to load, fall back to plain text instead of unsafe HTML.
    function setBotText(el, text) {
        if (window.marked && window.DOMPurify) {
            el.innerHTML = window.DOMPurify.sanitize(window.marked.parse(formatPageMarkers(text)));
        } else {
            el.textContent = text;
        }
    }

    // FIX: messages from the server or from errors are shown as text, never as HTML.
    function showError(el, message) {
        el.replaceChildren();
        const span = document.createElement('span');
        span.className = 'error-text';
        span.textContent = message;
        el.appendChild(span);
    }

    function renderCitationsUI(citations, targetDiv) {
        if (!citations || citations.length === 0) return;

        const unique = Array.from(
            new Map(citations.map(c => [`${c.file_name}:${c.page_number}`, c])).values()
        );

        const box = document.createElement('div');
        box.className = 'citations-container';

        const label = document.createElement('span');
        label.className = 'citations-label';
        label.textContent = 'Sources:';
        box.appendChild(label);

        unique.forEach(c => {
            const badge = document.createElement('span');
            badge.className = 'citation-badge';
            badge.textContent = `📄 ${c.file_name} (p. ${c.page_number})`;
            box.appendChild(badge);
        });

        targetDiv.appendChild(box);
    }

    async function submitFeedback(query, response, rating, container) {
        try {
            const res = await fetch('/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, response, rating })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            container.replaceChildren();
            const thanks = document.createElement('span');
            thanks.className = 'thanks-text';
            thanks.textContent = 'Thank you for your feedback!';
            container.appendChild(thanks);
        } catch (err) {
            console.error('Feedback failed:', err);
        }
    }

    // FIX: no inline onclick strings with hand-escaped quotes; buttons use closures.
    function addFeedbackButtons(botDiv, query, fullText) {
        const box = document.createElement('div');
        box.className = 'feedback-actions';

        const label = document.createElement('span');
        label.textContent = 'Rate this response:';
        box.appendChild(label);

        [['👍', 5], ['👎', 1]].forEach(([icon, rating]) => {
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.className = 'star-btn';
            btn.textContent = icon;
            btn.addEventListener('click', () => submitFeedback(query, fullText, rating, box));
            box.appendChild(btn);
        });

        botDiv.appendChild(box);
    }

    function handleEvent(payload, botDiv, state) {
        if (payload.event === 'token') {
            state.fullText += payload.data;
            setBotText(botDiv, state.fullText);
        } else if (payload.event === 'citations') {
            renderCitationsUI(payload.data, botDiv);
        } else if (payload.event === 'error') {
            showError(botDiv, payload.data || 'Something went wrong, try again');
        } else if (payload.event === 'notice') {
            const note = document.createElement('div');
            note.className = 'notice-text';
            note.textContent = payload.data;
            botDiv.appendChild(note);
        }
        scrollToBottom();
    }

    async function sendMessage() {
        const query = queryInput.value.trim();
        if (!query) return;

        // FIX: the user's text is inserted as text, not HTML.
        const userRow = document.createElement('div');
        userRow.className = 'message-row';
        const userDiv = document.createElement('div');
        userDiv.className = 'message user-message';
        userDiv.textContent = query;
        userRow.appendChild(userDiv);
        chatContainer.appendChild(userRow);

        queryInput.value = '';
        queryInput.disabled = true;
        sendBtn.disabled = true;

        const botRow = document.createElement('div');
        botRow.className = 'message-row';
        const botDiv = document.createElement('div');
        botDiv.className = 'message bot-message';
        botDiv.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
        botRow.appendChild(botDiv);
        chatContainer.appendChild(botRow);
        scrollToBottom();

        try {
            let sessionId = sessionStorage.getItem('sessionId');
            if (!sessionId) {
                sessionId = crypto.randomUUID();
                sessionStorage.setItem('sessionId', sessionId);
            }

            const response = await fetch('/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, session_id: sessionId })
            });

            if (response.status === 429 || response.status === 400) {
                let msg = 'Too many requests. Please try again later.';
                try { msg = (await response.json()).detail || msg; } catch (e) {}
                showError(botDiv, msg);
                return;
            }
            if (!response.ok) throw new Error(`HTTP error: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            const state = { fullText: '' };
            let firstToken = true;
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const parts = buffer.split('\n\n');
                buffer = parts.pop();

                for (const part of parts) {
                    const line = part.trim();
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const payload = JSON.parse(line.slice(6));
                        if (payload.event === 'token' && firstToken) {
                            botDiv.replaceChildren();
                            firstToken = false;
                        }
                        handleEvent(payload, botDiv, state);
                    } catch (e) {
                        console.error('Failed to parse SSE payload:', e, part);
                    }
                }
            }

            addFeedbackButtons(botDiv, query, state.fullText);
        } catch (error) {
            showError(botDiv, `Something went wrong, try again (${error.message})`);
        } finally {
            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.focus();
            scrollToBottom();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    queryInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') sendMessage();
    });
})();
