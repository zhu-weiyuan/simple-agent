/**
 * SimpleAgent 前端应用逻辑 - 增强版
 * 集成 6 个增强模块的可视化
 */

// ========================================
// 全局状态
// ========================================
let currentChatId = 'chat-' + Date.now();
let chatHistory = {};
let isTyping = false;

// ========================================
// 初始化
// ========================================
document.addEventListener('DOMContentLoaded', () => {
    loadChatHistory();
    updateChatList();
    setupEventListeners();
    initTheme();
    console.log('🤖 SimpleAgent 已启动');
});

function setupEventListeners() {
    const input = document.getElementById('messageInput');
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// ========================================
// 聊天管理
// ========================================
function newChat() {
    currentChatId = 'chat-' + Date.now();
    chatHistory[currentChatId] = [];
    
    const messages = document.getElementById('chatMessages');
    messages.innerHTML = `
        <div class="message welcome-message">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <h3>你好！我是 SimpleAgent AI 助手 👋</h3>
                <p>我集成了 <strong>6 个增强模块</strong>，可以：</p>
                <ul>
                    <li>🔍 <strong>查询路由</strong> — 自动分析你的问题复杂度</li>
                    <li>🧠 <strong>Persona 记忆</strong> — 记住你的偏好和画像</li>
                    <li>🚨 <strong>幻觉检测</strong> — 实时检测回答是否可能产生幻觉</li>
                    <li>📚 <strong>确定性引用</strong> — 从回答中提取引用来源</li>
                    <li>🔎 <strong>多索引检索</strong> — 混合检索相关文档</li>
                    <li>💬 <strong>多轮对话</strong> — 支持上下文连贯的对话</li>
                </ul>
                <p>试试发送一条消息吧！</p>
            </div>
        </div>
    `;
    
    updateChatList();
}

function saveChat() {
    localStorage.setItem('simpleagent_chats', JSON.stringify(chatHistory));
    localStorage.setItem('simpleagent_current', currentChatId);
}

function loadChatHistory() {
    const saved = localStorage.getItem('simpleagent_chats');
    if (saved) {
        chatHistory = JSON.parse(saved);
    }
    
    const current = localStorage.getItem('simpleagent_current');
    if (current && chatHistory[current]) {
        currentChatId = current;
        restoreChat(current);
    }
}

function restoreChat(chatId) {
    const messages = document.getElementById('chatMessages');
    messages.innerHTML = '';
    
    const chats = chatHistory[chatId] || [];
    chats.forEach(msg => {
        if (msg.enhanced) {
            appendEnhancedCard(msg.enhanced, false);
        }
        appendMessage(msg.role, msg.content, false);
    });
    
    scrollToBottom();
}

function updateChatList() {
    const chatList = document.getElementById('chatList');
    chatList.innerHTML = '';
    
    Object.keys(chatHistory).forEach(chatId => {
        const chats = chatHistory[chatId] || [];
        const firstMsg = chats.find(m => m.role === 'user');
        const title = firstMsg ? firstMsg.content.substring(0, 20) + '...' : '新对话';
        
        const item = document.createElement('div');
        item.className = `chat-item ${chatId === currentChatId ? 'active' : ''}`;
        item.textContent = title;
        item.onclick = () => {
            currentChatId = chatId;
            restoreChat(chatId);
            updateChatList();
        };
        
        chatList.appendChild(item);
    });
}

// ========================================
// 消息发送（增强版）
// ========================================
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    
    if (!message || isTyping) return;
    
    appendMessage('user', message);
    input.value = '';
    autoResize(input);
    
    if (!chatHistory[currentChatId]) {
        chatHistory[currentChatId] = [];
    }
    chatHistory[currentChatId].push({ role: 'user', content: message });
    saveChat();
    updateChatList();
    
    showTypingIndicator();
    isTyping = true;
    document.getElementById('sendBtn').disabled = true;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                chatId: currentChatId,
                stream: true
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        removeTypingIndicator();

        // 读取 SSE 流
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let fullReply = '';
        let enhancedData = null;
        let aiMsgEl = null;
        let contentDiv = null;

        // 创建 AI 消息气泡（打字机效果）
        const messages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ai';
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = '🤖';
        contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<span class="cursor">▊</span>';
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messages.appendChild(messageDiv);
        scrollToBottom();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 解析 SSE 事件
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('event: start')) continue;
                if (line.startsWith('event: done')) continue;
                if (line.startsWith('event: error')) continue;
                if (!line.startsWith('data: ')) continue;

                const dataStr = line.slice(6);
                try {
                    const data = JSON.parse(dataStr);

                    if (data.content !== undefined) {
                        fullReply += data.content;
                        contentDiv.innerHTML = formatMarkdown(fullReply) + '<span class="cursor">▊</span>';
                        scrollToBottom();
                    }

                    if (data.enhanced) {
                        enhancedData = data.enhanced;
                    }

                    if (data.reply !== undefined) {
                        fullReply = data.reply;
                        contentDiv.innerHTML = formatMarkdown(fullReply);
                    }
                } catch (e) {
                    // ignore parse errors on partial chunks
                }
            }
        }

        // 渲染增强卡片
        if (enhancedData && Object.keys(enhancedData).length > 0) {
            appendEnhancedCard(enhancedData);
        }

        // 保存到历史
        chatHistory[currentChatId].push({ role: 'ai', content: fullReply });
        saveChat();

    } catch (error) {
        removeTypingIndicator();
        appendMessage('ai', '❌ 错误: ' + error.message);
        console.error('发送消息失败:', error);
    }
    
    isTyping = false;
    document.getElementById('sendBtn').disabled = false;
    scrollToBottom();
}

// ========================================
// 消息显示
// ========================================
function appendMessage(role, content, animate = true) {
    const messages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'ai' ? '🤖' : '👤';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatMarkdown(content);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    messages.appendChild(messageDiv);
    
    if (animate) scrollToBottom();
}

function appendEnhancedCard(enhanced, existingCard = null, animate = true) {
    const messages = document.getElementById('chatMessages');
    const card = existingCard || document.createElement('div');
    if (!existingCard) {
        card.className = 'enhanced-card';
        messages.appendChild(card);
    }
    
    let html = '';
    
    // 查询路由分析
    if (enhanced.router) {
        const tierColors = {
            'tier_1_simple': '#4caf50',
            'tier_2_multi_fact': '#ff9800',
            'tier_3_cross_ref': '#f44336',
            'tier_4_synthesis': '#9c27b0'
        };
        const tierLabels = {
            'tier_1_simple': '简单',
            'tier_2_multi_fact': '中等',
            'tier_3_cross_ref': '复杂',
            'tier_4_synthesis': '高度综合'
        };
        const color = tierColors[enhanced.router.tier] || '#666';
        const label = tierLabels[enhanced.router.tier] || enhanced.router.tier;
        
        html += `
            <div class="enhanced-section">
                <div class="enhanced-header">🔍 查询路由分析</div>
                <div class="enhanced-grid">
                    <div class="enhanced-item">
                        <span class="label">复杂度</span>
                        <span class="value" style="color:${color};font-weight:bold">${label}</span>
                    </div>
                    <div class="enhanced-item">
                        <span class="label">策略</span>
                        <span class="value">${enhanced.router.strategy}</span>
                    </div>
                    <div class="enhanced-item">
                        <span class="label">置信度</span>
                        <span class="value">${(enhanced.router.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Persona 记忆
    if (enhanced.persona && enhanced.persona.length > 0) {
        html += `
            <div class="enhanced-section">
                <div class="enhanced-header">🧠 Persona 记忆 (${enhanced.persona.length} 条)</div>
                <div class="enhanced-list">
                    ${enhanced.persona.map(p => `<div class="persona-tag">${p}</div>`).join('')}
                </div>
            </div>
        `;
    }
    
    // 幻觉检测
    if (enhanced.hallucination) {
        const isHallucination = enhanced.hallucination.is_hallucination;
        const icon = isHallucination ? '⚠️' : '✅';
        const color = isHallucination ? '#f44336' : '#4caf50';
        html += `
            <div class="enhanced-section">
                <div class="enhanced-header">${icon} 幻觉检测</div>
                <div class="enhanced-grid">
                    <div class="enhanced-item">
                        <span class="label">状态</span>
                        <span class="value" style="color:${color};font-weight:bold">${isHallucination ? '可能幻觉' : '正常'}</span>
                    </div>
                    <div class="enhanced-item">
                        <span class="label">类型</span>
                        <span class="value">${enhanced.hallucination.hallucination_type}</span>
                    </div>
                </div>
                ${enhanced.hallucination.correction_suggestion && enhanced.hallucination.correction_suggestion !== '未发现幻觉' ? 
                    `<div class="enhanced-note" style="color:#f44336">💡 ${enhanced.hallucination.correction_suggestion}</div>` : ''}
            </div>
        `;
    }
    
    // 确定性引用
    if (enhanced.citation && enhanced.citation.has_citation) {
        html += `
            <div class="enhanced-section">
                <div class="enhanced-header">📚 引用 (${enhanced.citation.citations.length} 条)</div>
                <div class="enhanced-list">
                    ${enhanced.citation.citations.map(c => `<div class="citation-item">📌 ${c.content}</div>`).join('')}
                </div>
            </div>
        `;
    }
    
    // 多索引检索
    if (enhanced.retrieval && enhanced.retrieval.length > 0) {
        html += `
            <div class="enhanced-section">
                <div class="enhanced-header">🔎 检索结果 (${enhanced.retrieval.length} 条)</div>
                <div class="enhanced-list">
                    ${enhanced.retrieval.map(r => `<div class="retrieval-item">[${r.domain}] ${r.content}</div>`).join('')}
                </div>
            </div>
        `;
    }
    
    if (html) {
        card.innerHTML = html;
        messages.appendChild(card);
        if (animate) scrollToBottom();
    }
}

function showTypingIndicator() {
    const messages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai';
    typingDiv.id = 'typingIndicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    typingDiv.appendChild(avatar);
    typingDiv.appendChild(contentDiv);
    messages.appendChild(typingDiv);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();
}

function scrollToBottom() {
    const messages = document.getElementById('chatMessages');
    messages.scrollTop = messages.scrollHeight;
}

// ========================================
// Markdown 格式化
// ========================================
function formatMarkdown(text) {
    if (!text || typeof text !== 'string') return '';
    text = text.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>');
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    text = text.replace(/^\s*[-*+]\s+(.+)$/gm, '<li>$1</li>');
    text = text.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    text = text.replace(/^\s*\d+\.\s+(.+)$/gm, '<li>$1</li>');
    text = text.replace(/\n/g, '<br>');
    return text;
}

// ========================================
// 工具函数
// ========================================
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('collapsed');
}

function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('simpleagent_theme', isDark ? 'dark' : 'light');
    document.querySelector('.theme-toggle').textContent = isDark ? '☀️' : '🌙';
}

function clearChat() {
    if (confirm('确定要清空当前对话吗？')) {
        const messages = document.getElementById('chatMessages');
        messages.innerHTML = `
            <div class="message welcome-message">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <h3>对话已清空 🧹</h3>
                    <p>请输入你的问题，我将为你解答。</p>
                </div>
            </div>
        `;
    }
}

// ========================================
// 增强工具面板
// ========================================
function showEnhancedTools() {
    document.getElementById('enhancedToolsPanel').classList.add('open');
}

function hideEnhancedTools() {
    document.getElementById('enhancedToolsPanel').classList.remove('open');
}

async function classifyQuery() {
    const query = document.getElementById('queryClassifierInput').value;
    if (!query) return;
    
    const resultDiv = document.getElementById('queryResult');
    resultDiv.innerHTML = '<div class="loading-spinner"></div> 分析中...';
    
    try {
        const response = await fetch('/api/query/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: query })
        });
        const data = await response.json();
        const tierLabels = {
            'tier_1_simple': '简单', 'tier_2_multi_fact': '中等',
            'tier_3_cross_ref': '复杂', 'tier_4_synthesis': '高度综合'
        };
        const tierLabelsZh = {
            'tier_1_simple': '🟢 简单', 'tier_2_multi_fact': '🟡 中等',
            'tier_3_cross_ref': '🔴 复杂', 'tier_4_synthesis': '🟣 高度综合'
        };
        resultDiv.innerHTML = `
            <div class="enhanced-result">
                <div><strong>复杂度级别：</strong>${tierLabelsZh[data.tier] || data.tier}</div>
                <div><strong>检索策略：</strong>${data.strategy}</div>
                <div><strong>置信度：</strong>${(data.confidence * 100).toFixed(0)}%</div>
                <div><strong>复杂度分数：</strong>${data.complexity_score.toFixed(2)}</div>
                ${data.indicators && data.indicators.length ? `<div style="margin-top:8px;color:#888;font-size:12px">指标：${data.indicators.join(', ')}</div>` : ''}
            </div>
        `;
    } catch (error) {
        resultDiv.innerHTML = '❌ 错误: ' + error.message;
    }
}

async function extractPersona() {
    const text = document.getElementById('personaInput').value;
    if (!text) return;
    
    const resultDiv = document.getElementById('personaResult');
    resultDiv.innerHTML = '<div class="loading-spinner"></div> 提取中...';
    
    try {
        const response = await fetch('/api/persona/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        const facts = data.facts || [];
        let html = `<div class="enhanced-result"><strong>提取到 ${facts.length} 条事实：</strong><br>`;
        data.facts.forEach(f => {
            html += `<div style="margin:4px 0;padding:4px 8px;background:#f0f0f0;border-radius:4px">
                <span style="color:#667eea;font-weight:bold">[${f.domain}]</span> ${f.fact}
                <span style="color:#999;font-size:11px">(${(f.confidence*100).toFixed(0)}%)</span>
            </div>`;
        });
        html += '</div>';
        resultDiv.innerHTML = html;
    } catch (error) {
        resultDiv.innerHTML = '❌ 错误: ' + error.message;
    }
}

async function detectHallucination() {
    const text = document.getElementById('hallucinationInput').value;
    if (!text) return;
    
    const resultDiv = document.getElementById('hallucinationResult');
    resultDiv.innerHTML = '<div class="loading-spinner"></div> 检测中...';
    
    try {
        const response = await fetch('/api/hallucination/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        const icon = data.is_hallucination ? '⚠️' : '✅';
        const color = data.is_hallucination ? '#f44336' : '#4caf50';
        let html = `<div class="enhanced-result">
            <div style="font-size:18px;margin-bottom:8px">${icon} <span style="color:${color};font-weight:bold">${data.is_hallucination ? '发现潜在幻觉' : '未发现幻觉'}</span></div>
            <div><strong>置信度：</strong>${(data.confidence * 100).toFixed(0)}%</div>
            <div><strong>类型：</strong>${data.hallucination_type}</div>
            ${data.correction_suggestion && data.correction_suggestion !== '未发现幻觉' ? `<div style="margin-top:8px;color:#f44336">💡 ${data.correction_suggestion}</div>` : ''}
            ${data.evidence && data.evidence.length ? `<div style="margin-top:8px"><strong>证据：</strong><ul>${data.evidence.map(e => `<li>${e}</li>`).join('')}</ul></div>` : ''}
        </div>`;
        resultDiv.innerHTML = html;
    } catch (error) {
        resultDiv.innerHTML = '❌ 错误: ' + error.message;
    }
}

async function extractCitation() {
    const text = document.getElementById('citationInput').value;
    if (!text) return;
    
    const resultDiv = document.getElementById('citationResult');
    resultDiv.innerHTML = '<div class="loading-spinner"></div> 提取中...';
    
    try {
        const response = await fetch('/api/citation/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await response.json();
        let html = `<div class="enhanced-result">`;
        if (data.has_citation) {
            const citations = data.citations || [];
            html += `<div style="color:#4caf50;font-weight:bold">✅ 发现 ${citations.length} 条引用</div>`;
            citations.forEach(c => {
                html += `<div style="margin:4px 0;padding:4px 8px;background:#f0f0f0;border-radius:4px">📌 ${c.content}</div>`;
            });
        } else {
            html += `<div style="color:#999">未找到引用</div>`;
        }
        html += `</div>`;
        resultDiv.innerHTML = html;
    } catch (error) {
        resultDiv.innerHTML = '❌ 错误: ' + error.message;
    }
}

async function searchMultiIndex() {
    const query = document.getElementById('multiIndexInput').value;
    if (!query) return;
    
    const resultDiv = document.getElementById('multiIndexResult');
    resultDiv.innerHTML = '<div class="loading-spinner"></div> 搜索中...';
    
    try {
        const response = await fetch('/api/multi-index/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: query })
        });
        const data = await response.json();
        let html = `<div class="enhanced-result"><strong>搜索 "${query}" 的结果：</strong>`;
        if (data.results.length > 0) {
            data.results.forEach((r, i) => {
                html += `<div style="margin:6px 0;padding:6px 8px;background:#f0f0f0;border-radius:4px">
                    <div>${i + 1}. <span style="color:#667eea;font-weight:bold">[${r.domain}]</span> Score: ${r.score.toFixed(3)}</div>
                    <div style="margin-left:16px">${r.content}</div>
                </div>`;
            });
        } else {
            html += `<div style="color:#999">未找到相关结果</div>`;
        }
        html += `</div>`;
        resultDiv.innerHTML = html;
    } catch (error) {
        resultDiv.innerHTML = '❌ 错误: ' + error.message;
    }
}

// ========================================
// 主题初始化
// ========================================
function initTheme() {
    const saved = localStorage.getItem('simpleagent_theme');
    if (saved === 'dark') {
        document.body.classList.add('dark-mode');
        document.querySelector('.theme-toggle').textContent = '☀️';
    }
}
