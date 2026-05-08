import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import api from '../api/client'
import PageHeader from '../components/PageHeader'

const SUGGESTIONS = [
  'What is our total revenue?',
  'Show me the 30-day sales forecast',
  'Which customer segment has the highest churn risk?',
  'Why might revenue be declining?',
  'What are our top product categories?',
  'Analyze our delivery performance',
  'What is the average order value?',
  'How can we improve customer retention?',
]

function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const WELCOME_MSG = {
  id: 'welcome',
  role: 'ai',
  content: `👋 **Hello! I'm your AI Business Intelligence Consultant.**

I'm powered by a LangGraph multi-agent pipeline with access to:
- 📊 **Live Olist database** (KPIs, SQL queries)
- 🔮 **Prophet forecasting model** (30-day revenue projections)
- 🎯 **Churn DNN** (customer segment risk analysis)
- 📚 **Business knowledge base** (strategy playbooks via RAG)
- 🔬 **11 analytical tools** (revenue trends, delivery, CLV, etc.)

Ask me anything about your business! Try one of the suggestions below, or type your own question.`,
  time: new Date(),
}

export default function Chat() {
  const [messages, setMessages]   = useState([WELCOME_MSG])
  const [input, setInput]         = useState('')
  const [loading, setLoading]     = useState(false)
  const [threadId]                = useState(() => `session_${Date.now()}`)
  const [agentReady, setReady]    = useState(null)
  const bottomRef                 = useRef(null)
  const textareaRef               = useRef(null)

  // Check agent health
  useEffect(() => {
    api.get('/chat/health')
      .then(data => setReady(data.ready))
      .catch(() => setReady(false))
  }, [])

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const sendMessage = async (text) => {
    const msg = text || input.trim()
    if (!msg || loading) return
    setInput('')

    const userMsg = { id: Date.now(), role: 'user', content: msg, time: new Date() }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await api.post('/chat', { message: msg, thread_id: threadId })
      const aiMsg = {
        id: Date.now() + 1,
        role: 'ai',
        content: res.response || 'No response received.',
        time: new Date(),
      }
      setMessages(prev => [...prev, aiMsg])
    } catch (err) {
      const errMsg = {
        id: Date.now() + 1,
        role: 'ai',
        content: `⚠️ **Error:** ${err.message}\n\nMake sure the FastAPI server is running and GROQ_API_KEY is set in \`.env\`.`,
        time: new Date(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const clearChat = () => {
    setMessages([WELCOME_MSG])
  }

  return (
    <div className="fade-up" style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      <PageHeader
        title="AI Business Consultant"
        subtitle="LangGraph · Groq Llama 3.3 70B · RAG + Tools + Long-Term Memory"
      >
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          {agentReady !== null && (
            <span className={`badge ${agentReady ? 'badge-success' : 'badge-danger'}`}>
              {agentReady ? '🟢 Agent Ready' : '🔴 Agent Offline'}
            </span>
          )}
          <button className="btn btn-ghost" onClick={clearChat} style={{ fontSize: 13 }}>
            🗑️ Clear
          </button>
        </div>
      </PageHeader>

      {/* Suggestion pills */}
      {messages.length <= 1 && (
        <div className="suggestion-pills" style={{ marginBottom: 16 }}>
          {SUGGESTIONS.map((s, i) => (
            <button key={i} className="pill" onClick={() => sendMessage(s)}>
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Chat area */}
      <div className="chat-container" style={{ flex: 1 }}>
        {/* Messages */}
        <div className="chat-messages" id="chat-messages">
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-message ${msg.role}`}>
              <div className={`msg-avatar ${msg.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
                {msg.role === 'user' ? '👤' : '🧠'}
              </div>
              <div>
                <div className="msg-bubble">
                  {msg.role === 'ai' ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content}
                    </ReactMarkdown>
                  ) : (
                    <p>{msg.content}</p>
                  )}
                </div>
                <div className="msg-time">{formatTime(msg.time)}</div>
              </div>
            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="chat-message ai">
              <div className="msg-avatar ai-avatar">🧠</div>
              <div className="msg-bubble" style={{ padding: '10px 18px' }}>
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <span style={{ fontSize: 12, color: 'var(--text-muted)', marginLeft: 8 }}>
                    Thinking...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input Area */}
        <div className="chat-input-area">
          <div className="chat-input-row">
            <textarea
              ref={textareaRef}
              className="chat-input"
              placeholder="Ask about revenue, churn, forecasts, delivery... (Enter to send, Shift+Enter for new line)"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              disabled={loading}
            />
            <button
              id="send-btn"
              className="btn btn-primary"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              style={{ minWidth: 100, height: 46 }}
            >
              {loading ? '⏳' : '➤ Send'}
            </button>
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
            Thread: <code style={{ fontFamily: 'JetBrains Mono', fontSize: 10 }}>{threadId}</code>
            {' · '}
            {messages.length - 1} message(s) in session
          </div>
        </div>
      </div>
    </div>
  )
}
