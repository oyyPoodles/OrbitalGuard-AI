import React, { useState, useRef, useEffect, useCallback } from 'react'

const API = 'http://localhost:8000'

const QUICK = [
  'What is debris 25544?',
  'How many objects tracked?',
  'Tell me about STARLINK',
  'What are rocket bodies?',
]

export default function OrbitalChat() {
  const [messages, setMessages] = useState([{
    role: 'bot',
    text: 'I am Orbital Bot 🛰️\nAsk about any orbital object by NORAD ID, name, or type.\n\nTry: "What is debris 25544?"',
    ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  }])
  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef()
  const inputRef              = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const send = useCallback(async (q) => {
    const query = (q || input).trim()
    if (!query || loading) return
    setInput('')
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    setMessages(prev => [...prev, { role: 'user', text: query, ts }])
    setLoading(true)
    try {
      const res  = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'bot',
        text: data.answer || 'No response.',
        ts: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }])
    } catch {
      setMessages(prev => [...prev, {
        role: 'bot',
        text: 'Cannot reach intelligence engine — backend offline.',
        ts,
      }])
    }
    setLoading(false)
    inputRef.current?.focus()
  }, [input, loading])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  return (
    <div className="narad-panel">
      {/* Header */}
      <div className="narad-header">
        <div className="narad-icon">🧠</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="narad-title">Orbital Bot</div>
          <div className="narad-badge">Debris Intelligence Engine</div>
        </div>
        <span style={{
          fontSize: 7, color: 'var(--green)', fontWeight: 700,
          padding: '2px 6px', background: 'var(--green-dim)',
          border: '1px solid rgba(16,185,129,0.3)', borderRadius: 10,
          textTransform: 'uppercase', letterSpacing: '0.5px', flexShrink: 0,
        }}>AI</span>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className={`msg-avatar ${m.role === 'bot' ? 'bot-av' : 'user-av'}`}>
              {m.role === 'bot' ? '🛰' : '👤'}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2, alignItems: m.role === 'user' ? 'flex-end' : 'flex-start' }}>
              <div className="msg-bubble">{m.text}</div>
              {m.ts && (
                <span style={{ fontSize: 8, color: 'var(--text-dim)', paddingLeft: 4, paddingRight: 4 }}>
                  {m.ts}
                </span>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-msg bot">
            <div className="msg-avatar bot-av">🛰</div>
            <div className="msg-typing">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Quick actions */}
      <div className="chat-quick-actions">
        {QUICK.map((q, i) => (
          <button key={i} className="quick-btn" onClick={() => send(q)}>
            {q.length > 22 ? q.slice(0, 22) + '…' : q}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="chat-input-row">
        <input
          ref={inputRef}
          className="chat-input"
          type="text"
          placeholder='"IE124" or "what is 25544?"'
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <button
          className="chat-send-btn"
          onClick={() => send()}
          disabled={loading || !input.trim()}
        >▶</button>
      </div>
    </div>
  )
}
