import React, { useState, useRef, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { sendMessage } from '../store/slices/chatSlice'

export default function ChatInterface() {
  const dispatch = useDispatch()
  const messages = useSelector((state) => state.chat.messages)
  const isStreaming = useSelector((state) => state.chat.isStreaming)
  const [draft, setDraft] = useState('')
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!draft.trim() || isStreaming) return
    dispatch(sendMessage(draft.trim()))
    setDraft('')
  }

  return (
    <div className="card chat-window">
      <h2>Chat with the AI agent</h2>
      <div className="chat-messages" ref={scrollRef}>
        {messages.map((m, idx) => (
          <React.Fragment key={idx}>
            <div className={`chat-bubble ${m.role}`}>
              {m.content || (isStreaming && idx === messages.length - 1 ? '…' : '')}
            </div>
            {m.tools && m.tools.length > 0 && (
              <div className="chat-bubble tool">
                {m.tools.map((t, i) => (
                  <div key={i}>
                    <strong>{t.label}</strong>
                    <div style={{ opacity: 0.75 }}>{t.detail}</div>
                  </div>
                ))}
              </div>
            )}
          </React.Fragment>
        ))}
        {isStreaming && <div className="typing-indicator">Agent is thinking…</div>}
      </div>
      <div className="chat-input-row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Describe the visit, call, or ask to edit a past entry…"
          disabled={isStreaming}
        />
        <button className="btn btn-primary" onClick={handleSend} disabled={isStreaming}>
          Send
        </button>
      </div>
      <p className="hint-text">
        Try: "Log a call with Dr. Rohan Kulkarni about GlucoBalance, he was neutral, follow up in 2 weeks."
      </p>
    </div>
  )
}
