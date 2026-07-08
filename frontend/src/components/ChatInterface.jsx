import React, { useState, useRef, useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { sendMessage } from '../store/slices/chatSlice'

const formatToolName = (name = '') =>
  name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())

function ToolActivity({ tools, formFilled }) {
  const [open, setOpen] = useState(false)
  if (!tools?.length) return null

  const completedCount = tools.filter((tool) => tool.status === 'done').length
  const allComplete = completedCount === tools.length
  const statusTitle = allComplete
    ? formFilled
      ? 'Agentic AI updated the form'
      : 'Agentic AI finished'
    : 'Agentic AI is working'
  const statusMeta = allComplete
    ? `${tools.length} tool${tools.length === 1 ? '' : 's'} completed`
    : `${completedCount} of ${tools.length} tools completed`

  return (
    <div className="tool-activity">
      <button className="tool-toggle" type="button" onClick={() => setOpen((value) => !value)}>
        <span className={`tool-summary-icon ${allComplete ? 'complete' : 'running'}`} />
        <span className="tool-summary-copy">
          <strong>{statusTitle}</strong>
          <small>{statusMeta}</small>
        </span>
        <span className="tool-toggle-action">{open ? 'Hide' : 'View'}</span>
      </button>
      {open && (
        <div className="tool-list">
          {tools.map((tool, index) => (
            <div className="tool-item" key={`${tool.name}-${index}`}>
              <div className="tool-item-header">
                <span className={`tool-status tool-status-${tool.status}`} />
                <strong>{formatToolName(tool.name)}</strong>
                <span>{tool.status === 'calling' ? 'Running' : 'Done'}</span>
              </div>
              {tool.detail && <pre className="tool-detail">{tool.detail}</pre>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

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
    <aside className="chat-window">
      <div className="assistant-header">
        <h2>AI Assistant</h2>
        <p>Log interaction details here via chat</p>
      </div>
      <div className="chat-messages" ref={scrollRef}>
        {messages.map((m, idx) => (
          <React.Fragment key={idx}>
            <div className={`chat-bubble ${m.role}`}>
              {m.content || (isStreaming && idx === messages.length - 1 ? '…' : '')}
            </div>
            <ToolActivity tools={m.tools} formFilled={m.formFilled} />
          </React.Fragment>
        ))}
        {isStreaming && <div className="typing-indicator">Agentic AI is working…</div>}
      </div>
      <div className="chat-input-row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Describe the visit, call, or ask to edit a past entry…"
          disabled={isStreaming}
        />
        <button className="btn btn-primary log-btn" onClick={handleSend} disabled={isStreaming}>
          Log
        </button>
      </div>
    </aside>
  )
}
