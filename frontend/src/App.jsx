import React from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { setMode } from './store/slices/uiSlice'
import ChatInterface from './components/ChatInterface'
import StructuredForm from './components/StructuredForm'

export default function App() {
  const dispatch = useDispatch()
  const mode = useSelector((state) => state.ui.mode)

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Log Interaction</h1>
          <p>HCP Module · AI-First CRM for field representatives</p>
        </div>
        <div className="mode-toggle">
          <button
            className={mode === 'chat' ? 'active' : ''}
            onClick={() => dispatch(setMode('chat'))}
          >
            Chat
          </button>
          <button
            className={mode === 'form' ? 'active' : ''}
            onClick={() => dispatch(setMode('form'))}
          >
            Structured Form
          </button>
        </div>
      </header>

      <div className="layout-grid">
        {mode === 'chat' ? (
          <>
            <ChatInterface />
            <StructuredForm readOnlyHint />
          </>
        ) : (
          <StructuredForm />
        )}
      </div>
    </div>
  )
}
