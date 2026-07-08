import React from 'react'
import ChatInterface from './components/ChatInterface'
import StructuredForm from './components/StructuredForm'

export default function App() {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>Log Interaction</h1>
          <p>HCP Module · AI-First CRM for field representatives</p>
        </div>
      </header>

      <div className="layout-grid">
        <StructuredForm />
        <ChatInterface />
      </div>
    </div>
  )
}
