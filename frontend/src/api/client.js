const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'

export async function apiGet(path) {
  const res = await fetch(`${BASE_URL}${path}`)
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`)
  return res.json()
}

export async function apiPost(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`)
  return res.json()
}

export async function apiPatch(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`PATCH ${path} failed: ${res.status}`)
  return res.json()
}

/**
 * Streams the LangGraph agent's SSE response for a chat message.
 * Calls `onEvent({ event, data })` for every server-sent event:
 * "token", "tool_call", "tool_result", "form_update", "done".
 */
export async function streamChat({ sessionId, message }, onEvent) {
  const res = await fetch(`${BASE_URL}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  })
  if (!res.ok || !res.body) throw new Error(`chat stream failed: ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // Normalize \r\n -> \n so event boundaries split correctly regardless
    // of which line-ending style the SSE server uses.
    buffer = buffer.replace(/\r\n/g, '\n')

    const events = buffer.split('\n\n')
    buffer = events.pop() // last chunk may be incomplete

    for (const raw of events) {
      if (!raw.trim()) continue
      let eventName = 'message'
      let data = ''
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) eventName = line.slice(6).trim()
        else if (line.startsWith('data:')) data += line.slice(5).trim()
      }
      let parsed = null
      try {
        parsed = JSON.parse(data)
      } catch {
        parsed = { raw: data }
      }
      onEvent({ event: eventName, data: parsed })
    }
  }
}