import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { v4 as uuidv4 } from 'uuid'
import { streamChat } from '../../api/client'
import { applyAgentFormUpdate } from './interactionSlice'

const sessionId = uuidv4()

const formatJson = (value) => {
  try {
    const parsed = typeof value === 'string' ? JSON.parse(value) : value
    return JSON.stringify(parsed, null, 2)
  } catch {
    return typeof value === 'string' ? value : JSON.stringify(value, null, 2)
  }
}

export const sendMessage = createAsyncThunk(
  'chat/sendMessage',
  async (userText, { dispatch }) => {
    dispatch(chatSlice.actions.addMessage({ role: 'user', content: userText }))
    dispatch(chatSlice.actions.setStreaming(true))
    dispatch(chatSlice.actions.beginAssistantMessage())

    await streamChat({ sessionId, message: userText }, ({ event, data }) => {
      if (event === 'tool_call') {
        dispatch(
          chatSlice.actions.addToolEvent({
            name: data.tool,
            status: 'calling',
            detail: formatJson(data.args),
          })
        )
      } else if (event === 'tool_result') {
        dispatch(
          chatSlice.actions.addToolEvent({
            name: data.tool,
            status: 'done',
            detail: formatJson(data.result),
          })
        )
      } else if (event === 'token') {
        dispatch(chatSlice.actions.appendAssistantText(data.text))
      } else if (event === 'form_update') {
        dispatch(applyAgentFormUpdate(data))
        dispatch(chatSlice.actions.markFormFilled(data))
      } else if (event === 'done') {
        dispatch(chatSlice.actions.finalizeAssistantMessage(data.final_text))
      }
    })

    dispatch(chatSlice.actions.setStreaming(false))
  }
)

const chatSlice = createSlice({
  name: 'chat',
  initialState: {
    sessionId,
    messages: [
      {
        role: 'assistant',
        content:
          'Hi! Tell me about the HCP interaction you\'d like to log — e.g. "I met Dr. Anjali Mehta today, discussed CardioMax, she was positive and wants a follow-up study next month."',
      },
    ],
    isStreaming: false,
  },
  reducers: {
    addMessage(state, action) {
      state.messages.push(action.payload)
    },
    beginAssistantMessage(state) {
      state.messages.push({ role: 'assistant', content: '', tools: [], formFilled: null })
    },
    appendAssistantText(state, action) {
      const last = state.messages[state.messages.length - 1]
      last.content = (last.content || '') + (last.content ? '\n' : '') + action.payload
    },
    addToolEvent(state, action) {
      const last = state.messages[state.messages.length - 1]
      if (!last.tools) last.tools = []
      const incoming = action.payload
      if (incoming.status === 'done') {
        const existing = [...last.tools]
          .reverse()
          .find((tool) => tool.name === incoming.name && tool.status === 'calling')
        if (existing) {
          existing.status = 'done'
          existing.detail = incoming.detail
          return
        }
      }
      last.tools.push(incoming)
    },
    markFormFilled(state, action) {
      const last = state.messages[state.messages.length - 1]
      const interaction = action.payload || {}
      const filledFields = [
        interaction.hcp_name,
        interaction.summary,
        interaction.next_action,
        interaction.topics_discussed?.length,
        interaction.products_discussed?.length,
        interaction.samples_distributed?.length,
      ].filter(Boolean).length

      last.formFilled = {
        hcpName: interaction.hcp_name || 'interaction',
        filledFields,
        interactionId: interaction.id,
      }
    },
    finalizeAssistantMessage(state, action) {
      const last = state.messages[state.messages.length - 1]
      const finalText = action.payload
      if (!last.content && finalText) {
        last.content = finalText
      }
    },
    setStreaming(state, action) {
      state.isStreaming = action.payload
    },
  },
})

export default chatSlice.reducer
