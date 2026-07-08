import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { v4 as uuidv4 } from 'uuid'
import { streamChat } from '../../api/client'
import { applyAgentFormUpdate } from './interactionSlice'

const sessionId = uuidv4()

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
            label: `Calling tool: ${data.tool}`,
            detail: JSON.stringify(data.args),
          })
        )
      } else if (event === 'tool_result') {
        dispatch(
          chatSlice.actions.addToolEvent({
            label: `${data.tool} → done`,
            detail: data.result,
          })
        )
      } else if (event === 'token') {
        dispatch(chatSlice.actions.appendAssistantText(data.text))
      } else if (event === 'form_update') {
        dispatch(applyAgentFormUpdate(data))
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
      state.messages.push({ role: 'assistant', content: '', tools: [] })
    },
    appendAssistantText(state, action) {
      const last = state.messages[state.messages.length - 1]
      last.content = (last.content || '') + (last.content ? '\n' : '') + action.payload
    },
    addToolEvent(state, action) {
      const last = state.messages[state.messages.length - 1]
      if (!last.tools) last.tools = []
      last.tools.push(action.payload)
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