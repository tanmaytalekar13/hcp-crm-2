import { configureStore } from '@reduxjs/toolkit'
import interactionReducer from './slices/interactionSlice'
import chatReducer from './slices/chatSlice'
import uiReducer from './slices/uiSlice'

export const store = configureStore({
  reducer: {
    interaction: interactionReducer,
    chat: chatReducer,
    ui: uiReducer,
  },
})
