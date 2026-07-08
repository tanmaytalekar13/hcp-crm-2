import { createSlice } from '@reduxjs/toolkit'

const uiSlice = createSlice({
  name: 'ui',
  initialState: {
    mode: 'chat', // 'chat' | 'form'
  },
  reducers: {
    setMode(state, action) {
      state.mode = action.payload
    },
  },
})

export const { setMode } = uiSlice.actions
export default uiSlice.reducer
