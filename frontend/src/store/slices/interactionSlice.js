import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { apiGet } from '../../api/client'

const emptyForm = {
  id: null,
  hcp_id: '',
  hcp_name: '',
  interaction_type: 'visit',
  interaction_date: new Date().toISOString().slice(0, 16),
  raw_notes: '',
  summary: '',
  topics_discussed: [],
  products_discussed: [],
  samples_distributed: [],
  sentiment: 'neutral',
  next_action: '',
  follow_up_date: '',
  is_edited: false,
}

export const fetchHcps = createAsyncThunk('interaction/fetchHcps', async (q) => {
  const query = q ? `?q=${encodeURIComponent(q)}` : ''
  return apiGet(`/hcps${query}`)
})

const interactionSlice = createSlice({
  name: 'interaction',
  initialState: {
    form: emptyForm,
    hcps: [],
    status: 'idle',
  },
  reducers: {
    /** Merge fields streamed back from the LangGraph agent (auto-fill) */
    applyAgentFormUpdate(state, action) {
      const interaction = action.payload
      state.form = {
        ...state.form,
        id: interaction.id,
        hcp_id: interaction.hcp_id,
        hcp_name: interaction.hcp_name || state.form.hcp_name,
        interaction_type: interaction.interaction_type || state.form.interaction_type,
        interaction_date: interaction.interaction_date
          ? interaction.interaction_date.slice(0, 16)
          : state.form.interaction_date,
        raw_notes: interaction.raw_notes || state.form.raw_notes,
        summary: interaction.summary || state.form.summary,
        topics_discussed: interaction.topics_discussed || [],
        products_discussed: interaction.products_discussed || [],
        samples_distributed: interaction.samples_distributed || [],
        sentiment: interaction.sentiment || state.form.sentiment,
        next_action: interaction.next_action || state.form.next_action,
        follow_up_date: interaction.follow_up_date
          ? interaction.follow_up_date.slice(0, 16)
          : state.form.follow_up_date,
        is_edited: interaction.is_edited || false,
      }
    },
    resetForm(state) {
      state.form = emptyForm
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchHcps.fulfilled, (state, action) => {
        state.hcps = action.payload
      })
  },
})

export const { applyAgentFormUpdate, resetForm } = interactionSlice.actions
export default interactionSlice.reducer
