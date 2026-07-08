import { createSlice, createAsyncThunk } from '@reduxjs/toolkit'
import { apiGet, apiPost, apiPatch } from '../../api/client'

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

export const submitFormInteraction = createAsyncThunk(
  'interaction/submitForm',
  async (form) => {
    const payload = {
      hcp_id: form.hcp_id,
      interaction_type: form.interaction_type,
      interaction_date: form.interaction_date,
      raw_notes: form.raw_notes,
      summary: form.summary,
      topics_discussed: form.topics_discussed,
      products_discussed: form.products_discussed,
      samples_distributed: form.samples_distributed,
      sentiment: form.sentiment,
      next_action: form.next_action,
      follow_up_date: form.follow_up_date || null,
      created_via: 'form',
    }
    if (form.id) {
      return apiPatch(`/interactions/${form.id}`, payload)
    }
    return apiPost('/interactions', payload)
  }
)

const interactionSlice = createSlice({
  name: 'interaction',
  initialState: {
    form: emptyForm,
    hcps: [],
    status: 'idle',
  },
  reducers: {
    updateField(state, action) {
      const { field, value } = action.payload
      state.form[field] = value
    },
    /** Merge fields streamed back from the LangGraph agent (auto-fill) */
    applyAgentFormUpdate(state, action) {
      const interaction = action.payload
      state.form = {
        ...state.form,
        id: interaction.id,
        hcp_id: interaction.hcp_id,
        interaction_type: interaction.interaction_type || state.form.interaction_type,
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
      .addCase(submitFormInteraction.pending, (state) => {
        state.status = 'saving'
      })
      .addCase(submitFormInteraction.fulfilled, (state, action) => {
        state.status = 'saved'
        state.form.id = action.payload.id
      })
      .addCase(submitFormInteraction.rejected, (state) => {
        state.status = 'error'
      })
  },
})

export const { updateField, applyAgentFormUpdate, resetForm } = interactionSlice.actions
export default interactionSlice.reducer
