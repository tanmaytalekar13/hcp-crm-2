import React, { useEffect, useState } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { fetchHcps, updateField, submitFormInteraction, resetForm } from '../store/slices/interactionSlice'

const arrayToText = (arr) => (Array.isArray(arr) ? arr.join(', ') : '')
const textToArray = (text) =>
  text.split(',').map((s) => s.trim()).filter(Boolean)

export default function StructuredForm({ readOnlyHint = false }) {
  const dispatch = useDispatch()
  const form = useSelector((state) => state.interaction.form)
  const hcps = useSelector((state) => state.interaction.hcps)
  const status = useSelector((state) => state.interaction.status)

  useEffect(() => {
    dispatch(fetchHcps())
  }, [dispatch])

  const set = (field) => (e) => dispatch(updateField({ field, value: e.target.value }))
  const setArray = (field) => (e) =>
    dispatch(updateField({ field, value: textToArray(e.target.value) }))

  const handleSave = () => dispatch(submitFormInteraction(form))

  return (
    <div className="card">
      <h2>{readOnlyHint ? 'Structured Interaction (auto-filled by chat)' : 'Log Interaction — Form'}</h2>

      {readOnlyHint && (
        <p className="hint-text" style={{ marginTop: -8, marginBottom: 14 }}>
          These fields fill in automatically as you chat. You can tweak anything
          below and hit Save to persist your edits.
        </p>
      )}

      <div className="field">
        <label>HCP</label>
        <select value={form.hcp_id} onChange={set('hcp_id')}>
          <option value="">Select an HCP…</option>
          {hcps.map((h) => (
            <option key={h.id} value={h.id}>
              {h.name} — {h.specialty} ({h.city})
            </option>
          ))}
        </select>
      </div>

      <div className="field">
        <label>Interaction Type</label>
        <select value={form.interaction_type} onChange={set('interaction_type')}>
          <option value="visit">In-person Visit</option>
          <option value="call">Phone Call</option>
          <option value="email">Email</option>
          <option value="conference">Conference</option>
          <option value="sample_drop">Sample Drop</option>
        </select>
      </div>

      <div className="field">
        <label>Date</label>
        <input
          type="datetime-local"
          value={form.interaction_date}
          onChange={set('interaction_date')}
        />
      </div>

      <div className="field">
        <label>Notes (what happened)</label>
        <textarea value={form.raw_notes} onChange={set('raw_notes')} />
      </div>

      <div className="field">
        <label>AI Summary</label>
        <textarea value={form.summary} onChange={set('summary')} />
      </div>

      <div className="field">
        <label>Topics Discussed</label>
        <input
          value={arrayToText(form.topics_discussed)}
          onChange={setArray('topics_discussed')}
          placeholder="comma-separated"
        />
        <div className="chip-row" style={{ marginTop: 6 }}>
          {form.topics_discussed?.map((t, i) => (
            <span className="chip" key={i}>{t}</span>
          ))}
        </div>
      </div>

      <div className="field">
        <label>Products Discussed</label>
        <input
          value={arrayToText(form.products_discussed)}
          onChange={setArray('products_discussed')}
          placeholder="comma-separated"
        />
      </div>

      <div className="field">
        <label>Samples Distributed</label>
        <input
          value={arrayToText(form.samples_distributed)}
          onChange={setArray('samples_distributed')}
          placeholder="comma-separated"
        />
      </div>

      <div className="field">
        <label>Sentiment</label>
        <select value={form.sentiment} onChange={set('sentiment')}>
          <option value="positive">Positive</option>
          <option value="neutral">Neutral</option>
          <option value="negative">Negative</option>
        </select>
        <span className={`sentiment-badge sentiment-${form.sentiment}`} style={{ marginTop: 6 }}>
          {form.sentiment}
        </span>
      </div>

      <div className="field">
        <label>Next Action</label>
        <input value={form.next_action} onChange={set('next_action')} />
      </div>

      <div className="field">
        <label>Follow-up Date</label>
        <input type="datetime-local" value={form.follow_up_date || ''} onChange={set('follow_up_date')} />
      </div>

      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn btn-primary" onClick={handleSave} disabled={!form.hcp_id || status === 'saving'}>
          {form.id ? 'Save Changes' : 'Log Interaction'}
        </button>
        <button className="btn btn-ghost" onClick={() => dispatch(resetForm())}>
          Clear
        </button>
      </div>
      {status === 'saved' && <p className="hint-text">Saved ✓</p>}
      {status === 'error' && <p className="hint-text" style={{ color: 'var(--color-negative)' }}>Failed to save.</p>}
    </div>
  )
}
