import React, { useEffect } from 'react'
import { useDispatch, useSelector } from 'react-redux'
import { fetchHcps } from '../store/slices/interactionSlice'

const arrayToText = (arr) => (Array.isArray(arr) ? arr.join(', ') : '')
const toDateInput = (value) => (value ? value.slice(0, 10) : '')
const toTimeInput = (value) => (value ? value.slice(11, 16) : '')

export default function StructuredForm() {
  const dispatch = useDispatch()
  const form = useSelector((state) => state.interaction.form)
  const hcps = useSelector((state) => state.interaction.hcps)
  const selectedHcp = hcps.find((h) => h.id === form.hcp_id)
  const hcpDisplayName = selectedHcp
    ? `${selectedHcp.name} - ${selectedHcp.specialty}`
    : form.hcp_name

  useEffect(() => {
    dispatch(fetchHcps())
  }, [dispatch])

  return (
    <section className="interaction-panel">
      <div className="panel-title">Interaction Details</div>

      <div className="form-grid two-col">
        <div className="field">
          <label>HCP Name</label>
          <input
            value={hcpDisplayName || ''}
            placeholder="Search or select HCP..."
            readOnly
            disabled
          />
        </div>

        <div className="field">
          <label>Interaction Type</label>
          <select value={form.interaction_type} disabled>
            <option value="visit">Meeting</option>
            <option value="call">Phone Call</option>
            <option value="email">Email</option>
            <option value="conference">Conference</option>
            <option value="sample_drop">Sample Drop</option>
          </select>
        </div>

        <div className="field">
          <label>Date</label>
          <input type="date" value={toDateInput(form.interaction_date)} readOnly disabled />
        </div>

        <div className="field">
          <label>Time</label>
          <input type="time" value={toTimeInput(form.interaction_date)} readOnly disabled />
        </div>
      </div>

      <div className="field">
        <label>Attendees</label>
        <input value={selectedHcp?.name || form.hcp_name || ''} placeholder="Enter names or search..." readOnly disabled />
      </div>

      <div className="field">
        <label>Topics Discussed</label>
        <textarea
          value={arrayToText(form.topics_discussed) || form.raw_notes}
          placeholder="Enter key discussion points..."
          readOnly
          disabled
        />
      </div>

      <button className="link-action" type="button" disabled>
        Summarize from Voice Note (Requires Consent)
      </button>

      <div className="section-heading">Materials Shared / Samples Distributed</div>
      <div className="material-box">
        <div>
          <strong>Materials Shared</strong>
          <p>{arrayToText(form.products_discussed) || 'No materials added.'}</p>
        </div>
        <button className="mini-btn" type="button" disabled>Search/Add</button>
      </div>

      <div className="material-box">
        <div>
          <strong>Samples Distributed</strong>
          <p>{arrayToText(form.samples_distributed) || 'No samples added.'}</p>
        </div>
        <button className="mini-btn" type="button" disabled>Add Sample</button>
      </div>

      <div className="field">
        <label>Observed/Inferred HCP Sentiment</label>
        <div className="sentiment-options" aria-disabled="true">
          {['positive', 'neutral', 'negative'].map((sentiment) => (
            <label key={sentiment} className="sentiment-option">
              <input
                type="radio"
                checked={form.sentiment === sentiment}
                readOnly
                disabled
              />
              <span>{sentiment[0].toUpperCase() + sentiment.slice(1)}</span>
            </label>
          ))}
        </div>
      </div>

      <div className="field">
        <label>Outcomes</label>
        <textarea
          value={form.summary}
          placeholder="Key outcomes or agreements..."
          readOnly
          disabled
        />
      </div>

      <div className="field">
        <label>Follow-up Actions</label>
        <textarea
          value={form.next_action}
          placeholder="Enter next steps or tasks..."
          readOnly
          disabled
        />
      </div>

      <div className="ai-suggestions">
        <strong>AI Suggested Follow-ups:</strong>
        <span>{form.next_action || 'Follow-up suggestions will appear after the assistant logs the interaction.'}</span>
      </div>
    </section>
  )
}
