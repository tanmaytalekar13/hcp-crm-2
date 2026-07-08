# AI-First CRM — HCP Module: Log Interaction Screen

An AI-first "Log Interaction" screen for a pharma CRM, built for the Round 1
technical assignment. Field reps can log or edit a Healthcare Professional
(HCP) interaction either through a **structured form** or by **chatting** with
an AI agent that fills the form out for them in real time.

## What this project does

- Rep opens the **Log Interaction** screen and picks **Chat** or **Form** mode.
- In **Chat mode**, the rep describes the visit/call in plain language (e.g.
  *"I met Dr. Anjali Mehta today, discussed CardioMax, she was positive and
  wants a follow-up study next month"*). A **LangGraph agent** streams back a
  response token-by-token, calls tools to look up the doctor, summarize the
  conversation, extract structured entities (products, topics, samples,
  sentiment, next steps), save the interaction to Postgres, and **live-fills
  the structured form** next to the chat so the rep can see/verify/edit exactly
  what will be saved.
- In **Form mode**, the rep can fill and submit the same fields directly,
  bypassing the agent entirely — useful for quick, unambiguous entries.
- Every interaction is persisted to Postgres and fully editable later (via
  chat — *"actually, change the sentiment to positive"* — or via the form).

## Tech stack

| Layer      | Technology                                   |
|------------|-----------------------------------------------|
| Frontend   | React + Redux Toolkit, Google **Inter** font  |
| Backend    | Python + FastAPI (Server-Sent Events streaming) |
| AI Agent   | LangGraph (StateGraph + ToolNode)             |
| LLM        | Groq — `gemma2-9b-it` (primary, mandated by task), `llama-3.3-70b-versatile` (available as a fallback for heavier reasoning) |
| Database   | PostgreSQL (SQLAlchemy ORM)                   |

## Architecture

```
frontend (React/Redux)
   │  POST /api/chat/stream (SSE)         POST/GET/PATCH /api/interactions
   ▼                                              ▼
FastAPI backend  ───────────────────────────────────────────────►  Postgres
   │
   ▼
LangGraph agent (StateGraph: agent ⇄ tools)
   │
   ▼
Groq LLM (gemma2-9b-it) — reasoning + tool-calling + summarization
```

### Role of the LangGraph agent

The agent is the single orchestrator between the rep's free-form chat and the
structured `interactions` table. It:

1. Keeps short-term conversational memory per `session_id` (backed by a
   `chat_messages` table) so a rep can log an interaction, then a few turns
   later say "change the follow-up date" without repeating context.
2. Decides — turn by turn — which tool(s) to call based on intent, in a
   `agent → tools → agent` loop until it has a final natural-language reply.
3. Streams every step back to the frontend over SSE: assistant tokens,
   `tool_call` events (which tool, with what arguments), `tool_result` events,
   and a `form_update` event whenever a `log_interaction`/`edit_interaction`
   tool finishes — which the frontend uses to auto-fill the structured form.

### The 5 LangGraph tools (`backend/app/agent/tools.py`)

1. **`search_hcp(query)`** — looks up an HCP by name/specialty/city so the
   agent can resolve "Dr. Mehta" to a concrete `hcp_id` before logging.
2. **`log_interaction(hcp_id, raw_notes, interaction_date?)`** *(required tool #1)* —
   creates a new interaction. Internally calls `gemma2-9b-it` with the rep's
   raw account of the visit and asks it to return structured JSON: a short
   summary, `topics_discussed`, `products_discussed`, `samples_distributed`,
   `sentiment`, `next_action`, and `interaction_type`. The parsed result is
   saved to Postgres and returned to the graph (and, via SSE, to the form).
3. **`edit_interaction(interaction_id, fields_to_update)`** *(required tool #2)* —
   updates any subset of fields on a previously logged interaction. Before
   applying changes it snapshots the prior version into an `edit_history` JSON
   column (audit trail). If only `raw_notes` changes, it re-runs the same LLM
   extraction so summary/topics/sentiment stay consistent with the new notes.
4. **`get_interaction_history(hcp_id, limit)`** — fetches the rep's most recent
   past interactions with that HCP, so the agent has context for questions
   like "what did we discuss last time" or to avoid repeat sample drops.
5. **`schedule_followup(interaction_id, follow_up_date, next_action)`** — sets
   or updates the follow-up date and next action on an interaction once the
   rep mentions a future commitment (e.g. "send her the new study in two
   weeks").

## Repository structure

```
hcp-crm/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, CORS, router registration
│   │   ├── config.py          # env-based settings (Groq key/model, DB URL)
│   │   ├── database.py        # SQLAlchemy engine/session
│   │   ├── models.py          # HCP, Interaction, ChatMessage tables
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── seed.py            # seeds a few sample HCPs for demoing
│   │   ├── agent/
│   │   │   ├── llm.py         # Groq (ChatGroq) wrappers
│   │   │   ├── tools.py       # the 5 LangGraph tools
│   │   │   └── graph.py       # the LangGraph StateGraph definition
│   │   └── routers/
│   │       ├── interactions.py  # direct-form REST CRUD (bypasses agent)
│   │       └── chat.py          # SSE streaming endpoint that runs the agent
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx                    # mode toggle (Chat / Form) + layout
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx      # chat window, streams SSE, shows tool calls
│   │   │   └── StructuredForm.jsx     # the structured log-interaction form
│   │   ├── store/
│   │   │   ├── store.js
│   │   │   └── slices/
│   │   │       ├── uiSlice.js          # chat/form mode toggle
│   │   │       ├── chatSlice.js        # messages + SSE-driven thunk
│   │   │       └── interactionSlice.js # form state + direct REST thunks
│   │   ├── api/client.js               # fetch helpers + SSE parser
│   │   └── styles/index.css            # Inter font, layout, chat bubbles
│   ├── index.html
│   └── package.json
├── docker-compose.yml   # spins up Postgres locally
└── README.md
```

## Running it locally

### 1. Get a free Groq API key

Go to **https://console.groq.com/keys**, sign in (no credit card required),
and create a key. Groq's free developer tier covers `gemma2-9b-it` and
`llama-3.3-70b-versatile` with generous rate limits — plenty for this demo.

### 2. Start Postgres

```bash
docker compose up -d
```

(Or point `DATABASE_URL` in `backend/.env` at any Postgres/MySQL-compatible
instance you already have running.)

### 3. Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your GROQ_API_KEY

python -m app.seed        # seeds a few sample HCPs
uvicorn app.main:app --reload --port 8000
```

The API is now at `http://localhost:8000`, docs at `http://localhost:8000/docs`.

### 4. Frontend

```bash
cd frontend
npm install
cp .env.example .env    # defaults to http://localhost:8000/api, adjust if needed
npm run dev
```

Open `http://localhost:5173`.

## Trying it out

- **Chat mode**: type something like *"I visited Dr. Anjali Mehta at Fortis
  Hospital, we discussed CardioMax and she seemed very receptive, left 2
  samples, and asked me to send the phase III study by next Friday."* Watch
  the tool calls appear (`search_hcp` → `log_interaction` → `schedule_followup`)
  and the form on the right fill itself in.
- Follow up with *"actually mark the sentiment as neutral instead"* to see
  `edit_interaction` fire and the form update in place.
- **Form mode**: fill and submit the fields directly — this path never touches
  the LLM and writes straight to Postgres, for reps who prefer it.

## Notes on the tech choices

- **`gemma2-9b-it`** does the mandated LLM work: summarization and entity
  extraction (topics/products/samples/sentiment/next-action) inside the
  `log_interaction` and `edit_interaction` tools (see `app/agent/tools.py`).
  Note that Groq has deprecated `gemma2-9b-it` in favor of `llama-3.1-8b-instant`,
  and it is currently kept only for older projects — it still works via the
  API, but you may want to swap in `llama-3.1-8b-instant` if Groq removes it
  entirely (single line change: `GROQ_MODEL` in `.env`).
- **`llama-3.3-70b-versatile`** drives the LangGraph agent's tool-calling /
  orchestration loop (`app/agent/graph.py`). Small, non-tool-tuned models like
  `gemma2-9b-it` are unreliable at knowing when to *stop* calling tools —
  in testing this caused the agent to loop between the `agent` and `tools`
  nodes indefinitely, burning Groq API calls without ever returning a
  response, until it hit LangGraph's recursion limit and crashed. Using
  `llama-3.3-70b-versatile` for orchestration (with `gemma2-9b-it` still doing
  the actual mandated extraction work inside the tools) fixed this. As a
  second line of defense, `MAX_TOOL_CALLS_PER_TURN` in `graph.py` forces the
  agent to answer in plain text after 6 tool calls in a single turn, so the
  graph can never loop forever even if a model misbehaves. The chat SSE
  endpoint also now wraps the whole stream in a try/except so any failure is
  sent to the frontend as a message instead of hanging silently.
- Server-Sent Events (rather than WebSockets) were used for streaming since
  the interaction is one-directional (agent → client) per turn and SSE is
  simpler to reason about and works over plain HTTP.
