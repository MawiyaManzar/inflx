# AutoStream Social-to-Lead Agent

This project implements a conversational AI agent for the fictional SaaS product **AutoStream**.  
The agent can:
- detect user intent (`greeting`, `product_inquiry`, `high_intent`)
- answer product/policy questions using local RAG knowledge
- collect lead details (`name`, `email`, `platform`)
- call a mock lead-capture tool only after all required fields are valid

## How To Run Locally

### 1) Prerequisites
- Python 3.10+
- `uv` installed

### 2) Install dependencies
```bash
uv pip install -r requirements.txt
```

### 3) Configure environment
Create `.env` in the project root:
```env
GEMINI_API_KEY=your_google_ai_api_key
```

Optional (to avoid Hugging Face rate-limit warning on model download):
```env
HF_TOKEN=your_huggingface_token
```

### 4) Run the agent
```bash
python -m src.main
```

### 5) Run tests
```bash
uv run python -m pytest -q tests/test_task5_gating.py
```

## Architecture Explanation

I chose **LangGraph** because this assignment requires explicit multi-step control flow and reliable state transitions, not just free-form chat completion. LangGraph lets us model the conversation as nodes and edges (`classify`, `retrieve`, `lead_progress`, `capture_lead`, `respond`) with deterministic routing logic. That makes tool-gating rules easy to enforce and debug. For knowledge retrieval, the project uses a local markdown knowledge base (`data/autostream_kb.md`) that is chunked and indexed in an in-memory vector store with SentenceTransformer embeddings (`all-MiniLM-L6-v2`). This gives lightweight, local RAG behavior while keeping implementation simple.

State is managed as a structured dictionary (`AgentState`) carried across turns in the graph loop. It stores conversation memory (`messages`), current `intent`, RAG context (`retrieved_context`), lead fields (`lead`), remaining requirements (`missing_fields`), and tool completion (`lead_captured`). A `pending_question` field preserves product questions when users mix inquiry and buying intent in one message. After lead capture succeeds, the agent can automatically answer that pending question without forcing the user to repeat it. This state-first design keeps behavior predictable, testable, and aligned with real lead qualification flows.

## WhatsApp Integration Using Webhooks

To integrate this agent with WhatsApp in production, I would use the **WhatsApp Cloud API** (Meta) with a webhook-based backend service.

1. Create a webhook endpoint (for example, using FastAPI) that receives incoming WhatsApp message events.
2. Verify webhook signatures/tokens from Meta for security.
3. Map each WhatsApp sender ID (phone number) to a persisted agent state record (Redis/Postgres) so memory is preserved between messages.
4. On each inbound message:
   - load state for that sender
   - set `last_user_message` and append to `messages`
   - run the LangGraph app
   - persist updated state
5. Send the agent’s latest response back to the user with WhatsApp’s send-message API endpoint.
6. Log tool events (lead capture), add retries/dead-letter handling, and include monitoring/alerts for reliability.

This design keeps the same core graph logic while adding transport, persistence, authentication, and operational reliability needed for real deployment.
