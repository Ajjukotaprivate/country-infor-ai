# 🌍 Country Information AI Agent

A production-grade AI agent that answers questions about countries using the public [REST Countries API](https://restcountries.com), built with **LangGraph**, **Groq**, and **Streamlit**.

This project implements a scalable AI workflow strictly following the requirements:
- **LangGraph** orchestration (no single-prompt wrappers)
- **Three-step flow**: Intent Identification → Tool Invocation → Answer Synthesis
- **Zero-State Constraints**: No authentication, no database, no embeddings, and no RAG.

---

## 🚀 Deliverables

- 🔗 **Live Demo**: [Insert Streamlit Cloud / HuggingFace Spaces Link Here]
- 📺 **Video Walkthrough**: [Insert Loom / YouTube Link Here]

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│              LangGraph StateGraph               │
│                                                 │
│  intent_node ──► router_node                   │
│                      │                          │
│               ┌──────┴──────┐                   │
│               ▼             ▼                   │
│          tool_node     reject_node              │
│         (REST API)    (off-topic guard)         │
│               │                                 │
│               ▼                                 │
│        synthesis_node                           │
│       (grounded answer)                         │
└─────────────────────────────────────────────────┘
```

### Node Responsibilities

| Node | Role |
|------|------|
| `intent_node` | Extracts country name + requested fields via Pydantic structured output |
| `router_node` | Pure-logic conditional edge — no LLM call |
| `tool_node` | Async httpx call to REST Countries API with cache + retry |
| `synthesis_node` | LLM answer grounded strictly on API JSON |
| `reject_node` | Politely refuses off-topic queries |
| `error_node` | Global fallback for unexpected failures |

---

## Folder Structure

```
├── src/
│   ├── agent/
│   │   ├── graph.py          # StateGraph assembly + MemorySaver
│   │   ├── nodes.py          # All node functions
│   │   └── state.py          # AgentState + IntentResult schema
│   ├── tools/
│   │   └── country_api.py    # Async httpx REST Countries client
│   └── core/
│       ├── config.py         # Pydantic Settings (all env vars)
│       └── memory_manager.py # Token-based sliding window
├── tests/
│   ├── unit/                 # Logic tests (no LLM)
│   └── integration/          # Full graph tests (requires API key)
├── evals/
│   ├── dataset.json          # 18 ground truth Q&A pairs
│   └── scorer.py             # Evaluation script
├── main.py                   # Streamlit UI
├── pytest.ini
├── requirements.txt
└── .env.example
```

---

## Production Design Decisions

### Security
- **ChatML role separation**: User input never touches the system prompt string
- **Out-of-scope guardrail**: Pydantic-validated `is_off_topic` flag rejects non-country queries before any tool call
- **Prompt injection defence**: System prompt explicitly instructs the LLM to ignore instructions embedded in user messages

### Resilience
- **Timeouts**: 10-second HTTP timeout on every API call
- **Retry**: 1 retry with exponential backoff on timeout
- **Graceful 404**: Returns friendly "country not found" message, not a stack trace
- **Global error node**: Catches unexpected graph failures

### Performance
- **Async all the way**: All nodes are `async def` — safe for high-concurrency serving
- **TTL cache**: In-memory country data cache (5-minute TTL) — repeated queries don't hit the API
- **Token trimming**: Sliding window ensures the LLM context never overflows

### Observability
- **Structured logging**: Every node logs `[Node: name]` with key state values
- **LangSmith compatible**: Graph structure is natively traceable via LangSmith

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Ajjukotaprivate/country-infor-ai.git
cd country-infor-ai
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com
```

### 3. Run

```bash
streamlit run main.py
```

---

## Evaluation

```bash
# Unit tests (no API key needed)
pytest tests/unit/ -v

# Integration tests (requires GROQ_API_KEY + internet)
pytest tests/integration/ -v

# Eval against golden dataset
python -m evals.scorer
```

---

## Known Limitations & Trade-offs

| Limitation | Trade-off |
|-----------|-----------|
| In-memory cache resets on restart | Simplicity > persistence; Redis would solve this in production |
| No streaming response | Avoids complexity; Streamlit streaming can be added with `st.write_stream` |
| Single-model (Groq) | No fallback LLM; production would add OpenAI as a backup |
| Entity normalisation via LLM | LLM may mis-normalise rare country aliases; a lookup table would be more reliable |
| Lexical eval scoring | Substring matching is fast but misses paraphrase; LLM-as-a-Judge would be more accurate |

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — Agent orchestration
- **[Groq](https://groq.com)** — `llama-3.1-8b-instant` inference
- **[httpx](https://www.python-httpx.org/)** — Async HTTP
- **[Pydantic](https://docs.pydantic.dev/)** — Structured output validation
- **[Streamlit](https://streamlit.io)** — Chat UI
- **[REST Countries API](https://restcountries.com)** — Free, no auth
