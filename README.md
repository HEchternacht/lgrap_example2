# lgrap

OpenAI-compatible HTTP API in front of a **LangGraph agent**. Any OpenAI client (or UI) can talk to it at `/v1/chat/completions`, with streaming, while the agent behind it can call tools. Points at a local `llama-server` / any OpenAI-compatible backend by default; LangSmith tracing is optional.

## What it does
- `POST /v1/chat/completions` (streaming and non-streaming) and `GET /v1/models` (allowlist from config), OpenAI schema.
- LangGraph agent (`app/agent/graph.py`) with tools: `calculator`, `search`, `weather` (`app/tools/`, registered in `registry.py`).
- Per-user chat history and logs in SQL (`db_app/`: SQLAlchemy models for users, chat history, logs; history routes under `/v1`), session signed with `SECRET_KEY`.
- Run manager for cancelling/tracking runs, SSE streaming helpers (`app/managers`, `app/utils/streaming.py`).
- Optional LangSmith tracing (`LANGSMITH_*`).

## Install
Python 3.12+. With [uv](https://github.com/astral-sh/uv):
```bash
git clone https://github.com/HEchternacht/lgrap_example2
cd lgrap_example2
uv sync                 # or: pip install -e .
cp .env.example .env    # then edit
```

## Configure (`.env`)
| Var | Default | |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | backend (llama-server, OpenAI, ...) |
| `OPENAI_API_KEY` | `not-needed` | |
| `MODEL_NAME` | `gpt-4` | used when the request names no model |
| `AVAILABLE_MODELS` | `gpt-4` | comma-separated allowlist |
| `SECRET_KEY` | `change-me-in-production` | session signing - change it |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | off | optional tracing |

## Run
```bash
python main.py                  # or the console script:  lgrap
uvicorn main:app --reload       # development
curl http://localhost:8000/v1/models
```
Use it from any OpenAI client by setting its base URL to `http://localhost:8000/v1`. `wb.ipynb` is a scratch notebook.
