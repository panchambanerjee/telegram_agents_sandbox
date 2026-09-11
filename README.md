# telegram_agents_sandbox

Three Telegram bots — **Nietzsche**, **Sartre**, and **Camus** — debate in a group chat. Each has its own BotFather token, a persona prompt, a conviction score that drifts as the conversation goes, and a RAG index over source texts from that philosopher’s tradition.

Drop a topic in the group. The bots poll new messages, retrieve passages from their books, call OpenAI, show a typing indicator, pause, then post short replies in the same group (not in DMs).

## How it works

```
Telegram group
      │
      ▼
Dispatcher (listens with NietzscheBot)
      │  logs humans → SQLite
      ▼
Nietzsche / Sartre / Camus loops
      │  unread messages → RAG → gpt-5-nano → typing → post
      ▼
Same group chat
```

| Piece | Role |
|---|---|
| `telegram_agents/main.py` | Loads `.env`, SQLite, Chroma, starts dispatcher + three agent loops |
| `core/dispatcher.py` | Polls Telegram, logs human text, handles `/help` `/reset` `/summary` `/convictions` |
| `core/base_agent.py` | Shared poll / RAG / LLM / typing / post loop |
| `agents/personas.py` | Nietzsche, Sartre, Camus subclasses |
| `core/database.py` | Async SQLite: `message_log`, `agent_cursor`, `conviction_scores` |
| `rag/embedder.py` | Chunks `.pdf` / `.txt` / `.md` into Chroma per agent |
| `rag/retriever.py` | Query-time retrieval, biased by conviction |

Only **NietzscheBot** listens for updates. All three bots must be **members of the group**, and they all **post** with their own tokens. Human messages in the group are visible to NietzscheBot and written to SQLite; the other two read that log.

## Prerequisites

- Python 3.9+
- An OpenAI API key
- Three Telegram bots from [@BotFather](https://t.me/BotFather)
- A Telegram **group** with you and all three bots
- Source texts on disk (PDFs are gitignored; keep them locally)

## Telegram setup

1. Create three bots (suggested names: NietzscheBot, SartreBot, CamusBot). Copy each token.
2. For **each** bot in BotFather: `/setprivacy` → **Disable**. If privacy is on, they ignore ordinary group messages.
3. Add all three bots to your group.
4. Copy `telegram_agents/.env.example` to `telegram_agents/.env` and fill tokens + `OPENAI_API_KEY`.
5. Put a placeholder `GROUP_CHAT_ID` first, start the app, then in the group send `/help`. NietzscheBot replies with the real chat id (a negative number). Put that in `.env` and restart.

Private chats with a single bot are ignored unless that chat id is `GROUP_CHAT_ID`. The intended surface is the **group**.

## Source texts (RAG)

Place files under `telegram_agents/texts/`. Folder names can be the handle or the tradition:

```
telegram_agents/texts/
├── nihilist/            # or nihilism/
│   └── genealogy_of_morality.pdf
├── existentialism/      # or existentialist/
│   └── being_and_nothingess.pdf
└── absurdism/           # or absurdist/
    └── myth_of_sisyphus.pdf
```

Supported: `.pdf`, `.txt`, `.md`. First run extracts, chunks (~400 words, 50-word overlap), and upserts into Chroma at `CHROMA_PATH`. Later starts skip files that are already loaded. PDFs are not committed (copyright).

Conviction steers retrieval: low score appends doubt language to the query; high score appends certainty language.

## Install and run

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r telegram_agents/requirements.txt
```

```bash
cd telegram_agents
cp .env.example .env   # then edit
python main.py
```

You should see RAG load (or “already loaded”), three agents ready, then:

```
✅ 3 philosopher(s) running. Drop a topic in your Telegram group.
```

Ctrl+C shuts down cleanly.

First ingest of a long PDF (e.g. *Being and Nothingness*) can take a few minutes. The first OpenAI calls can also take a while; the bots should show **typing** in Telegram during that wait.

## Group commands

| Command | What it does |
|---|---|
| `/help` | Short usage text. If `GROUP_CHAT_ID` is wrong, replies with this chat’s id |
| `/reset` | Deletes that chat’s `message_log` rows and resets agent cursors |
| `/convictions` | Text bars for each philosopher’s score |
| `/summary` | Camus summarises the last 40 messages |

Plain text (not a command) is logged as `sender=user` and becomes the next debate turn.

## Conversation storage

Chats are **not** stored in Telegram history for the app. They go to SQLite:

`telegram_agents/data/conversations.db` (`DB_PATH`)

| Table | Contents |
|---|---|
| `message_log` | Humans and bots: `sender`, `sender_name`, `content`, `telegram_msg_id`, `created_at` |
| `agent_cursor` | Last `message_log.id` each agent has processed |
| `conviction_scores` | Float in `[0, 1]`, default `0.5` |

Inspect:

```bash
cd telegram_agents
sqlite3 -header -column data/conversations.db \
  "SELECT sender_name, content, created_at FROM message_log ORDER BY id;"
```

Embeddings live in `telegram_agents/data/chroma`. Both `data/` and `.env` are gitignored.

## Environment variables

See `telegram_agents/.env.example`.

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Chat completions (`gpt-5-nano` in code) |
| `NIHILIST_BOT_TOKEN` | NietzscheBot; also the **listener** |
| `EXISTENTIALIST_BOT_TOKEN` | SartreBot |
| `ABSURDIST_BOT_TOKEN` | CamusBot |
| `GROUP_CHAT_ID` | Target group (negative integer) |
| `RESPONSE_DELAY_MIN` / `MAX` | Typing pause before each post (seconds) |
| `CONTEXT_WINDOW` | Recent SQLite rows sent to the model |
| `DB_PATH` | SQLite file |
| `CHROMA_PATH` | Chroma persistence dir |
| `TEXTS_PATH` | Source texts directory |

Each agent has a `response_probability` (Nietzsche 0.80, Sartre 0.75, Camus 0.70) so they do not all answer every turn.

## Layout

```
telegram_agents_sandbox/
├── README.md
├── .gitignore
└── telegram_agents/
    ├── main.py
    ├── requirements.txt
    ├── .env.example
    ├── agents/personas.py
    ├── core/
    │   ├── base_agent.py
    │   ├── database.py
    │   └── dispatcher.py
    ├── rag/
    │   ├── embedder.py
    │   └── retriever.py
    ├── texts/          # local PDFs; not in git
    └── data/           # SQLite + Chroma; not in git
```

## Troubleshooting

**Bots never reply, no `[Dispatcher] Logged from …` in the terminal**  
`GROUP_CHAT_ID` is still the example, or NietzscheBot is not in the group, or privacy mode is on. Send `/help` in the group and copy the id it prints.

**Messages log but nobody posts, or `[Name] LLM error: Unsupported parameter: max_tokens`**  
Reasoning models need `max_completion_tokens`, not `max_tokens`. That is already in `base_agent.py`. Restart after pulling.

**Empty replies / silent skip after “Calling LLM”**  
`gpt-5-nano` can spend the budget on hidden reasoning. The code uses `reasoning_effort=low` and a higher completion limit. Send a **new** message; already-processed rows will not be retried (`agent_cursor`).

**`RuntimeError: threads can only be started once`**  
Old aiosqlite pattern `async with await connect()`. Current `core/database.py` uses a single context manager.

**Sartre `Timed out` on send**  
Telegram send timeouts were increased. Restart so that build is running.

**fontTools / urllib3 warnings on startup**  
Filtered in `main.py`. Already-ingested PDFs are not re-parsed.

**Agents skip a lot**  
Normal: random skip plus cursor advance. `/reset` if you want a clean thread.

## License and texts

Application code in this repo is yours to use as a sandbox. Do **not** commit scanned or publisher PDFs of Nietzsche, Sartre, or Camus; keep them under `texts/` locally.
