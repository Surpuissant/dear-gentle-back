---
title: "Dear Gentle API"
description: |
  Compact reference for the HTTP endpoints that power the Dear Gentle backend.
  The same content is available as Markdown (for humans) and as structured JSON
  via `/api/docs` so that tooling or AI agents can consume it without scraping
  the repository.
---

### Overview

- **Base URL**: all routes are mounted under `/api/...`.
- **Success envelope**: most responses follow `{ "ok": true, ... }`.
- **Errors**: standard FastAPI HTTP errors (4xx/5xx) with a JSON body.
- **Authentication**: none (the service relies on `user_id` in the payloads).

### Models

| Model | Description | Fields |
| --- | --- | --- |
| `ChatRequest` | Launch a conversation or authoring turn. | `user_id`, `book_id`, `session_id`, `message`, `snapshot_override?` |
| `ChatResponse` | Response payload for `/api/chat`. | `output`, `mode`, `used_facets`, `used_memory_ids`, `chapter_id?` |
| `SeedPrefRequest` | Set or remove user preferences. | `user_id`, `items[]` of `{key, value}` |
| `SeedMemoryRequest` | Insert manual memory snippets. | `user_id`, `texts[]` |
| `SetSnapshotRequest` | Replace the full session snapshot. | `session_id`, `snapshot` |
| `Book` | Book metadata persisted for each user. | `id`, `user_id`, `title`, `outline`, `themes`, `style`, timestamps |
| `ChapterEditRequest` | Apply an instruction to rewrite a chapter. | `user_id`, `chapter_id`, `edit_instruction` |
| `AutoMemActionRequest` | Accept or reject auto-captured memories. | `user_id`, `ids[]` |

### Core endpoints

#### POST `/api/chat`
- **Purpose**: process a conversation turn or author a new chapter section.
- **Body**: `ChatRequest`.
- **Notes**:
  - Validates `book_id`.
  - Detects mode (`conversation`, `author`, `instruction`, ...).
  - In `instruction` mode, stores the instruction and returns an empty `output`.
  - Persists responses, snapshots, facets and memory recency.
- **Response**: `ChatResponse`.

#### GET `/api/conversations/{session_id}`
- **Purpose**: retrieve the recent messages for a session so the UI can rehydrate after a refresh.
- **Query params**:
  - `limit` (default: 50) — cap the number of returned messages (set to `0` for no cap).
  - `include_instructions` (default: `false`) — include `>>` instruction turns when `true`.
- **Response**: `{ "ok": true, "items": [{role, content, ts, mode}, ...], "summary": str|null }`.

### Auto-memories

- **GET** `/api/memories/auto/pending?user_id=...`
  - Returns `{ "ok": true, "items": [...] }` with pending memories.
- **POST** `/api/memories/auto/accept`
  - Body: `AutoMemActionRequest`.
  - Returns accepted counters and IDs.
- **POST** `/api/memories/auto/reject`
  - Body: `AutoMemActionRequest`.
  - Deletes the listed proposals.

### User preferences

- **POST** `/api/seed/preferences`
  - Body: `SeedPrefRequest`.
  - Upserts or deletes key/value pairs.
- **GET** `/api/preferences?user_id=...`
  - Returns all `{key, value}` entries.

### Manual memories

- **POST** `/api/seed/memories`
  - Body: `SeedMemoryRequest`.
  - Creates `MemoryItem` entries and embeddings.
  - Response: `{ "ok": true, "count": <int> }`.

### Session snapshot

- **POST** `/api/snapshot/set`
  - Body: `SetSnapshotRequest`.
  - Stores a full snapshot for the session.

### Books

- **POST** `/api/book/upsert`
  - Body: `Book` (with `id`).
  - Updates `updated_at`, stores the book, returns `{ "ok": true, "book_id": ... }`.

### Instruction overrides

- **GET** `/api/instructions?user_id=...`
  - Lists custom instructions with `index` and `active` flag.
- **POST** `/api/instructions/toggle`
  - Body: `{ "user_id": str, "index": int, "active": bool }`.
- **DELETE** `/api/instructions`
  - Body: `{ "user_id": str, "indexes": [int, ...] }`.

### Styles

- **GET** `/api/styles`
  - Returns the available style packs metadata.
- **GET** `/api/styles/{style_id}`
  - Returns the details for a style pack.
- **POST** `/api/styles/select`
  - Body: `{ "user_id": str, "style_id": str }`.

### Chapters

- **GET** `/api/chapters?book_id=...`
  - Lists chapters for a book (sorted by `index`).
- **GET** `/api/chapters/{chapter_id}`
  - Returns the chapter and its versions.
- **POST** `/api/chapters/{chapter_id}/edit`
  - Body: `ChapterEditRequest`.
  - Returns `{ "ok": true, "item": <Chapter>, "used_facets": [...], "used_memory_ids": [...] }`.

### Healthcheck

- **GET** `/api/health`
  - Returns `{ "ok": true }`.

