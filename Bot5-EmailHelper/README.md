---
title: 5thEmailBot
emoji: ⚡
colorFrom: red
colorTo: red
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Overview
5thEmailBot is a lightweight Flask web app (packaged for Hugging Face Space via Docker) that assists customer support agents in drafting, reviewing, and justifying email replies. It combines:
* Retrieval of internal Q&A knowledge (`Questions.txt`)
* Example response and policy similarity search (SentenceTransformers)
* Multi‑agent style prompting (analyzer, drafter, reviewer, sentiment, justifiers) using Together AI LLMs

The app produces: analysis, sentiment, draft reply, review notes, policy + example rationales, and knowledge source attribution.

## Key Features
* Knowledge base lookup with semantic fallback to keyword matching
* Example + policy embedding retrieval (SentenceTransformer with graceful degradation)
* Structured multi‑role prompts (analyzer/drafter/reviewer/etc.)
* Justification of selected examples and policies
* Simple web UI (index page with sample emails)
* JSON API endpoints for automation
* Dockerized for reproducible deployment (HF Space compatible)

## Architecture (High Level)
```
Browser → Flask (app.py)
	├─ EmailProcessingSystem
	│    ├─ KnowledgeRetriever (Questions.txt)
	│    ├─ EmailResponseRetriever (examples)
	│    ├─ PolicyRetriever (policies)
	│    └─ Multiple EmailAgent roles (analyzer / drafter / reviewer / etc.)
	└─ Together AI (chat completions)
```

## Project Structure
```
app.py              # Flask app + processing pipeline
config.json         # Local dev config (tokens & model name) – DO NOT commit real secrets
Questions.txt       # Plaintext Q/A knowledge base
templates/index.html
static/             # Static assets placeholder
requirements.txt
Dockerfile
space.yml           # (HF Space config)
```

## Prerequisites
* Python 3.11+
* (Optional) Docker
* Together AI API key
* (Optional) Hugging Face token (if private models are needed)

## Installation (Local)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
You can supply credentials either via `config.json` (development only) or environment variables (recommended for production).

### Option 1: config.json (development)
Example (use placeholders, never commit real keys):
```json
{
	"model": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",
	"hf_token": "<hf_token_optional>",
	"together_ai_token": "<together_api_key>"
}
```

### Option 2: Environment Variables
Set before running:
```bash
export TOGETHER_API_KEY="<together_api_key>"
export HF_TOKEN="<hf_token_optional>"
```
Then adjust `app.py` to read from `os.environ` if you remove `config.json`.

## Running the App
```bash
python app.py
```
Default host: `0.0.0.0` (for container use) on port `5005` (or `PORT` env var). Navigate to: http://localhost:5005

## Docker
Build & run locally:
```bash
docker build -t 5themailbot .
docker run -e TOGETHER_API_KEY="<together_api_key>" -p 7860:7860 5themailbot
```
If deploying as a Hugging Face Space, the `Dockerfile` + `space.yml` are already prepared (Space expects the app to listen on `$PORT`, currently set to 7860 inside the container). Ensure the container invokes the correct port (`ENV PORT=7860`).

## API Endpoints
| Method | Path        | Description                                    |
|--------|-------------|------------------------------------------------|
| GET    | /           | UI with sample emails                          |
| POST   | /process    | Process an email; returns JSON with analysis   |
| POST   | /approve    | Increment approved counter                     |
| POST   | /disapprove | Increment disapproved counter                  |
| GET    | /stats      | Return approval/disapproval counts             |

### /process Response (fields)
* status
* analysis
* final_draft
* review
* policies
* examples
* policy_justification
* example_justification
* sentiment
* knowledge_source (Direct match | Semantic search | Not found)

## Knowledge Base (`Questions.txt`)
Plain text file, alternating lines of question and answer (blank lines allowed). Avoid including URLs-only lines or they will be skipped. Update this file to expand coverage; semantic embeddings regenerate on app start.

## Updating Example Responses / Policies
Edit the dictionaries inside `EmailResponseRetriever.examples` and `PolicyRetriever.policies` in `app.py`. Restart to apply changes.

## Error Handling & Fallbacks
* If SentenceTransformer models fail to load, the system falls back to keyword matching.
* If no knowledge is found, draft indicates missing knowledge base coverage.
* Model loading tries several candidates for robustness.

## Performance Notes
Initial load may download transformer weights; use a persistent volume or Docker layer caching to avoid repeated downloads.

## Security / Privacy
IMPORTANT: Remove real API keys from `config.json` before committing. Prefer environment variables or a secrets manager. Never expose customer PII in logs.

## Troubleshooting
| Issue | Fix |
|-------|-----|
| SentenceTransformer load errors | Check CPU/RAM; ensure internet access for first download |
| Empty knowledge responses | Populate / correct `Questions.txt` format |
| 401 / Together API errors | Verify `TOGETHER_API_KEY` value and account quota |
| Container port mismatch | Ensure `ENV PORT` and `EXPOSE` match and host mapping `-p host:container` |

## Roadmap (Ideas)
* Replace fixed example/policy dicts with external JSON store
* Add authentication for admin updates
* Add vector database (FAISS already listed, not yet wired) for scalable retrieval
* Add unit tests & CI
* Stream responses

## License
Add a LICENSE file if distributing publicly.

---
Maintainer tips: keep secrets out of version control, pin critical model versions for reproducibility, and monitor token usage.
