# Enterprise Document Intelligence — Deliverables

AI Solution Architect assessment submission: a zero-hallucination Document
AI pipeline for verbatim key-value extraction from Amazon.com, Inc. Form
10-K (FY2025), with a local Streamlit runner and a production architecture
that is scalable, authenticated, cached, circuit-broken, and
GPU-autoscaled.

---

## Files in this folder

| File | What it is |
|------|------------|
| `streamlit_app.py` | **Local app.** Two tabs: single-file extraction and batch queue + dashboard. |
| `engine/` | Shared extraction engine — used by both Streamlit and the production workers. |
| `extraction_output.json` | **Deliverable 2.** Final JSON payload (5 extractions + citations). |
| `ARCHITECTURE.md` | **Deliverable 1 (text).** Full production architecture. |
| `ARCHITECTURE_DIAGRAM.svg` | **Deliverable 1 (diagram).** Single-page visual. |
| `DEBRIEF.md` | **Deliverable 3.** The hardest technical challenge in this document. |
| `pipeline/` | Production stack — FastAPI + Celery + JWT + Kong + Kubernetes manifests. |
| `requirements.txt` | Python dependencies. |
| `case-study.pdf` | Source document (Amazon 10-K, FY ended 2025-12-31). |

---

## Quick start

```bash
git clone <this-repo> && cd RAG
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open <http://localhost:8501>, switch to whichever tab you want, upload
PDFs, and click **Run** / **Submit**. The deterministic backend runs
fully offline — no LLM or internet required.

---

## 1. Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.10 or 3.11 | `engine/` uses 3.10-style union types (`str \| None`). |
| `pip` | ≥ 23 | For the venv install path. |
| `git` | any recent | Cloning the repo. |
| *(optional)* Ollama | 0.3+ | Runs a local LLM for LLM-grounded extraction. |

Disk: ~300 MB for Python deps, ~5 GB extra if you pull `llama3.1:8b`.

---

## 2. Environment setup

### 2.1 — Clone the project

```bash
git clone <this-repo>
cd RAG
```

### 2.2 — Create an isolated Python environment

Pick one.

**Option A — built-in `venv` (recommended).**

```bash
python3.10 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip wheel
```

**Option B — `conda`.**

```bash
conda create -n doc-intel python=3.10 -y
conda activate doc-intel
```

### 2.3 — Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the core extraction stack (`pdfplumber`,
`rank-bm25`, `numpy`), local embeddings (`sentence-transformers`,
`torch`, `torchvision`), the Streamlit UI, caching, and the production
FastAPI / Celery / Kong / circuit-breaker libraries. `torchvision` is
listed explicitly because `transformers.models.zoedepth` eagerly imports
it at module-load time and some environments omit it — pinning it here
prevents the `ModuleNotFoundError: No module named 'torchvision'` seen
on fresh installs.

### 2.4 — *(Optional)* Install Ollama for LLM-grounded extraction

The Streamlit demo works fully offline with the deterministic backend.
If you want to see the LLM-grounded path, install Ollama and pull a
model:

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh
# Windows: https://ollama.com/download/windows

ollama pull llama3.1:8b
```

Start the Ollama daemon in a spare terminal (it auto-starts on macOS /
Windows installers):

```bash
ollama serve
```

The Streamlit sidebar auto-detects Ollama and switches its status dot
from red to green when reachable.

---

## 3. Launching the application

### 3.1 — Start the Streamlit app

Make sure your venv is active, then:

```bash
streamlit run streamlit_app.py
```

The terminal will print:

```
  You can now view your Streamlit app in your browser.

  Local URL:   http://localhost:8501
  Network URL: http://<your-ip>:8501
```

Open <http://localhost:8501>.

### 3.2 — Using the UI

**Sidebar.** Pick the backend (`auto` / `llm` / `deterministic`) and see
the live Ollama status.

**Tab 1 — Single extraction.** Upload one PDF, click **▶ Run
extraction**, and get the JSON output inline plus a download button.
Results are also persisted into the local SQLite store so they appear
on the dashboard.

**Tab 2 — Batch & Dashboard.** Upload many PDFs at once, click **Submit
N file(s) to queue**, and watch the live counters (Queued / Processing
/ Completed / Failed / In flight). The document table shows per-file
status, page count, latency, backend, and — on failures — the first 80
characters of the error reason. Select a row to drill into its
extractions or failure detail. The **↻ Refresh** button polls the
store (the production build uses WebSocket push).

### 3.3 — Configuration knobs

Everything is environment-variable driven, so you don't need to edit
code for normal tuning.

| Env var | Default | What it controls |
|---|---|---|
| `DOC_INTEL_WORKERS` | `2` | Concurrent extractions in the local worker pool (`engine/job_queue.py`). |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Alternate sentence-transformers model id. |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | If Ollama runs elsewhere. |
| `OLLAMA_MODEL` | `llama3.1:8b` | Swap in a different local model. |

Example — more concurrent extractions on a beefy laptop:

```bash
DOC_INTEL_WORKERS=6 streamlit run streamlit_app.py
```

### 3.4 — Where local data lives

| What | Path |
|---|---|
| Document + extraction store (SQLite) | `~/.cache/doc-intel/store.sqlite` |
| Content-addressed result/embedding cache | `~/.cache/doc-intel/cache/` |

Delete either directory to start from scratch. The dashboard has a
**🧹 Reset dashboard** button that wipes the store from inside the UI.

### 3.5 — Stopping the app

`Ctrl+C` in the terminal running Streamlit. Background worker threads
are daemon threads and exit with the process.

---

## 4. The 5 extracted answers (summary)

| Field | Answer (verbatim) | Page |
|-------|-------------------|------|
| AWS Net Sales (2025) | `$ 128,725` million | p. 107 (Note 10 — Segment Information) |
| Employee headcount | `approximately 1,576,000 full-time and part-time employees` | p. 6 (Item 1 — Human Capital) |
| Fulfillment equipment useful life | `Three to ten years` (under the "Other equipment" row) | p. 75 (Note 1 — Property and Equipment) |
| Primary retail competition | *"We believe that the principal competitive factors in our retail businesses include selection, price, and convenience, including fast and reliable fulfillment."* | p. 6 (Item 1 — Competition) |
| Foreign Exchange Risk | *"The results of operations of, and certain of our intercompany balances associated with, our internationally-focused stores and AWS are exposed to foreign exchange rate fluctuations."* | p. 56 (Item 7A — Foreign Exchange Risk) |

Every value is substring-verified against its cited page before
emission — values that fail verification are dropped, never paraphrased.
The extraction rules are value-free (see `engine/patterns.py`), so the
same pipeline runs against any year's 10-K.

---

## 5. Production deploy (reference)

```bash
# API gateway (auth + rate limit + proxy cache)
deck gateway sync pipeline/deploy/api-gateway/kong.yaml

# Stateless API tier (HPA on CPU + RPS, 3 → 50 replicas)
kubectl apply -f pipeline/deploy/kubernetes/api-deployment.yaml

# GPU worker pool (KEDA 0 → 20 on queue depth, Karpenter for nodes)
kubectl apply -f pipeline/deploy/kubernetes/gpu-workers.yaml
```

See `ARCHITECTURE.md` and `ARCHITECTURE_DIAGRAM.svg` for the full picture
of the production stack (API gateway, message queue, circuit breakers,
GPU autoscaling, observability, security model).