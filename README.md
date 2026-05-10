# Enterprise Agentic RAG Platform 🏢🤖

A production-grade, **Agentic Retrieval-Augmented Generation (RAG)** system built with LangChain, Groq, FAISS, and Streamlit. Features real-time streaming, multi-turn conversational memory, hybrid retrieval, hallucination guardrails, and a live telemetry dashboard.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **Real-Time Streaming** | Answers stream token-by-token like ChatGPT using `st.write_stream` |
| 🧠 **Conversational Memory** | Query Decontextualizer resolves pronouns in follow-up questions ("How do I claim *it*?") |
| 🔍 **Hybrid Retrieval** | Combines FAISS (dense) + BM25 (sparse) for maximum recall |
| 🎯 **Intent-Aware Search** | LLM classifies query intent to dynamically adjust retrieval depth |
| 🏆 **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranks retrieved chunks for precision |
| ✂️ **Context Compression** | Filters redundant sentences and enforces token limits before generation |
| 🛡️ **Hallucination Guardrails** | LLM verifier audits every answer against retrieved context |
| 📊 **Admin Dashboard** | Live telemetry: hallucination rate, retrieval scores, AI auto-tuning recommendations |
| 🗃️ **Feedback Database** | SQLite-backed logging of every query for continuous improvement |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Query Decontextualizer] ──► Resolves pronouns using chat history
    │
    ▼
[Intent Classifier] ──► Adjusts top_k for simple/complex queries
    │
    ▼
[Hybrid Retriever] ──► FAISS (dense) + BM25 (sparse) fusion
    │
    ▼
[Cross-Encoder Reranker] ──► Scores query-document pairs precisely
    │
    ▼
[Context Compressor] ──► Removes redundancy, enforces token limits
    │
    ▼
[LLM Generator (Groq)] ──► Generates cited, context-only answer
    │
    ▼
[Hallucination Verifier] ──► Audits answer faithfulness
    │
    ▼
[Evaluator + Feedback DB] ──► Logs metrics to SQLite
    │
    ▼
Streamlit UI (Streaming)
```

---

## 📁 Project Structure

```
enterprise_rag/
│
├── main.py                     # Streamlit UI (Chat + Admin Dashboard)
│
├── rag/
│   ├── __init__.py
│   ├── chunking.py             # SmartChunker (token-aware splitting)
│   ├── ingest.py               # Ingestion pipeline (FAISS + BM25 indexing)
│   ├── hybrid_retriever.py     # Intent classifier + hybrid search fusion
│   ├── reranker.py             # Cross-encoder reranking (sigmoid normalized)
│   ├── compressor.py           # Semantic context compression
│   ├── generator.py            # LLM answer generation (streaming)
│   ├── verifier.py             # Hallucination detection via structured output
│   ├── evaluator.py            # Latency, cost, faithfulness metrics
│   ├── memory.py               # Query Decontextualizer (multi-turn memory)
│   ├── feedback.py             # SQLite feedback manager + dashboard stats
│   └── orchestrator.py         # Master pipeline orchestrator
│
├── data/                       # Place your .txt documents here
│   ├── remote_work_policy.txt
│   └── it_security_policy.txt
│
├── storage/                    # Auto-generated indexes (gitignored)
│   ├── vector_store/           # FAISS index
│   ├── bm25_corpus.json        # BM25 sparse index
│   └── feedback.db             # SQLite telemetry database
│
├── .env                        # API keys (gitignored)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/enterprise-rag.git
cd enterprise-rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Add Your Documents
Place `.txt` files in the `data/` folder. Sample files are included.

### 6. Run the Application
```bash
python -m streamlit run main.py
```

### 7. Index Your Documents
In the sidebar, click **"🔄 Run Data Ingestion"** to build the FAISS + BM25 indexes.

---

## 💬 Usage

### Copilot Chat Tab
Ask questions about your documents. The AI will:
- Stream the answer in real-time
- Cite the exact document chunk used
- Show a ⚠️ Guardrail Alert if the verifier detects unsupported claims

### Admin Dashboard Tab
Monitor system health including:
- **Total Queries Processed**
- **Hallucination Rate** (% of answers flagged by the verifier)
- **Avg Retrieval Score** (normalized 0–1 cross-encoder score)
- **AI Auto-Tuning Recommendation** (suggests chunk size adjustments)
- **Flagged Query Log** (all queries that triggered guardrails)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM** | Groq API (Llama 3.1 8B Instant) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS |
| **Sparse Search** | BM25 (rank-bm25) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Orchestration** | LangChain |
| **Database** | SQLite |
| **Language** | Python 3.10+ |

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list. Key dependencies:
```
langchain
langchain-groq
langchain-huggingface
langchain-community
faiss-cpu
rank-bm25
sentence-transformers
streamlit
python-dotenv
tiktoken
pandas
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)
