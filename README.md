# ⚡ Agentic AI Chatbot

An end-to-end, production-grade **Agentic AI Chatbot** powered by **LangGraph**, **Groq (`openai/gpt-oss-120b`)**, and **Streamlit**. Designed with modular stateful workflows, multi-tool orchestration, thread-isolated PDF RAG, human-in-the-loop (HITL) safety approvals, persistent SQLite memory, and multi-user session isolation.

---

## 🌟 Key Features

- **🧠 Agentic Multi-Tool Routing**: Dynamically decides when to answer directly or delegate queries to specialized tools.
- **📄 Thread-Isolated PDF RAG**: Upload resumes, documents, or reports via the chat UI. Indexed with FAISS and Google Gemini embeddings (`gemini-embedding-001`), completely isolated by conversation thread to prevent cross-chat data leakage.
- **🛡️ Human-in-the-Loop (HITL) Approval**: Safe action gating using LangGraph `interrupt()`. Actions such as stock purchases pause execution until explicit human confirmation is received from the UI.
- **👥 Multi-User Session Isolation**: Browser sessions are isolated via URL session keys (`?session=...`), ensuring visitors on shared or deployed instances only access their own conversation history.
- **💾 Persistent Checkpoint Memory**: Built-in conversation memory backed by `SqliteSaver` (`chatbot.db`), allowing users to switch between conversations seamlessly.
- **⚡ High-Speed Inference**: Powered by Groq LPU inference using `openai/gpt-oss-120b` for token streaming.
- **🔍 Real-Time External Capabilities**:
  - **Web Search**: Real-time information discovery via **Tavily Search**.
  - **Financial Data**: Live stock quotes via **Alpha Vantage**.
  - **Weather Reports**: Real-time conditions via **OpenWeatherMap**.
  - **Mathematical Reasoning**: Safe sandboxed expression evaluation with Python `math`.
- **🎨 Modern Dark UI**: ChatGPT-style sleek interface featuring dynamic thread naming, document attachment chips, action cards, and responsive sidebar navigation.
- **📊 Observability & Tracing**: Full execution traces and state graphs logged with **LangSmith**.
- **🐳 Docker & Cloud Deployment Ready**: Includes production-ready `DockerFile`, `.dockerignore`, and deployment configurations for **Render** and **AWS**.

---

## 🏗️ Architecture & Workflow

```mermaid
flowchart TD
    User([👤 User Prompt / PDF Upload]) --> UI[🎨 Streamlit Frontend]
    UI --> LangGraph[⚙️ LangGraph StateGraph]
    
    subgraph Core Agent Engine
        LangGraph --> Checkpointer[(💾 SQLite Checkpointer)]
        LangGraph --> ChatNode[🧠 Chat Node: Groq gpt-oss-120b]
        ChatNode --> ToolCondition{Tool Required?}
        
        ToolCondition -- No --> Response[💬 Stream Response to UI]
        ToolCondition -- Yes --> ToolDispatcher[🛠️ Tool Node]
    end

    subgraph Integrated Tools
        ToolDispatcher --> RAG[📄 PDF RAG: FAISS + Google Embeddings]
        ToolDispatcher --> Tavily[🌐 Web Search: Tavily API]
        ToolDispatcher --> Calc[🧮 Safe Math Calculator]
        ToolDispatcher --> Stocks[📈 Stock Quotes: Alpha Vantage]
        ToolDispatcher --> Weather[🌤️ Weather: OpenWeatherMap]
        ToolDispatcher --> HITL[🛡️ Purchase Stock: HITL Interrupt]
    end

    HITL --> Decision{Human Approval?}
    Decision -- Yes / No --> ToolDispatcher
    Integrated Tools --> ChatNode
```

---

## 🛠️ Tool Ecosystem

| Tool | Functionality | Backing Service / Technology |
| :--- | :--- | :--- |
| **`rag_tool`** | Ingests and semantically searches uploaded PDFs (resumes, documents) | `PyPDFLoader`, `FAISS`, `GoogleGenerativeAIEmbeddings` |
| **`search_tool`** | Performs deep, up-to-date web searches for news and facts | Tavily Search API (`langchain-tavily`) |
| **`calculator`** | Solves mathematical and statistical expressions | Python `math` module (restricted evaluation) |
| **`get_stock_price`**| Fetches real-time market data and quotes for tickers | Alpha Vantage API |
| **`purchase_stock`**| Simulates stock purchases with Human-in-the-Loop gating | LangGraph `interrupt()` |
| **`get_current_weather`** | Fetches live weather conditions for cities globally | OpenWeatherMap Geocoding & Weather APIs |

---

## 📁 Repository Structure

```text
agentic-chatbot/
├── app.py                  # Streamlit frontend with modern UI & session routing
├── backend.py              # LangGraph workflow, tools, checkpointer & nodes
├── DockerFile              # Production Dockerfile (Python 3.11-slim + FAISS support)
├── requirements.txt        # Pinned dependencies
├── .python-version         # Enforces Python 3.11 runtime on cloud hosts
├── .dockerignore           # Excludes local databases, caches, and secrets
├── .gitignore              # Ignores virtualenvs, database files, and .env
├── .env.example            # Sample environment variables template
└── README.md               # Project documentation
```

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Populate the keys:

```ini
# LLM Providers
GROQ_API_KEY="your-groq-api-key"
OPENAI_API_KEY="your-openai-api-key"               # Optional fallback
GOOGLE_API_KEY="your-google-api-key"               # Required for RAG embeddings

# External Search & API Services
TAVILY_API_KEY="your-tavily-api-key"               # Required for Web Search
OPENWEATHER_API_KEY="your-openweather-api-key"     # Required for Weather tool

# LangSmith Tracing & Observability (Optional)
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="agentic-chatbot"
```

---

## 🚀 Quickstart (Local Installation)

### 1. Clone the Repository
```bash
git clone https://github.com/its-vikash-Kushwaha/agentic-chatbot.git
cd agentic-chatbot
```

### 2. Set Up a Virtual Environment (Python 3.11 recommended)
```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 Docker Setup

### 1. Build the Docker Image
```bash
docker build -t agentic-chatbot:latest .
```

### 2. Run the Container
```bash
docker run -d \
  -p 8501:8501 \
  --env-file .env \
  --name agentic-chatbot \
  agentic-chatbot:latest
```
Visit `http://localhost:8501` to use the chatbot.

---

## ☁️ Deployment Guide (Render)

1. **Create Web Service**: Connect your GitHub repository on [Render](https://render.com/).
2. **Environment**: Select **Python** or **Docker**.
3. **Build & Start Commands** (if using Native Python):
   - **Python Version**: Ensure `.python-version` is committed (`3.11.9`).
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```
4. **Environment Variables**: Add all variables defined in your `.env` under Render's **Environment** tab.

---

## 🔒 Multi-User Privacy & Security

- **Session Isolation**: Every user accessing the web application receives an isolated session key (`?session=<uuid>`). Checkpoint threads and conversation histories are strictly filtered by this identifier.
- **RAG Data Isolation**: Uploaded documents and FAISS indexes are partitioned into subdirectories by thread ID (`faiss_db/{thread_id}/`), guaranteeing that uploaded files cannot be accessed or queried across different chat sessions.
- **Sandboxed Operations**: Calculator evaluations are strictly isolated from system built-ins to prevent arbitrary code execution.

---

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).
