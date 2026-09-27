# FactGraph AI: Autonomous Multi-Agent Research & Fact-Checking Engine
### Built with LangGraph Cyclical Orchestration & Model Context Protocol (MCP)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF4B4B?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Model Context Protocol](https://img.shields.io/badge/Protocol-MCP%20FastMCP-000000?style=for-the-badge)](https://modelcontextprotocol.io)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**FactGraph AI** is a decoupled multi-agent intelligence system designed to autonomously investigate complex topics, claims, and assertions. It combines **LangGraph's cyclical graph execution** with Anthropic's **Model Context Protocol (MCP)**, enforcing an adversarial **critic loop** and multi-source cross-verification before publishing factual reports.

---




## 🌟 Key Features

1. **Model Context Protocol (MCP) Tool-Serving Layer**:
   - Tools are **not hardcoded** inside the agent. Instead, they are exposed over standard MCP (`stdio` transport) using the official `fastmcp` SDK.
   - Includes **MCP Tools** (`web_search`, `wikipedia_lookup`, `analyze_source_credibility`), an **MCP Resource** (`factcheck://rubric`), and an **MCP Prompt** (`decompose_query_prompt`).
2. **Adversarial Reflection & Cyclical Loop**:
   - The **Critic Agent** reads the verification rubric served directly over MCP and evaluates the Researcher's draft. If claims are speculative or lack attribution, the graph loops back with targeted directives.
3. **Structured State Machine**:
   - Unlike basic chat scripts that only track `messages: list`, the graph tracks an explicit `ResearchAgentState` containing atomic claims, raw evidence, editorial feedback, and per-claim confidence scores.
4. **Source Credibility Heuristics**:
   - Audits sources into **HIGH** (peer-reviewed, government, top news wires), **MEDIUM** (established tech/media), or **CAUTION** tiers.
5. **Persistent State & Resumable Sessions**:
   - Powered by `SqliteSaver` in `chatbot.db`, allowing users to resume past research threads or delete completed investigations.
6. **100% Free & Zero-Cost**:
   - No paid APIs required. Powered by Google Gemini Flash free tier, DuckDuckGo, and Wikipedia REST.

---

## 📂 Repository Structure

```
├── backend_db.py      # LangGraph Multi-Agent Orchestrator & State Machine
├── mcp_server.py      # FastMCP Server (Tools, Resources, Prompts)
├── frontend_db.py     # Streamlit UI with Live Pipeline Tracker
├── test_mcp.py        # Automated test suite for MCP Server over stdio
├── chatbot.db         # SQLite database (LangGraph checkpoints & thread titles)
├── requirements.txt   # Project dependencies
└── README.md          # System documentation & interview guide
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/FactGraph-AI.git
cd FactGraph-AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_free_google_ai_studio_api_key_here
```
*(Alternatively, you can input your API key directly in the Streamlit sidebar at runtime).*

### 3. Verify the MCP Server

Run the automated MCP test suite to confirm the server and tool transports are functioning:
```bash
python test_mcp.py
```

### 4. Launch the Application

```bash
streamlit run frontend_db.py
```



## 📄 License

Distributed under the MIT License.
