"""
Streamlit Frontend for FactGraph AI: Autonomous Multi-Agent Research & Fact-Checking Assistant.
Built with LangGraph cyclical orchestration and Model Context Protocol (MCP).
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import streamlit as st
import os
import uuid
from dotenv import load_dotenv

# Load backend
from backend_db import (
    research_assistant,
    save_chat_name,
    load_chat_names,
    delete_chat_thread
)

load_dotenv()

# ============================================================================
# Page Configuration & Styling
# ============================================================================
st.set_page_config(
    page_title="FactGraph AI | Multi-Agent Fact-Checking",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive polish
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-verified {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #34d399;
    }
    .badge-disputed {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #f59e0b;
    }
    .badge-unsupported {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
    }
    .agent-step-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = load_chat_names()

if "current_report" not in st.session_state:
    st.session_state["current_report"] = None

if "current_logs" not in st.session_state:
    st.session_state["current_logs"] = []

if "current_fact_checks" not in st.session_state:
    st.session_state["current_fact_checks"] = []


# ============================================================================
# Sidebar Configuration
# ============================================================================
with st.sidebar:
    st.title("🛡️ FactGraph AI")
    st.caption("Autonomous Multi-Agent Fact-Checking Engine")
    
    st.markdown("---")
    
    # API Key Configuration
    st.subheader("🔑 Credentials & Model")
    existing_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key") or ""
    user_api_key = st.text_input(
        "Google Gemini API Key",
        value=existing_key,
        type="password",
        help="Required for Gemini Flash. Get your free key at aistudio.google.com"
    )
    
    selected_model = st.selectbox(
        "Primary Gemini Model",
        options=[
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash"
        ],
        index=0,
        help="Flash-Lite models provide the highest free-tier quota (15+ RPM) and auto-failover."
    )
    
    # System Architecture Badges
    st.markdown("---")
    st.markdown("**System Architecture:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("🧠 **LangGraph**\nCyclic Agent Graph")
    with col_b:
        st.success("🔌 **MCP Server**\nStdio Transport")

    st.markdown("---")
    
    # Thread Controls
    if st.button("➕ New Research Thread", use_container_width=True):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["current_report"] = None
        st.session_state["current_logs"] = []
        st.session_state["current_fact_checks"] = []
        st.rerun()

    st.subheader("📚 Saved Investigations")
    threads = load_chat_names()
    for t_id, name in threads.items():
        col_t1, col_t2 = st.columns([0.8, 0.2])
        with col_t1:
            if st.button(f"📄 {name[:24]}...", key=f"sel_{t_id}", use_container_width=True):
                st.session_state["thread_id"] = t_id
                # Load saved checkpoint
                config = {"configurable": {"thread_id": t_id}}
                state = research_assistant.get_state(config)
                vals = state.values if state else {}
                st.session_state["current_report"] = vals.get("final_report")
                st.session_state["current_logs"] = vals.get("logs", [])
                st.session_state["current_fact_checks"] = vals.get("fact_check_results", [])
                st.rerun()
        with col_t2:
            if st.button("🗑️", key=f"del_{t_id}"):
                delete_chat_thread(t_id)
                if st.session_state["thread_id"] == t_id:
                    st.session_state["thread_id"] = str(uuid.uuid4())
                    st.session_state["current_report"] = None
                    st.session_state["current_logs"] = []
                st.rerun()


# ============================================================================
# Main Workspace Header
# ============================================================================
st.title("Autonomous Research & Fact-Checking Engine")
st.markdown("""
Investigate complex claims with a **collaborative multi-agent swarm**. 
An adversarial **Critic Agent** audits research drafts against an official **MCP Verification Rubric**, 
looping back for evidence until claims meet rigorous journalistic and scientific standards.
""")

# Visual Multi-Agent Architecture Pipeline
st.markdown("""
```
[1. Researcher Agent]  ──►  [2. Adversarial Critic]  ──►  [3. Fact-Checker]  ──►  [4. Final Synthesizer]
        ▲                             │ (If weak claims)
        └─────────────────────────────┘
```
""")

# ============================================================================
# Input Form
# ============================================================================
with st.form("research_form"):
    user_query = st.text_area(
        "Enter a topic, assertion, or question to investigate:",
        placeholder="e.g., Did the James Webb Space Telescope disprove the Big Bang theory?",
        height=90
    )
    submit_btn = st.form_submit_button("🚀 Launch Autonomous Fact-Check", use_container_width=True)

# Sample Starters
st.caption("💡 Sample queries: 'Is intermittent fasting scientifically proven to extend lifespan?' | 'Did quantum computers break RSA encryption?' | 'Origin of the Apollo 11 moon landing flags'")

# ============================================================================
# Agent Pipeline Execution
# ============================================================================
if submit_btn and user_query.strip():
    if not user_api_key.strip():
        st.error("⚠️ Please enter a Google Gemini API Key in the left sidebar to run the multi-agent system.")
        st.stop()

    active_thread = st.session_state["thread_id"]
    save_chat_name(active_thread, user_query[:35])

    config = {
        "configurable": {"thread_id": active_thread}
    }

    initial_state = {
        "query": user_query,
        "sub_claims": [],
        "raw_evidence": [],
        "research_draft": "",
        "critique_feedback": "",
        "needs_revision": False,
        "iteration_count": 0,
        "fact_check_results": [],
        "final_report": "",
        "current_agent": "Initializing",
        "logs": [],
        "api_key": user_api_key.strip(),
        "selected_model": selected_model
    }

    # Visual Status Container
    with st.status("🤖 Multi-Agent Swarm Initializing...", expanded=True) as status_box:
        step_container = st.empty()
        
        try:
            # Stream execution across graph nodes
            for event in research_assistant.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    current_agent = node_output.get("current_agent", node_name)
                    logs = node_output.get("logs", [])
                    
                    if current_agent == "Researcher":
                        status_box.update(label="🔍 Researcher Agent: Gathering external evidence & Wikipedia records...", state="running")
                    elif current_agent == "Critic":
                        status_box.update(label="⚖️ Critic Agent: Reviewing draft against MCP Fact-Checking Rubric...", state="running")
                    elif current_agent == "Fact-Checker":
                        status_box.update(label="🛡️ Fact-Checker Agent: Auditing source credibility & assigning confidence scores...", state="running")
                    elif current_agent == "Complete":
                        status_box.update(label="📝 Final Synthesizer: Compiling verified report & confidence matrix...", state="complete", expanded=False)

                    # Update session state with latest updates
                    if "final_report" in node_output:
                        st.session_state["current_report"] = node_output["final_report"]
                    if "logs" in node_output:
                        st.session_state["current_logs"] = node_output["logs"]
                    if "fact_check_results" in node_output:
                        st.session_state["current_fact_checks"] = node_output["fact_check_results"]

        except Exception as e:
            status_box.update(label=f"❌ Error during multi-agent execution: {str(e)}", state="error")
            st.error(f"Execution Error: {e}")
            st.stop()


# ============================================================================
# Display Results Tabs
# ============================================================================
if st.session_state["current_report"]:
    st.markdown("---")
    tab_report, tab_matrix, tab_trace = st.tabs([
        "📄 Final Intelligence Report",
        "📊 Audited Claims Matrix",
        "🔍 Multi-Agent Execution Trace"
    ])

    with tab_report:
        st.markdown(st.session_state["current_report"])

    with tab_matrix:
        st.subheader("Audited Claims & Confidence Breakdown")
        fact_checks = st.session_state.get("current_fact_checks", [])
        if fact_checks:
            for item in fact_checks:
                verdict = item.get("verdict", "UNSUPPORTED")
                badge_class = (
                    "badge-verified" if verdict == "VERIFIED" 
                    else "badge-disputed" if verdict == "DISPUTED" 
                    else "badge-unsupported"
                )
                conf = int(item.get("confidence_score", 0.0) * 100)
                
                with st.expander(f"Claim: {item.get('claim')}", expanded=True):
                    st.markdown(f"""
                    **Verdict:** <span class="metric-badge {badge_class}">{verdict} ({conf}% Confidence)</span>
                    
                    **Evidence Analysis:**  
                    {item.get('evidence')}
                    
                    **Attributed Sources:**  
                    {', '.join(item.get('sources', [])) if item.get('sources') else 'No authoritative source cited.'}
                    """, unsafe_allow_html=True)
        else:
            st.info("No structured claims matrix available for this run.")

    with tab_trace:
        st.subheader("Multi-Agent Execution Log")
        st.caption("Inspect the exact sequence of agent thought processes and MCP tool invocations.")
        logs = st.session_state.get("current_logs", [])
        for log_entry in logs:
            st.markdown(f"`{log_entry}`")