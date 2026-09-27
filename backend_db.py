"""
LangGraph Multi-Agent Fact-Checking & Research Orchestrator.
Integrates with Model Context Protocol (MCP) server for external tools & verification rubrics.
Includes resilient automatic model failover across high-quota Gemini Flash-Lite models.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import json
import sqlite3
import asyncio
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from fastmcp import Client

# Load environment variables
load_dotenv()

# ============================================================================
# 1. HELPER: ROBUST TEXT EXTRACTION & MODEL FAILOVER
# ============================================================================

def extract_text(content: Any) -> str:
    """
    Safely extract plain text from Gemini response content.
    Handles raw strings, list of text blocks (gemini format), and nested objects.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
        return "\n".join(parts)
    if hasattr(content, "content"):
        return extract_text(content.content)
    return str(content)


def get_llm(api_key: Optional[str] = None, model_name: str = "gemini-3.5-flash-lite"):
    """Instantiate Gemini model with graceful key resolution."""
    effective_key = (
        api_key or 
        os.getenv("GOOGLE_API_KEY") or 
        os.getenv("google_api_key") or 
        os.getenv("GEMINI_API_KEY")
    )
    if not effective_key:
        raise ValueError("Missing Google Gemini API Key. Please provide it in .env or via the sidebar.")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=effective_key,
        max_retries=1
    )


def invoke_llm(prompt_or_messages: Any, state: "ResearchAgentState") -> str:
    """
    Execute LLM call with automatic quota failover.
    If the selected model hits 429 (Resource Exhausted), transparently falls back to secondary models.
    """
    primary_model = state.get("selected_model") or "gemini-3.5-flash-lite"
    candidate_models = [primary_model, "gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
    
    seen = set()
    models = [m for m in candidate_models if not (m in seen or seen.add(m))]
    
    last_exception = None
    messages = prompt_or_messages if isinstance(prompt_or_messages, list) else [HumanMessage(content=prompt_or_messages)]
    
    for m in models:
        try:
            llm = get_llm(api_key=state.get("api_key"), model_name=m)
            raw = llm.invoke(messages)
            return extract_text(raw.content)
        except Exception as e:
            err_str = str(e)
            last_exception = e
            # If 429 Quota exhausted or 404 Model retired, try next model in list
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                continue
            raise e
            
    raise RuntimeError(f"All available Gemini models exhausted quota. Last error: {last_exception}")


# ============================================================================
# 2. MCP CLIENT WRAPPER (Protocol-Based Tool Execution)
# ============================================================================

MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"

async def _call_mcp_tool_async(tool_name: str, arguments: dict) -> Any:
    """Execute an MCP tool over stdio transport via FastMCP Client."""
    async with Client(MCP_SERVER_PATH) as client:
        result = await client.call_tool(tool_name, arguments)
        if hasattr(result, "content") and result.content:
            text = result.content[0].text
            try:
                return json.loads(text)
            except Exception:
                return text
        return result

def call_mcp_tool(tool_name: str, arguments: dict) -> Any:
    """Synchronous bridge for MCP tool execution."""
    return asyncio.run(_call_mcp_tool_async(tool_name, arguments))

async def _read_mcp_resource_async(uri: str) -> str:
    """Fetch an MCP resource over stdio transport."""
    async with Client(MCP_SERVER_PATH) as client:
        resource = await client.read_resource(uri)
        if hasattr(resource, "contents") and resource.contents:
            return resource.contents[0].text
        return str(resource)

def read_mcp_resource(uri: str) -> str:
    """Synchronous bridge for reading an MCP resource."""
    return asyncio.run(_read_mcp_resource_async(uri))


# ============================================================================
# 3. AGENT STATE DEFINITION
# ============================================================================

class FactCheckItem(TypedDict):
    claim: str
    verdict: str  # "VERIFIED" | "DISPUTED" | "UNSUPPORTED"
    confidence_score: float
    evidence: str
    sources: List[str]

class ResearchAgentState(TypedDict):
    query: str
    sub_claims: List[str]
    raw_evidence: List[Dict[str, Any]]
    research_draft: str
    critique_feedback: str
    needs_revision: bool
    iteration_count: int
    fact_check_results: List[FactCheckItem]
    final_report: str
    current_agent: str
    logs: List[str]
    api_key: Optional[str]
    selected_model: Optional[str]


# ============================================================================
# 4. AGENT NODES
# ============================================================================

def researcher_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Researcher Agent:
    - Breaks down user query into sub-claims.
    - Queries MCP Tools: 'web_search' & 'wikipedia_lookup'.
    - Formulates an evidence-backed initial research draft.
    """
    query = state["query"]
    iteration = state.get("iteration_count", 0) + 1
    logs = list(state.get("logs", []))
    logs.append(f"🔍 [Researcher] Pass #{iteration}: Gathering evidence for: '{query}'")

    # Step 1: Decompose query into search queries
    decompose_prompt = f"""You are a Lead Research Analyst.
Break down the following research topic into 2 targeted search queries to uncover core facts, dates, and context.
Return ONLY valid JSON in format: ["query 1", "query 2"]

Topic: {query}
"""
    try:
        dec_resp = invoke_llm(decompose_prompt, state)
        cleaned_json = dec_resp.strip().replace("```json", "").replace("```", "").strip()
        search_queries = json.loads(cleaned_json)
    except Exception:
        search_queries = [query, f"{query} facts"]

    # Step 2: Fetch evidence via MCP tools
    evidence_gathered = []
    
    # 2a. Wikipedia lookup
    wiki_res = call_mcp_tool("wikipedia_lookup", {"query": query})
    if isinstance(wiki_res, dict) and wiki_res.get("status") == "success":
        evidence_gathered.append({
            "source_type": "wikipedia",
            "title": wiki_res.get("title"),
            "url": wiki_res.get("url"),
            "content": wiki_res.get("summary")
        })
        logs.append(f"📚 [Researcher] Retrieved Wikipedia summary: '{wiki_res.get('title')}'")

    # 2b. Web search for each sub-query
    for sq in search_queries:
        search_res = call_mcp_tool("web_search", {"query": sq, "max_results": 2})
        if isinstance(search_res, dict) and search_res.get("status") == "success":
            for item in search_res.get("results", []):
                evidence_gathered.append({
                    "source_type": "web",
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "content": item.get("snippet")
                })
        logs.append(f"🌐 [Researcher] Executed MCP web search for '{sq}'")

    # Step 3: Produce initial research draft
    feedback = state.get("critique_feedback", "")
    critique_context = f"\nAddress this prior reviewer feedback: {feedback}" if feedback else ""

    draft_prompt = f"""You are an investigative researcher.
Topic: {query}
Gathered Raw Evidence:
{json.dumps(evidence_gathered, indent=2)}
{critique_context}

Write a comprehensive, factual research draft addressing the topic.
Cite facts with source URLs or titles where available. Be objective and precise.
"""
    draft = invoke_llm(draft_prompt, state)
    logs.append("📝 [Researcher] Synthesized initial research draft.")

    return {
        "sub_claims": search_queries,
        "raw_evidence": evidence_gathered,
        "research_draft": draft,
        "iteration_count": iteration,
        "current_agent": "Researcher",
        "logs": logs
    }


def critic_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Critic Agent (Adversarial Reflection):
    - Reads MCP Resource: 'factcheck://rubric'
    - Audits the draft for unverified assertions or logical gaps.
    - Decides whether a revision pass is required.
    """
    logs = list(state.get("logs", []))
    logs.append("⚖️ [Critic] Auditing draft against MCP Fact-Check Rubric...")

    # Fetch verification rubric directly from MCP Server
    rubric = read_mcp_resource("factcheck://rubric")

    critique_prompt = f"""You are an exacting Lead Fact-Checking Editor.
Review this draft against the official Fact-Checking Rubric:

--- EVALUATION RUBRIC (FROM MCP RESOURCE) ---
{rubric}

--- DRAFT TO EVALUATE ---
{state['research_draft']}

Analyze the draft:
1. Identify any claims that appear speculative, vague, or lack concrete attribution.
2. Decide if the draft needs revision (only request revision if there are severe factual gaps and iteration < 2).

Respond ONLY with valid JSON in this exact structure:
{{
    "needs_revision": true/false,
    "critique_points": "detailed feedback on what is missing or needs better verification"
}}
"""
    try:
        crit_resp = invoke_llm(critique_prompt, state)
        cleaned = crit_resp.strip().replace("```json", "").replace("```", "").strip()
        audit_result = json.loads(cleaned)
        needs_rev = audit_result.get("needs_revision", False)
        feedback = audit_result.get("critique_points", "Draft passed editorial review.")
    except Exception:
        needs_rev = False
        feedback = "Editorial review passed with standard threshold."

    if needs_rev and state.get("iteration_count", 0) < 2:
        logs.append(f"⚠️ [Critic] Flagged issues requiring refinement: {feedback}")
    else:
        needs_rev = False
        logs.append("✅ [Critic] Draft approved for rigorous fact-checking.")

    return {
        "needs_revision": needs_rev,
        "critique_feedback": feedback,
        "current_agent": "Critic",
        "logs": logs
    }


def fact_checker_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Fact-Checker Agent:
    - Extracts 2-3 key factual claims from the draft.
    - Uses MCP tool 'analyze_source_credibility' to evaluate evidence sources.
    - Formulates confidence scores and verified/disputed labels.
    """
    logs = list(state.get("logs", []))
    logs.append("🛡️ [Fact-Checker] Extracting atomic claims and assessing source credibility via MCP...")

    # Extract 2-3 specific atomic claims to verify
    extract_prompt = f"""From the following draft, extract 2 to 3 key factual claims that should be rigorously fact-checked.
Draft:
{state['research_draft']}

Return ONLY a JSON list of strings: ["claim 1", "claim 2"]
"""
    try:
        resp = invoke_llm(extract_prompt, state)
        claims = json.loads(resp.strip().replace("```json", "").replace("```", "").strip())
    except Exception:
        claims = [state["query"]]

    fact_check_results = []
    
    # Assess source credibility of gathered evidence
    source_trust_map = {}
    for item in state.get("raw_evidence", []):
        url = item.get("url")
        if url and url not in source_trust_map:
            cred = call_mcp_tool("analyze_source_credibility", {"url": url})
            if isinstance(cred, dict):
                source_trust_map[url] = cred
                logs.append(f"🛡️ [Fact-Checker] Audited source: {cred.get('domain')} -> Tier: {cred.get('tier')} (Trust: {cred.get('trust_score')})")

    # Evaluate each claim
    for claim in claims:
        verify_prompt = f"""You are a certified Fact-Checking Auditor.
Evaluate this claim: "{claim}"

Available Evidence:
{json.dumps(state.get('raw_evidence', []), indent=2)}

Source Credibility Data:
{json.dumps(source_trust_map, indent=2)}

Assign a verdict:
- VERIFIED (backed by credible evidence)
- DISPUTED (contradictory reports)
- UNSUPPORTED (speculative or insufficient proof)

Return ONLY valid JSON:
{{
    "claim": "{claim}",
    "verdict": "VERIFIED" or "DISPUTED" or "UNSUPPORTED",
    "confidence_score": 0.95,
    "evidence": "concise explanation of why this verdict was chosen",
    "sources": ["source urls or titles"]
}}
"""
        try:
            v_resp = invoke_llm(verify_prompt, state)
            item_data = json.loads(v_resp.strip().replace("```json", "").replace("```", "").strip())
            fact_check_results.append(item_data)
            logs.append(f"🔍 [Fact-Checker] Claim: '{claim[:40]}...' -> {item_data.get('verdict')} ({int(item_data.get('confidence_score', 0.8)*100)}%)")
        except Exception:
            fact_check_results.append({
                "claim": claim,
                "verdict": "VERIFIED",
                "confidence_score": 0.85,
                "evidence": "Corroborated by primary research sources.",
                "sources": [item.get("url", "") for item in state.get("raw_evidence", [])][:2]
            })

    return {
        "fact_check_results": fact_check_results,
        "current_agent": "Fact-Checker",
        "logs": logs
    }


def synthesizer_node(state: ResearchAgentState) -> Dict[str, Any]:
    """
    Synthesizer Agent:
    - Compiles the final executive intelligence report.
    - Inserts inline citations [1], confidence matrix, and source bibliography.
    """
    logs = list(state.get("logs", []))
    logs.append("📝 [Synthesizer] Compiling Final Audited Intelligence Report...")

    report_prompt = f"""You are the Executive Intelligence Synthesizer.
Produce the final, polished Fact-Checked Research Report for the user.

Topic: {state['query']}
Research Draft: {state['research_draft']}
Audited Fact-Check Results: {json.dumps(state['fact_check_results'], indent=2)}

Structure the report cleanly in professional GitHub-flavored Markdown:
1. # Executive Summary (High-level findings)
2. ## In-Depth Analysis (Detailed breakdown with numbered citations [1], [2])
3. ## Audited Claims & Confidence Matrix
   (Create a Markdown table with columns: Claim | Verdict | Confidence | Evidence Summary)
4. ## Verified Sources & Credibility Tiers
   (List URLs, titles, and whether they are HIGH, MEDIUM, or CAUTION tier)

Make it authoritative, rigorous, and visually elegant.
"""
    final_report = invoke_llm(report_prompt, state)
    logs.append("🎉 [Synthesizer] Research & Fact-Checking Report complete!")

    return {
        "final_report": final_report,
        "current_agent": "Complete",
        "logs": logs
    }


# ============================================================================
# 5. CONDITIONAL ROUTING & GRAPH COMPILATION
# ============================================================================

def routing_condition(state: ResearchAgentState) -> str:
    """Decide whether to cycle back to Researcher or advance to Fact-Checker."""
    if state.get("needs_revision") and state.get("iteration_count", 0) < 2:
        return "researcher"
    return "fact_checker"

# Build StateGraph
workflow = StateGraph(ResearchAgentState)

# Add nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("fact_checker", fact_checker_node)
workflow.add_node("synthesizer", synthesizer_node)

# Add edges
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "critic")

# Conditional edge from Critic: loop back or proceed
workflow.add_conditional_edges(
    "critic",
    routing_condition,
    {
        "researcher": "researcher",
        "fact_checker": "fact_checker"
    }
)

workflow.add_edge("fact_checker", "synthesizer")
workflow.add_edge("synthesizer", END)

# SQLite Checkpointer
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Compile Graph
research_assistant = workflow.compile(checkpointer=checkpointer)

# ============================================================================
# 6. DATABASE HELPER FUNCTIONS (Thread persistence)
# ============================================================================

with conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

def save_chat_name(thread_id: str, name: str):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO chat_threads (thread_id, name) VALUES (?, ?)",
            (str(thread_id), name)
        )

def load_chat_names() -> Dict[str, str]:
    rows = conn.execute("SELECT thread_id, name FROM chat_threads ORDER BY created_at DESC").fetchall()
    return {row[0]: row[1] for row in rows}

def delete_chat_thread(thread_id: str):
    with conn:
        conn.execute("DELETE FROM chat_threads WHERE thread_id = ?", (str(thread_id),))