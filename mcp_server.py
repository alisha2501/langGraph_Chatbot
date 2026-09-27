"""
MCP Server for Autonomous Research & Fact-Checking Assistant
Built with the Model Context Protocol (MCP) using FastMCP.
Exposes Tools, Resources, and Prompts to LangGraph agents.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from fastmcp import FastMCP
from urllib.parse import urlparse, quote
import json
import logging
import requests

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_server")

# Initialize FastMCP Server
mcp = FastMCP("Research-FactCheck-Server")

# ============================================================================
# 1. MCP TOOLS
# ============================================================================

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the live web for real-time information, articles, and authoritative sources.
    Returns a JSON string of results with title, link, and snippet.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("href", ""),
                    "snippet": item.get("body", "")
                })
        if not results:
            return json.dumps({"status": "no_results", "query": query, "results": []})
        return json.dumps({"status": "success", "query": query, "results": results}, indent=2)
    except Exception as e:
        logger.error(f"Error during web search for query '{query}': {e}")
        return json.dumps({"status": "error", "message": str(e), "results": []})


@mcp.tool()
def wikipedia_lookup(query: str) -> str:
    """
    Query Wikipedia REST API to retrieve factual encyclopedic summaries, background context, and official page links.
    Ideal for verified facts, historical events, organizations, and scientific concepts.
    """
    try:
        headers = {"User-Agent": "AutonomousResearchFactChecker/2.0 (contact@researcher.ai)"}
        
        # Step 1: Search Wikipedia for the best matching page title
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "opensearch",
            "search": query,
            "limit": 3,
            "namespace": 0,
            "format": "json"
        }
        search_resp = requests.get(search_url, params=search_params, headers=headers, timeout=8)
        search_data = search_resp.json()
        
        titles = search_data[1] if len(search_data) > 1 else []
        if not titles:
            return json.dumps({"status": "not_found", "query": query})
        
        best_title = titles[0]
        
        # Step 2: Fetch clean summary extract from REST API
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(best_title)}"
        sum_resp = requests.get(summary_url, headers=headers, timeout=8)
        
        if sum_resp.status_code == 200:
            sum_data = sum_resp.json()
            return json.dumps({
                "status": "success",
                "title": sum_data.get("title", best_title),
                "url": sum_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{best_title}"),
                "summary": sum_data.get("extract", "No extract available."),
                "related_titles": titles[1:]
            }, indent=2)
        else:
            return json.dumps({
                "status": "success",
                "title": best_title,
                "url": f"https://en.wikipedia.org/wiki/{best_title}",
                "summary": f"Page found on Wikipedia: {best_title}",
                "related_titles": titles[1:]
            }, indent=2)

    except Exception as e:
        logger.error(f"Error querying Wikipedia for '{query}': {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def analyze_source_credibility(url: str) -> str:
    """
    Analyze the domain credibility and reliability tier of a source URL.
    Classifies sources into HIGH, MEDIUM, or UNVERIFIED/CAUTION tiers based on domain heuristics.
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        # High-credibility benchmarks
        academic_gov = any(domain.endswith(ext) for ext in [".edu", ".gov", ".ac.uk", ".mil"])
        major_wire_academic = any(key in domain for key in [
            "reuters.com", "apnews.com", "bbc.com", "nature.com", "science.org", 
            "nih.gov", "cdc.gov", "who.int", "bloomberg.com", "wsj.com",
            "nytimes.com", "arxiv.org", "wikipedia.org", "economist.com"
        ])

        # Tech / Community / Commercial platforms
        medium_sources = any(key in domain for key in [
            "github.com", "medium.com", "forbes.com", "techcrunch.com", 
            "theverge.com", "wired.com", "cnet.com", "substack.com"
        ])

        if academic_gov or major_wire_academic:
            tier = "HIGH"
            reason = "Peer-reviewed, governmental, or globally recognized primary news wire."
            trust_score = 0.95
        elif medium_sources:
            tier = "MEDIUM"
            reason = "Established media outlet, tech publication, or curated community platform."
            trust_score = 0.70
        else:
            tier = "CAUTION"
            reason = "Unindexed domain, personal blog, or non-peer-reviewed source. Requires secondary cross-referencing."
            trust_score = 0.45

        return json.dumps({
            "domain": domain,
            "url": url,
            "tier": tier,
            "trust_score": trust_score,
            "reason": reason
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ============================================================================
# 2. MCP RESOURCES
# ============================================================================

@mcp.resource("factcheck://rubric")
def get_factcheck_rubric() -> str:
    """
    Standard Operating Procedure (SOP) & Evaluation Rubric for Fact-Checking Claims.
    Served as a contextual resource to Critic and Fact-Checker agents over MCP.
    """
    return """
# FACT-CHECKING & CLAIM EVALUATION RUBRIC (v2.0)

### 1. Classification Labels:
- **[VERIFIED]**: Supported by at least two independent credible sources, or one high-trust tier (e.g. .gov, .edu, peer-reviewed paper, major wire service like Reuters/AP).
- **[DISPUTED]**: Multiple reputable sources present contradictory findings or ongoing empirical debate.
- **[UNSUPPORTED]**: No authoritative documentation found; claim rests on rumor, conjecture, or speculation.

### 2. Evidence Hierarchy:
- **Tier 1 (High Trust, 0.9 - 1.0)**: Academic journals, official government/regulatory agencies (.gov/.edu), primary statistical datasets, major news wires (Reuters, AP).
- **Tier 2 (Medium Trust, 0.6 - 0.8)**: Established national news publications, reputable industry analysts, encyclopedias (Wikipedia).
- **Tier 3 (Caution, < 0.6)**: Unverified blogs, sponsored advertorials, self-published forum posts, anonymous social media accounts.

### 3. Verification Directives:
- An assertion is only as strong as its primary attribution.
- Distinguish between correlation and direct causation.
- Check date of citations to prevent citing obsolete data.
"""


# ============================================================================
# 3. MCP PROMPTS
# ============================================================================

@mcp.prompt()
def decompose_query_prompt(topic: str) -> str:
    """
    Standard prompt template for decomposing complex research questions into atomic verifiable claims.
    """
    return f"""You are an elite Investigative Fact-Checking Specialist.
Decompose the following user question/topic into 2-4 atomic, verifiable sub-claims:

Topic: {topic}

For each sub-claim:
1. Formulate a specific factual statement to confirm or refute.
2. Provide a targeted keyword search query to execute against external sources.

Return your analysis in clear, numbered points.
"""


# ============================================================================
# Server Entry Point
# ============================================================================
if __name__ == "__main__":
    # Runs over stdio transport by default (standard for MCP clients)
    mcp.run()
