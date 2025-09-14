# backend.py

import requests
import wikipedia
from typing import TypedDict, Annotated

# --- LangChain & LangGraph Imports ---
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoints.sqlite import SqliteSaver

class BackendService:
    def __init__(self, secrets):
        """
        Initializes the backend services, tools, and the chatbot graph.
        'secrets' is the st.secrets object passed from the frontend.
        """
        self.secrets = secrets
        self.llm = self._initialize_llm()
        self.tools = self._initialize_tools()
        self.chatbot = self._compile_chatbot()

    def _initialize_llm(self):
        """Initializes the Google Generative AI model with an API key."""
        google_api_key = self.secrets.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in secrets.")
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    def _initialize_tools(self):
        """Initializes all the tools the chatbot can use."""
        
        @tool
        def get_weather(city: str) -> dict:
            """Provide weather report of the query city"""
            api_key = self.secrets.get("OPENWEATHER_API_KEY")
            if not api_key:
                return {"error": "OpenWeather API key not found"}
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            return response.json()

        @tool
        def get_stock_price(symbol: str) -> dict:
            """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')"""
            api_key = self.secrets.get("ALPHA_VANTAGE_API_KEY")
            if not api_key:
                return {"error": "Alpha Vantage API key not found"}
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
            response = requests.get(url)
            return response.json()

        @tool
        def wikipedia_search(query: str, sentences: int = 3) -> dict:
            """Search Wikipedia for a given query and return a summary."""
            try:
                return {"summary": wikipedia.summary(query, sentences=sentences)}
            except Exception as e:
                return {"error": str(e)}

        # Add other tools like calculator and DuckDuckGo search
        search_tool = DuckDuckGoSearchRun(region="us-en")
        
        # We can define the calculator tool directly as it needs no secrets
        @tool
        def calculator(first_num: float, second_num: float, operation: str) -> dict:
            """Perform a basic arithmetic operation."""
            # ... (calculator logic from your original code) ...
            if operation == "add": return {"result": first_num + second_num}
            if operation == "sub": return {"result": first_num - second_num}
            if operation == "mul": return {"result": first_num * first_num}
            if operation == "div": return {"result": first_num / second_num}
            return {"error": "Invalid operation"}


        return [search_tool, get_weather, get_stock_price, wikipedia_search, calculator]

    def _compile_chatbot(self):
        """Compiles and returns the LangGraph chatbot."""
        class ChatState(TypedDict):
            messages: Annotated[list[BaseMessage], add_messages]

        llm_with_tools = self.llm.bind_tools(self.tools)
        tool_node = ToolNode(self.tools)

        # WARNING: In-memory database is not persistent on Streamlit Cloud.
        memory = SqliteSaver.from_conn_string(":memory:")

        graph = StateGraph(ChatState)
        graph.add_node("chatbot", llm_with_tools)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "chatbot")
        graph.add_conditional_edges("chatbot", tools_condition)
        graph.add_edge("tools", "chatbot")
        
        return graph.compile(checkpointer=memory)
