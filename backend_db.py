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
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash")

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_weather(city: str) -> dict:
    """_
    Provide weather report of the query city
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return {"error": "OpenWeather API key not found"}
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return {"error": f"Could not fetch weather data: {data.get('message', 'Unknown error')}"}
        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "description": data["weather"]["description"]
        }
    except Exception as e:
        return {"error": str(e)}


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
