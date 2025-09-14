from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()


# ======================= Model =======================
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    max_tokens=1024
)


# ======================= Tools =======================
# search
search_tool = DuckDuckGoSearchRun(region="us-en")

# calculator
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

# stock price 
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the current stock price for a given ticker symbol using Alpha Vantage.
    """
    try:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return {"error": "Missing ALPHAVANTAGE_API_KEY in environment variables"}

        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        )
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


# Make tool list
tool_list = [search_tool, calculator, get_stock_price]

# Make the LLM tool-aware
model_with_tools = model.bind_tools(tool_list)


# ======================= State =======================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ======================= Graph Nodes =======================
def chat_node(state: ChatState):
    # take user query from state
    messages = state['messages']

    # send to llm
    response = model_with_tools.invoke(messages)

    # response store state
    return {'messages': [response]}

tool_node = ToolNode(tool_list)


# ======================= Database Conn =======================
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpointer = SqliteSaver(conn= conn)


# ======================= Graph =======================
graph = StateGraph(ChatState)

# add node
graph.add_node('chat_node', chat_node)
graph.add_node("tools", tool_node)

# add edges
graph.add_edge(START, 'chat_node')
# If the LLM asked for a tool, go to the tool node; else finish
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer= checkpointer)



with conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_threads (
            thread_id TEXT PRIMARY KEY,
            name TEXT
        )
    """)

def save_chat_name(thread_id, name):
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO chat_threads (thread_id, name) VALUES (?, ?)",
            (str(thread_id), name)
        )

def load_chat_names():
    rows = conn.execute("SELECT thread_id, name FROM chat_threads").fetchall()
    return {row[0]: row[1] for row in rows}