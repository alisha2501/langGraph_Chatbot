import streamlit as st
from backend_db import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash")
# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def generate_thread_name(user_input=None):
    """
    Generate a short title (3–5 words) for a chat based on the first user message,
    using Google Gemini.
    """
    # user_text = conversation[0]["content"] if conversation else "General Chat"

    # response = llm.invoke([
    #     {"role": "system", "content": "You are a title generator. Create a short, 3-5 word title summarizing the chat."},
    #     {"role": "user", "content": user_text}
    # ])


    if user_input:
        # Clean and truncate the input to create a name
        response = llm.invoke([
            {"role": "system", "content": "You are a title generator. Create a short, 3-5 word title summarizing the chat."},
            {"role": "user", "content": user_input}
        ])
    return response.content.strip()


    #     cleaned_input = user_input.strip()
    #     if len(cleaned_input) > 30:
    #         return cleaned_input[:27] + "..."
    #     return cleaned_input
    # return "New Chat"

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["thread_names"][thread_id] = "New Chat"
    st.session_state["message_history"] = []

def add_thread(thread_id, name="New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    if thread_id not in st.session_state["thread_names"]:
        st.session_state["thread_names"][thread_id] = name

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = {}
    # Initialize with default names for existing threads
    for thread_id in st.session_state["chat_threads"]:
        st.session_state["thread_names"][thread_id] = "Chat Conversation"

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    thread_name = st.session_state["thread_names"].get(thread_id, "Chat Conversation")
    
    # Display thread name with a button to load it
    if st.sidebar.button(thread_name, key=f"btn_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Update thread name if this is the first message
    if len(st.session_state["message_history"]) == 0:
        thread_name = generate_thread_name(user_input)
        st.session_state["thread_names"][st.session_state["thread_id"]] = thread_name
    
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )