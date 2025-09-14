# frontend.py

import streamlit as st
import uuid
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash")
# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

@st.cache_data # Cache the name generation to avoid re-running on every interaction
def generate_thread_name(user_input):
    prompt = f"Generate a short, 3-5 word title for a conversation that starts with this message: '{user_input}'"
    response = llm.invoke(prompt)
    return response.content.strip().strip('"')

# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
    st.session_state.thread_name = "New Chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar UI ---
st.sidebar.title("🤖 LangGraph Chatbot")
st.sidebar.markdown("This chatbot uses a separate frontend and backend.")

if st.sidebar.button("New Chat", use_container_width=True):
    st.session_state.messages = []
    st.session_state.thread_id = generate_thread_id()
    st.session_state.thread_name = "New Chat"
    st.rerun()

st.sidebar.header("Current Conversation")
st.sidebar.write(st.session_state.thread_name)

# --- Main Chat Interface UI ---
st.header(st.session_state.thread_name)

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Set thread name on the very first message
    if len(st.session_state.messages) == 1:
        st.session_state.thread_name = generate_thread_name(user_input)
        st.rerun()

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        events = chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )
        for event in events:
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage):
                    full_response = last_message.content
                    response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
