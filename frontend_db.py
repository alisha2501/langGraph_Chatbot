import streamlit as st
from backend_db import chatbot, model, load_chat_names, save_chat_name
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# ======================= Utility functions ========================
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'][thread_id] = "New Chat"

def reset_chat():
    existing_new_chat = None
    for t_id, name in st.session_state['chat_threads'].items():
        if name == "New Chat":
            state = chatbot.get_state(config={'configurable': {'thread_id': t_id}})
            messages = state.values.get("messages", [])
            if not messages:  # completely unused
                existing_new_chat = t_id
                break

    if existing_new_chat:
        # Just switch back to the existing empty new chat
        st.session_state['thread_id'] = existing_new_chat
        st.session_state['message_history'] = []
    else:
        # or make a fresh one
        thread_id = generate_thread_id()
        st.session_state['thread_id'] = thread_id
        add_thread(thread_id)
        st.session_state['message_history'] = []

        
def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get("messages", [])
    return messages

# ======================= Session Setup =======================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = load_chat_names()

add_thread(st.session_state['thread_id'])

# *======================= Sidebar =======================
st.sidebar.title("LangGraph ChatBot")

if st.sidebar.button('Create New Chat'):
    reset_chat()

st.sidebar.header('My conversations')

for thread_id, name in reversed(list(st.session_state['chat_threads'].items())):
    if st.sidebar.button(name, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            elif isinstance(message, AIMessage):
                role = 'assistant'
            else:
                role = 'tool'
            temp_messages.append({'role': role, 'content': message.content})
        st.session_state['message_history'] = temp_messages


# ======================= Main UI =======================

# loading the conversation history
for message in st.session_state['message_history']:
    if message['role'] == 'tool' or not message['content']:
        continue
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    if st.session_state['chat_threads'][st.session_state['thread_id']] == "New Chat":
        summary_state = model.invoke(
            f"Summarize in max 4 words: {user_input}"
        ).content
        st.session_state['chat_threads'][st.session_state['thread_id']] = summary_state
        save_chat_name(st.session_state['thread_id'], summary_state)

    CONFIG = {
        'configurable': {'thread_id':st.session_state['thread_id']},
        "metadata": {
            "thread_id": st.session_state['thread_id']
        },
        "run_name": "chat_turn"
    }


    status_holder = {"box": None}

    def ai_only_stream():
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            
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

            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content

    with st.chat_message("assistant"):
        ai_message = st.write_stream(ai_only_stream())

    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished", state="complete", expanded=False
        )
    
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})