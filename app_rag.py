from agentic_rag_backend import chatbot, get_all_threads, ingest_rag_document
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
import tempfile
import uuid
from pathlib import Path

# Requires Streamlit 1.43+.
# Chat input documentation:
# https://docs.streamlit.io/develop/api-reference/chat/st.chat_input

st.title("Agentic Chatbot with LangGraph")


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


def get_thread_label(thread_id):
    messages = load_conversation(thread_id)

    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            if not isinstance(content, str):
                content = str(content)

            snippet = " ".join(content.split())
            if len(snippet) > 40:
                snippet = snippet[:37].rstrip() + "..."

            return snippet or "Empty conversation"

    return "New conversation"


def get_message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return str(content) if content is not None else ""


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = list(get_all_threads())

add_thread(st.session_state["thread_id"])

st.sidebar.title("My Conversations")

if st.sidebar.button("+ New Chat"):
    reset_chat()
    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(get_thread_label(thread_id), key=thread_id):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            content = get_message_text(message.content)
            if content:
                temp_messages.append(
                    {"role": role, "content": content}
                )

        st.session_state["message_history"] = temp_messages
        st.rerun()

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# A PDF can be attached directly inside the chat input.
# Accept one PDF because the supplied ingestion function replaces faiss_db.
chat_submission = st.chat_input(
    "Type a message or attach a PDF",
    accept_file=True,
    file_type=["pdf"],
)

if chat_submission:
    user_input = chat_submission.text.strip()
    uploaded_files = chat_submission.files

    display_parts = []
    if user_input:
        display_parts.append(user_input)

    for uploaded_file in uploaded_files:
        display_parts.append(f"📎 {uploaded_file.name}")

    display_message = "\n\n".join(display_parts)

    st.session_state["message_history"].append(
        {"role": "user", "content": display_message}
    )

    with st.chat_message("user"):
        st.markdown(display_message)

    # Save the uploaded bytes temporarily because the backend expects a path.
    if uploaded_files:
        uploaded_file = uploaded_files[0]

        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = Path(temp_dir) / "uploaded_document.pdf"
                    pdf_path.write_bytes(uploaded_file.getvalue())
                    ingest_rag_document(str(pdf_path))

        except Exception as exc:
            with st.chat_message("assistant"):
                st.error(f"Could not process the PDF: {exc}")
            st.stop()

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Store attachment information in the conversation without sending PDF bytes
    # to the language model.
    if uploaded_files:
        attachment_note = (
            f'A PDF named "{uploaded_files[0].name}" was uploaded '
            "and indexed for retrieval."
        )

        if user_input:
            backend_input = f"{attachment_note}\n\n{user_input}"
        else:
            backend_input = (
                f"{attachment_note}\n\n"
                "Briefly confirm the upload and ask what I would like "
                "to know about the PDF."
            )
    else:
        backend_input = user_input

    with st.chat_message("assistant"):
        tool_status = st.status("Agent is thinking...", expanded=True)
        used_tools = set()

        def stream_response():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=backend_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                calls = list(
                    getattr(message_chunk, "tool_calls", []) or []
                )
                calls.extend(
                    getattr(message_chunk, "tool_call_chunks", []) or []
                )

                for call in calls:
                    name = (
                        call.get("name")
                        if isinstance(call, dict)
                        else None
                    )

                    if name and name not in used_tools:
                        used_tools.add(name)
                        tool_status.write(f"Using tool: **{name}**")
                        tool_status.update(label=f"Using tool: {name}")

                if isinstance(message_chunk, AIMessage):
                    content = get_message_text(message_chunk.content)
                    if content:
                        yield content

        try:
            ai_message = st.write_stream(stream_response())
            tool_status.update(
                label="Response complete",
                state="complete",
                expanded=bool(used_tools),
            )
        except Exception as exc:
            tool_status.update(
                label="Response failed",
                state="error",
                expanded=True,
            )
            st.error(f"Could not generate a response: {exc}")
            st.stop()

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

# Backend requirement: the retrieval tool must load/use the updated faiss_db.
# The supplied ingestion function overwrites a shared index; it does not
# provide separate document indexes for different conversations or users.