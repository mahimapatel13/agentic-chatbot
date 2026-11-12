"""RAG-style chat interface with human-in-the-loop decisions."""
from uuid import uuid4
from pathlib import Path
import tempfile

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from chatbot_with_hitl import chatbot, checkpoint, ingest_rag_document

st.title("Agentic Chatbot with LangGraph")

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


if "hitl_thread_id" not in st.session_state:
    st.session_state["hitl_thread_id"] = str(uuid4())

try:
    saved_threads = list(dict.fromkeys(
        item.config["configurable"]["thread_id"]
        for item in checkpoint.list(None)
    ))
except Exception as exc:
    st.error(f"Could not load conversations: {exc}")
    st.stop()

threads = st.session_state.setdefault("hitl_chat_threads", [])
for thread_id in reversed(saved_threads):
    if thread_id not in threads:
        threads.append(thread_id)
if st.session_state["hitl_thread_id"] not in threads:
    threads.append(st.session_state["hitl_thread_id"])


def load_state(thread_id):
    return chatbot.get_state({"configurable": {"thread_id": thread_id}})


def get_thread_label(thread_id):
    for message in load_state(thread_id).values.get("messages", []):
        if isinstance(message, HumanMessage):
            snippet = " ".join(get_message_text(message.content).split())
            return snippet[:37].rstrip() + "..." if len(snippet) > 40 else snippet or "Empty conversation"
    return "New conversation"


st.sidebar.title("My Conversations")
if st.sidebar.button("+ New Chat"):
    st.session_state["hitl_thread_id"] = str(uuid4())
    st.rerun()

try:
    for thread_id in threads[::-1]:
        if st.sidebar.button(get_thread_label(thread_id), key=thread_id):
            st.session_state["hitl_thread_id"] = thread_id
            st.rerun()
    state = load_state(st.session_state["hitl_thread_id"])
except Exception as exc:
    st.error(f"Could not load the conversation: {exc}")
    st.stop()

config = {
    "configurable": {"thread_id": st.session_state["hitl_thread_id"]},
    "metadata": {"thread_id": st.session_state["hitl_thread_id"]},
    "run_name": "chat_trace",
}


def run_chat(graph_input):
    """Stream answers and tool activity, then reload checkpoints and interrupts."""
    with st.chat_message("assistant"):
        tool_status = st.status("Agent is thinking...", expanded=True)
        used_tools = set()

        def stream_response():
            for message, metadata in chatbot.stream(
                graph_input, config=config, stream_mode="messages"
            ):
                calls = list(getattr(message, "tool_calls", []) or [])
                calls.extend(getattr(message, "tool_call_chunks", []) or [])
                for call in calls:
                    name = call.get("name") if isinstance(call, dict) else None
                    if name and name not in used_tools:
                        used_tools.add(name)
                        tool_status.write(f"Using tool: **{name}**")
                        tool_status.update(label=f"Using tool: {name}")
                if isinstance(message, AIMessage):
                    text = get_message_text(message.content)
                    if text:
                        yield text

        try:
            st.write_stream(stream_response())
            latest = load_state(st.session_state["hitl_thread_id"])
            paused = any(task.interrupts for task in latest.tasks)
            tool_status.update(
                label="Waiting for your decision" if paused else "Response complete",
                state="complete", expanded=bool(used_tools),
            )
        except Exception as exc:
            tool_status.update(label="Response failed", state="error", expanded=True)
            st.session_state["hitl_error"] = str(exc)
    st.rerun()


if error := st.session_state.pop("hitl_error", None):
    st.error(f"Could not complete the request: {error}")

for message in state.values.get("messages", []):
    if isinstance(message, (HumanMessage, AIMessage)):
        content = get_message_text(message.content)
        if content:
            with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
                st.markdown(content)

pending = [item for task in state.tasks for item in task.interrupts]
if pending:
    st.warning("The assistant is paused. Approve or disapprove each action to continue.")
    decision_key = "hitl-decisions-" + st.session_state["hitl_thread_id"] + "-" + "-".join(
        item.id for item in pending
    )
    responses = st.session_state.setdefault(decision_key, {})
    for index, item in enumerate(pending, start=1):
        st.subheader(f"Decision {index}")
        if isinstance(item.value, str):
            st.write(item.value)
        else:
            st.json(item.value)
        approve_column, disapprove_column = st.columns(2)
        button_key = f"{st.session_state['hitl_thread_id']}-{item.id}"
        if approve_column.button("Approve", key=f"approve-{button_key}", type="primary"):
            responses[item.id] = "yes"
        if disapprove_column.button("Disapprove", key=f"disapprove-{button_key}"):
            responses[item.id] = "no"
        if item.id in responses:
            st.caption("Approved" if responses[item.id] == "yes" else "Disapproved")

    # Resume all pending calls together once every action has a decision.
    if all(item.id in responses for item in pending):
        decisions = dict(responses)
        del st.session_state[decision_key]
        run_chat(Command(resume=decisions))

chat_submission = st.chat_input(
    "Type a message or attach a PDF",
    accept_file=True,
    file_type=["pdf"],
    disabled=bool(pending),
)
if chat_submission:
    user_input = chat_submission.text.strip()
    uploaded_files = chat_submission.files
    if not user_input and not uploaded_files:
        st.stop()
    display_parts = [user_input] if user_input else []
    display_parts.extend(f"Attached PDF: {file.name}" for file in uploaded_files)
    with st.chat_message("user"):
        st.markdown("\n\n".join(display_parts))

    if uploaded_files:
        uploaded_file = uploaded_files[0]
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = Path(temp_dir) / "uploaded_document.pdf"
                    pdf_path.write_bytes(uploaded_file.getvalue())
                    ingest_rag_document(str(pdf_path))
        except Exception as exc:
            st.error(f"Could not process the PDF: {exc}")
            st.stop()
        attachment_note = f'A PDF named "{uploaded_file.name}" was uploaded and indexed for retrieval.'
        user_input = attachment_note + "\n\n" + (
            user_input or "Briefly confirm the upload and ask what I would like to know about the PDF."
        )
    run_chat({"messages": [HumanMessage(content=user_input)]})
