from agentic_chatbot_db_backend import chatbot, get_all_threads
from langchain_core.messages import BaseMessage,AIMessage, HumanMessage
import streamlit as st
import uuid

# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the conversation list
def add_thread(thread_id):

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state['thread_id'] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state['message_history'] = []

    # Add the new thread to the conversation list
    add_thread(st.session_state['thread_id'])

# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            'configurable':{
                'thread_id': thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get('messages', [])

# Use the first user message as a readable label for each conversation
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

st.title("Agentic Chatbot with LangGraph")

# CONFIG = {'configurable': {'thread_id': 'thread-2'}}

# Create message history when the app runs for the first time
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Create a thread ID when the app runs for the first time
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# Create a list to store all conversation thread IDs
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()


# Add the current thread to the conversation list
add_thread(st.session_state['thread_id'])

# Display the sidebar title
st.sidebar.title("My Conversations")

# A button for starting a new conversation
if st.sidebar.button("+ New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()

# Display all conversation threads in reverse order
# This shows thenewest conversation first
for thread_id in st.session_state['chat_threads'][::-1]:

    # One sidebar button for every conversation
    if st.sidebar.button(
        get_thread_label(thread_id),
        key=thread_id
    ):
        # Set the selected thread as the current thread
        st.session_state['thread_id'] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all messages
        for message in messages:

            # Check whether the mssage was sent by the user:
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else: 
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state['message_history'] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()



# loading the convo history
for message in st.session_state['message_history']:
    # st.text(message)
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

# Run this block after the user submits a message
if user_input:

    # First add the message to message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    # Display the user's message in the chat interface
    with st.chat_message('user'):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable":{
            "thread_id": st.session_state['thread_id']
        },
        "metadata": {
            "thread_id": st.session_state['thread_id']
        },
        "run_name": "chat_trace",
    }

    # Create the assistant chat-message container
    with st.chat_message('assistant'):

        # Stream the assistant response token by token
        ai_message = st.write_stream(

            # Return the assistant response of AI message chunks
            message_chunk.content 

            # Stream the messages from the Langgraph chatbot
            for message_chunk, metadata in chatbot.stream(
                {
                    # Send the latest user message to the chatbot
                    'messages': [
                        HumanMessage(content=user_input)
                    ]
                },

                # Use the current conversation thread
                config=CONFIG,

                # Stream individual message chunks
                stream_mode ='messages'

            )

            # Display only AI messages
            # This prevents tool and user messages form appearing
            if isinstance(message_chunk, AIMessage)
        )

    # Save the complete assistant response in session state
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    
