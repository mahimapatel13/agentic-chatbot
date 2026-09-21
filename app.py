from agentic_chatbot_backend import chatbot
from langchain_core.messages import BaseMessage, HumanMessage
import streamlit as st

st.title("Agentic Chatbot with Langgraph")

CONFIG = {'configurable': {'thread_id': 'thread-2'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# loading the convo history
for message in st.session_state['message_history']:
    # st.text(message)
    with st.chat_message(message['role']):
        st.text(message['content'])




user_input = st.chat_input('Type here')


if user_input:
    # first add the message to message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})

    with st.chat_message('user'):
        st.text(user_input)


    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config={'configurable':{'thread_id': 'thread-1'}},
                stream_mode ='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    
    
# thread_id="1"
# config = {'configurable': {'thread_id': thread_id}}
# resp = chatbot.invoke({'messages': [HumanMessage(content="What is the capital of india?")]}, config=config)

# print(resp['messages'][-1].content)