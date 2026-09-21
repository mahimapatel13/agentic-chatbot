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
    resp = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)

    ai=resp['messages'][-1].content
    # first add the message to message history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai})
        
    with st.chat_message('assistant'):
            st.text(ai)

    
    
# thread_id="1"
# config = {'configurable': {'thread_id': thread_id}}
# resp = chatbot.invoke({'messages': [HumanMessage(content="What is the capital of india?")]}, config=config)

# print(resp['messages'][-1].content)