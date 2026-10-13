from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    message = state['messages']
    # send to llm
    resp = llm.invoke(message)
    # response store state
    return {'messages': [resp]}

conn=sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpoint)
checkpoints = checkpoint.list(None)

def get_all_threads():
    all_threads = set()

    for checkpoint in checkpoints:
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)



