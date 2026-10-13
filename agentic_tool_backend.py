from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition 
from langchain_tavily import TavilySearch 
from langchain_core.tools import tool
import requests
import math
import os 

load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """ 
    Useful for simple math calculations.
    Input should be a valid mathematical expression.
    Example: 2+2, math.sqrt(16),10*5
    """

    try:
        allowed={
            "math": math,
            "abs" : abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """ 
    Fetch latest stock price for a given symbol
  .
    """

    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('ALPHA_VANTAGE_API_KEY')}"
    r = requests.get(url)
    return r.json()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""

    # 1. Convert location -> latitude/longitude
    geo_url = "https://api.openweathermap.org/geo/1.0/direct"

    geo_params = {
        "q": location,
        "limit": 1,
        "appid": OPENWEATHER_API_KEY
    }

    try:
        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )
        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if not geo_data:
            return f"Could not find the location: {location}"

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # 2. Get current weather using coordinates
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )
        weather_response.raise_for_status()

        data = weather_response.json()

        return (
            f"Current weather in {data['name']}, {data['sys']['country']}: "
            f"{data['weather'][0]['description']}, "
            f"temperature {data['main']['temp']}°C, "
            f"feels like {data['main']['feels_like']}°C, "
            f"humidity {data['main']['humidity']}%, "
            f"wind speed {data['wind']['speed']} m/s."
        )

    except requests.RequestException as e:
        return f"Failed to fetch weather: {e}"
    
from langgraph.graph.message import add_messages

tools = [search_tool, calculator, get_stock_price, get_current_weather]

llm_with_tools = llm.bind_tools(tools)
# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    message = state['messages']
    # send to llm
    resp = llm_with_tools.invoke(message)
    # response store state
    return {'messages': [resp]}

# Tool node
tool_node = ToolNode(tools)

conn=sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# add edges
graph.add_edge(START, "chat_node")

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)
checkpoints = checkpoint.list(None)

# helper function to get all threads for streamlit
def get_all_threads():
    all_threads = set()

    for checkpoint in checkpoints:
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)



