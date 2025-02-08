from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain_core.tools import tool
import requests
import json
import getpass
import os

from typing import Annotated
import json
import numexpr
import math
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class State(TypedDict):
    messages: Annotated[list, add_messages]
    executor_messages: Annotated[list, add_messages]
    user_task: str
    plan: str

class RelevantTask(BaseModel):
    is_relevant: bool = Field(description="Whether the user's task is relevant to a bookstore.")

class SQLQuery(BaseModel):
    sql_query: str = Field(description="The SQL query generated from the user's query. Default is ''")
    
class QueryResponse(BaseModel):
    ai_response: str = Field(description="The AI response to the user's query. Default is ''")

class QueryRequest(BaseModel):
    user_query: str = Field(description="The user's query. Default is ''")
    thread_id: str = Field(description="The thread id")

class ChatResponse(BaseModel):
    content: str
    role: str = "assistant"

