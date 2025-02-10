"""
Tools module providing various utility functions for the bookstore application.
Contains tools for SQL operations, calculations, and task management.
"""

from typing import Any, Dict, List, Annotated
import json
import math
import numexpr
import requests

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from core import settings
from core.logger import setup_logger
from models.schemas import SQLQuery


# Initialize logger
logger = setup_logger(__name__, level=settings.LOG_LEVEL)

class PromptLoader:
    """Handles loading and managing system prompts."""
    
    @staticmethod
    def load_prompt(path: str) -> SystemMessage:
        """Load a prompt file and return as SystemMessage."""
        try:
            with open(path, 'r', encoding='utf-8') as file:
                return SystemMessage(content=file.read())
        except FileNotFoundError as e:
            logger.error(f"Failed to load prompt file {path}: {e}")
            raise

class SystemPrompts:
    """Container for all system prompts used in the application."""
    
    def __init__(self):
        """Initialize all system prompts."""
        self.text_to_sql = PromptLoader.load_prompt(settings.TEXT_TO_SQL_PROMPT_PATH)
        self.is_safe_sql = PromptLoader.load_prompt(settings.IS_SAFE_SQL_PROMPT_PATH)
        self.plan_generation = PromptLoader.load_prompt(settings.PLAN_GENERATION_PROMPT_PATH)
        self.executor = PromptLoader.load_prompt(settings.EXECUTOR_PROMPT_PATH)
        self.task_relevency = PromptLoader.load_prompt(settings.TASK_RELEVENCY_PROMPT_PATH)
        self.task_generation = PromptLoader.load_prompt(settings.TASK_GENERATION_PROMPT_PATH)

class SQLExecutionError(Exception):
    """Custom exception for SQL execution errors."""
    pass

class DatabaseClient:
    """Handles database interactions."""
    
    @staticmethod
    def execute_query(sql_query: str) -> Dict[str, Any]:
        """Execute SQL query against the database."""
        try:
            payload = json.dumps({"query": sql_query})
            headers = {'Content-Type': 'application/json'}
            
            response = requests.post(
                settings.DB_URL,
                headers=headers,
                data=payload
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Database request failed: {e}")
            raise SQLExecutionError(f"Database request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid response format: {e}")
            raise SQLExecutionError(f"Invalid response format: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise SQLExecutionError(f"Unexpected error: {str(e)}")

# Initialize global instances
prompts = SystemPrompts()
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)


@tool
def text_to_sql(user_query: str) -> str:
    """Convert natural language query to SQL."""
    try:
        messages = [prompts.text_to_sql, HumanMessage(content=f"User's query: {user_query}")]
        structured_llm = llm.with_structured_output(SQLQuery, method="json_schema", strict=True)
        sql_query = structured_llm.invoke(messages)
        logger.info(f"Generated SQL query: {sql_query.sql_query}")
        return sql_query.sql_query
    except Exception as e:
        logger.error(f"Failed to convert text to SQL: {e}")
        raise


@tool
def is_safe_sql(sql_query: str) -> str:
    """Check if the SQL query is safe to execute."""
    try:
        messages = [prompts.is_safe_sql, HumanMessage(content=f"SQL query: {sql_query}")]
        response = llm.invoke(messages)
        logger.info(f"SQL safety check completed: {response.content.lower()}")
        return response.content.lower()
    except Exception as e:
        logger.error(f"Failed to check SQL safety: {e}")
        return "not safe"


@tool
def execute_sql(sql_query: str) -> Dict[str, Any]:
    """Execute SQL query and return results."""
    logger.info(f"Executing SQL query: {sql_query}")
    return DatabaseClient.execute_query(sql_query)


@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions safely."""
    try:
        local_dict = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "abs": abs
        }
        
        cleaned_expression = expression.strip()
        if not cleaned_expression:
            raise ValueError("Empty expression")
            
        result = numexpr.evaluate(
            cleaned_expression,
            global_dict={},
            local_dict=local_dict,
        )
        
        logger.info(f"Calculated: {cleaned_expression} = {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        raise ValueError(f"Invalid mathematical expression: {str(e)}")


@tool
def is_relevant(user_query: str) -> str:
    """Check if the user's query is relevant to the bookstore."""
    try:
        messages = [prompts.task_relevency, HumanMessage(content=f"User's query: {user_query}")]
        response = llm.invoke(messages)
        logger.info(f"Relevance check completed: {response.content.lower()}")
        return response.content.lower()
    except Exception as e:
        logger.error(f"Failed to check relevance: {e}")
        return "not relevant"


@tool
def generate_user_task(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
    chat_history: Annotated[List, InjectedState("messages")]
) -> Command:
    """Generate the user's intended task based on the chat history."""
    messages = [prompts.task_generation]
    
    messages.extend(chat_history[:-1])
    
    task = llm.invoke(messages).content
    logger.info(f"Generated user task: {task}")
    
    return Command(
        update={
            "user_task": str(task),
            "messages": [ToolMessage(f"Successfully generated user's task: {str(task)}", tool_call_id=tool_call_id)]
        }
    )
