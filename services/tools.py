from langgraph.types import Command

from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langchain_core.messages import ToolMessage

from typing import Annotated
from langchain_core.runnables import RunnableConfig
import json
import math
from typing import Any, Dict
import numexpr
import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from core import settings
from core.logger import setup_logger
from models.schemas import SQLQuery

# Initialize logger
logger = setup_logger(__name__, level=settings.LOG_LEVEL)

# Initialize LLM
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY
)

generate_user_task_system_message = SystemMessage(content=open(settings.TASK_GENERATION_PROMPT_PATH).read())
# Load system prompts
try:
    text_to_sql_system_message = SystemMessage(content=open(settings.TEXT_TO_SQL_PROMPT_PATH).read())
    is_safe_sql_system_message = SystemMessage(content=open(settings.IS_SAFE_SQL_PROMPT_PATH).read())
    plan_generation_system_message = SystemMessage(content=open(settings.PLAN_GENERATION_PROMPT_PATH).read())
    executor_system_message = SystemMessage(content=open(settings.EXECUTOR_PROMPT_PATH).read())
    task_relevency_system_message = SystemMessage(content=open(settings.TASK_RELEVENCY_PROMPT_PATH).read())
    task_generation_system_message = SystemMessage(content=open(settings.TASK_GENERATION_PROMPT_PATH).read())
except FileNotFoundError as e:
    logger.error(f"Failed to load prompt files: {e}")
    raise

class SQLExecutionError(Exception):
    """Custom exception for SQL execution errors"""
    pass

@tool
def text_to_sql(user_query: str) -> str:
    """
    Convert natural language query to SQL.
    
    Args:
        user_query (str): The user's natural language query
        
    Returns:
        str: Generated SQL query
        
    Raises:
        Exception: If SQL generation fails
    """
    try:
        messages = [text_to_sql_system_message, HumanMessage(content=f"User's query: {user_query}")]
        structured_llm = llm.with_structured_output(SQLQuery, method="json_schema", strict=True)
        sql_query = structured_llm.invoke(messages)
        logger.info(f"Generated SQL query: {sql_query.sql_query}")
        return sql_query.sql_query
    except Exception as e:
        logger.error(f"Failed to convert text to SQL: {e}")
        raise

@tool
def is_safe_sql(sql_query: str) -> bool:
    """
    Check if the SQL query is safe to execute.
    
    Args:
        sql_query (str): The SQL query to check
        
    Returns:
        bool: True if safe, False otherwise
    """
    try:
        messages = [is_safe_sql_system_message, HumanMessage(content=f"SQL query: {sql_query}")]
        response = llm.invoke(messages)
        ## TODO: Return the full response content
        is_safe = 'safe' in response.content.lower() and 'not safe' not in response.content.lower()
        logger.info(f"SQL safety check result: {is_safe}")
        return response.content.lower()
    except Exception as e:
        logger.error(f"Failed to check SQL safety: {e}")
        return False

@tool
def execute_sql(sql_query: str) -> Dict[str, Any]:
    """
    Execute SQL query and return results.
    
    Args:
        sql_query (str): The SQL query to execute
        
    Returns:
        Dict[str, Any]: Query results
        
    Raises:
        SQLExecutionError: If query execution fails
    """
    try:
        payload = json.dumps({"query": sql_query})
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(
            settings.DB_URL,
            headers=headers,
            data=payload,
            timeout=30  # Add timeout
        )
        
        response.raise_for_status()  # Raise exception for bad status codes
        
        result = response.json()
        logger.info(f"Successfully executed SQL query")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to execute SQL query: {e}")
        raise SQLExecutionError(f"Database request failed: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SQL query response: {e}")
        raise SQLExecutionError(f"Invalid response format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during SQL execution: {e}")
        raise SQLExecutionError(f"Unexpected error: {str(e)}")

@tool
def calculator(expression: str) -> str:
    """
    Evaluate mathematical expressions safely.
    
    Args:
        expression (str): Mathematical expression to evaluate
        
    Returns:
        str: Result of the calculation
        
    Raises:
        ValueError: If expression is invalid
    """
    try:
        # Define allowed mathematical constants
        local_dict = {
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "abs": abs
        }
        
        # Clean and validate expression
        cleaned_expression = expression.strip()
        if not cleaned_expression:
            raise ValueError("Empty expression")
            
        result = numexpr.evaluate(
            cleaned_expression,
            global_dict={},  # Restrict access to globals
            local_dict=local_dict,
        )
        
        logger.info(f"Calculated expression: {cleaned_expression} = {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        raise ValueError(f"Invalid mathematical expression: {str(e)}")
    
@tool
def is_relevant(user_query: str) -> str:
    """
    Check if the user's query is relevant to the bookstore.
    
    Args:
        user_query (str): The user's query
        
    Returns:
        bool: True if relevant, False otherwise
    """
    try:
        messages = [
            SystemMessage(
                content="You are an AI assistant tasked with answering questions related to a bookstore called https://books.toscrape.com. The bookstore's data is stored in a PostgreSQL database, and you can query this database to respond to user inquiries. If the user asks about something that is not related to the bookstore, you should politely inform them that you are not able to answer that question."
            ), 
            HumanMessage(content=f"User's query: {user_query}")
        ]
        response = llm.invoke(messages)
        is_relevant = response.content.lower()
        logger.info(f"Relevance check result: {is_relevant}")
        return is_relevant
    except Exception as e:
        logger.error(f"Failed to check relevance: {e}")
        return False

@tool
def generate_user_task(tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig, chat_history: Annotated[list, InjectedState("messages")]) -> str:
    """Generate the user's intended task based on the chat history"""
    messages = [generate_user_task_system_message]
    messages.extend(chat_history[:-1])
    
    return Command(
        update={
            # update the state keys
            "user_task": llm.invoke(messages).content,
            # update the message history
            "messages": [ToolMessage("Successfully generated user's task", tool_call_id=tool_call_id)]
        }
    )
