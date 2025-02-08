from langgraph.graph import END
from models import State
from typing import Union, List
from openai.types.chat import ChatCompletionMessage

def route_tools(state: Union[State, List[ChatCompletionMessage]]) -> str:
    """
    Routes to the ToolNode if the last message has tool calls, otherwise routes to END.
    
    Args:
        state: Either a State object or a list of messages. The state must contain
              at least one AI message to check for tool calls.
    
    Returns:
        str: Either "tools" if tool calls are present, or END if no tool calls found
    
    Raises:
        ValueError: If no messages can be found in the input state
    """
    # Extract the last AI message based on input type
    if isinstance(state, list):
        last_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get("executor_messages")):
        last_message = messages[-1]
    else:
        raise ValueError(f"Unable to extract messages from state: {state}")

    # Check for tool calls
    has_tool_calls = (
        hasattr(last_message, "tool_calls") and 
        last_message.tool_calls and 
        len(last_message.tool_calls) > 0
    )
    
    return "tools" if has_tool_calls else END
