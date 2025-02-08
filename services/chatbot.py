from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any
from langgraph.graph import StateGraph, START
from models import State, ChatResponse
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from pathlib import Path

from langgraph.prebuilt import ToolNode, tools_condition
from .tools import text_to_sql, is_safe_sql, execute_sql, calculator
from .tools import llm

from core import setup_logger, settings,Settings
from .chat_history import ChatHistoryService

# Initialize logger
logger = setup_logger(__name__, level=settings.LOG_LEVEL)

class ChatbotService:
    def __init__(self, prompt_path: str, chat_history_table_name: str, chat_history_primary_key_name: str):
        """Initialize the chatbot service with configuration."""
        self.memory = MemorySaver()
        self.tools = [text_to_sql, is_safe_sql, execute_sql, calculator]
        
        # Load system prompt
        try:
            prompt_path = Path(prompt_path)
            self.system_message = SystemMessage(content=prompt_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            raise RuntimeError("Could not initialize chatbot service")

        # Initialize LLM
        self.llm = llm
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Setup graph
        self.graph = self._build_graph()
        

    def _build_graph(self) -> StateGraph:
        """Build the conversation flow graph."""
        def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
            messages = [self.system_message]
            messages.extend(state["messages"])
            try:
                chat_response = self.llm_with_tools.invoke(messages)
                return {"messages": [chat_response]}
            except Exception as e:
                logger.error(f"Error in chatbot node: {e}")
                error_message = AIMessage(content="I apologize, but I encountered an error. Please try again.")
                return {"messages": [error_message]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        
        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)
        
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")
        
        return graph_builder.compile(checkpointer=self.memory)

    async def chat(self, user_input: str, session_id: str) -> ChatResponse:
        """
        Process a user message and return a response.
        
        Args:
            user_input: The user's message
            config: Configuration dictionary for the conversation
            
        Returns:
            ChatResponse: The AI's response
            
        Raises:
            ValueError: If the input is invalid
        """
        config = self.get_config(session_id)
        
        if not user_input.strip():
            raise ValueError("User input cannot be empty")

        try:
            chat_history = ChatHistoryService(settings.CHAT_HISTORY_TABLE_NAME, settings.CHAT_HISTORY_PRIMARY_KEY_NAME, session_id)
            logger.info(f'Previous state chat history: {list(self.graph.get_state_history(config))}')
            logger.info(f'Current state chat history from db: {chat_history.get_messages()}')
            if not list(self.graph.get_state_history(config)):
                self.graph.update_state(config, values={'messages': chat_history.get_messages()})
            
            logger.info(f'Updated state chat history: {list(self.graph.get_state_history(config))}')
            events = self.graph.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode='values'
            )
            
            for event in events:
                # event["messages"][-1].pretty_print()
                logger.info(
                    f"Processing chat response: Content: {event['messages'][-1].content}, Type: {event['messages'][-1].type}"
                )
            
            final_message = event["messages"][-1]
            final_response = final_message.content
            
            chat_history.add_user_message(HumanMessage(content=user_input.strip()))
            chat_history.add_assistant_message(AIMessage(content=final_response))
            
            # Log conversation for monitoring
            logger.info(
                "Chat completed",
                extra={
                    "user_input_length": len(user_input),
                    "response_length": len(final_response)
                }
            )
            
            return ChatResponse(
                content=final_response,
                role="assistant"
            )

        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process message: {str(e)}")
        
    def get_config(self, thread_id: str) -> dict:
        """Get configuration for the chat service.
        
        Args:
            thread_id (str): The thread identifier
            
        Returns:
            dict: Configuration dictionary
        """
        if not thread_id.strip():
            raise ValueError("thread_id cannot be empty")
        return {"configurable": {"thread_id": thread_id.strip()}}
