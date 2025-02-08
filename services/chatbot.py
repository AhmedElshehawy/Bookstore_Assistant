from pathlib import Path
from typing import Dict, List, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core import setup_logger, settings
from models import ChatResponse, RelevantTask, State
from services.chat_history import ChatHistoryService
from services.helpers import route_tools
from .tools import (
    calculator,
    execute_sql,
    generate_user_task,
    is_safe_sql,
    llm,
    text_to_sql,
)
from langsmith import traceable

logger = setup_logger(__name__, level=settings.LOG_LEVEL)

class ChatbotService:
    """Service for managing conversational AI interactions.
    
    This service handles the chat workflow, including message processing,
    tool execution, and response generation.
    """
    
    def __init__(self) -> None:
        """Initialize the chatbot service with required components."""
        try:
            self._initialize_components()
            self.graph = self._build_graph()
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotService: {e}")
            raise RuntimeError("ChatbotService initialization failed") from e

    def _initialize_components(self) -> None:
        """Initialize all required components for the chatbot."""
        self.memory = MemorySaver()
        self._initialize_tools()
        self._load_system_prompts()
        self._initialize_llms()

    def _initialize_tools(self) -> None:
        """Set up tool configurations for different components."""
        self.executor_tools = [text_to_sql, is_safe_sql, execute_sql, calculator]
        self.user_task_tools = [generate_user_task]

    def _load_system_prompts(self) -> None:
        """Load and validate system prompts from configuration files.
        
        Raises:
            RuntimeError: If any prompt file cannot be loaded
        """
        prompt_configs = {
            'chatbot': settings.CHATBOT_PROMPT_PATH,
            'relevant_task': settings.TASK_RELEVENCY_PROMPT_PATH,
            'plan_generation': settings.PLAN_GENERATION_PROMPT_PATH,
            'executor': settings.EXECUTOR_PROMPT_PATH
        }
        
        try:
            self.system_messages = {
                key: SystemMessage(content=Path(path).read_text())
                for key, path in prompt_configs.items()
            }
        except Exception as e:
            logger.error(f"Failed to load system prompts: {e}")
            raise RuntimeError("System prompts initialization failed") from e

    def _initialize_llms(self) -> None:
        """Configure language models with their respective tools."""
        self.llm = llm
        self.executor_llm = self.llm.bind_tools(self.executor_tools)
        self.chat_llm = self.llm.bind_tools(self.user_task_tools)

    def _build_graph(self) -> StateGraph:
        """Construct the conversation flow graph.
        
        Returns:
            StateGraph: Compiled conversation flow graph
        """
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("chatbot", self._chatbot_node)
        graph_builder.add_node("user_task_tools", 
                             ToolNode(tools=self.user_task_tools, 
                                    name='user_task_tools'))
        graph_builder.add_node("executor_tools", 
                             ToolNode(tools=self.executor_tools, 
                                    messages_key='executor_messages', 
                                    name='executor_tools'))
        graph_builder.add_node("relevance_checker", self._relevance_checker)
        graph_builder.add_node("plan_generator", self._plan_generator)
        graph_builder.add_node("executor", self._executor)
        graph_builder.add_node("ask_user_to_rephrase", self._ask_user_to_rephrase)

        # Configure graph edges
        self._configure_graph_edges(graph_builder)
        
        return graph_builder.compile(checkpointer=self.memory)

    def _chatbot_node(self, state: State) -> Dict[str, List[BaseMessage]]:
        """Process initial user input in the conversation.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict containing message list
        """
        messages = [self.system_messages['chatbot'], *state["messages"]]
        try:
            response = self.chat_llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error in chatbot node: {e}")
            return {"messages": [AIMessage(content="I apologize, but I encountered an error. Please try again.")]}

    def _relevance_checker(self, state: State) -> Command[Literal["plan_generator", "ask_user_to_rephrase"]]:
        """Check if the user's task is relevant and processable.
        
        Args:
            state: Current conversation state
            
        Returns:
            Command indicating next step
        """
        messages = [
            self.system_messages['relevant_task'],
            HumanMessage(content=f"User's task: {state['user_task']}")
        ]
        is_relevant = self.llm.with_structured_output(
            RelevantTask, 
            method="json_schema", 
            strict=True
        ).invoke(messages)
        
        return Command(goto="plan_generator" if is_relevant.is_relevant else "ask_user_to_rephrase")

    def _plan_generator(self, state: State) -> Dict[str, str]:
        """Generate execution plan for the user's task.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict containing the generated plan
        """
        messages = [
            self.system_messages['plan_generation'],
            HumanMessage(content=f"User's task: {state['user_task']}")
        ]
        plan = self.llm.invoke(messages)
        return {"plan": plan.content}

    def _executor(self, state: State) -> Dict[str, List[BaseMessage]]:
        """Execute the generated plan.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict containing executor messages
        """
        executor_messages = state["executor_messages"] or []
        messages = [
            self.system_messages['executor'],
            HumanMessage(content=f"Task: {state['user_task']}\n\nPlan: {state['plan']}")
        ] + executor_messages
        
        result = self.executor_llm.invoke(messages)
        return {"executor_messages": [result]}

    def _ask_user_to_rephrase(self, state: State) -> Dict[str, List[BaseMessage]]:
        """Generate a rephrasing request when task is unclear.
        
        Args:
            state: Current conversation state
            
        Returns:
            Dict containing rephrasing request message
        """
        messages = [
            HumanMessage(
                content=f"You are an AI assistant specialized in answering questions about a bookstore. "
                       f"The user wants to perform this task: {state['user_task']}, "
                       f"And you can't perform it. Ask the user to rephrase their question."
            )
        ]
        result = self.llm.invoke(messages)
        return {"messages": [result], "executor_messages": [result]}

    def _configure_graph_edges(self, graph: StateGraph) -> None:
        """Configure the edges between nodes in the conversation graph.
        
        Args:
            graph: StateGraph instance to configure
        """
        graph.add_conditional_edges(
            "chatbot",
            tools_condition,
            {
                "tools": "user_task_tools",
                "__end__": END
            }
        )
        
        graph.add_edge("user_task_tools", "relevance_checker")
        graph.add_edge("ask_user_to_rephrase", END)
        graph.add_edge("plan_generator", "executor")
        
        graph.add_conditional_edges(
            "executor",
            route_tools,
            {
                "tools": "executor_tools",
                "__end__": END
            }
        )
        
        graph.add_edge("executor_tools", "executor")
        graph.add_edge(START, "chatbot")

    @traceable
    async def chat(self, user_input: str, session_id: str) -> ChatResponse:
        """Process a user message and generate a response.
        
        Args:
            user_input: The user's message
            session_id: Unique session identifier
            
        Returns:
            ChatResponse: The AI's response
            
        Raises:
            ValueError: If the input is invalid
            RuntimeError: If message processing fails
        """
        if not user_input.strip():
            raise ValueError("User input cannot be empty")

        try:
            return await self._process_chat(user_input, session_id)
        except Exception as e:
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise RuntimeError(f"Failed to process message: {str(e)}")

    async def _process_chat(self, user_input: str, session_id: str) -> ChatResponse:
        """Internal method to process chat messages.
        
        Args:
            user_input: The user's message
            session_id: Unique session identifier
            
        Returns:
            ChatResponse: The AI's response
        """
        config = self._get_config(session_id)
        chat_history = ChatHistoryService(
            settings.CHAT_HISTORY_TABLE_NAME,
            settings.CHAT_HISTORY_PRIMARY_KEY_NAME,
            session_id
        )
        
        # update the chat history in the current graph state with the history retrieved from db
        if not list(self.graph.get_state_history(config)):
            self.graph.update_state(config, values={'messages': chat_history.messages})
        
        events = self.graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode='values'
        )
        
        final_response = await self._process_events(events)
        
        chat_history.add_user_message(HumanMessage(content=user_input.strip()))
        chat_history.add_assistant_message(AIMessage(content=final_response))
        
        return ChatResponse(content=final_response, role="assistant")

    @staticmethod
    def _get_config(thread_id: str) -> dict:
        """Get configuration for the chat service.
        
        Args:
            thread_id: The thread identifier
            
        Returns:
            dict: Configuration dictionary
            
        Raises:
            ValueError: If thread_id is empty
        """
        if not thread_id.strip():
            raise ValueError("thread_id cannot be empty")
        return {"configurable": {"thread_id": thread_id.strip()}}

    async def _process_events(self, events) -> str:
        """Process chat events and extract final response.
        
        Args:
            events: Iterator of chat events
            
        Returns:
            str: Final response content
            
        Raises:
            RuntimeError: If event processing fails
        """
        final_response = "I apologize, but I encountered an error. Please try again."
        
        for event in events:
            self._log_event(event)
            try:
                if event.get("executor_messages"):
                    final_response = event["executor_messages"][-1].content
                elif event.get("messages"):
                    final_response = event["messages"][-1].content
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)
                raise RuntimeError(f"Failed to process event: {str(e)}")
        
        return final_response

    @staticmethod
    def _log_event(event: Dict) -> None:
        """Log chat event details.
        
        Args:
            event: Chat event dictionary
        """
        if event.get('executor_messages'):
            logger.info(
                f"Processing chat response: "
                f"Content: {event['executor_messages'][-1].content}, "
                f"Type: {event['executor_messages'][-1].type}"
            )
        elif event.get('messages'):
            logger.info(
                f"Processing chat response: "
                f"Content: {event['messages'][-1].content}, "
                f"Type: {event['messages'][-1].type}"
            )
