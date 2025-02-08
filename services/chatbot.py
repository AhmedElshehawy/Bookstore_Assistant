from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any, Literal
from langgraph.graph import StateGraph, START
from models import State, ChatResponse, RelevantTask
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from pathlib import Path

from langgraph.prebuilt import ToolNode, tools_condition
from .tools import text_to_sql, is_safe_sql, execute_sql, calculator, generate_user_task
from .tools import llm

from core import setup_logger, settings,Settings
from .chat_history import ChatHistoryService
from langgraph.types import Command
from services.helpers import route_tools
from langgraph.graph import END
# Initialize logger
logger = setup_logger(__name__, level=settings.LOG_LEVEL)

class ChatbotService:
    def __init__(self):
        """Initialize the chatbot service with configuration."""
        self.memory = MemorySaver()
        self.executor_tools = [text_to_sql, is_safe_sql, execute_sql, calculator]
        self.user_task_tools = [generate_user_task]
        # Load system prompt
        try:
            chatbot_prompt_path = Path(settings.CHATBOT_PROMPT_PATH)
            relevant_task_prompt_path = Path(settings.TASK_RELEVENCY_PROMPT_PATH)
            plan_generation_prompt_path = Path(settings.PLAN_GENERATION_PROMPT_PATH)
            executor_prompt_path = Path(settings.EXECUTOR_PROMPT_PATH)
            
            self.chatbot_system_message = SystemMessage(content=chatbot_prompt_path.read_text())
            self.relevant_task_system_message = SystemMessage(content=relevant_task_prompt_path.read_text())
            self.plan_generation_system_message = SystemMessage(content=plan_generation_prompt_path.read_text())
            self.executor_system_message = SystemMessage(content=executor_prompt_path.read_text())
        except Exception as e:
            logger.error(f"Failed to load system prompt: {e}")
            raise RuntimeError("Could not initialize chatbot service")

        # Initialize LLM
        self.llm = llm
        self.executor_llm = self.llm.bind_tools(self.executor_tools)
        self.chat_llm = self.llm.bind_tools(self.user_task_tools)
        
        # Setup graph
        self.graph = self._build_graph()
        

    def _build_graph(self) -> StateGraph:
        """Build the conversation flow graph."""
        def chatbot(state: State) -> Dict[str, List[BaseMessage]]:
            messages = [self.chatbot_system_message]
            messages.extend(state["messages"])
            try:
                chat_response = self.chat_llm.invoke(messages)
                return {"messages": [chat_response]}
            except Exception as e:
                logger.error(f"Error in chatbot node: {e}")
                error_message = AIMessage(content="I apologize, but I encountered an error. Please try again.")
                return {"messages": [error_message]}
        
        def relevance_checker(state: State) -> Command[Literal["plan_generator", "ask_user_to_rephrase"]]:
            messages = [self.relevant_task_system_message, HumanMessage(content=f"User's task: {state['user_task']}")]
            is_relevant = self.llm.with_structured_output(RelevantTask, method="json_schema", strict=True).invoke(messages)
            
            if is_relevant.is_relevant:
                goto = "plan_generator"
            else:
                goto = "ask_user_to_rephrase"
                
            return Command(
                goto=goto
            )


        def plan_generator(state: State):
            messages = [self.plan_generation_system_message, HumanMessage(content=f"User's task: {state['user_task']}")]
            plan = self.llm.invoke(messages)
            return {"plan": plan.content}

        def executor(state: State):
            executor_messages = state["executor_messages"] or []
            messages = [self.executor_system_message, HumanMessage(content=f"Task: {state['user_task']}\n\nPlan: {state['plan']}")] + executor_messages
            result = self.executor_llm.invoke(messages)
            return {"executor_messages": [result]}
        
        def ask_user_to_rephrase(state: State):
            messages = [HumanMessage(content=f"You are an AI assistant specialized in answering questions about a bookstore. The user wants to perform this task: {state['user_task']}, And you can't perform it. Ask the user to rephrase their question.")]
            result = self.llm.invoke(messages)
            return {"messages": [result]}
    
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("chatbot", chatbot)
        executor_tools_node = ToolNode(tools=self.executor_tools, messages_key='executor_messages', name='executor_tools')
        user_task_node = ToolNode(tools=self.user_task_tools, name='user_task_tools')

        graph_builder.add_node("user_task_tools", user_task_node)
        graph_builder.add_node("relevance_checker", relevance_checker)
        graph_builder.add_node("plan_generator", plan_generator)
        graph_builder.add_node("executor", executor)
        graph_builder.add_node("ask_user_to_rephrase", ask_user_to_rephrase)

        graph_builder.add_node("executor_tools", executor_tools_node)


        graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {
                "tools": "user_task_tools",
                "__end__": END
            }
        )
        
        graph_builder.add_edge("user_task_tools", "relevance_checker")
        graph_builder.add_edge("ask_user_to_rephrase", END)
        graph_builder.add_edge("plan_generator", "executor")
        graph_builder.add_conditional_edges(
            "executor",
            route_tools,
            {
                "tools": "executor_tools",
                "__end__": END
            }
        )
        graph_builder.add_edge("executor_tools", "executor")
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
                if event['executor_messages']:
                    logger.info(
                        f"Processing chat response: Content: {event['executor_messages'][-1].content}, Type: {event['executor_messages'][-1].type}"
                    )
                else:
                    logger.info(
                        f"Processing chat response: Content: {event['messages'][-1].content}, Type: {event['messages'][-1].type}"
                    )
            
            try:
                if event["executor_messages"]:
                    final_message = event["executor_messages"][-1]
                    final_response = final_message.content
                else:
                    final_message = event["messages"][-1]
                    final_response = final_message.content
            except Exception as e:
                logger.error(f"Error in chat: {e}", exc_info=True)
                final_response = "I apologize, but I encountered an error. Please try again."
                raise RuntimeError(f"Failed to process message: {str(e)}")
            
            chat_history.add_user_message(HumanMessage(content=user_input.strip()))
            chat_history.add_assistant_message(AIMessage(content=final_response))
            
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
