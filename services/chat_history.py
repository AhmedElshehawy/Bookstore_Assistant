from typing import List, Optional
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class ChatHistoryService:
    """Service class for managing chat history using DynamoDB.
    
    This class provides an interface to store and retrieve chat messages
    using DynamoDB as the backend storage.
    
    Attributes:
        chat_history: DynamoDB chat message history instance
    """
    
    def __init__(self, table_name: str, primary_key_name: str, session_id: str) -> None:
        """Initialize ChatHistoryService.
        
        Args:
            table_name: Name of the DynamoDB table
            primary_key_name: Name of the primary key in the DynamoDB table
            session_id: Unique identifier for the chat session
        """
        try:
            self._chat_history = DynamoDBChatMessageHistory(
                table_name=table_name,
                primary_key_name=primary_key_name,
                session_id=session_id
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DynamoDB chat history: {str(e)}")

    def add_user_message(self, message: HumanMessage) -> None:
        """Add a user message to the chat history.
        
        Args:
            message: The user's message to add
            
        Raises:
            ValueError: If message is None or empty
        """
        if not message or not message.content:
            raise ValueError("Message cannot be None or empty")
        
        try:
            self._chat_history.add_user_message(message)
        except Exception as e:
            raise RuntimeError(f"Failed to add user message: {str(e)}")

    def add_assistant_message(self, message: AIMessage) -> None:
        """Add an assistant message to the chat history.
        
        Args:
            message: The assistant's message to add
            
        Raises:
            ValueError: If message is None or empty
        """
        if not message or not message.content:
            raise ValueError("Message cannot be None or empty")
            
        try:
            self._chat_history.add_ai_message(message)
        except Exception as e:
            raise RuntimeError(f"Failed to add assistant message: {str(e)}")

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages from the chat history.
        
        Returns:
            List of chat messages. Empty list if no messages exist.
        """
        try:
            return self._chat_history.messages if self._chat_history.messages else []
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve messages: {str(e)}")

    def clear(self) -> None:
        """Clear all messages from the chat history.
        
        Raises:
            RuntimeError: If clearing the history fails
        """
        try:
            self._chat_history.clear()
        except Exception as e:
            raise RuntimeError(f"Failed to clear chat history: {str(e)}")
