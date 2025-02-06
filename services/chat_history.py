from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class ChatHistoryService:
    def __init__(self, table_name: str, primary_key_name: str, session_id: str):
        self.chat_history = DynamoDBChatMessageHistory(table_name=table_name, primary_key_name=primary_key_name, session_id=session_id)

    def add_user_message(self, message: HumanMessage):
        self.chat_history.add_user_message(message)
    
    def add_assistant_message(self, message: AIMessage):
        self.chat_history.add_ai_message(message)

    def get_messages(self):
        if self.chat_history.messages:
            return self.chat_history.messages
        else:
            return []
