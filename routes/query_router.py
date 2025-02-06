from fastapi import APIRouter, HTTPException, status
from models import QueryRequest, QueryResponse
from core import setup_logger, settings
from services.chatbot import ChatbotService
# Initialize router with prefix and tags for better API documentation
query_router = APIRouter(
    tags=["query"],
    responses={404: {"description": "Not found"}},
)

def get_config(thread_id: str) -> dict:
    """Get configuration for the chat service.
    
    Args:
        thread_id (str): The thread identifier
        
    Returns:
        dict: Configuration dictionary
    """
    if not thread_id:
        raise ValueError("thread_id cannot be empty")
    return {"configurable": {"thread_id": thread_id}}

chatbot_service = ChatbotService(settings.CHATBOT_PROMPT_PATH)
logger = setup_logger(__name__, settings.LOG_LEVEL)

@query_router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        500: {"description": "Internal server error"},
        400: {"description": "Bad request"},
    }
)
async def query_books(query_request: QueryRequest) -> QueryResponse:
    """Process a user query and return AI-generated response.
    
    Args:
        query_request (QueryRequest): The query request containing user query and thread ID
        
    Returns:
        QueryResponse: The AI-generated response
        
    Raises:
        HTTPException: If there's an error processing the request
    """
    logger.info(f"Processing query request: {query_request}")
    
    try:
        # Validate input
        if not query_request.user_query.strip():
            raise ValueError("User query cannot be empty")
            
        # Get configuration
        config = get_config(query_request.thread_id)
        
        # Process query
        ai_response = await chatbot_service.chat(query_request.user_query, config)
        
        logger.info(f"Successfully processed query. Response: {ai_response.content}")
        return QueryResponse(ai_response=ai_response.content)
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )
