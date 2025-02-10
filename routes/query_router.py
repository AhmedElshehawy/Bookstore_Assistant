from fastapi import APIRouter, HTTPException, status
from models import QueryRequest, QueryResponse
from core import setup_logger, settings
from services.chatbot import ChatbotService
from services.chat_history import ChatHistoryService


# Initialize router
query_router = APIRouter(
    tags=["query"],
    responses={404: {"description": "Not found"}},
)


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
    chatbot_service = ChatbotService()
    
    logger.info(f"Processing query request: {query_request}") 
    try:
        # Validate input
        if not query_request.user_query.strip():
            raise ValueError("User query cannot be empty")

        # Process query
        ai_response = await chatbot_service.chat(query_request.user_query, query_request.thread_id)
        
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
