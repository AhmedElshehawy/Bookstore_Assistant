from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables.
    
    Attributes:
        OPENAI_API_KEY: OpenAI API key for authentication
        DB_URL: Database connection string
        APP_NAME: Name of the application
        API_PREFIX: Prefix for all API endpoints
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # API Keys and External Services
    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    DB_URL: str
    
    # Application Settings
    APP_NAME: str = "FastAPI App"
    API_PREFIX: str = "/api/v1"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Chatbot
    CHATBOT_PROMPT_PATH: str
    TEXT_TO_SQL_PROMPT_PATH: str
    IS_SAFE_SQL_PROMPT_PATH: str
       
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()
