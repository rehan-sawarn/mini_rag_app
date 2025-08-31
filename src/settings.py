import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# LOAD VARIABLES FROM .env
load_dotenv()

class Settings(BaseSettings):
    # QDRANT
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str

    # LLM / EMBEDDINGS
    OPENAI_API_KEY: str
    COHERE_API_KEY: str
    GROQ_API_KEY: str

    EMBEDDING_DIM: int = 384

    class Config:
        case_sensitive = True

# INSTANTIATE SETTINGS
settings = Settings()
