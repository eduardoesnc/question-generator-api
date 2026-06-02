"""
Configurações centralizadas da aplicação
"""
import os
from typing import List, Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configurações da aplicação"""
    
    # API
    APP_NAME: str = "NLP Educational API"
    APP_VERSION: str = "2.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ENVIRONMENT: str = "development"
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,https://question-generator-seven-delta.vercel.app"
    
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    MODELS_DIR: str = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

    # Model source: "local" | "minio" | "s3" | "huggingface"
    MODEL_SOURCE: str = "local"

    # MinIO / S3 (usado quando MODEL_SOURCE=minio ou s3)
    MINIO_ENDPOINT: str = ""
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "ml-models"
    MINIO_MODEL_PATH: str = "bncc-embeddings-finetuned"

    # HuggingFace (usado quando MODEL_SOURCE=huggingface)
    HF_MODEL_ID: str = ""

    # Gemini (usado pelo modo LLM)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"

    @property
    def bncc_data_path(self) -> str:
        return os.path.join(self.DATA_DIR, "bncc-data.json")
    
    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.DATA_DIR, "bncc_embeddings.json")
    
    @property
    def sentence_transformer_model(self) -> str:
        return os.path.join(self.MODELS_DIR, "bncc-embeddings-finetuned")
    
    # Compatibilidade com código existente
    @property
    def BNCC_DATA_PATH(self) -> str:
        return self.bncc_data_path
    
    @property
    def EMBEDDINGS_PATH(self) -> str:
        return self.embeddings_path
    
    @property
    def SENTENCE_TRANSFORMER_MODEL(self) -> str:
        return self.sentence_transformer_model
    
    # Thresholds
    EMBEDDINGS_THRESHOLD: float = 0.30
    CONFIDENCE_MIN: float = 0.50
    CONFIDENCE_MAX: float = 0.95
    
    # Performance
    MAX_TEXT_LENGTH: int = 1000
    SHORT_TEXT_WORDS: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def api_host(self) -> str:
        return self.API_HOST
    
    @property
    def api_port(self) -> int:
        return self.API_PORT
    
    @property
    def environment(self) -> str:
        return self.ENVIRONMENT
    
    @property
    def cors_origins(self) -> List[str]:
        return self.cors_origins_list


# Instância global
settings = Settings()
