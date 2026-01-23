"""
Configurações centralizadas da aplicação
"""
import os
from typing import List
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
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    BNCC_DATA_PATH: str = os.path.join(DATA_DIR, "bncc-data.json")
    EMBEDDINGS_PATH: str = os.path.join(DATA_DIR, "bncc_embeddings.json")
    
    # NLP Models
    SPACY_MODEL: str = "pt_core_news_sm"
    SENTENCE_TRANSFORMER_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"
    
    # Thresholds
    KEYWORDS_THRESHOLD: float = 0.20
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
        """Retorna lista de origens CORS"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def api_host(self) -> str:
        """Retorna host da API"""
        return self.API_HOST
    
    @property
    def api_port(self) -> int:
        """Retorna porta da API"""
        return self.API_PORT
    
    @property
    def environment(self) -> str:
        """Retorna ambiente"""
        return self.ENVIRONMENT
    
    @property
    def cors_origins(self) -> List[str]:
        """Alias para cors_origins_list"""
        return self.cors_origins_list


# Instância global
settings = Settings()
