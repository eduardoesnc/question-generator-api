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
    
    # Paths (podem ser sobrescritos via ENV)
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    MODELS_DIR: str = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))
    
    @property
    def bncc_data_path(self) -> str:
        return os.path.join(self.DATA_DIR, "bncc-data.json")
    
    @property
    def embeddings_path(self) -> str:
        return os.path.join(self.DATA_DIR, "bncc_embeddings.json")
    
    # NLP Models (apenas embeddings)
    @property
    def sentence_transformer_model(self) -> str:
        return os.path.join(self.MODELS_DIR, "bncc-embeddings-finetuned")
    
    # Manter compatibilidade com código antigo
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
