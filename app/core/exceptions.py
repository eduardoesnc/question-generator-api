"""
Exceções customizadas da aplicação
"""
from typing import Optional, Dict, Any


class NLPAPIException(Exception):
    """Exceção base da API"""
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(NLPAPIException):
    """Modelo NLP não foi carregado"""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Modelo '{model_name}' não foi carregado",
            status_code=503,
            details={"model": model_name}
        )


class BNCCDataNotFoundError(NLPAPIException):
    """Dados da BNCC não encontrados"""
    
    def __init__(self, path: str):
        super().__init__(
            message=f"Arquivo BNCC não encontrado: {path}",
            status_code=500,
            details={"path": path}
        )


class EmbeddingsNotAvailableError(NLPAPIException):
    """Embeddings não estão disponíveis"""
    
    def __init__(self):
        super().__init__(
            message="Embeddings não disponíveis. Execute: python scripts/generate_embeddings.py",
            status_code=503,
            details={"solution": "Run generate_embeddings.py script"}
        )


class InvalidTextInputError(NLPAPIException):
    """Texto de entrada inválido"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"Texto inválido: {reason}",
            status_code=400,
            details={"reason": reason}
        )


class ExtractionFailedError(NLPAPIException):
    """Falha na extração de informações"""
    
    def __init__(self, text: str, error: str):
        super().__init__(
            message=f"Falha ao extrair informações do texto",
            status_code=500,
            details={"text": text[:100], "error": error}
        )


class MatcherNotFoundError(NLPAPIException):
    """Matcher não encontrado"""
    
    def __init__(self, matcher_type: str):
        super().__init__(
            message=f"Matcher '{matcher_type}' não encontrado ou não inicializado",
            status_code=503,
            details={"matcher_type": matcher_type}
        )


class ProcessingError(NLPAPIException):
    """Erro genérico durante processamento"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details or {}
        )
