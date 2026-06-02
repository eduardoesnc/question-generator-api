"""
Response models
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class Suggestion(BaseModel):
    """Sugestão de valor para um campo"""
    field: str
    values: List[str]
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "field": "unidadeTematica",
                "values": ["Álgebra", "Geometria"],
                "message": "Possíveis unidades temáticas"
            }
        }


class ExtractionResponse(BaseModel):
    """Response da extração de informações"""
    
    extracted: Dict[str, Any] = Field(
        ...,
        description="Campos extraídos do texto"
    )
    
    confidence: Dict[str, float] = Field(
        ...,
        description="Confiança de cada campo (0.0 a 1.0)"
    )
    
    suggestions: List[Suggestion] = Field(
        default=[],
        description="Sugestões de valores alternativos"
    )
    
    missing_fields: List[str] = Field(
        default=[],
        description="Campos que não foram extraídos"
    )
    
    original_text: str = Field(
        ...,
        description="Texto original enviado"
    )
    
    method_used: Optional[str] = Field(
        default=None,
        description="Método usado para extração"
    )
    
    similarity_score: Optional[float] = Field(
        default=None,
        description="Score de similaridade (para embeddings)"
    )
    
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Tempo de processamento em milissegundos"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "extracted": {
                    "disciplina": "História",
                    "ano": "9º",
                    "nivelBloom": "analise"
                },
                "confidence": {
                    "disciplina": 0.85,
                    "ano": 0.95,
                    "nivelBloom": 0.82
                },
                "suggestions": [],
                "missing_fields": ["tipoQuestao", "tipoTextoBase"],
                "original_text": "História sobre Era Vargas, 9º ano, análise",
                "method_used": "hybrid",
                "processing_time_ms": 125.5
            }
        }


class TokenUsage(BaseModel):
    """Uso de tokens de uma chamada ao Gemini"""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LLMTokenSummary(BaseModel):
    """Resumo de tokens das duas chamadas ao Gemini"""
    call_1: TokenUsage = Field(..., description="Tokens da chamada 1 (extração de campos)")
    call_2: TokenUsage = Field(..., description="Tokens da chamada 2 (seleção BNCC + geração)")
    total_input: int = Field(..., description="Total de tokens de entrada")
    total_output: int = Field(..., description="Total de tokens de saída")
    total: int = Field(..., description="Total geral de tokens")


class LLMGenerationResponse(BaseModel):
    """Response do modo LLM (Gemini)"""

    filled_prompt: str = Field(..., description="Prompt padrão preenchido com os campos extraídos")
    extracted_fields: Dict[str, Any] = Field(..., description="Campos extraídos pelo Gemini")
    generated_question: str = Field(..., description="Questão gerada pelo Gemini (JSON bruto)")
    original_text: str = Field(..., description="Texto original enviado")
    processing_time_ms: float = Field(..., description="Tempo total de processamento em ms")
    token_usage: Optional[LLMTokenSummary] = Field(default=None, description="Consumo de tokens por chamada")


class HealthResponse(BaseModel):
    """Response do health check"""
    
    status: str = Field(..., description="Status da API")
    model_loaded: bool = Field(..., description="Modelo spaCy carregado")
    embeddings_available: bool = Field(..., description="Embeddings disponíveis")
    hybrid_available: bool = Field(..., description="Matcher híbrido disponível")
    version: str = Field(..., description="Versão da API")
