"""
Request models
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, Literal


class ExtractionRequest(BaseModel):
    """Request para extração de informações"""
    
    text: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Texto para extrair informações educacionais"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexto adicional (opcional)"
    )
    
    method: Literal['embeddings'] = Field(
        default='embeddings',
        description="Método de extração: embeddings (similaridade semântica com modelo fine-tuned)"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Valida texto de entrada"""
        if not v or not v.strip():
            raise ValueError("Texto não pode ser vazio")
        
        if len(v.strip()) < 3:
            raise ValueError("Texto muito curto (mínimo 3 caracteres)")
        
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "História sobre Era Vargas, 9º ano, análise",
                "method": "embeddings"
            }
        }
