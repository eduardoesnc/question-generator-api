from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import uvicorn
import os
from dotenv import load_dotenv
from nlp_processor import NLPProcessor

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI(
    title="API NLP - Gerador de Prompts Educacionais",
    description="API para extrair informações educacionais de texto livre",
    version="1.0.0"
)

# CORS - configurável via env
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001")
origins_list = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar processador NLP
nlp_processor = NLPProcessor()


class TextInput(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None


class ExtractionResponse(BaseModel):
    extracted: Dict[str, Any]
    confidence: Dict[str, float]
    suggestions: List[Dict[str, Any]]
    missing_fields: List[str]
    original_text: str


@app.get("/")
async def root():
    return {
        "message": "API NLP - Gerador de Prompts Educacionais",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "nlp_model_loaded": nlp_processor.is_loaded()
    }


@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_information(input_data: TextInput):
    """
    Extrai informações educacionais de texto livre.
    
    Campos extraídos:
    - disciplina: Matéria escolar
    - ano: Ano/série escolar
    - unidadeTematica: Tópico principal
    - objetoConhecimento: Assunto específico
    - nivelBloom: Nível de dificuldade cognitiva
    - tipoQuestao: Formato da questão
    - tipoTextoBase: Tipo de texto de apoio
    - perfilAluno: Perfil do estudante
    """
    try:
        if not input_data.text or len(input_data.text.strip()) < 3:
            raise HTTPException(
                status_code=400,
                detail="Texto muito curto. Por favor, forneça mais informações."
            )
        
        result = nlp_processor.process(input_data.text, input_data.context)
        
        return ExtractionResponse(
            extracted=result["extracted"],
            confidence=result["confidence"],
            suggestions=result["suggestions"],
            missing_fields=result["missing_fields"],
            original_text=input_data.text
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar texto: {str(e)}"
        )


if __name__ == "__main__":
    # Configurações via env
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    environment = os.getenv("ENVIRONMENT", "development")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=(environment == "development")
    )
