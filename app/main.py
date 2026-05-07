from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

from app.core.logging import logger
from app.core.exceptions import ModelNotLoadedError, ProcessingError
from app.models.requests import ExtractionRequest
from app.models.responses import ExtractionResponse
from app.services.nlp_service import NLPService
from app.config import settings

load_dotenv()

app = FastAPI(
    title="API NLP - Gerador de Prompts Educacionais",
    description="API para extrair informações educacionais de texto livre usando NLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp_service = NLPService()

@app.on_event("startup")
async def startup_event():
    """Evento executado ao iniciar a aplicação"""
    logger.info("Iniciando API NLP...")
    logger.info(f"Ambiente: {settings.environment}")
    logger.info(f"CORS Origins: {settings.cors_origins}")
    
    if not nlp_service.is_loaded():
        logger.error("Modelo NLP não foi carregado!")
    else:
        logger.success("Modelo NLP carregado com sucesso!")

@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "API NLP - Gerador de Prompts Educacionais",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Verifica o status da API e do modelo NLP"""
    is_loaded = nlp_service.is_loaded()
    
    return {
        "status": "healthy" if is_loaded else "degraded",
        "nlp_model_loaded": is_loaded,
        "environment": settings.environment
    }

@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_information(input_data: ExtractionRequest):
    """
    Extrai informações educacionais de texto livre usando NLP com embeddings.
    
    ## Método:
    - **embeddings**: Similaridade semântica com modelo fine-tuned (~200ms)
    
    ## Campos Extraídos:
    - **disciplina**: Matéria escolar (Matemática, História, etc.)
    - **ano**: Ano/série escolar (1º ao 9º)
    - **unidadeTematica**: Tópico principal da BNCC
    - **objetoConhecimento**: Assunto específico da BNCC
    - **habilidade**: Código da habilidade BNCC
    - **nivelBloom**: Nível cognitivo (conhecimento, compreensão, aplicação, análise, síntese, avaliação)
    - **tipoQuestao**: Formato da questão (múltipla escolha, dissertativa, etc.)
    - **tipoTextoBase**: Tipo de texto de apoio (documento histórico, gráfico, etc.)
    - **perfilAluno**: Perfil do estudante (bom domínio, dificuldade, etc.)
    
    ## Exemplo de Entrada:
    ```json
    {
        "text": "História sobre Era Vargas, 9º ano, análise, dissertativa longa com documento histórico",
        "method": "embeddings"
    }
    ```
    
    ## Retorno:
    - **extracted**: Dicionário com os campos extraídos
    - **confidence**: Scores de confiança para cada campo (0.0 a 1.0)
    - **suggestions**: Sugestões de valores alternativos
    - **missing_fields**: Lista de campos que não foram extraídos
    - **original_text**: Texto original enviado
    """
    try:
        if not nlp_service.is_loaded():
            logger.error("Tentativa de processar texto sem modelo carregado")
            raise ModelNotLoadedError("Modelo NLP não está carregado")
        
        logger.info(f"Processando texto com método '{input_data.method}': '{input_data.text[:100]}...'")
        result = nlp_service.process(input_data.text, input_data.context, method=input_data.method)
        
        logger.success(f"Processamento concluído. Campos extraídos: {len(result['extracted'])}")
        
        return ExtractionResponse(
            extracted=result["extracted"],
            confidence=result["confidence"],
            suggestions=result["suggestions"],
            missing_fields=result["missing_fields"],
            original_text=input_data.text
        )
    
    except ModelNotLoadedError as e:
        logger.error(f"Erro: Modelo não carregado - {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    
    except ProcessingError as e:
        logger.error(f"Erro de processamento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Erro inesperado: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar texto: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=(settings.environment == "development")
    )
