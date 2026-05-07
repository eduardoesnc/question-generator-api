import os
from typing import Dict, List, Any, Optional

from app.core.logging import logger
from app.core.exceptions import ModelNotLoadedError
from app.models.responses import Suggestion
from app.config import settings
from matchers.embeddings_matcher import EmbeddingsMatcher

class NLPService:
    """Serviço principal para processamento NLP usando apenas embeddings"""
    
    def __init__(self):
        self.embeddings_matcher = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo de embeddings de acordo com MODEL_SOURCE"""
        try:
            model_path = self._resolve_model_path()
            logger.info(f"Carregando modelo de embeddings de: {model_path}")
            self.embeddings_matcher = EmbeddingsMatcher(model_path=model_path)
            
            if self.embeddings_matcher.embeddings_cache:
                logger.success("Modelo de embeddings carregado com sucesso!")
            else:
                logger.error("Cache de embeddings vazio")
                self.embeddings_matcher = None
                    
        except Exception as e:
            logger.error(f"Erro ao carregar embeddings: {str(e)}")
            self.embeddings_matcher = None

    def _resolve_model_path(self) -> str:
        """Resolve o caminho do modelo de acordo com MODEL_SOURCE"""
        source = settings.MODEL_SOURCE.lower()

        if source == "local":
            return settings.sentence_transformer_model

        elif source == "minio":
            return self._download_from_minio()

        elif source == "s3":
            return self._download_from_s3()

        elif source == "huggingface":
            return settings.HF_MODEL_ID  # sentence-transformers aceita HF model ID direto

        else:
            logger.warning(f"MODEL_SOURCE '{source}' desconhecido, usando local")
            return settings.sentence_transformer_model

    def _download_from_minio(self) -> str:
        """Baixa modelo do MinIO para MODELS_DIR e retorna o caminho local"""
        try:
            from minio import Minio

            dest = settings.sentence_transformer_model
            if os.path.exists(dest):
                logger.info("Modelo já existe localmente, pulando download")
                return dest

            logger.info(f"Baixando modelo do MinIO: {settings.MINIO_BUCKET}/{settings.MINIO_MODEL_PATH}")
            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=True,
            )
            objects = client.list_objects(settings.MINIO_BUCKET, prefix=settings.MINIO_MODEL_PATH, recursive=True)
            for obj in objects:
                local_file = os.path.join(settings.MODELS_DIR, obj.object_name)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                client.fget_object(settings.MINIO_BUCKET, obj.object_name, local_file)

            logger.success("Modelo baixado do MinIO com sucesso!")
            return dest

        except ImportError:
            raise RuntimeError("Pacote 'minio' não instalado. Adicione ao requirements.txt: minio")
        except Exception as e:
            raise RuntimeError(f"Erro ao baixar modelo do MinIO: {e}")

    def _download_from_s3(self) -> str:
        """Baixa modelo do S3 para MODELS_DIR e retorna o caminho local"""
        try:
            import boto3

            dest = settings.sentence_transformer_model
            if os.path.exists(dest):
                logger.info("Modelo já existe localmente, pulando download")
                return dest

            logger.info(f"Baixando modelo do S3: {settings.MINIO_BUCKET}/{settings.MINIO_MODEL_PATH}")
            s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.MINIO_ACCESS_KEY,
                aws_secret_access_key=settings.MINIO_SECRET_KEY,
            )
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=settings.MINIO_BUCKET, Prefix=settings.MINIO_MODEL_PATH):
                for obj in page.get("Contents", []):
                    local_file = os.path.join(settings.MODELS_DIR, obj["Key"])
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    s3.download_file(settings.MINIO_BUCKET, obj["Key"], local_file)

            logger.success("Modelo baixado do S3 com sucesso!")
            return dest

        except ImportError:
            raise RuntimeError("Pacote 'boto3' não instalado. Adicione ao requirements.txt: boto3")
        except Exception as e:
            raise RuntimeError(f"Erro ao baixar modelo do S3: {e}")
    
    def is_loaded(self) -> bool:
        """Verifica se o modelo foi carregado"""
        return self.embeddings_matcher is not None
    
    def process(self, text: str, context: Optional[Dict[str, Any]] = None, method: str = "embeddings") -> Dict[str, Any]:
        """
        Processa o texto e extrai informações educacionais usando embeddings
        
        Args:
            text: Texto de entrada para análise
            context: Contexto adicional (opcional)
            method: Método de matching - apenas "embeddings" é suportado
        
        Returns:
            Dicionário com campos extraídos, confiança, sugestões e campos faltantes
        
        Raises:
            ModelNotLoadedError: Se o modelo NLP não estiver carregado
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Modelo de embeddings não está disponível")
        
        # Apenas embeddings é suportado agora
        if method != "embeddings":
            logger.warning(f"Método '{method}' não suportado. Usando 'embeddings'")
        
        return self._process_with_embeddings(text, context)
    
    def _process_with_embeddings(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa usando embeddings (Sentence Transformers fine-tuned)"""
        extracted, confidence, suggestions = self._initialize_result(context)
        
        # Busca global com ensemble de embeddings
        embeddings_result = self.embeddings_matcher.search_with_ensemble(text, top_k=5)
        
        if embeddings_result:
            for field in ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']:
                if field in embeddings_result and embeddings_result[field]:
                    if field not in extracted:
                        extracted[field] = embeddings_result[field]
                        confidence[field] = embeddings_result['confidence'][field]
        
        # Inferir campos não-BNCC
        bloom_result = self.embeddings_matcher.infer_bloom(text)
        if bloom_result and 'nivelBloom' not in extracted:
            extracted['nivelBloom'] = bloom_result[0]
            confidence['nivelBloom'] = bloom_result[1]
        
        perfil_result = self.embeddings_matcher.infer_perfil_aluno(text)
        if perfil_result and 'perfilAluno' not in extracted:
            extracted['perfilAluno'] = perfil_result[0]
            confidence['perfilAluno'] = perfil_result[1]
        
        tipo_questao_result = self.embeddings_matcher.infer_tipo_questao(text)
        if tipo_questao_result and 'tipoQuestao' not in extracted:
            extracted['tipoQuestao'] = tipo_questao_result[0]
            confidence['tipoQuestao'] = tipo_questao_result[1]
        
        tipo_texto_result = self.embeddings_matcher.infer_tipo_texto_base(text)
        if tipo_texto_result and 'tipoTextoBase' not in extracted:
            extracted['tipoTextoBase'] = tipo_texto_result[0]
            confidence['tipoTextoBase'] = tipo_texto_result[1]
        
        missing_fields = self._get_missing_fields(extracted, confidence)
        
        return {
            "extracted": extracted,
            "confidence": confidence,
            "suggestions": suggestions,
            "missing_fields": missing_fields
        }
    
    def _initialize_result(self, context: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Inicializa resultado com contexto bloqueado
        
        Returns:
            Tupla (extracted, confidence, suggestions)
        """
        extracted = {}
        confidence = {}
        suggestions = []
        
        if context:
            for key, value in context.items():
                if value:
                    extracted[key] = value
                    confidence[key] = 1.0
        
        return extracted, confidence, suggestions
    
    def _get_missing_fields(
        self,
        extracted: Dict[str, Any],
        confidence: Dict[str, float],
        threshold: float = 0.5
    ) -> List[str]:
        """Identifica campos faltantes ou com baixa confiança"""
        all_fields = [
            "disciplina", "ano", "perfilAluno",
            "unidadeTematica", "objetoConhecimento", "habilidade",
            "nivelBloom", "tipoQuestao", "tipoTextoBase"
        ]
        return [
            field for field in all_fields
            if field not in extracted or confidence.get(field, 0) < threshold
        ]

