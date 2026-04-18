import spacy
from typing import Dict, List, Any, Optional

from app.core.logging import logger
from app.core.exceptions import ModelNotLoadedError
from app.models.responses import Suggestion
from matchers.pipeline import NLPPipeline
from matchers.embeddings_matcher import EmbeddingsMatcher
from matchers.hybrid_matcher import HybridMatcher
from app.services.discipline_detector import DisciplineDetector
from app.services.text_cleaner import TextCleaner
from app.services.year_validator import YearValidator

class NLPService:
    """Serviço principal para processamento NLP"""
    
    def __init__(self):
        self.nlp = None
        self.pipeline = None
        self.embeddings_matcher = None
        self.hybrid_matcher = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo spaCy para português"""
        try:
            self.nlp = spacy.load("pt_core_news_lg")
            self.pipeline = NLPPipeline(self.nlp)
            
            try:
                self.embeddings_matcher = EmbeddingsMatcher()
                
                if self.embeddings_matcher.embeddings_cache:
                    self.hybrid_matcher = HybridMatcher(self.nlp)
                else:
                    self.embeddings_matcher = None
                    
            except Exception as e:
                logger.warning(f"Embeddings matcher não disponível: {str(e)}")
                self.embeddings_matcher = None
            
        except OSError:
            try:
                self.nlp = spacy.load("pt_core_news_sm")
                self.pipeline = NLPPipeline(self.nlp)
                
            except OSError:
                logger.error("Nenhum modelo spaCy encontrado. Execute: python -m spacy download pt_core_news_lg")
                self.nlp = None
                self.pipeline = None
    
    def is_loaded(self) -> bool:
        """Verifica se o modelo foi carregado"""
        return self.nlp is not None and self.pipeline is not None
    
    def process(self, text: str, context: Optional[Dict[str, Any]] = None, method: str = "keywords") -> Dict[str, Any]:
        """
        Processa o texto e extrai informações educacionais usando a pipeline modular
        
        Args:
            text: Texto de entrada para análise
            context: Contexto adicional (opcional)
            method: Método de matching - "keywords", "embeddings" ou "hybrid"
        
        Returns:
            Dicionário com campos extraídos, confiança, sugestões e campos faltantes
        
        Raises:
            ModelNotLoadedError: Se o modelo NLP não estiver carregado
        """
        if not self.is_loaded():
            raise ModelNotLoadedError("Modelo NLP não está disponível")
        
        if method == "embeddings":
            if not self.embeddings_matcher:
                logger.warning("Embeddings não disponível, usando keywords")
                result = self.pipeline.classify(text, context)
            else:
                result = self._process_with_embeddings(text, context)
        
        elif method == "hybrid":
            if not self.hybrid_matcher:
                logger.warning("Hybrid não disponível, usando keywords")
                result = self.pipeline.classify(text, context)
            else:
                result = self._process_with_hybrid(text, context)
        
        else:
            result = self.pipeline.classify(text, context)
        
        return result
    
    def _process_with_embeddings(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa usando APENAS embeddings (100% Sentence Transformers, SEM SPACY)"""
        extracted, confidence, suggestions = self._initialize_result(context)
        
        embeddings_result = self.embeddings_matcher.search_with_ensemble(text, top_k=5)
        
        if embeddings_result:
            for field in ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']:
                if field in embeddings_result and embeddings_result[field]:
                    if field not in extracted:
                        extracted[field] = embeddings_result[field]
                        confidence[field] = embeddings_result['confidence'][field]
        
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
    
    def _process_with_hybrid(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa usando método híbrido com pesos calibrados
        
        REGRAS DE PRIORIDADE:
        - Keywords SEMPRE vence: ano, disciplina, tipoQuestao
        - Embeddings pode vencer: unidadeTematica, objetoConhecimento, habilidade
        - Pesos: keywords=0.6, embeddings=0.35, sintático=0.05
        """
        
        extracted = {}
        confidence = {}
        suggestions = []
        missing_fields = []
        
        if context:
            for key, value in context.items():
                if value:
                    extracted[key] = value
                    confidence[key] = 1.0
        
        keywords_result = self.pipeline.classify(text, context)
        
        import re
        text_lower = text.lower()
        disciplina_hint = None
        disciplina_confidence = 0.0
        
        math_terms = ['função', 'funcao', 'funções', 'funcoes', 'equação', 'equacao', 'equações', 'equacoes',
                      'álgebra', 'algebra', 'geometria', 'trigonometria', 'probabilidade', 'estatística',
                      'fração', 'fracao', 'frações', 'fracoes', 'matemática', 'matematica']
        if any(term in text_lower for term in math_terms):
            disciplina_hint = 'Matemática'
            disciplina_confidence = 0.90
        elif any(term in text_lower for term in ['algoritmo', 'programação', 'programacao', 'código', 'codigo', 
                                                   'variável', 'variavel', 'booleano', 'computação', 'computacao']):
            disciplina_hint = 'Computação'
            disciplina_confidence = 0.90
        elif any(term in text_lower for term in ['texto', 'leitura', 'escrita', 'modalização', 'modalizacao',
                                                   'português', 'portugues', 'portuguesa', 'gramatica', 'gramática']):
            disciplina_hint = 'Língua Portuguesa'
            disciplina_confidence = 0.85
        elif any(term in text_lower for term in ['história', 'historia', 'histórico', 'historico', 'vargas',
                                                   'república', 'republica', 'ditadura', 'guerra']):
            disciplina_hint = 'História'
            disciplina_confidence = 0.85
        elif any(term in text_lower for term in ['ciências', 'ciencias', 'científico', 'cientifico', 'biologia',
                                                   'física', 'fisica', 'química', 'quimica']):
            disciplina_hint = 'Ciências'
            disciplina_confidence = 0.85
        
        instruction_words = {
            'quero', 'crie', 'elabore', 'faça', 'gere', 'desenvolva', 'construa',
            'questões', 'questoes', 'questão', 'questao', 
            'estilo', 'tipo', 'tipos', 'formato',
            'enem', 'prova', 'exame', 'teste', 'avaliação', 'avaliacao',
            'alunos', 'aluno', 'estudantes', 'estudante',
            'para', 'sobre', 'com', 'de', 'do', 'da', 'dos', 'das', 'uma', 'um', 'o', 'a', 'as', 'os'
        }
        
        cleaned_text = text.lower()
        cleaned_text = re.sub(r'\d+[ºª°]\s*ano', '', cleaned_text)
        cleaned_text = re.sub(r'(primeiro|segundo|terceiro|quarto|quinto|sexto|sétimo|setimo|oitavo|nono)\s*ano', '', cleaned_text)
        
        words = cleaned_text.split()
        cleaned_words = [w for w in words if w not in instruction_words and len(w) > 2]
        search_text = ' '.join(cleaned_words) if cleaned_words else text
        
        embeddings_result = self.embeddings_matcher.search_global(search_text, disciplina=disciplina_hint) if self.embeddings_matcher else None
        
        keywords_priority_fields = ['ano', 'tipoQuestao']
        for field in keywords_priority_fields:
            if field in keywords_result['extracted'] and field not in extracted:
                extracted[field] = keywords_result['extracted'][field]
                confidence[field] = keywords_result['confidence'][field]
        
        if 'disciplina' not in extracted:
            if disciplina_hint:
                extracted['disciplina'] = disciplina_hint
                confidence['disciplina'] = disciplina_confidence
            elif 'disciplina' in keywords_result['extracted']:
                extracted['disciplina'] = keywords_result['extracted']['disciplina']
                confidence['disciplina'] = keywords_result['confidence']['disciplina']
            elif embeddings_result and 'disciplina' in embeddings_result:
                extracted['disciplina'] = embeddings_result['disciplina']
                confidence['disciplina'] = embeddings_result['confidence']['disciplina']

        other_non_bncc = ['nivelBloom', 'tipoTextoBase', 'perfilAluno']
        for field in other_non_bncc:
            if field in keywords_result['extracted'] and field not in extracted:
                extracted[field] = keywords_result['extracted'][field]
                confidence[field] = keywords_result['confidence'][field]
        
        bncc_fields = ['unidadeTematica', 'objetoConhecimento', 'habilidade']
        
        if embeddings_result:
            ano_extraido = extracted.get('ano')
            ano_embedding = embeddings_result.get('ano')
            valid_embedding = True
            
            if ano_extraido and ano_embedding:
                ano_map = {'1º': 1, '2º': 2, '3º': 3, '4º': 4, '5º': 5, '6º': 6, '7º': 7, '8º': 8, '9º': 9}
                ano_num_extraido = ano_map.get(ano_extraido, 0)
                ano_num_embedding = ano_map.get(ano_embedding, 0)
                
                if abs(ano_num_embedding - ano_num_extraido) > 2:
                    logger.warning(f"⚠️  Embedding bloqueado: ano {ano_embedding} vs {ano_extraido} (diferença > 2)")
                    valid_embedding = False
            
            if valid_embedding and disciplina_hint:
                disciplina_embedding = embeddings_result.get('disciplina')
                if disciplina_embedding != disciplina_hint:
                    logger.warning(f"⚠️  Embedding bloqueado: disciplina {disciplina_embedding} vs {disciplina_hint}")
                    valid_embedding = False
            
            if valid_embedding:
                for field in bncc_fields:
                    if field not in extracted:
                        conf_keywords = keywords_result['confidence'].get(field, 0)
                        conf_embeddings = embeddings_result['confidence'].get(field, 0)
                        
                        weighted_keywords = conf_keywords * 0.6
                        weighted_embeddings = conf_embeddings * 0.35
                        
                        if weighted_embeddings > weighted_keywords and field in embeddings_result:
                            extracted[field] = embeddings_result[field]
                            confidence[field] = min(0.85, conf_embeddings)
                        elif field in keywords_result['extracted']:
                            extracted[field] = keywords_result['extracted'][field]
                            confidence[field] = min(0.85, conf_keywords)
            else:
                for field in bncc_fields:
                    if field in keywords_result['extracted'] and field not in extracted:
                        extracted[field] = keywords_result['extracted'][field]
                        confidence[field] = keywords_result['confidence'][field]
        else:
            for field in bncc_fields:
                if field in keywords_result['extracted'] and field not in extracted:
                    extracted[field] = keywords_result['extracted'][field]
                    confidence[field] = keywords_result['confidence'][field]
        
        if 'unidadeTematica' not in extracted or confidence.get('unidadeTematica', 0) < 0.5:
            if 'disciplina' in extracted:
                fallback_map = {
                    'Matemática': ('Álgebra', 0.40),
                    'História': ('História do Brasil', 0.40),
                    'Língua Portuguesa': ('Leitura', 0.40),
                    'Ciências': ('Vida e evolução', 0.40),
                    'Computação': ('Pensamento computacional', 0.40)
                }
                if extracted['disciplina'] in fallback_map:
                    fallback_unidade, fallback_conf = fallback_map[extracted['disciplina']]
                    extracted['unidadeTematica'] = fallback_unidade
                    confidence['unidadeTematica'] = fallback_conf
        
        ambiguous_fields = []
        for field, conf in confidence.items():
            if conf <= 0.7 and field in ['disciplina', 'unidadeTematica', 'objetoConhecimento', 'habilidade']:
                ambiguous_fields.append(field)
        
        if ambiguous_fields:
            logger.warning(f"⚠️  Campos ambíguos (conf ≤ 0.7): {ambiguous_fields}")
            from app.models.responses import Suggestion
            suggestions.append(Suggestion(
                field="ambiguidade",
                values=ambiguous_fields,
                message=f"Campos com baixa confiança (≤ 0.7): {', '.join(ambiguous_fields)}"
            ))
        
        all_fields = [
            "disciplina", "ano", "perfilAluno",
            "unidadeTematica", "objetoConhecimento", "habilidade",
            "nivelBloom", "tipoQuestao", "tipoTextoBase"
        ]
        missing_fields = [
            field for field in all_fields
            if field not in extracted or confidence.get(field, 0) < 0.5
        ]
        
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
    
    def _copy_non_bncc_fields(
        self,
        source_result: Dict[str, Any],
        extracted: Dict[str, Any],
        confidence: Dict[str, float]
    ) -> None:
        """Copia campos não-BNCC do resultado fonte"""
        non_bncc_fields = ['ano', 'nivelBloom', 'tipoQuestao', 'tipoTextoBase', 'perfilAluno']
        for field in non_bncc_fields:
            if field in source_result['extracted'] and field not in extracted:
                extracted[field] = source_result['extracted'][field]
                confidence[field] = source_result['confidence'][field]
    
    def _validate_embeddings_result(
        self,
        embeddings_result: Dict[str, Any],
        year_extracted: Optional[str],
        discipline_hint: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Valida resultado de embeddings (progressão pedagógica e disciplina)"""
        year_embedding = embeddings_result.get('ano')
        if not YearValidator.is_valid_progression(year_extracted, year_embedding, max_difference=2):
            return None
        
        if discipline_hint:
            discipline_embedding = embeddings_result.get('disciplina')
            if discipline_embedding != discipline_hint:
                return None
        
        return embeddings_result
    
    def _apply_embeddings_result(
        self,
        embeddings_result: Dict[str, Any],
        extracted: Dict[str, Any],
        confidence: Dict[str, float]
    ) -> None:
        """Aplica resultado de embeddings aos campos extraídos"""
        bncc_fields = ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']
        for field in bncc_fields:
            if field in embeddings_result and embeddings_result[field]:
                if field not in extracted or confidence.get(field, 0) < 0.85:
                    extracted[field] = embeddings_result[field]
                    confidence[field] = embeddings_result['confidence'][field]
    
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
