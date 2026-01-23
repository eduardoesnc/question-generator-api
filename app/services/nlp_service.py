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
        print("="*80)
        print("🚀 INICIANDO CARREGAMENTO DE MODELOS NLP")
        print("="*80)
        
        try:
            logger.info("📦 Carregando modelo spaCy pt_core_news_lg...")
            print("📦 Carregando modelo spaCy pt_core_news_lg...")
            self.nlp = spacy.load("pt_core_news_lg")
            self.pipeline = NLPPipeline(self.nlp)
            logger.success("✅ Modelo pt_core_news_lg carregado com sucesso!")
            print("✅ Modelo pt_core_news_lg carregado!")
            
            # Tentar carregar embeddings matcher
            try:
                logger.info("📦 Carregando embeddings matcher...")
                print("📦 Carregando embeddings matcher...")
                self.embeddings_matcher = EmbeddingsMatcher()
                
                if self.embeddings_matcher.embeddings_cache:
                    logger.success(f"✅ Embeddings matcher carregado! ({len(self.embeddings_matcher.embeddings_cache)} objetos)")
                    print(f"✅ Embeddings matcher carregado! ({len(self.embeddings_matcher.embeddings_cache)} objetos)")
                    self.hybrid_matcher = HybridMatcher(self.nlp)
                    logger.success("✅ Hybrid matcher carregado!")
                    print("✅ Hybrid matcher carregado!")
                else:
                    logger.warning("⚠️  Embeddings cache vazio!")
                    print("⚠️  Embeddings cache vazio!")
                    self.embeddings_matcher = None
                    
            except Exception as e:
                logger.warning(f"⚠️  Embeddings matcher não disponível: {str(e)}")
                print(f"⚠️  Embeddings matcher não disponível: {str(e)}")
                logger.warning("Execute: python scripts/generate_embeddings.py")
                self.embeddings_matcher = None
            
        except OSError:
            logger.warning("⚠️  Modelo pt_core_news_lg não encontrado. Tentando modelo menor...")
            print("⚠️  Modelo pt_core_news_lg não encontrado. Tentando modelo menor...")
            try:
                self.nlp = spacy.load("pt_core_news_sm")
                self.pipeline = NLPPipeline(self.nlp)
                logger.success("✅ Modelo pt_core_news_sm carregado com sucesso!")
                print("✅ Modelo pt_core_news_sm carregado!")
                
            except OSError:
                logger.error("❌ Nenhum modelo spaCy encontrado!")
                print("❌ Nenhum modelo spaCy encontrado!")
                logger.error("Execute: python -m spacy download pt_core_news_lg")
                self.nlp = None
                self.pipeline = None
        
        print("="*80)
        print("✅ CARREGAMENTO CONCLUÍDO")
        print("="*80)
    
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
            logger.error("Tentativa de processar texto sem modelo carregado")
            raise ModelNotLoadedError("Modelo NLP não está disponível")
        
        logger.debug(f"Processando texto com método '{method}': '{text[:100]}...'")
        
        # Escolher método
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
        
        else:  # keywords (padrão)
            result = self.pipeline.classify(text, context)
        
        return result
    
    def _process_with_embeddings(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa usando apenas embeddings"""
        logger.info("🧠 Processando com embeddings...")
        
        # Inicializar resultado
        extracted, confidence, suggestions = self._initialize_result(context)
        
        # Extrair campos não-BNCC com keywords
        temp_result = self.pipeline.classify(text, context)
        self._copy_non_bncc_fields(temp_result, extracted, confidence)
        
        # Detectar disciplina por termos fortes
        disciplina_hint, disciplina_confidence = DisciplineDetector.detect(text)
        if disciplina_hint:
            logger.debug(f"🎯 Disciplina detectada por termo forte: {disciplina_hint}")
        
        # Limpar texto para embeddings
        search_text = TextCleaner.clean_for_embeddings(text)
        logger.debug(f"📝 Texto original: '{text}'")
        logger.debug(f"🧹 Texto limpo para embeddings: '{search_text}'")
        
        # Buscar com embeddings
        embeddings_result = self.embeddings_matcher.search_global(search_text, disciplina=disciplina_hint, ano=None)
        
        # Validar e aplicar resultados de embeddings
        if embeddings_result:
            embeddings_result = self._validate_embeddings_result(
                embeddings_result,
                extracted.get('ano'),
                disciplina_hint
            )
        
        if embeddings_result:
            self._apply_embeddings_result(embeddings_result, extracted, confidence)
        
        # Aplicar disciplina detectada se não foi extraída
        if disciplina_hint and 'disciplina' not in extracted:
            extracted['disciplina'] = disciplina_hint
            confidence['disciplina'] = disciplina_confidence
        
        # Identificar campos faltantes
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
        logger.info("🔀 Processando com método híbrido...")
        
        # Inicializar resultado vazio
        extracted = {}
        confidence = {}
        suggestions = []
        missing_fields = []
        
        # REGRA HARD 1: Contexto explícito vence SEMPRE
        if context:
            for key, value in context.items():
                if value:
                    extracted[key] = value
                    confidence[key] = 1.0
                    logger.debug(f"🔒 Campo bloqueado por contexto: {key} = {value}")
        
        # Obter resultados de KEYWORDS
        keywords_result = self.pipeline.classify(text, context)
        
        # Obter resultados de EMBEDDINGS (usando mesma lógica de detecção)
        import re
        text_lower = text.lower()
        disciplina_hint = None
        disciplina_confidence = 0.0
        
        # Detectar disciplina por termos fortes
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
        
        # Limpar texto para embeddings
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
        
        # APLICAR REGRAS DE PRIORIDADE HÍBRIDA
        
        # 1. KEYWORDS SEMPRE VENCE: ano, disciplina, tipoQuestao
        keywords_priority_fields = ['ano', 'tipoQuestao']
        for field in keywords_priority_fields:
            if field in keywords_result['extracted'] and field not in extracted:
                extracted[field] = keywords_result['extracted'][field]
                confidence[field] = keywords_result['confidence'][field]
                logger.debug(f"✅ KEYWORDS vence: {field} = {extracted[field]}")
        
        # Disciplina: usar termo forte se detectado, senão keywords, senão embeddings
        if 'disciplina' not in extracted:
            if disciplina_hint:
                extracted['disciplina'] = disciplina_hint
                confidence['disciplina'] = disciplina_confidence
                logger.debug(f"✅ TERMO FORTE vence: disciplina = {disciplina_hint}")
            elif 'disciplina' in keywords_result['extracted']:
                extracted['disciplina'] = keywords_result['extracted']['disciplina']
                confidence['disciplina'] = keywords_result['confidence']['disciplina']
                logger.debug(f"✅ KEYWORDS vence: disciplina = {extracted['disciplina']}")
            elif embeddings_result and 'disciplina' in embeddings_result:
                extracted['disciplina'] = embeddings_result['disciplina']
                confidence['disciplina'] = embeddings_result['confidence']['disciplina']
                logger.debug(f"✅ EMBEDDINGS vence: disciplina = {extracted['disciplina']} (fallback)")

        
        # 2. OUTROS CAMPOS NÃO-BNCC: keywords (confiança original)
        other_non_bncc = ['nivelBloom', 'tipoTextoBase', 'perfilAluno']
        for field in other_non_bncc:
            if field in keywords_result['extracted'] and field not in extracted:
                extracted[field] = keywords_result['extracted'][field]
                confidence[field] = keywords_result['confidence'][field]  # Confiança original, não ponderada
        
        # 3. CAMPOS BNCC: comparar keywords vs embeddings
        bncc_fields = ['unidadeTematica', 'objetoConhecimento', 'habilidade']
        
        if embeddings_result:
            # Validar progressão pedagógica
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
            
            # Validar disciplina
            if valid_embedding and disciplina_hint:
                disciplina_embedding = embeddings_result.get('disciplina')
                if disciplina_embedding != disciplina_hint:
                    logger.warning(f"⚠️  Embedding bloqueado: disciplina {disciplina_embedding} vs {disciplina_hint}")
                    valid_embedding = False
            
            # Aplicar embeddings para campos BNCC se passou validação
            if valid_embedding:
                for field in bncc_fields:
                    if field not in extracted:  # Não sobrescrever contexto
                        conf_keywords = keywords_result['confidence'].get(field, 0)
                        conf_embeddings = embeddings_result['confidence'].get(field, 0)
                        
                        # Ponderar para comparação (decidir qual vence)
                        weighted_keywords = conf_keywords * 0.6
                        weighted_embeddings = conf_embeddings * 0.35
                        
                        # Comparar confiânças ponderadas para decidir vencedor
                        if weighted_embeddings > weighted_keywords and field in embeddings_result:
                            extracted[field] = embeddings_result[field]
                            # Usar confiança ORIGINAL do embeddings (não ponderada)
                            confidence[field] = min(0.85, conf_embeddings)
                            logger.debug(f"✅ EMBEDDINGS vence: {field} (weighted: {weighted_embeddings:.2f} > {weighted_keywords:.2f}, conf original: {conf_embeddings:.2f})")
                        elif field in keywords_result['extracted']:
                            extracted[field] = keywords_result['extracted'][field]
                            # Usar confiança ORIGINAL do keywords (não ponderada)
                            confidence[field] = min(0.85, conf_keywords)
                            logger.debug(f"✅ KEYWORDS vence: {field} (weighted: {weighted_keywords:.2f} >= {weighted_embeddings:.2f}, conf original: {conf_keywords:.2f})")
            else:
                # Embeddings bloqueado, usar apenas keywords
                for field in bncc_fields:
                    if field in keywords_result['extracted'] and field not in extracted:
                        extracted[field] = keywords_result['extracted'][field]
                        confidence[field] = keywords_result['confidence'][field]  # Confiança original
        else:
            # Sem embeddings, usar keywords com confiança original
            for field in bncc_fields:
                if field in keywords_result['extracted'] and field not in extracted:
                    extracted[field] = keywords_result['extracted'][field]
                    confidence[field] = keywords_result['confidence'][field]  # Confiança original
        
        # FALLBACK PEDAGÓGICO: se incerto em campos BNCC, usar genérico
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
                    logger.debug(f"🔄 FALLBACK pedagógico: unidadeTematica = {fallback_unidade}")
        
        # EMITIR FLAG DE AMBIGUIDADE
        ambiguous_fields = []
        for field, conf in confidence.items():
            if conf <= 0.7 and field in ['disciplina', 'unidadeTematica', 'objetoConhecimento', 'habilidade']:
                ambiguous_fields.append(field)
        
        if ambiguous_fields:
            logger.warning(f"⚠️  Campos ambíguos (conf ≤ 0.7): {ambiguous_fields}")
            # Criar sugestão estruturada
            from app.models.responses import Suggestion
            suggestions.append(Suggestion(
                field="ambiguidade",
                values=ambiguous_fields,
                message=f"Campos com baixa confiança (≤ 0.7): {', '.join(ambiguous_fields)}"
            ))
        
        # Identificar campos faltantes
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

    
    # ========== Métodos Auxiliares (DRY) ==========
    
    def _initialize_result(self, context: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Inicializa resultado com contexto bloqueado
        
        Returns:
            Tupla (extracted, confidence, suggestions)
        """
        extracted = {}
        confidence = {}
        suggestions = []
        
        # REGRA HARD: Contexto explícito vence SEMPRE
        if context:
            for key, value in context.items():
                if value:
                    extracted[key] = value
                    confidence[key] = 1.0
                    logger.debug(f"🔒 Campo bloqueado por contexto: {key} = {value}")
        
        return extracted, confidence, suggestions
    
    def _copy_non_bncc_fields(
        self,
        source_result: Dict[str, Any],
        extracted: Dict[str, Any],
        confidence: Dict[str, float]
    ) -> None:
        """
        Copia campos não-BNCC do resultado fonte
        
        Args:
            source_result: Resultado fonte (keywords)
            extracted: Dicionário de campos extraídos (modificado in-place)
            confidence: Dicionário de confiânças (modificado in-place)
        """
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
        """
        Valida resultado de embeddings (progressão pedagógica e disciplina)
        
        Args:
            embeddings_result: Resultado do embeddings
            year_extracted: Ano extraído do texto
            discipline_hint: Disciplina detectada por termo forte
            
        Returns:
            embeddings_result se válido, None se bloqueado
        """
        # Validar progressão pedagógica
        year_embedding = embeddings_result.get('ano')
        if not YearValidator.is_valid_progression(year_extracted, year_embedding, max_difference=2):
            logger.warning(f"⚠️  Embedding retornou ano {year_embedding} mas texto menciona {year_extracted} (diferença > 2 anos)")
            logger.warning(f"🚫 BLOQUEANDO resultado de embeddings por progressão pedagógica")
            return None
        
        # Validar disciplina
        if discipline_hint:
            discipline_embedding = embeddings_result.get('disciplina')
            if discipline_embedding != discipline_hint:
                logger.warning(f"⚠️  Embedding retornou {discipline_embedding} mas termo forte indica {discipline_hint}")
                logger.warning(f"🚫 VETANDO disciplina do embedding (keyword vence)")
                return None
        
        return embeddings_result
    
    def _apply_embeddings_result(
        self,
        embeddings_result: Dict[str, Any],
        extracted: Dict[str, Any],
        confidence: Dict[str, float]
    ) -> None:
        """
        Aplica resultado de embeddings aos campos extraídos
        
        Args:
            embeddings_result: Resultado do embeddings
            extracted: Dicionário de campos extraídos (modificado in-place)
            confidence: Dicionário de confiânças (modificado in-place)
        """
        bncc_fields = ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']
        for field in bncc_fields:
            if field in embeddings_result and embeddings_result[field]:
                # Não sobrescrever campos já extraídos com alta confiança
                if field not in extracted or confidence.get(field, 0) < 0.85:
                    extracted[field] = embeddings_result[field]
                    confidence[field] = embeddings_result['confidence'][field]
    
    def _get_missing_fields(
        self,
        extracted: Dict[str, Any],
        confidence: Dict[str, float],
        threshold: float = 0.5
    ) -> List[str]:
        """
        Identifica campos faltantes ou com baixa confiança
        
        Args:
            extracted: Campos extraídos
            confidence: Confiânças dos campos
            threshold: Threshold mínimo de confiança
            
        Returns:
            Lista de campos faltantes
        """
        all_fields = [
            "disciplina", "ano", "perfilAluno",
            "unidadeTematica", "objetoConhecimento", "habilidade",
            "nivelBloom", "tipoQuestao", "tipoTextoBase"
        ]
        return [
            field for field in all_fields
            if field not in extracted or confidence.get(field, 0) < threshold
        ]
