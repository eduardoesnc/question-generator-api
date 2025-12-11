import spacy
from typing import Dict, List, Any, Optional
from matchers.pipeline import NLPPipeline


class NLPProcessor:
    def __init__(self):
        self.nlp = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo spaCy para português"""
        try:
            self.nlp = spacy.load("pt_core_news_lg")
            self.pipeline = NLPPipeline(self.nlp)
        except OSError:
            print("Modelo pt_core_news_lg não encontrado. Tentando carregar modelo menor...")
            try:
                self.nlp = spacy.load("pt_core_news_sm")
                self.pipeline = NLPPipeline(self.nlp)
            except OSError:
                print("AVISO: Nenhum modelo spaCy encontrado. Execute: python -m spacy download pt_core_news_lg")
                self.nlp = None
                self.pipeline = None
    
    def is_loaded(self) -> bool:
        """Verifica se o modelo foi carregado"""
        return self.nlp is not None and self.pipeline is not None
    
    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa o texto e extrai informações educacionais usando a pipeline modular
        """
        if not self.is_loaded():
            return {
                "extracted": {},
                "confidence": {},
                "suggestions": [],
                "missing_fields": ["disciplina", "ano", "nivelBloom", "tipoQuestao", "tipoTextoBase", "perfilAluno"]
            }
        
        return self.pipeline.classify(text, context)

