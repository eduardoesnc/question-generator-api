"""
Base matcher class for all educational field matchers
"""
from typing import Dict, List, Optional, Tuple
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import unicodedata


def remove_accents(text: str) -> str:
    """Remove acentos de um texto"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_text(text: str) -> str:
    """Normaliza texto: lowercase e remove acentos"""
    return remove_accents(text.lower())


class BaseMatcher:
    """Classe base para matchers educacionais"""
    
    def __init__(self, nlp, patterns: Dict[str, List[str]]):
        """
        Args:
            nlp: modelo spaCy carregado
            patterns: dicionário {categoria: [palavras-chave]}
        """
        self.nlp = nlp
        self.patterns = patterns
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._build_matcher()
    
    def _build_matcher(self):
        """Constrói o PhraseMatcher com os padrões"""
        for category, keywords in self.patterns.items():
            # Criar docs para cada keyword
            patterns = [self.nlp.make_doc(kw) for kw in keywords]
            self.matcher.add(category, patterns)
    
    def match(self, text: str) -> Optional[Tuple[str, float]]:
        """
        Encontra a melhor correspondência no texto
        
        Returns:
            Tuple (categoria, confiança) ou None
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        if not matches:
            return None
        
        # Encontrar o melhor match (maior span)
        best_match = None
        best_length = 0
        
        for match_id, start, end in matches:
            span = doc[start:end]
            span_length = len(span.text)
            
            if span_length > best_length:
                best_length = span_length
                category = self.nlp.vocab.strings[match_id]
                best_match = (category, self._calculate_confidence(span))
        
        return best_match
    
    def _calculate_confidence(self, span) -> float:
        """Calcula confiança baseada no tamanho do match"""
        # Quanto maior o span, maior a confiança
        base_confidence = 0.75
        length_bonus = min(0.20, len(span.text) / 100)
        return min(0.98, base_confidence + length_bonus)
    
    def match_all(self, text: str) -> List[Tuple[str, float]]:
        """Retorna todos os matches encontrados"""
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        results = []
        for match_id, start, end in matches:
            span = doc[start:end]
            category = self.nlp.vocab.strings[match_id]
            confidence = self._calculate_confidence(span)
            results.append((category, confidence))
        
        return results
