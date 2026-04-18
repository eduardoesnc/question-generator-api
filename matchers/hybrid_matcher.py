"""
Matcher híbrido: combina termos-chave + embeddings
"""
import sys
import os
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers.bncc_matcher import BNCCMatcher
from matchers.embeddings_matcher import EmbeddingsMatcher
from app.core.logging import logger

class HybridMatcher:
    """
    Matcher híbrido que combina:
        pass
    1. Termos-chave (rápido, explicável)
    2. Embeddings (preciso, semântico)
    """
    
    def __init__(self, nlp):
        self.bncc_matcher = BNCCMatcher(nlp)
        self.embeddings_matcher = EmbeddingsMatcher()
    
    def search_global(self, text: str) -> Optional[Dict]:
        """
        Busca híbrida:
            pass
        1. Tenta termos-chave primeiro (rápido)
        2. Se score baixo, usa embeddings (preciso)
        3. Combina resultados com pesos
        """
        
        keywords_result = self.bncc_matcher.search_global(text)
        
        embeddings_result = self.embeddings_matcher.search_global(text)
        
        if keywords_result and embeddings_result:
            kw_conf = keywords_result['confidence']['objetoConhecimento']
            emb_conf = embeddings_result['confidence']['objetoConhecimento']
            
            if keywords_result['objetoConhecimento'] == embeddings_result['objetoConhecimento']:
                keywords_result['confidence'] = {
                    k: min(0.95, v * 1.1) for k, v in keywords_result['confidence'].items()
                }
                keywords_result['method'] = 'hybrid_consensus'
                return keywords_result
            
            if kw_conf >= 0.60:
                keywords_result['method'] = 'hybrid_keywords_preferred'
                return keywords_result
            elif emb_conf > kw_conf * 1.3:
                embeddings_result['method'] = 'hybrid_embeddings'
                return embeddings_result
            else:
                keywords_result['method'] = 'hybrid_keywords_default'
                return keywords_result
        
        elif keywords_result:
            keywords_result['method'] = 'hybrid_keywords_only'
            return keywords_result
        
        elif embeddings_result:
            embeddings_result['method'] = 'hybrid_embeddings_only'
            return embeddings_result
        
        return None
