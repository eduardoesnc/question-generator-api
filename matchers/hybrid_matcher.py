"""
Matcher híbrido: combina termos-chave + embeddings
"""
import sys
import os
from typing import Dict, Optional

# Adicionar path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers.bncc_matcher import BNCCMatcher
from matchers.embeddings_matcher import EmbeddingsMatcher
from app.core.logging import logger


class HybridMatcher:
    """
    Matcher híbrido que combina:
    1. Termos-chave (rápido, explicável)
    2. Embeddings (preciso, semântico)
    """
    
    def __init__(self, nlp):
        self.bncc_matcher = BNCCMatcher(nlp)
        self.embeddings_matcher = EmbeddingsMatcher()
    
    def search_global(self, text: str) -> Optional[Dict]:
        """
        Busca híbrida:
        1. Tenta termos-chave primeiro (rápido)
        2. Se score baixo, usa embeddings (preciso)
        3. Combina resultados com pesos
        """
        logger.debug(f"🔀 BUSCA HÍBRIDA para: '{text}'")
        
        # FASE 1: Termos-chave
        logger.debug("📝 Fase 1: Termos-chave...")
        keywords_result = self.bncc_matcher.search_global(text)
        
        # FASE 2: Embeddings
        logger.debug("🧠 Fase 2: Embeddings...")
        embeddings_result = self.embeddings_matcher.search_global(text)
        
        # DECISÃO: Qual usar?
        if keywords_result and embeddings_result:
            # Ambos encontraram - comparar scores
            kw_conf = keywords_result['confidence']['objetoConhecimento']
            emb_conf = embeddings_result['confidence']['objetoConhecimento']
            
            logger.debug("⚖️  Comparando resultados:")
            logger.debug(f"   Termos-chave: {keywords_result['objetoConhecimento'][:50]}... (conf: {kw_conf:.2f})")
            logger.debug(f"   Embeddings: {embeddings_result['objetoConhecimento'][:50]}... (conf: {emb_conf:.2f})")
            
            # Se são o mesmo objeto, aumentar confiança
            if keywords_result['objetoConhecimento'] == embeddings_result['objetoConhecimento']:
                logger.success("✅ CONSENSO! Ambos métodos concordam.")
                keywords_result['confidence'] = {
                    k: min(0.95, v * 1.1) for k, v in keywords_result['confidence'].items()
                }
                keywords_result['method'] = 'hybrid_consensus'
                return keywords_result
            
            # Se diferentes, SEMPRE preferir keywords quando tem confiança razoável
            # Keywords é mais confiável para termos específicos do domínio
            if kw_conf >= 0.60:  # Threshold mais baixo para keywords
                logger.info("📝 Usando termos-chave (mais confiável para domínio específico)")
                keywords_result['method'] = 'hybrid_keywords_preferred'
                return keywords_result
            elif emb_conf > kw_conf * 1.3:  # Embeddings precisa ser MUITO melhor
                logger.info("🧠 Usando embeddings (confiança significativamente maior)")
                embeddings_result['method'] = 'hybrid_embeddings'
                return embeddings_result
            else:
                logger.info("📝 Usando termos-chave (default para empate)")
                keywords_result['method'] = 'hybrid_keywords_default'
                return keywords_result
        
        elif keywords_result:
            logger.info("📝 Apenas termos-chave encontrou resultado")
            keywords_result['method'] = 'hybrid_keywords_only'
            return keywords_result
        
        elif embeddings_result:
            logger.info("🧠 Apenas embeddings encontrou resultado")
            embeddings_result['method'] = 'hybrid_embeddings_only'
            return embeddings_result
        
        logger.debug("❌ Nenhum método encontrou resultado")
        return None
