"""
Matcher usando embeddings semânticos (sentence-transformers)
"""
import json
import os
import sys
import numpy as np
from typing import Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Adicionar path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.logging import logger


class EmbeddingsMatcher:
    """Matcher baseado em embeddings semânticos"""
    
    def __init__(self):
        self.model = None
        self.embeddings_cache = {}
        self.reverse_index = {}
        self._load_model()
        self._load_embeddings()
    
    def _load_model(self):
        """Carrega modelo sentence-transformers"""
        logger.info("📦 Carregando modelo sentence-transformers...")
        # Modelo maior e mais preciso (768 dimensões)
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        logger.success("✅ Modelo sentence-transformers carregado!")
    
    def _load_embeddings(self):
        """Carrega embeddings pré-computados"""
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 
            'bncc_embeddings.json'
        )
        
        if not os.path.exists(embeddings_path):
            logger.warning("⚠️  Embeddings não encontrados! Execute: python scripts/generate_embeddings.py")
            return
        
        logger.info("📚 Carregando embeddings pré-computados...")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Converter embeddings de lista para numpy array
        # Nova estrutura: chave = "disciplina|ano|tipo|texto"
        # tipo pode ser: 'unidade', 'objeto', 'habilidade'
        for key, info in data.items():
            self.embeddings_cache[key] = np.array(info['embedding'])
            
            # Extrair informações
            self.reverse_index[key] = {
                'texto': info.get('texto', ''),
                'tipo': info.get('tipo', 'objeto'),  # backward compatibility
                'objeto': info.get('objeto', info.get('texto', '')),
                'disciplina': info['disciplina'],
                'ano': info['ano'],
                'unidade': info['unidade'],
                'habilidades': info['habilidades']
            }
        
        logger.success(f"✅ {len(self.embeddings_cache)} embeddings carregados!")
    
    def search_global(self, text: str, disciplina: str = None, ano: str = None) -> Optional[Dict]:
        """
        Busca global usando similaridade de embeddings
        
        Args:
            text: Texto para buscar
            disciplina: Filtrar por disciplina (opcional)
            ano: Filtrar por ano (opcional)
        """
        if not self.embeddings_cache:
            logger.error("❌ Embeddings não carregados!")
            return None
        
        logger.debug(f"🌍 BUSCA POR EMBEDDINGS para: '{text}'")
        if disciplina:
            logger.debug(f"   Filtrando por disciplina: {disciplina}")
        if ano:
            logger.debug(f"   Filtrando por ano: {ano}")
        
        # Gerar embedding do texto de entrada
        text_embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Calcular similaridade com todos os objetos (ou filtrados)
        similarities = []
        for key, cached_embedding in self.embeddings_cache.items():
            context = self.reverse_index[key]
            
            # Filtrar por disciplina/ano se fornecido
            if disciplina and context['disciplina'] != disciplina:
                continue
            if ano and context['ano'] != ano:
                continue
            
            # Reshape para sklearn
            text_emb = text_embedding.reshape(1, -1)
            obj_emb = cached_embedding.reshape(1, -1)
            
            # Calcular similaridade coseno
            similarity = cosine_similarity(text_emb, obj_emb)[0][0]
            
            similarities.append({
                'key': key,
                'texto': context['texto'],
                'tipo': context['tipo'],
                'objeto': context['objeto'],
                'similarity': float(similarity),
                'context': context
            })
        
        if not similarities:
            logger.warning(f"❌ Nenhum objeto encontrado com os filtros aplicados")
            return None
        
        # Ordenar por similaridade
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Mostrar top 3
        logger.debug("🏆 Top 3 matches (embeddings):")
        for i, match in enumerate(similarities[:3]):
            tipo_emoji = {'unidade': '📚', 'objeto': '📖', 'habilidade': '🎯'}.get(match['tipo'], '📄')
            logger.debug(f"   {i+1}. {tipo_emoji} Similaridade: {match['similarity']:.3f} | {match['context']['disciplina']} {match['context']['ano']}")
            logger.debug(f"      {match['tipo'].title()}: '{match['texto'][:80]}...'")
            logger.debug(f"      Unidade: {match['context']['unidade']}")
            if match['context']['objeto']:
                logger.debug(f"      Objeto: '{match['context']['objeto'][:60]}...'")
        
        # Retornar melhor match se passar threshold
        threshold = 0.40 if (disciplina or ano) else 0.30  # Threshold maior se filtrado
        if similarities and similarities[0]['similarity'] > threshold:
            best = similarities[0]
            context = best['context']
            
            logger.info(f"✅ MATCH SELECIONADO (embeddings) - tipo: {best['tipo']}, similaridade: {best['similarity']:.3f}")
            
            return {
                'disciplina': context['disciplina'],
                'ano': context['ano'],
                'unidadeTematica': context['unidade'],
                'objetoConhecimento': context['objeto'],
                'habilidade': context['habilidades'][0] if context['habilidades'] else None,
                'confidence': {
                    'disciplina': 0.85,
                    'ano': 0.85,
                    'unidadeTematica': min(0.85, 0.55 + best['similarity'] * 0.30),
                    'objetoConhecimento': min(0.85, 0.55 + best['similarity'] * 0.30),
                    'habilidade': 0.75 if context['habilidades'] else 0.0
                },
                'method': 'embeddings',
                'similarity_score': best['similarity']
            }
        else:
            logger.debug(f"❌ Similaridade insuficiente: {similarities[0]['similarity']:.3f} (threshold: {threshold})")
        
        return None
