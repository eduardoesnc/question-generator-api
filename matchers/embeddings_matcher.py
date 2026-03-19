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
from matchers.synonyms import get_key_terms


class EmbeddingsMatcher:
    """Matcher baseado em embeddings semânticos"""
    
    def __init__(self):
        self.model = None
        self.embeddings_cache = {}
        self.reverse_index = {}
        self._load_model()
        self._load_embeddings()
    
    def _load_model(self):
        """Carrega modelo fine-tuned (OBRIGATÓRIO)"""
        logger.info("📦 Carregando modelo FINE-TUNED...")
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bncc-embeddings-finetuned')
        
        if not os.path.exists(model_path):
            error_msg = (
                f"❌ ERRO: Modelo fine-tuned não encontrado em: {model_path}\n"
                f"Execute os seguintes comandos:\n"
                f"  1. python scripts/generate_training_data.py\n"
                f"  2. python scripts/finetune_embeddings.py\n"
                f"  3. python scripts/generate_embeddings.py"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"🎯 Carregando modelo de: {model_path}")
        
        # Suprimir warnings do tokenizer
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
            self.model = SentenceTransformer(model_path)
        
        logger.success("✅ Modelo FINE-TUNED carregado com sucesso!")
    
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
        
        # Gerar embedding do texto original
        # NOTA: Para usar contexto, precisa regenerar embeddings com: python scripts/generate_embeddings.py
        text_embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Calcular similaridade com todos os objetos (ou filtrados)
        similarities = []
        
        # Extrair termos-chave do texto apenas para boost (não para gerar embedding)
        key_terms_text_weighted = get_key_terms(text, include_weights=True)
        
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
            
            # Calcular similaridade coseno base (pura, sem interferência)
            base_similarity = cosine_similarity(text_emb, obj_emb)[0][0]
            similarity = base_similarity
            
            # ✅ BOOST OPCIONAL: Se houver termos importantes em comum, dar um boost
            # Isso ajuda quando o modelo não captura bem termos específicos do domínio
            objeto_texto = context['texto']
            key_terms_objeto_weighted = get_key_terms(objeto_texto, include_weights=True)
            common_terms = set(key_terms_text_weighted.keys()) & set(key_terms_objeto_weighted.keys())
            
            boost_applied = 1.0
            if common_terms:
                # Contar termos de alto valor (peso >= 2.0)
                high_value_matches = sum(1 for t in common_terms if key_terms_text_weighted.get(t, 1.0) >= 2.0)
                
                # Aplicar boost MODERADO (não queremos sobrescrever o embedding)
                if high_value_matches >= 2:
                    boost_applied = 1.3  # 2+ termos importantes
                    similarity *= boost_applied
                elif high_value_matches >= 1:
                    boost_applied = 1.2  # 1 termo importante
                    similarity *= boost_applied
                elif len(common_terms) >= 3:
                    boost_applied = 1.1  # 3+ termos comuns
                    similarity *= boost_applied
            
            similarities.append({
                'key': key,
                'texto': context['texto'],
                'tipo': context['tipo'],
                'objeto': context['objeto'],
                'similarity': float(similarity),
                'base_similarity': float(base_similarity),
                'boost': boost_applied,
                'context': context,
                'common_terms': len(common_terms),
                'high_value_terms': sum(1 for t in common_terms if key_terms_text_weighted.get(t, 1.0) >= 2.0) if common_terms else 0
            })
        
        if not similarities:
            logger.warning(f"❌ Nenhum objeto encontrado com os filtros aplicados")
            return None
        
        # Ordenar por similaridade
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # ✅ MELHORIA: Preferir objetos/habilidades sobre unidades
        # Objetos têm hierarquia completa, unidades não têm objeto específico
        # Aplicar boost para objetos e habilidades
        for match in similarities:
            if match['tipo'] == 'objeto':
                match['similarity'] *= 1.2  # Boost de 20% para objetos
            elif match['tipo'] == 'habilidade':
                match['similarity'] *= 1.15  # Boost de 15% para habilidades
        
        # Re-ordenar após aplicar boost
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Mostrar top 3
        logger.debug("🏆 Top 3 matches (embeddings):")
        for i, match in enumerate(similarities[:3]):
            tipo_emoji = {'unidade': '📚', 'objeto': '📖', 'habilidade': '🎯'}.get(match['tipo'], '📄')
            boost_info = f" (boost: {match['boost']:.2f}x)" if match['boost'] > 1.0 else ""
            tipo_boost = ""
            if match['tipo'] == 'objeto':
                tipo_boost = " [+20% objeto]"
            elif match['tipo'] == 'habilidade':
                tipo_boost = " [+15% habilidade]"
            
            logger.debug(f"   {i+1}. {tipo_emoji} Similaridade: {match['similarity']:.3f}{boost_info}{tipo_boost} | {match['context']['disciplina']} {match['context']['ano']}")
            if match['boost'] > 1.0:
                logger.debug(f"      Base: {match['base_similarity']:.3f} → Final: {match['similarity']:.3f}")
            logger.debug(f"      {match['tipo'].title()}: '{match['texto'][:80]}...'")
            logger.debug(f"      Termos comuns: {match['common_terms']} | Termos importantes: {match['high_value_terms']}")
            logger.debug(f"      Unidade: {match['context']['unidade']}")
            if match['context']['objeto']:
                logger.debug(f"      Objeto: '{match['context']['objeto'][:60]}...'")
            else:
                logger.debug(f"      ⚠️  Sem objeto específico (match foi em unidade)")
        
        # ✅ Threshold mais baixo (principal melhoria)
        threshold = 0.20 if (disciplina or ano) else 0.15
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
                'similarity_score': best['similarity'],
                'common_terms': best['common_terms'],
                'high_value_terms': best['high_value_terms']
            }
        else:
            logger.debug(f"❌ Similaridade insuficiente: {similarities[0]['similarity']:.3f} (threshold: {threshold})")
        
        return None
