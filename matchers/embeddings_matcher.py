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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.logging import logger
from app.config import settings

class EmbeddingsMatcher:
    """Matcher baseado em embeddings semânticos"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.embeddings_cache = {}
        self.reverse_index = {}
        self.non_bncc_embeddings = {}
        self._model_path_override = model_path
        self._load_model()
        self._load_embeddings()
        self._load_non_bncc_embeddings()
    
    def _load_model(self):
        """Carrega modelo fine-tuned (OBRIGATÓRIO)"""
        
        # Usa model_path passado no construtor, depois ENV, depois padrão
        model_path = self._model_path_override or settings.sentence_transformer_model
        
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
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
            self.model = SentenceTransformer(model_path)
        
    def _load_embeddings(self):
        """Carrega embeddings pré-computados"""
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 
            'bncc_embeddings.json'
        )
        
        if not os.path.exists(embeddings_path):
            logger.warning("Embeddings não encontrados! Execute: python scripts/generate_embeddings.py")
            return
        
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key, info in data.items():
            self.embeddings_cache[key] = np.array(info['embedding'])
            
            self.reverse_index[key] = {
                'texto': info.get('texto', ''),
                'tipo': info.get('tipo', 'objeto'),
                'objeto': info.get('objeto', info.get('texto', '')),
                'disciplina': info['disciplina'],
                'ano': info['ano'],
                'unidade': info['unidade'],
                'habilidades': info['habilidades']
            }
    
    def _load_non_bncc_embeddings(self):
        """Carrega embeddings para campos não-BNCC"""
        embeddings_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 
            'non_bncc_embeddings.json'
        )
        
        if not os.path.exists(embeddings_path):
            logger.warning("Embeddings não-BNCC não encontrados! Execute: python scripts/generate_non_bncc_embeddings.py")
            return
        
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for category, items in data.items():
            self.non_bncc_embeddings[category] = {}
            for key, info in items.items():
                self.non_bncc_embeddings[category][key] = np.array(info['embedding'])
        
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
        
        text_embedding = self.model.encode(text, convert_to_numpy=True)
        
        similarities = []
        
        key_terms_text_weighted = get_key_terms(text, include_weights=True)
        
        for key, cached_embedding in self.embeddings_cache.items():
            context = self.reverse_index[key]
            
            if disciplina and context['disciplina'] != disciplina:
                continue
            if ano and context['ano'] != ano:
                continue
            
            text_emb = text_embedding.reshape(1, -1)
            obj_emb = cached_embedding.reshape(1, -1)
            
            base_similarity = cosine_similarity(text_emb, obj_emb)[0][0]
            similarity = base_similarity
            
            objeto_texto = context['texto']
            key_terms_objeto_weighted = get_key_terms(objeto_texto, include_weights=True)
            common_terms = set(key_terms_text_weighted.keys()) & set(key_terms_objeto_weighted.keys())
            
            boost_applied = 1.0
            if common_terms:
                high_value_matches = sum(1 for t in common_terms if key_terms_text_weighted.get(t, 1.0) >= 2.0)
                
                if high_value_matches >= 2:
                    boost_applied = 1.3
                    similarity *= boost_applied
                elif high_value_matches >= 1:
                    boost_applied = 1.2
                    similarity *= boost_applied
                elif len(common_terms) >= 3:
                    boost_applied = 1.1
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
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        for match in similarities:
            if match['tipo'] == 'objeto':
                match['similarity'] *= 1.2
            elif match['tipo'] == 'habilidade':
                match['similarity'] *= 1.15
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        for i, match in enumerate(similarities[:3]):
            tipo_emoji = {'unidade': '📚', 'objeto': '📖', 'habilidade': '🎯'}.get(match['tipo'], '📄')
            boost_info = f" (boost: {match['boost']:.2f}x)" if match['boost'] > 1.0 else ""
            tipo_boost = ""
            if match['tipo'] == 'objeto':
                tipo_boost = " [+20% objeto]"
            elif match['tipo'] == 'habilidade':
                tipo_boost = " [+15% habilidade]"

        threshold = 0.20 if (disciplina or ano) else 0.15
        if similarities and similarities[0]['similarity'] > threshold:
            best = similarities[0]
            context = best['context']
            
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

    
    def infer_bloom(self, text: str) -> Optional[Tuple[str, float]]:
        """Infere nível de Bloom usando embeddings"""
        if 'bloom' not in self.non_bncc_embeddings:
            return None
        
        text_embedding = self.model.encode(text, convert_to_numpy=True).reshape(1, -1)
        
        best_nivel = None
        best_similarity = 0.0
        
        for nivel, embedding in self.non_bncc_embeddings['bloom'].items():
            nivel_emb = embedding.reshape(1, -1)
            similarity = cosine_similarity(text_embedding, nivel_emb)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_nivel = nivel
        
        if best_similarity > 0.30:
            confidence = min(0.85, 0.50 + best_similarity * 0.35)
            return (best_nivel, confidence)
        
        return None
    
    def infer_perfil_aluno(self, text: str) -> Optional[Tuple[str, float]]:
        """Infere perfil de aluno usando embeddings"""
        if 'perfil_aluno' not in self.non_bncc_embeddings:
            return None
        
        text_embedding = self.model.encode(text, convert_to_numpy=True).reshape(1, -1)
        
        best_perfil = None
        best_similarity = 0.0
        
        for perfil, embedding in self.non_bncc_embeddings['perfil_aluno'].items():
            perfil_emb = embedding.reshape(1, -1)
            similarity = cosine_similarity(text_embedding, perfil_emb)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_perfil = perfil
        
        if best_similarity > 0.30:
            confidence = min(0.85, 0.50 + best_similarity * 0.35)
            return (best_perfil, confidence)
        
        return None
    
    def infer_tipo_questao(self, text: str) -> Optional[Tuple[str, float]]:
        """Infere tipo de questão usando embeddings"""
        if 'tipo_questao' not in self.non_bncc_embeddings:
            return None
        
        text_embedding = self.model.encode(text, convert_to_numpy=True).reshape(1, -1)
        
        best_tipo = None
        best_similarity = 0.0
        
        for tipo, embedding in self.non_bncc_embeddings['tipo_questao'].items():
            tipo_emb = embedding.reshape(1, -1)
            similarity = cosine_similarity(text_embedding, tipo_emb)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_tipo = tipo
        
        if best_similarity > 0.30:
            confidence = min(0.85, 0.50 + best_similarity * 0.35)
            return (best_tipo, confidence)
        
        return None
    
    def infer_tipo_texto_base(self, text: str) -> Optional[Tuple[str, float]]:
        """Infere tipo de texto base usando embeddings"""
        if 'tipo_texto_base' not in self.non_bncc_embeddings:
            return None
        
        text_embedding = self.model.encode(text, convert_to_numpy=True).reshape(1, -1)
        
        best_tipo = None
        best_similarity = 0.0
        
        for tipo, embedding in self.non_bncc_embeddings['tipo_texto_base'].items():
            tipo_emb = embedding.reshape(1, -1)
            similarity = cosine_similarity(text_embedding, tipo_emb)[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_tipo = tipo
        
        if best_similarity > 0.30:
            confidence = min(0.85, 0.50 + best_similarity * 0.35)
            return (best_tipo, confidence)
        
        return None
    
    def search_with_ensemble(self, text: str, top_k: int = 5) -> Optional[Dict]:
        """
        Busca com ensemble: retorna top-K matches e escolhe o melhor com validação semântica
        Inclui expansão semântica automática
        """
        if not self.embeddings_cache:
            logger.error("❌ Embeddings não carregados!")
            return None
        
        text_variations = self._expand_query(text)
        
        all_similarities = []
        
        for text_var in text_variations:
            text_embedding = self.model.encode(text_var, convert_to_numpy=True)
            key_terms_text_weighted = get_key_terms(text_var, include_weights=True)
            
            for key, cached_embedding in self.embeddings_cache.items():
                context = self.reverse_index[key]
                
                text_emb = text_embedding.reshape(1, -1)
                obj_emb = cached_embedding.reshape(1, -1)
                
                base_similarity = cosine_similarity(text_emb, obj_emb)[0][0]
                similarity = base_similarity
                
                objeto_texto = context['texto']
                key_terms_objeto_weighted = get_key_terms(objeto_texto, include_weights=True)
                common_terms = set(key_terms_text_weighted.keys()) & set(key_terms_objeto_weighted.keys())
                
                boost_applied = 1.0
                if common_terms:
                    high_value_matches = sum(1 for t in common_terms if key_terms_text_weighted.get(t, 1.0) >= 2.0)
                    
                    if high_value_matches >= 2:
                        boost_applied = 1.3
                        similarity *= boost_applied
                    elif high_value_matches >= 1:
                        boost_applied = 1.2
                        similarity *= boost_applied
                    elif len(common_terms) >= 3:
                        boost_applied = 1.1
                        similarity *= boost_applied
                
                all_similarities.append({
                    'key': key,
                    'texto': context['texto'],
                    'tipo': context['tipo'],
                    'objeto': context['objeto'],
                    'similarity': float(similarity),
                    'base_similarity': float(base_similarity),
                    'boost': boost_applied,
                    'context': context,
                    'common_terms': len(common_terms),
                    'high_value_terms': sum(1 for t in common_terms if key_terms_text_weighted.get(t, 1.0) >= 2.0) if common_terms else 0,
                    'query_variation': text_var
                })
        
        if not all_similarities:
            return None
        
        similarities_by_key = {}
        for sim in all_similarities:
            key = sim['key']
            if key not in similarities_by_key or sim['similarity'] > similarities_by_key[key]['similarity']:
                similarities_by_key[key] = sim
        
        similarities = list(similarities_by_key.values())
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        for match in similarities:
            if match['tipo'] == 'objeto':
                match['similarity'] *= 1.2
            elif match['tipo'] == 'habilidade':
                match['similarity'] *= 1.15
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        top_matches = similarities[:top_k]
        
        disciplinas_count = {}
        for match in top_matches:
            disc = match['context']['disciplina']
            disciplinas_count[disc] = disciplinas_count.get(disc, 0) + 1
        
        disciplina_mais_votada = max(disciplinas_count, key=disciplinas_count.get) if disciplinas_count else None
        
        best_match = None
        for match in top_matches:
            if match['context']['disciplina'] == disciplina_mais_votada:
                best_match = match
                break
        
        if not best_match:
            best_match = top_matches[0]
        
        if best_match['similarity'] > 0.15:
            context = best_match['context']
            
            return {
                'disciplina': context['disciplina'],
                'ano': context['ano'],
                'unidadeTematica': context['unidade'],
                'objetoConhecimento': context['objeto'],
                'habilidade': context['habilidades'][0] if context['habilidades'] else None,
                'confidence': {
                    'disciplina': 0.85,
                    'ano': 0.85,
                    'unidadeTematica': min(0.85, 0.55 + best_match['similarity'] * 0.30),
                    'objetoConhecimento': min(0.85, 0.55 + best_match['similarity'] * 0.30),
                    'habilidade': 0.75 if context['habilidades'] else 0.0
                },
                'method': 'embeddings_ensemble',
                'similarity_score': best_match['similarity'],
                'votes': disciplinas_count,
                'best_query': best_match['query_variation']
            }
        
        return None
    
    def _expand_query(self, text: str, max_variations: int = 5) -> list:
        """
        Expande a query usando busca semântica na BNCC
        Retorna variações do texto que podem melhorar a busca
        """
        variations = [text]
        
        if len(text.split()) <= 3:
            text_embedding = self.model.encode(text, convert_to_numpy=True).reshape(1, -1)
            
            candidates = []
            for key, cached_embedding in self.embeddings_cache.items():
                obj_emb = cached_embedding.reshape(1, -1)
                similarity = cosine_similarity(text_embedding, obj_emb)[0][0]
                
                if similarity > 0.25:
                    context = self.reverse_index[key]
                    candidates.append({
                        'texto': context['texto'],
                        'similarity': similarity
                    })
            
            candidates.sort(key=lambda x: x['similarity'], reverse=True)
            
            for candidate in candidates[:3]:
                texto = candidate['texto']
                palavras = texto.lower().split()
                
                for palavra in palavras:
                    if len(palavra) > 4 and palavra not in text.lower():
                        variation = f"{text} {palavra}"
                        if variation not in variations:
                            variations.append(variation)
                            if len(variations) >= max_variations:
                                return variations
        
        return variations

