"""
Matcher para dados da BNCC (Unidades Temáticas, Objetos de Conhecimento, Habilidades)
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from spacy.matcher import PhraseMatcher
from .synonyms import expand_query, normalize_term, get_key_terms
import numpy as np


class BNCCMatcher:
    """Matcher para extrair informações da BNCC"""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.bncc_data = self._load_bncc_data()
        self.unidades_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self.objetos_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._build_matchers()
        # Cache para busca reversa (objeto -> contexto)
        self._build_reverse_index()
    
    def _load_bncc_data(self) -> Dict:
        """Carrega dados da BNCC do JSON"""
        try:
            # Path relativo ao diretório do módulo
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            bncc_path = os.path.join(BASE_DIR, 'data', 'bncc-data.json')
            
            if os.path.exists(bncc_path):
                with open(bncc_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"AVISO: Arquivo bncc-data.json não encontrado em {bncc_path}")
                return {}
        except Exception as e:
            print(f"Erro ao carregar BNCC: {e}")
            return {}
    
    def _build_matchers(self):
        """Constrói matchers para unidades e objetos"""
        if not self.bncc_data:
            return
        
        # Coletar todas as unidades temáticas e objetos
        unidades_set = set()
        objetos_set = set()
        
        for disciplina, anos in self.bncc_data.items():
            for ano, unidades in anos.items():
                for unidade, objetos in unidades.items():
                    unidades_set.add(unidade)
                    for objeto in objetos.keys():
                        objetos_set.add(objeto)
        
        # Adicionar ao matcher
        if unidades_set:
            patterns = [self.nlp.make_doc(u) for u in unidades_set]
            self.unidades_matcher.add("UNIDADE", patterns)
        
        if objetos_set:
            patterns = [self.nlp.make_doc(o) for o in objetos_set]
            self.objetos_matcher.add("OBJETO", patterns)
    
    def _build_reverse_index(self):
        """Constrói índice reverso: objeto -> {disciplina, ano, unidade, habilidades}"""
        self.reverse_index = {}
        
        for disciplina, anos in self.bncc_data.items():
            for ano, unidades in anos.items():
                for unidade, objetos in unidades.items():
                    for objeto, habilidades in objetos.items():
                        self.reverse_index[objeto] = {
                            'disciplina': disciplina,
                            'ano': ano,
                            'unidade': unidade,
                            'habilidades': habilidades
                        }
    
    def search_global(self, text: str) -> Optional[Dict]:
        """
        Busca GLOBAL na BNCC - procura em todas disciplinas/anos
        Retorna TUDO: disciplina, ano, unidade, objeto, habilidade
        """
        print(f"\n🌍 BUSCA GLOBAL na BNCC para: '{text}'")
        
        # Expandir com sinônimos
        text_variations = expand_query(text)
        print(f"   📝 Variações ({len(text_variations)}): {text_variations[:3]}...")
        
        best_matches = []  # Lista dos top 3 matches
        
        # Buscar em TODOS os objetos de conhecimento
        for objeto, context in self.reverse_index.items():
            max_similarity = 0
            best_variation = None
            
            # Testar cada variação
            for idx, text_var in enumerate(text_variations):
                # Usar termos-chave ponderados ao invés de similaridade simples
                key_terms_var = get_key_terms(text_var, include_weights=True)
                key_terms_obj = get_key_terms(objeto, include_weights=True)
                
                # Termos em comum
                common_terms = set(key_terms_var.keys()) & set(key_terms_obj.keys())
                
                if common_terms:
                    # Score ponderado
                    peso_comuns = sum(key_terms_var.get(t, 1.0) for t in common_terms)
                    peso_total_obj = sum(key_terms_obj.values())
                    
                    score = peso_comuns / max(peso_total_obj, 1.0)
                    
                    # Bonus MASSIVO para termos de alto valor (nomes próprios, etc.)
                    high_value_matches = sum(1 for t in common_terms if key_terms_var.get(t, 1.0) >= 2.0)
                    if high_value_matches >= 2:
                        score *= 3.0  # Aumentado de 2.0 para 3.0
                    elif high_value_matches >= 1:
                        score *= 2.5  # Aumentado de 1.5 para 2.5
                    
                    # Bonus para variações com sinônimos
                    if idx > 0:
                        score *= 1.5  # Aumentado de 1.3 para 1.5
                    
                    if score > max_similarity:
                        max_similarity = score
                        best_variation = text_var
                
                # Fallback: similaridade semântica APENAS se não encontrou termos
                # E com penalização PESADA
                # DESABILITADO - causa muitos falsos positivos
                # if max_similarity == 0:
                #     similarity = self._semantic_similarity(text_var, objeto)
                #     if similarity > max_similarity:
                #         max_similarity = similarity * 0.3
                #         best_variation = text_var
            
            if max_similarity > 0:
                best_matches.append({
                    'score': max_similarity,
                    'objeto': objeto,
                    'context': context,
                    'variation': best_variation
                })
        
        # Ordenar por score
        best_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Mostrar top 3
        print(f"\n   🏆 Top 3 matches:")
        for i, match in enumerate(best_matches[:3]):
            print(f"      {i+1}. Score: {match['score']:.3f} | {match['context']['disciplina']} {match['context']['ano']}")
            print(f"         Objeto: '{match['objeto'][:60]}...'")
            print(f"         Via: '{match['variation']}'")
        
        if best_matches and best_matches[0]['score'] > 0.20:  # Threshold
            best = best_matches[0]
            print(f"\n   ✅ MATCH GLOBAL SELECIONADO!")
            
            # Retornar tudo
            return {
                'disciplina': best['context']['disciplina'],
                'ano': best['context']['ano'],
                'unidadeTematica': best['context']['unidade'],
                'objetoConhecimento': best['objeto'],
                'habilidade': best['context']['habilidades'][0] if best['context']['habilidades'] else None,
                'confidence': {
                    'disciplina': 0.85,
                    'ano': 0.85,
                    'unidadeTematica': min(0.85, 0.55 + best['score'] * 0.30),
                    'objetoConhecimento': min(0.85, 0.55 + best['score'] * 0.30),
                    'habilidade': 0.75 if best['context']['habilidades'] else 0.0
                }
            }
        else:
            print(f"   ❌ Nenhum match global suficiente")
        
        return None
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade semântica entre dois textos usando embeddings do spaCy
        Retorna valor entre 0 e 1
        """
        try:
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Verificar se o modelo tem vetores
            if not doc1.has_vector or not doc2.has_vector:
                # Fallback: usar overlap de lemas (mais inteligente que palavras brutas)
                lemmas1 = set(token.lemma_.lower() for token in doc1 if not token.is_stop and not token.is_punct and len(token.text) > 2)
                lemmas2 = set(token.lemma_.lower() for token in doc2 if not token.is_stop and not token.is_punct and len(token.text) > 2)
                
                if not lemmas1 or not lemmas2:
                    return 0.0
                
                # Overlap básico
                overlap = len(lemmas1 & lemmas2)
                base_score = overlap / max(len(lemmas1), len(lemmas2))
                
                # BONUS: Se encontrou nomes próprios ou termos importantes em comum
                # Extrair termos importantes (substantivos próprios, nomes)
                important1 = set(token.lemma_.lower() for token in doc1 if (token.pos_ == "PROPN" or len(token.text) > 6) and not token.is_stop)
                important2 = set(token.lemma_.lower() for token in doc2 if (token.pos_ == "PROPN" or len(token.text) > 6) and not token.is_stop)
                
                important_overlap = len(important1 & important2)
                if important_overlap > 0:
                    # Dar bonus significativo para termos importantes
                    base_score *= (1.0 + important_overlap * 0.5)
                
                return min(1.0, base_score)
            
            # Usar similaridade do spaCy (baseada em word vectors)
            similarity = doc1.similarity(doc2)
            return max(0.0, similarity)  # Garantir que não seja negativo
        except Exception as e:
            print(f"      ⚠️  Erro na similaridade: {e}")
            return 0.0
    
    def match_unidade_tematica(self, text: str, disciplina: str = None, ano: str = None) -> Optional[Tuple[str, float]]:
        """
        Encontra unidade temática no texto
        Busca primeiro nos objetos de conhecimento com sinônimos
        """
        if not disciplina or not ano:
            return None
        
        # Expandir consulta com sinônimos
        text_variations = expand_query(text)
        print(f"   📝 Variações do texto ({len(text_variations)}): {text_variations[:3]}...")
        
        # Extrair termos-chave do texto
        key_terms_text = get_key_terms(text)
        print(f"   🔑 Termos-chave do texto: {key_terms_text}")
        
        try:
            unidades_data = self.bncc_data.get(disciplina, {}).get(ano, {})
            
            best_match = None
            best_score = 0
            best_objeto = None
            
            for unidade, objetos in unidades_data.items():
                for objeto in objetos.keys():
                    max_score_for_objeto = 0
                    
                    # Testar cada variação do texto
                    for idx, text_var in enumerate(text_variations):
                        # Extrair termos-chave COM PESOS
                        key_terms_var_weighted = get_key_terms(text_var, include_weights=True)
                        key_terms_objeto_weighted = get_key_terms(objeto, include_weights=True)
                        
                        # Palavras-chave em comum
                        key_terms_comuns = set(key_terms_var_weighted.keys()) & set(key_terms_objeto_weighted.keys())
                        
                        if key_terms_comuns:
                            # Score baseado em termos-chave PONDERADOS
                            # Somar os pesos dos termos em comum
                            peso_comuns = sum(key_terms_var_weighted.get(t, 1.0) for t in key_terms_comuns)
                            peso_total_objeto = sum(key_terms_objeto_weighted.values())
                            
                            score = peso_comuns / max(peso_total_objeto, 1.0)
                            
                            # Contar quantos termos de alto valor foram encontrados
                            high_value_matches = sum(1 for t in key_terms_comuns if key_terms_var_weighted.get(t, 1.0) >= 2.0)
                            
                            # Bonus progressivo para termos de alto valor
                            if high_value_matches >= 3:
                                score *= 2.0  # 3+ termos importantes
                            elif high_value_matches >= 2:
                                score *= 1.7  # 2 termos importantes
                            elif high_value_matches >= 1:
                                score *= 1.4  # 1 termo importante
                            
                            # Bonus se encontrou muitos termos no total
                            if len(key_terms_comuns) >= 3:
                                score *= 1.3
                            elif len(key_terms_comuns) >= 2:
                                score *= 1.2
                            
                            # Bonus extra se o match veio de uma variação com sinônimo (não o texto original)
                            if idx > 0:  # Não é o texto original
                                score *= 1.3
                            
                            max_score_for_objeto = max(max_score_for_objeto, score)
                    
                    if max_score_for_objeto > best_score:
                        best_score = max_score_for_objeto
                        best_match = unidade
                        best_objeto = objeto
                        key_terms_objeto_weighted = get_key_terms(objeto, include_weights=True)
                        high_value = {k: v for k, v in key_terms_objeto_weighted.items() if v > 1.0}
                        print(f"   🎯 Candidato: '{objeto[:60]}...'")
                        print(f"      Score: {max_score_for_objeto:.3f} | Termos importantes: {high_value if high_value else 'nenhum'}")
            
            if best_match and best_score > 0.15:  # Threshold mais baixo para permitir matches parciais
                confidence = min(0.85, 0.55 + best_score * 0.30)
                print(f"   ✅ MATCH! Unidade: '{best_match[:60]}...'")
                print(f"      Via objeto: '{best_objeto[:60]}...' (score: {best_score:.3f}, conf: {confidence:.2f})")
                return (best_match, confidence)
            else:
                print(f"   ❌ Score por termos-chave insuficiente: {best_score:.3f}")
                print(f"   🔄 Tentando busca semântica...")
                
                # FALLBACK: Busca semântica usando embeddings
                best_semantic_match = None
                best_semantic_score = 0
                best_semantic_objeto = None
                
                for unidade, objetos in unidades_data.items():
                    for objeto in objetos.keys():
                        # Calcular similaridade semântica com cada variação
                        max_similarity = 0
                        for text_var in text_variations:
                            similarity = self._semantic_similarity(text_var, objeto)
                            max_similarity = max(max_similarity, similarity)
                        
                        if max_similarity > best_semantic_score:
                            best_semantic_score = max_similarity
                            best_semantic_match = unidade
                            best_semantic_objeto = objeto
                
                if best_semantic_match and best_semantic_score > 0.35:  # Threshold para similaridade semântica
                    confidence = min(0.80, 0.50 + best_semantic_score * 0.30)
                    print(f"   ✅ MATCH SEMÂNTICO! Unidade: '{best_semantic_match[:60]}...'")
                    print(f"      Via objeto: '{best_semantic_objeto[:60]}...' (similaridade: {best_semantic_score:.3f}, conf: {confidence:.2f})")
                    return (best_semantic_match, confidence)
                else:
                    print(f"   ❌ Similaridade semântica insuficiente: {best_semantic_score:.3f} (mínimo: 0.35)")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        
        return None
    
    def match_objeto_conhecimento(self, text: str, disciplina: str = None, ano: str = None, unidade: str = None) -> Optional[Tuple[str, float]]:
        """Encontra objeto de conhecimento no texto com sinônimos"""
        if not disciplina or not ano:
            return None
        
        # Expandir consulta com sinônimos
        text_variations = expand_query(text)
        print(f"   📝 Buscando objeto com {len(text_variations)} variações...")
        
        # Extrair termos-chave do texto
        key_terms_text = get_key_terms(text)
        print(f"   🔑 Termos-chave: {key_terms_text}")
        
        try:
            # Se tem unidade, buscar apenas nela
            if unidade:
                objetos = self.bncc_data.get(disciplina, {}).get(ano, {}).get(unidade, {}).keys()
                print(f"   🎯 Buscando apenas na unidade: '{unidade[:50]}...'")
            else:
                # Buscar em todas as unidades
                unidades_data = self.bncc_data.get(disciplina, {}).get(ano, {})
                objetos = [obj for unidade_objs in unidades_data.values() for obj in unidade_objs.keys()]
                print(f"   🔍 Buscando em todas as unidades ({len(objetos)} objetos)")
            
            best_match = None
            best_score = 0
            
            for objeto in objetos:
                max_score_for_objeto = 0
                
                # Testar cada variação
                for idx, text_var in enumerate(text_variations):
                    # Extrair termos-chave COM PESOS
                    key_terms_var_weighted = get_key_terms(text_var, include_weights=True)
                    key_terms_objeto_weighted = get_key_terms(objeto, include_weights=True)
                    
                    # Termos em comum
                    key_terms_comuns = set(key_terms_var_weighted.keys()) & set(key_terms_objeto_weighted.keys())
                    
                    if key_terms_comuns:
                        # Score baseado em termos-chave PONDERADOS
                        peso_comuns = sum(key_terms_var_weighted.get(t, 1.0) for t in key_terms_comuns)
                        peso_total_objeto = sum(key_terms_objeto_weighted.values())
                        
                        score = peso_comuns / max(peso_total_objeto, 1.0)
                        
                        # Contar quantos termos de alto valor foram encontrados
                        high_value_matches = sum(1 for t in key_terms_comuns if key_terms_var_weighted.get(t, 1.0) >= 2.0)
                        
                        # Bonus progressivo para termos de alto valor
                        if high_value_matches >= 3:
                            score *= 2.0  # 3+ termos importantes
                        elif high_value_matches >= 2:
                            score *= 1.7  # 2 termos importantes
                        elif high_value_matches >= 1:
                            score *= 1.4  # 1 termo importante
                        
                        # Bonus se encontrou muitos termos no total
                        if len(key_terms_comuns) >= 3:
                            score *= 1.3
                        elif len(key_terms_comuns) >= 2:
                            score *= 1.2
                        
                        # Bonus extra se o match veio de uma variação com sinônimo
                        if idx > 0:  # Não é o texto original
                            score *= 1.3
                        
                        max_score_for_objeto = max(max_score_for_objeto, score)
                
                if max_score_for_objeto > best_score:
                    best_score = max_score_for_objeto
                    best_match = objeto
                    key_terms_objeto_weighted = get_key_terms(objeto, include_weights=True)
                    high_value = {k: v for k, v in key_terms_objeto_weighted.items() if v > 1.0}
                    print(f"   🎯 Candidato: '{objeto[:60]}...'")
                    print(f"      Score: {max_score_for_objeto:.3f} | Termos importantes: {high_value if high_value else 'nenhum'}")
            
            if best_match and best_score > 0.15:  # Threshold mais baixo para permitir matches parciais
                confidence = min(0.85, 0.55 + best_score * 0.30)
                print(f"   ✅ MATCH! Objeto: '{best_match[:60]}...'")
                print(f"      Score: {best_score:.3f}, Confiança: {confidence:.2f}")
                return (best_match, confidence)
            else:
                print(f"   ❌ Score por termos-chave insuficiente: {best_score:.3f}")
                print(f"   🔄 Tentando busca semântica...")
                
                # FALLBACK: Busca semântica usando embeddings
                best_semantic_match = None
                best_semantic_score = 0
                
                for objeto in objetos:
                    # Calcular similaridade semântica com cada variação
                    max_similarity = 0
                    for text_var in text_variations:
                        similarity = self._semantic_similarity(text_var, objeto)
                        max_similarity = max(max_similarity, similarity)
                    
                    if max_similarity > best_semantic_score:
                        best_semantic_score = max_similarity
                        best_semantic_match = objeto
                
                if best_semantic_match and best_semantic_score > 0.35:  # Threshold para similaridade semântica
                    confidence = min(0.80, 0.50 + best_semantic_score * 0.30)
                    print(f"   ✅ MATCH SEMÂNTICO! Objeto: '{best_semantic_match[:60]}...'")
                    print(f"      Similaridade: {best_semantic_score:.3f}, Confiança: {confidence:.2f}")
                    return (best_semantic_match, confidence)
                else:
                    print(f"   ❌ Similaridade semântica insuficiente: {best_semantic_score:.3f} (mínimo: 0.35)")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        
        return None
    
    def match_habilidade(self, text: str, disciplina: str = None, ano: str = None, unidade: str = None, objeto: str = None) -> Optional[Tuple[str, float]]:
        """Encontra habilidade baseada no contexto"""
        if not all([disciplina, ano, unidade, objeto]):
            return None
        
        try:
            habilidades = self.bncc_data.get(disciplina, {}).get(ano, {}).get(unidade, {}).get(objeto, [])
            
            if habilidades:
                # Se tem múltiplas habilidades, tentar fazer match com o texto
                if len(habilidades) > 1:
                    text_lower = text.lower()
                    best_match = None
                    best_score = 0
                    
                    for hab in habilidades:
                        hab_lower = hab.lower()
                        palavras_texto = set(text_lower.split())
                        palavras_hab = set(hab_lower.split())
                        palavras_comuns = palavras_texto & palavras_hab
                        
                        if palavras_comuns:
                            score = len(palavras_comuns)
                            if score > best_score:
                                best_score = score
                                best_match = hab
                    
                    if best_match:
                        return (best_match, 0.75)
                
                # Pegar a primeira habilidade como padrão
                return (habilidades[0], 0.70)
        except Exception as e:
            print(f"   Erro ao buscar habilidade: {e}")
        
        return None
    
    def _fuzzy_match_unidade(self, text: str, disciplina: str = None, ano: str = None) -> Optional[Tuple[str, float]]:
        """Busca fuzzy por unidade temática"""
        if not self.bncc_data or not disciplina or not ano:
            return None
        
        text_lower = text.lower()
        unidades = self.bncc_data.get(disciplina, {}).get(ano, {}).keys()
        
        for unidade in unidades:
            # Buscar palavras-chave da unidade no texto
            palavras_unidade = unidade.lower().split()
            matches = sum(1 for palavra in palavras_unidade if palavra in text_lower)
            
            if matches >= len(palavras_unidade) * 0.5:  # 50% das palavras
                return (unidade, 0.70)
        
        return None
    
    def _fuzzy_match_objeto(self, text: str, disciplina: str = None, ano: str = None, unidade: str = None) -> Optional[Tuple[str, float]]:
        """Busca fuzzy por objeto de conhecimento"""
        if not self.bncc_data or not all([disciplina, ano, unidade]):
            return None
        
        text_lower = text.lower()
        objetos = self.bncc_data.get(disciplina, {}).get(ano, {}).get(unidade, {}).keys()
        
        for objeto in objetos:
            palavras_objeto = objeto.lower().split()
            matches = sum(1 for palavra in palavras_objeto if palavra in text_lower)
            
            if matches >= len(palavras_objeto) * 0.5:
                return (objeto, 0.70)
        
        return None
    
    def _is_valid_unidade(self, unidade: str, disciplina: str = None, ano: str = None) -> bool:
        """Verifica se unidade é válida para disciplina/ano"""
        if not disciplina or not ano:
            return True
        
        try:
            unidades = self.bncc_data.get(disciplina, {}).get(ano, {}).keys()
            return unidade in unidades
        except:
            return False
    
    def _is_valid_objeto(self, objeto: str, disciplina: str = None, ano: str = None, unidade: str = None) -> bool:
        """Verifica se objeto é válido"""
        if not all([disciplina, ano, unidade]):
            return True
        
        try:
            objetos = self.bncc_data.get(disciplina, {}).get(ano, {}).get(unidade, {}).keys()
            return objeto in objetos
        except:
            return False
    
    def match_unidade_any_year(self, text: str, disciplina: str) -> Optional[Tuple[str, float]]:
        """Busca unidade temática em qualquer ano da disciplina usando busca semântica"""
        if not disciplina:
            return None
        
        print(f"   🔍 Buscando em TODOS os anos de {disciplina}...")
        
        # Expandir com sinônimos
        text_variations = expand_query(text)
        print(f"   📝 Variações: {text_variations[:3]}...")
        
        best_match_unidade = None
        best_match_ano = None
        best_score = 0
        best_objeto = None
        
        try:
            anos = self.bncc_data.get(disciplina, {})
            
            # FASE 1: Busca semântica (mais eficaz para textos curtos)
            print(f"   🔄 Usando busca semântica...")
            for ano, unidades in anos.items():
                for unidade, objetos in unidades.items():
                    for objeto in objetos.keys():
                        # Calcular similaridade semântica com cada variação
                        max_similarity = 0
                        for text_var in text_variations:
                            similarity = self._semantic_similarity(text_var, objeto)
                            max_similarity = max(max_similarity, similarity)
                        
                        if max_similarity > best_score:
                            best_score = max_similarity
                            best_match_unidade = unidade
                            best_match_ano = ano
                            best_objeto = objeto
            
            if best_match_unidade and best_score > 0.30:  # Threshold mais baixo para any_year
                confidence = min(0.80, 0.50 + best_score * 0.30)
                print(f"   ✅ MATCH SEMÂNTICO! Unidade: '{best_match_unidade[:60]}...'")
                print(f"      Ano: {best_match_ano}, Via objeto: '{best_objeto[:60]}...'")
                print(f"      Similaridade: {best_score:.3f}, Confiança: {confidence:.2f}")
                return (best_match_unidade, confidence)
            else:
                print(f"   ❌ Similaridade insuficiente: {best_score:.3f} (mínimo: 0.30)")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
        
        return None
    
    def get_ano_from_unidade(self, disciplina: str, unidade: str) -> Optional[str]:
        """Retorna o ano escolar de uma unidade temática"""
        try:
            anos = self.bncc_data.get(disciplina, {})
            for ano, unidades in anos.items():
                if unidade in unidades:
                    return ano
        except:
            pass
        return None
    
    def match_habilidade_any_year(self, text: str, disciplina: str, unidade: str = None, objeto: str = None) -> Optional[Tuple[str, float]]:
        """Busca habilidade em qualquer ano da disciplina"""
        if not disciplina:
            return None
        
        try:
            anos = self.bncc_data.get(disciplina, {})
            for ano, unidades in anos.items():
                if unidade and unidade in unidades:
                    objetos = unidades[unidade]
                    if objeto and objeto in objetos:
                        habilidades = objetos[objeto]
                        if habilidades:
                            return (habilidades[0], 0.65)
                    elif objetos:
                        # Pegar primeira habilidade da unidade
                        primeiro_objeto = list(objetos.values())[0]
                        if primeiro_objeto:
                            return (primeiro_objeto[0], 0.60)
        except:
            pass
        
        return None
    
    def get_all_for_context(self, disciplina: str, ano: str) -> Dict:
        """Retorna todas as opções disponíveis para um contexto"""
        try:
            data = self.bncc_data.get(disciplina, {}).get(ano, {})
            return {
                "unidades": list(data.keys()),
                "objetos": [obj for unidade in data.values() for obj in unidade.keys()],
                "total_habilidades": sum(len(habs) for unidade in data.values() for habs in unidade.values())
            }
        except:
            return {"unidades": [], "objetos": [], "total_habilidades": 0}
