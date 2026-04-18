"""
Pipeline principal de classificação NLP
"""
from typing import Dict, Any, Optional
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers.disciplinas_matcher import DisciplinasMatcher
from matchers.bloom_matcher import BloomMatcher
from matchers.bncc_matcher import BNCCMatcher
from app.core.mappings import (
    ANOS_MAP, TIPOS_QUESTAO_MAP, TIPOS_TEXTO_BASE_MAP, PERFIS_ALUNO_MAP
)
from app.core.logging import logger
from app.models.responses import Suggestion

class NLPPipeline:
    """Pipeline de processamento NLP para extração educacional"""
    
    def __init__(self, nlp):
        self.nlp = nlp
        self.disciplinas_matcher = DisciplinasMatcher(nlp)
        self.bloom_matcher = BloomMatcher(nlp)
        self.bncc_matcher = BNCCMatcher(nlp)
    
    def classify(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Classifica o texto e extrai todas as informações educacionais
        
        Returns:
            Dict com extracted, confidence, suggestions, missing_fields
        """
        
        text_lower = text.lower()
        
        extracted = {}
        confidence = {}
        suggestions = []
        
        if context:
            for key, value in context.items():
                if value:
                    extracted[key] = value
                    confidence[key] = 1.0
        
        if len(text.split()) <= 5:
            global_result = self.bncc_matcher.search_global(text)
            if global_result:
                for field in ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']:
                    if field in global_result and global_result[field]:
                        extracted[field] = global_result[field]
                        confidence[field] = global_result['confidence'][field]
                
                if all(f in extracted for f in ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']):
                    pass
                else:
                    logger.warning(" Busca global parcial - continuando extração normal...")
            else:
                pass
        
        if "disciplina" not in extracted:
            disc_result = self.disciplinas_matcher.match(text_lower)
            if disc_result:
                extracted["disciplina"] = disc_result[0]
                confidence["disciplina"] = disc_result[1]
            else:
                pass
        
        if "ano" not in extracted:
            ano_result = self._extract_ano(text)
            if ano_result:
                extracted["ano"] = ano_result["value"]
                confidence["ano"] = ano_result["confidence"]
            else:
                pass
        
        if "nivelBloom" not in extracted:
            bloom_result = self.bloom_matcher.match(text_lower)
            if bloom_result:
                extracted["nivelBloom"] = bloom_result[0]
                confidence["nivelBloom"] = bloom_result[1]
            else:
                pass
        
        if "tipoQuestao" not in extracted:
            tipo_q = self._extract_by_keywords(text_lower, TIPOS_QUESTAO_MAP)
            if tipo_q:
                extracted["tipoQuestao"] = tipo_q["value"]
                confidence["tipoQuestao"] = tipo_q["confidence"]
            else:
                pass
        
        if "tipoTextoBase" not in extracted:
            tipo_t = self._extract_by_keywords(text_lower, TIPOS_TEXTO_BASE_MAP)
            if tipo_t:
                extracted["tipoTextoBase"] = tipo_t["value"]
                confidence["tipoTextoBase"] = tipo_t["confidence"]
            else:
                pass
        
        if "perfilAluno" not in extracted:
            perfil = self._extract_by_keywords(text_lower, PERFIS_ALUNO_MAP)
            if perfil:
                extracted["perfilAluno"] = perfil["value"]
                confidence["perfilAluno"] = perfil["confidence"]
            else:
                pass
        
        if "unidadeTematica" not in extracted:
            disciplina = extracted.get("disciplina")
            ano = extracted.get("ano")
            
            if disciplina and not ano:
                unidade_result = self.bncc_matcher.match_unidade_any_year(text, disciplina)
                if unidade_result:
                    extracted["unidadeTematica"] = unidade_result[0]
                    confidence["unidadeTematica"] = unidade_result[1]
                    ano_inferido = self.bncc_matcher.get_ano_from_unidade(disciplina, unidade_result[0])
                    if ano_inferido and "ano" not in extracted:
                        extracted["ano"] = ano_inferido
                        confidence["ano"] = 0.75
            elif disciplina and ano:
                unidade_result = self.bncc_matcher.match_unidade_tematica(text, disciplina, ano)
                if unidade_result:
                    extracted["unidadeTematica"] = unidade_result[0]
                    confidence["unidadeTematica"] = unidade_result[1]
                else:
                    pass
            else:
                pass
        
        if "objetoConhecimento" not in extracted:
            disciplina = extracted.get("disciplina")
            ano = extracted.get("ano")
            unidade = extracted.get("unidadeTematica")
            
            if disciplina and ano:
                objeto_result = self.bncc_matcher.match_objeto_conhecimento(text, disciplina, ano, unidade)
                if objeto_result:
                    extracted["objetoConhecimento"] = objeto_result[0]
                    confidence["objetoConhecimento"] = objeto_result[1]
                else:
                    pass
            else:
                pass
        
        if "habilidade" not in extracted:
            disciplina = extracted.get("disciplina")
            ano = extracted.get("ano")
            unidade = extracted.get("unidadeTematica")
            objeto = extracted.get("objetoConhecimento")
            
            if all([disciplina, ano, unidade, objeto]):
                habilidade_result = self.bncc_matcher.match_habilidade(text, disciplina, ano, unidade, objeto)
                if habilidade_result:
                    extracted["habilidade"] = habilidade_result[0]
                    confidence["habilidade"] = habilidade_result[1]
                else:
                    habilidade_any = self.bncc_matcher.match_habilidade_any_year(text, disciplina, unidade, objeto)
                    if habilidade_any:
                        extracted["habilidade"] = habilidade_any[0]
                        confidence["habilidade"] = habilidade_any[1]
                    else:
                        habilidade_generica = f"Compreender e analisar {objeto} no contexto de {unidade}"
                        extracted["habilidade"] = habilidade_generica
                        confidence["habilidade"] = 0.50
            else:
                pass
        
        if "unidadeTematica" not in extracted:
            topicos = self._extract_free_topics(text)
            if topicos:
                suggestions.append(Suggestion(
                    field="unidadeTematica",
                    values=topicos,
                    message="Tópicos identificados no texto (não encontrados na BNCC)"
                ))
        
        self._apply_smart_defaults(extracted, confidence, text_lower)
        
        self._validate_consistency(extracted, confidence)
        
        for field, value in extracted.items():
            conf = confidence.get(field, 0)
            value_display = value if len(str(value)) < 50 else str(value)[:50] + "..."
        
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
    
    def _extract_ano(self, text: str) -> Optional[Dict[str, Any]]:
        """Extrai ano escolar usando regex"""
        for ano, patterns in ANOS_MAP.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return {"value": ano, "confidence": 0.95}
        return None
    
    def _extract_by_keywords(self, text: str, mapping: Dict) -> Optional[Dict[str, Any]]:
        """Extração genérica por keywords"""
        best_match = None
        best_confidence = 0.0
        best_length = 0
        
        for category, keywords in mapping.items():
            for keyword in keywords:
                if keyword in text:
                    length = len(keyword)
                    confidence = 0.80 + min(0.15, length / 100)
                    
                    if f" {keyword} " in f" {text} ":
                        confidence = min(0.98, confidence + 0.1)
                    
                    if confidence > best_confidence or (confidence == best_confidence and length > best_length):
                        best_confidence = confidence
                        best_match = category
                        best_length = length
        
        if best_match:
            return {"value": best_match, "confidence": best_confidence}
        return None
    
    def _extract_free_topics(self, text: str) -> list:
        """Extrai tópicos livres usando NER e noun chunks"""
        doc = self.nlp(text)
        topics = set()
        
        blacklist = {
            'documento histórico', 'documento historico', 'texto literário', 'texto literario',
            'artigo jornal', 'charge', 'gráfico', 'grafico', 'tabela', 'imagem', 'mapa',
            'múltipla escolha', 'multipla escolha', 'dissertativa', 'verdadeiro falso',
            'análise', 'analise', 'síntese', 'sintese', 'aplicação', 'aplicacao',
            'conhecimento', 'compreensão', 'compreensao', 'avaliação', 'avaliacao'
        }
        
        for ent in doc.ents:
            if ent.label_ in ["PER", "ORG", "LOC", "EVENT", "MISC"]:
                ent_lower = ent.text.lower()
                if (not re.match(r'^\d+[º°]?\s*ano', ent_lower) and
                    not any(bl in ent_lower for bl in blacklist)):
                    topics.add(ent.text.title())
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            chunk_lower = chunk_text.lower()
            if (len(chunk_text.split()) >= 2 and 
                not re.match(r'^\d+[º°]?\s*ano', chunk_lower) and
                not any(bl in chunk_lower for bl in blacklist) and
                chunk.root.pos_ == "NOUN" and 
                not chunk.root.is_stop):
                topics.add(chunk_text.title())
        
        if not topics:
            for token in doc:
                token_lower = token.text.lower()
                if (token.pos_ == "PROPN" and 
                    not token.is_stop and 
                    not token.text.isdigit() and
                    not any(bl in token_lower for bl in blacklist)):
                    topics.add(token.text.title())
        
        topics = {t for t in topics if len(t.strip()) > 2 and not t.strip().startswith(',')}
        
        return sorted(list(topics))[:5]
    
    def _apply_smart_defaults(self, extracted: Dict, confidence: Dict, text: str):
        """Aplica defaults inteligentes"""
        
        if "perfilAluno" not in extracted and "ano" in extracted:
            ano = extracted["ano"]
            if any(x in ano for x in ["1º", "2º", "3º", "4º", "5º"]):
                extracted["perfilAluno"] = "conhecimento_basico"
                confidence["perfilAluno"] = 0.6
            elif any(x in ano for x in ["6º", "7º", "8º", "9º"]):
                extracted["perfilAluno"] = "bom_dominio"
                confidence["perfilAluno"] = 0.6
        
        if "tipoQuestao" not in extracted:
            if any(w in text for w in ["alternativa", "opção", "opcao", "a)", "b)"]):
                extracted["tipoQuestao"] = "multipla_escolha"
                confidence["tipoQuestao"] = 0.65
            elif any(w in text for w in ["explique", "desenvolva", "argumente"]):
                extracted["tipoQuestao"] = "dissertativa_longa"
                confidence["tipoQuestao"] = 0.65
        
        if "nivelBloom" not in extracted:
            if any(w in text for w in ["compare", "relacione", "diferencie", "analise"]):
                extracted["nivelBloom"] = "analise"
                confidence["nivelBloom"] = 0.65
            elif any(w in text for w in ["calcule", "resolva", "aplique"]):
                extracted["nivelBloom"] = "aplicacao"
                confidence["nivelBloom"] = 0.65
            else:
                extracted["nivelBloom"] = "compreensao"
                confidence["nivelBloom"] = 0.5
        
    def _validate_consistency(self, extracted: Dict, confidence: Dict):
        """
        Valida consistência entre campos extraídos
        Reduz confiança se houver inconsistências
        """
        
        if all(k in extracted for k in ['disciplina', 'ano', 'unidadeTematica']):
            disciplina = extracted['disciplina']
            ano = extracted['ano']
            unidade = extracted['unidadeTematica']
            
            if disciplina in self.bncc_matcher.bncc_data:
                if ano in self.bncc_matcher.bncc_data[disciplina]:
                    if unidade not in self.bncc_matcher.bncc_data[disciplina][ano]:
                        logger.warning(f"  Inconsistência: Unidade '{unidade}' não existe em {disciplina} {ano}")
                        if 'unidadeTematica' in confidence:
                            confidence['unidadeTematica'] *= 0.7
        
        if 'disciplina' in extracted and 'tipoTextoBase' in extracted:
            disc = extracted['disciplina'].lower()
            tipo = extracted['tipoTextoBase']
            
            incompatible = False
            if 'matemática' in disc or 'matematica' in disc:
                if tipo in ['documento_historico', 'texto_literario', 'poema']:
                    incompatible = True
            elif 'história' in disc or 'historia' in disc:
                if tipo in ['grafico_barras', 'grafico_linhas']:
                    incompatible = True
            
            if incompatible:
                logger.warning(f"  Inconsistência: Tipo de texto '{tipo}' incomum para {extracted['disciplina']}")
                if 'tipoTextoBase' in confidence:
                    confidence['tipoTextoBase'] *= 0.8
        
        if 'ano' in extracted and 'nivelBloom' in extracted:
            ano = extracted['ano']
            bloom = extracted['nivelBloom']
            
            if any(x in ano for x in ["1º", "2º", "3º"]):
                if bloom in ['sintese', 'avaliacao']:
                    logger.warning(f"  Inconsistência: Nível Bloom '{bloom}' avançado para {ano}")
                    if 'nivelBloom' in confidence:
                        confidence['nivelBloom'] *= 0.7
        
        if 'ano' in extracted and 'perfilAluno' in extracted:
            ano = extracted['ano']
            perfil = extracted['perfilAluno']
            
            if any(x in ano for x in ["1º", "2º", "3º"]):
                if perfil == 'conhecimento_avancado':
                    logger.warning(f"  Inconsistência: Perfil '{perfil}' avançado para {ano}")
                    if 'perfilAluno' in confidence:
                        confidence['perfilAluno'] *= 0.8
        
        if 'nivelBloom' in extracted and 'tipoQuestao' in extracted:
            bloom = extracted['nivelBloom']
            tipo_q = extracted['tipoQuestao']
            
            if bloom in ['sintese', 'avaliacao'] and tipo_q == 'multipla_escolha':
                logger.warning(f"  Inconsistência: Múltipla escolha para nível '{bloom}'")
                if 'tipoQuestao' in confidence:
                    confidence['tipoQuestao'] *= 0.8
        
