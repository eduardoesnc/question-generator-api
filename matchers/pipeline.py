"""
Pipeline principal de classificação NLP
"""
from typing import Dict, Any, Optional
import re
import sys
import os

# Adicionar diretório pai ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers.disciplinas_matcher import DisciplinasMatcher
from matchers.bloom_matcher import BloomMatcher
from matchers.bncc_matcher import BNCCMatcher
from educational_mappings import (
    ANOS_MAP, TIPOS_QUESTAO_MAP, TIPOS_TEXTO_BASE_MAP, PERFIS_ALUNO_MAP
)


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
        print(f"\n{'='*60}")
        print(f"DEBUG: Processando texto: '{text}'")
        print(f"{'='*60}\n")
        
        text_lower = text.lower()
        
        extracted = {}
        confidence = {}
        suggestions = []
        
        # Usar contexto se fornecido
        if context:
            for key, value in context.items():
                if value:
                    extracted[key] = value
                    confidence[key] = 1.0
        
        # 🌍 BUSCA GLOBAL PRIMEIRO - tenta encontrar tudo de uma vez
        # Isso é especialmente útil para textos curtos como "Vargas", "Era Vargas", etc.
        if len(text.split()) <= 5:  # Textos curtos (até 5 palavras)
            print(f"\n🎯 Texto curto detectado - tentando busca global na BNCC...")
            global_result = self.bncc_matcher.search_global(text)
            if global_result:
                # Extrair tudo que foi encontrado
                for field in ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']:
                    if field in global_result and global_result[field]:
                        extracted[field] = global_result[field]
                        confidence[field] = global_result['confidence'][field]
                        print(f"   ✅ {field}: {str(global_result[field])[:60]}... (conf: {global_result['confidence'][field]:.2f})")
                
                # Se encontrou tudo na BNCC, pular extração individual
                if all(f in extracted for f in ['disciplina', 'ano', 'unidadeTematica', 'objetoConhecimento', 'habilidade']):
                    print(f"\n   🎉 BUSCA GLOBAL COMPLETA! Todos os campos BNCC encontrados.")
                    # Continuar para extrair apenas campos não-BNCC (bloom, tipo questão, etc.)
                else:
                    print(f"\n   ⚠️  Busca global parcial - continuando extração normal...")
            else:
                print(f"   ❌ Busca global não encontrou matches - continuando extração normal...")
        
        # Extrair disciplina com PhraseMatcher
        if "disciplina" not in extracted:
            disc_result = self.disciplinas_matcher.match(text_lower)
            if disc_result:
                extracted["disciplina"] = disc_result[0]
                confidence["disciplina"] = disc_result[1]
                print(f"✅ Disciplina: {disc_result[0]} (confiança: {disc_result[1]:.2f})")
            else:
                print("❌ Disciplina não encontrada")
        
        # Extrair ano escolar (regex) - usar texto original para preservar números
        if "ano" not in extracted:
            ano_result = self._extract_ano(text)  # Usar texto original, não lowercase
            if ano_result:
                extracted["ano"] = ano_result["value"]
                confidence["ano"] = ano_result["confidence"]
                print(f"✅ Ano: {ano_result['value']} (confiança: {ano_result['confidence']:.2f})")
            else:
                print("❌ Ano não encontrado")
        
        # Extrair nível Bloom com PhraseMatcher
        if "nivelBloom" not in extracted:
            bloom_result = self.bloom_matcher.match(text_lower)
            if bloom_result:
                extracted["nivelBloom"] = bloom_result[0]
                confidence["nivelBloom"] = bloom_result[1]
                print(f"✅ Nível Bloom: {bloom_result[0]} (confiança: {bloom_result[1]:.2f})")
            else:
                print("❌ Nível Bloom não encontrado")
        
        # Extrair tipo de questão (keyword matching)
        if "tipoQuestao" not in extracted:
            tipo_q = self._extract_by_keywords(text_lower, TIPOS_QUESTAO_MAP)
            if tipo_q:
                extracted["tipoQuestao"] = tipo_q["value"]
                confidence["tipoQuestao"] = tipo_q["confidence"]
                print(f"✅ Tipo Questão: {tipo_q['value']} (confiança: {tipo_q['confidence']:.2f})")
            else:
                print("❌ Tipo Questão não encontrado")
        
        # Extrair tipo de texto base
        if "tipoTextoBase" not in extracted:
            tipo_t = self._extract_by_keywords(text_lower, TIPOS_TEXTO_BASE_MAP)
            if tipo_t:
                extracted["tipoTextoBase"] = tipo_t["value"]
                confidence["tipoTextoBase"] = tipo_t["confidence"]
                print(f"✅ Tipo Texto Base: {tipo_t['value']} (confiança: {tipo_t['confidence']:.2f})")
            else:
                print("❌ Tipo Texto Base não encontrado")
        
        # Extrair perfil do aluno
        if "perfilAluno" not in extracted:
            perfil = self._extract_by_keywords(text_lower, PERFIS_ALUNO_MAP)
            if perfil:
                extracted["perfilAluno"] = perfil["value"]
                confidence["perfilAluno"] = perfil["confidence"]
                print(f"✅ Perfil Aluno: {perfil['value']} (confiança: {perfil['confidence']:.2f})")
            else:
                print("❌ Perfil Aluno não encontrado")
        
        # Extrair Unidade Temática da BNCC (ou tópicos livres)
        if "unidadeTematica" not in extracted:
            disciplina = extracted.get("disciplina")
            ano = extracted.get("ano")
            
            print(f"\n🔍 Tentando extrair Unidade Temática...")
            print(f"   Disciplina: {disciplina}, Ano: {ano}")
            
            # Se não tem ano mas tem disciplina, tentar buscar em todos os anos
            if disciplina and not ano:
                print(f"   ⚙️  Chamando match_unidade_any_year('{text}', '{disciplina}')...")
                unidade_result = self.bncc_matcher.match_unidade_any_year(text, disciplina)
                print(f"   ⚙️  Resultado: {unidade_result}")
                if unidade_result:
                    extracted["unidadeTematica"] = unidade_result[0]
                    confidence["unidadeTematica"] = unidade_result[1]
                    # Se encontrou unidade, tentar inferir o ano
                    ano_inferido = self.bncc_matcher.get_ano_from_unidade(disciplina, unidade_result[0])
                    if ano_inferido and "ano" not in extracted:
                        extracted["ano"] = ano_inferido
                        confidence["ano"] = 0.75
                        print(f"✅ Ano inferido: {ano_inferido} (confiança: 0.75)")
                    print(f"✅ Unidade Temática (BNCC): {unidade_result[0]} (confiança: {unidade_result[1]:.2f})")
            # Primeiro tentar na BNCC com ano específico
            elif disciplina and ano:
                unidade_result = self.bncc_matcher.match_unidade_tematica(text, disciplina, ano)
                if unidade_result:
                    extracted["unidadeTematica"] = unidade_result[0]
                    confidence["unidadeTematica"] = unidade_result[1]
                    print(f"✅ Unidade Temática (BNCC): {unidade_result[0]} (confiança: {unidade_result[1]:.2f})")
                else:
                    # BNCC não encontrou nada - não usar tópicos livres
                    # Deixar vazio para o usuário preencher manualmente
                    print("❌ Unidade Temática não encontrada na BNCC")
            else:
                print("⚠️  Unidade Temática: precisa de disciplina primeiro")
        
        # Extrair Objeto de Conhecimento
        if "objetoConhecimento" not in extracted:
            disciplina = extracted.get("disciplina")
            ano = extracted.get("ano")
            unidade = extracted.get("unidadeTematica")
            
            print(f"\n🔍 Tentando extrair Objeto de Conhecimento...")
            print(f"   Disciplina: {disciplina}, Ano: {ano}, Unidade: {unidade}")
            
            if disciplina and ano:
                objeto_result = self.bncc_matcher.match_objeto_conhecimento(text, disciplina, ano, unidade)
                if objeto_result:
                    extracted["objetoConhecimento"] = objeto_result[0]
                    confidence["objetoConhecimento"] = objeto_result[1]
                    print(f"✅ Objeto Conhecimento (BNCC): {objeto_result[0][:80]}... (confiança: {objeto_result[1]:.2f})")
                else:
                    print("❌ Objeto de Conhecimento não encontrado na BNCC")
            else:
                print("⚠️  Objeto Conhecimento: precisa de disciplina e ano primeiro")
        
        # Extrair Habilidade
        if "habilidade" not in extracted:
            disciplina = extracted.get("disciplina")
            ano = extracted.get("ano")
            unidade = extracted.get("unidadeTematica")
            objeto = extracted.get("objetoConhecimento")
            
            print(f"\n🔍 Tentando extrair Habilidade...")
            print(f"   Disciplina: {disciplina}, Ano: {ano}")
            print(f"   Unidade: {unidade}, Objeto: {objeto}")
            
            if all([disciplina, ano, unidade, objeto]):
                habilidade_result = self.bncc_matcher.match_habilidade(text, disciplina, ano, unidade, objeto)
                if habilidade_result:
                    extracted["habilidade"] = habilidade_result[0]
                    confidence["habilidade"] = habilidade_result[1]
                    print(f"✅ Habilidade (BNCC): {habilidade_result[0][:50]}... (confiança: {habilidade_result[1]:.2f})")
                else:
                    # Se não encontrou na BNCC, buscar em qualquer ano da mesma disciplina
                    print("   Buscando habilidade em outros anos...")
                    habilidade_any = self.bncc_matcher.match_habilidade_any_year(text, disciplina, unidade, objeto)
                    if habilidade_any:
                        extracted["habilidade"] = habilidade_any[0]
                        confidence["habilidade"] = habilidade_any[1]
                        print(f"✅ Habilidade (outro ano): {habilidade_any[0][:50]}... (confiança: {habilidade_any[1]:.2f})")
                    else:
                        # Gerar habilidade genérica baseada no contexto
                        habilidade_generica = f"Compreender e analisar {objeto} no contexto de {unidade}"
                        extracted["habilidade"] = habilidade_generica
                        confidence["habilidade"] = 0.50
                        print(f"✅ Habilidade (genérica): {habilidade_generica} (confiança: 0.50)")
            else:
                print("⚠️  Habilidade: precisa de disciplina, ano, unidade e objeto primeiro")
        
        # Extrair tópicos livres como sugestões (fallback se não encontrou na BNCC)
        if "unidadeTematica" not in extracted:
            topicos = self._extract_free_topics(text)
            if topicos:
                suggestions.append({
                    "field": "unidadeTematica",
                    "values": topicos,
                    "message": "Tópicos identificados no texto (não encontrados na BNCC)"
                })
        
        # Aplicar defaults inteligentes
        self._apply_smart_defaults(extracted, confidence, text_lower)
        
        print(f"\n{'='*60}")
        print(f"📊 RESULTADO FINAL:")
        print(f"{'='*60}")
        for field, value in extracted.items():
            conf = confidence.get(field, 0)
            value_display = value if len(str(value)) < 50 else str(value)[:50] + "..."
            print(f"  {field}: {value_display} (conf: {conf:.2f})")
        print(f"\n❌ Campos faltantes: {missing_fields if 'missing_fields' in locals() else 'calculando...'}")
        print(f"{'='*60}\n")
        
        # Identificar campos faltantes (TODOS os 10 campos)
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
                    print(f"DEBUG: Ano encontrado: {ano} com padrão '{pattern}' em '{text}'")
                    return {"value": ano, "confidence": 0.95}
        print(f"DEBUG: Nenhum ano encontrado em '{text}'")
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
                    
                    # Palavra completa aumenta confiança
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
        
        # Palavras que NÃO devem ser consideradas tópicos (tipos de questão, texto base, etc.)
        blacklist = {
            'documento histórico', 'documento historico', 'texto literário', 'texto literario',
            'artigo jornal', 'charge', 'gráfico', 'grafico', 'tabela', 'imagem', 'mapa',
            'múltipla escolha', 'multipla escolha', 'dissertativa', 'verdadeiro falso',
            'análise', 'analise', 'síntese', 'sintese', 'aplicação', 'aplicacao',
            'conhecimento', 'compreensão', 'compreensao', 'avaliação', 'avaliacao'
        }
        
        # Entidades nomeadas
        for ent in doc.ents:
            if ent.label_ in ["PER", "ORG", "LOC", "EVENT", "MISC"]:
                ent_lower = ent.text.lower()
                # Filtrar entidades que são anos, tipos de questão ou texto base
                if (not re.match(r'^\d+[º°]?\s*ano', ent_lower) and
                    not any(bl in ent_lower for bl in blacklist)):
                    topics.add(ent.text.title())
        
        # Noun chunks relevantes (2+ palavras)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            chunk_lower = chunk_text.lower()
            # Filtrar chunks que são anos escolares, tipos de questão ou muito genéricos
            if (len(chunk_text.split()) >= 2 and 
                not re.match(r'^\d+[º°]?\s*ano', chunk_lower) and
                not any(bl in chunk_lower for bl in blacklist) and
                chunk.root.pos_ == "NOUN" and 
                not chunk.root.is_stop):
                topics.add(chunk_text.title())
        
        # Se não encontrou nada, tentar extrair substantivos próprios simples
        if not topics:
            for token in doc:
                token_lower = token.text.lower()
                if (token.pos_ == "PROPN" and 
                    not token.is_stop and 
                    not token.text.isdigit() and
                    not any(bl in token_lower for bl in blacklist)):
                    topics.add(token.text.title())
        
        # Filtrar tópicos vazios ou muito curtos
        topics = {t for t in topics if len(t.strip()) > 2 and not t.strip().startswith(',')}
        
        return sorted(list(topics))[:5]
    
    def _apply_smart_defaults(self, extracted: Dict, confidence: Dict, text: str):
        """Aplica defaults inteligentes"""
        
        # Perfil baseado no ano
        if "perfilAluno" not in extracted and "ano" in extracted:
            ano = extracted["ano"]
            if any(x in ano for x in ["1º", "2º", "3º", "4º", "5º"]):
                extracted["perfilAluno"] = "conhecimento_basico"
                confidence["perfilAluno"] = 0.6
            elif any(x in ano for x in ["6º", "7º", "8º", "9º"]):
                extracted["perfilAluno"] = "bom_dominio"
                confidence["perfilAluno"] = 0.6
        
        # Tipo de questão baseado em palavras-chave
        if "tipoQuestao" not in extracted:
            if any(w in text for w in ["alternativa", "opção", "opcao", "a)", "b)"]):
                extracted["tipoQuestao"] = "multipla_escolha"
                confidence["tipoQuestao"] = 0.65
            elif any(w in text for w in ["explique", "desenvolva", "argumente"]):
                extracted["tipoQuestao"] = "dissertativa_longa"
                confidence["tipoQuestao"] = 0.65
        
        # Nível Bloom baseado em verbos
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
        
        # Tipo de texto baseado na disciplina
        if "tipoTextoBase" not in extracted and "disciplina" in extracted:
            disc = extracted["disciplina"].lower()
            if "matemática" in disc or "matematica" in disc:
                extracted["tipoTextoBase"] = "grafico_barras"
                confidence["tipoTextoBase"] = 0.55
            elif "história" in disc or "historia" in disc:
                extracted["tipoTextoBase"] = "documento_historico"
                confidence["tipoTextoBase"] = 0.55
            elif "geografia" in disc:
                extracted["tipoTextoBase"] = "mapa"
                confidence["tipoTextoBase"] = 0.55
            elif "português" in disc or "portugues" in disc:
                extracted["tipoTextoBase"] = "texto_literario"
                confidence["tipoTextoBase"] = 0.55
