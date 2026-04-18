"""
Gera automaticamente exemplos de treino baseados na BNCC
Cria exemplos positivos (similares) e negativos (não similares)
Inclui exemplos para: BNCC, Bloom, Tipos de Questão, Tipos de Texto Base
"""
import json
import os
import random
import sys

# Adicionar path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matchers.synonyms import SYNONYMS_MAP
from app.core.mappings import NIVEIS_BLOOM_MAP, TIPOS_QUESTAO_MAP, TIPOS_TEXTO_BASE_MAP

def load_bncc_data():
    """Carrega dados da BNCC"""
    script_dir = os.path.dirname(__file__)
    bncc_path = os.path.join(script_dir, '..', 'data', 'bncc-data.json')
    
    with open(bncc_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_key_terms(text):
    """Extrai termos-chave de um texto"""
    # Palavras genéricas para ignorar
    stop_words = {
        'o', 'a', 'os', 'as', 'de', 'da', 'do', 'das', 'dos',
        'em', 'na', 'no', 'nas', 'nos', 'e', 'ou', 'para', 'com'
    }
    
    words = text.lower().split()
    key_terms = [w for w in words if len(w) > 3 and w not in stop_words]
    return key_terms[:3]  # Top 3 termos

def generate_positive_examples(bncc_data):
    """Gera exemplos positivos (similares)"""
    examples = []
    
    print("📝 Gerando exemplos POSITIVOS...")
    
    for disciplina, anos in bncc_data.items():
        for ano, unidades in anos.items():
            for unidade, objetos in unidades.items():
                for objeto in objetos.keys():
                    # Extrair termos-chave do objeto
                    key_terms = extract_key_terms(objeto)
                    
                    if not key_terms:
                        continue
                    
                    # Exemplo 1: Termo principal → Objeto completo
                    main_term = key_terms[0]
                    examples.append({
                        "text1": main_term.title(),
                        "text2": objeto,
                        "score": 0.85,
                        "type": "main_term",
                        "disciplina": disciplina
                    })
                    
                    # Exemplo 2: Combinação de termos → Objeto
                    if len(key_terms) >= 2:
                        combined = f"{key_terms[0]} {key_terms[1]}"
                        examples.append({
                            "text1": combined.title(),
                            "text2": objeto,
                            "score": 0.90,
                            "type": "combined_terms",
                            "disciplina": disciplina
                        })
                    
                    # Exemplo 3: Usar sinônimos do dicionário
                    for term in key_terms:
                        if term in SYNONYMS_MAP:
                            for synonym in SYNONYMS_MAP[term][:2]:  # Top 2 sinônimos
                                examples.append({
                                    "text1": synonym.title(),
                                    "text2": objeto,
                                    "score": 0.95,
                                    "type": "synonym",
                                    "disciplina": disciplina
                                })
    
    print(f"   ✅ {len(examples)} exemplos positivos gerados")
    return examples

def generate_negative_examples(bncc_data):
    """Gera exemplos negativos (não similares)"""
    examples = []
    
    print("📝 Gerando exemplos NEGATIVOS...")
    
    # Coletar todos os objetos por disciplina
    objetos_por_disciplina = {}
    for disciplina, anos in bncc_data.items():
        objetos_por_disciplina[disciplina] = []
        for ano, unidades in anos.items():
            for unidade, objetos in unidades.items():
                objetos_por_disciplina[disciplina].extend(objetos.keys())
    
    # Tipo 1: Disciplinas diferentes (score 0.0)
    disciplinas = list(objetos_por_disciplina.keys())
    for i, disc1 in enumerate(disciplinas):
        for disc2 in disciplinas[i+1:]:
            examples.append({
                "text1": disc1,
                "text2": disc2,
                "score": 0.0,
                "type": "different_disciplines",
                "disciplina": "cross"
            })
    
    # Tipo 2: Objetos de disciplinas diferentes (score 0.1-0.2)
    for _ in range(50):  # 50 exemplos
        disc1, disc2 = random.sample(disciplinas, 2)
        if objetos_por_disciplina[disc1] and objetos_por_disciplina[disc2]:
            obj1 = random.choice(objetos_por_disciplina[disc1])
            obj2 = random.choice(objetos_por_disciplina[disc2])
            
            # Extrair termo do obj1
            terms1 = extract_key_terms(obj1)
            if terms1:
                examples.append({
                    "text1": terms1[0].title(),
                    "text2": obj2,
                    "score": 0.1,
                    "type": "cross_discipline_object",
                    "disciplina": "cross"
                })
    
    # Tipo 3: Objetos da mesma disciplina mas não relacionados (score 0.2-0.4)
    for disciplina, objetos_list in objetos_por_disciplina.items():
        if len(objetos_list) < 10:
            continue
        
        # Pegar 20 pares aleatórios
        for _ in range(20):
            obj1, obj2 = random.sample(objetos_list, 2)
            
            # Verificar se são realmente diferentes
            terms1 = set(extract_key_terms(obj1))
            terms2 = set(extract_key_terms(obj2))
            
            # Se não têm termos em comum, são bem diferentes
            if not (terms1 & terms2):
                term1 = extract_key_terms(obj1)
                if term1:
                    examples.append({
                        "text1": term1[0].title(),
                        "text2": obj2,
                        "score": random.uniform(0.2, 0.4),
                        "type": "same_discipline_different",
                        "disciplina": disciplina
                    })
    
    # Tipo 4: Termos genéricos vs objetos específicos (score 0.3-0.5)
    generic_terms = [
        "História", "Matemática", "Ciências", "Geografia", "Português",
        "Educação", "Ensino", "Aprendizagem", "Conhecimento", "Habilidade"
    ]
    
    for disciplina, objetos_list in objetos_por_disciplina.items():
        if not objetos_list:
            continue
        
        for term in generic_terms[:5]:  # Top 5 termos genéricos
            obj = random.choice(objetos_list)
            examples.append({
                "text1": term,
                "text2": obj,
                "score": random.uniform(0.3, 0.5),
                "type": "generic_vs_specific",
                "disciplina": disciplina
            })
    
    print(f"   ✅ {len(examples)} exemplos negativos gerados")
    return examples

def generate_bloom_examples():
    """Gera exemplos para níveis de Bloom"""
    examples = []
    
    print("📝 Gerando exemplos de NÍVEIS BLOOM...")
    
    # Exemplos positivos: frases típicas → nível Bloom
    bloom_phrases = {
        "conhecimento": [
            "Memorizar datas históricas importantes",
            "Listar os estados brasileiros",
            "Definir o conceito de fotossíntese",
            "Identificar as partes do corpo humano",
            "Recordar eventos da Segunda Guerra",
            "Nomear os planetas do sistema solar"
        ],
        "compreensao": [
            "Explicar o processo de fotossíntese",
            "Interpretar um texto literário",
            "Resumir os principais eventos da Era Vargas",
            "Descrever o ciclo da água",
            "Compreender as causas da Revolução Industrial",
            "Parafrasear um poema"
        ],
        "aplicacao": [
            "Resolver equações do segundo grau",
            "Aplicar regras gramaticais em frases",
            "Calcular a área de um triângulo",
            "Usar fórmulas matemáticas em problemas",
            "Demonstrar experimentos científicos",
            "Executar operações com frações"
        ],
        "analise": [
            "Analisar as causas da guerra",
            "Comparar diferentes sistemas políticos",
            "Diferenciar células vegetais e animais",
            "Examinar dados de um gráfico",
            "Investigar relações entre variáveis",
            "Relacionar eventos históricos"
        ],
        "sintese": [
            "Criar um projeto de ciências",
            "Desenvolver uma redação argumentativa",
            "Elaborar um plano de ação",
            "Construir uma maquete",
            "Produzir um vídeo educativo",
            "Planejar uma apresentação"
        ],
        "avaliacao": [
            "Avaliar a qualidade de um argumento",
            "Julgar a validade de uma teoria",
            "Criticar uma obra literária",
            "Justificar uma decisão política",
            "Argumentar sobre questões ambientais",
            "Defender um ponto de vista"
        ]
    }
    
    # Positivos: frase → nível correto (score alto)
    for nivel, frases in bloom_phrases.items():
        for frase in frases:
            examples.append({
                "text1": frase,
                "text2": nivel,
                "score": 0.95,
                "type": "bloom_positive",
                "disciplina": "bloom"
            })
    
    # Negativos: frase → nível errado (score baixo)
    niveis = list(bloom_phrases.keys())
    for nivel_correto, frases in bloom_phrases.items():
        for frase in frases[:2]:  # 2 frases por nível
            nivel_errado = random.choice([n for n in niveis if n != nivel_correto])
            examples.append({
                "text1": frase,
                "text2": nivel_errado,
                "score": 0.1,
                "type": "bloom_negative",
                "disciplina": "bloom"
            })
    
    print(f"   ✅ {len(examples)} exemplos de Bloom gerados")
    return examples

def generate_question_type_examples():
    """Gera exemplos para tipos de questão"""
    examples = []
    
    print("📝 Gerando exemplos de TIPOS DE QUESTÃO...")
    
    question_phrases = {
        "multipla_escolha": [
            "Marque a alternativa correta",
            "Assinale a opção que apresenta",
            "Escolha entre as alternativas a), b), c)",
            "Selecione a resposta adequada",
            "Qual das opções abaixo está correta?",
            "Indique a alternativa verdadeira"
        ],
        "dissertativa_curta": [
            "Responda brevemente",
            "Explique em poucas palavras",
            "Descreva resumidamente",
            "Cite dois exemplos",
            "Defina o conceito",
            "Responda de forma objetiva"
        ],
        "dissertativa_longa": [
            "Desenvolva um texto argumentativo",
            "Escreva uma redação sobre",
            "Elabore um texto dissertativo",
            "Discorra sobre o tema",
            "Produza um texto explicando",
            "Argumente a favor ou contra"
        ],
        "verdadeiro_falso": [
            "Marque V para verdadeiro e F para falso",
            "Indique se as afirmações são verdadeiras ou falsas",
            "Julgue os itens como certo ou errado",
            "Classifique as sentenças em V ou F",
            "Determine a veracidade das afirmações",
            "Avalie se cada item é verdadeiro ou falso"
        ],
        "associacao": [
            "Relacione a coluna A com a coluna B",
            "Ligue os itens correspondentes",
            "Associe os conceitos às definições",
            "Conecte os termos às suas descrições",
            "Combine os elementos das duas listas",
            "Estabeleça correspondência entre"
        ]
    }
    
    # Positivos
    for tipo, frases in question_phrases.items():
        for frase in frases:
            examples.append({
                "text1": frase,
                "text2": tipo,
                "score": 0.95,
                "type": "question_type_positive",
                "disciplina": "question_type"
            })
    
    # Negativos
    tipos = list(question_phrases.keys())
    for tipo_correto, frases in question_phrases.items():
        for frase in frases[:2]:
            tipo_errado = random.choice([t for t in tipos if t != tipo_correto])
            examples.append({
                "text1": frase,
                "text2": tipo_errado,
                "score": 0.1,
                "type": "question_type_negative",
                "disciplina": "question_type"
            })
    
    print(f"   ✅ {len(examples)} exemplos de tipos de questão gerados")
    return examples

def generate_text_base_examples():
    """Gera exemplos para tipos de texto base"""
    examples = []
    
    print("📝 Gerando exemplos de TIPOS DE TEXTO BASE...")
    
    text_base_phrases = {
        "documento_historico": [
            "Leia o documento histórico abaixo",
            "Analise a fonte primária",
            "Observe o trecho do documento oficial",
            "Com base no registro histórico",
            "Segundo o documento da época",
            "De acordo com a fonte histórica"
        ],
        "texto_literario": [
            "Leia o fragmento literário",
            "Analise o trecho do romance",
            "Observe o poema abaixo",
            "Com base no texto literário",
            "Segundo o conto apresentado",
            "De acordo com a obra literária"
        ],
        "artigo_jornal": [
            "Leia a notícia abaixo",
            "Analise a reportagem",
            "Observe o artigo jornalístico",
            "Com base na matéria do jornal",
            "Segundo a notícia publicada",
            "De acordo com o texto jornalístico"
        ],
        "charge": [
            "Observe a charge abaixo",
            "Analise o cartum apresentado",
            "Com base na tirinha",
            "Segundo a charge política",
            "De acordo com a história em quadrinhos",
            "Interprete a caricatura"
        ],
        "grafico_barras": [
            "Observe o gráfico de barras",
            "Analise o gráfico em colunas",
            "Com base no gráfico vertical",
            "Segundo os dados do gráfico de barras",
            "De acordo com o gráfico apresentado",
            "Interprete o gráfico de colunas"
        ],
        "grafico_linhas": [
            "Observe o gráfico de linhas",
            "Analise a evolução temporal",
            "Com base no gráfico linear",
            "Segundo a série temporal",
            "De acordo com o gráfico de evolução",
            "Interprete o gráfico de linhas"
        ],
        "tabela": [
            "Observe a tabela abaixo",
            "Analise os dados tabulados",
            "Com base na planilha",
            "Segundo a tabela apresentada",
            "De acordo com o quadro de dados",
            "Interprete a matriz de dados"
        ],
        "imagem": [
            "Observe a imagem abaixo",
            "Analise a fotografia",
            "Com base na figura",
            "Segundo a ilustração",
            "De acordo com a foto apresentada",
            "Interprete a representação visual"
        ],
        "mapa": [
            "Observe o mapa abaixo",
            "Analise a carta geográfica",
            "Com base no mapa geográfico",
            "Segundo o planisfério",
            "De acordo com a representação cartográfica",
            "Interprete o mapa apresentado"
        ],
        "infografico": [
            "Observe o infográfico abaixo",
            "Analise a visualização de dados",
            "Com base no gráfico informativo",
            "Segundo o infográfico apresentado",
            "De acordo com a infografia",
            "Interprete o infográfico"
        ],
        "poema": [
            "Leia o poema abaixo",
            "Analise a letra da música",
            "Observe os versos apresentados",
            "Com base no poema",
            "Segundo a estrofe",
            "De acordo com o soneto"
        ]
    }
    
    # Positivos
    for tipo, frases in text_base_phrases.items():
        for frase in frases:
            examples.append({
                "text1": frase,
                "text2": tipo,
                "score": 0.95,
                "type": "text_base_positive",
                "disciplina": "text_base"
            })
    
    # Negativos
    tipos = list(text_base_phrases.keys())
    for tipo_correto, frases in text_base_phrases.items():
        for frase in frases[:2]:
            tipo_errado = random.choice([t for t in tipos if t != tipo_correto])
            examples.append({
                "text1": frase,
                "text2": tipo_errado,
                "score": 0.1,
                "type": "text_base_negative",
                "disciplina": "text_base"
            })
    
    print(f"   ✅ {len(examples)} exemplos de tipos de texto base gerados")
    return examples

def generate_perfil_aluno_examples():
    """Gera exemplos para perfis de aluno"""
    examples = []
    
    print("📝 Gerando exemplos de PERFIS DE ALUNO...")
    
    perfil_phrases = {
        "bom_dominio": [
            "Alunos com bom domínio de leitura",
            "Estudantes que leem bem",
            "Turma com boa compreensão textual",
            "Alunos com leitura fluente",
            "Estudantes avançados em leitura",
            "Turma com boa interpretação"
        ],
        "dificuldade_conexao": [
            "Alunos com dificuldade em conectar conceitos",
            "Estudantes que têm dificuldade para relacionar ideias",
            "Turma com dificuldade de interpretação",
            "Alunos que sabem o básico mas não conectam",
            "Estudantes com dificuldade em relacionar",
            "Turma com dificuldade de conexão"
        ],
        "conhecimento_basico": [
            "Alunos com conhecimento básico",
            "Estudantes iniciantes no assunto",
            "Turma em nível fundamental",
            "Alunos no nível básico",
            "Estudantes com conhecimento elementar",
            "Turma iniciante"
        ],
        "conhecimento_avancado": [
            "Alunos com conhecimento avançado",
            "Estudantes em nível profundo",
            "Turma avançada no assunto",
            "Alunos com alto nível",
            "Estudantes especializados",
            "Turma com conhecimento aprofundado"
        ]
    }
    
    for perfil, frases in perfil_phrases.items():
        for frase in frases:
            examples.append({
                "text1": frase,
                "text2": perfil,
                "score": 0.95,
                "type": "perfil_positive",
                "disciplina": "perfil_aluno"
            })
    
    perfis = list(perfil_phrases.keys())
    for perfil_correto, frases in perfil_phrases.items():
        for frase in frases[:2]:
            perfil_errado = random.choice([p for p in perfis if p != perfil_correto])
            examples.append({
                "text1": frase,
                "text2": perfil_errado,
                "score": 0.1,
                "type": "perfil_negative",
                "disciplina": "perfil_aluno"
            })
    
    print(f"   ✅ {len(examples)} exemplos de perfis de aluno gerados")
    return examples

def balance_and_sample(positive, negative, max_total=200):
    """Balanceia e amostra exemplos"""
    print(f"\n⚖️  Balanceando dataset...")
    print(f"   Positivos: {len(positive)}")
    print(f"   Negativos: {len(negative)}")
    
    # Queremos 50% positivos, 50% negativos
    target_per_type = max_total // 2
    
    # Amostrar positivos (priorizar synonyms e combined_terms)
    positive_priority = sorted(positive, key=lambda x: {
        'synonym': 3,
        'combined_terms': 2,
        'main_term': 1
    }.get(x['type'], 0), reverse=True)
    
    sampled_positive = positive_priority[:target_per_type]
    
    # Amostrar negativos (distribuir entre tipos)
    negative_by_type = {}
    for ex in negative:
        t = ex['type']
        if t not in negative_by_type:
            negative_by_type[t] = []
        negative_by_type[t].append(ex)
    
    sampled_negative = []
    per_type = target_per_type // len(negative_by_type)
    
    for examples in negative_by_type.values():
        sampled_negative.extend(random.sample(examples, min(per_type, len(examples))))
    
    # Completar se necessário
    if len(sampled_negative) < target_per_type:
        remaining = [ex for ex in negative if ex not in sampled_negative]
        sampled_negative.extend(random.sample(remaining, min(target_per_type - len(sampled_negative), len(remaining))))
    
    print(f"   ✅ Selecionados: {len(sampled_positive)} positivos, {len(sampled_negative)} negativos")
    
    return sampled_positive + sampled_negative

def balance_and_sample(positive, negative, max_total=200):
    """Balanceia e amostra exemplos"""
    print(f"\n⚖️  Balanceando dataset...")
    print(f"   Positivos: {len(positive)}")
    print(f"   Negativos: {len(negative)}")
    
    # Queremos 50% positivos, 50% negativos
    target_per_type = max_total // 2
    
    # Amostrar positivos (priorizar synonyms e combined_terms)
    positive_priority = sorted(positive, key=lambda x: {
        'synonym': 3,
        'combined_terms': 2,
        'main_term': 1
    }.get(x['type'], 0), reverse=True)
    
    sampled_positive = positive_priority[:target_per_type]
    
    # Amostrar negativos (distribuir entre tipos)
    negative_by_type = {}
    for ex in negative:
        t = ex['type']
        if t not in negative_by_type:
            negative_by_type[t] = []
        negative_by_type[t].append(ex)
    
    sampled_negative = []
    per_type = target_per_type // len(negative_by_type)
    
    for examples in negative_by_type.values():
        sampled_negative.extend(random.sample(examples, min(per_type, len(examples))))
    
    # Completar se necessário
    if len(sampled_negative) < target_per_type:
        remaining = [ex for ex in negative if ex not in sampled_negative]
        sampled_negative.extend(random.sample(remaining, min(target_per_type - len(sampled_negative), len(remaining))))
    
    print(f"   ✅ Selecionados: {len(sampled_positive)} positivos, {len(sampled_negative)} negativos")
    
    return sampled_positive + sampled_negative

def save_training_data(examples, output_path):
    """Salva exemplos em JSON"""
    # Converter para formato final (remover campos auxiliares)
    final_examples = []
    for ex in examples:
        final_examples.append({
            "text1": ex["text1"],
            "text2": ex["text2"],
            "score": ex["score"]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_examples, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Salvo em: {output_path}")

def generate_statistics(examples):
    """Gera estatísticas do dataset"""
    print("\n" + "="*80)
    print("📊 ESTATÍSTICAS DO DATASET")
    print("="*80)
    
    total = len(examples)
    by_score = {
        "Alta (0.8-1.0)": 0,
        "Média (0.5-0.8)": 0,
        "Baixa (0.0-0.5)": 0
    }
    
    by_type = {}
    by_disciplina = {}
    
    for ex in examples:
        score = ex['score']
        if score >= 0.8:
            by_score["Alta (0.8-1.0)"] += 1
        elif score >= 0.5:
            by_score["Média (0.5-0.8)"] += 1
        else:
            by_score["Baixa (0.0-0.5)"] += 1
        
        t = ex.get('type', 'unknown')
        by_type[t] = by_type.get(t, 0) + 1
        
        d = ex.get('disciplina', 'unknown')
        by_disciplina[d] = by_disciplina.get(d, 0) + 1
    
    print(f"\nTotal de exemplos: {total}")
    
    print(f"\nPor similaridade:")
    for category, count in by_score.items():
        pct = (count / total) * 100
        print(f"  {category}: {count} ({pct:.1f}%)")
    
    print(f"\nPor tipo:")
    for t, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / total) * 100
        print(f"  {t}: {count} ({pct:.1f}%)")
    
    print(f"\nPor disciplina (top 5):")
    for d, count in sorted(by_disciplina.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = (count / total) * 100
        print(f"  {d}: {count} ({pct:.1f}%)")

def main():
    print("🚀 GERADOR AUTOMÁTICO DE DADOS DE TREINO")
    print("="*80)
    
    # Carregar BNCC
    print("\n📚 Carregando BNCC...")
    bncc_data = load_bncc_data()
    print(f"   ✅ {len(bncc_data)} disciplinas carregadas")
    
    # Gerar exemplos BNCC
    positive = generate_positive_examples(bncc_data)
    negative = generate_negative_examples(bncc_data)
    
    # ⚖️ BALANCEAR BNCC para evitar overfitting
    # Usar apenas uma amostra representativa, não todos os dados
    print(f"\n⚖️  Balanceando exemplos BNCC para evitar overfitting...")
    print(f"   Total gerado: {len(positive)} positivos, {len(negative)} negativos")
    
    # Amostrar 300 positivos (priorizar synonyms e combined_terms)
    positive_priority = sorted(positive, key=lambda x: {
        'synonym': 3,
        'combined_terms': 2,
        'main_term': 1
    }.get(x['type'], 0), reverse=True)
    sampled_positive = positive_priority[:300]
    
    # Amostrar 150 negativos (distribuir entre tipos)
    negative_by_type = {}
    for ex in negative:
        t = ex['type']
        if t not in negative_by_type:
            negative_by_type[t] = []
        negative_by_type[t].append(ex)
    
    sampled_negative = []
    per_type = 150 // len(negative_by_type)
    for examples in negative_by_type.values():
        sampled_negative.extend(random.sample(examples, min(per_type, len(examples))))
    
    # Completar se necessário
    if len(sampled_negative) < 150:
        remaining = [ex for ex in negative if ex not in sampled_negative]
        sampled_negative.extend(random.sample(remaining, min(150 - len(sampled_negative), len(remaining))))
    
    print(f"   ✅ Selecionados: {len(sampled_positive)} positivos, {len(sampled_negative)} negativos")
    
    # Gerar exemplos Bloom, Tipos de Questão, Tipos de Texto Base, Perfil Aluno
    bloom_examples = generate_bloom_examples()
    question_examples = generate_question_type_examples()
    text_base_examples = generate_text_base_examples()
    perfil_examples = generate_perfil_aluno_examples()
    
    # Combinar todos os exemplos
    all_examples = sampled_positive + sampled_negative + bloom_examples + question_examples + text_base_examples + perfil_examples
    
    # Embaralhar
    random.shuffle(all_examples)
    
    # Estatísticas
    generate_statistics(all_examples)
    
    # Salvar
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_pairs_auto.json')
    save_training_data(all_examples, output_path)
    
    print("\n" + "="*80)
    print("✅ CONCLUÍDO!")
    print("="*80)
    print(f"\n💡 Próximos passos:")
    print(f"   1. Revise o arquivo: data/training_pairs_auto.json")
    print(f"   2. Execute fine-tuning: python scripts/finetune_embeddings.py")
    print(f"   3. Teste resultados: python test_finetuning_results.py")
    print(f"\n📊 Dataset balanceado ({len(all_examples)} exemplos):")
    print(f"   ✅ 300 exemplos BNCC positivos (amostra representativa)")
    print(f"   ✅ 150 exemplos BNCC negativos (evita falsos positivos)")
    print(f"   ✅ {len(bloom_examples)} exemplos Bloom (generalização)")
    print(f"   ✅ {len(question_examples)} exemplos Questão (generalização)")
    print(f"   ✅ {len(text_base_examples)} exemplos Texto Base (generalização)")
    print(f"   ✅ {len(perfil_examples)} exemplos Perfil Aluno (generalização)")
    print(f"\n💡 Balanceamento evita overfitting e melhora generalização!")

if __name__ == "__main__":
    main()
