"""
Gera embeddings para campos não-BNCC (Bloom, Perfil Aluno, Tipos de Questão, Tipos de Texto Base)
"""
import json
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_finetuned_model():
    """Carrega modelo fine-tuned"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bncc-embeddings-finetuned')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo fine-tuned não encontrado em: {model_path}")
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
        model = SentenceTransformer(model_path)
    
    return model

def generate_bloom_embeddings(model):
    """Gera embeddings para níveis de Bloom"""
    print("\n📝 Gerando embeddings para NÍVEIS BLOOM...")
    
    bloom_examples = {
        "conhecimento": [
            "Memorizar datas históricas importantes",
            "Listar os estados brasileiros",
            "Definir o conceito de fotossíntese",
            "Identificar as partes do corpo humano",
            "Recordar eventos da Segunda Guerra",
            "Nomear os planetas do sistema solar",
            "Lembrar fórmulas matemáticas básicas",
            "Reconhecer símbolos químicos"
        ],
        "compreensao": [
            "Explicar o processo de fotossíntese",
            "Interpretar um texto literário",
            "Resumir os principais eventos da Era Vargas",
            "Descrever o ciclo da água",
            "Compreender as causas da Revolução Industrial",
            "Parafrasear um poema",
            "Entender conceitos matemáticos",
            "Classificar tipos de animais"
        ],
        "aplicacao": [
            "Resolver equações do segundo grau",
            "Aplicar regras gramaticais em frases",
            "Calcular a área de um triângulo",
            "Usar fórmulas matemáticas em problemas",
            "Demonstrar experimentos científicos",
            "Executar operações com frações",
            "Implementar algoritmos",
            "Praticar conversação em inglês"
        ],
        "analise": [
            "Analisar as causas da guerra",
            "Comparar diferentes sistemas políticos",
            "Diferenciar células vegetais e animais",
            "Examinar dados de um gráfico",
            "Investigar relações entre variáveis",
            "Relacionar eventos históricos",
            "Desconstruir argumentos",
            "Organizar informações complexas"
        ],
        "sintese": [
            "Criar um projeto de ciências",
            "Desenvolver uma redação argumentativa",
            "Elaborar um plano de ação",
            "Construir uma maquete",
            "Produzir um vídeo educativo",
            "Planejar uma apresentação",
            "Inventar uma solução criativa",
            "Sintetizar múltiplas fontes"
        ],
        "avaliacao": [
            "Avaliar a qualidade de um argumento",
            "Julgar a validade de uma teoria",
            "Criticar uma obra literária",
            "Justificar uma decisão política",
            "Argumentar sobre questões ambientais",
            "Defender um ponto de vista",
            "Recomendar melhorias",
            "Decidir entre alternativas"
        ]
    }
    
    bloom_embeddings = {}
    for nivel, exemplos in bloom_examples.items():
        texto_combinado = " ".join(exemplos)
        embedding = model.encode(texto_combinado, convert_to_numpy=True)
        bloom_embeddings[nivel] = {
            "embedding": embedding.tolist(),
            "exemplos": exemplos
        }
        print(f"  ✓ {nivel}: {len(exemplos)} exemplos")
    
    return bloom_embeddings

def generate_perfil_embeddings(model):
    """Gera embeddings para perfis de aluno"""
    print("\n📝 Gerando embeddings para PERFIS DE ALUNO...")
    
    perfil_examples = {
        "bom_dominio": [
            "Alunos com bom domínio de leitura",
            "Estudantes que leem bem",
            "Turma com boa compreensão textual",
            "Alunos com leitura fluente",
            "Estudantes avançados em leitura",
            "Turma com boa interpretação de textos"
        ],
        "dificuldade_conexao": [
            "Alunos com dificuldade em conectar conceitos",
            "Estudantes que têm dificuldade para relacionar ideias",
            "Turma com dificuldade de interpretação",
            "Alunos que sabem o básico mas não conectam",
            "Estudantes com dificuldade em relacionar",
            "Turma com dificuldade de conexão entre temas"
        ],
        "conhecimento_basico": [
            "Alunos com conhecimento básico",
            "Estudantes iniciantes no assunto",
            "Turma em nível fundamental",
            "Alunos no nível básico",
            "Estudantes com conhecimento elementar",
            "Turma iniciante no tema"
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
    
    perfil_embeddings = {}
    for perfil, exemplos in perfil_examples.items():
        texto_combinado = " ".join(exemplos)
        embedding = model.encode(texto_combinado, convert_to_numpy=True)
        perfil_embeddings[perfil] = {
            "embedding": embedding.tolist(),
            "exemplos": exemplos
        }
        print(f"  ✓ {perfil}: {len(exemplos)} exemplos")
    
    return perfil_embeddings

def generate_tipo_questao_embeddings(model):
    """Gera embeddings para tipos de questão"""
    print("\n📝 Gerando embeddings para TIPOS DE QUESTÃO...")
    
    tipo_questao_examples = {
        "multipla_escolha": [
            "Marque a alternativa correta",
            "Assinale a opção que apresenta",
            "Escolha entre as alternativas a, b, c",
            "Selecione a resposta adequada",
            "Qual das opções abaixo está correta",
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
    
    tipo_questao_embeddings = {}
    for tipo, exemplos in tipo_questao_examples.items():
        texto_combinado = " ".join(exemplos)
        embedding = model.encode(texto_combinado, convert_to_numpy=True)
        tipo_questao_embeddings[tipo] = {
            "embedding": embedding.tolist(),
            "exemplos": exemplos
        }
        print(f"  ✓ {tipo}: {len(exemplos)} exemplos")
    
    return tipo_questao_embeddings

def generate_tipo_texto_base_embeddings(model):
    """Gera embeddings para tipos de texto base"""
    print("\n📝 Gerando embeddings para TIPOS DE TEXTO BASE...")
    
    tipo_texto_examples = {
        "documento_historico": [
            "Leia o documento histórico abaixo",
            "Analise a fonte primária",
            "Observe o trecho do documento oficial",
            "Com base no registro histórico"
        ],
        "texto_literario": [
            "Leia o fragmento literário",
            "Analise o trecho do romance",
            "Observe o poema abaixo",
            "Com base no texto literário"
        ],
        "artigo_jornal": [
            "Leia a notícia abaixo",
            "Analise a reportagem",
            "Observe o artigo jornalístico",
            "Com base na matéria do jornal"
        ],
        "charge": [
            "Observe a charge abaixo",
            "Analise o cartum apresentado",
            "Com base na tirinha",
            "Interprete a caricatura"
        ],
        "grafico_barras": [
            "Observe o gráfico de barras",
            "Analise o gráfico em colunas",
            "Com base no gráfico vertical"
        ],
        "grafico_linhas": [
            "Observe o gráfico de linhas",
            "Analise a evolução temporal",
            "Com base no gráfico linear"
        ],
        "tabela": [
            "Observe a tabela abaixo",
            "Analise os dados tabulados",
            "Com base na planilha"
        ],
        "imagem": [
            "Observe a imagem abaixo",
            "Analise a fotografia",
            "Com base na figura"
        ],
        "mapa": [
            "Observe o mapa abaixo",
            "Analise a carta geográfica",
            "Com base no mapa geográfico"
        ],
        "infografico": [
            "Observe o infográfico abaixo",
            "Analise a visualização de dados",
            "Com base no gráfico informativo"
        ],
        "poema": [
            "Leia o poema abaixo",
            "Analise a letra da música",
            "Observe os versos apresentados"
        ]
    }
    
    tipo_texto_embeddings = {}
    for tipo, exemplos in tipo_texto_examples.items():
        texto_combinado = " ".join(exemplos)
        embedding = model.encode(texto_combinado, convert_to_numpy=True)
        tipo_texto_embeddings[tipo] = {
            "embedding": embedding.tolist(),
            "exemplos": exemplos
        }
        print(f"  ✓ {tipo}: {len(exemplos)} exemplos")
    
    return tipo_texto_embeddings

def main():
    print("🚀 GERADOR DE EMBEDDINGS PARA CAMPOS NÃO-BNCC")
    print("="*80)
    
    print("\n📦 Carregando modelo fine-tuned...")
    model = load_finetuned_model()
    print("✅ Modelo carregado!")
    
    bloom_embeddings = generate_bloom_embeddings(model)
    perfil_embeddings = generate_perfil_embeddings(model)
    tipo_questao_embeddings = generate_tipo_questao_embeddings(model)
    tipo_texto_embeddings = generate_tipo_texto_base_embeddings(model)
    
    output_data = {
        "bloom": bloom_embeddings,
        "perfil_aluno": perfil_embeddings,
        "tipo_questao": tipo_questao_embeddings,
        "tipo_texto_base": tipo_texto_embeddings
    }
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'non_bncc_embeddings.json')
    
    print(f"\n💾 Salvando embeddings...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Salvo em: {output_path}")
    
    print("\n" + "="*80)
    print("📊 RESUMO:")
    print(f"  • Bloom: {len(bloom_embeddings)} níveis")
    print(f"  • Perfil Aluno: {len(perfil_embeddings)} perfis")
    print(f"  • Tipo Questão: {len(tipo_questao_embeddings)} tipos")
    print(f"  • Tipo Texto Base: {len(tipo_texto_embeddings)} tipos")
    print(f"\n✅ CONCLUÍDO!")

if __name__ == "__main__":
    main()
