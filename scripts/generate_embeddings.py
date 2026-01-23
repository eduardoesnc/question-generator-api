"""
Script para gerar embeddings pré-computados dos objetos BNCC
Executa offline, salva em JSON para uso rápido na aplicação
"""
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

def load_bncc_data():
    """Carrega dados da BNCC"""
    # Caminho correto: scripts/../data/bncc-data.json
    script_dir = os.path.dirname(__file__)
    bncc_path = os.path.join(script_dir, '..', 'data', 'bncc-data.json')
    bncc_path = os.path.abspath(bncc_path)
    
    print(f"📂 Procurando BNCC em: {bncc_path}")
    
    if not os.path.exists(bncc_path):
        raise FileNotFoundError(f"Arquivo BNCC não encontrado: {bncc_path}")
    
    with open(bncc_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_embeddings():
    """Gera embeddings para todos os objetos BNCC"""
    print("🚀 Iniciando geração de embeddings...")
    
    # Carregar modelo (multilíngue, otimizado para português)
    # MPNet é mais preciso que MiniLM (768 vs 384 dimensões)
    print("📦 Carregando modelo sentence-transformers...")
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    print("✅ Modelo carregado! (768 dimensões)")
    
    # Carregar BNCC
    print("📚 Carregando dados BNCC...")
    bncc_data = load_bncc_data()
    
    embeddings_data = {}
    total = 0
    duplicates = 0
    
    # Gerar embeddings para TUDO: unidades, objetos e habilidades
    print("🔄 Gerando embeddings para unidades, objetos e habilidades...")
    for disciplina, anos in bncc_data.items():
        for ano, unidades in anos.items():
            for unidade, objetos in unidades.items():
                
                # 1. EMBEDDING PARA UNIDADE TEMÁTICA
                unidade_key = f"{disciplina}|{ano}|unidade|{unidade}"
                if unidade_key not in embeddings_data:
                    embedding = model.encode(unidade, convert_to_numpy=True)
                    embeddings_data[unidade_key] = {
                        'embedding': embedding.tolist(),
                        'texto': unidade,
                        'tipo': 'unidade',
                        'disciplina': disciplina,
                        'ano': ano,
                        'unidade': unidade,
                        'objeto': None,
                        'habilidades': []
                    }
                    total += 1
                
                for objeto, habilidades in objetos.items():
                    
                    # 2. EMBEDDING PARA OBJETO DE CONHECIMENTO
                    objeto_key = f"{disciplina}|{ano}|objeto|{objeto}"
                    if objeto_key not in embeddings_data:
                        embedding = model.encode(objeto, convert_to_numpy=True)
                        embeddings_data[objeto_key] = {
                            'embedding': embedding.tolist(),
                            'texto': objeto,
                            'tipo': 'objeto',
                            'disciplina': disciplina,
                            'ano': ano,
                            'unidade': unidade,
                            'objeto': objeto,
                            'habilidades': habilidades
                        }
                        total += 1
                    else:
                        duplicates += 1
                    
                    # 3. EMBEDDINGS PARA CADA HABILIDADE
                    for habilidade in habilidades:
                        habilidade_key = f"{disciplina}|{ano}|habilidade|{habilidade}"
                        if habilidade_key not in embeddings_data:
                            embedding = model.encode(habilidade, convert_to_numpy=True)
                            embeddings_data[habilidade_key] = {
                                'embedding': embedding.tolist(),
                                'texto': habilidade,
                                'tipo': 'habilidade',
                                'disciplina': disciplina,
                                'ano': ano,
                                'unidade': unidade,
                                'objeto': objeto,
                                'habilidades': [habilidade]
                            }
                            total += 1
                        else:
                            duplicates += 1
                    
                    if total % 100 == 0:
                        print(f"   Processados: {total} embeddings...")
    
    print(f"✅ Total de embeddings gerados: {total}")
    print(f"⚠️  Objetos duplicados ignorados: {duplicates}")
    
    # Salvar em arquivo (na pasta data, não scripts/data)
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, '..', 'data', 'bncc_embeddings.json')
    output_path = os.path.abspath(output_path)
    
    print(f"💾 Salvando embeddings em: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
    
    # Calcular tamanho
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"📊 Tamanho do arquivo: {file_size:.2f} MB")
    print("🎉 Embeddings gerados com sucesso!")

if __name__ == "__main__":
    generate_embeddings()
