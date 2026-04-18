"""
Adiciona embeddings de palavras-chave importantes extraídas dos objetos da BNCC
Isso melhora a busca para textos curtos como "Era vargas"
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

def extract_keywords(text):
    """Extrai palavras-chave importantes de um texto"""
    stop_words = {
        'o', 'a', 'os', 'as', 'de', 'da', 'do', 'das', 'dos',
        'em', 'na', 'no', 'nas', 'nos', 'e', 'ou', 'para', 'com',
        'sua', 'suas', 'seu', 'seus', 'entre', 'sobre'
    }
    
    words = text.lower().split()
    keywords = []
    
    for i, word in enumerate(words):
        if len(word) > 4 and word not in stop_words:
            keywords.append(word)
            
            if i + 1 < len(words) and len(words[i+1]) > 4:
                bigram = f"{word} {words[i+1]}"
                keywords.append(bigram)
    
    return keywords[:5]

def main():
    print("🚀 ADICIONANDO EMBEDDINGS DE PALAVRAS-CHAVE")
    print("="*80)
    
    print("\n📦 Carregando modelo...")
    model = load_finetuned_model()
    
    print("📂 Carregando embeddings existentes...")
    embeddings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'bncc_embeddings.json')
    
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)
    
    print(f"✅ {len(embeddings_data)} embeddings carregados")
    
    print("\n🔍 Extraindo palavras-chave dos objetos...")
    new_embeddings = {}
    count = 0
    
    for key, data in embeddings_data.items():
        if data['tipo'] == 'objeto':
            texto = data['texto']
            keywords = extract_keywords(texto)
            
            for keyword in keywords:
                if len(keyword.split()) <= 2:
                    new_key = f"{key}|keyword|{keyword}"
                    
                    if new_key not in embeddings_data and new_key not in new_embeddings:
                        embedding = model.encode(keyword, convert_to_numpy=True)
                        
                        new_embeddings[new_key] = {
                            'embedding': embedding.tolist(),
                            'texto': keyword,
                            'tipo': 'keyword',
                            'disciplina': data['disciplina'],
                            'ano': data['ano'],
                            'unidade': data['unidade'],
                            'objeto': data['objeto'],
                            'habilidades': data['habilidades']
                        }
                        count += 1
    
    print(f"✅ {count} novos embeddings de palavras-chave gerados")
    
    print("\n💾 Salvando embeddings atualizados...")
    embeddings_data.update(new_embeddings)
    
    with open(embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Total de embeddings: {len(embeddings_data)}")
    print(f"   Originais: {len(embeddings_data) - count}")
    print(f"   Novos (keywords): {count}")
    
    print("\n" + "="*80)
    print("✅ CONCLUÍDO!")
    print("\n💡 Agora teste novamente com 'Era vargas'")

if __name__ == "__main__":
    main()
