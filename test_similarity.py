import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Carregar modelo
model_path = 'models/bncc-embeddings-finetuned'
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    model = SentenceTransformer(model_path)

# Carregar embeddings
data = json.load(open('data/bncc_embeddings.json', 'r', encoding='utf-8'))

# Texto de teste
text = "Era vargas"
text_emb = model.encode(text, convert_to_numpy=True).reshape(1, -1)

# Buscar top 10 mais similares
similarities = []
for key, info in data.items():
    emb = np.array(info['embedding']).reshape(1, -1)
    sim = cosine_similarity(text_emb, emb)[0][0]
    similarities.append({
        'key': key,
        'similarity': sim,
        'texto': info.get('texto', ''),
        'disciplina': info.get('disciplina', ''),
        'ano': info.get('ano', ''),
        'tipo': info.get('tipo', '')
    })

similarities.sort(key=lambda x: x['similarity'], reverse=True)

print(f'Texto: "{text}"')
print(f'\nTop 10 mais similares:\n')
for i, match in enumerate(similarities[:10], 1):
    print(f'{i}. Similaridade: {match["similarity"]:.4f}')
    print(f'   Disciplina: {match["disciplina"]} - Ano: {match["ano"]}')
    print(f'   Tipo: {match["tipo"]}')
    print(f'   Texto: {match["texto"][:80]}...')
    print(f'   Key: {match["key"][:100]}...')
    print()
