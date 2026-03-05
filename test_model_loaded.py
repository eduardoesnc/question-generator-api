"""
Testa qual modelo está sendo carregado pelo EmbeddingsMatcher
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from matchers.embeddings_matcher import EmbeddingsMatcher

print("🧪 TESTE: Qual modelo está sendo carregado?")
print("="*80)

# Criar instância
matcher = EmbeddingsMatcher()

# Verificar qual modelo foi carregado
model_name = matcher.model.model_card_data.model_name if hasattr(matcher.model, 'model_card_data') else "Unknown"

print(f"\n📦 Modelo carregado:")
print(f"   Nome: {model_name}")
print(f"   Tipo: {type(matcher.model)}")

# Verificar se é fine-tuned
model_path = os.path.join('models', 'bncc-embeddings-finetuned')
is_finetuned = os.path.exists(model_path)

print(f"\n🔍 Verificação:")
print(f"   Modelo fine-tuned existe? {is_finetuned}")
print(f"   Path: {os.path.abspath(model_path)}")

# Testar similaridade
print(f"\n🧪 Teste de similaridade:")
text1 = "Era Vargas"
text2 = "O período varguista e suas contradições"

emb1 = matcher.model.encode(text1)
emb2 = matcher.model.encode(text2)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]

print(f"   '{text1}' ↔ '{text2}'")
print(f"   Similaridade: {sim:.4f}")

if sim > 0.80:
    print(f"   ✅ ALTA similaridade - Provavelmente usando modelo FINE-TUNED!")
elif sim > 0.50:
    print(f"   🟡 Similaridade média - Pode ser fine-tuned ou base")
else:
    print(f"   ❌ Baixa similaridade - Provavelmente usando modelo BASE")

print("\n" + "="*80)
print("💡 Conclusão:")
if sim > 0.80:
    print("   O modelo FINE-TUNED está sendo usado corretamente! 🎉")
else:
    print("   O modelo BASE está sendo usado. Execute:")
    print("   1. python scripts/finetune_embeddings.py")
    print("   2. python scripts/generate_embeddings.py")
