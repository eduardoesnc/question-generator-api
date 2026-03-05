"""
Testa se o fine-tuning melhorou os resultados
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

print("🧪 TESTE DE RESULTADOS DO FINE-TUNING")
print("="*80)

# Verificar se modelo fine-tuned existe
model_path = os.path.join('models', 'bncc-embeddings-finetuned')
if not os.path.exists(model_path):
    print(f"❌ Modelo fine-tuned não encontrado em: {model_path}")
    print("   Execute: python scripts/finetune_embeddings.py")
    exit(1)

print(f"✅ Modelo fine-tuned encontrado: {model_path}\n")

# Carregar ambos os modelos
print("📦 Carregando modelo base...")
model_base = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("📦 Carregando modelo fine-tuned...")
model_ft = SentenceTransformer(model_path)

print("\n" + "="*80)
print("🔬 COMPARANDO SIMILARIDADES")
print("="*80 + "\n")

# Casos de teste
test_cases = [
    ("Era Vargas", "O período varguista e suas contradições"),
    ("Era Vargas", "A história recente"),  # Falso positivo
    ("Células vegetais", "Célula como unidade da vida"),
    ("Ditadura militar", "A ditadura civil-militar e os processos de resistência"),
    ("Trabalhismo", "O trabalhismo e seu protagonismo político"),
]

improvements = []

for text1, text2 in test_cases:
    # Base
    emb1_base = model_base.encode(text1).reshape(1, -1)
    emb2_base = model_base.encode(text2).reshape(1, -1)
    sim_base = cosine_similarity(emb1_base, emb2_base)[0][0]
    
    # Fine-tuned
    emb1_ft = model_ft.encode(text1).reshape(1, -1)
    emb2_ft = model_ft.encode(text2).reshape(1, -1)
    sim_ft = cosine_similarity(emb1_ft, emb2_ft)[0][0]
    
    improvement = sim_ft - sim_base
    improvements.append(improvement)
    
    emoji = "✅" if improvement > 0.05 else "⚠️" if improvement > 0 else "❌"
    
    print(f"{emoji} '{text1}' ↔ '{text2[:50]}...'")
    print(f"   Base:       {sim_base:.4f}")
    print(f"   Fine-tuned: {sim_ft:.4f}")
    print(f"   Melhoria:   {improvement:+.4f}")
    
    if improvement > 0.05:
        print(f"   💡 Melhoria significativa!")
    elif improvement < -0.05:
        print(f"   ⚠️  Piorou!")
    
    print()

# Resumo
print("="*80)
print("📊 RESUMO")
print("="*80)

avg_improvement = sum(improvements) / len(improvements)
positive_improvements = sum(1 for i in improvements if i > 0)

print(f"\nMelhoria média: {avg_improvement:+.4f}")
print(f"Casos que melhoraram: {positive_improvements}/{len(improvements)}")

if avg_improvement > 0.05:
    print("\n✅ SUCESSO! O fine-tuning melhorou significativamente!")
elif avg_improvement > 0:
    print("\n🟡 MELHORIA MODESTA. Considere adicionar mais exemplos de treino.")
else:
    print("\n❌ SEM MELHORIA. Possíveis problemas:")
    print("   - Poucos exemplos de treino (adicione mais em training_pairs.json)")
    print("   - Exemplos não representativos")
    print("   - Overfitting (modelo decorou ao invés de aprender)")

print("\n💡 Próximos passos:")
if avg_improvement <= 0.05:
    print("   1. Adicione mais exemplos em data/training_pairs.json")
    print("   2. Execute novamente: python scripts/finetune_embeddings.py")
    print("   3. Regenere embeddings: python scripts/generate_embeddings.py")
else:
    print("   1. Verifique se a API está usando o modelo fine-tuned")
    print("   2. Teste com casos reais na API")
    print("   3. Se ainda não funcionar, verifique os logs da API")
