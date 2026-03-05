"""
Script para fazer fine-tuning do modelo de embeddings com dados da BNCC
"""
import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

def load_training_data():
    """Carrega pares de treino do JSON"""
    script_dir = os.path.dirname(__file__)
    
    # Tentar carregar training_pairs_auto.json primeiro (gerado automaticamente)
    auto_path = os.path.join(script_dir, '..', 'data', 'training_pairs_auto.json')
    manual_path = os.path.join(script_dir, '..', 'data', 'training_pairs.json')
    
    if os.path.exists(auto_path):
        training_path = auto_path
        print(f"📂 Usando dados AUTOMÁTICOS: {training_path}")
    elif os.path.exists(manual_path):
        training_path = manual_path
        print(f"📂 Usando dados MANUAIS: {training_path}")
    else:
        print("❌ Nenhum arquivo de treino encontrado!")
        print("   Execute: python scripts/generate_training_data.py")
        exit(1)
    
    with open(training_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ {len(data)} pares de treino carregados!")
    return data

def create_train_examples(data):
    """Converte dados JSON para InputExample"""
    examples = []
    
    for pair in data:
        example = InputExample(
            texts=[pair['text1'], pair['text2']],
            label=float(pair['score'])
        )
        examples.append(example)
    
    return examples

def finetune_model():
    """Executa fine-tuning do modelo"""
    print("🚀 INICIANDO FINE-TUNING DO MODELO DE EMBEDDINGS")
    print("="*80)
    
    # 1. Carregar modelo base
    print("\n📦 Carregando modelo base...")
    model_name = 'paraphrase-multilingual-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    print(f"✅ Modelo {model_name} carregado!")
    
    # 2. Carregar dados de treino
    print("\n📚 Preparando dados de treino...")
    training_data = load_training_data()
    train_examples = create_train_examples(training_data)
    
    # Split: 80% treino, 20% validação
    split_idx = int(len(train_examples) * 0.8)
    train_set = train_examples[:split_idx]
    val_set = train_examples[split_idx:]
    
    print(f"   Treino: {len(train_set)} exemplos")
    print(f"   Validação: {len(val_set)} exemplos")
    
    # 3. Criar DataLoader
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=8)
    
    # 4. Definir loss function
    # CosineSimilarityLoss: otimiza para que textos similares tenham embeddings próximos
    train_loss = losses.CosineSimilarityLoss(model)
    
    # 5. Criar evaluator para validação
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        val_set,
        name='bncc-validation'
    )
    
    # 6. Configurar fine-tuning
    print("\n🔧 Configurando fine-tuning...")
    num_epochs = 10
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% warmup
    
    print(f"   Épocas: {num_epochs}")
    print(f"   Batch size: 8")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Loss function: CosineSimilarityLoss")
    
    # 7. Executar fine-tuning
    print("\n🏋️  Iniciando treinamento...")
    print("   (Isso pode levar alguns minutos...)")
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'bncc-embeddings-finetuned')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        evaluation_steps=50,
        save_best_model=True,
        show_progress_bar=True
    )
    
    print("\n✅ Fine-tuning concluído!")
    print(f"📁 Modelo salvo em: {output_path}")
    
    # 8. Testar modelo fine-tuned
    print("\n🧪 TESTANDO MODELO FINE-TUNED")
    print("="*80)
    
    test_pairs = [
        ("Era Vargas", "O período varguista e suas contradições"),
        ("Células vegetais", "Célula como unidade da vida"),
        ("Ditadura militar", "A ditadura civil-militar e os processos de resistência"),
    ]
    
    print("\nComparando modelo ORIGINAL vs FINE-TUNED:\n")
    
    # Modelo original
    model_original = SentenceTransformer(model_name)
    
    for text1, text2 in test_pairs:
        # Original
        emb1_orig = model_original.encode(text1)
        emb2_orig = model_original.encode(text2)
        sim_orig = model_original.similarity(emb1_orig, emb2_orig).item()
        
        # Fine-tuned
        emb1_ft = model.encode(text1)
        emb2_ft = model.encode(text2)
        sim_ft = model.similarity(emb1_ft, emb2_ft).item()
        
        improvement = sim_ft - sim_orig
        emoji = "✅" if improvement > 0 else "❌"
        
        print(f"{emoji} '{text1}' ↔ '{text2[:40]}...'")
        print(f"   Original: {sim_orig:.4f}")
        print(f"   Fine-tuned: {sim_ft:.4f}")
        print(f"   Melhoria: {improvement:+.4f}\n")
    
    print("="*80)
    print("🎉 PROCESSO COMPLETO!")
    print(f"\n💡 Para usar o modelo fine-tuned, atualize embeddings_matcher.py:")
    print(f"   model = SentenceTransformer('{output_path}')")

if __name__ == "__main__":
    finetune_model()
