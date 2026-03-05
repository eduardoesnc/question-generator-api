# Fine-tuning do Modelo de Embeddings

Este guia explica como fazer fine-tuning do modelo de embeddings para melhorar a precisão no domínio educacional brasileiro (BNCC).

## O que é Fine-tuning?

Fine-tuning é o processo de retreinar um modelo pré-treinado com dados específicos do seu domínio. No nosso caso, ensinamos o modelo que:
- "Era Vargas" é similar a "O período varguista e suas contradições"
- "Células vegetais" é similar a "Célula como unidade da vida"
- E assim por diante...

## Pré-requisitos

```bash
pip install sentence-transformers torch
```

## Passo 1: Preparar dados de treino

Os dados de treino estão em `data/training_pairs.json`. Cada par contém:
- `text1`: Texto que o usuário pode digitar
- `text2`: Texto correspondente na BNCC
- `score`: Similaridade esperada (0.0 a 1.0)

**Exemplo:**
```json
{
  "text1": "Era Vargas",
  "text2": "O período varguista e suas contradições",
  "score": 1.0
}
```

### Adicionar mais exemplos

Edite `data/training_pairs.json` e adicione mais pares. Quanto mais exemplos, melhor o modelo aprende!

**Dicas:**
- Adicione variações de termos (sinônimos, abreviações)
- Inclua exemplos de todas as disciplinas
- Use scores entre 0.7 e 1.0 para pares relacionados
- Mínimo recomendado: 50-100 pares
- Ideal: 200-500 pares

## Passo 2: Executar fine-tuning

```bash
cd nlp-api
python scripts/finetune_embeddings.py
```

**O que acontece:**
1. Carrega o modelo base (`paraphrase-multilingual-mpnet-base-v2`)
2. Treina com seus dados por 10 épocas
3. Salva o modelo fine-tuned em `models/bncc-embeddings-finetuned/`
4. Mostra comparação antes/depois

**Tempo estimado:** 5-15 minutos (depende do hardware)

## Passo 3: Usar modelo fine-tuned

### Opção A: Atualização automática

```bash
python scripts/use_finetuned_model.py
```

Isso atualiza `embeddings_matcher.py` automaticamente.

### Opção B: Atualização manual

Edite `matchers/embeddings_matcher.py`, linha ~24:

```python
# ANTES:
self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# DEPOIS:
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'bncc-embeddings-finetuned')
if os.path.exists(model_path):
    self.model = SentenceTransformer(model_path)
else:
    self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
```

## Passo 4: Regenerar embeddings

Após o fine-tuning, regenere os embeddings da BNCC:

```bash
python scripts/generate_embeddings.py
```

Isso cria novos embeddings usando o modelo fine-tuned.

## Passo 5: Testar

Reinicie a API e teste:

```bash
python app/main.py
```

Teste com exemplos que antes não funcionavam bem:
- "Era Vargas"
- "Células vegetais"
- "Ditadura militar"

## Resultados esperados

**Antes do fine-tuning:**
- "Era Vargas" → "O período varguista..." = 0.38 similaridade ❌

**Depois do fine-tuning:**
- "Era Vargas" → "O período varguista..." = 0.70+ similaridade ✅

## Melhorias incrementais

Você pode fazer fine-tuning múltiplas vezes:

1. Adicione mais exemplos em `training_pairs.json`
2. Execute `finetune_embeddings.py` novamente
3. Regenere embeddings
4. Teste

O modelo vai melhorando gradualmente!

## Troubleshooting

### Erro: "CUDA out of memory"

Reduza o batch size em `finetune_embeddings.py`:
```python
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=4)  # Era 8
```

### Modelo não melhora

- Adicione mais exemplos de treino (mínimo 50)
- Verifique se os scores estão corretos (0.7-1.0 para similares)
- Aumente o número de épocas (10 → 20)

### Modelo piorou

- Pode ter overfitting (treinou demais)
- Reduza épocas (10 → 5)
- Adicione mais variedade nos exemplos

## Arquivos importantes

- `data/training_pairs.json` - Dados de treino
- `scripts/finetune_embeddings.py` - Script de fine-tuning
- `models/bncc-embeddings-finetuned/` - Modelo treinado
- `matchers/embeddings_matcher.py` - Usa o modelo

## Próximos passos

1. ✅ Fine-tuning básico (este guia)
2. 🔄 Coletar mais exemplos reais de uso
3. 🔄 Fine-tuning incremental
4. 🔄 Avaliar performance em dataset de teste
5. 🔄 Ajustar hiperparâmetros (épocas, batch size, learning rate)
