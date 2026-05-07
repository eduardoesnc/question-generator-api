# NLP API

API FastAPI que extrai informações educacionais de texto livre usando **embeddings semânticos fine-tuned** para BNCC (Base Nacional Comum Curricular).

Dado um texto como *"Questão de matemática sobre frações para o 7º ano, nível de análise"*, a API retorna:
- Disciplina, ano, unidade temática, objeto de conhecimento
- Habilidade BNCC correspondente
- Nível da Taxonomia de Bloom
- Tipo de questão e perfil do aluno

## Stack

FastAPI • Sentence Transformers • PyTorch • scikit-learn

## Rodar Local

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

Acesse: `http://localhost:8000/docs`

## API

### Extração de Informações

**POST** `/api/extract`

```bash
curl -X POST http://localhost:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "História sobre Era Vargas, 9º ano, análise, dissertativa com documento histórico",
    "method": "hybrid"
  }'
```

**Resposta:**
```json
{
  "extracted": {
    "disciplina": "História",
    "ano": "9º ano",
    "nivelBloom": "Análise",
    "tipoQuestao": "Dissertativa",
    "tipoTextoBase": "Documento histórico"
  },
  "confidence": { "disciplina": 0.95, "ano": 0.98 },
  "missing_fields": ["unidadeTematica", "objetoConhecimento"]
}
```

**Métodos:**
- `keywords` - Matching por palavras-chave (~100ms)
- `embeddings` - Similaridade semântica (~200ms)
- `hybrid` - Combinação recomendada (~250ms)

## Scripts de Treinamento

```bash
# Gerar embeddings dos dados BNCC
python scripts/generate_embeddings.py

# Gerar embeddings de campos não-BNCC (Bloom, tipos de questão)
python scripts/generate_non_bncc_embeddings.py

# Fine-tuning do modelo
python scripts/finetune_embeddings.py

# Gerar dados de treinamento
python scripts/generate_training_data.py
```

## Modelo

O modelo fine-tuned está em `models/bncc-embeddings-finetuned/` e é carregado automaticamente pela API. Baseado em `sentence-transformers/all-MiniLM-L6-v2` com fine-tuning em dados da BNCC.

Tamanho: ~1GB | Linguagem: Português

## Variáveis de Ambiente

Copie `.env.example` para `.env`:

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `API_HOST` | Host da API | `0.0.0.0` |
| `API_PORT` | Porta da API | `8000` |
| `ENVIRONMENT` | Ambiente (development/production) | `development` |
| `CORS_ORIGINS` | Domínios permitidos (separados por vírgula) | `http://localhost:3000` |
| `KEYWORDS_THRESHOLD` | Threshold para matching por keywords | `0.20` |
| `EMBEDDINGS_THRESHOLD` | Threshold para similaridade semântica | `0.30` |
| `CONFIDENCE_MIN` | Confiança mínima | `0.50` |
| `CONFIDENCE_MAX` | Confiança máxima | `0.95` |
| `MAX_TEXT_LENGTH` | Tamanho máximo do texto de entrada | `1000` |
