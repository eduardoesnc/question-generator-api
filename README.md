# 🤖 NLP API - Extração de Informações Educacionais

API FastAPI que extrai informações educacionais de texto livre usando embeddings semânticos fine-tuned para BNCC.

## 🚀 Setup Rápido

```bash
# Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt

# Baixar modelo spaCy (usado para keywords e análise sintática)
python -m spacy download pt_core_news_lg

# Configurar variáveis de ambiente
cp .env.example .env

# Executar API
python -m uvicorn app.main:app --reload --port 8000
```

API disponível em: `http://localhost:8000/docs`

## 🎯 Métodos de Extração

A API oferece 3 métodos (parâmetro `method`):

- **keywords**: Matching por palavras-chave (~100ms)
- **embeddings**: Similaridade semântica com modelo fine-tuned (~200ms)
- **hybrid**: Combinação de ambos - recomendado (~250ms)

## 📝 Exemplo de Uso

```json
POST /api/extract
{
  "text": "História sobre Era Vargas, 9º ano, análise, dissertativa com documento histórico",
  "method": "hybrid"
}
```

Retorna: disciplina, ano, unidadeTematica, objetoConhecimento, habilidade BNCC, nivelBloom, tipoQuestao, tipoTextoBase, perfilAluno.

## 🔧 Scripts Disponíveis

```bash
# Gerar embeddings dos dados da BNCC
python scripts/generate_embeddings.py

# Gerar embeddings de campos não-BNCC (Bloom, perfil aluno, tipos de questão/texto)
python scripts/generate_non_bncc_embeddings.py

# Criar pares de treinamento automaticamente
python scripts/generate_training_data.py
# Esses pares serão modificados, está dentro dos meus planos trocar a forma de treinamento

# Fine-tuning do modelo com os dados gerados
python scripts/finetune_embeddings.py

# Adicionar embeddings de palavras-chave extraídas
python scripts/add_keyword_embeddings.py
```

## 📦 Modelo

O modelo fine-tuned está em `models/bncc-embeddings-finetuned/` e é carregado automaticamente pela API. Baseado em `sentence-transformers` com dados da BNCC.

## 🛠️ Stack

FastAPI • spaCy • Sentence Transformers • PyTorch • scikit-learn
