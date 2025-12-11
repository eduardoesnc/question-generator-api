# 🤖 NLP API - Processamento de Linguagem Natural

API FastAPI para processar texto livre e extrair informações educacionais usando NLP com spaCy.

> **Nota:** Este é o backend do projeto. Para instruções completas de instalação e execução do sistema completo, veja o [README principal](../README.md).

## 🚀 Instalação

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual (Windows)
venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Baixar modelo de português do spaCy
python -m spacy download pt_core_news_lg
```

## 🏃 Executar

```bash
python main.py
```

A API estará disponível em: `http://localhost:8000`

## 📚 Documentação da API

Acesse a documentação interativa em: `http://localhost:8000/docs`

## 🔍 Endpoints

### `GET /`
Informações básicas da API

### `GET /health`
Health check - verifica se o modelo NLP está carregado

### `POST /api/extract`
Extrai informações educacionais de texto livre

**Request:**
```json
{
  "text": "Quero uma questão de matemática para o 7º ano sobre frações",
  "context": {}  // opcional
}
```

**Response:**
```json
{
  "extracted": {
    "disciplina": "Matemática",
    "ano": "7º ano"
  },
  "confidence": {
    "disciplina": 0.95,
    "ano": 0.98
  },
  "suggestions": [],
  "missing_fields": ["nivelBloom", "tipoQuestao", "tipoTextoBase"],
  "original_text": "Quero uma questão de matemática para o 7º ano sobre frações"
}
```

## 🧪 Testar

```bash
python test_api.py
```

## 📊 Campos Extraídos

- **disciplina**: Matéria escolar (Matemática, Português, etc.)
- **ano**: Ano/série escolar (1º ano a 9º ano)
- **nivelBloom**: Nível cognitivo (conhecimento, compreensão, aplicação, análise, síntese, avaliação)
- **tipoQuestao**: Formato da questão (múltipla escolha, dissertativa, etc.)
- **tipoTextoBase**: Tipo de texto de apoio (charge, gráfico, tabela, etc.)
- **perfilAluno**: Perfil do estudante (conhecimento básico, avançado, etc.)

## 🛠️ Tecnologias

- **FastAPI** - Framework web moderno e rápido
- **spaCy** - Biblioteca de NLP para português
- **Pydantic** - Validação de dados
- **Uvicorn** - Servidor ASGI
