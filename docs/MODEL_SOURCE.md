# Configurando a origem do modelo

A variável `MODEL_SOURCE` no `.env` define de onde o modelo será carregado no startup.

---

## Local (volume externo)

```env
MODEL_SOURCE=local
MODELS_DIR=/app/models
```

No `docker-compose.yml`, monte o diretório com os modelos:

```yaml
volumes:
  - /caminho/no/servidor/models:/app/models:ro
```

---

## MinIO

```env
MODEL_SOURCE=minio
MINIO_ENDPOINT=minio.machlev.com.br
MINIO_ACCESS_KEY=sua-access-key
MINIO_SECRET_KEY=sua-secret-key
MINIO_BUCKET=ml-models
MINIO_MODEL_PATH=bncc-embeddings-finetuned
```

> Requer `pip install minio` no requirements.txt

---

## S3

```env
MODEL_SOURCE=s3
MINIO_ACCESS_KEY=sua-aws-access-key
MINIO_SECRET_KEY=sua-aws-secret-key
MINIO_BUCKET=nome-do-bucket
MINIO_MODEL_PATH=bncc-embeddings-finetuned
```

> Requer `pip install boto3` no requirements.txt

---

## HuggingFace

```env
MODEL_SOURCE=huggingface
HF_MODEL_ID=seu-usuario/nome-do-modelo
```

---

## Observações

- O modelo é baixado **uma única vez no startup** do container
- Se o modelo já existir localmente, o download é pulado
- Use a imagem `bncc-ai-service:1.0-runtime` para os modos MinIO, S3 e HuggingFace
- Use a imagem `bncc-ai-service:1.0-model` se quiser o modelo já embedado (não precisa de `MODEL_SOURCE`)
