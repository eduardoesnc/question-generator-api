"""
Teste rápido para validar o modelo Suggestion
"""
from app.models.responses import Suggestion, ExtractionResponse

# Teste 1: Criar Suggestion
suggestion = Suggestion(
    field="ambiguidade",
    values=["disciplina", "unidadeTematica"],
    message="Campos com baixa confiança"
)

print("✅ Suggestion criado:", suggestion)
print("   JSON:", suggestion.model_dump())

# Teste 2: Criar ExtractionResponse com suggestions
response = ExtractionResponse(
    extracted={"disciplina": "Matemática", "ano": "7º"},
    confidence={"disciplina": 0.65, "ano": 0.95},
    suggestions=[suggestion],
    missing_fields=["unidadeTematica"],
    original_text="Teste"
)

print("\n✅ ExtractionResponse criado:", response)
print("   Suggestions:", response.suggestions)
print("   JSON:", response.model_dump_json(indent=2))

print("\n✅ Todos os testes passaram!")
