# Regras de Priorização - Sistema NLP

## ✅ Implementado

### Prioridade MÁXIMA (Hard Constraints)

#### 1. Campos Explícitos Vencem
- **Regra**: Qualquer campo fornecido no `context` tem confiança 1.0 e não pode ser sobrescrito
- **Implementação**: `nlp_service.py` - linhas de contexto bloqueado
- **Exemplo**: Se `context = {"ano": "7º"}`, nenhum método pode alterar para 9º

#### 2. Disciplina por Palavra-Chave Forte
- **Regra**: Termos específicos determinam disciplina com alta confiança
- **Mapeamento**:
  - `função, equação, álgebra, geometria` → Matemática (0.90)
  - `algoritmo, programação, código, variável` → Computação (0.90)
  - `texto, leitura, modalização, gramática` → Língua Portuguesa (0.85)
  - `história, vargas, república, ditadura` → História (0.85)
  - `ciências, biologia, física, química` → Ciências (0.85)
- **Implementação**: `_process_with_embeddings()` e `_process_with_hybrid()`

#### 3. Bloqueio de Progressão Pedagógica
- **Regra**: Rejeitar habilidades/objetos se diferença de ano > 2
- **Exemplo**: Se texto menciona 7º ano, rejeitar resultados do 1º ou 9º+ ano
- **Implementação**: Validação em `_process_with_embeddings()` e `_process_with_hybrid()`

#### 4. Veto de Disciplina Conflitante
- **Regra**: Se keyword forte detecta disciplina X mas embedding retorna Y, vetar embedding
- **Exemplo**: Texto com "função" → Matemática detectada → embedding retorna Computação → VETO
- **Implementação**: Validação antes de aplicar resultados de embeddings

### Alta Prioridade (Desambiguação e Fluxo)

#### 5. Pipeline Obrigatório
- **Ordem**: texto → ano → disciplina → unidade temática → objeto → habilidade
- **Implementação**: Estrutura do `pipeline.py` e ordem de extração

#### 6. Desambiguador Semântico
- **Regra**: Detectar disciplina antes de buscar embeddings para filtrar contexto
- **Implementação**: Detecção de `disciplina_hint` antes de `search_global()`

#### 7. Tipo de Representação Apenas com Evidência
- **Regra**: Não defaultar `tipoTextoBase` sem palavra-chave clara
- **Mudança**: Removido default de `grafico_barras` para Matemática
- **Implementação**: `pipeline.py` - seção de defaults removida

### Média Prioridade (Calibração Hybrid)

#### 8. Pesos do Método Hybrid
- **Pesos**: keywords=0.6, embeddings=0.35, sintático=0.05
- **Uso dos pesos**: Apenas para **decidir qual método vence** (comparação)
- **Confiança final**: Usa a **confiança original** do método vencedor (não ponderada)
- **Exemplo**: 
  - Keywords: 65% × 0.6 = 39% (ponderado)
  - Embeddings: 78% × 0.35 = 27% (ponderado)
  - Keywords vence (39% > 27%), mas retorna **confiança original de 65%**
- **Keywords SEMPRE vence**: ano, disciplina, tipoQuestao
- **Embeddings pode vencer**: unidadeTematica, objetoConhecimento, habilidade
- **Implementação**: `_process_with_hybrid()` com comparação de confiânças ponderadas, mas retorna confiança original

#### 9. Fallback Pedagógico Seguro
- **Regra**: Quando incerto em unidade temática, usar genérico da disciplina
- **Mapeamento**:
  - Matemática → Álgebra (0.40)
  - História → História do Brasil (0.40)
  - Língua Portuguesa → Leitura (0.40)
  - Ciências → Vida e evolução (0.40)
  - Computação → Pensamento computacional (0.40)
- **Implementação**: `_process_with_hybrid()` - seção de fallback

#### 10. Emissão de Ambiguidade
- **Regra**: Marcar campos com confiança ≤ 0.7 como ambíguos
- **Output**: Adiciona sugestão no campo `suggestions`
- **Implementação**: `_process_with_hybrid()` - flag de ambiguidade

## 🔧 Como Usar

### Método Hybrid (Recomendado)
```python
result = nlp_service.process(text, method="hybrid")
# Combina keywords e embeddings com pesos calibrados
# Keywords vence: ano, disciplina, tipoQuestao
# Embeddings pode vencer: unidade, objeto, habilidade
# Retorna confiança ORIGINAL do vencedor (não ponderada)
```

## 📊 Exemplo de Comportamento Corrigido

### Antes (Incorreto)
```
Input: "Questão com tabela sobre razão e proporção para 7º ano"
Embeddings: unidadeTematica: 78%
Hybrid: unidadeTematica: 32% (78% × 0.35 + algo) ❌ ERRADO
Resultado: Campo marcado como "faltante" (< 50%)
```

### Depois (Correto)
```
Input: "Questão com tabela sobre razão e proporção para 7º ano"
Keywords: unidadeTematica: 0% (não encontrou)
Embeddings: unidadeTematica: 78%

Comparação ponderada:
- Keywords: 0% × 0.6 = 0%
- Embeddings: 78% × 0.35 = 27%
- Embeddings vence (27% > 0%)

Resultado: unidadeTematica com confiança 78% ✅ CORRETO
```
