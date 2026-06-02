# Modo LLM — Geração de Questões via Gemini

Este documento descreve a arquitetura, o fluxo de execução, as decisões de projeto e as melhorias planejadas para o modo LLM da API.

---

## Visão Geral

O modo LLM substitui o pipeline de embeddings locais por chamadas à API do Google Gemini. A partir de uma descrição em linguagem natural, o sistema extrai os campos educacionais necessários, ancora as habilidades BNCC em dados reais e gera a questão completa — tudo via LLM.

**Endpoint:** `POST /api/generate-llm`

**Configuração necessária no `.env`:**
```env
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.0-flash-lite   # ou outro modelo disponível
```

---

## Arquitetura em 2 Chamadas

O fluxo foi dividido em duas chamadas sequenciais ao Gemini para evitar alucinação de códigos BNCC — principal problema de uma abordagem em chamada única.

```
Usuário (texto livre)
        │
        ▼
┌─────────────────────────────────────┐
│  Chamada 1 — Extração de campos     │
│  Gemini recebe o texto e retorna    │
│  JSON com disciplina, ano, Bloom,   │
│  tipo de questão, perfil do aluno   │
│  e tipo de texto base.              │
└─────────────────────────────────────┘
        │
        ▼ Python filtra bncc-data.json localmente (~0ms)
        │  usando disciplina + ano extraídos
        │
        ▼
┌─────────────────────────────────────┐
│  Chamada 2 — Seleção BNCC + Geração │
│  Gemini recebe a estrutura BNCC     │
│  REAL filtrada para aquela          │
│  disciplina/ano e o template de     │
│  questão. Escolhe a habilidade      │
│  correta e já gera a questão        │
│  completa no mesmo prompt.          │
└─────────────────────────────────────┘
        │
        ▼
   Resposta ao cliente
```

### Por que não uma chamada única?

Para selecionar a habilidade BNCC correta, é preciso filtrar o `bncc-data.json` pelo par disciplina/ano. Esse filtro só é possível depois de conhecer a disciplina e o ano — informações que vêm da Chamada 1. Uma abordagem em chamada única exigiria enviar o JSON completo da BNCC (~700KB) para o Gemini, o que tornaria o prompt inviável em tamanho e custo.

### Por que não três chamadas?

Uma iteração anterior separava: (1) extração de campos não-BNCC, (2) seleção de habilidade BNCC, (3) geração da questão. Isso adicionava latência sem ganho real de qualidade, pois as etapas 2 e 3 podem ser combinadas no mesmo prompt sem perda de precisão.

---

## Prevenção de Alucinação de Códigos BNCC

O principal risco de abordagens LLM para conteúdo BNCC é a **alucinação de códigos**: o modelo inventa um código plausível (ex: `EF09HI07`) mas com a descrição errada, ou cria códigos inexistentes.

A solução adotada:

1. **Chamada 1** não pede campos BNCC — pede apenas disciplina e ano.
2. **Python** carrega a seção exata do `bncc-data.json` para aquele par disciplina/ano.
3. **Chamada 2** recebe essa seção como parte do prompt com instrução explícita de copiar os textos fielmente, sem alterações.

Resultado: o Gemini escolhe entre opções reais, nunca inventa.

---

## Resposta da API

```json
{
  "filled_prompt": "Prompt padrão preenchido com os campos extraídos...",
  "extracted_fields": {
    "disciplina": "História",
    "ano": "9º",
    "perfilAluno": "Bom domínio do conteúdo",
    "unidadeTematica": "O nascimento da República no Brasil...",
    "objetoConhecimento": "O Brasil na era Vargas",
    "habilidade": "(EF09HI05) Identificar e discutir...",
    "nivelBloom": "Análise",
    "tipoQuestao": "Múltipla escolha",
    "tipoTextoBase": "Documento histórico"
  },
  "generated_question": "{...}",
  "original_text": "Texto enviado pelo usuário",
  "processing_time_ms": 28400.0,
  "token_usage": {
    "call_1": { "input_tokens": 312, "output_tokens": 89, "total_tokens": 401 },
    "call_2": { "input_tokens": 4201, "output_tokens": 823, "total_tokens": 5024 },
    "total_input": 4513,
    "total_output": 912,
    "total": 5425
  }
}
```

O campo `generated_question` é o JSON bruto retornado pelo Gemini, com a seguinte estrutura quando válido:

```json
{
  "unidadeTematica": "...",
  "objetoConhecimento": "...",
  "habilidade": "...",
  "disciplina": "...",
  "topico": "...",
  "assunto_especifico": "...",
  "habilidade_avaliada": "...",
  "nivel_dificuldade": "...",
  "texto_base": "...",
  "enunciado": "...",
  "alternativas": [
    { "letra": "A", "texto": "..." },
    { "letra": "B", "texto": "..." },
    { "letra": "C", "texto": "..." },
    { "letra": "D", "texto": "..." },
    { "letra": "E", "texto": "..." }
  ],
  "alternativa_correta": "A",
  "justificativa_pedagogica": {
    "justificativa_correta": "...",
    "justificativa_distrator_A": "...",
    "justificativa_distrator_B": "...",
    "justificativa_distrator_C": "...",
    "justificativa_distrator_D": "...",
    "justificativa_distrator_E": "..."
  }
}
```

---

## Monitoramento de Tokens

Cada chamada ao Gemini retorna `usage_metadata` com os tokens consumidos. O serviço captura esses valores e os agrega:

- **input_tokens** (`prompt_token_count`): tokens do prompt enviado
- **output_tokens** (`candidates_token_count`): tokens da resposta gerada
- **total_tokens** (`total_token_count`): soma reportada pela API

Os valores são exibidos no frontend tanto no modo LLM quanto na página de comparação, permitindo análise de custo por requisição.

---

## Melhorias Identificadas

### 1. Prompt de produção sem justificativas pedagógicas

**Impacto:** alto — **Esforço:** baixo

As justificativas são a maior porção do output da Chamada 2. Um segundo template de prompt sem esse bloco reduz ~40% dos tokens de saída e, proporcionalmente, a latência. Implementação: parâmetro `mode: "producao" | "detalhado"` na request; o backend seleciona o template adequado.

---

### 2. Modelo mais rápido

**Impacto:** alto — **Esforço:** baixo

O modelo é configurável via `GEMINI_MODEL` no `.env`. Modelos como `gemini-2.0-flash-lite` e `gemini-1.5-flash-8b` são significativamente mais rápidos que versões maiores, com qualidade aceitável para o domínio educacional estruturado. Recomenda-se comparar latência e qualidade de saída antes de fixar o modelo para produção.

---

### 3. Streaming da resposta

**Impacto:** alto (percepção de velocidade) — **Esforço:** médio

O tempo total não diminui, mas o usuário vê a questão sendo gerada progressivamente em vez de aguardar ~30s em tela branca. A API do Gemini suporta streaming via `client.models.generate_content_stream()`. No frontend, o endpoint precisaria retornar `text/event-stream` (SSE) ou `ndjson`.

---

### 4. Redução do JSON da BNCC enviado na Chamada 2

**Impacto:** alto — **Esforço:** médio

Atualmente a seção inteira da disciplina/ano é enviada ao Gemini. Para História do 9º ano, por exemplo, isso inclui todas as unidades temáticas e dezenas de habilidades. Uma pré-filtragem por similaridade de palavras-chave entre o texto do usuário e os nomes das unidades temáticas reduziria o input da Chamada 2 para apenas as 2-3 unidades mais prováveis, diminuindo tokens e latência sem perda de precisão.

---

### 5. Fusão das duas chamadas em uma única

**Impacto:** alto — **Esforço:** médio

Eliminaria a latência de rede de uma chamada (~15s). Viável se o prompt incluir a lista de disciplinas disponíveis já na primeira chamada, permitindo que o Gemini identifique disciplina e ano, e o backend então filtre o BNCC e reenvie tudo em sequência síncrona — ou, alternativamente, enviando as disciplinas e anos possíveis junto com o JSON BNCC completo de forma truncada.

---

### 6. Cache de extrações

**Impacto:** médio — **Esforço:** baixo

Textos idênticos ou muito similares produzem a mesma extração. Um cache em memória (dict Python com hash do texto) ou Redis elimina chamadas repetidas, especialmente útil em ambiente de testes e demonstrações.

---

## Análise: Tudo em Uma Única Chamada?

Uma dúvida natural é: por que não enviar o texto do usuário e pedir ao Gemini que, num único prompt, faça a extração dos campos, escolha a habilidade BNCC e gere a questão completa?

### Por que parece atraente

- Eliminaria a latência de rede de uma chamada inteira (~15s)
- Simplificaria o código: sem orquestração, sem estado intermediário
- Menos pontos de falha

### Por que não funciona na prática

O problema central é o `bncc-data.json`. Para garantir que o Gemini escolha um código BNCC real — e não um inventado —, é preciso enviar os dados reais da BNCC no prompt. Mas para saber *quais* dados enviar, é preciso saber a disciplina e o ano. E a disciplina e o ano só são conhecidos *depois* da extração.

Isso cria uma dependência circular que não tem solução em chamada única sem comprometer uma das garantias:

| Abordagem única | Consequência |
|---|---|
| Enviar o JSON completo da BNCC (~700KB) | Prompt inviável em tamanho, custo e latência — provavelmente excede o contexto útil |
| Enviar apenas a lista de disciplinas e deixar o Gemini inferir a habilidade livremente | Voltamos ao problema original de alucinação de códigos BNCC |
| Enviar uma versão comprimida da BNCC (só códigos e descrições curtas) | Reduz o risco de alucinação mas não elimina; ainda é um prompt enorme |

### Conclusão

A chamada única é um trade-off: você ganha velocidade e perde garantia de precisão nos códigos BNCC. Para um sistema de avaliação educacional, onde o código errado invalida o alinhamento curricular, esse trade-off não compensa.

A arquitetura em 2 chamadas existe exatamente por esse motivo: a primeira chamada é barata (poucos tokens, resposta pequena) e desbloqueia o filtro local que torna a segunda chamada precisa. O custo da primeira chamada em latência (~3–5s) é muito menor que o custo de enviar todo o BNCC numa única vez.

**Quando a chamada única faria sentido:** se o requisito de precisão BNCC for relaxado — por exemplo, em um protótipo rápido onde o educador revisará a habilidade antes de usar a questão. Nesse caso, um único prompt com os campos livres e sem âncora no JSON real seria suficiente e mais simples.

---

## Junção com Embeddings: análise e proposta

### Valeria a pena?

Sim — e o ganho seria significativo, especialmente para a etapa de seleção de habilidade BNCC.

O problema atual é que o JSON da BNCC enviado na Chamada 2 pode conter dezenas de habilidades para a disciplina/ano. O Gemini precisa ler e comparar todas para escolher a mais adequada ao texto do usuário — tarefa custosa em tokens e tempo.

Os embeddings já resolvem exatamente esse problema: o modelo fine-tuned foi treinado para medir similaridade semântica entre textos e habilidades BNCC. Ele faz essa busca em milissegundos, localmente, sem custo de API.

### Como seria a junção (arquitetura híbrida)

```
Usuário (texto livre)
        │
        ▼
┌──────────────────────────────────────────┐
│  Chamada 1 — Gemini (mantida)            │
│  Extrai: disciplina, ano, Bloom,         │
│  tipo de questão, perfil, tipo de texto  │
└──────────────────────────────────────────┘
        │
        ▼ Python — EmbeddingsMatcher (local, ~50ms)
        │  Busca as top-3 habilidades BNCC mais
        │  similares ao texto para aquela disciplina/ano
        │
        ▼
┌──────────────────────────────────────────┐
│  Chamada 2 — Gemini (muito menor)        │
│  Recebe apenas as 3 habilidades          │
│  candidatas (não o JSON inteiro) +       │
│  o template de questão.                  │
│  Confirma a habilidade e gera a questão. │
└──────────────────────────────────────────┘
```

### Ganhos esperados

| Aspecto | Sem embeddings | Com embeddings |
|---|---|---|
| JSON BNCC no prompt da Chamada 2 | Seção inteira (~50 habilidades) | 3 candidatas |
| Tokens de input (Chamada 2) | ~4.000–6.000 | ~800–1.200 |
| Latência estimada | ~28–35s | ~12–18s |
| Risco de escolha errada de habilidade | Baixo (Gemini lê tudo) | Muito baixo (embeddings filtram semanticamente) |
| Dependência do modelo local | Nenhuma | Modelo fine-tuned necessário |

### Custo da integração

A integração é direta: o `EmbeddingsMatcher` já está disponível no projeto. Bastaria instanciá-lo no `LLMService.__init__` e chamar `search_with_ensemble(text)` após a Chamada 1, passando o resultado filtrado para construir o prompt da Chamada 2.

A principal ressalva é que a abordagem híbrida mantém dependência do modelo local (~1GB), o que pode ser indesejável em cenários onde o modo LLM seria a alternativa ao modelo de embeddings (ex: ambiente sem GPU ou sem o modelo fine-tuned disponível).

---

### Variante mais impactante: substituir a Chamada 1 pelo embeddings

A proposta acima usa embeddings para auxiliar a Chamada 2. Mas há uma abordagem mais interessante: **substituir a Chamada 1 inteiramente pelo pipeline de embeddings**, eliminando uma chamada de API e reduzindo o Gemini a uma única responsabilidade — gerar a questão.

Isso funciona porque o `EmbeddingsMatcher` já extrai exatamente os campos que a Chamada 1 extrai hoje:

| Campo | Chamada 1 (Gemini) | EmbeddingsMatcher (local) |
|---|---|---|
| disciplina | sim | sim |
| ano | sim | sim |
| unidadeTematica | não extrai | sim |
| objetoConhecimento | não extrai | sim |
| habilidade | não extrai | sim (código real) |
| nivelBloom | sim | sim |
| tipoQuestao | sim | sim |
| tipoTextoBase | sim | sim |
| perfilAluno | sim | sim |

O embeddings faz mais do que a Chamada 1: além dos campos não-BNCC, já retorna a habilidade BNCC com código real e score de confiança — exatamente o que hoje exige a Chamada 2 também.

**Arquitetura resultante:**

```
Usuário (texto livre)
        │
        ▼
┌──────────────────────────────────────────┐
│  EmbeddingsMatcher (local, ~200ms)       │
│  Extrai todos os 9 campos incluindo      │
│  disciplina, ano e habilidade BNCC real  │
└──────────────────────────────────────────┘
        │
        ▼ Python preenche o template com os campos extraídos
        │
        ▼
┌──────────────────────────────────────────┐
│  Chamada única — Gemini                  │
│  Recebe o template já preenchido e       │
│  apenas gera a questão.                  │
│  Sem necessidade de extrair ou escolher. │
└──────────────────────────────────────────┘
```

**Comparação de latência:**

| Arquitetura | Chamadas à API | Latência estimada |
|---|---|---|
| Atual (2 chamadas Gemini) | 2 | ~28–35s |
| Híbrida (embeddings filtra Chamada 2) | 2 | ~12–18s |
| Híbrida (embeddings substitui Chamada 1) | 1 | ~8–12s |

**Vale a pena?**

Sim, é a abordagem mais eficiente das três. O ganho é duplo: elimina ~15s de latência da Chamada 1 e entrega os campos BNCC com código real sem custo adicional, já que o modelo local já faz isso.

A única ressalva relevante é o comportamento em textos muito vagos. O pipeline de embeddings depende de similaridade semântica — se o texto for curto demais (ex: "questão de matemática"), os scores de confiança serão baixos e vários campos ficarão sem extração. O Gemini na Chamada 1 atual consegue inferir campos implícitos com mais flexibilidade por entender contexto de forma generativa.

Uma solução para isso seria usar os scores de confiança do embeddings como critério: campos com confiança acima do threshold seguem direto para o template; campos abaixo do threshold são complementados por uma chamada Gemini focada apenas nesses campos faltantes. Isso manteria a robustez para inputs vagos sem sacrificar velocidade nos casos comuns.
