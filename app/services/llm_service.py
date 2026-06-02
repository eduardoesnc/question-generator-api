"""
LLM Service - extrai campos educacionais via Gemini e gera a questão final.

Fluxo em 2 chamadas:
  1. Gemini extrai disciplina, ano e campos não-BNCC (Bloom, tipo de questão, etc.)
     Python filtra o bncc-data.json localmente para a disciplina/ano encontrados.
  2. Gemini recebe a estrutura BNCC real filtrada + o template da questão num único
     prompt — escolhe a habilidade correta e já gera a questão completa.
"""
import json
import re
import time
from typing import Any, Dict, Optional

from google import genai

from app.config import settings
from app.core.logging import logger

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_STEP1_PROMPT = """\
Você é um especialista em educação brasileira.
Analise o texto abaixo e extraia APENAS os campos solicitados.
Retorne SOMENTE o objeto JSON, sem markdown, sem texto adicional.

Texto: "{text}"

Campos a extrair (use null quando não for possível inferir):
{{
  "disciplina": "use EXATAMENTE um dos nomes: Língua Portuguesa | Matemática | Ciências | Computação | Geografia | História | Arte | Educação Física | Língua Inglesa | Ensino Religioso",
  "ano": "ordinal do ano do ensino fundamental SEM a palavra 'ano' (ex: 1º, 2º, 3º, 4º, 5º, 6º, 7º, 8º, 9º) — use null se não mencionado",
  "perfilAluno": "um de: Bom domínio do conteúdo | Dificuldades pontuais | Muita dificuldade | Perfil diversificado — use null se não mencionado",
  "nivelBloom": "um de: Conhecimento | Compreensão | Aplicação | Análise | Síntese | Avaliação — use null se não mencionado",
  "tipoQuestao": "um de: Múltipla escolha | Dissertativa curta | Dissertativa longa | Verdadeiro ou falso | Associação de colunas — use null se não mencionado",
  "tipoTextoBase": "um de: Documento histórico | Gráfico ou tabela | Poema ou crônica | Notícia ou reportagem | Trecho literário | Charge ou cartum | Mapa ou infográfico | Enunciado matemático | Imagem ou fotografia | Experimento científico | Sem texto base — use null se não mencionado"
}}"""

# Prompt unificado: escolhe habilidade BNCC E gera a questão de uma vez.
_STEP2_PROMPT = """\
Você é um especialista em design educacional e elaborador de itens para exames de larga escala, \
com profundo conhecimento da BNCC e da Taxonomia de Bloom.

## TAREFA
Dado o tema do usuário e a estrutura BNCC real abaixo, você deve:
1. Escolher a unidadeTematica, objetoConhecimento e habilidade mais adequados ao tema (use EXATAMENTE os textos do JSON fornecido).
2. Gerar UMA questão inédita, original e pedagogicamente robusta com base nas especificações.

## TEMA DO USUÁRIO
"{text}"

## ESTRUTURA BNCC REAL — {disciplina}, {ano} ano
(Escolha apenas opções que aparecem aqui. Cópia fiel, sem alterações.)
{bncc_json}

## ESPECIFICAÇÕES DA QUESTÃO
- Nível de Ensino: {ano} ano, ensino fundamental
- Perfil do Aluno: {perfilAluno}
- Disciplina: {disciplina}
- Tipo de Questão: {tipoQuestao}
- Nível de Dificuldade (Taxonomia de Bloom): {nivelBloom}
- Tipo de Texto Base: {tipoTextoBase}
  (A questão DEVE ser precedida por um texto de apoio conciso que sirva de ponto de partida, sem entregar a resposta.)

## REGRAS PARA AS ALTERNATIVAS
- Uma e apenas uma alternativa correta, 100% suportada pelo texto base.
- Distrator 1: interpretação equivocada ou superficial do texto base.
- Distrator 2: afirmação verdadeira para outro período/contexto, incorreta para o assunto.
- Distrator 3: generalização indevida ou simplificação exagerada.
- Distrator 4: inversão de causa/consequência ou confusão entre conceitos relacionados.
- Evite "NÃO", "EXCETO", "INCORRETA" no enunciado. Alternativas com comprimento similar.

## FORMATO DE SAÍDA
Retorne APENAS um bloco JSON válido, sem texto adicional antes ou depois:
```json
{{
  "unidadeTematica": "",
  "objetoConhecimento": "",
  "habilidade": "",
  "disciplina": "{disciplina}",
  "topico": "",
  "assunto_especifico": "",
  "habilidade_avaliada": "",
  "nivel_dificuldade": "{nivelBloom}",
  "texto_base": "",
  "enunciado": "",
  "alternativas": [
    {{ "letra": "A", "texto": "" }},
    {{ "letra": "B", "texto": "" }},
    {{ "letra": "C", "texto": "" }},
    {{ "letra": "D", "texto": "" }},
    {{ "letra": "E", "texto": "" }}
  ],
  "alternativa_correta": "",
  "justificativa_pedagogica": {{
    "justificativa_correta": "",
    "justificativa_distrator_A": "",
    "justificativa_distrator_B": "",
    "justificativa_distrator_C": "",
    "justificativa_distrator_D": "",
    "justificativa_distrator_E": ""
  }}
}}
```"""

_QUESTION_TEMPLATE = """\
Aja como um especialista em design educacional e elaborador de itens para exames de larga escala, com profundo conhecimento da Matriz de Referência do Enem e da Base Nacional Comum Curricular (BNCC). Você é um mestre em taxonomia de Bloom e na criação de questões que avaliam habilidades cognitivas complexas, não apenas memorização. Sua missão principal é gerar UMA questão inédita, original e pedagogicamente robusta, seguindo rigorosamente todas as especificidades, regras e formatos definidos abaixo. A questão deve ser desafiadora, justa e livre de qualquer tipo de viés.

**CONTEXTO PEDAGÓGICO E ALUNO-ALVO**
- Nível de Ensino: [ANO_ESCOLAR] ano, ensino fundamental
- Perfil do Aluno: [PERFIL_DO_ALUNO]
- Disciplina: [DISCIPLINA]
- Tópico Principal: [TOPICO_PRINCIPAL]
- Assunto Específico: [ASSUNTO_ESPECIFICO]

**ESPECIFICAÇÕES DA QUESTÃO**
- Tipo de Questão: [TIPO_DE_QUESTAO]
- Nível de Dificuldade (Taxonomia de Bloom): [NIVEL_DIFICULDADE_BLOOM]
- Habilidade da Matriz de Referência:
  - Código: [CODIGO_HABILIDADE_ENEM]
  - Descrição: [DESCRICAO_HABILIDADE_ENEM]
- Texto Base: A questão DEVE ser precedida por um texto de apoio.
  - Tipo de Texto Base: [TIPO_TEXTO_BASE]
  - O texto deve ser conciso, relevante para o assunto específico e servir como ponto de partida para o raciocínio do aluno, não entregando a resposta diretamente.
- Comando da Questão: O enunciado deve ser claro, direto e conectar o texto base com a habilidade a ser avaliada. Deve instruir o aluno a realizar uma ação cognitiva específica.

**REGRAS PARA AS ALTERNATIVAS**
- Alternativa Correta: Deve haver UMA e APENAS UMA alternativa que responda corretamente ao comando da questão e seja 100% suportada pelos fatos e pela análise do texto base.
- Alternativas Incorretas (Distratores): Os distratores devem ser plausíveis, mas conceitualmente incorretos:
  - Distrator 1: Uma interpretação equivocada ou superficial do texto base.
  - Distrator 2: Uma afirmação que é verdadeira para outro período ou contexto, mas incorreta para o assunto específico da questão.
  - Distrator 3: Uma generalização indevida ou uma simplificação exagerada do tema.
  - Distrator 4: Uma afirmação que inverte a relação de causa e consequência ou confunde conceitos relacionados.
- Linguagem: Evite o uso de "NÃO", "EXCETO", "INCORRETA" no comando da questão. Todas as alternativas devem ter um comprimento e complexidade estrutural semelhantes.

**FORMATO DE SAÍDA (JSON)**
Sua resposta final deve ser APENAS um bloco de código JSON válido, sem nenhum texto ou explicação adicional antes ou depois:
```json
{
  "disciplina": "[DISCIPLINA]",
  "topico": "[TOPICO_PRINCIPAL]",
  "assunto_especifico": "[ASSUNTO_ESPECIFICO]",
  "habilidade_avaliada": "[CODIGO_HABILIDADE_ENEM]",
  "nivel_dificuldade": "[NIVEL_DIFICULDADE_BLOOM]",
  "texto_base": "",
  "enunciado": "",
  "alternativas": [
    { "letra": "A", "texto": "" },
    { "letra": "B", "texto": "" },
    { "letra": "C", "texto": "" },
    { "letra": "D", "texto": "" },
    { "letra": "E", "texto": "" }
  ],
  "alternativa_correta": "",
  "justificativa_pedagogica": {
    "justificativa_correta": "",
    "justificativa_distrator_A": "",
    "justificativa_distrator_B": "",
    "justificativa_distrator_C": "",
    "justificativa_distrator_D": "",
    "justificativa_distrator_E": ""
  }
}
```"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    return json.loads(text.strip())


def _extract_habilidade_code(habilidade: str) -> str:
    match = re.search(r'\(([A-Z0-9]+)\)', habilidade or '')
    return match.group(1) if match else (habilidade or '')[:20]


def _normalize(s: str) -> str:
    return s.lower().strip()


def _find_bncc_section(bncc_data: dict, disciplina: str, ano: str) -> Optional[dict]:
    for disc_key, anos in bncc_data.items():
        if _normalize(disc_key) == _normalize(disciplina):
            for ano_key, unidades in anos.items():
                if _normalize(ano_key.replace('º', '')) == _normalize(ano.replace('º', '')):
                    return unidades
    return None


def _fill_template(fields: Dict[str, Any]) -> str:
    habilidade = fields.get('habilidade') or ''
    codigo = _extract_habilidade_code(habilidade)

    prompt = _QUESTION_TEMPLATE
    prompt = prompt.replace('[ANO_ESCOLAR]', fields.get('ano') or '')
    prompt = prompt.replace('[PERFIL_DO_ALUNO]', fields.get('perfilAluno') or '')
    prompt = prompt.replace('[DISCIPLINA]', fields.get('disciplina') or '')
    prompt = prompt.replace('[TOPICO_PRINCIPAL]', fields.get('unidadeTematica') or '')
    prompt = prompt.replace('[ASSUNTO_ESPECIFICO]', fields.get('objetoConhecimento') or '')
    prompt = prompt.replace('[CODIGO_HABILIDADE_ENEM]', codigo)
    prompt = prompt.replace('[DESCRICAO_HABILIDADE_ENEM]', habilidade)
    prompt = prompt.replace('[TIPO_DE_QUESTAO]', fields.get('tipoQuestao') or '')
    prompt = prompt.replace('[NIVEL_DIFICULDADE_BLOOM]', fields.get('nivelBloom') or '')
    prompt = prompt.replace('[TIPO_TEXTO_BASE]', fields.get('tipoTextoBase') or '')
    return prompt


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LLMService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY não configurada no arquivo .env")
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        with open(settings.bncc_data_path, 'r', encoding='utf-8') as f:
            self._bncc_data = json.load(f)

    def _call_gemini(self, prompt: str) -> tuple[str, dict]:
        """Retorna (texto, uso_de_tokens)."""
        response = self._client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        usage = response.usage_metadata
        tokens = {
            'input_tokens': usage.prompt_token_count or 0,
            'output_tokens': usage.candidates_token_count or 0,
            'total_tokens': usage.total_token_count or 0,
        }
        return response.text, tokens

    def _step1_extract_base_fields(self, text: str) -> tuple[Dict[str, Any], dict]:
        raw, tokens = self._call_gemini(_STEP1_PROMPT.format(text=text))
        try:
            fields = _parse_json_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"Gemini retornou JSON inválido (step 1): {raw[:300]}") from exc
        return {k: (v if v is not None else '') for k, v in fields.items()}, tokens

    def _step2_pick_bncc_and_generate(self, text: str, base_fields: Dict[str, Any]) -> Dict[str, Any]:
        disciplina = base_fields.get('disciplina', '')
        ano = base_fields.get('ano', '')

        section = _find_bncc_section(self._bncc_data, disciplina, ano)
        if not section:
            logger.warning(f"[LLM] Seção BNCC não encontrada para '{disciplina}' / '{ano}'. Usando template simples.")
            filled_prompt = _fill_template(base_fields)
            raw, tokens = self._call_gemini(filled_prompt)
            return {'bncc_found': False, 'raw': raw, 'filled_prompt': filled_prompt, 'tokens': tokens}

        bncc_json = json.dumps(section, ensure_ascii=False, indent=2)
        prompt = _STEP2_PROMPT.format(
            text=text,
            disciplina=disciplina,
            ano=ano,
            bncc_json=bncc_json,
            perfilAluno=base_fields.get('perfilAluno', ''),
            tipoQuestao=base_fields.get('tipoQuestao', ''),
            nivelBloom=base_fields.get('nivelBloom', ''),
            tipoTextoBase=base_fields.get('tipoTextoBase', ''),
        )
        raw, tokens = self._call_gemini(prompt)
        return {'bncc_found': True, 'raw': raw, 'filled_prompt': prompt, 'tokens': tokens}

    def process(self, text: str) -> Dict[str, Any]:
        start = time.time()

        logger.info(f"[LLM] Chamada 1 — extraindo campos base: '{text[:80]}'")
        base_fields, tokens_1 = self._step1_extract_base_fields(text)
        logger.info(f"[LLM] Chamada 1 OK: disciplina={base_fields.get('disciplina')}, ano={base_fields.get('ano')} | tokens: {tokens_1}")

        logger.info("[LLM] Chamada 2 — selecionando habilidade BNCC e gerando questão...")
        step2 = self._step2_pick_bncc_and_generate(text, base_fields)
        tokens_2 = step2['tokens']
        logger.info(f"[LLM] Chamada 2 OK | tokens: {tokens_2}")

        if step2['bncc_found']:
            try:
                parsed = _parse_json_response(step2['raw'])
                bncc_fields = {
                    'unidadeTematica': parsed.get('unidadeTematica', ''),
                    'objetoConhecimento': parsed.get('objetoConhecimento', ''),
                    'habilidade': parsed.get('habilidade', ''),
                }
            except (json.JSONDecodeError, ValueError):
                bncc_fields = {'unidadeTematica': '', 'objetoConhecimento': '', 'habilidade': ''}
        else:
            bncc_fields = {'unidadeTematica': '', 'objetoConhecimento': '', 'habilidade': ''}

        extracted_fields = {**base_fields, **bncc_fields}
        logger.success("[LLM] Questão gerada com sucesso.")

        token_usage = {
            'call_1': tokens_1,
            'call_2': tokens_2,
            'total_input': tokens_1['input_tokens'] + tokens_2['input_tokens'],
            'total_output': tokens_1['output_tokens'] + tokens_2['output_tokens'],
            'total': tokens_1['total_tokens'] + tokens_2['total_tokens'],
        }

        elapsed_ms = (time.time() - start) * 1000
        return {
            "filled_prompt": step2['filled_prompt'],
            "extracted_fields": extracted_fields,
            "generated_question": step2['raw'],
            "original_text": text,
            "processing_time_ms": round(elapsed_ms, 1),
            "token_usage": token_usage,
        }
