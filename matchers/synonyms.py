"""
Mapeamento de sinônimos e variações para melhorar matching
"""

# Sinônimos e variações de termos educacionais
SYNONYMS_MAP = {
    # História - Era Vargas e República
    "era vargas": ["período varguista", "vargas", "getúlio vargas", "governo vargas", "varguista", "getúlio", "era getúlio", "era de vargas"],
    "vargas": ["período varguista", "era vargas", "getúlio vargas", "varguista", "getúlio", "governo vargas", "era de vargas"],
    "período varguista": ["era vargas", "vargas", "getúlio vargas", "varguista", "governo vargas", "era de vargas"],
    "getúlio": ["vargas", "getúlio vargas", "período varguista", "era vargas", "governo vargas", "era de vargas"],
    "getúlio vargas": ["vargas", "período varguista", "era vargas", "varguista", "governo vargas", "era de vargas"],
    "governo vargas": ["era vargas", "período varguista", "vargas", "getúlio vargas", "varguista", "era de vargas"],
    "varguista": ["vargas", "era vargas", "período varguista", "getúlio vargas", "governo vargas", "era de vargas"],
    "era de vargas": ["era vargas", "período varguista", "vargas", "getúlio vargas", "varguista", "governo vargas"],
    
    # História - República e períodos
    "república": ["republicano", "republicana", "regime republicano", "proclamação república"],
    "primeira república": ["república velha", "oligarquias", "café com leite"],
    "república velha": ["primeira república", "oligarquias"],
    "proclamação": ["proclamação república", "início república"],
    
    # História - Trabalhismo e movimentos sociais
    "trabalhismo": ["trabalhista", "movimento trabalhista", "trabalhadores", "direitos trabalhistas"],
    "trabalhista": ["trabalhismo", "movimento trabalhista", "trabalhadores"],
    "trabalhadores": ["trabalhismo", "trabalhista", "classe trabalhadora", "operários"],
    "operários": ["trabalhadores", "classe operária", "proletariado"],
    
    # História - Urbanização
    "urbanização": ["crescimento urbano", "cidades", "metropolização", "urbano", "vida urbana"],
    "urbano": ["urbanização", "cidade", "vida urbana", "crescimento urbano"],
    "urbana": ["urbanização", "cidade", "vida urbana", "crescimento urbano"],
    "vida urbana": ["urbanização", "urbano", "cidade", "crescimento urbano"],
    "segregação espacial": ["segregação urbana", "divisão espacial", "desigualdade espacial"],
    
    # História - Guerras e conflitos
    "segunda guerra": ["segunda guerra mundial", "guerra mundial", "wwii", "2ª guerra", "ii guerra"],
    "segunda guerra mundial": ["segunda guerra", "guerra mundial", "wwii", "2ª guerra"],
    "primeira guerra": ["primeira guerra mundial", "grande guerra", "wwi", "1ª guerra"],
    "guerra fria": ["confronto eua urss", "bipolarização", "capitalismo vs socialismo", "guerra ideológica"],
    
    # História - Ditadura
    "ditadura militar": ["ditadura civil-militar", "regime militar", "golpe de 64", "golpe 1964", "regime autoritário"],
    "ditadura civil-militar": ["ditadura militar", "regime militar", "golpe 64"],
    "regime militar": ["ditadura militar", "ditadura civil-militar", "autoritarismo"],
    "golpe": ["golpe militar", "golpe estado", "tomada poder"],
    
    # História - Outros
    "revolução": ["revolucionário", "movimento revolucionário", "revolta"],
    "independência": ["independente", "emancipação", "libertação"],
    "abolição": ["abolicionismo", "fim escravidão", "libertação escravos"],
    "escravidão": ["escravos", "trabalho escravo", "regime escravista"],
    "colonização": ["colonial", "colonialismo", "período colonial"],
    "império": ["imperial", "período imperial", "monarquia"],
    "constituição": ["carta magna", "lei fundamental", "constituinte"],
    "redemocratização": ["retorno democracia", "abertura política", "transição democrática"],
    
    # Matemática - Criptografia
    "criptografia": ["codificação", "codificação da informação", "sistemas de criptografia", "codificar", "criptografar"],
    "sistemas de criptografia": ["codificação", "codificação da informação", "criptografia", "criptografar"],
    "codificação": ["criptografia", "sistemas de criptografia", "codificar", "código"],
    "codificação da informação": ["criptografia", "sistemas de criptografia", "codificação"],
    
    # Matemática - Outros
    "frações": ["números fracionários", "divisão", "razão", "fração"],
    "equações": ["equação", "expressões algébricas", "álgebra", "igualdade"],
    "geometria": ["geométrico", "figuras geométricas", "formas"],
    "álgebra": ["algébrico", "expressões algébricas", "equações"],
    "probabilidade": ["chance", "possibilidade", "estatística"],
    "porcentagem": ["percentual", "taxa", "razão centesimal"],
    
    # Ciências - Corpo e saúde
    "corpo humano": ["anatomia", "fisiologia", "sistemas do corpo", "organismo"],
    "anatomia": ["corpo humano", "estrutura corporal", "órgãos"],
    "sistema": ["sistemas", "aparelho", "conjunto"],
    
    # Ciências - Meio ambiente
    "meio ambiente": ["ecologia", "natureza", "sustentabilidade", "ambiente"],
    "ecologia": ["meio ambiente", "ecossistema", "natureza"],
    "sustentabilidade": ["sustentável", "preservação", "conservação"],
    "energia": ["formas de energia", "transformação de energia", "energético"],
    
    # Ciências - Outros
    "célula": ["celular", "células", "estrutura celular"],
    "evolução": ["evolutivo", "seleção natural", "darwin"],
    "matéria": ["substância", "material", "composição"],
    
    # Geografia - Clima e relevo
    "clima": ["climatologia", "tempo atmosférico", "fenômenos climáticos", "climático"],
    "relevo": ["formas de relevo", "geomorfologia", "topografia"],
    "vegetação": ["flora", "bioma", "cobertura vegetal"],
    
    # Geografia - População e espaço
    "população": ["demográfico", "demografia", "habitantes", "populoso"],
    "migração": ["migratório", "imigração", "emigração", "deslocamento"],
    "território": ["territorial", "espaço geográfico", "área"],
    "fronteira": ["limite", "divisa", "fronteiras"],
    
    # Geografia - Economia
    "agricultura": ["agrícola", "agropecuária", "cultivo"],
    "indústria": ["industrial", "industrialização", "fábrica"],
    "comércio": ["comercial", "mercado", "trocas"],
    "globalização": ["global", "mundialização", "integração mundial"],
    
    # Português - Interpretação
    "interpretação": ["compreensão textual", "leitura", "análise de texto", "interpretar"],
    "compreensão": ["entendimento", "interpretação", "compreender"],
    "leitura": ["ler", "texto", "interpretação"],
    
    # Português - Gramática
    "gramática": ["sintaxe", "morfologia", "análise linguística", "gramatical"],
    "sintaxe": ["estrutura sintática", "análise sintática", "frase"],
    "morfologia": ["estrutura morfológica", "formação palavras"],
    "ortografia": ["escrita", "grafia", "ortográfico"],
    
    # Português - Gêneros textuais
    "narrativa": ["narração", "conto", "história", "narrativo"],
    "dissertação": ["dissertativo", "argumentação", "texto argumentativo"],
    "descrição": ["descritivo", "caracterização"],
    "poesia": ["poema", "poético", "verso"],
    "crônica": ["cronista", "texto jornalístico"],
}


def expand_query(text: str) -> list:
    """
    Expande uma consulta com sinônimos de forma inteligente
    
    Args:
        text: texto original
        
    Returns:
        lista com texto original + variações (ordenadas por relevância)
    """
    text_lower = text.lower()
    variations = [text_lower]
    
    # Buscar sinônimos - procurar por matches mais longos primeiro
    # Isso evita que "vargas" substitua "era vargas" incorretamente
    sorted_keys = sorted(SYNONYMS_MAP.keys(), key=len, reverse=True)
    
    matched_keys = []
    for key in sorted_keys:
        if key in text_lower:
            matched_keys.append(key)
            # Adicionar sinônimos diretos
            variations.extend(SYNONYMS_MAP[key])
            
            # Substituir no texto original (apenas se não foi substituído antes)
            for syn in SYNONYMS_MAP[key]:
                # Evitar substituições duplicadas
                new_text = text_lower.replace(key, syn)
                if new_text != text_lower:
                    variations.append(new_text)
    
    # Remover duplicatas mantendo ordem
    seen = set()
    unique_variations = []
    for v in variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    return unique_variations


def normalize_term(term: str) -> str:
    """
    Normaliza um termo para comparação
    Remove artigos, preposições e pontuação
    """
    import re
    
    term_lower = term.lower()
    
    # Remover pontuação
    term_lower = re.sub(r'[^\w\s]', ' ', term_lower)
    
    # Remover artigos, preposições e conectivos comuns
    stop_words = {
        'o', 'a', 'os', 'as', 
        'de', 'da', 'do', 'das', 'dos',
        'em', 'na', 'no', 'nas', 'nos',
        'um', 'uma', 'uns', 'umas',
        'ao', 'aos', 'à', 'às',
        'por', 'para', 'com', 'sem',
        'e', 'ou', 'mas', 'que',
        'seu', 'sua', 'seus', 'suas'
    }
    
    words = term_lower.split()
    filtered = [w for w in words if w and w not in stop_words]
    
    return ' '.join(filtered)


def get_key_terms(text: str, include_weights: bool = False) -> set:
    """
    Extrai termos-chave de um texto (substantivos importantes)
    Remove palavras genéricas e mantém apenas termos significativos
    
    Args:
        text: texto para extrair termos
        include_weights: se True, retorna dict com pesos, senão retorna set
        
    Returns:
        conjunto de termos-chave ou dict {termo: peso}
    """
    normalized = normalize_term(text)
    words = normalized.split()
    
    # Palavras muito genéricas que devem ser ignoradas
    generic_words = {
        'história', 'historia', 'brasil', 'mundo', 'país', 'pais',
        'período', 'periodo', 'época', 'epoca', 'tempo', 'ano', 'anos',
        'processos', 'processo', 'questão', 'questao', 'questões', 'questoes',
        'aspectos', 'aspecto', 'características', 'caracteristicas',
        'contexto', 'situação', 'situacao', 'momento', 'fase',
        'parte', 'partes', 'elemento', 'elementos', 'forma', 'formas',
        'sobre', 'com', 'longa', 'análise', 'dissertativa', 'histórico', 'documento',
        'ideal', 'nação', 'nacão', 'moderna', 'moderno',  # Palavras muito genéricas de períodos
        'transformação', 'transformacao', 'desdobramentos', 'desdobramento',
        'nascimento', 'metade', 'século', 'seculo', 'primeiros', 'primeira',
        'era'  # MUITO genérico - causa falsos positivos (era JK vs Era Vargas)
    }
    
    # Termos muito específicos que têm peso maior
    high_value_terms = {
        'vargas', 'varguista', 'getúlio', 'getulio',
        'trabalhismo', 'trabalhista', 'trabalhadores',
        'urbanização', 'urbanizacao', 'urbana', 'urbano',
        'segregação', 'segregacao', 'espacial',
        'ditadura', 'militar', 'golpe', 'autoritário', 'autoritario',
        'guerra', 'revolução', 'revolucao', 'conflito',
        'república', 'republica', 'republicano', 'republicana',
        'abolição', 'abolicao', 'escravidão', 'escravidao',
        'independência', 'independencia', 'colonial', 'colonização', 'colonizacao',
        'jk', 'juscelino', 'kubitschek',  # Para diferenciar "era JK" de "Era Vargas"
        'redemocratização', 'redemocratizacao', 'constituição', 'constituicao',
        'negros', 'indígena', 'indigena', 'quilombolas', 'afrodescendentes',
        'feminino', 'anarquismo', 'totalitarismo', 'fascismo', 'nazismo',
        'holocausto', 'onu', 'direitos', 'humanos'
    }
    
    if include_weights:
        key_terms = {}
        for w in words:
            if w and w not in generic_words and len(w) > 2:
                # Termos específicos têm peso 2.0, outros têm peso 1.0
                weight = 2.0 if w in high_value_terms else 1.0
                key_terms[w] = weight
        # NÃO incluir "era" - é muito genérico e causa falsos positivos
        return key_terms
    else:
        key_terms = {w for w in words if w and w not in generic_words and len(w) > 2}
        # NÃO incluir "era"
        return key_terms
