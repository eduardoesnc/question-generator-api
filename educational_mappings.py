"""
Mapeamentos de palavras-chave para campos educacionais
"""

DISCIPLINAS_MAP = {
    "Matemática": [
        "matemática", "matematica", "math", "cálculo", "calculo",
        "álgebra", "algebra", "geometria", "aritmética", "aritmetica",
        "números", "numeros", "equações", "equacoes", "frações", "fracoes"
    ],
    "Língua Portuguesa": [
        "português", "portugues", "lingua portuguesa", "língua portuguesa",
        "gramática", "gramatica", "redação", "redacao", "literatura",
        "interpretação", "interpretacao", "texto", "leitura"
    ],
    "Ciências": [
        "ciências", "ciencias", "biologia", "física", "fisica",
        "química", "quimica", "natureza", "meio ambiente", "ecologia",
        "corpo humano", "animais", "plantas"
    ],
    "História": [
        "história", "historia", "histórico", "historico",
        "brasil", "mundo", "guerra", "revolução", "revolucao",
        "período", "periodo", "era", "século", "seculo",
        # Adicionar figuras históricas e eventos importantes
        "vargas", "getúlio", "getulio", "dom pedro", "tiradentes",
        "república", "republica", "império", "imperio", "colonial",
        "ditadura", "democracia", "independência", "independencia",
        "abolição", "abolicao", "escravidão", "escravidao",
        "revolução industrial", "revolucao industrial", "feudalismo",
        "capitalismo", "socialismo", "comunismo", "fascismo", "nazismo"
    ],
    "Geografia": [
        "geografia", "geográfico", "geografico", "mapa", "mapas",
        "região", "regiao", "clima", "relevo", "população", "populacao",
        "território", "territorio", "país", "pais", "continente"
    ],
    "Inglês": [
        "inglês", "ingles", "english", "língua inglesa", "lingua inglesa"
    ],
    "Arte": [
        "arte", "artes", "música", "musica", "pintura", "escultura",
        "teatro", "dança", "danca", "artístico", "artistico"
    ],
    "Educação Física": [
        "educação física", "educacao fisica", "esporte", "esportes",
        "atividade física", "atividade fisica", "ginástica", "ginastica"
    ]
}

# Mapeamento de anos - retorna no formato da BNCC (sem "ano")
ANOS_MAP = {
    "1º": [r"1[oº°]?\s*ano", r"primeiro\s+ano", r"\b1\s*ano\b"],
    "2º": [r"2[oº°]?\s*ano", r"segundo\s+ano", r"\b2\s*ano\b"],
    "3º": [r"3[oº°]?\s*ano", r"terceiro\s+ano", r"\b3\s*ano\b"],
    "4º": [r"4[oº°]?\s*ano", r"quarto\s+ano", r"\b4\s*ano\b"],
    "5º": [r"5[oº°]?\s*ano", r"quinto\s+ano", r"\b5\s*ano\b"],
    "6º": [r"6[oº°]?\s*ano", r"sexto\s+ano", r"\b6\s*ano\b"],
    "7º": [r"7[oº°]?\s*ano", r"s[eé]timo\s+ano", r"\b7\s*ano\b"],
    "8º": [r"8[oº°]?\s*ano", r"oitavo\s+ano", r"\b8\s*ano\b"],
    "9º": [r"9[oº°]?\s*ano", r"nono\s+ano", r"\b9\s*ano\b"],
}

NIVEIS_BLOOM_MAP = {
    "conhecimento": [
        "lembrar", "recordar", "memorizar", "listar", "definir",
        "identificar", "nomear", "reconhecer", "básico", "basico",
        "conhecimento", "memorização", "memorizacao", "recordação", "recordacao",
        "lembrete", "lembrança", "lembranca", "saber", "conhecer"
    ],
    "compreensao": [
        "compreender", "entender", "explicar", "interpretar",
        "resumir", "descrever", "classificar", "comparar",
        "compreensão", "compreensao", "entendimento", "interpretação", "interpretacao",
        "explicação", "explicacao", "descrição", "descricao", "parafrasear"
    ],
    "aplicacao": [
        "aplicar", "usar", "executar", "implementar", "resolver",
        "demonstrar", "praticar", "calcular", "aplicação", "aplicacao",
        "uso", "execução", "execucao", "implementação", "implementacao",
        "resolução", "resolucao", "prática", "pratica", "cálculo", "calculo"
    ],
    "analise": [
        "analisar", "examinar", "investigar", "comparar",
        "diferenciar", "organizar", "desconstruir", "relacionar",
        "análise", "analise", "exame", "investigação", "investigacao",
        "comparação", "comparacao", "diferenciação", "diferenciacao",
        "organização", "organizacao", "relação", "relacao", "conexão", "conexao"
    ],
    "sintese": [
        "criar", "desenvolver", "construir", "planejar",
        "produzir", "inventar", "elaborar", "sintetizar",
        "síntese", "sintese", "criação", "criacao", "desenvolvimento",
        "construção", "construcao", "planejamento", "produção", "producao",
        "invenção", "invencao", "elaboração", "elaboracao", "design", "projetar"
    ],
    "avaliacao": [
        "avaliar", "julgar", "criticar", "justificar",
        "argumentar", "defender", "recomendar", "decidir",
        "avaliação", "avaliacao", "julgamento", "crítica", "critica",
        "justificativa", "argumentação", "argumentacao", "defesa",
        "recomendação", "recomendacao", "decisão", "decisao", "opinar", "opinião", "opiniao"
    ]
}

TIPOS_QUESTAO_MAP = {
    "multipla_escolha": [
        "múltipla escolha", "multipla escolha", "alternativas",
        "opções", "opcoes", "a, b, c", "marcar", "assinalar",
        "escolha múltipla", "escolha multipla", "teste", "quiz",
        "marque", "assinale", "selecione", "escolha a alternativa"
    ],
    "dissertativa_curta": [
        "dissertativa curta", "resposta curta", "breve",
        "resumida", "pequeno texto", "curta", "objetiva curta",
        "responda brevemente", "responda em poucas palavras"
    ],
    "dissertativa_longa": [
        "dissertativa longa", "dissertativa", "redação", "redacao",
        "texto longo", "desenvolver", "argumentar", "longa",
        "escreva um texto", "desenvolva", "argumente", "discorra",
        "elabore um texto", "produza um texto"
    ],
    "verdadeiro_falso": [
        "verdadeiro ou falso", "verdadeiro falso", "v ou f",
        "certo ou errado", "true false", "v/f", "c/e",
        "verdadeiro e falso", "certo e errado"
    ],
    "associacao": [
        "associação", "associacao", "correspondência", "correspondencia",
        "relacionar", "ligar", "conectar", "combinar",
        "relacione", "ligue", "conecte", "combine", "associe",
        "correlação", "correlacao", "matching"
    ]
}

TIPOS_TEXTO_BASE_MAP = {
    "documento_historico": [
        "documento histórico", "documento historico", "documento",
        "fonte histórica", "fonte historica", "trecho histórico",
        "fonte primária", "fonte primaria", "documento original",
        "registro histórico", "registro historico"
    ],
    "texto_literario": [
        "texto literário", "texto literario", "literatura",
        "fragmento literário", "poesia", "prosa", "literário", "literario",
        "trecho literário", "trecho literario", "obra literária", "obra literaria",
        "conto", "romance", "crônica", "cronica"
    ],
    "artigo_jornal": [
        "artigo", "jornal", "notícia", "noticia", "reportagem",
        "matéria", "materia", "jornalístico", "jornalistico",
        "artigo de jornal", "texto jornalístico", "texto jornalistico",
        "manchete", "editorial"
    ],
    "charge": [
        "charge", "cartum", "cartoon", "tirinha", "quadrinho",
        "história em quadrinhos", "historia em quadrinhos", "hq",
        "caricatura", "desenho satírico", "desenho satirico"
    ],
    "grafico_barras": [
        "gráfico de barras", "grafico de barras", "gráfico em barras",
        "barras", "gráfico vertical", "grafico vertical",
        "gráfico de colunas", "grafico de colunas"
    ],
    "grafico_linhas": [
        "gráfico de linhas", "grafico de linhas", "gráfico linear",
        "linhas", "evolução", "evolucao", "grafico linear",
        "gráfico temporal", "grafico temporal", "série temporal", "serie temporal"
    ],
    "tabela": [
        "tabela", "dados tabulados", "planilha", "dados em tabela",
        "quadro", "matriz de dados"
    ],
    "imagem": [
        "imagem", "foto", "fotografia", "figura", "ilustração", "ilustracao",
        "fotografia", "picture", "visual", "representação visual", "representacao visual"
    ],
    "mapa": [
        "mapa", "cartográfico", "cartografico", "geográfico", "geografico",
        "mapa geográfico", "mapa geografico", "carta geográfica", "carta geografica",
        "planisfério", "planisferio", "globo"
    ],
    "infografico": [
        "infográfico", "infografico", "infografia",
        "gráfico informativo", "grafico informativo", "visualização de dados",
        "visualizacao de dados"
    ],
    "poema": [
        "poema", "poesia", "verso", "letra de música", "letra de musica",
        "poético", "poetico", "estrofe", "rima", "soneto"
    ]
}

PERFIS_ALUNO_MAP = {
    "bom_dominio": [
        "bom domínio", "bom dominio", "boa leitura", "avançado em leitura",
        "lê bem", "le bem", "domina bem", "boa compreensão", "boa compreensao",
        "leitura fluente", "bom leitor", "boa interpretação", "boa interpretacao"
    ],
    "dificuldade_conexao": [
        "dificuldade em conectar", "dificuldade de conexão",
        "dificuldade de conexao", "básico mas com dificuldade",
        "dificuldade para relacionar", "dificuldade de interpretação",
        "dificuldade de interpretacao", "dificuldade em relacionar"
    ],
    "conhecimento_basico": [
        "conhecimento básico", "conhecimento basico", "básico", "basico",
        "iniciante", "fundamental", "nível básico", "nivel basico",
        "introdutório", "introdutorio", "elementar", "inicial"
    ],
    "conhecimento_avancado": [
        "conhecimento avançado", "conhecimento avancado", "avançado",
        "avancado", "profundo", "expert", "nível avançado", "nivel avancado",
        "aprofundado", "especializado", "superior", "alto nível", "alto nivel"
    ]
}
