"""
Matcher para disciplinas escolares
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers.base_matcher import BaseMatcher

DISCIPLINAS_PATTERNS = {
    "Matemática": [
        "matemática", "matematica", "math", "cálculo", "calculo",
        "álgebra", "algebra", "geometria", "aritmética", "aritmetica",
        "números", "numeros", "equações", "equacoes", "frações", "fracoes",
        "trigonometria", "estatística", "estatistica", "probabilidade"
    ],
    "Língua Portuguesa": [
        "português", "portugues", "lingua portuguesa", "língua portuguesa",
        "gramática", "gramatica", "redação", "redacao", "literatura",
        "interpretação", "interpretacao", "texto", "leitura", "escrita",
        "ortografia", "sintaxe", "morfologia", "semântica", "semantica"
    ],
    "Ciências": [
        "ciências", "ciencias", "biologia", "física", "fisica",
        "química", "quimica", "natureza", "meio ambiente", "ecologia",
        "corpo humano", "animais", "plantas", "células", "celulas",
        "energia", "matéria", "materia"
    ],
    "História": [
        "história", "historia", "histórico", "historico",
        "brasil", "mundo", "guerra", "revolução", "revolucao",
        "período", "periodo", "era", "século", "seculo",
        "civilização", "civilizacao", "império", "imperio",
        "república", "republica", "ditadura", "democracia"
    ],
    "Geografia": [
        "geografia", "geográfico", "geografico", "mapa", "mapas",
        "região", "regiao", "clima", "relevo", "população", "populacao",
        "território", "territorio", "país", "pais", "continente",
        "urbanização", "urbanizacao", "globalização", "globalizacao"
    ],
    "Inglês": [
        "inglês", "ingles", "english", "língua inglesa", "lingua inglesa",
        "vocabulary", "grammar", "reading", "writing"
    ],
    "Arte": [
        "arte", "artes", "música", "musica", "pintura", "escultura",
        "teatro", "dança", "danca", "artístico", "artistico",
        "desenho", "cultura", "estética", "estetica"
    ],
    "Educação Física": [
        "educação física", "educacao fisica", "esporte", "esportes",
        "atividade física", "atividade fisica", "ginástica", "ginastica",
        "jogos", "atletismo", "saúde", "saude"
    ]
}


class DisciplinasMatcher(BaseMatcher):
    def __init__(self, nlp):
        super().__init__(nlp, DISCIPLINAS_PATTERNS)
