"""
Matcher para níveis da Taxonomia de Bloom
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchers.base_matcher import BaseMatcher

BLOOM_PATTERNS = {
    "conhecimento": [
        "lembrar", "recordar", "memorizar", "listar", "definir",
        "identificar", "nomear", "reconhecer", "conhecimento",
        "memorização", "memorizacao", "saber", "conhecer",
        "relembrar", "citar", "enumerar", "rotular"
    ],
    "compreensao": [
        "compreender", "entender", "explicar", "interpretar",
        "resumir", "descrever", "classificar", "comparar",
        "compreensão", "compreensao", "entendimento", "interpretação",
        "interpretacao", "parafrasear", "ilustrar", "exemplificar"
    ],
    "aplicacao": [
        "aplicar", "usar", "executar", "implementar", "resolver",
        "demonstrar", "praticar", "calcular", "aplicação", "aplicacao",
        "utilizar", "empregar", "operar", "solucionar"
    ],
    "analise": [
        "analisar", "examinar", "investigar", "comparar",
        "diferenciar", "organizar", "desconstruir", "relacionar",
        "análise", "analise", "distinguir", "categorizar",
        "contrastar", "separar", "dividir"
    ],
    "sintese": [
        "criar", "desenvolver", "construir", "planejar",
        "produzir", "inventar", "elaborar", "sintetizar",
        "síntese", "sintese", "design", "projetar",
        "formular", "compor", "gerar", "combinar"
    ],
    "avaliacao": [
        "avaliar", "julgar", "criticar", "justificar",
        "argumentar", "defender", "recomendar", "decidir",
        "avaliação", "avaliacao", "opinar", "validar",
        "verificar", "testar", "medir", "estimar"
    ]
}


class BloomMatcher(BaseMatcher):
    def __init__(self, nlp):
        super().__init__(nlp, BLOOM_PATTERNS)
