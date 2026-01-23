"""
Detector de disciplina por termos fortes (Single Responsibility Principle)
"""
from typing import Optional, Tuple


class DisciplineDetector:
    """Detecta disciplina baseado em termos fortes no texto"""
    
    # Mapeamento de termos → (disciplina, confiança)
    DISCIPLINE_TERMS = {
        'Matemática': {
            'terms': [
                'função', 'funcao', 'funções', 'funcoes', 'equação', 'equacao', 'equações', 'equacoes',
                'álgebra', 'algebra', 'geometria', 'trigonometria', 'probabilidade', 'estatística',
                'fração', 'fracao', 'frações', 'fracoes', 'matemática', 'matematica'
            ],
            'confidence': 0.90
        },
        'Computação': {
            'terms': [
                'algoritmo', 'programação', 'programacao', 'código', 'codigo',
                'variável', 'variavel', 'booleano', 'computação', 'computacao'
            ],
            'confidence': 0.90
        },
        'Língua Portuguesa': {
            'terms': [
                'texto', 'leitura', 'escrita', 'modalização', 'modalizacao',
                'português', 'portugues', 'portuguesa', 'gramatica', 'gramática'
            ],
            'confidence': 0.85
        },
        'História': {
            'terms': [
                'história', 'historia', 'histórico', 'historico', 'vargas',
                'república', 'republica', 'ditadura', 'guerra'
            ],
            'confidence': 0.85
        },
        'Ciências': {
            'terms': [
                'ciências', 'ciencias', 'científico', 'cientifico', 'biologia',
                'física', 'fisica', 'química', 'quimica'
            ],
            'confidence': 0.85
        }
    }
    
    @classmethod
    def detect(cls, text: str) -> Tuple[Optional[str], float]:
        """
        Detecta disciplina por termos fortes
        
        Args:
            text: Texto para análise
            
        Returns:
            Tupla (disciplina, confiança) ou (None, 0.0) se não detectado
        """
        text_lower = text.lower()
        
        for discipline, config in cls.DISCIPLINE_TERMS.items():
            if any(term in text_lower for term in config['terms']):
                return discipline, config['confidence']
        
        return None, 0.0
