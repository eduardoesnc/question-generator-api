"""
Limpador de texto para embeddings (Single Responsibility Principle)
"""
import re
from typing import Set

class TextCleaner:
    """Limpa texto removendo instruções e ruído para busca semântica"""
    
    INSTRUCTION_WORDS: Set[str] = {
        'quero', 'crie', 'elabore', 'faça', 'gere', 'desenvolva', 'construa',
        'questões', 'questoes', 'questão', 'questao',
        'estilo', 'tipo', 'tipos', 'formato',
        'enem', 'prova', 'exame', 'teste', 'avaliação', 'avaliacao',
        'alunos', 'aluno', 'estudantes', 'estudante',
        'para', 'sobre', 'com', 'de', 'do', 'da', 'dos', 'das', 'uma', 'um', 'o', 'a', 'as', 'os'
    }
    
    YEAR_PATTERNS = [
        r'\d+[ºª°]\s*ano',
        r'(primeiro|segundo|terceiro|quarto|quinto|sexto|sétimo|setimo|oitavo|nono)\s*ano'
    ]
    
    @classmethod
    def clean_for_embeddings(cls, text: str, min_word_length: int = 2) -> str:
        """
        Limpa texto para busca por embeddings
        
        Args:
            text: Texto original
            min_word_length: Tamanho mínimo de palavra para manter
            
        Returns:
            Texto limpo
        """
        cleaned = text.lower()
        
        for pattern in cls.YEAR_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned)
        
        words = cleaned.split()
        cleaned_words = [
            w for w in words
            if w not in cls.INSTRUCTION_WORDS and len(w) > min_word_length
        ]
        
        return ' '.join(cleaned_words) if cleaned_words else text
