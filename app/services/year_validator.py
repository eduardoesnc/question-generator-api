"""
Validador de progressão pedagógica (Single Responsibility Principle)
"""
from typing import Optional

class YearValidator:
    """Valida progressão pedagógica entre anos escolares"""
    
    YEAR_MAP = {
        '1º': 1, '2º': 2, '3º': 3, '4º': 4, '5º': 5,
        '6º': 6, '7º': 7, '8º': 8, '9º': 9
    }
    
    @classmethod
    def is_valid_progression(
        cls,
        year_extracted: Optional[str],
        year_embedding: Optional[str],
        max_difference: int = 2
    ) -> bool:
        """
        Valida se a diferença entre anos está dentro da progressão pedagógica
        
        Args:
            year_extracted: Ano extraído do texto (ex: "7º")
            year_embedding: Ano retornado pelo embedding (ex: "9º")
            max_difference: Diferença máxima permitida em anos
            
        Returns:
            True se válido, False caso contrário
        """
        if not year_extracted or not year_embedding:
            return True
        
        year_num_extracted = cls.YEAR_MAP.get(year_extracted, 0)
        year_num_embedding = cls.YEAR_MAP.get(year_embedding, 0)
        
        if year_num_extracted == 0 or year_num_embedding == 0:
            return True
        
        return abs(year_num_embedding - year_num_extracted) <= max_difference
