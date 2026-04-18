"""
Sistema de logging estruturado
"""
import logging
import sys
from typing import Optional
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Formatter com cores para terminal"""
    
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)

class CustomLogger(logging.Logger):
    """Logger customizado com método success"""
    
    def success(self, message, *args, **kwargs):
        """Log de sucesso (mesmo nível que INFO mas com emoji)"""
        if self.isEnabledFor(logging.INFO):
            self.info(f"✅ {message}", *args, **kwargs)

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configura logging da aplicação
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Logger configurado
    """
    logging.setLoggerClass(CustomLogger)
    
    logger = logging.getLogger("nlp_api")
    logger.setLevel(getattr(logging, level.upper()))
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: Optional[str] = None) -> CustomLogger:
    """
    Retorna logger para um módulo específico
    
    Args:
        name: Nome do módulo (opcional)
    
    Returns:
        Logger configurado
    """
    logging.setLoggerClass(CustomLogger)
    if name:
        return logging.getLogger(f"nlp_api.{name}")
    return logging.getLogger("nlp_api")

logger = get_logger()
