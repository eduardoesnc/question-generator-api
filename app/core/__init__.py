"""
Core utilities
"""
from .logging import setup_logging, get_logger, logger
from .exceptions import (
    NLPAPIException,
    ModelNotLoadedError,
    BNCCDataNotFoundError,
    EmbeddingsNotAvailableError,
    InvalidTextInputError,
    ExtractionFailedError,
    MatcherNotFoundError
)

__all__ = [
    'setup_logging',
    'get_logger',
    'logger',
    'NLPAPIException',
    'ModelNotLoadedError',
    'BNCCDataNotFoundError',
    'EmbeddingsNotAvailableError',
    'InvalidTextInputError',
    'ExtractionFailedError',
    'MatcherNotFoundError',
]
