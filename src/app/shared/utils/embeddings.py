from functools import cache

import numpy

from src.app.shared.utils.decorators import log_time_and_error_sync


@log_time_and_error_sync
@cache
def convert_embedding_bytes(
    embeddings_byte: bytes, dtype=numpy.float32
) -> numpy.ndarray:
    """
    Converts a byte representation of embeddings to a numpy ndarray.
    Args:
        dtype: The desired data type of the output array. Default is numpy.float32. Only numpy types.
        embeddings_byte: The byte representation of the embeddings.
    Returns: A numpy ndarray of the embeddings.
    """
    if not isinstance(embeddings_byte, bytes):
        raise ValueError(
            f"Embedding must be of type bytes, received type: {type(embeddings_byte).__name__}"
        )
    return numpy.frombuffer(embeddings_byte, dtype=dtype)
