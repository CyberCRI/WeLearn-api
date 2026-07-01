"""
Format validation tool - Check output structure using Pydantic
"""

import logging
from typing import Any, Dict, Type

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ValidationResult(BaseModel):
    """Result of format validation"""

    valid: bool
    errors: list[str] = []


def validate_format(
    content: Dict[str, Any], schema: Type[BaseModel]
) -> ValidationResult:
    """
    Validate content against Pydantic schema

    Args:
        content: Content to validate (as dict)
        schema: Pydantic model class to validate against

    Returns:
        ValidationResult with valid status and any errors
    """
    logger.info(f"Validating content against schema: {schema.__name__}")

    try:
        # Attempt to instantiate the schema with content
        schema(**content)
        logger.info("Validation passed")
        return ValidationResult(valid=True, errors=[])

    except ValidationError as e:
        errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
        logger.warning(f"Validation failed: {errors}")
        return ValidationResult(valid=False, errors=errors)

    except Exception as e:
        logger.error(f"Unexpected validation error: {str(e)}")
        return ValidationResult(valid=False, errors=[str(e)])
