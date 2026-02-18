"""IR validation package — public API."""

from app.validator.ir_validator import validate
from app.validator.code_validator import validate_code

__all__ = ["validate", "validate_code"]
