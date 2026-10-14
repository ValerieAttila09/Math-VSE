class MathVSEError(ValueError):
    """Base exception for MathVSE."""


class InvalidExpressionError(MathVSEError):
    """Raised when the expression grammar is invalid."""


class UnsupportedExpressionError(MathVSEError):
    """Raised for equations outside the supported linear subset."""


class SolverValidationError(MathVSEError):
    """Raised when a candidate solution fails validation."""
