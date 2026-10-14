class MathVSEError(ValueError):
    """Base exception for MathVSE."""


class ImageValidationError(MathVSEError):
    """Raised when an uploaded image cannot be safely validated or decoded."""


class EmptyImageError(ImageValidationError):
    """Raised when a file is empty or contains no image data."""


class UnsupportedImageFormatError(ImageValidationError):
    """Raised for unsupported raster formats."""


class ImageTooLargeError(ImageValidationError):
    """Raised when file size or decoded dimensions exceed configured limits."""


class InvalidImageDataError(ImageValidationError):
    """Raised when image data is corrupted or cannot be decoded safely."""


class InvalidExpressionError(MathVSEError):
    """Raised when the expression grammar is invalid."""


class UnsupportedExpressionError(MathVSEError):
    """Raised for equations outside the supported linear subset."""


class SolverValidationError(MathVSEError):
    """Raised when a candidate solution fails validation."""
