"""Services layer for business logic."""

from app.services.storage import (
    StorageService,
    LocalStorageService,
    StorageError,
    StorageFullError,
    get_storage_service,
)
from app.services.onnx import (
    ONNXService,
    ONNXError,
    ONNXLoadError,
    ONNXValidationError,
    ONNXInferenceError,
    ONNXInputError,
    TensorSchema,
    ValidationResult,
    InferenceResult,
    get_onnx_service,
)
from app.services.cache import (
    CacheService,
    CacheError,
    get_cache_service,
    close_cache_service,
    set_cache_service,
)
from app.services.model_cache import (
    ModelCache,
    model_to_cache_dict,
)

__all__ = [
    # Storage
    "StorageService",
    "LocalStorageService",
    "StorageError",
    "StorageFullError",
    "get_storage_service",
    # ONNX
    "ONNXService",
    "ONNXError",
    "ONNXLoadError",
    "ONNXValidationError",
    "ONNXInferenceError",
    "ONNXInputError",
    "TensorSchema",
    "ValidationResult",
    "InferenceResult",
    "get_onnx_service",
    # Cache
    "CacheService",
    "CacheError",
    "get_cache_service",
    "close_cache_service",
    "set_cache_service",
    # Model Cache
    "ModelCache",
    "model_to_cache_dict",
]
