"""ONNX model service for validation and metadata extraction.

This module provides functionality to load, validate, and extract metadata
from ONNX models using the ONNX Runtime. It handles schema extraction for
input/output tensors and model metadata like opset version and producer info.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import onnxruntime as ort


class ONNXError(Exception):
    """Base exception for ONNX operations."""

    pass


class ONNXLoadError(ONNXError):
    """Raised when an ONNX model fails to load."""

    pass


class ONNXValidationError(ONNXError):
    """Raised when ONNX model validation fails."""

    pass


@dataclass
class TensorSchema:
    """Schema for a single input or output tensor.

    Attributes:
        name: Tensor name in the model
        dtype: Data type (e.g., 'float32', 'int64')
        shape: Tensor dimensions, None values indicate dynamic axes
    """

    name: str
    dtype: str
    shape: list[Optional[int]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
        }


@dataclass
class ValidationResult:
    """Result of ONNX model validation.

    Attributes:
        valid: Whether the model is valid and loadable
        input_schema: List of input tensor schemas
        output_schema: List of output tensor schemas
        metadata: Model metadata (opset, producer, etc.)
        error_message: Error description if validation failed
    """

    valid: bool
    input_schema: list[TensorSchema] = field(default_factory=list)
    output_schema: list[TensorSchema] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "input_schema": [s.to_dict() for s in self.input_schema],
            "output_schema": [s.to_dict() for s in self.output_schema],
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


# Mapping from ONNX element type enum to human-readable dtype strings
# Based on onnx.TensorProto.DataType enum values
_ONNX_DTYPE_MAP: dict[str, str] = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(double)": "float64",
    "tensor(int8)": "int8",
    "tensor(int16)": "int16",
    "tensor(int32)": "int32",
    "tensor(int64)": "int64",
    "tensor(uint8)": "uint8",
    "tensor(uint16)": "uint16",
    "tensor(uint32)": "uint32",
    "tensor(uint64)": "uint64",
    "tensor(bool)": "bool",
    "tensor(string)": "string",
    "tensor(bfloat16)": "bfloat16",
}


class ONNXService:
    """Service for ONNX model operations.

    Provides functionality to:
    - Load and validate ONNX models
    - Extract input/output tensor schemas
    - Extract model metadata (opset version, producer info, etc.)

    This service uses ONNX Runtime for model loading which validates
    that models are compatible with the runtime and can be used for inference.
    """

    def __init__(self, providers: Optional[list[str]] = None):
        """Initialize ONNX service.

        Args:
            providers: List of execution providers to use.
                      Defaults to ['CPUExecutionProvider'].
        """
        self.providers = providers or ["CPUExecutionProvider"]

    def validate(self, model_path: Path | str) -> ValidationResult:
        """Validate an ONNX model and extract its schemas.

        Attempts to load the model with ONNX Runtime, which performs
        validation checks. If successful, extracts input/output schemas
        and metadata.

        Args:
            model_path: Path to the .onnx model file

        Returns:
            ValidationResult with validity status, schemas, and metadata
        """
        path = Path(model_path)

        if not path.exists():
            return ValidationResult(
                valid=False,
                error_message=f"Model file not found: {path}",
            )

        if not path.suffix.lower() == ".onnx":
            return ValidationResult(
                valid=False,
                error_message=f"Invalid file extension: {path.suffix}. Expected .onnx",
            )

        try:
            session = self._load_session(path)
            input_schema = self._extract_input_schema(session)
            output_schema = self._extract_output_schema(session)
            metadata = self._extract_metadata(session, path)

            return ValidationResult(
                valid=True,
                input_schema=input_schema,
                output_schema=output_schema,
                metadata=metadata,
            )

        except ort.capi.onnxruntime_pybind11_state.InvalidGraph as e:
            return ValidationResult(
                valid=False,
                error_message=f"Invalid ONNX graph: {str(e)}",
            )
        except ort.capi.onnxruntime_pybind11_state.InvalidArgument as e:
            return ValidationResult(
                valid=False,
                error_message=f"Invalid ONNX model argument: {str(e)}",
            )
        except ort.capi.onnxruntime_pybind11_state.NoSuchFile as e:
            return ValidationResult(
                valid=False,
                error_message=f"ONNX file not found: {str(e)}",
            )
        except ort.capi.onnxruntime_pybind11_state.Fail as e:
            return ValidationResult(
                valid=False,
                error_message=f"ONNX Runtime error: {str(e)}",
            )
        except Exception as e:
            # Catch-all for unexpected errors during model loading
            return ValidationResult(
                valid=False,
                error_message=f"Failed to load model: {type(e).__name__}: {str(e)}",
            )

    def load_session(self, model_path: Path | str) -> ort.InferenceSession:
        """Load an ONNX model and return an inference session.

        Args:
            model_path: Path to the .onnx model file

        Returns:
            ONNX Runtime InferenceSession ready for inference

        Raises:
            ONNXLoadError: If model fails to load
        """
        path = Path(model_path)

        if not path.exists():
            raise ONNXLoadError(f"Model file not found: {path}")

        try:
            return self._load_session(path)
        except Exception as e:
            raise ONNXLoadError(f"Failed to load model: {str(e)}") from e

    def _load_session(self, path: Path) -> ort.InferenceSession:
        """Internal method to create inference session.

        Args:
            path: Path to the ONNX model

        Returns:
            ONNX Runtime InferenceSession
        """
        # Use session options for better error messages
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Error level only

        return ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=self.providers,
        )

    def _extract_input_schema(self, session: ort.InferenceSession) -> list[TensorSchema]:
        """Extract input tensor schemas from session.

        Args:
            session: Loaded ONNX Runtime session

        Returns:
            List of TensorSchema for each input
        """
        schemas = []
        for input_meta in session.get_inputs():
            dtype = self._convert_dtype(input_meta.type)
            shape = self._convert_shape(input_meta.shape)
            schemas.append(TensorSchema(
                name=input_meta.name,
                dtype=dtype,
                shape=shape,
            ))
        return schemas

    def _extract_output_schema(self, session: ort.InferenceSession) -> list[TensorSchema]:
        """Extract output tensor schemas from session.

        Args:
            session: Loaded ONNX Runtime session

        Returns:
            List of TensorSchema for each output
        """
        schemas = []
        for output_meta in session.get_outputs():
            dtype = self._convert_dtype(output_meta.type)
            shape = self._convert_shape(output_meta.shape)
            schemas.append(TensorSchema(
                name=output_meta.name,
                dtype=dtype,
                shape=shape,
            ))
        return schemas

    def _extract_metadata(
        self, session: ort.InferenceSession, path: Path
    ) -> dict[str, Any]:
        """Extract model metadata from session.

        Extracts:
        - Opset versions for each domain
        - Producer name and version
        - Model description (if present)
        - Graph name
        - Custom metadata (if any)

        Args:
            session: Loaded ONNX Runtime session
            path: Path to the model file (for additional info)

        Returns:
            Dictionary of metadata
        """
        metadata: dict[str, Any] = {}

        # Get model metadata from session
        model_meta = session.get_modelmeta()

        # Producer information - use getattr for compatibility across onnxruntime versions
        producer_name = getattr(model_meta, "producer_name", None)
        if producer_name:
            metadata["producer_name"] = producer_name

        producer_version = getattr(model_meta, "producer_version", None)
        if producer_version:
            metadata["producer_version"] = producer_version

        # Graph name
        graph_name = getattr(model_meta, "graph_name", None)
        if graph_name:
            metadata["graph_name"] = graph_name

        # Description
        description = getattr(model_meta, "description", None)
        if description:
            metadata["description"] = description

        # Domain
        domain = getattr(model_meta, "domain", None)
        if domain:
            metadata["domain"] = domain

        # Model version
        version = getattr(model_meta, "version", None)
        if version:
            metadata["version"] = version

        # Custom metadata from model
        custom_metadata_map = getattr(model_meta, "custom_metadata_map", None)
        if custom_metadata_map:
            metadata["custom_metadata"] = dict(custom_metadata_map)

        # Provider information
        metadata["providers"] = session.get_providers()

        return metadata

    def _convert_dtype(self, onnx_type: str) -> str:
        """Convert ONNX type string to standard dtype name.

        Args:
            onnx_type: ONNX type string like 'tensor(float)'

        Returns:
            Standard dtype string like 'float32'
        """
        return _ONNX_DTYPE_MAP.get(onnx_type, onnx_type)

    def _convert_shape(self, shape: list) -> list[Optional[int]]:
        """Convert ONNX shape to list of dimensions.

        Dynamic dimensions (strings like 'batch_size') are converted to None.

        Args:
            shape: ONNX shape list

        Returns:
            List of dimensions with None for dynamic axes
        """
        result = []
        for dim in shape:
            if isinstance(dim, int):
                result.append(dim)
            else:
                # Dynamic dimension (string name or None)
                result.append(None)
        return result


# Singleton instance for dependency injection
_onnx_service: Optional[ONNXService] = None


def get_onnx_service() -> ONNXService:
    """Get the ONNX service instance.

    Returns a singleton ONNXService by default.
    Can be overridden for testing.
    """
    global _onnx_service
    if _onnx_service is None:
        _onnx_service = ONNXService()
    return _onnx_service


def set_onnx_service(service: ONNXService) -> None:
    """Set the ONNX service instance (for testing)."""
    global _onnx_service
    _onnx_service = service


def reset_onnx_service() -> None:
    """Reset the ONNX service to None (for testing cleanup)."""
    global _onnx_service
    _onnx_service = None
