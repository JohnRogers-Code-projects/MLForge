"""ONNX model service for validation and metadata extraction.

This module provides functionality to load, validate, and extract metadata
from ONNX models using the ONNX Runtime. It handles schema extraction for
input/output tensors and model metadata like opset version and producer info.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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


class ONNXInferenceError(ONNXError):
    """Raised when ONNX inference fails."""

    pass


class ONNXInputError(ONNXError):
    """Raised when input data is invalid for the model."""

    pass


class PostCommitmentInvariantViolation(ONNXError):
    """The pipeline is in a state that must not exist.

    POST-COMMITMENT INVARIANT VIOLATION
    ====================================

    This exception is raised when post-commitment assumptions are violated.
    It is not an error to be handled. It is a statement that the pipeline
    contract has been broken and continuation is not permitted.

    Violated Invariant:
        After commitment, `file_path` points to a valid, loadable ONNX file.

    Observed State:
        A committed model's file no longer exists on disk.

    Why This Stops Execution:
        The commitment boundary guarantees that post-boundary code may rely
        on certain invariants. If those invariants do not hold, the guarantee
        is broken. Continuing would mean operating on assumptions known to
        be false. The system refuses.

    This Is Not:
        - A transient error (do not retry)
        - An input validation failure (the model was already committed)
        - A recoverable state (there is no fallback)
        - A warning (execution stops here)

    If you are reading this because this exception was raised:
        The pipeline's contract was violated. The only correct response
        is to stop and determine how the invariant was broken.
    """

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
    shape: list[int | None]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
        }


@dataclass
class InferenceResult:
    """Result of running inference on an ONNX model.

    Attributes:
        outputs: Dictionary mapping output names to numpy arrays (converted to lists)
        inference_time_ms: Time taken for inference in milliseconds
    """

    outputs: dict[str, Any]
    inference_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "outputs": self.outputs,
            "inference_time_ms": self.inference_time_ms,
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
    error_message: str | None = None

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

    PURE EXECUTION - NO POLICY DECISIONS
    =====================================

    This service is pure execution code. It makes NO decisions about:
    - Whether to run inference (caller decides)
    - Whether the model is valid for use (commitment boundary decides)
    - Whether to retry on failure (orchestration code decides)
    - Whether to cache results (PredictionCache decides)

    Given a path and input data, this service runs inference. That's it.

    If you find yourself wanting to add policy here (retries, fallbacks,
    confidence checks), that policy belongs in the calling code, not here.

    What This Service Does
    ----------------------
    - Load ONNX files into inference sessions
    - Cache loaded sessions for performance
    - Run inference given a path and input data
    - Detect post-commitment invariant violations (file missing from cache)
    - Convert between ONNX types and numpy types

    Why This Matters
    ----------------
    Separating execution from policy means:
    - This service can be tested in isolation
    - Policy changes require visible changes in orchestration code
    - Adding retries/fallbacks requires modifying callers, not this service
    """

    def __init__(self, providers: list[str] | None = None):
        """Initialize ONNX service.

        Args:
            providers: List of execution providers to use.
                      Defaults to ['CPUExecutionProvider'].
        """
        self.providers = providers or ["CPUExecutionProvider"]
        # Session cache: maps resolved path string to (session, input_names, output_names)
        self._session_cache: dict[
            str, tuple[ort.InferenceSession, list[str], list[str]]
        ] = {}

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

    def get_cached_session(
        self, model_path: Path | str
    ) -> tuple[ort.InferenceSession, list[str], list[str]]:
        """Get a cached inference session, loading if necessary.

        Args:
            model_path: Path to the .onnx model file

        Returns:
            Tuple of (session, input_names, output_names)

        Raises:
            ONNXLoadError: If model fails to load
            PostCommitmentInvariantViolation: If committed model's file no longer exists
        """
        path = Path(model_path).resolve()
        cache_key = str(path)

        # ---------------------------------------------------------------------
        # POST-COMMITMENT INVARIANT CHECK
        # ---------------------------------------------------------------------
        # Invariant: After commitment, file_path points to a valid ONNX file.
        # If we have a cached session but the file is gone, the invariant is
        # violated. This is not corruption detection. This is a statement that
        # the pipeline is in a state that must not exist.
        if cache_key in self._session_cache:
            if not path.exists():
                del self._session_cache[cache_key]
                raise PostCommitmentInvariantViolation(
                    f"POST-COMMITMENT INVARIANT VIOLATED. "
                    f"Invariant: file_path points to a valid ONNX file. "
                    f"Observed: file '{path}' no longer exists. "
                    f"The pipeline contract is broken. Execution cannot continue."
                )

        if cache_key not in self._session_cache:
            session = self.load_session(path)
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            self._session_cache[cache_key] = (session, input_names, output_names)

        return self._session_cache[cache_key]

    def run_inference(
        self,
        model_path: Path | str,
        input_data: dict[str, Any],
    ) -> InferenceResult:
        """Run inference on the model with the given input.

        Args:
            model_path: Path to the .onnx model file
            input_data: Dictionary mapping input names to data.
                       Data can be lists or numpy arrays.

        Returns:
            InferenceResult with outputs and timing

        Raises:
            ONNXLoadError: If model fails to load
            ONNXInputError: If input data is invalid
            ONNXInferenceError: If inference fails
        """
        session, input_names, output_names = self.get_cached_session(model_path)

        # Validate all required inputs are provided
        missing_inputs = set(input_names) - set(input_data.keys())
        if missing_inputs:
            raise ONNXInputError(
                f"Missing required inputs: {', '.join(sorted(missing_inputs))}. "
                f"Expected inputs: {', '.join(input_names)}"
            )

        # Convert inputs to numpy arrays with proper dtype
        try:
            numpy_inputs = self._prepare_inputs(session, input_data)
        except Exception as e:
            raise ONNXInputError(f"Failed to prepare inputs: {str(e)}") from e

        # Run inference with timing
        try:
            start_time = time.perf_counter()
            results = session.run(output_names, numpy_inputs)
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
        except Exception as e:
            raise ONNXInferenceError(f"Inference failed: {str(e)}") from e

        # Convert outputs to serializable format
        outputs = {}
        for name, result in zip(output_names, results, strict=True):
            # Convert numpy arrays to nested lists for JSON serialization
            if isinstance(result, np.ndarray):
                outputs[name] = result.tolist()
            else:
                outputs[name] = result

        return InferenceResult(
            outputs=outputs,
            inference_time_ms=inference_time_ms,
        )

    def _prepare_inputs(
        self,
        session: ort.InferenceSession,
        input_data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Prepare input data by converting to numpy arrays with correct dtypes.

        Args:
            session: ONNX Runtime session
            input_data: Raw input data

        Returns:
            Dictionary of numpy arrays ready for inference
        """
        numpy_inputs = {}
        input_metas = {inp.name: inp for inp in session.get_inputs()}

        for name, data in input_data.items():
            if name not in input_metas:
                # Skip extra inputs (not an error, just ignore)
                continue

            meta = input_metas[name]
            onnx_type = meta.type

            # Determine numpy dtype from ONNX type
            dtype = self._onnx_type_to_numpy_dtype(onnx_type)

            # Convert to numpy array
            if isinstance(data, np.ndarray):
                arr = data.astype(dtype)
            else:
                arr = np.array(data, dtype=dtype)

            numpy_inputs[name] = arr

        return numpy_inputs

    def _onnx_type_to_numpy_dtype(self, onnx_type: str) -> np.dtype:
        """Convert ONNX type string to numpy dtype.

        Args:
            onnx_type: ONNX type string like 'tensor(float)'

        Returns:
            Numpy dtype
        """
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
            "tensor(bool)": np.bool_,
        }
        return np.dtype(dtype_map.get(onnx_type, np.float32))

    def clear_cache(self) -> None:
        """Clear all cached sessions."""
        self._session_cache.clear()

    def remove_from_cache(self, model_path: Path | str) -> bool:
        """Remove a specific model from the session cache.

        Args:
            model_path: Path to the model to remove

        Returns:
            True if model was in cache and removed, False otherwise
        """
        cache_key = str(Path(model_path).resolve())
        if cache_key in self._session_cache:
            del self._session_cache[cache_key]
            return True
        return False

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

    def _extract_input_schema(
        self, session: ort.InferenceSession
    ) -> list[TensorSchema]:
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
            schemas.append(
                TensorSchema(
                    name=input_meta.name,
                    dtype=dtype,
                    shape=shape,
                )
            )
        return schemas

    def _extract_output_schema(
        self, session: ort.InferenceSession
    ) -> list[TensorSchema]:
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
            schemas.append(
                TensorSchema(
                    name=output_meta.name,
                    dtype=dtype,
                    shape=shape,
                )
            )
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

    def _convert_shape(self, shape: list) -> list[int | None]:
        """Convert ONNX shape to list of dimensions.

        Dynamic dimensions (strings like 'batch_size') are converted to None.

        Args:
            shape: ONNX shape list

        Returns:
            List of dimensions with None for dynamic axes
        """
        result: list[int | None] = []
        for dim in shape:
            if isinstance(dim, int):
                result.append(dim)
            else:
                # Dynamic dimension (string name or None)
                result.append(None)
        return result


# Singleton instance for dependency injection
_onnx_service: ONNXService | None = None


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
