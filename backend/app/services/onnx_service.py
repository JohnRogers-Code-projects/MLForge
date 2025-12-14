"""ONNX Runtime service for model validation and inference."""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class ONNXValidationError(Exception):
    """Raised when ONNX model validation fails."""

    pass


class ONNXInferenceError(Exception):
    """Raised when inference fails."""

    pass


class ONNXService:
    """
    Service for ONNX model validation and inference.

    Manages ONNX Runtime sessions with caching for performance.
    Provides schema extraction and input validation.
    """

    def __init__(self) -> None:
        # Cache of loaded ONNX sessions: model_id -> InferenceSession
        self._sessions: dict[str, ort.InferenceSession] = {}
        # Session options for all models
        self._session_options = ort.SessionOptions()
        self._session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        # Use CPU by default; GPU support can be added via providers
        self._providers = ["CPUExecutionProvider"]

    def validate_model(self, file_path: str) -> dict[str, Any]:
        """
        Validate an ONNX model file and extract metadata.

        Args:
            file_path: Path to the ONNX model file

        Returns:
            Dictionary containing:
            - input_schema: List of input specifications
            - output_schema: List of output specifications
            - file_hash: SHA256 hash of the file
            - file_size_bytes: Size of the file
            - opset_version: ONNX opset version
            - ir_version: ONNX IR version

        Raises:
            ONNXValidationError: If the model is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise ONNXValidationError(f"Model file not found: {file_path}")

        if not path.suffix.lower() == ".onnx":
            raise ONNXValidationError(f"Invalid file extension: {path.suffix}")

        # Calculate file hash and size
        file_size = path.stat().st_size
        file_hash = self._calculate_file_hash(path)

        try:
            # Try to load the model - this validates the ONNX format
            session = ort.InferenceSession(
                str(path),
                sess_options=self._session_options,
                providers=self._providers,
            )
        except Exception as e:
            raise ONNXValidationError(f"Failed to load ONNX model: {e}") from e

        # Extract input schema
        input_schema = []
        for inp in session.get_inputs():
            input_schema.append({
                "name": inp.name,
                "shape": inp.shape,
                "dtype": inp.type,
            })

        # Extract output schema
        output_schema = []
        for out in session.get_outputs():
            output_schema.append({
                "name": out.name,
                "shape": out.shape,
                "dtype": out.type,
            })

        # Extract model metadata
        model_meta = session.get_modelmeta()
        metadata = {
            "producer_name": model_meta.producer_name,
            "graph_name": model_meta.graph_name,
            "domain": model_meta.domain,
            "description": model_meta.description,
            "version": model_meta.version,
            "custom_metadata": dict(model_meta.custom_metadata_map),
        }

        return {
            "input_schema": input_schema,
            "output_schema": output_schema,
            "file_hash": file_hash,
            "file_size_bytes": file_size,
            "metadata": metadata,
        }

    def load_model(self, model_id: str, file_path: str) -> None:
        """
        Load an ONNX model into the session cache.

        Args:
            model_id: Unique identifier for the model
            file_path: Path to the ONNX model file

        Raises:
            ONNXValidationError: If the model cannot be loaded
        """
        if model_id in self._sessions:
            logger.info(f"Model {model_id} already loaded, skipping")
            return

        try:
            session = ort.InferenceSession(
                file_path,
                sess_options=self._session_options,
                providers=self._providers,
            )
            self._sessions[model_id] = session
            logger.info(f"Loaded model {model_id} from {file_path}")
        except Exception as e:
            raise ONNXValidationError(f"Failed to load model {model_id}: {e}") from e

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from the session cache.

        Args:
            model_id: The model to unload

        Returns:
            True if the model was unloaded, False if not found
        """
        if model_id in self._sessions:
            del self._sessions[model_id]
            logger.info(f"Unloaded model {model_id}")
            return True
        return False

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded in the session cache."""
        return model_id in self._sessions

    def warmup_model(self, model_id: str, file_path: str) -> float:
        """
        Warm up a model by loading and running a dummy inference.

        This ensures the model is ready for fast inference.

        Args:
            model_id: Unique identifier for the model
            file_path: Path to the ONNX model file

        Returns:
            Warmup time in milliseconds

        Raises:
            ONNXValidationError: If warmup fails
        """
        start_time = time.perf_counter()

        self.load_model(model_id, file_path)
        session = self._sessions[model_id]

        # Create dummy inputs based on input schema
        dummy_inputs = self._create_dummy_inputs(session)

        # Run a warmup inference
        try:
            session.run(None, dummy_inputs)
        except Exception as e:
            logger.warning(f"Warmup inference failed for {model_id}: {e}")
            # Don't fail warmup if dummy inference fails
            # The model is still loaded and valid

        warmup_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Model {model_id} warmed up in {warmup_time_ms:.2f}ms")
        return warmup_time_ms

    def predict(
        self,
        model_id: str,
        input_data: dict[str, Any],
        file_path: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        """
        Run inference on a model.

        Args:
            model_id: The model to use for inference
            input_data: Dictionary mapping input names to numpy arrays or lists
            file_path: Optional path to load model if not cached

        Returns:
            Tuple of (output_data dict, inference_time_ms)

        Raises:
            ONNXInferenceError: If inference fails
        """
        # Load model if not in cache
        if model_id not in self._sessions:
            if file_path is None:
                raise ONNXInferenceError(
                    f"Model {model_id} not loaded and no file_path provided"
                )
            self.load_model(model_id, file_path)

        session = self._sessions[model_id]

        # Validate and prepare inputs
        prepared_inputs = self._prepare_inputs(session, input_data)

        # Run inference
        start_time = time.perf_counter()
        try:
            outputs = session.run(None, prepared_inputs)
        except Exception as e:
            raise ONNXInferenceError(f"Inference failed: {e}") from e
        inference_time_ms = (time.perf_counter() - start_time) * 1000

        # Build output dictionary
        output_names = [out.name for out in session.get_outputs()]
        output_data = {}
        for name, value in zip(output_names, outputs):
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(value, np.ndarray):
                output_data[name] = value.tolist()
            else:
                output_data[name] = value

        return output_data, inference_time_ms

    def get_input_schema(self, model_id: str) -> list[dict[str, Any]]:
        """Get the input schema for a loaded model."""
        if model_id not in self._sessions:
            raise ONNXInferenceError(f"Model {model_id} not loaded")

        session = self._sessions[model_id]
        return [
            {"name": inp.name, "shape": inp.shape, "dtype": inp.type}
            for inp in session.get_inputs()
        ]

    def get_output_schema(self, model_id: str) -> list[dict[str, Any]]:
        """Get the output schema for a loaded model."""
        if model_id not in self._sessions:
            raise ONNXInferenceError(f"Model {model_id} not loaded")

        session = self._sessions[model_id]
        return [
            {"name": out.name, "shape": out.shape, "dtype": out.type}
            for out in session.get_outputs()
        ]

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return list(self._sessions.keys())

    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _prepare_inputs(
        self,
        session: ort.InferenceSession,
        input_data: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Prepare and validate inputs for inference.

        Converts lists to numpy arrays with correct dtypes.
        """
        prepared = {}
        expected_inputs = {inp.name: inp for inp in session.get_inputs()}

        for name, spec in expected_inputs.items():
            if name not in input_data:
                raise ONNXInferenceError(f"Missing required input: {name}")

            value = input_data[name]

            # Convert to numpy array if needed
            if not isinstance(value, np.ndarray):
                value = np.array(value)

            # Convert to expected dtype
            dtype = self._onnx_dtype_to_numpy(spec.type)
            if dtype is not None and value.dtype != dtype:
                value = value.astype(dtype)

            prepared[name] = value

        return prepared

    def _create_dummy_inputs(
        self,
        session: ort.InferenceSession,
    ) -> dict[str, np.ndarray]:
        """Create dummy inputs for warmup inference."""
        dummy = {}
        for inp in session.get_inputs():
            # Replace dynamic dimensions with 1
            shape = [1 if isinstance(d, str) or d is None else d for d in inp.shape]
            dtype = self._onnx_dtype_to_numpy(inp.type) or np.float32
            dummy[inp.name] = np.zeros(shape, dtype=dtype)
        return dummy

    def _onnx_dtype_to_numpy(self, onnx_type: str) -> np.dtype | None:
        """Convert ONNX type string to numpy dtype."""
        type_map = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(uint8)": np.uint8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
            "tensor(bool)": np.bool_,
            "tensor(string)": np.object_,
        }
        return type_map.get(onnx_type)


    def clear_all_sessions(self) -> None:
        """
        Clear all loaded ONNX Runtime sessions and free resources.
        """
        for session in self._sessions.values():
            try:
                session._sess = None  # Help GC, though not strictly necessary
                del session
            except Exception:
                pass
        self._sessions.clear()
# Singleton instance
onnx_service = ONNXService()
