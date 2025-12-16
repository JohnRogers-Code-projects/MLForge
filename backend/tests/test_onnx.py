"""Unit tests for ONNX service."""

from pathlib import Path

import onnx
import pytest

from app.services.onnx import (
    ONNXService,
    ONNXLoadError,
    TensorSchema,
    ValidationResult,
    get_onnx_service,
    set_onnx_service,
)
from tests.conftest import create_simple_onnx_model


class TestONNXServiceValidation:
    """Tests for ONNX model validation."""

    def test_validate_valid_model(
        self, onnx_service: ONNXService, onnx_model_path: Path
    ):
        """Validate a valid ONNX model successfully."""
        result = onnx_service.validate(onnx_model_path)

        assert result.valid is True
        assert result.error_message is None
        assert len(result.input_schema) == 1
        assert len(result.output_schema) == 1

    def test_validate_extracts_input_schema(
        self, onnx_service: ONNXService, onnx_model_path: Path
    ):
        """Validate extracts correct input schema."""
        result = onnx_service.validate(onnx_model_path)

        assert len(result.input_schema) == 1
        input_tensor = result.input_schema[0]
        assert input_tensor.name == "input"
        assert input_tensor.dtype == "float32"
        # Shape should have None for batch dimension, 10 for features
        assert input_tensor.shape == [None, 10]

    def test_validate_extracts_output_schema(
        self, onnx_service: ONNXService, onnx_model_path: Path
    ):
        """Validate extracts correct output schema."""
        result = onnx_service.validate(onnx_model_path)

        assert len(result.output_schema) == 1
        output_tensor = result.output_schema[0]
        assert output_tensor.name == "output"
        assert output_tensor.dtype == "float32"
        assert output_tensor.shape == [None, 10]

    def test_validate_extracts_metadata(
        self, onnx_service: ONNXService, onnx_model_path: Path
    ):
        """Validate extracts model metadata."""
        result = onnx_service.validate(onnx_model_path)

        # Producer name should be available
        assert "producer_name" in result.metadata
        assert result.metadata["producer_name"] == "test_producer"
        # producer_version and graph_name may not be available in all onnxruntime versions
        # so we just check providers which is always set
        assert "providers" in result.metadata

    def test_validate_invalid_model_returns_error(
        self, onnx_service: ONNXService, invalid_onnx_path: Path
    ):
        """Validate returns error for invalid ONNX file."""
        result = onnx_service.validate(invalid_onnx_path)

        assert result.valid is False
        assert result.error_message is not None
        assert len(result.input_schema) == 0
        assert len(result.output_schema) == 0

    def test_validate_nonexistent_file(self, onnx_service: ONNXService, tmp_path: Path):
        """Validate returns error for nonexistent file."""
        result = onnx_service.validate(tmp_path / "nonexistent.onnx")

        assert result.valid is False
        assert "not found" in result.error_message.lower()

    def test_validate_wrong_extension(
        self, onnx_service: ONNXService, tmp_path: Path, simple_onnx_model
    ):
        """Validate returns error for wrong file extension."""
        # Save valid ONNX content with wrong extension
        wrong_ext_path = tmp_path / "model.txt"
        onnx.save(simple_onnx_model, str(wrong_ext_path))

        result = onnx_service.validate(wrong_ext_path)

        assert result.valid is False
        assert "extension" in result.error_message.lower()

    def test_validate_accepts_string_path(
        self, onnx_service: ONNXService, onnx_model_path: Path
    ):
        """Validate accepts string path as well as Path object."""
        result = onnx_service.validate(str(onnx_model_path))

        assert result.valid is True


class TestONNXServiceLoadSession:
    """Tests for ONNX session loading."""

    def test_load_session_success(
        self, onnx_service: ONNXService, onnx_model_path: Path
    ):
        """Load session successfully for valid model."""
        session = onnx_service.load_session(onnx_model_path)

        assert session is not None
        assert len(session.get_inputs()) == 1
        assert len(session.get_outputs()) == 1

    def test_load_session_nonexistent_raises(
        self, onnx_service: ONNXService, tmp_path: Path
    ):
        """Load session raises for nonexistent file."""
        with pytest.raises(ONNXLoadError) as exc_info:
            onnx_service.load_session(tmp_path / "nonexistent.onnx")

        assert "not found" in str(exc_info.value).lower()

    def test_load_session_invalid_raises(
        self, onnx_service: ONNXService, invalid_onnx_path: Path
    ):
        """Load session raises for invalid model."""
        with pytest.raises(ONNXLoadError):
            onnx_service.load_session(invalid_onnx_path)


class TestTensorSchema:
    """Tests for TensorSchema dataclass."""

    def test_to_dict(self):
        """TensorSchema converts to dict correctly."""
        schema = TensorSchema(
            name="input",
            dtype="float32",
            shape=[None, 10, 20],
        )

        result = schema.to_dict()

        assert result == {
            "name": "input",
            "dtype": "float32",
            "shape": [None, 10, 20],
        }


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_to_dict_valid_result(self):
        """ValidationResult converts to dict for valid model."""
        result = ValidationResult(
            valid=True,
            input_schema=[TensorSchema("input", "float32", [None, 10])],
            output_schema=[TensorSchema("output", "float32", [None, 10])],
            metadata={"producer_name": "test"},
        )

        result_dict = result.to_dict()

        assert result_dict["valid"] is True
        assert result_dict["error_message"] is None
        assert len(result_dict["input_schema"]) == 1
        assert len(result_dict["output_schema"]) == 1
        assert result_dict["metadata"]["producer_name"] == "test"

    def test_to_dict_error_result(self):
        """ValidationResult converts to dict for error."""
        result = ValidationResult(
            valid=False,
            error_message="Model failed to load",
        )

        result_dict = result.to_dict()

        assert result_dict["valid"] is False
        assert result_dict["error_message"] == "Model failed to load"
        assert result_dict["input_schema"] == []
        assert result_dict["output_schema"] == []


class TestONNXServiceSingleton:
    """Tests for ONNX service singleton pattern."""

    def test_get_onnx_service_returns_instance(self):
        """get_onnx_service returns an ONNXService instance."""
        service = get_onnx_service()

        assert isinstance(service, ONNXService)

    def test_get_onnx_service_returns_same_instance(self):
        """get_onnx_service returns same instance on multiple calls."""
        service1 = get_onnx_service()
        service2 = get_onnx_service()

        assert service1 is service2

    def test_set_onnx_service_overrides(self):
        """set_onnx_service allows overriding the singleton."""
        custom_service = ONNXService(providers=["CPUExecutionProvider"])
        set_onnx_service(custom_service)

        retrieved = get_onnx_service()

        assert retrieved is custom_service


class TestONNXServiceMultiInput:
    """Tests for models with multiple inputs/outputs."""

    def test_validate_multi_input_model(
        self, onnx_service: ONNXService, tmp_path: Path
    ):
        """Validate model with multiple inputs."""
        # Create model with two inputs
        from onnx import TensorProto, helper

        X1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, ["batch", 5])
        X2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, ["batch", 5])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 5])

        add_node = helper.make_node(
            "Add", inputs=["input1", "input2"], outputs=["output"], name="add"
        )

        graph = helper.make_graph(
            nodes=[add_node],
            name="multi_input_graph",
            inputs=[X1, X2],
            outputs=[Y],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8  # Set IR version for onnxruntime compatibility
        model_path = tmp_path / "multi_input.onnx"
        onnx.save(model, str(model_path))

        result = onnx_service.validate(model_path)

        assert result.valid is True
        assert len(result.input_schema) == 2
        assert result.input_schema[0].name == "input1"
        assert result.input_schema[1].name == "input2"
        assert len(result.output_schema) == 1


class TestONNXServiceDtypeConversion:
    """Tests for ONNX dtype conversion."""

    def test_int64_dtype_conversion(self, onnx_service: ONNXService, tmp_path: Path):
        """Validate correctly converts int64 dtype."""
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("input", TensorProto.INT64, ["batch", 5])
        Y = helper.make_tensor_value_info("output", TensorProto.INT64, ["batch", 5])

        identity_node = helper.make_node(
            "Identity", inputs=["input"], outputs=["output"], name="identity"
        )

        graph = helper.make_graph(
            nodes=[identity_node],
            name="int64_graph",
            inputs=[X],
            outputs=[Y],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8  # Set IR version for onnxruntime compatibility
        model_path = tmp_path / "int64_model.onnx"
        onnx.save(model, str(model_path))

        result = onnx_service.validate(model_path)

        assert result.valid is True
        assert result.input_schema[0].dtype == "int64"
        assert result.output_schema[0].dtype == "int64"

    def test_double_dtype_conversion(self, onnx_service: ONNXService, tmp_path: Path):
        """Validate correctly converts double (float64) dtype."""
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("input", TensorProto.DOUBLE, [1, 10])
        Y = helper.make_tensor_value_info("output", TensorProto.DOUBLE, [1, 10])

        identity_node = helper.make_node(
            "Identity", inputs=["input"], outputs=["output"], name="identity"
        )

        graph = helper.make_graph(
            nodes=[identity_node],
            name="double_graph",
            inputs=[X],
            outputs=[Y],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 8  # Set IR version for onnxruntime compatibility
        model_path = tmp_path / "double_model.onnx"
        onnx.save(model, str(model_path))

        result = onnx_service.validate(model_path)

        assert result.valid is True
        assert result.input_schema[0].dtype == "float64"
