"""Pytest fixtures for testing."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator

import onnx
from onnx import TensorProto, helper
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.database import Base, get_db
from app.main import app
from app.services.storage import LocalStorageService, get_storage_service
from app.services.onnx import ONNXService, get_onnx_service, reset_onnx_service
from app.services.cache import CacheService, get_cache_service


# Use SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test settings override."""
    return Settings(
        database_url=TEST_DATABASE_URL,
        environment="test",
        debug=True,
    )


@pytest_asyncio.fixture(scope="function")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def test_storage(tmp_path: Path) -> LocalStorageService:
    """Create test storage service with temp directory."""
    return LocalStorageService(base_path=str(tmp_path / "models"), max_size_mb=10)


@pytest_asyncio.fixture(scope="function")
async def test_cache() -> CacheService:
    """Create test cache service (disabled for tests)."""
    return CacheService(enabled=False)


@pytest_asyncio.fixture(scope="function")
async def client(
    db_session: AsyncSession, test_storage: LocalStorageService, test_cache: CacheService
) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with overridden dependencies."""

    async def override_get_db():
        yield db_session

    def override_get_storage():
        return test_storage

    async def override_get_cache():
        return test_cache

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_storage_service] = override_get_storage
    app.dependency_overrides[get_cache_service] = override_get_cache

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


def create_simple_onnx_model(
    input_name: str = "input",
    output_name: str = "output",
    input_shape: list[int] = None,
    producer_name: str = "test_producer",
) -> onnx.ModelProto:
    """Create a simple ONNX model for testing.

    Creates a model that performs: output = input + 1
    This is the simplest possible ONNX model with real operations.

    Args:
        input_name: Name of input tensor
        output_name: Name of output tensor
        input_shape: Shape of input tensor (default: [None, 10] for batch + 10 features)
        producer_name: Producer name in metadata

    Returns:
        ONNX ModelProto
    """
    if input_shape is None:
        input_shape = [None, 10]

    # Convert None to string for dynamic dimensions
    onnx_input_shape = [
        d if d is not None else "batch_size" for d in input_shape
    ]

    # Define input
    X = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, onnx_input_shape)

    # Define output
    Y = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, onnx_input_shape)

    # Create constant tensor for adding 1
    one_tensor = helper.make_tensor(
        name="one",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[1.0],
    )

    # Create Add node: output = input + 1
    add_node = helper.make_node(
        "Add",
        inputs=[input_name, "one"],
        outputs=[output_name],
        name="add_one",
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[add_node],
        name="test_graph",
        inputs=[X],
        outputs=[Y],
        initializer=[one_tensor],
    )

    # Create model with IR version 8 for onnxruntime compatibility
    model = helper.make_model(
        graph,
        producer_name=producer_name,
        producer_version="1.0.0",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Set IR version to 8 for compatibility with onnxruntime
    model.ir_version = 8

    # Validate model
    onnx.checker.check_model(model)

    return model


@pytest.fixture
def simple_onnx_model() -> onnx.ModelProto:
    """Fixture providing a simple ONNX model."""
    return create_simple_onnx_model()


@pytest.fixture
def onnx_model_path(tmp_path: Path, simple_onnx_model: onnx.ModelProto) -> Path:
    """Fixture providing path to a saved ONNX model file."""
    model_path = tmp_path / "test_model.onnx"
    onnx.save(simple_onnx_model, str(model_path))
    return model_path


@pytest.fixture
def invalid_onnx_path(tmp_path: Path) -> Path:
    """Fixture providing path to an invalid ONNX file (just random bytes)."""
    invalid_path = tmp_path / "invalid_model.onnx"
    invalid_path.write_bytes(b"this is not a valid onnx model")
    return invalid_path


@pytest.fixture
def onnx_service() -> ONNXService:
    """Fixture providing an ONNXService instance."""
    return ONNXService()


@pytest.fixture(autouse=True)
def reset_onnx_singleton():
    """Reset ONNX service singleton before each test."""
    reset_onnx_service()
    yield
    reset_onnx_service()
