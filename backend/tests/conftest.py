"""Pytest fixtures for testing."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings, settings
from app.database import Base, get_db
from app.main import app
from app.services.onnx_service import onnx_service
from app.services.storage_service import storage_service


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
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def test_model_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test models."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def simple_onnx_model(test_model_dir: Path) -> Path:
    """
    Create a simple ONNX model for testing.

    This creates a minimal model that adds two inputs.
    """
    try:
        import onnx
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx package required for this test")

    # Create a simple Add model: output = input_a + input_b
    input_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [1, 3])
    input_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 3])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])

    add_node = helper.make_node("Add", ["input_a", "input_b"], ["output"])

    graph = helper.make_graph([add_node], "test_add", [input_a, input_b], [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    model_path = test_model_dir / "test_add.onnx"
    onnx.save(model, str(model_path))

    return model_path


@pytest.fixture(scope="function")
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary storage directory for each test."""
    temp_dir = Path(tempfile.mkdtemp())
    # Override storage service path
    original_path = storage_service._base_path
    storage_service._base_path = temp_dir
    yield temp_dir
    storage_service._base_path = original_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def clean_onnx_service() -> Generator[None, None, None]:
    """Clean up ONNX service sessions after each test."""
    try:
        yield
    finally:
        # Unload all models
        for model_id in list(onnx_service._sessions.keys()):
            onnx_service.unload_model(model_id)
