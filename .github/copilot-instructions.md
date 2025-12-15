# GitHub Copilot Instructions for ModelForge

## Project Overview

ModelForge is a production-ready ML model serving platform that allows users to deploy and serve ONNX models at scale. The platform provides:

- **High-performance API** for model management and inference
- **Model versioning** and metadata storage
- **Prediction caching** for optimized performance
- **Async job processing** for long-running inference tasks
- **Management dashboard** (planned) for operations

**Target Users**: ML Engineers, Data Scientists, DevOps teams deploying machine learning models in production.

**Current Status**: Phase 1 complete (Core Backend Foundation). Next: ONNX Runtime Integration (Phase 2).

**Architecture**:
```
Next.js UI → FastAPI API → PostgreSQL DB
              ↓    ↓    ↓
            Redis ONNX Celery
            Cache Runtime Worker
```

## Tech Stack

### Backend (Current Focus)
- **Python**: 3.11+
- **Framework**: FastAPI 0.109.0
- **Database**: PostgreSQL with SQLAlchemy 2.0 (async) and Alembic migrations
- **Caching**: Redis 5.0+
- **ML Runtime**: ONNX Runtime 1.16+ (Phase 2)
- **Task Queue**: Celery with Redis broker (Phase 4)
- **Testing**: pytest with pytest-asyncio and pytest-cov
- **Validation**: Pydantic 2.5+

### Frontend (Phase 5 - Future)
- **Framework**: Next.js 14 (App Router)
- **Auth**: NextAuth.js

### Deployment
- **Platform**: Railway
- **Container**: Docker with docker-compose for local development

## Coding Standards and Guidelines

### Python Style
- Follow **PEP 8** style guide
- Use **type hints** for all function parameters and return values
- Use **async/await** for all database and I/O operations
- Maximum line length: 100 characters (not the default 88)
- Use double quotes for strings (not single quotes)

### Code Organization
- **Models**: SQLAlchemy ORM models in `backend/app/models/`
- **Schemas**: Pydantic validation schemas in `backend/app/schemas/`
- **CRUD**: Database operations in `backend/app/crud/`
- **API Routes**: FastAPI route handlers in `backend/app/api/`
- **Config**: Application settings in `backend/app/config.py` using pydantic-settings

### Naming Conventions
- **Files**: Snake_case (e.g., `ml_model.py`, `predictions.py`)
- **Classes**: PascalCase (e.g., `ModelCreate`, `PredictionResponse`)
- **Functions/Variables**: snake_case (e.g., `create_model`, `model_id`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `API_PREFIX`, `DATABASE_URL`)
- **Database tables**: Plural snake_case (e.g., `models`, `predictions`, `jobs`)

### API Design Patterns
- Use **resource-based routing** (e.g., `/api/v1/models`, `/api/v1/predictions`)
- Return **Pydantic schemas** for response validation
- Use **HTTP status codes** correctly (201 for created, 404 for not found, etc.)
- Include **pagination** for list endpoints (using `skip` and `limit` query params)
- Use **dependency injection** via `Depends()` for database sessions and common parameters

### Database Patterns
- Use **async SQLAlchemy** (AsyncSession) for all database operations
- Define models with `DeclarativeBase` base class
- Use **Alembic** for all schema migrations
- Add **indexes** for frequently queried fields (name, version, status)
- Include `created_at` and `updated_at` timestamps on all models

### Error Handling
- Raise **HTTPException** for API errors with appropriate status codes
- Use specific error messages that don't expose sensitive information
- Log errors before raising exceptions (when logging is implemented)

### Documentation
- Include **docstrings** for all public functions and classes (Google-style format)
- Add **inline comments** for complex logic, algorithms, or non-obvious decisions
- Keep OpenAPI/Swagger documentation up to date (FastAPI auto-generates this)

## Security Practices

### Authentication & Authorization
- Plan to use **JWT tokens** for API authentication (Phase 5+)
- Never commit secrets or API keys to version control
- Use **environment variables** for all sensitive configuration

### Input Validation
- **Always validate** user input using Pydantic schemas
- Use **query parameter validation** with FastAPI Query/Path validators
- Sanitize file uploads (especially for ONNX models in Phase 2)

### Database Security
- Use **parameterized queries** (SQLAlchemy ORM handles this)
- Never construct SQL queries with string concatenation
- Limit database user permissions in production

### Dependencies
- Keep dependencies **up to date** for security patches
- Review security advisories before adding new dependencies
- Use specific version pins in `requirements.txt` (not ranges)

### CORS
- Configure CORS restrictively in production (currently allows all origins for development)

## Testing Requirements

### Test Coverage
- Target: **90%+ code coverage** for backend
- Run tests with: `cd backend && pytest -v --cov=app`

### Test Organization
- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test database operations and API endpoints
- Use **fixtures** for database setup and teardown (see `tests/conftest.py`)

### Test Patterns
- Use **async test functions** with `@pytest.mark.asyncio` decorator
- Use **TestClient** from `httpx` for API endpoint testing
- Mock external dependencies (ONNX runtime, Celery, Redis when needed)
- Write tests that **catch real bugs**, not just happy-path scenarios

### Test Naming
- Name test files: `test_<module_name>.py`
- Name test functions: `test_<function_name>_<scenario>`
- Example: `test_create_model_duplicate_name_raises_error`

## Development Workflow

### Local Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Database (Docker)
docker-compose up -d db redis

# Run migrations
alembic upgrade head

# Start server
uvicorn app.main:app --reload
```

### Database Migrations
```bash
# Create migration after model changes
alembic revision --autogenerate -m "description of changes"

# Apply migrations
alembic upgrade head

# Rollback last migration
alembic downgrade -1
```

### Running Tests
```bash
cd backend
pytest -v --cov=app
pytest tests/test_api/test_models.py -v  # Run specific test file
```

### Docker Development
```bash
docker-compose up -d        # Start all services
docker-compose logs -f api  # View logs
docker-compose down         # Stop services
```

## Project-Specific Patterns

### Async Database Sessions
- Always use `async with` context manager for database sessions
- Use dependency injection: `db: DBSession = Depends(get_db)`
- Don't manually commit/rollback - CRUD operations handle this

### CRUD Operations
- Use CRUD classes in `backend/app/crud/` for database operations
- Inherit from `CRUDBase` for common operations (get, get_multi, create, update, delete)
- Add custom methods for specific queries

### Pydantic Schemas
- Separate schemas for **Create**, **Update**, and **Response**
- Use `ConfigDict(from_attributes=True)` for ORM model conversion
- Add field validation with Pydantic validators when needed

### FastAPI Dependencies
- `DBSession`: Database session (via `get_db()`)
- `ModelDep`: Model lookup by ID with automatic 404 handling
- Add new dependencies in `backend/app/api/deps.py`

## Phase-Specific Guidance

### Current Phase: Phase 1 Complete ✅
The core backend foundation is established with:
- FastAPI app structure
- Database models and migrations
- Basic CRUD operations
- Health check endpoints
- Docker Compose setup

### Next Phase: Phase 2 (ONNX Runtime Integration)
When working on Phase 2, focus on:
- Model file upload and storage (consider local filesystem vs S3)
- ONNX model validation before saving
- Model metadata extraction (input/output shapes, types)
- Synchronous inference endpoint
- Input/output schema validation per model
- Model warmup on startup/upload

### Future Phases
- **Phase 3**: Redis caching with TTL and invalidation
- **Phase 4**: Celery async jobs with status tracking
- **Phase 5**: Next.js dashboard with authentication
- **Phase 6**: Comprehensive testing and documentation
- **Phase 7**: Railway deployment and CI/CD

## Communication Style Preferences

### Be Direct and Honest
- **No sycophancy** - Be critical when code has issues
- **Question requirements** if they seem unclear or suboptimal
- **Suggest better alternatives** when applicable
- **Explain architectural decisions** before implementing

### Code Quality Focus
- **Point out code smells** and technical debt
- **Suggest refactors** rather than just adding features
- **Use industry-standard patterns** and explain why
- **Flag dependencies** that are overkill or outdated

### Clarity and Explanation
- **Explain complex logic** inline (regex, algorithms, gnarly code)
- **Rubber-duck** complex decisions step by step
- **Flag performance implications** proactively
- **Flag security concerns** proactively

### Testing and Safety
- Write **tests that catch real bugs** (not just happy-path)
- Consider edge cases and error scenarios
- Think about production implications

## External References

For detailed project roadmap and phase-specific deliverables, see:
- [CLAUDE.md](../CLAUDE.md) - Complete build plan and architecture
- [README.md](../README.md) - Quick start guide and API documentation

## Additional Notes

### Dependencies
- Avoid adding new dependencies unless necessary
- When adding dependencies, explain the choice and why alternatives weren't suitable
- Check for security advisories before adding packages

### Performance Considerations
- Use async operations for all I/O (database, file operations, external APIs)
- Consider caching strategies for frequently accessed data
- Profile code for performance bottlenecks when adding inference capabilities

### Future Considerations
- Keep scalability in mind (horizontal scaling of API servers)
- Design for observability (structured logging, metrics)
- Plan for model versioning and A/B testing capabilities
