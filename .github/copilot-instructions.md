# ModelForge Copilot Instructions

## Repository Overview
**ModelForge** is a production-ready ML model serving platform built with FastAPI for backend API, ONNX Runtime for model inference, PostgreSQL for data storage, Redis for caching, Celery for async job processing, and Next.js for the management dashboard (Phase 5, not yet implemented).

**Project Size**: Small-to-medium Python backend (~2,000 lines of code)  
**Languages**: Python 3.11+ (tested with 3.12.3)  
**Frameworks**: FastAPI 0.109, SQLAlchemy 2.0 (async), Alembic, pytest  
**Dependencies**: asyncpg, Redis, ONNX Runtime, Celery, Pydantic 2.5  
**Status**: Phase 1 complete (core backend), Phase 2-7 planned

---

## Critical Known Issues ⚠️

### 1. SQLAlchemy Reserved Name Conflict (BLOCKING)
**Problem**: The `MLModel` class in `backend/app/models/ml_model.py` has a field named `metadata` (line 56), which is a reserved name in SQLAlchemy's Declarative API.

**Impact**: Application, tests, and migrations **will not run** until fixed.

**Error**:
```
sqlalchemy.exc.InvalidRequestError: Attribute name 'metadata' is reserved when using the Declarative API.
```

**Workaround**: Rename the field to `model_metadata` or `meta_data` in:
- `backend/app/models/ml_model.py` (line 56)
- `backend/app/schemas/ml_model.py` (corresponding schema field)
- `backend/alembic/versions/001_initial_migration.py` (line 38)
- Any CRUD operations or API handlers using this field

### 2. ONNX Runtime Version Incompatibility
**Problem**: `requirements.txt` specifies `onnxruntime==1.16.3`, which is unavailable in PyPI.

**Error**:
```
ERROR: Could not find a version that satisfies the requirement onnxruntime==1.16.3
```

**Workaround**: Use `onnxruntime>=1.17.0` instead. Version 1.23.2 has been tested and works.

### 3. Pydantic Protected Namespace Warning
**Problem**: `backend/app/config.py` has a field `model_storage_path` that conflicts with Pydantic's protected "model_" namespace.

**Warning**:
```
UserWarning: Field "model_storage_path" has conflict with protected namespace "model_".
```

**Workaround**: Add to `Settings` class in `backend/app/config.py`:
```python
model_config = SettingsConfigDict(
    ...,
    protected_namespaces=('settings_',)
)
```

---

## Environment Setup

### Prerequisites
- **Python**: 3.11+ (tested with 3.12.3)
- **Docker & Docker Compose**: For PostgreSQL and Redis
- **System packages**: gcc, libpq-dev (for psycopg2)

### Initial Setup (First Time)
```bash
# 1. Navigate to backend directory
cd backend

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# 4. Install dependencies
# NOTE: Fix onnxruntime version first if not already done
pip install -r requirements.txt

# 5. Copy environment file
cp .env.example .env
# Edit .env if needed (defaults work for local development)

# 6. Start PostgreSQL and Redis via Docker
# From repository root:
cd ..
docker compose up -d db redis

# 7. Wait for database to be healthy (5-10 seconds)
# Check with: docker compose ps
# Wait until mlforge-db-1 shows "healthy" status

# 8. Run database migrations
cd backend
alembic upgrade head

# NOTE: This will FAIL if the SQLAlchemy metadata issue is not fixed
```

**Time Estimates**:
- Initial setup: 5-10 minutes
- Docker image pulls: 30-60 seconds
- Dependencies install: 60-120 seconds
- Database ready: 5-10 seconds

---

## Build & Test Commands

### Running the Application

**Start API Server (Development)**:
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- Server starts on: `http://localhost:8000`
- API docs: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

**IMPORTANT**: The application will **not start** until the SQLAlchemy `metadata` field conflict is resolved.

### Running Tests

**Run All Tests**:
```bash
cd backend
source venv/bin/activate
pytest -v
```

**Run with Coverage**:
```bash
pytest -v --cov=app
```

**Run Specific Test File**:
```bash
pytest tests/test_health.py -v
```

**Test Configuration**: Defined in `backend/pytest.ini`
- Uses SQLite for testing (`sqlite+aiosqlite:///./test.db`)
- Async mode enabled
- Test fixtures in `backend/tests/conftest.py`

**IMPORTANT**: Tests will **not run** until the SQLAlchemy `metadata` field conflict is resolved.

### Database Migrations

**Apply Migrations**:
```bash
cd backend
source venv/bin/activate
alembic upgrade head
```

**Create New Migration** (after model changes):
```bash
alembic revision --autogenerate -m "description of changes"
# Review the generated file in backend/alembic/versions/
alembic upgrade head
```

**Rollback Migration**:
```bash
alembic downgrade -1  # Go back one version
```

**Migration Files**: Located in `backend/alembic/versions/`
- Current: `001_initial_migration.py` (creates ml_models, predictions, jobs tables)

### Docker Commands

**Start All Services** (PostgreSQL, Redis):
```bash
# From repository root
docker compose up -d db redis
```

**Stop Services**:
```bash
docker compose down
```

**View Logs**:
```bash
docker compose logs -f db    # PostgreSQL logs
docker compose logs -f redis # Redis logs
```

**Clean Everything** (removes volumes):
```bash
docker compose down -v
```

---

## Project Architecture

### Directory Structure
```
ModelForge/
├── .github/                  # GitHub configuration (no CI/CD workflows yet)
├── backend/                  # Python FastAPI backend
│   ├── app/
│   │   ├── api/             # API route handlers (health, models, predictions, jobs)
│   │   ├── crud/            # Database CRUD operations (~350 lines)
│   │   ├── models/          # SQLAlchemy ORM models (MLModel, Prediction, Job)
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── config.py        # Pydantic Settings (env var management)
│   │   ├── database.py      # Async SQLAlchemy engine & session
│   │   └── main.py          # FastAPI app entry point
│   ├── alembic/             # Database migrations
│   │   ├── env.py           # Alembic async configuration
│   │   └── versions/        # Migration scripts
│   ├── tests/               # Pytest test suite
│   │   ├── conftest.py      # Test fixtures (SQLite test DB, async client)
│   │   ├── test_health.py   # Health endpoint tests
│   │   ├── test_models.py   # Model CRUD tests
│   │   └── test_jobs.py     # Job management tests
│   ├── alembic.ini          # Alembic configuration
│   ├── pytest.ini           # Pytest configuration
│   ├── requirements.txt     # Python dependencies (FIX onnxruntime version!)
│   └── Dockerfile           # Container image definition
├── docker-compose.yml       # Local dev services (PostgreSQL, Redis)
├── README.md                # User-facing documentation
└── CLAUDE.md                # Phased build plan (Phase 1 complete)
```

### Key Configuration Files

**Environment Variables** (`backend/.env`):
- `DATABASE_URL`: PostgreSQL connection (default: `postgresql+asyncpg://postgres:postgres@localhost:5432/modelforge`)
- `REDIS_URL`: Redis connection (default: `redis://localhost:6379/0`)
- `ENVIRONMENT`: Runtime env (development/production)
- `SECRET_KEY`: App secret (change in production)
- `MODEL_STORAGE_PATH`: Model file storage location

**Pytest Configuration** (`backend/pytest.ini`):
- Async mode: auto
- Test paths: `tests/`
- File pattern: `test_*.py`
- Ignores deprecation warnings

**Alembic Configuration** (`backend/alembic.ini`):
- Script location: `alembic/`
- Database URL: Read from `app.config.settings.database_url` in `alembic/env.py`
- Async migrations supported

### API Structure

**Health Endpoints** (`/api/v1/`):
- `GET /health` - Basic health check
- `GET /ready` - Readiness probe (checks DB connection)
- `GET /live` - Liveness probe

**Models Endpoints** (`/api/v1/models`):
- `POST /` - Create model
- `GET /` - List models (with pagination)
- `GET /{id}` - Get model by ID
- `PATCH /{id}` - Update model
- `DELETE /{id}` - Delete model

**Predictions Endpoints** (`/api/v1/predictions`):
- `POST /models/{id}/predict` - Create prediction
- `GET /models/{id}/predictions` - List predictions for model

**Jobs Endpoints** (`/api/v1/jobs`):
- `POST /` - Create async job
- `GET /` - List jobs
- `GET /{id}` - Get job by ID
- `POST /{id}/cancel` - Cancel job

### Database Models

**MLModel** (`backend/app/models/ml_model.py`):
- Fields: id (UUID), name, version, status, file_path, input_schema, output_schema, **metadata** ⚠️
- Status enum: PENDING, VALIDATING, READY, ERROR, ARCHIVED
- Relationships: predictions (one-to-many), jobs (one-to-many)

**Prediction** (`backend/app/models/prediction.py`):
- Fields: id, model_id (FK), input_data, output_data, inference_time_ms, cached
- Used for storing synchronous prediction results

**Job** (`backend/app/models/job.py`):
- Fields: id, model_id (FK), status, priority, input_data, output_data, celery_task_id
- Status enum: PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED
- Priority enum: LOW, NORMAL, HIGH
- Used for async inference jobs (Phase 4, not fully implemented)

---

## Validation & Testing

### No CI/CD Pipeline Yet
**There are NO GitHub Actions workflows configured.** All validation must be done locally:

1. **Lint** (not configured): No linter setup (flake8, black, ruff, mypy)
2. **Format** (not configured): No formatter configured
3. **Type Check** (not configured): No mypy or type checking
4. **Tests**: Run `pytest -v --cov=app` manually
5. **Migrations**: Run `alembic upgrade head` to validate DB schema

### Manual Validation Checklist
Before committing changes:
- [ ] Fix SQLAlchemy `metadata` field conflict if modifying models
- [ ] Activate virtual environment
- [ ] Run `pytest -v` (all tests pass)
- [ ] Run `alembic upgrade head` (migrations apply cleanly)
- [ ] Start API server and check `http://localhost:8000/api/v1/docs`
- [ ] Test modified endpoints via Swagger UI or curl
- [ ] Check for Python import errors: `python -c "from app.main import app"`

---

## Common Pitfalls & Solutions

### Issue: "Module 'app' has no attribute X"
**Cause**: Import ordering or missing `__init__.py`  
**Solution**: Ensure all directories have `__init__.py` and check circular imports

### Issue: "connection refused" errors
**Cause**: PostgreSQL or Redis not running  
**Solution**: Run `docker compose up -d db redis` and wait 5-10 seconds

### Issue: "Alembic can't find models"
**Cause**: Models not imported in `alembic/env.py`  
**Solution**: Add imports: `from app.models import MLModel, Prediction, Job`

### Issue: "Database already exists" on fresh setup
**Cause**: Previous Docker volumes  
**Solution**: Run `docker compose down -v` to remove volumes

### Issue: "Test database file locked"
**Cause**: Previous test run didn't clean up  
**Solution**: Delete `backend/test.db` file manually

### Issue: Docker Compose version warning
**Warning**: `version` attribute is obsolete in `docker-compose.yml`  
**Impact**: Harmless warning, can be ignored or remove `version: "3.8"` line

---

## Development Workflow

### Making Code Changes
1. **Always activate virtual environment first**: `source venv/bin/activate`
2. **Make changes** to code
3. **Run tests**: `pytest -v` to catch regressions
4. **Test manually**: Start server and test via `/api/v1/docs`
5. **Create migration** if models changed: `alembic revision --autogenerate -m "..."`
6. **Commit changes**

### Adding New Models
1. Create model in `backend/app/models/`
2. Create schema in `backend/app/schemas/`
3. Create CRUD operations in `backend/app/crud/`
4. Import model in `backend/alembic/env.py`
5. Generate migration: `alembic revision --autogenerate -m "add new model"`
6. Review and apply: `alembic upgrade head`
7. Create API endpoints in `backend/app/api/`
8. Write tests in `backend/tests/`

### Adding Dependencies
1. Add to `requirements.txt`
2. **Check onnxruntime version compatibility** (use >=1.17.0)
3. Install: `pip install -r requirements.txt`
4. Test: Ensure application still starts

---

## Quick Reference

### File Locations
- **Main app**: `backend/app/main.py`
- **Config**: `backend/app/config.py`
- **Database setup**: `backend/app/database.py`
- **Models**: `backend/app/models/`
- **API routes**: `backend/app/api/`
- **Tests**: `backend/tests/`
- **Migrations**: `backend/alembic/versions/`

### Environment Variables
Defined in `backend/.env` (copy from `.env.example`):
```
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/modelforge
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=change-me-in-production
MODEL_STORAGE_PATH=./models
MAX_MODEL_SIZE_MB=500
```

### Port Mappings
- FastAPI: `8000`
- PostgreSQL: `5432`
- Redis: `6379`
- Next.js (Phase 5): `3000` (not implemented yet)

---

## Trust These Instructions

**These instructions have been validated by**:
1. Creating a fresh virtual environment
2. Installing dependencies (with onnxruntime fix)
3. Starting Docker services
4. Attempting migrations (fails due to known SQLAlchemy issue)
5. Reviewing all configuration files
6. Checking test infrastructure

**Only search for additional information if**:
- These instructions are incomplete for your specific task
- You encounter an error not documented here
- You need to understand implementation details beyond this reference

**For routine tasks** (adding endpoints, tests, CRUD operations), trust these instructions and proceed directly.
