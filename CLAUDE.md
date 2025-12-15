# ModelForge - ML Model Serving Platform

## Project Overview
ModelForge is a production-ready ML model serving platform with FastAPI backend, ONNX Runtime inference, Redis caching, PostgreSQL storage, async job processing, and a Next.js dashboard.

---

## Development Guidelines

### Communication Style
- **No sycophancy** - Be direct, honest, and critical when needed
- **Question requirements** - Push back on unclear or suboptimal approaches
- **Explain before implementing** - Describe architectural decisions and trade-offs first

### Code Quality
- **Point out code smells and technical debt** as you encounter them
- **Suggest better alternatives** when the current approach is suboptimal
- **Suggest refactors** for existing code - don't just add features and bloat
- **Use industry-standard patterns** and explain why they're standard

### Documentation & Clarity
- **Rubber-duck complex logic** - explain reasoning step by step
- **Explain inline**: regex, complex queries, gnarly algorithms - no cryptic code dumps
- **Flag dependencies** that are overkill or outdated

### Testing & Safety
- **Write tests that catch real bugs** - not just happy-path nonsense
- **Flag performance implications** proactively
- **Flag security concerns** proactively

---

## Tech Stack
- **Backend**: FastAPI (Python 3.11+)
- **ML Runtime**: ONNX Runtime
- **Database**: PostgreSQL
- **Caching**: Redis
- **Job Queue**: Celery with Redis broker
- **Frontend**: Next.js 14 (App Router)
- **Deployment**: Railway

## Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI API   │────▶│  PostgreSQL DB  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Redis   │ │  ONNX    │ │  Celery  │
              │  Cache   │ │  Runtime │ │  Worker  │
              └──────────┘ └──────────┘ └──────────┘
```

---

## Phased Build Plan

### Phase 1: Core Backend Foundation ✅ COMPLETE
**Goal**: Establish FastAPI backend with database models and basic API structure

**Deliverables**:
- [x] Project structure and configuration
- [x] FastAPI application setup with CORS, health checks
- [x] SQLAlchemy models (Model, Prediction, Job)
- [x] Pydantic schemas for API validation
- [x] Database connection and migrations (Alembic)
- [x] Basic CRUD endpoints for models
- [x] Docker Compose for local development
- [x] pytest setup with initial tests

**Key Files**:
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Settings management
│   ├── database.py          # DB connection
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── api/                 # Route handlers
│   └── crud/                # Database operations
├── alembic/                 # Migrations
├── tests/
├── requirements.txt
└── Dockerfile
```

---

### Phase 2: ONNX Runtime Integration
**Goal**: Add ML model upload, storage, and inference capabilities

#### PR 2.0: Fix Known Blocking Issues ⚠️ REQUIRED FIRST
- [ ] Fix `onnxruntime` version in requirements.txt (1.16.3 doesn't exist → use >=1.17.0)
- [ ] Rename `MLModel.metadata` field to `model_metadata` (SQLAlchemy reserved name conflict)
- [ ] Fix Pydantic `model_` namespace warnings in schemas
- [ ] Add Alembic migration for field rename
- [ ] Verify all tests pass

#### PR 2.1: Storage Service Foundation
- [ ] Create `backend/app/services/` directory structure
- [ ] Implement `StorageService` base class with interface
- [ ] Add local filesystem storage implementation
- [ ] Add storage configuration to `config.py`
- [ ] Unit tests for storage service

#### PR 2.2: Model File Upload Endpoint
- [ ] Add `/models/{id}/upload` POST endpoint
- [ ] Implement file validation (size limits, extension check)
- [ ] Store file via StorageService
- [ ] Update MLModel record with file_path, file_size, file_hash
- [ ] Integration tests for upload

#### PR 2.3: ONNX Validation Service
- [ ] Implement `ONNXService` for model operations
- [ ] Add ONNX model validation (load and verify)
- [ ] Extract input/output schema from ONNX model
- [ ] Extract model metadata (opset version, producer, etc.)
- [ ] Update MLModel status to READY or ERROR
- [ ] Unit tests with sample ONNX models

#### PR 2.4: Synchronous Inference Endpoint
- [ ] Add `/models/{id}/predict` POST endpoint
- [ ] Load ONNX model into runtime (with session caching)
- [ ] Validate input against model's input schema
- [ ] Run inference and return results
- [ ] Create Prediction record in database
- [ ] Integration tests for inference

#### PR 2.5: Model Versioning
- [ ] Add version uniqueness constraint (name + version)
- [ ] Implement version comparison logic
- [ ] Add `/models/{id}/versions` endpoint to list versions
- [ ] Add "latest" alias resolution
- [ ] Migration and tests

---

### Phase 3: Redis Caching Layer

#### PR 3.1: Redis Connection Manager
- [ ] Implement `CacheService` with Redis client
- [ ] Add connection pooling and health checks
- [ ] Add Redis configuration to `config.py`
- [ ] Graceful degradation when Redis unavailable
- [ ] Unit tests with Redis mocking

#### PR 3.2: Model Metadata Caching
- [ ] Cache model metadata on first load
- [ ] Add TTL configuration
- [ ] Invalidate cache on model update/delete
- [ ] Add cache headers to API responses

#### PR 3.3: Prediction Caching
- [ ] Generate cache keys from model ID + input hash
- [ ] Cache prediction results with configurable TTL
- [ ] Add cache hit/miss metrics endpoint
- [ ] Bypass cache option in predict request

---

### Phase 4: Async Job Queue

#### PR 4.1: Celery Setup
- [ ] Add Celery application configuration
- [ ] Create worker entry point
- [ ] Add Celery to Docker Compose
- [ ] Health check for worker status

#### PR 4.2: Async Inference Jobs
- [ ] Create `inference_task` Celery task
- [ ] Add `/jobs` POST endpoint to submit async inference
- [ ] Update Job model with task_id
- [ ] Return job ID immediately

#### PR 4.3: Job Status & Results
- [ ] Add `/jobs/{id}` GET endpoint for status
- [ ] Add `/jobs/{id}/result` GET endpoint
- [ ] Implement job polling with status transitions
- [ ] Store results in database

#### PR 4.4: Job Management
- [ ] Add `/jobs/{id}/cancel` POST endpoint
- [ ] Implement retry logic with exponential backoff
- [ ] Add job timeout handling
- [ ] Job cleanup/archival for old jobs

---

### Phase 5: Next.js Dashboard

#### PR 5.1: Project Setup
- [ ] Initialize Next.js 14 with App Router
- [ ] Configure TypeScript, ESLint, Tailwind
- [ ] Set up API client with fetch wrapper
- [ ] Add environment configuration

#### PR 5.2: Authentication
- [ ] Integrate NextAuth.js
- [ ] Add login/logout pages
- [ ] Protect routes with middleware
- [ ] Add user context provider

#### PR 5.3: Model Management UI
- [ ] Model list page with pagination
- [ ] Model detail page with metadata
- [ ] Model upload form with drag-and-drop
- [ ] Delete/archive model actions

#### PR 5.4: Prediction Interface
- [ ] Prediction form with JSON input
- [ ] Prediction history table
- [ ] Prediction detail view
- [ ] Export predictions as CSV

#### PR 5.5: Job Monitoring
- [ ] Job queue dashboard
- [ ] Real-time status updates (polling or SSE)
- [ ] Job cancellation UI
- [ ] Job logs/error display

---

### Phase 6: Testing & Documentation

#### PR 6.1: Test Coverage Expansion
- [ ] Add missing unit tests to reach 90%+ coverage
- [ ] Add edge case and error path tests
- [ ] Add load/stress tests for inference endpoint

#### PR 6.2: Integration Tests
- [ ] Full API workflow tests
- [ ] Database transaction tests
- [ ] Cache integration tests

#### PR 6.3: E2E Tests
- [ ] Set up Playwright
- [ ] Test critical user flows
- [ ] Add to CI pipeline

#### PR 6.4: Documentation
- [ ] Complete OpenAPI descriptions
- [ ] Write deployment guide
- [ ] Add architecture decision records

---

### Phase 7: Railway Deployment

#### PR 7.1: Railway Configuration
- [ ] Add `railway.toml` configuration
- [ ] Configure environment variables
- [ ] Set up PostgreSQL and Redis services

#### PR 7.2: CI/CD Pipeline
- [ ] GitHub Actions for test on PR
- [ ] Auto-deploy to Railway on main merge
- [ ] Add deployment status badges

#### PR 7.3: Observability
- [ ] Add structured logging
- [ ] Configure health monitoring
- [ ] Set up error alerting

---

## Development Commands

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "description"

# Tests
pytest -v --cov=app

# Docker
docker-compose up -d

# Frontend (Phase 5)
cd frontend
npm install
npm run dev
```

## Environment Variables

```env
# Backend
DATABASE_URL=postgresql://user:pass@localhost:5432/modelforge
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
ENVIRONMENT=development

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Current Status
- **Phase 1**: ✅ COMPLETE
- **Phase 2**: 🚧 NOT STARTED
- **Next PR**: PR 2.0 (Fix Known Blocking Issues)
- **Last Updated**: 2025-12-15

## Known Blocking Issues
These MUST be fixed before any Phase 2 work (see PR 2.0):

1. **`onnxruntime==1.16.3` does not exist** - pip install fails
   - Fix: Change to `onnxruntime>=1.17.0` in requirements.txt

2. **`MLModel.metadata` conflicts with SQLAlchemy** - reserved attribute name
   - Fix: Rename to `model_metadata`, add migration

3. **Pydantic `model_` namespace warning** - fields starting with `model_` trigger warnings
   - Fix: Use `model_config` to allow the namespace or rename fields

## Session Notes
- 2025-12-15: Merged Copilot onboarding PRs (#10, #6), closed duplicate (#9), removed blocking ruleset
- 2025-12-15: Assessed project state - Phase 2 ONNX work not actually merged, need to start fresh
- 2025-12-15: Broke down Phases 2-7 into 22 smaller PRs for easier review
- Next session: Start with PR 2.0 to fix blocking issues, then PR 2.1 for storage service
