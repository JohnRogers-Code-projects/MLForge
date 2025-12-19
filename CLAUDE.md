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

#### PR 2.0: Fix Known Blocking Issues ✅ COMPLETE (PR #12)
- [x] Fix `onnxruntime` version in requirements.txt (>=1.17.0)
- [x] Rename `MLModel.metadata` field to `model_metadata`
- [x] Fix Pydantic `model_` namespace warnings in schemas
- [x] Add Alembic migration for field rename
- [x] All 35 tests pass

#### PR 2.1: Storage Service Foundation ✅ COMPLETE (PR #13)
- [x] Create `backend/app/services/` directory structure
- [x] Implement `StorageService` abstract base class
- [x] Add `LocalStorageService` with directory traversal protection
- [x] Add storage configuration to `config.py`
- [x] 19 unit tests for storage service

#### PR 2.2: Model File Upload Endpoint ✅ COMPLETE (PR #15)
- [x] Add `/models/{id}/upload` POST endpoint
- [x] File validation (.onnx extension, size limits)
- [x] Store file via StorageService with SHA-256 hash
- [x] Update MLModel record with file_path, file_size, file_hash
- [x] Added `UPLOADED` status to ModelStatus enum
- [x] Cleanup of stored file if DB update fails
- [x] 8 integration tests for upload (42 total tests)

#### PR 2.3: ONNX Validation Service ✅ COMPLETE
- [x] Implement `ONNXService` for model operations
- [x] Add ONNX model validation (load and verify)
- [x] Extract input/output schema from ONNX model
- [x] Extract model metadata (opset version, producer, etc.)
- [x] Update MLModel status to READY or ERROR
- [x] Unit tests with sample ONNX models
- [x] Add `/models/{id}/validate` POST endpoint
- [x] Integration tests for validation endpoint (69 total tests)

#### PR 2.4: Synchronous Inference Endpoint ✅ COMPLETE
- [x] Add `/models/{id}/predict` POST endpoint
- [x] Load ONNX model into runtime (with session caching)
- [x] Validate input against model's input schema
- [x] Run inference and return results
- [x] Create Prediction record in database
- [x] Integration tests for inference (22 new tests, 91 total tests)

#### PR 2.5: Model Versioning ✅ COMPLETE
- [x] Add version uniqueness constraint (name + version)
- [x] Implement version comparison logic (semantic versioning)
- [x] Add `/models/by-name/{name}/versions` endpoint to list versions
- [x] Add `/models/by-name/{name}/latest` endpoint with ready_only filter
- [x] Migration and tests (22 new tests, 113 total tests)

---

### Phase 3: Redis Caching Layer

#### PR 3.1: Redis Connection Manager ✅ COMPLETE
- [x] Implement `CacheService` with Redis client
- [x] Add connection pooling and health checks
- [x] Add Redis configuration to `config.py`
- [x] Graceful degradation when Redis unavailable
- [x] Unit tests with Redis mocking (29 new tests, 142 total)

#### PR 3.2: Model Metadata Caching ✅ COMPLETE
- [x] Cache model metadata on first load
- [x] Add TTL configuration (cache_model_ttl, cache_model_list_ttl)
- [x] Invalidate cache on model update/delete/upload/validate
- [x] Add cache headers to API responses (X-Cache, Cache-Control)
- [x] ModelCache helper class for cache operations
- [x] 17 new tests (159 total tests)

#### PR 3.3: Prediction Caching ✅ COMPLETE
- [x] Generate cache keys from model ID + input hash (MD5-based deterministic hashing)
- [x] Cache prediction results with configurable TTL (cache_prediction_ttl setting)
- [x] Add cache hit/miss metrics endpoint (/cache/metrics GET and POST reset)
- [x] Bypass cache option in predict request (skip_cache parameter)
- [x] Invalidate prediction cache on model upload/validate/delete
- [x] X-Cache response header (HIT/MISS)
- [x] 25 new tests (184 total tests)

---

### Phase 4: Async Job Queue

**Existing scaffolding**: Job model/schemas, JobCRUD, and basic endpoints (`POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`) already exist but are not wired to Celery. Jobs currently stay in PENDING forever.

#### PR 4.1: Celery Infrastructure Setup ✅ COMPLETE
- [x] Create `backend/app/celery.py` - Celery app instance with proper configuration
- [x] Add Celery settings to `config.py` (broker_url, result_backend, task serializer, time limits)
- [x] Create `backend/app/worker.py` - Worker entry point
- [x] Uncomment and configure worker service in docker-compose.yml
- [x] Add `/health/celery` endpoint to check worker connectivity
- [x] Add Flower service to docker-compose (task monitoring UI)
- [x] Unit tests for Celery configuration (23 new tests, 207 total)

#### PR 4.2: Async Inference Task ✅ COMPLETE
- [x] Create `backend/app/tasks/__init__.py`
- [x] Create `backend/app/tasks/inference.py` with `run_inference_task`
- [x] Task implementation: load model → run inference → store result
- [x] Update `POST /jobs` endpoint to queue task after creating job record
- [x] Implement status transitions: PENDING → QUEUED → RUNNING → COMPLETED/FAILED
- [x] Store `celery_task_id` and `worker_id` on job record
- [x] Track `queue_time_ms` and `inference_time_ms`
- [x] Integration tests with Celery eager mode (27 new tests, 228 total)

#### PR 4.3: Job Results Endpoint
- [ ] Add `GET /jobs/{id}/result` endpoint
- [ ] Return result directly if job completed
- [ ] Return 404 if job not found, 202 if still processing
- [ ] Add optional `wait` query param with timeout (poll until complete)
- [ ] Store `error_traceback` for failed jobs
- [ ] Tests for result retrieval and waiting behavior (~10 tests)

#### PR 4.4: Job Management & Cleanup
- [ ] Update `POST /jobs/{id}/cancel` to revoke Celery task
- [ ] Configure retry with exponential backoff (on transient failures)
- [ ] Add `task_soft_time_limit` and `task_time_limit` configuration
- [ ] Create periodic cleanup task for old completed/failed jobs
- [ ] Add `job_retention_days` setting (configurable retention period)
- [ ] Add `DELETE /jobs/{id}` endpoint for manual deletion
- [ ] Tests for cancellation, retries, and cleanup (~12 tests)

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
- **Phase 2**: ✅ COMPLETE
- **Phase 3**: ✅ COMPLETE
- **Next Phase**: Phase 4 (Async Job Queue)
- **Last Updated**: 2025-12-18
- **Test Count**: 228 tests passing

## Session Notes
- 2025-12-15: Merged Copilot onboarding PRs (#10, #6), closed duplicate (#9), removed blocking ruleset
- 2025-12-15: Assessed project state - Phase 2 ONNX work not actually merged, need to start fresh
- 2025-12-15: Broke down Phases 2-7 into 22 smaller PRs for easier review
- 2025-12-15: Completed PR 2.0 (blocking issues), PR 2.1 (storage service), PR 2.2 (upload endpoint)
- 2025-12-16: Completed PR 2.3 - ONNX Validation Service (ONNXService, validation endpoint, 27 new tests)
- 2025-12-16: Completed PR 2.4 - Synchronous Inference Endpoint (session caching, predict endpoint, 22 new tests)
- 2025-12-16: Completed PR 2.5 - Model Versioning (unique constraint, semver comparison, version endpoints)
- 2025-12-17: Completed PR 3.1 - Redis Connection Manager (CacheService, health checks, graceful degradation)
- 2025-12-17: Completed PR 3.2 - Model Metadata Caching (ModelCache helper, cache invalidation, cache headers)
- 2025-12-17: Completed PR 3.3 - Prediction Caching (PredictionCache, input hashing, metrics endpoint, cache invalidation)
- 2025-12-17: Starting Phase 4 - Refined PR breakdown; Job model/CRUD/endpoints already scaffolded, need Celery wiring
- 2025-12-17: Completed PR 4.1 - Celery Infrastructure Setup (celery app, worker, flower, health checks)
- 2025-12-18: Completed PR 4.2 - Async Inference Task (run_inference_task, status transitions, timing metrics)
