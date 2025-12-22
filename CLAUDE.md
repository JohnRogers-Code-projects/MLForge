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

#### PR 4.3: Job Results Endpoint ✅ COMPLETE
- [x] Add `GET /jobs/{id}/result` endpoint
- [x] Return result directly if job completed (200 with JobResultResponse)
- [x] Return 404 if job not found, 202 if still processing
- [x] Add optional `wait` query param with timeout (poll until complete, max 30s)
- [x] Expose `error_traceback` in JobResponse schema for failed jobs
- [x] Add JobResultResponse schema for result endpoint
- [x] Tests for result retrieval and waiting behavior (11 new tests, 239 total)

#### PR 4.4: Job Management & Cleanup ✅ COMPLETE
- [x] Update `POST /jobs/{id}/cancel` to revoke Celery task (also supports RUNNING jobs now)
- [x] Retry with exponential backoff already configured in PR 4.2 (verified)
- [x] Apply `soft_time_limit` and `time_limit` to inference task decorator
- [x] Create periodic cleanup task (`cleanup_old_jobs`) with Celery beat schedule
- [x] `job_retention_days` setting already in config (verified)
- [x] Add `DELETE /jobs/{id}` endpoint for manual deletion (terminal states only)
- [x] Tests for cancellation, deletion, and cleanup (14 new tests, 253 total)

---

### Phase 5: Next.js Dashboard

#### PR 5.1: Project Setup ✅ COMPLETE
- [x] Initialize Next.js 14 with App Router
- [x] Configure TypeScript, ESLint, Tailwind
- [x] Set up API client with fetch wrapper (`src/lib/api.ts`)
- [x] Add environment configuration (`.env.example`, `src/lib/config.ts`)
- [x] Add TypeScript types for API responses (`src/types/api.ts`)
- [x] Create placeholder home page with ModelForge branding

#### PR 5.2: Authentication ✅ COMPLETE
- [x] Integrate NextAuth.js with GitHub OAuth provider
- [x] Add login page with GitHub sign-in
- [x] Add auth-aware home page with sign-out
- [x] Protect routes with middleware (models, jobs, predictions, dashboard, settings)
- [x] Add SessionProvider to app layout
- [x] Type augmentations for NextAuth session
- [x] Configure Next.js for external avatar images

#### PR 5.3: Model Management UI ✅ COMPLETE
- [x] Model list page with pagination
- [x] Model detail page with metadata display
- [x] Model upload form with drag-and-drop
- [x] Delete/archive/validate model actions
- [x] Shared Header component with navigation
- [x] Model API service functions

#### PR 5.4: Prediction Interface ✅ COMPLETE
- [x] Prediction form with JSON input
- [x] Prediction history table
- [x] Prediction detail view
- [x] Export predictions as CSV

#### PR 5.5: Job Monitoring ✅ COMPLETE
- [x] Job queue dashboard
- [x] Real-time status updates (polling)
- [x] Job cancellation UI
- [x] Job logs/error display

---

### Phase 6: Testing & Documentation

#### PR 6.1: Test Coverage Expansion ✅ COMPLETE
- [x] Add missing unit tests (253 → 309 tests)
- [x] Add edge case and error path tests for CRUD operations
- [x] Coverage improved from 81% to 86%
- [x] Added tests for: health endpoints, job CRUD, model CRUD, prediction CRUD, cache service

#### PR 6.2: Integration Tests ✅ COMPLETE
- [x] Full API workflow tests (model lifecycle, async jobs)
- [x] Model versioning workflow tests
- [x] Cache integration tests (disabled cache behavior)
- [x] Data consistency tests (deletion cascade, status transitions)
- [x] Pagination integration tests
- [x] Error path integration tests
- [x] 20 new integration tests (309 → 329 tests)

#### PR 6.3: E2E Tests ✅ COMPLETE
- [x] Set up Playwright with chromium/firefox/webkit projects
- [x] Test critical user flows (16 E2E tests)
  - Login page accessibility and error handling
  - Protected route redirects (models, jobs, predictions)
  - Home page rendering and navigation
- [x] Add CI pipeline with GitHub Actions (.github/workflows/ci.yml)
  - Backend tests with PostgreSQL and Redis services
  - Frontend E2E tests with Playwright
  - ESLint checks

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
NEXTAUTH_SECRET=your-nextauth-secret
NEXTAUTH_URL=http://localhost:3000
GITHUB_ID=your-github-client-id
GITHUB_SECRET=your-github-client-secret
```

---

## Current Status
- **Phase 1**: ✅ COMPLETE
- **Phase 2**: ✅ COMPLETE
- **Phase 3**: ✅ COMPLETE
- **Phase 4**: ✅ COMPLETE
- **Phase 5**: ✅ COMPLETE
- **Phase 6**: In Progress (PR 6.1 ✅, PR 6.2 ✅, PR 6.3 ✅)
- **Next PR**: PR 6.4 (Documentation)
- **Last Updated**: 2025-12-22
- **Test Count**: 329 backend tests + 16 E2E tests
- **Coverage**: 86%

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
- 2025-12-19: PR 4.2 merged after 3 rounds of Copilot review (error handling, path traversal security, rollback fixes)
- 2025-12-19: Completed PR 4.3 - Job Results Endpoint (GET /jobs/{id}/result, wait parameter, error_traceback exposure)
- 2025-12-19: Completed PR 4.4 - Job Management & Cleanup (cancel revokes task, DELETE endpoint, periodic cleanup task)
- 2025-12-20: Started Phase 5, completed PR 5.1 - Next.js Dashboard Setup (Next.js 14, TypeScript, Tailwind, API client, typed API types matching backend schemas)
- 2025-12-20: Completed PR 5.2 - Authentication (NextAuth.js with GitHub OAuth, login page, route protection via middleware, SessionProvider, auth-aware home page)
- 2025-12-20: Completed PR 5.3 - Model Management UI (models list with pagination, detail page with metadata, drag-and-drop upload, delete/archive/validate actions)
- 2025-12-20: Merged PR #29 - Fixed type definitions for upload/validate responses, case-insensitive extension check
- 2025-12-20: Completed PR 5.4 - Prediction Interface (prediction form with JSON input/schema hints, prediction history table with pagination, detail modal, CSV export with security measures)
- 2025-12-20: Completed PR 5.5 - Job Monitoring (job queue dashboard, status filtering, 5-second polling for real-time updates, cancel/delete actions, detail modal with error tracebacks)
- 2025-12-20: Started Phase 6 - Created branch `feature/pr-6.1-test-coverage`
- 2025-12-22: Completed PR 6.1 - Test Coverage Expansion (56 new tests, 81% → 86% coverage, CRUD tests for all models)
- 2025-12-22: Completed PR 6.2 - Integration Tests (20 new tests covering full API workflows, versioning, error paths, pagination)
- 2025-12-22: Completed PR 6.3 - E2E Tests (Playwright setup, 16 E2E tests, GitHub Actions CI pipeline)

## Development Notes

**Environment Note**: Local Python is 3.14 which is too new for onnxruntime. Tests MUST be run via Docker:
```bash
# Start required services
docker-compose up -d db redis

# Run tests with coverage in Docker container
docker-compose run --rm -e DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/modelforge \
  -e REDIS_URL=redis://redis:6379/0 \
  api python -m pytest --cov=app --cov-report=term-missing -q
```

**Next Steps for PR 6.4 (Documentation)**:
1. Complete OpenAPI descriptions for all endpoints
2. Write deployment guide for Railway
3. Add architecture decision records (ADRs)
