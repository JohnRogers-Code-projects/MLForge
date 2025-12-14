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

### Phase 2: ONNX Runtime Integration ✅ COMPLETE
**Goal**: Add ML model upload, storage, and inference capabilities

**Deliverables**:
- [x] Model file upload endpoint (local filesystem storage)
- [x] ONNX model validation and metadata extraction
- [x] Model versioning system (name + version unique constraint)
- [x] Synchronous inference endpoint
- [x] Input/output schema validation per model
- [x] Model warmup and health checks

**Key Files Added**:
```
backend/app/services/
├── __init__.py
├── onnx_service.py      # ONNX Runtime wrapper (validation, inference, caching)
└── storage_service.py   # Model file storage (local filesystem)
```

**New API Endpoints**:
- `POST /api/v1/models/upload` - Upload and validate ONNX model
- `POST /api/v1/predictions/models/{id}/predict` - Run synchronous inference
- `GET /api/v1/models/{id}/schema` - Get model input/output schema
- `POST /api/v1/models/{id}/warmup` - Warm up model for fast inference

---

### Phase 3: Redis Caching Layer
**Goal**: Implement caching for predictions and model metadata

**Deliverables**:
- [ ] Redis connection manager
- [ ] Prediction caching with TTL
- [ ] Model metadata caching
- [ ] Cache invalidation strategies
- [ ] Cache hit/miss metrics

---

### Phase 4: Async Job Queue
**Goal**: Handle long-running inference jobs asynchronously

**Deliverables**:
- [ ] Celery worker setup
- [ ] Async inference job submission
- [ ] Job status tracking and polling
- [ ] Job result retrieval
- [ ] Job cancellation
- [ ] Retry logic with exponential backoff

---

### Phase 5: Next.js Dashboard
**Goal**: Build management UI for models and predictions

**Deliverables**:
- [ ] Next.js 14 project setup
- [ ] Authentication (NextAuth.js)
- [ ] Model management pages (list, detail, upload)
- [ ] Prediction history and analytics
- [ ] Job queue monitoring
- [ ] Real-time status updates (WebSocket/SSE)

---

### Phase 6: Testing & Documentation
**Goal**: Comprehensive test coverage and documentation

**Deliverables**:
- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests
- [ ] E2E tests (Playwright)
- [ ] API documentation (OpenAPI/Swagger)
- [ ] User guide and README
- [ ] Architecture decision records

---

### Phase 7: Railway Deployment
**Goal**: Production deployment with CI/CD

**Deliverables**:
- [ ] Railway project configuration
- [ ] Environment variable management
- [ ] PostgreSQL and Redis provisioning
- [ ] GitHub Actions CI/CD pipeline
- [ ] Health monitoring and alerts
- [ ] Logging and observability

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
- **Phase 1**: COMPLETE
- **Phase 2**: COMPLETE
- **Next Phase**: Phase 3 (Redis Caching Layer)
- **Last Updated**: 2025-12-14

## Session Notes
- Phase 2 ONNX Runtime integration is complete
- Services: `ONNXService` for model validation/inference, `StorageService` for file handling
- Model upload flow: upload → validate ONNX → extract schema → store → warmup → ready
- Inference: validates model status, runs ONNX inference, stores prediction record
- Tests added for ONNX service, storage service, and API endpoints
- Next session: Start with Phase 3 - Redis caching for predictions
