# ModelForge - ML Model Serving Platform

## Project Overview
ModelForge is a production-ready ML model serving platform with FastAPI backend, ONNX Runtime inference, Redis caching, PostgreSQL storage, async job processing, and a Next.js dashboard.

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

### Phase 1: Core Backend Foundation ✅ CURRENT
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

**Deliverables**:
- [ ] Model file upload endpoint (S3-compatible or local)
- [ ] ONNX model validation and metadata extraction
- [ ] Model versioning system
- [ ] Synchronous inference endpoint
- [ ] Input/output schema validation per model
- [ ] Model warmup and health checks

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
- **Active Phase**: Phase 1
- **Last Updated**: 2025-12-14
