# ModelForge

[![CI](https://github.com/JohnRogers-Code-projects/MLForge/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnRogers-Code-projects/MLForge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Generic ONNX model serving platform. Upload any model, get predictions via REST API.

## Why ModelForge?

- **Fast** - ONNX Runtime for optimized inference + Redis caching for repeated predictions
- **Simple** - REST API with no ML framework lock-in; works with any ONNX model
- **Scalable** - Async job queue (Celery) for batch processing and long-running inference
- **Observable** - Health checks, metrics endpoints, and optional Sentry integration

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────────┐
│   Clients   │────▶│            FastAPI Application          │
└─────────────┘     │  ┌─────────┬──────────┬───────┬───────┐ │
                    │  │ Models  │Predictions│ Jobs  │ Cache │ │
                    │  └────┬────┴─────┬────┴───┬───┴───┬───┘ │
                    └───────┼──────────┼────────┼───────┼─────┘
                            │          │        │       │
                    ┌───────▼───┐ ┌────▼────┐ ┌─▼─┐ ┌───▼───┐
                    │ PostgreSQL│ │  ONNX   │ │   │ │ Redis │
                    │ (metadata)│ │ Runtime │ │ C │ │(cache)│
                    └───────────┘ └────┬────┘ │ e │ └───────┘
                                       │      │ l │
                                  ┌────▼────┐ │ e │
                                  │  File   │ │ r │
                                  │ Storage │ │ y │
                                  └─────────┘ └───┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Features

- **FastAPI Backend** - High-performance async API with automatic OpenAPI docs
- **ONNX Runtime** - Framework-agnostic inference (PyTorch, TensorFlow, scikit-learn)
- **PostgreSQL** - Model metadata, prediction history, and job tracking
- **Redis Caching** - Sub-millisecond cached predictions with configurable TTL
- **Celery Workers** - Background job processing with retry and timeout handling

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)

### Running with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Local Development

```bash
# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start PostgreSQL and Redis (via Docker)
docker-compose up -d db redis

# Run migrations
alembic upgrade head

# Start API server
uvicorn app.main:app --reload
```

### Running Tests

```bash
cd backend
pytest -v --cov=app
```

## API Examples

### Create a Model

```bash
curl -X POST http://localhost:8000/api/v1/models \
  -H "Content-Type: application/json" \
  -d '{"name": "my-classifier", "version": "1.0.0"}'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "my-classifier",
  "version": "1.0.0",
  "status": "pending"
}
```

### Upload and Validate Model

```bash
# Upload ONNX file
curl -X POST http://localhost:8000/api/v1/models/{model_id}/upload \
  -F "file=@model.onnx"

# Validate model (extracts schema)
curl -X POST http://localhost:8000/api/v1/models/{model_id}/validate
```

### Run Prediction

```bash
curl -X POST http://localhost:8000/api/v1/models/{model_id}/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"input": [[1.0, 2.0, 3.0, 4.0, 5.0]]}}'
```

Response:
```json
{
  "id": "prediction-uuid",
  "output_data": {"output": [[0.85]]},
  "inference_time_ms": 12.5,
  "cached": false
}
```

## API Documentation

Once running, access the interactive API docs:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## Project Structure

```
ModelForge/
├── backend/
│   ├── app/
│   │   ├── api/          # API route handlers
│   │   ├── crud/         # Database operations
│   │   ├── models/       # SQLAlchemy models
│   │   ├── schemas/      # Pydantic schemas
│   │   ├── services/     # Business logic (ONNX, cache, storage)
│   │   ├── tasks/        # Celery async tasks
│   │   ├── config.py     # Settings
│   │   └── main.py       # FastAPI app
│   ├── alembic/          # Database migrations
│   ├── tests/            # Pytest tests
│   └── Dockerfile
├── docs/
│   ├── ARCHITECTURE.md   # System architecture
│   └── deployment.md     # Deployment guide
├── docker-compose.yml
└── CLAUDE.md
```

## API Endpoints

### Health
- `GET /api/v1/health` - Health check with service status
- `GET /api/v1/ready` - Kubernetes readiness probe
- `GET /api/v1/live` - Kubernetes liveness probe
- `GET /api/v1/metrics` - Application metrics

### Models
- `POST /api/v1/models` - Create model metadata
- `GET /api/v1/models` - List all models
- `GET /api/v1/models/{model_id}` - Get model details
- `POST /api/v1/models/{model_id}/upload` - Upload ONNX file
- `POST /api/v1/models/{model_id}/validate` - Validate and activate model
- `DELETE /api/v1/models/{model_id}` - Delete model

### Predictions
- `POST /api/v1/models/{model_id}/predict` - Run synchronous prediction
- `GET /api/v1/models/{model_id}/predictions` - List prediction history

### Jobs (Async Inference)
- `POST /api/v1/jobs` - Create async inference job
- `GET /api/v1/jobs` - List jobs
- `GET /api/v1/jobs/{job_id}` - Get job status
- `GET /api/v1/jobs/{job_id}/result` - Poll for job result

### Cache
- `GET /api/v1/cache/metrics` - Cache hit/miss statistics

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `ENVIRONMENT` | Runtime environment | `development` |
| `SECRET_KEY` | Application secret | `change-me-in-production` |

## Deployment

ModelForge is configured for deployment to [Railway](https://railway.app).

### Railway Setup

1. Connect your GitHub repository to Railway
2. Railway will auto-deploy from the `main` branch
3. Configure the required environment variables (see `.env.railway.example`)
4. Add PostgreSQL and Redis services

For detailed deployment instructions, see [docs/deployment.md](docs/deployment.md).

## License

MIT
