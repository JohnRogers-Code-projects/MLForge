# ModelForge

ML Model Serving Platform - Deploy and serve ONNX models at scale.

## Features

- **FastAPI Backend** - High-performance async API
- **ONNX Runtime** - Efficient ML model inference
- **PostgreSQL** - Reliable model metadata storage
- **Redis Caching** - Fast prediction caching
- **Async Job Queue** - Background inference processing
- **Next.js Dashboard** - Modern management UI

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend)

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
│   │   ├── config.py     # Settings
│   │   ├── database.py   # DB connection
│   │   └── main.py       # FastAPI app
│   ├── alembic/          # Database migrations
│   ├── tests/            # Pytest tests
│   └── Dockerfile
├── frontend/             # Next.js dashboard (Phase 5)
├── docker-compose.yml
└── CLAUDE.md             # Build plan
```

## API Endpoints

### Health
- `GET /api/v1/health` - Health check
- `GET /api/v1/ready` - Readiness probe
- `GET /api/v1/live` - Liveness probe

### Models
- `POST /api/v1/models` - Create model
- `GET /api/v1/models` - List models
- `GET /api/v1/models/{id}` - Get model
- `PATCH /api/v1/models/{id}` - Update model
- `DELETE /api/v1/models/{id}` - Delete model

### Predictions
- `POST /api/v1/predictions/models/{id}/predict` - Create prediction
- `GET /api/v1/predictions/models/{id}/predictions` - List predictions

### Jobs
- `POST /api/v1/jobs` - Create async job
- `GET /api/v1/jobs` - List jobs
- `GET /api/v1/jobs/{id}` - Get job
- `POST /api/v1/jobs/{id}/cancel` - Cancel job

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `ENVIRONMENT` | Runtime environment | `development` |
| `SECRET_KEY` | Application secret | `change-me-in-production` |

## License

MIT
