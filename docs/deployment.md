# ModelForge Deployment Guide

This guide covers deploying ModelForge to Railway, though the concepts apply to other PaaS providers.

## Prerequisites

- A [Railway](https://railway.app) account
- GitHub repository connected to Railway
- Basic understanding of environment variables and Docker

## Architecture Overview

ModelForge consists of the following services:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI API   │────▶│  PostgreSQL DB  │
│   (frontend)    │     │   (backend)     │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Redis   │ │  Celery  │ │  Model   │
              │  Cache   │ │  Worker  │ │  Storage │
              └──────────┘ └──────────┘ └──────────┘
```

## Railway Services Setup

### 1. Create a New Project

1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Connect your ModelForge repository

### 2. Add PostgreSQL

1. In your project, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway will automatically provision and configure the database
4. Note the `DATABASE_URL` - it will be available as an environment variable

### 3. Add Redis

1. Click "New" → "Database" → "Redis"
2. Railway will provision Redis and provide `REDIS_URL`

### 4. Configure Backend Service

Create a new service from your repo with these settings:

**Build Settings:**
- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Environment Variables:**
```env
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}
ENVIRONMENT=production
SECRET_KEY=<generate-a-secure-key>
CORS_ORIGINS=https://your-frontend-domain.railway.app
```

### 5. Configure Celery Worker

Create another service for the Celery worker:

**Build Settings:**
- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `celery -A app.worker worker --loglevel=info`

**Environment Variables:**
(Same as backend service)

### 6. Configure Frontend Service

**Build Settings:**
- Root Directory: `frontend`
- Build Command: `npm ci && npm run build`
- Start Command: `npm start`

**Environment Variables:**
```env
NEXT_PUBLIC_API_URL=https://your-backend-domain.railway.app
NEXTAUTH_URL=https://your-frontend-domain.railway.app
NEXTAUTH_SECRET=<generate-a-secure-key>
GITHUB_ID=<your-github-oauth-app-id>
GITHUB_SECRET=<your-github-oauth-app-secret>
```

## Environment Variables Reference

### Backend Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://user:pass@host:5432/db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `SECRET_KEY` | Application secret key | 32+ character random string |
| `ENVIRONMENT` | Environment name | `production`, `staging`, `development` |

### Backend Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |
| `MAX_MODEL_SIZE_BYTES` | `104857600` | Maximum model file size (100MB) |
| `CACHE_MODEL_TTL` | `3600` | Model cache TTL in seconds |
| `CACHE_PREDICTION_TTL` | `300` | Prediction cache TTL in seconds |
| `JOB_RETENTION_DAYS` | `7` | Days to keep completed jobs |

### Frontend Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://api.example.com` |
| `NEXTAUTH_URL` | Frontend URL for NextAuth | `https://example.com` |
| `NEXTAUTH_SECRET` | NextAuth encryption key | 32+ character random string |
| `GITHUB_ID` | GitHub OAuth App Client ID | From GitHub Developer Settings |
| `GITHUB_SECRET` | GitHub OAuth App Client Secret | From GitHub Developer Settings |

## Database Migrations

Run migrations after deployment:

```bash
# Using Railway CLI
railway run alembic upgrade head

# Or connect to the service and run manually
railway shell
cd backend
alembic upgrade head
```

## Health Checks

Configure Railway health checks to use these endpoints:

- **Liveness**: `GET /api/v1/live`
- **Readiness**: `GET /api/v1/ready`
- **Full Health**: `GET /api/v1/health`

## Scaling Considerations

### Horizontal Scaling

- **Backend API**: Stateless, can scale horizontally
- **Celery Workers**: Scale based on job queue depth
- **Frontend**: Stateless, can scale horizontally

### Vertical Scaling

- **PostgreSQL**: Scale based on connection count and data size
- **Redis**: Scale based on cache size and throughput

## Monitoring

### Recommended Metrics

1. **API Response Times**: P50, P95, P99 latencies
2. **Error Rates**: 4xx and 5xx responses
3. **Cache Hit Rate**: Monitor via `/api/v1/cache/metrics`
4. **Job Queue Depth**: Monitor pending/queued jobs
5. **Database Connections**: Active and idle connections

### Logging

All services output JSON-structured logs. Configure Railway's log drain to forward logs to your observability platform (DataDog, Grafana, etc.).

## Troubleshooting

### Common Issues

**Database Connection Errors**
- Verify `DATABASE_URL` format includes `+asyncpg`
- Check PostgreSQL service is running
- Verify network connectivity between services

**Redis Connection Errors**
- Application gracefully degrades without Redis
- Check `REDIS_URL` format
- Verify Redis service is running

**Model Upload Failures**
- Check `MAX_MODEL_SIZE_BYTES` setting
- Verify storage directory permissions
- Check available disk space

**Celery Worker Not Processing Jobs**
- Verify worker is connected to Redis broker
- Check `/api/v1/health/celery` endpoint
- Review worker logs for errors

## Security Checklist

- [ ] Generate unique `SECRET_KEY` for each environment
- [ ] Configure `CORS_ORIGINS` to only allow your frontend domain
- [ ] Set up HTTPS (Railway provides this automatically)
- [ ] Configure GitHub OAuth with correct callback URLs
- [ ] Review and restrict network access to databases
- [ ] Enable database connection encryption (SSL)
