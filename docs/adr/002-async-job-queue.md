# ADR-002: Celery for Async Job Processing

## Status

Accepted

## Context

ML inference can be time-consuming, especially for:
- Large models with high latency
- Batch predictions processing many inputs
- Models with complex preprocessing

HTTP requests have timeouts (typically 30-60 seconds), making synchronous inference unsuitable for long-running predictions. We need an async processing mechanism.

Options considered:

1. **Celery with Redis**: Battle-tested Python task queue
2. **RQ (Redis Queue)**: Simpler Redis-based queue
3. **Dramatiq**: Modern alternative to Celery
4. **AWS SQS + Lambda**: Serverless approach
5. **Background threads**: In-process async handling

## Decision

We will use **Celery with Redis** as our async job queue.

### Rationale

1. **Maturity**: Celery is battle-tested with excellent documentation
2. **Redis reuse**: We already use Redis for caching
3. **Features**: Retries, rate limiting, task routing, monitoring
4. **Scaling**: Easy to scale workers horizontally
5. **Ecosystem**: Flower for monitoring, extensive integrations
6. **Python native**: Natural fit with FastAPI backend

## Consequences

### Positive

- Jobs can run for arbitrary duration
- Automatic retry with exponential backoff
- Worker scaling independent of API servers
- Job status tracking and result storage
- Task prioritization via queues
- Monitoring via Flower dashboard

### Negative

- Additional infrastructure (Celery workers)
- Complexity of distributed systems
- Redis becomes critical infrastructure
- Need to handle task serialization carefully

### Implementation Details

```python
# Task configuration
@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    max_retries=3,
    soft_time_limit=300,
    time_limit=330,
)
def run_inference_task(self, job_id: str):
    ...
```

### Job State Machine

```
PENDING → QUEUED → RUNNING → COMPLETED
                          ↘ FAILED
                          ↘ CANCELLED
```

### Monitoring

- Flower dashboard at `/flower` (development)
- `/api/v1/health/celery` endpoint for health checks
- Job metrics via API endpoints
