# Performance Benchmarks

MLForge is designed for low-latency model serving. This document details performance characteristics and benchmarks.

## Single Prediction

| Scenario | Latency | Notes |
|----------|---------|-------|
| Cold (first request) | < 100ms | Includes model loading |
| Warm (model cached) | < 50ms | Model already in memory |
| Cache hit (Redis) | < 10ms | Prediction retrieved from cache |

### Factors Affecting Latency

1. **Model complexity**: Larger models with more operations take longer
2. **Input size**: Batch size directly impacts inference time
3. **Model loading**: First inference after startup includes model load time
4. **Network**: Redis cache latency depends on network to Redis server

## Throughput

| Batch Size | Throughput | Notes |
|------------|------------|-------|
| 1 | ~100-500 pred/sec | Single sample inference |
| 10 | ~500-1000 pred/sec | Small batch |
| 100 | ~1000-5000 pred/sec | Medium batch |
| 1000 | ~5000+ pred/sec | Large batch |

*Throughput varies significantly based on model complexity and hardware.*

## Caching Strategy

MLForge uses Redis for prediction caching:

```
Cache Key: pred:{model_id}:{input_hash}
TTL: 3600 seconds (configurable)
```

### Cache Behavior

1. **Cache Miss**: Full inference pipeline executes, result cached
2. **Cache Hit**: Result returned directly from Redis (< 10ms)
3. **Cache Invalidation**: Automatic on model update/deletion

### Cache Configuration

```bash
# Environment variables
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600
CACHE_ENABLED=true
```

## Optimization Tips

### 1. Batch Requests

Instead of multiple single predictions:
```python
# Slow - N network round trips
for sample in samples:
    predict(sample)

# Fast - 1 network round trip
predict(samples)
```

### 2. Model Optimization

- Use ONNX Runtime's graph optimizations
- Quantize models when precision allows
- Remove unused model outputs

### 3. Hardware Considerations

- CPU: More cores improve batch throughput
- Memory: Ensure models fit in RAM
- Redis: Low-latency connection to cache

## Benchmark Environment

Tests run in CI with the following baseline:

| Component | Specification |
|-----------|---------------|
| Python | 3.11+ |
| ONNX Runtime | Latest stable |
| Redis | 7.x |
| Database | PostgreSQL 15+ |

## Running Performance Tests

```bash
cd backend

# Run all performance tests
pytest tests/test_performance.py -v

# Run with timing details
pytest tests/test_performance.py -v --durations=0

# Run specific benchmark
pytest tests/test_performance.py::TestPerformanceBenchmarks::test_single_prediction_latency_under_100ms -v
```

## Metrics Collection

MLForge exposes the following performance metrics:

### Response Fields

```json
{
  "prediction_id": "uuid",
  "output_data": {...},
  "inference_time_ms": 12.5,
  "cached": false
}
```

### Health Endpoint

```bash
GET /health
```

Returns:
```json
{
  "status": "healthy",
  "database": "connected",
  "cache": {
    "status": "healthy",
    "connected": true
  }
}
```

## Performance Regression Testing

The test suite includes performance assertions:

| Test | Threshold | Purpose |
|------|-----------|---------|
| `test_single_prediction_latency_under_100ms` | < 100ms | Catch latency regressions |
| `test_batch_prediction_throughput` | > 100 pred/sec | Ensure minimum throughput |
| `test_cache_hit_latency_under_10ms` | < 50ms* | Verify cache effectiveness |

*Note: The 50ms threshold is for test environments without real Redis. Production cache hits are typically < 10ms.

## Profiling

For detailed performance analysis:

```bash
# Profile a prediction
python -m cProfile -o profile.stats -c "
import asyncio
from app.services.onnx import ONNXService
# ... profiling code
"

# Analyze results
python -m pstats profile.stats
```
