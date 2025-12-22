# ADR-003: Redis Caching with Graceful Degradation

## Status

Accepted

## Context

ML model serving benefits significantly from caching:
- Model metadata is frequently accessed and rarely changes
- Identical predictions should return cached results
- Reducing database queries improves response times

However, cache infrastructure can fail, and the system should remain operational.

Options considered:

1. **Redis only**: External cache, fast, feature-rich
2. **Memcached**: Simpler, multi-threaded
3. **In-process cache**: No network overhead, but not shared
4. **Database query cache**: Limited to DB queries
5. **Multi-tier caching**: Combine approaches

## Decision

We will use **Redis as our primary cache** with **graceful degradation** when Redis is unavailable.

### Rationale

1. **Performance**: Sub-millisecond latency for cached data
2. **Shared cache**: All API instances share the same cache
3. **Features**: TTL, atomic operations, pub/sub for invalidation
4. **Reuse**: Same Redis instance used for Celery broker
5. **Graceful degradation**: System works without cache (slower)

## Consequences

### Positive

- Fast response times for cached data
- Shared cache across API instances
- Automatic TTL-based expiration
- Cache metrics for observability
- System remains functional if Redis fails

### Negative

- Redis becomes important (but not critical) infrastructure
- Cache invalidation complexity
- Memory usage grows with cached data
- Network latency for cache operations

### Implementation Details

#### Cache Service Pattern

```python
class CacheService:
    async def get(self, key: str) -> Any | None:
        if not self._connected:
            return None  # Graceful degradation
        try:
            return await self._redis.get(key)
        except Exception:
            return None  # Fail open
```

#### Cache Key Strategy

```
model:{model_id}              # Model metadata
model:list:page:{n}           # Paginated model list
prediction:{model_id}:{hash}  # Prediction result
```

#### TTL Configuration

| Cache Type | Default TTL | Rationale |
|------------|-------------|-----------|
| Model metadata | 1 hour | Rarely changes |
| Model list | 5 minutes | May change frequently |
| Predictions | 5 minutes | Balance freshness/performance |

#### Cache Invalidation

Caches are invalidated on:
- Model update/delete
- Model file upload
- Model validation
- Manual cache clear via API

#### Cache Headers

API responses include cache headers:
```
X-Cache: HIT | MISS
Cache-Control: max-age={ttl}
```
