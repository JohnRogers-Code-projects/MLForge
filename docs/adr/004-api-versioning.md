# ADR-004: URL Path API Versioning

## Status

Accepted

## Context

APIs evolve over time, and breaking changes are sometimes necessary. We need a versioning strategy that:
- Allows introducing breaking changes safely
- Maintains backward compatibility for existing clients
- Is clear and intuitive for API consumers

Options considered:

1. **URL path versioning**: `/api/v1/models`
2. **Query parameter**: `/api/models?version=1`
3. **Header versioning**: `Accept: application/vnd.modelforge.v1+json`
4. **No versioning**: Break clients when needed

## Decision

We will use **URL path versioning** with the format `/api/v{major}/...`.

### Rationale

1. **Explicit**: Version is visible in every URL
2. **Simple**: No special headers or parameters needed
3. **Cacheable**: Different versions are different URLs
4. **Documentation**: Easy to document and discover
5. **Industry standard**: Used by most major APIs

### Version Format

```
/api/v1/models      # Version 1
/api/v2/models      # Version 2 (future)
```

## Consequences

### Positive

- Clear which version is being used
- Easy to test different versions
- Browser-friendly (works in URL bar)
- Cache keys naturally separated by version
- Simple routing configuration

### Negative

- URLs change between versions
- Slightly longer URLs
- Need to maintain multiple versions simultaneously

### Versioning Policy

1. **Major versions** (`v1`, `v2`): Breaking changes
   - Removing fields
   - Changing field types
   - Changing endpoint behavior

2. **Minor changes** (no version bump):
   - Adding new optional fields
   - Adding new endpoints
   - Deprecating (but not removing) features

### Deprecation Process

1. Announce deprecation in release notes
2. Add `Deprecation` header to responses
3. Maintain deprecated version for 6+ months
4. Remove deprecated version with major release

### Current Routes

```
/api/v1/
├── health              # Health checks
├── health/celery       # Celery health
├── live                # Liveness probe
├── ready               # Readiness probe
├── models/             # Model management
│   └── {model_id}/
│       ├── predict     # Synchronous inference
│       └── predictions # Prediction history
├── jobs/               # Async job queue
└── cache/              # Cache management
```

### Future Considerations

When `v2` is needed:
1. Create new router with `/api/v2` prefix
2. Implement new endpoints in parallel
3. Deprecate `v1` endpoints
4. Eventually sunset `v1`
