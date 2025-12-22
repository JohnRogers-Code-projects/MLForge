# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for ModelForge.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences.

## ADR Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-001](001-onnx-runtime.md) | Use ONNX Runtime for ML Inference | Accepted |
| [ADR-002](002-async-job-queue.md) | Celery for Async Job Processing | Accepted |
| [ADR-003](003-caching-strategy.md) | Redis Caching with Graceful Degradation | Accepted |
| [ADR-004](004-api-versioning.md) | URL Path API Versioning | Accepted |

## Template

When creating a new ADR, use this template:

```markdown
# ADR-XXX: Title

## Status

Proposed | Accepted | Deprecated | Superseded

## Context

What is the issue we're addressing?

## Decision

What is the change we're proposing?

## Consequences

What are the results of this decision?
```
