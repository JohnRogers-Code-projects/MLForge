# CLAUDE.md - MLForge

## Interaction Style

- Be direct and critical. No sycophancy.
- Point out flaws and inefficiencies directly
- Honest technical assessment over politeness

## Project Context

MLForge is a **generic ONNX model serving platform**. It serves any ONNX model uploaded to it—not tied to any specific domain or application.

**It is NOT specialized for MTG, ForgeBreaker, or any specific use case.**

Current consumers:
- ForgeBreaker uploads a deck win-rate predictor model
- Any other app could upload their own models

---

## CRITICAL: Work Item Rules

### 🚨 DO NOT BUILD MONOLITHIC PRs

**STOP. READ THIS BEFORE WRITING ANY CODE.**

Each work item = ONE PR. Do not combine work items. Do not "get ahead." 

**Before starting ANY work item, confirm with the user:**
> "I'm about to start Work Item X. This will create/modify these files: [list]. Should I proceed?"

**After completing ANY work item, STOP and say:**
> "Work Item X is complete. Tests pass. Ready for review before starting Work Item Y."

DO NOT automatically continue to the next work item.

### 🚨 LINT BEFORE EVERY COMMIT

**Run these commands before every commit, no exceptions:**

```bash
# Format code first
ruff format .

# Then check for lint errors
ruff check . --fix

# Verify no errors remain
ruff check .

# Only then run tests
pytest -v
```

**If ruff check fails, fix ALL errors before committing.** Do not commit code with lint errors. Do not defer lint fixes to "a later PR."

### 🚨 CI Must Pass At Every Commit

Every single commit must:
1. Pass `ruff format --check .`
2. Pass `ruff check .`
3. Pass `pytest`

If CI fails, fix it immediately in the same PR. Do not merge broken code.

---

## Linting Configuration

Ensure this exists in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by formatter)
]

[tool.ruff.format]
quote-style = "double"

[tool.ruff.lint.isort]
known-first-party = ["app"]
```

---

## Work Items (Do ONE at a time, get approval, then next)

### Work Item 1: Verify Existing Prediction Endpoint

**Goal:** Confirm the existing `/api/v1/models/{id}/predict` endpoint works for generic ONNX models.

**Files to check/modify:**
- `backend/app/api/predictions.py`
- `backend/tests/test_predictions.py`

**Test cases to verify exist:**
```python
test_predict_accepts_arbitrary_features
test_returns_prediction_array
test_validates_input_matches_model_schema
test_rejects_missing_features
test_caches_predictions_in_redis
test_handles_model_not_found
```

**Definition of Done:**
- [ ] `ruff format .` passes
- [ ] `ruff check .` passes
- [ ] `pytest -v` passes
- [ ] User has reviewed and approved

**STOP after this. Do not proceed to Work Item 2 without explicit approval.**

---

### Work Item 2: Add Model Upload Endpoint (if missing)

**Goal:** Ensure models can be uploaded via API.

**Files:**
- `backend/app/api/models.py`
- `backend/tests/test_models.py`

**Test cases:**
```python
test_upload_onnx_model
test_upload_rejects_invalid_file
test_upload_stores_metadata
test_list_models_returns_uploaded
```

**Definition of Done:**
- [ ] All lint checks pass
- [ ] All tests pass
- [ ] User has reviewed and approved

**STOP after this.**

---

### Work Item 3: Performance Tests for ONNX Inference

**Goal:** Measure and document model throughput and latency.

**Files:**
- `backend/tests/test_performance.py`
- `docs/PERFORMANCE.md`

**Test cases:**
```python
test_single_prediction_latency_under_100ms
test_batch_prediction_throughput
test_cache_hit_latency_under_10ms
```

**Documentation to create:**
```markdown
# Performance Benchmarks

## Single Prediction
- Cold: Xms (no cache)
- Warm: Xms (cached)

## Throughput
- X predictions/second (single model)

## Environment
- CPU: X
- Memory: X
- ONNX Runtime version: X
```

---

### Work Item 4: Mock Tests for Caching Logic

**Goal:** Verify Redis caching works correctly in isolation.

**Files:**
- `backend/tests/test_caching.py`

**Test cases:**
```python
test_prediction_cached_after_first_call
test_cache_key_includes_model_and_input
test_cache_expires_after_ttl
test_cache_miss_calls_model
test_cache_hit_skips_model
```

---

### Work Item 5: Architecture Documentation

**Goal:** Explain system design visually.

**File:** `docs/ARCHITECTURE.md`

**Required content:**
```markdown
# MLForge Architecture

## System Overview
[Mermaid diagram showing: API → Model Registry → ONNX Runtime]
                                    ↓
                              Redis Cache
                                    ↓
                              PostgreSQL (metadata)
                                    ↓
                              Job Queue (async)

## Components
- **API Layer**: FastAPI endpoints for model CRUD and predictions
- **Model Registry**: Stores ONNX files and metadata
- **Inference Engine**: ONNX Runtime for model execution
- **Cache Layer**: Redis for prediction caching
- **Job Queue**: Async processing for batch predictions

## Data Flow
1. Client uploads ONNX model → stored in registry
2. Client requests prediction → check cache → run inference → cache result
3. Async jobs for batch processing
```

---

### Work Item 6: README Improvements

**Goal:** Make README story-driven and visually clear.

**Additions needed:**
- [ ] "Why MLForge?" section explaining uniqueness
- [ ] Architecture diagram (inline Mermaid or image)
- [ ] Example curl commands with responses
- [ ] CI badges (build, coverage)
- [ ] Screenshots of dashboard (if frontend exists)

**README structure:**
```markdown
# MLForge

![Build](badge) ![Coverage](badge)

## What is MLForge?
Generic ONNX model serving platform. Upload any model, get predictions via API.

## Why MLForge?
- Fast: ONNX Runtime + Redis caching
- Simple: REST API, no ML framework lock-in
- Scalable: Async job queue for batch processing

## Architecture
[diagram]

## Quick Start
[commands]

## API Examples
[curl examples with responses]

## Performance
[link to PERFORMANCE.md]
```

---

### Work Item 7: CI Workflow

**Goal:** Enforce quality on every PR.

**File:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install ruff mypy
      - run: ruff format --check .
      - run: ruff check .
      - run: mypy backend/app --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=app --cov-report=xml --cov-fail-under=70
      - uses: codecov/codecov-action@v4
```

---

### Work Item 8: .env.example

**Goal:** Safe template for environment setup.

**File:** `.env.example`

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/mlforge

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=change-me-in-production

# Environment
ENVIRONMENT=development
```

---

## Commands (Run in this order, every time)

```bash
cd backend

# 1. Format
ruff format .

# 2. Lint (fix auto-fixable)
ruff check . --fix

# 3. Lint (verify clean)
ruff check .

# 4. Test
pytest -v

# 5. Only now commit
git add .
git commit -m "feat: description here"
```

---

## File Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── health.py
│   │   ├── models.py
│   │   └── predictions.py
│   ├── crud/
│   ├── models/        # SQLAlchemy
│   ├── schemas/       # Pydantic
│   ├── config.py
│   ├── database.py
│   └── main.py
├── alembic/
├── tests/
├── pyproject.toml     # Must have ruff config
└── Dockerfile
```

---

## Quality Checklist (Every PR)

- [ ] Only ONE work item per PR
- [ ] `ruff format .` passes
- [ ] `ruff check .` passes (zero errors)
- [ ] `pytest -v` passes
- [ ] No railway config changes
- [ ] No domain-specific code
- [ ] User approved before merge

---

## Anti-Patterns to Avoid

❌ "I'll implement Work Items 1-3 together for efficiency"
❌ "I'll fix the lint errors in a follow-up PR"  
❌ "The tests pass so the lint warnings are fine"
❌ "I'll just disable that lint rule"
❌ Committing without running ruff first

✅ One work item at a time
✅ Lint before every commit
✅ Fix all errors before committing
✅ Wait for approval between work items

---

## AI Assistance

Built with Claude as AI pair programmer. John Rogers provides direction and review.
