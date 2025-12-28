# Claude Instructions — MLForge

You are assisting with changes to **MLForge**.

MLForge is **not** MCP-Demo.

It is intentionally more flexible, more operational, and more representative of real ML-adjacent backend systems.  
However, that flexibility must now be **explicit, bounded, and intentional**.

Your task is to help sharpen MLForge’s architectural intent without turning it into a framework or over-constraining it.

---

## High-Level Goal

MLForge should clearly demonstrate:

- Practical ML-oriented backend engineering
- Pipeline-style data and model flow
- Realistic operational tradeoffs
- Explicit architectural decisions (not accidental ones)

It must **complement** MCP-Demo, not imitate it.

---

## Non-Negotiable Principles

You must preserve the following:

1. MLForge remains **practical and realistic**
2. MLForge remains **flexible where experimentation is required**
3. MLForge does **not** become a constrained reference architecture
4. MLForge does **not** gain framework-like abstractions
5. MLForge does **not** hide complexity behind “magic” helpers

Flexibility is allowed — *but only when explicitly justified*.

---

## Architectural Corrections Required

The following weaknesses must be addressed **deliberately**, not incidentally.

---

### 1. Explicit Architectural Stance (PR 1)

MLForge must clearly state:

- What it optimizes for
- What it does *not* attempt to guarantee
- How it differs philosophically from MCP-Demo

This must be documented in the README and reflected in code structure.

You must **not** imply strong global guarantees that the system does not enforce.

---

### 2. Single Authoritative Pipeline Commitment (PR 2)

MLForge must introduce **one explicit commitment point** in the pipeline where:

- assumptions are locked in
- data shape is considered authoritative
- downstream stages may rely on invariants

This is *not* a global canonical context.

It is a **pipeline commitment boundary**.

Before this point:
- experimentation is allowed

After this point:
- assumptions must be enforced
- misuse must fail loudly

---

### 3. Deliberate Failure Mode (PR 3)

MLForge must include **one intentional failure mode** that demonstrates:

- misuse of the pipeline
- violation of assumptions
- invalid model or data artifact

The failure must be:
- explicit
- early
- loud
- non-recoverable

Do **not** add retries, fallbacks, or silent degradation.

This failure exists to demonstrate engineering judgment.

---

### 4. Tool / Model Invocation Responsibility (PR 4)

MLForge may retain flexible model invocation.

However:

- Responsibility for *why* a model is invoked must be explicit
- Tool/model interfaces must not silently accumulate policy
- “Manager” classes must justify their existence

If a component is making decisions, that must be obvious in code.

---

### 5. Testing Philosophy Alignment (PR 5)

MLForge tests must include:

- At least one negative test that proves:
    - misuse fails
    - invalid state cannot propagate

Do **not** attempt exhaustive coverage.

One strong, intentional negative test is sufficient.

---

## Hard Prohibitions

You must **not**:

- introduce plugin systems
- introduce registries
- add configuration layers “for flexibility”
- abstract for reuse across future domains
- unify MLForge and MCP-Demo concepts
- claim guarantees the system does not enforce
- add defensive recovery that hides failure

If something feels “cleaner” but less explicit, do not do it.

---

## PR Sequencing (Must Follow)

You must propose and implement changes in **this exact order**:

1. **PR 1 — Architectural Stance & README**
2. **PR 2 — Pipeline Commitment Boundary**
3. **PR 3 — Deliberate Failure Mode**
4. **PR 4 — Responsibility Clarification**
5. **PR 5 — Negative Test**

No PR may combine multiple goals.

Each PR must be:
- small
- reviewable
- conceptually singular

---

## Success Criteria

After all PRs:

- MLForge clearly communicates *why it is flexible*
- Authority and responsibility are explicit
- Failure is intentional, not accidental
- Reviewers can explain the architecture after one read
- MLForge complements MCP-Demo without copying it

If MLForge starts to feel like a framework or a demo of constraints, you have failed.

---

## Final Instruction

Do not optimize for elegance.

Optimize for **explicit tradeoffs, visible responsibility, and honest engineering**.

If a decision is pragmatic, say so in code and documentation.

Ambiguity is worse than imperfection.
