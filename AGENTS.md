# AGENTS.md - Engineering and Agent Governance for ACI

This file defines mandatory engineering rules for this repository.

Applies to:
- Human contributors
- AI coding agents

## 1. Purpose and Scope

This document is the single governance contract for:
- Requirements
- Design
- Programming
- Testing
- CI/CD
- Git workflow
- AI safety constraints
- AI commit discipline

If a change conflicts with this document, this document takes precedence unless a maintainer explicitly overrides it in the task context.

## 2. Roles and Audience

- Humans MUST follow these standards when planning, coding, testing, and reviewing.
- AI agents MUST follow these standards when proposing, editing, validating, and reporting changes.

## 3. Requirement Engineering Standards

- Every non-trivial task MUST define:
  - Objective
  - In-scope
  - Out-of-scope
  - Acceptance criteria
- Bugfixes MUST include:
  - Reproducible symptom
  - Expected behavior
- If acceptance criteria are undefined, implementation MUST NOT start.

## 4. Design Standards

- Non-trivial changes MUST include a brief design rationale in PR/task notes.
- Design rationale MUST state:
  - Selected approach
  - Alternatives considered
  - Tradeoffs
- Layering MUST be respected:
  - `core`: domain logic and cross-cutting primitives
  - `infrastructure`: external systems and adapters
  - `services`: orchestration and business workflows
  - `cli` / `http` / `mcp`: entrypoint adapters only

## 5. Programming Standards

- New code MUST include type annotations.
- Error handling and logs MUST preserve root causes and actionable context.
- Security-sensitive data MUST NOT be hardcoded or logged.
- Changes SHOULD minimize diff surface; avoid opportunistic refactors.
- Existing repository conventions MUST be preserved unless the task explicitly requests a redesign.

## 6. Testing Standards

- Logic changes MUST include tests appropriate to the change:
  - Unit tests
  - Property tests
  - Integration tests
- Flaky or non-deterministic tests MUST NOT be merged.
- Minimum local validation before PR:
  - `uv run ruff check src tests`
  - `uv run pytest tests/ -v --tb=short -q --durations=10`
- Type checking SHOULD be run and reviewed:
  - `uv run mypy src --ignore-missing-imports --no-error-summary`

## 7. CI/CD Standards

- CI source of truth:
  - `.github/workflows/test.yml`
  - `.github/workflows/release.yml`
- Required CI commands are:
  - `uv run ruff check src tests`
  - `uv run pytest tests/ -v --tb=short -q --durations=10`
- Current mypy execution in CI is informational/non-blocking:
  - `uv run mypy src --ignore-missing-imports --no-error-summary || true`
- Release tags MUST follow `v*`.
- A release is marked prerelease when tag name contains `-`.
- PRs MUST NOT merge with failing required checks.

## 8. Git Workflow and PR Standards

- Workflow MUST be Trunk-based development + PR.
- Commit messages MUST follow Conventional Commits with scope:
  - `type(scope): short imperative summary`
- Allowed types:
  - `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`, `perf`
- Branch naming SHOULD use:
  - `feat/<topic>`
  - `fix/<topic>`
  - `chore/<topic>`
  - `docs/<topic>`
- Every PR MUST include:
  - What changed
  - Why
  - Test evidence (commands + outcomes)
  - Risk/rollback notes for non-trivial changes

## 9. AI Safety and Dangerous-Command Policy

### 9.1 Strict Blocking Rules

AI MUST NOT run destructive or high-risk commands.

Default prohibited patterns:
- `rm -rf`
- `del /s /q`
- `format`
- `mkfs`
- `dd`
- shutdown/reboot commands
- recursive permission/ownership changes outside task scope
- `git reset --hard`
- `git clean -fdx`
- `git checkout -- .`
- force push to protected branches

### 9.2 Additional Safety Constraints

- AI MUST NOT modify files outside repository task scope without explicit user instruction.
- AI MUST explicitly call out risk before any potentially destructive operation.
- AI MUST prefer reversible operations.
- AI MUST preserve unrelated local changes.

## 10. AI Commit Discipline and Commit Message Standard

### 10.1 Milestone-Based Commit Discipline

- AI MUST commit at each verifiable milestone, not only at final completion.
- A milestone is:
  - A complete logical subtask
  - With relevant checks passing
  - With coherent rollback boundaries
- AI MUST avoid oversized mixed-purpose commits.
- AI SHOULD keep one intent per commit.

### 10.2 AI Commit Message Template

Header (required):
- `type(scope): short imperative summary`

Body (required):
- `Why:` context/problem
- `What:` key changes
- `Test:` commands executed and outcomes

Optional footer:
- `BREAKING CHANGE:` when applicable

Example:
- `fix(indexing): retry qdrant upsert on transient timeout`

## 11. Definition of Done (DoD) Checklist

Before marking work complete, ALL applicable items MUST be satisfied:

- Requirements are explicit and testable
- Design rationale is captured for non-trivial changes
- Code is scoped and consistent with project architecture
- Tests are added/updated and passing
- Lint checks are passing
- CI/CD impact is considered
- Documentation is updated when behavior changes
- Commit/PR metadata follows policy

