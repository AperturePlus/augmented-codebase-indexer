# Project ACI Code Diagnoses

- **Config-driven indexing/search settings ignored (Req 1.1, 2.5, 7.1, 7.2)**  
  - Evidence: `src/aci/cli/__init__.py:29-111` loads config but passes only embedding/vector settings; `IndexingService` is instantiated with default `FileScanner`/`Chunker`/`max_chunk_tokens`/ignore patterns instead of `config.indexing` (`src/aci/services/indexing_service.py:135-169`). `SearchService` is also built without the configured reranker or default limits.  
  - Impact: user-provided file extensions, ignore patterns, token window/overlap, and rerank toggles in YAML/env are silently ignored, breaking configurability promised in the requirements.

- **File-path filter not enforced in Qdrant search (Req 4.5 / Property 14)**  
  - Evidence: `src/aci/infrastructure/vector_store.py:244-307` leaves the filter construction as a `pass` placeholder and issues an unfiltered search, then client-side filters only the top `limit*5` results.  
  - Impact: queries scoped to a file/glob can return empty or partial results even when matching chunks exist but rank outside the small unfiltered window.

- **Embedding client lacks connection pooling (Req 6.5)**  
  - Evidence: each `_call_api` invocation spins up a fresh `httpx.AsyncClient` context (`src/aci/infrastructure/embedding_client.py:181`), with no reuse or shared session.  
  - Impact: no HTTP connection pooling; repeated batch calls incur extra TCP/TLS setup, violating the performance requirement.

- **Progress reporting omits ETA (Req 6.3)**  
  - Evidence: `_report_progress` only logs counts (`src/aci/services/indexing_service.py:171-175`), and no estimated completion time is calculated during indexing/update flows.  
  - Impact: CLI progress messages do not satisfy the requirement for ETA-aware reporting on large codebases.

- **Incremental change detection ignores modified timestamps (Req 5.4)**  
  - Evidence: `update_incremental` detects modifications solely by content hash (`src/aci/services/indexing_service.py:465-487`) despite the spec calling for both mtime and hash-based detection.  
  - Impact: deviates from the mandated strategy and cannot detect timestamp-only changes (e.g., touched files) per the acceptance criteria.

- **Evaluation service missing recall threshold alert (Req 9.5)**  
  - Evidence: `src/aci/services/evaluation_service.py:87-151` computes Recall@K/MRR but never flags Recall@10 < 0.7 for investigation.  
  - Impact: quality regressions below the required recall threshold go unreported.

- **Status command lacks health checks (Req 8.3)**  
  - Evidence: `src/aci/cli/__init__.py:162-185` prints cached metadata stats only; no vector store/embedding connectivity or health information is surfaced.  
  - Impact: the `status` CLI command does not meet the requirement to expose index health alongside statistics.
