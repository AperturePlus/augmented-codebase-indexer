# Issues Verification Report for Develop Branch

**Date:** 2026-03-21
**Branch:** develop (commit 9301964)
**Total Open Issues Checked:** 3

## Executive Summary

All 3 open issues (#14, #19, #20) have been **FIXED** in the develop branch and are ready for merge to master.

---

## Issue #14: Indexing always fails with Ollama/BERT-based embedding models due to tiktoken tokenizer mismatch

**Status:** ✅ FIXED
**Fixed by:** PR #17, PR #24
**Commits:** 61472a6, bafcf80

### Problem
ACI hardcoded the `cl100k_base` (OpenAI BPE) tokenizer, which caused token count mismatches with Ollama/BERT-based models (e.g., `nomic-embed-text`, `mxbai-embed-large`). This resulted in "input length exceeds context length" errors during indexing.

### Solution Implemented

1. **Added configurable tokenizer strategies** (`src/aci/core/tokenizer.py`):
   - `TiktokenTokenizer`: OpenAI BPE (default, accurate for OpenAI models)
   - `CharacterTokenizer`: Conservative `len(text)/4` estimator (works with any model)
   - `SimpleTokenizer`: Whitespace-based tokenizer (for generic non-BPE models)

2. **Added `ACI_TOKENIZER` environment variable** (`.env.example` lines 38-45):
   ```env
   # Use "character" or "simple" for Ollama/BERT-based models
   #   tiktoken  - OpenAI BPE (default, accurate for OpenAI models)
   #   character - len(text)/4 estimate (conservative, works with any model)
   #   simple    - whitespace split (for generic non-BPE models)
   ACI_TOKENIZER=tiktoken
   ```

3. **Fixed offline test failures**: Updated test fixtures to use `CharacterTokenizer()` instead of the default `tiktoken` to avoid network dependency on OpenAI's encoding vocabulary.

### Verification
- Code review: ✅ All three tokenizer classes implemented correctly
- Configuration: ✅ `ACI_TOKENIZER` documented in `.env.example`
- Tests: ✅ Test fixtures updated to work offline
- Integration: ✅ Tokenizer wired into service initialization via `IndexingConfig`

---

## Issue #19: BUG(CLI): .aci/index.db created in CWD instead of indexed project directory

**Status:** ✅ FIXED
**Fixed by:** PR #23
**Commit:** 80335dc

### Problem
When running `aci index /path/to/project` from a different directory, `.aci/index.db` was created in CWD instead of the target project directory. This caused `aci search` to fail with "Path has not been indexed" when run from the project directory.

### Solution Implemented

1. **Added `_project_metadata_db_path()` helper** (`src/aci/cli/__init__.py` lines 58-60):
   ```python
   def _project_metadata_db_path(path: Path) -> Path:
       """Return the metadata DB path scoped to a project root."""
       return path.resolve() / ".aci" / "index.db"
   ```

2. **Updated `get_services()` to accept `metadata_db_path`** (lines 63-84):
   - Accepts optional `metadata_db_path` parameter
   - Forwards it to `create_services()` for proper initialization

3. **Wired CLI commands to use project-scoped paths**:
   - `index` command (line 112): `get_services(metadata_db_path=_project_metadata_db_path(path))`
   - `search` command: Updated similarly when `--path` is specified
   - `update` command: Updated for consistency

### Verification
- Code review: ✅ Helper function correctly resolves to `<project>/.aci/index.db`
- CLI integration: ✅ All affected commands updated
- Tests: ✅ New test file `tests/unit/test_cli_metadata_db_path.py` with 3 tests:
  - `test_project_metadata_db_path_is_scoped_to_project_root`
  - `test_index_uses_project_scoped_metadata_db_path`
  - `test_search_uses_explicit_path_for_project_scoped_metadata_db_path`

---

## Issue #20: BUG: file_filter glob with relative path prefix returns no results

**Status:** ✅ FIXED
**Fixed by:** PR #22
**Commit:** 910c5a6

### Problem
The `file_filter` parameter silently returned zero results when using relative path prefixes (e.g., `apps/web/**/*.tsx`) because file paths are stored as absolute paths in the index, and relative globs never matched.

### Solution Implemented

1. **Added `resolve_file_filter_pattern()` function** (`src/aci/core/path_utils.py` lines 176-208):
   - Keeps wildcard-only patterns unchanged (e.g., `*.py`, `**/*.py`)
   - Keeps absolute patterns unchanged
   - Expands relative directory-prefixed patterns (e.g., `src/**/*.py`) to absolute patterns rooted at the indexed repository

2. **Applied normalization at all search entrypoints**:
   - MCP handler: `aci/mcp/handlers.py` line 5
   - HTTP server: `aci/http_server.py` line 9
   - CLI: `aci/cli/__init__.py` line 27

### Verification
- Code review: ✅ Pattern resolution logic correctly handles all cases
- Entry points: ✅ All three interfaces (MCP, HTTP, CLI) apply normalization
- Tests: ✅ New tests in `tests/unit/test_runtime_path_resolution.py` (lines 112-135):
  - `test_resolve_file_filter_pattern_keeps_wildcard_only_pattern`
  - `test_resolve_file_filter_pattern_expands_relative_prefixed_pattern`
  - `test_resolve_file_filter_pattern_keeps_absolute_pattern`

---

## Additional Improvements in Develop Branch

Beyond the three open issues, the develop branch includes:

1. **Graceful handling of oversized items** (PR #12, #16):
   - Skip oversized single items with zero-vector placeholder instead of aborting entire indexing
   - Improved embedding error classification for token-limit detection

2. **Better runtime path mapping** (multiple PRs):
   - Support for Windows/POSIX path mapping in containerized environments
   - `ACI_MCP_PATH_MAPPINGS` environment variable

3. **Test suite improvements**:
   - Fixed Windows system directory detection tests
   - Added property tests for embedding client fallback behavior
   - Improved offline test compatibility

---

## Recommendations

1. **Merge develop to master**: All critical bugs are fixed and well-tested
2. **Update documentation**: Consider adding a troubleshooting section for Ollama users
3. **Release notes**: Document the `ACI_TOKENIZER` environment variable for existing users

---

## Test Results Summary

All new tests added for fixes pass successfully:

- Issue #14: Tokenizer tests (offline-safe)
- Issue #19: 3/3 metadata DB path tests
- Issue #20: 3/3 file filter pattern resolution tests

Additional regression tests validate that fixes don't break existing functionality.

---

## Conclusion

The develop branch successfully addresses all 3 open issues:

- ✅ Issue #14: Tokenizer mismatch with Ollama - FIXED
- ✅ Issue #19: .aci/index.db creation path - FIXED
- ✅ Issue #20: file_filter relative paths - FIXED

All fixes include comprehensive tests and follow the repository's engineering standards (AGENTS.md). The branch is ready for merge to master.
