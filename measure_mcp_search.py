import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so we can import aci
sys.path.insert(0, os.path.abspath("."))

from aci.mcp.handlers import _handle_search_code, _handle_index_codebase

# Define QA Pairs
QA_PAIRS = [
    {
        "q": "How are grep search scores normalized against vector scores?",
        "expected_partial": "search_utils.py",
        "expected_term": "scale_factor",
        "desc": "Logic for normalizing grep vs vector scores"
    },
    {
        "q": "Why are summaries generated in the main process instead of worker processes?",
        "expected_partial": "indexing_service.py",
        "expected_term": "SummaryGenerator",
        "desc": "Architectural decision regarding process boundaries"
    },
    {
        "q": "What regex pattern is used to identify a function definition boundary in Python?",
        "expected_partial": "smart_splitter.py",
        "expected_term": "def\\s+",
        "desc": "Specific regex implementation detail"
    },
    {
        "q": "How does the system handle partial failures during batch indexing?",
        "expected_partial": "indexing_service.py",
        "expected_term": "pending_batch",
        "desc": "Data integrity mechanism"
    },
    {
        "q": "How does the indexing service avoid re-initializing Tree-sitter parsers in every worker task?",
        "expected_partial": "indexing_worker.py",
        "expected_term": "init_worker",
        "desc": "Performance optimization in multiprocessing"
    },
    {
        "q": "How can a user exclude specific paths from a search query?",
        "expected_partial": "search_utils.py",
        "expected_term": "exclude_patterns",
        "desc": "Search query syntax implementation"
    },
    {
        "q": "What context is added to the second chunk when a large method is split?",
        "expected_partial": "smart_splitter.py",
        "expected_term": "Context:",
        "desc": "Chunking context preservation logic"
    },
    {
        "q": "What design pattern is used for language-specific AST parsing?",
        "expected_partial": "ast_parser.py",
        "expected_term": "Strategy pattern",
        "desc": "Architectural pattern identification"
    },
    {
        "q": "How are duplicate results between vector and grep search handled?",
        "expected_partial": "search_utils.py",
        "expected_term": "deduplicate",
        "desc": "Search result merging logic"
    },
    {
        "q": "What happens if the embedding API returns fewer vectors than expected?",
        "expected_partial": "indexing_service.py",
        "expected_term": "IndexingError",
        "desc": "Error handling in external service calls"
    }
]

async def run_tests():
    # 1. Setup / Ensure Index
    target_path = str(Path("src/aci").resolve())
    print(f"Target Path: {target_path}")
    
    # Check if we should re-index (env var flag)
    if os.environ.get("REINDEX", "0") == "1":
        print("Forcing re-indexing...")
        idx_res = await _handle_index_codebase({"path": target_path})
        print(f"Index Result: {idx_res[0].text[:200]}...")
    else:
        print("Skipping re-indexing (set REINDEX=1 to force). Assuming index exists.")

    print("\n" + "="*80)
    print("STARTING COMPLEX SEARCH QUALITY TEST")
    print("="*80 + "\n")

    passed_file = 0
    passed_term = 0
    
    for i, item in enumerate(QA_PAIRS, 1):
        question = item["q"]
        print(f"Q{i}: {question}")
        print(f"    Target: {item['desc']}")
        
        # Call the MCP handler
        response_list = await _handle_search_code({
            "query": question,
            "path": target_path,
            "limit": 5,  # Increased limit for harder questions
            "mode": "hybrid"
        })
        
        # Parse output
        raw_text = response_list[0].text
        try:
            data = json.loads(raw_text)
            results = data.get("results", [])
            
            if not results:
                print("    ❌ NO RESULTS FOUND")
                print("-" * 60)
                continue
            
            # Check top 5 results
            found_file_idx = -1
            found_term_idx = -1
            
            for rank, res in enumerate(results, 1):
                file_path = res.get("file_path", "")
                content = res.get("content", "")
                
                # Check for file match
                if found_file_idx == -1 and item["expected_partial"] in file_path:
                    found_file_idx = rank
                
                # Check for term match (case-insensitive)
                if found_term_idx == -1 and item["expected_term"].lower() in content.lower():
                    found_term_idx = rank

            # Report File Finding
            if found_file_idx != -1:
                print(f"    ✅ File found at rank #{found_file_idx} ({item['expected_partial']})")
                passed_file += 1
            else:
                print(f"    ❌ File NOT found in top 5. Expected: {item['expected_partial']}")
                print(f"       Top result: {results[0].get('file_path')}")

            # Report Term Finding
            if found_term_idx != -1:
                print(f"    ✅ Term found at rank #{found_term_idx} ('{item['expected_term']}')")
                passed_term += 1
            else:
                print(f"    ⚠️  Term NOT found in top 5 snippets. Expected: '{item['expected_term']}'")

        except json.JSONDecodeError:
            print(f"    ❌ Error decoding JSON response")
            print(f"       Raw response: {raw_text[:200]}...")
        except Exception as e:
            print(f"    ❌ Exception: {e}")
            
        print("-" * 60)
        
    print(f"\nTEST SUMMARY:")
    print(f"Files Found: {passed_file}/{len(QA_PAIRS)}")
    print(f"Terms Found: {passed_term}/{len(QA_PAIRS)}")

if __name__ == "__main__":
    asyncio.run(run_tests())