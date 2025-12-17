"""
Test MCP index_codebase on current codebase.

Run this BEFORE test_stdio.py to ensure the index exists for update_index.

Usage: uv run python tests/test_mcp_call/test_index_codebase.py
"""

import asyncio
import json
import os
import time

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


pytestmark = pytest.mark.integration

if os.environ.get("ACI_RUN_MCP_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "MCP stdio integration test disabled by default; set ACI_RUN_MCP_INTEGRATION_TESTS=1 to run.",
        allow_module_level=True,
    )


async def test_index_codebase():
    """Test index_codebase on current codebase via stdio."""
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "aci-mcp"],
        env=None,
    )
    
    codebase_path = os.getcwd()
    print(f"=== Testing index_codebase ===")
    print(f"Codebase path: {codebase_path}")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Server initialized")
            
            # Index the codebase
            print(f"\nIndexing {codebase_path}...")
            print("This may take a few minutes...")
            start_time = time.time()
            
            result = await session.call_tool("index_codebase", {
                "path": codebase_path,
                "workers": 4,
            })
            
            elapsed = time.time() - start_time
            print(f"\nCompleted in {elapsed:.2f}s")
            
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                        
                        # Verify success
                        if data.get("status") == "success":
                            print(f"\n✓ Indexed {data.get('total_files', 0)} files")
                            print(f"✓ Created {data.get('total_chunks', 0)} chunks")
                        else:
                            print(f"\n✗ Indexing failed or returned unexpected status")
                    except json.JSONDecodeError:
                        print(content.text)
            
            # Verify by checking status
            print("\n=== Verifying index status ===")
            result = await session.call_tool("get_index_status", {"path": codebase_path})
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        vectors = data.get("vector_store", {}).get("total_vectors", 0)
                        files = data.get("metadata", {}).get("total_files", 0)
                        print(f"Vector store: {vectors} vectors")
                        print(f"Metadata: {files} files indexed")
                        
                        if vectors > 0 and files > 0:
                            print("\n✓ Index verification passed!")
                        else:
                            print("\n✗ Index verification failed - no data found")
                    except json.JSONDecodeError:
                        print(content.text)


if __name__ == "__main__":
    asyncio.run(test_index_codebase())
