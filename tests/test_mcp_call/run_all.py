"""
Run all MCP integration tests in order.

Usage: uv run python tests/test_mcp_call/run_all.py
"""

import subprocess
import sys


def run_test(script_name: str) -> bool:
    """Run a test script and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60 + "\n")
    
    result = subprocess.run(
        [sys.executable, f"tests/test_mcp_call/{script_name}"],
        cwd=".",
    )
    return result.returncode == 0


def main():
    tests = [
        ("test_index_codebase.py", "Index codebase (required first)"),
        ("test_stdio.py", "Update index and other operations"),
    ]
    
    results = []
    for script, description in tests:
        print(f"\n>>> {description}")
        success = run_test(script)
        results.append((script, success))
        
        if not success:
            print(f"\n✗ {script} failed!")
            # Continue anyway to see all results
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for script, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {script}")
    
    all_passed = all(s for _, s in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
