"""
MCP Client Test Script

用于测试 ACI MCP 服务器的工具调用。
可以直接运行: uv run python tests/test_mcp_call/test_mcp_client.py
"""

import asyncio
import json
from pathlib import Path

import pytest

# Integration-style script; skip during normal automated test runs.
pytestmark = pytest.mark.skip(reason="Manual MCP client exercise; requires running MCP server")

# 直接导入 MCP handlers 进行测试（绕过 stdio）
from aci.mcp.tools import list_tools
from aci.mcp.handlers import call_tool


async def test_list_tools():
    """测试列出所有可用工具"""
    print("=" * 60)
    print("测试: list_tools")
    print("=" * 60)
    
    tools = list_tools()
    print(f"可用工具数量: {len(tools)}\n")
    
    for tool in tools:
        print(f"📦 {tool.name}")
        print(f"   描述: {tool.description[:80]}...")
        required = tool.inputSchema.get("required", [])
        print(f"   必需参数: {required}")
        print()
    
    return tools


async def test_get_status(path: str | None = None):
    """测试获取索引状态"""
    print("=" * 60)
    print(f"测试: get_index_status (path={path})")
    print("=" * 60)
    
    args = {"path": path} if path else {}
    result = await call_tool("get_index_status", args)
    
    print("结果:")
    for item in result:
        if hasattr(item, "text"):
            print(item.text)
    print()
    return result


async def test_list_repos():
    """测试列出已索引的仓库"""
    print("=" * 60)
    print("测试: list_indexed_repos")
    print("=" * 60)
    
    result = await call_tool("list_indexed_repos", {})
    
    print("结果:")
    for item in result:
        if hasattr(item, "text"):
            print(item.text)
    print()
    return result


async def test_index(path: str):
    """测试索引目录"""
    print("=" * 60)
    print(f"测试: index_codebase (path={path})")
    print("=" * 60)
    
    result = await call_tool("index_codebase", {"path": path})
    
    print("结果:")
    for item in result:
        if hasattr(item, "text"):
            print(item.text)
    print()
    return result


async def test_search(query: str, path: str, limit: int = 5, mode: str = "hybrid"):
    """测试搜索"""
    print("=" * 60)
    print(f"测试: search_code")
    print(f"  query: {query}")
    print(f"  path: {path}")
    print(f"  mode: {mode}")
    print("=" * 60)
    
    result = await call_tool("search_code", {
        "query": query,
        "path": path,
        "limit": limit,
        "mode": mode,
    })
    
    print("结果:")
    for item in result:
        if hasattr(item, "text"):
            # 尝试解析 JSON 格式化输出
            try:
                data = json.loads(item.text)
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(item.text)
    print()
    return result


async def test_update(path: str):
    """测试增量更新"""
    print("=" * 60)
    print(f"测试: update_index (path={path})")
    print("=" * 60)
    
    result = await call_tool("update_index", {"path": path})
    
    print("结果:")
    for item in result:
        if hasattr(item, "text"):
            print(item.text)
    print()
    return result


async def run_all_tests():
    """运行完整测试流程"""
    # 使用当前项目目录作为测试路径
    test_path = str(Path.cwd())
    
    print("\n" + "🚀 开始 MCP 工具测试 ".center(60, "=") + "\n")
    
    # 1. 列出工具
    await test_list_tools()
    
    # 2. 列出已索引仓库
    await test_list_repos()
    
    # 3. 获取状态
    await test_get_status()
    
    # 4. 索引当前目录（如果需要）
    # await test_index(test_path)
    
    # 5. 搜索测试
    # await test_search("embedding client", test_path, limit=3)
    
    print("\n" + " 测试完成 ".center(60, "=") + "\n")


async def interactive_test():
    """交互式测试"""
    test_path = str(Path.cwd())
    
    while True:
        print("\n选择测试:")
        print("1. 列出工具")
        print("2. 列出已索引仓库")
        print("3. 获取索引状态")
        print("4. 索引目录")
        print("5. 搜索代码")
        print("6. 增量更新")
        print("0. 退出")
        
        choice = input("\n请选择 (0-6): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            await test_list_tools()
        elif choice == "2":
            await test_list_repos()
        elif choice == "3":
            path = input("输入路径 (留空获取全局状态): ").strip() or None
            await test_get_status(path)
        elif choice == "4":
            path = input(f"输入路径 (默认: {test_path}): ").strip() or test_path
            await test_index(path)
        elif choice == "5":
            query = input("输入搜索查询: ").strip()
            if not query:
                print("查询不能为空")
                continue
            path = input(f"输入路径 (默认: {test_path}): ").strip() or test_path
            mode = input("搜索模式 (hybrid/vector/grep, 默认: hybrid): ").strip() or "hybrid"
            await test_search(query, path, limit=5, mode=mode)
        elif choice == "6":
            path = input(f"输入路径 (默认: {test_path}): ").strip() or test_path
            await test_update(path)
        else:
            print("无效选择")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        # 交互模式
        asyncio.run(interactive_test())
    else:
        # 运行所有测试
        asyncio.run(run_all_tests())
