"""
Property-based tests for MCP handler context parameter requirements.

**Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
**Validates: Requirements 1.1, 1.3, 3.3**

**Feature: mcp-dependency-injection, Property 4: Context Passed to Handlers**
**Validates: Requirements 2.3**
"""

import asyncio
import inspect
from typing import get_type_hints

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aci.mcp.context import MCPContext
from aci.mcp.handlers import _HANDLERS, call_tool


class TestHandlerContextParameter:
    """
    **Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
    **Validates: Requirements 1.1, 1.3, 3.3**

    For any MCP handler function registered in `_HANDLERS`, the function
    signature SHALL include an `MCPContext` parameter, and the handler
    SHALL NOT import or call `get_initialized_services`.
    """

    def test_all_handlers_have_mcp_context_parameter(self):
        """
        **Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
        **Validates: Requirements 1.1, 1.3, 3.3**

        For any registered handler, the function signature SHALL include
        an MCPContext parameter as the second argument.
        """
        assert len(_HANDLERS) > 0, "No handlers registered"

        for name, handler in _HANDLERS.items():
            sig = inspect.signature(handler)
            params = list(sig.parameters.keys())

            # Handler should have at least 2 parameters: arguments and ctx
            assert len(params) >= 2, (
                f"Handler '{name}' should have at least 2 parameters "
                f"(arguments, ctx), got {len(params)}: {params}"
            )

            # Second parameter should be named 'ctx'
            assert params[1] == "ctx", (
                f"Handler '{name}' second parameter should be 'ctx', "
                f"got '{params[1]}'"
            )


    def test_all_handlers_have_correct_ctx_type_annotation(self):
        """
        **Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
        **Validates: Requirements 1.1, 1.3, 3.3**

        For any registered handler, the ctx parameter SHALL be annotated
        with MCPContext type.
        """
        for name, handler in _HANDLERS.items():
            hints = get_type_hints(handler)

            assert "ctx" in hints, (
                f"Handler '{name}' should have type annotation for 'ctx' parameter"
            )

            assert hints["ctx"] is MCPContext, (
                f"Handler '{name}' ctx parameter should be typed as MCPContext, "
                f"got {hints['ctx']}"
            )

    def test_handlers_do_not_import_get_initialized_services(self):
        """
        **Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
        **Validates: Requirements 1.1, 1.3, 3.3**

        Handler module SHALL NOT import or use get_initialized_services.
        """
        import aci.mcp.handlers as handlers_module

        # Check module doesn't have get_initialized_services in its namespace
        assert not hasattr(handlers_module, "get_initialized_services"), (
            "handlers module should not import get_initialized_services"
        )

    def test_handlers_do_not_import_get_indexing_lock(self):
        """
        **Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
        **Validates: Requirements 1.1, 1.3, 3.3**

        Handler module SHALL NOT import or use get_indexing_lock.
        """
        import aci.mcp.handlers as handlers_module

        # Check module doesn't have get_indexing_lock in its namespace
        assert not hasattr(handlers_module, "get_indexing_lock"), (
            "handlers module should not import get_indexing_lock"
        )

    @given(handler_name=st.sampled_from(list(_HANDLERS.keys())))
    @settings(max_examples=100)
    def test_handler_signature_includes_context_property(self, handler_name: str):
        """
        **Feature: mcp-dependency-injection, Property 1: Handler Context Parameter**
        **Validates: Requirements 1.1, 1.3, 3.3**

        For any handler name in _HANDLERS, the corresponding function
        SHALL have MCPContext as its second parameter type.
        """
        handler = _HANDLERS[handler_name]
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())

        # Must have at least 2 parameters
        assert len(params) >= 2

        # Second parameter must be ctx with MCPContext type
        ctx_param = params[1]
        assert ctx_param.name == "ctx"

        hints = get_type_hints(handler)
        assert hints.get("ctx") is MCPContext


class TestContextPassedToHandlers:
    """
    **Feature: mcp-dependency-injection, Property 4: Context Passed to Handlers**
    **Validates: Requirements 2.3**

    For any tool call dispatched via `call_tool()`, the handler SHALL
    receive the MCPContext as its second argument.
    """

    def test_call_tool_signature_includes_context(self):
        """
        **Feature: mcp-dependency-injection, Property 4: Context Passed to Handlers**
        **Validates: Requirements 2.3**

        call_tool function SHALL accept MCPContext as its third parameter.
        """
        sig = inspect.signature(call_tool)
        params = list(sig.parameters.keys())

        assert "ctx" in params, "call_tool should have 'ctx' parameter"

        hints = get_type_hints(call_tool)
        assert hints.get("ctx") is MCPContext, (
            "call_tool ctx parameter should be typed as MCPContext"
        )

    @pytest.mark.asyncio
    async def test_call_tool_returns_error_when_context_is_none(self):
        """
        **Feature: mcp-dependency-injection, Property 4: Context Passed to Handlers**
        **Validates: Requirements 2.3**

        When ctx is None, call_tool SHALL return an error message.
        """
        result = await call_tool("index_codebase", {"path": "/tmp"}, None)

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "MCPContext not initialized" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_returns_error_for_unknown_tool(self):
        """
        **Feature: mcp-dependency-injection, Property 4: Context Passed to Handlers**
        **Validates: Requirements 2.3**

        When tool name is unknown, call_tool SHALL return an error message.
        """
        from aci.core.chunker import create_chunker
        from aci.core.config import ACIConfig
        from aci.core.file_scanner import FileScanner
        from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
        from aci.infrastructure.metadata_store import IndexMetadataStore
        from aci.services import IndexingService, SearchService

        # Create minimal context
        config = ACIConfig()
        embedding_client = LocalEmbeddingClient()
        vector_store = InMemoryVectorStore()
        metadata_store = IndexMetadataStore(":memory:")
        file_scanner = FileScanner(extensions={".py"})
        chunker = create_chunker()

        search_service = SearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            reranker=None,
            grep_searcher=None,
            default_limit=10,
        )

        indexing_service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            file_scanner=file_scanner,
            chunker=chunker,
            batch_size=32,
            max_workers=1,
        )

        ctx = MCPContext(
            config=config,
            search_service=search_service,
            indexing_service=indexing_service,
            metadata_store=metadata_store,
            vector_store=vector_store,
            indexing_lock=asyncio.Lock(),
        )

        result = await call_tool("nonexistent_tool", {}, ctx)

        assert len(result) == 1
        assert "Unknown tool" in result[0].text
        assert "nonexistent_tool" in result[0].text
