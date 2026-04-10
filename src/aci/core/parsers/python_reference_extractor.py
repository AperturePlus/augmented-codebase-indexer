"""
Python reference extractor.

Extracts symbol references (calls, imports, type annotations, inheritance)
from Python source code using tree-sitter, complementing the PythonParser
which extracts symbol definitions.
"""

from typing import Any

from aci.core.parsers.base import SymbolReference
from aci.core.parsers.reference_extractor import ReferenceExtractorInterface


class PythonReferenceExtractor(ReferenceExtractorInterface):
    """Extract symbol references from Python tree-sitter AST."""

    def extract_references(
        self, root_node: Any, content: str, file_path: str
    ) -> list[SymbolReference]:
        """Extract all non-import references: calls, type annotations, inheritance."""
        refs: list[SymbolReference] = []
        self._traverse(root_node, content, file_path, refs, scope_stack=[])
        return refs

    def extract_imports(
        self, root_node: Any, content: str, file_path: str
    ) -> list[SymbolReference]:
        """Extract import references from the AST."""
        refs: list[SymbolReference] = []
        self._traverse_imports(root_node, content, file_path, refs, scope_stack=[])
        return refs

    # ------------------------------------------------------------------
    # Internal traversal — references (calls, types, inheritance)
    # ------------------------------------------------------------------

    def _traverse(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        scope_stack: list[str],
    ) -> None:
        """Recursively walk the AST collecting calls, type annotations, and inheritance."""
        parent_symbol = ".".join(scope_stack) if scope_stack else None

        if node.type == "class_definition":
            class_name = self._get_identifier(node, content)
            if class_name:
                self._extract_inheritance(node, content, file_path, refs, parent_symbol)
                new_scope = [*scope_stack, class_name]
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        if node.type == "function_definition":
            func_name = self._get_identifier(node, content)
            if func_name:
                self._extract_function_annotations(
                    node, content, file_path, refs, parent_symbol
                )
                new_scope = [*scope_stack, func_name]
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        if node.type == "call":
            name = self._extract_call_name(node, content)
            if name:
                line, _ = self.get_line_numbers(node)
                refs.append(
                    SymbolReference(
                        name=name,
                        ref_type="call",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
            # Recurse into arguments for nested calls
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        self._traverse(arg, content, file_path, refs, scope_stack)
            return

        # type nodes appear for annotations like `x: int`, `def f() -> str`
        if node.type == "type":
            type_name = self.get_node_content(node, content).strip()
            if type_name:
                line, _ = self.get_line_numbers(node)
                refs.append(
                    SymbolReference(
                        name=type_name,
                        ref_type="type_annotation",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
            return

        for child in node.children:
            self._traverse(child, content, file_path, refs, scope_stack)

    # ------------------------------------------------------------------
    # Internal traversal — imports only
    # ------------------------------------------------------------------

    def _traverse_imports(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        scope_stack: list[str],
    ) -> None:
        """Walk the AST collecting only import references."""
        parent_symbol = ".".join(scope_stack) if scope_stack else None

        if node.type == "import_statement":
            self._extract_import_statement(node, content, file_path, refs, parent_symbol)
            return

        if node.type == "import_from_statement":
            self._extract_import_from_statement(
                node, content, file_path, refs, parent_symbol
            )
            return

        # Track scope for parent_symbol on imports inside functions/classes
        if node.type == "class_definition":
            class_name = self._get_identifier(node, content)
            if class_name:
                new_scope = [*scope_stack, class_name]
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse_imports(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        if node.type == "function_definition":
            func_name = self._get_identifier(node, content)
            if func_name:
                new_scope = [*scope_stack, func_name]
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse_imports(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        for child in node.children:
            self._traverse_imports(child, content, file_path, refs, scope_stack)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _get_identifier(self, node: Any, content: str) -> str | None:
        """Return the first identifier child's text (class/function name)."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_content(child, content)
        return None

    def _extract_call_name(self, node: Any, content: str) -> str | None:
        """Extract the callable name from a ``call`` node.

        Handles:
        - ``foo()``            → ``"foo"``
        - ``obj.method()``     → ``"obj.method"``
        - ``a.b.c()``          → ``"a.b.c"``
        """
        if not node.children:
            return None
        func_node = node.children[0]
        return self._dotted_name(func_node, content)

    def _dotted_name(self, node: Any, content: str) -> str | None:
        """Resolve an identifier or attribute chain to a dotted string."""
        if node.type == "identifier":
            return self.get_node_content(node, content)
        if node.type == "attribute":
            return self.get_node_content(node, content)
        return None

    def _extract_inheritance(
        self,
        class_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract base classes from a ``class_definition`` node."""
        for child in class_node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    name = self._dotted_name(arg, content)
                    if name:
                        line, _ = self.get_line_numbers(arg)
                        refs.append(
                            SymbolReference(
                                name=name,
                                ref_type="inheritance",
                                file_path=file_path,
                                line=line,
                                parent_symbol=parent_symbol,
                            )
                        )

    def _extract_function_annotations(
        self,
        func_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract parameter type annotations and return type from a function."""
        for child in func_node.children:
            if child.type == "parameters":
                self._extract_param_annotations(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "type":
                # Return type annotation: `def f() -> str`
                type_name = self.get_node_content(child, content).strip()
                if type_name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=type_name,
                            ref_type="type_annotation",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )

    def _extract_param_annotations(
        self,
        params_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type annotations from function parameters."""
        for param in params_node.children:
            # typed_parameter, typed_default_parameter
            if param.type in ("typed_parameter", "typed_default_parameter"):
                for child in param.children:
                    if child.type == "type":
                        type_name = self.get_node_content(child, content).strip()
                        if type_name:
                            line, _ = self.get_line_numbers(child)
                            refs.append(
                                SymbolReference(
                                    name=type_name,
                                    ref_type="type_annotation",
                                    file_path=file_path,
                                    line=line,
                                    parent_symbol=parent_symbol,
                                )
                            )

    # ------------------------------------------------------------------
    # Import extraction helpers
    # ------------------------------------------------------------------

    def _extract_import_statement(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract references from ``import os`` / ``import os.path`` statements."""
        line, _ = self.get_line_numbers(node)
        for child in node.children:
            if child.type == "dotted_name":
                name = self.get_node_content(child, content)
                refs.append(
                    SymbolReference(
                        name=name,
                        ref_type="import",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
            elif child.type == "aliased_import":
                for sub in child.children:
                    if sub.type == "dotted_name":
                        name = self.get_node_content(sub, content)
                        refs.append(
                            SymbolReference(
                                name=name,
                                ref_type="import",
                                file_path=file_path,
                                line=line,
                                parent_symbol=parent_symbol,
                            )
                        )
                        break

    def _extract_import_from_statement(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract references from ``from X import Y`` statements.

        Emits one reference for the module and one for each imported name.
        """
        line, _ = self.get_line_numbers(node)
        module_name: str | None = None
        # Determine the module source: relative_import takes precedence,
        # otherwise the first dotted_name before the "import" keyword is the module.
        found_import_keyword = False
        first_dotted: str | None = None
        for child in node.children:
            if child.type == "relative_import":
                module_name = self.get_node_content(child, content)
            elif child.type == "dotted_name" and not found_import_keyword:
                first_dotted = self.get_node_content(child, content)
            elif child.type == "import":
                found_import_keyword = True

        if module_name is None:
            module_name = first_dotted

        # Emit the module reference
        if module_name:
            refs.append(
                SymbolReference(
                    name=module_name,
                    ref_type="import",
                    file_path=file_path,
                    line=line,
                    parent_symbol=parent_symbol,
                )
            )

        # Emit each imported name (children after the "import" keyword)
        after_import = False
        for child in node.children:
            if child.type == "import":
                after_import = True
                continue
            if not after_import:
                continue
            if child.type == "dotted_name":
                name = self.get_node_content(child, content)
                refs.append(
                    SymbolReference(
                        name=name,
                        ref_type="import",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
            elif child.type == "aliased_import":
                for sub in child.children:
                    if sub.type in ("dotted_name", "identifier"):
                        name = self.get_node_content(sub, content)
                        refs.append(
                            SymbolReference(
                                name=name,
                                ref_type="import",
                                file_path=file_path,
                                line=line,
                                parent_symbol=parent_symbol,
                            )
                        )
                        break
            elif child.type == "identifier":
                name = self.get_node_content(child, content)
                refs.append(
                    SymbolReference(
                        name=name,
                        ref_type="import",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
