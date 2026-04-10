"""
Go reference extractor.

Extracts symbol references (calls, imports, type annotations, inheritance/embedding)
from Go source code using tree-sitter, complementing the GoParser which extracts
symbol definitions.
"""

from typing import Any

from aci.core.parsers.base import SymbolReference
from aci.core.parsers.reference_extractor import ReferenceExtractorInterface


class GoReferenceExtractor(ReferenceExtractorInterface):
    """Extract symbol references from Go tree-sitter AST."""

    def extract_references(
        self, root_node: Any, content: str, file_path: str
    ) -> list[SymbolReference]:
        """Extract all non-import references: calls, type annotations, inheritance/embedding."""
        refs: list[SymbolReference] = []
        self._traverse(root_node, content, file_path, refs, scope_stack=[])
        return refs

    def extract_imports(
        self, root_node: Any, content: str, file_path: str
    ) -> list[SymbolReference]:
        """Extract import references from the AST."""
        refs: list[SymbolReference] = []
        self._traverse_imports(root_node, content, file_path, refs)
        return refs

    # ------------------------------------------------------------------
    # Internal traversal — references (calls, types, embedding)
    # ------------------------------------------------------------------

    def _traverse(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        scope_stack: list[str],
    ) -> None:
        """Recursively walk the AST collecting calls, type annotations, and embedding."""
        parent_symbol = ".".join(scope_stack) if scope_stack else None

        # --- function_declaration ---
        if node.type == "function_declaration":
            func_name = self._get_func_name(node, content)
            if func_name:
                self._extract_func_type_refs(
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

        # --- method_declaration (function with receiver) ---
        if node.type == "method_declaration":
            receiver_type = self._get_receiver_type(node, content)
            method_name = self._get_method_name(node, content)
            if method_name:
                scope_name = (
                    f"{receiver_type}.{method_name}" if receiver_type else method_name
                )
                self._extract_func_type_refs(
                    node, content, file_path, refs, parent_symbol
                )
                new_scope = [*scope_stack, scope_name]
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        # --- type_declaration (struct/interface embedding) ---
        if node.type == "type_declaration":
            self._extract_type_declaration_refs(
                node, content, file_path, refs, scope_stack
            )
            return

        # --- call_expression ---
        if node.type == "call_expression":
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
            # Recurse into argument_list for nested calls
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        self._traverse(arg, content, file_path, refs, scope_stack)
            return

        # --- type_assertion_expression: x.(Type) ---
        if node.type == "type_assertion_expression":
            type_name = self._extract_type_assertion_type(node, content)
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
            # Recurse into sub-expression
            for child in node.children:
                self._traverse(child, content, file_path, refs, scope_stack)
            return

        # Default: recurse into children
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
    ) -> None:
        """Walk the AST collecting only import references."""
        if node.type == "import_declaration":
            self._extract_import_declaration(node, content, file_path, refs)
            return

        for child in node.children:
            self._traverse_imports(child, content, file_path, refs)

    # ------------------------------------------------------------------
    # Name extraction helpers
    # ------------------------------------------------------------------

    def _get_func_name(self, node: Any, content: str) -> str | None:
        """Return the function name from a function_declaration node."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_content(child, content)
        return None

    def _get_method_name(self, node: Any, content: str) -> str | None:
        """Return the method name from a method_declaration node."""
        for child in node.children:
            if child.type == "field_identifier":
                return self.get_node_content(child, content)
        return None

    def _get_receiver_type(self, node: Any, content: str) -> str | None:
        """Extract the receiver type from a method_declaration's parameter_list."""
        found_receiver = False
        for child in node.children:
            if child.type == "parameter_list" and not found_receiver:
                found_receiver = True
                for param in child.children:
                    if param.type == "parameter_declaration":
                        for pc in param.children:
                            if pc.type == "type_identifier":
                                return self.get_node_content(pc, content)
                            if pc.type == "pointer_type":
                                for ptr_child in pc.children:
                                    if ptr_child.type == "type_identifier":
                                        return self.get_node_content(
                                            ptr_child, content
                                        )
        return None

    def _dotted_name(self, node: Any, content: str) -> str | None:
        """Resolve an identifier or selector expression to a dotted string."""
        if node.type == "identifier":
            return self.get_node_content(node, content)
        if node.type == "selector_expression":
            return self.get_node_content(node, content)
        return None

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    def _extract_call_name(self, node: Any, content: str) -> str | None:
        """Extract the callable name from a call_expression node.

        Handles:
        - ``foo()``            → ``"foo"``
        - ``pkg.Func()``      → ``"pkg.Func"``
        - ``obj.Method()``    → ``"obj.Method"``
        """
        if not node.children:
            return None
        func_node = node.children[0]
        return self._dotted_name(func_node, content)

    # ------------------------------------------------------------------
    # Type declaration refs (struct embedding, interface embedding)
    # ------------------------------------------------------------------

    def _extract_type_declaration_refs(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        scope_stack: list[str],
    ) -> None:
        """Extract embedding and type refs from a type_declaration node."""
        parent_symbol = ".".join(scope_stack) if scope_stack else None

        for child in node.children:
            if child.type == "type_spec":
                type_name = None
                for spec_child in child.children:
                    if spec_child.type == "type_identifier":
                        type_name = self.get_node_content(spec_child, content)

                    elif spec_child.type == "struct_type":
                        self._extract_struct_embedding(
                            spec_child,
                            content,
                            file_path,
                            refs,
                            parent_symbol,
                            type_name,
                        )

                    elif spec_child.type == "interface_type":
                        self._extract_interface_embedding(
                            spec_child,
                            content,
                            file_path,
                            refs,
                            parent_symbol,
                            type_name,
                        )

    def _extract_struct_embedding(
        self,
        struct_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
        type_name: str | None,
    ) -> None:
        """Extract embedded types from a struct_type node.

        Struct embedding: ``type Foo struct { Bar }`` — ``Bar`` is embedded.
        A field_declaration with only a type (no field name) is an embedding.
        """
        scope = type_name if type_name else parent_symbol
        for child in struct_node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        self._check_struct_field_embedding(
                            field, content, file_path, refs, scope
                        )

    def _check_struct_field_embedding(
        self,
        field_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Check if a field_declaration is an embedding (no field name, only type)."""
        has_field_name = False
        type_nodes: list[Any] = []

        for child in field_node.children:
            if child.type == "field_identifier":
                has_field_name = True
            elif child.type in ("type_identifier", "qualified_type"):
                type_nodes.append(child)
            elif child.type == "pointer_type":
                # *EmbeddedType
                for ptr_child in child.children:
                    if ptr_child.type in ("type_identifier", "qualified_type"):
                        type_nodes.append(ptr_child)

        # Embedding: field_declaration with type but no field name
        if not has_field_name and type_nodes:
            for tn in type_nodes:
                name = self.get_node_content(tn, content)
                if name:
                    line, _ = self.get_line_numbers(tn)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="inheritance",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
        elif has_field_name and type_nodes:
            # Regular field with a type — record as type_annotation
            for tn in type_nodes:
                name = self.get_node_content(tn, content)
                if name:
                    line, _ = self.get_line_numbers(tn)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="type_annotation",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )

    def _extract_interface_embedding(
        self,
        iface_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
        type_name: str | None,
    ) -> None:
        """Extract embedded interfaces from an interface_type node.

        Interface embedding: ``type Reader interface { io.Reader }``
        Look for type references that are not method signatures.
        """
        scope = type_name if type_name else parent_symbol
        for child in iface_node.children:
            # Embedded type in interface (type_identifier or qualified_type)
            if child.type in ("type_identifier", "qualified_type"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="inheritance",
                            file_path=file_path,
                            line=line,
                            parent_symbol=scope,
                        )
                    )
            # Also check inside constraint elements or method specs
            elif child.type == "constraint_elem":
                for cc in child.children:
                    if cc.type in ("type_identifier", "qualified_type"):
                        name = self.get_node_content(cc, content)
                        if name:
                            line, _ = self.get_line_numbers(cc)
                            refs.append(
                                SymbolReference(
                                    name=name,
                                    ref_type="inheritance",
                                    file_path=file_path,
                                    line=line,
                                    parent_symbol=scope,
                                )
                            )

    # ------------------------------------------------------------------
    # Function/method type references (params + return types)
    # ------------------------------------------------------------------

    def _extract_func_type_refs(
        self,
        func_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type references from function/method parameters and return types."""
        found_receiver = False
        for child in func_node.children:
            if child.type == "parameter_list":
                if func_node.type == "method_declaration" and not found_receiver:
                    # Skip the receiver parameter list
                    found_receiver = True
                    continue
                self._extract_param_type_refs(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "result":
                self._extract_result_type_refs(
                    child, content, file_path, refs, parent_symbol
                )

    def _extract_param_type_refs(
        self,
        params_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type references from parameter declarations."""
        for child in params_node.children:
            if child.type == "parameter_declaration":
                self._collect_type_refs(child, content, file_path, refs, parent_symbol)

    def _extract_result_type_refs(
        self,
        result_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type references from function return types."""
        self._collect_type_refs(result_node, content, file_path, refs, parent_symbol)

    def _collect_type_refs(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Recursively collect type_identifier and qualified_type nodes as type_annotation refs."""
        if node.type in ("type_identifier", "qualified_type"):
            name = self.get_node_content(node, content)
            if name:
                line, _ = self.get_line_numbers(node)
                refs.append(
                    SymbolReference(
                        name=name,
                        ref_type="type_annotation",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
            return
        for child in node.children:
            self._collect_type_refs(child, content, file_path, refs, parent_symbol)

    # ------------------------------------------------------------------
    # Type assertion extraction
    # ------------------------------------------------------------------

    def _extract_type_assertion_type(self, node: Any, content: str) -> str | None:
        """Extract the asserted type from a type_assertion_expression: x.(Type)."""
        for child in node.children:
            if child.type in ("type_identifier", "qualified_type"):
                return self.get_node_content(child, content)
        return None

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import_declaration(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
    ) -> None:
        """Extract import references from an import_declaration node.

        Handles:
        - Single: ``import "fmt"``
        - Grouped: ``import ("fmt"; "os")``
        - Aliased: ``import f "fmt"``
        """
        line, _ = self.get_line_numbers(node)

        for child in node.children:
            if child.type == "import_spec":
                self._extract_import_spec(child, content, file_path, refs, line)
            elif child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        spec_line, _ = self.get_line_numbers(spec)
                        self._extract_import_spec(
                            spec, content, file_path, refs, spec_line
                        )

    def _extract_import_spec(
        self,
        spec_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        line: int,
    ) -> None:
        """Extract the import path from an import_spec node."""
        for child in spec_node.children:
            if child.type == "interpreted_string_literal":
                raw = self.get_node_content(child, content)
                # Strip surrounding quotes
                path = raw.strip('"')
                if path:
                    refs.append(
                        SymbolReference(
                            name=path,
                            ref_type="import",
                            file_path=file_path,
                            line=line,
                            parent_symbol=None,
                        )
                    )
                return
