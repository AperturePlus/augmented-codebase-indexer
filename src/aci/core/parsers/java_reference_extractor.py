"""
Java reference extractor.

Extracts symbol references (calls, imports, type annotations, inheritance)
from Java source code using tree-sitter, complementing the JavaParser which
extracts symbol definitions.
"""

from typing import Any

from aci.core.parsers.base import SymbolReference
from aci.core.parsers.reference_extractor import ReferenceExtractorInterface


class JavaReferenceExtractor(ReferenceExtractorInterface):
    """Extract symbol references from Java tree-sitter AST."""

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
        self._traverse_imports(root_node, content, file_path, refs)
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

        # --- class_declaration / interface_declaration / enum_declaration ---
        if node.type in (
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ):
            class_name = self._get_identifier(node, content)
            if class_name:
                self._extract_inheritance(node, content, file_path, refs, parent_symbol)
                new_scope = [*scope_stack, class_name]
                for child in node.children:
                    if child.type in ("class_body", "interface_body", "enum_body"):
                        for body_child in child.children:
                            self._traverse(
                                body_child, content, file_path, refs, new_scope
                            )
            return

        # --- method_declaration / constructor_declaration ---
        if node.type in ("method_declaration", "constructor_declaration"):
            method_name = self._get_identifier(node, content)
            if method_name:
                self._extract_method_type_refs(
                    node, content, file_path, refs, parent_symbol
                )
                new_scope = [*scope_stack, method_name]
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        # --- method_invocation: foo(), obj.method(), ClassName.staticMethod() ---
        if node.type == "method_invocation":
            name = self._extract_method_invocation_name(node, content)
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

        # --- object_creation_expression: new ClassName() ---
        if node.type == "object_creation_expression":
            name = self._extract_object_creation_name(node, content)
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

        # --- field_declaration: private List<String> items; ---
        if node.type == "field_declaration":
            self._extract_field_type_ref(node, content, file_path, refs, parent_symbol)
            # Recurse for initializer expressions (may contain calls)
            for child in node.children:
                if child.type == "variable_declarator":
                    for vc in child.children:
                        self._traverse(vc, content, file_path, refs, scope_stack)
            return

        # --- local_variable_declaration: String name = "hello"; ---
        if node.type == "local_variable_declaration":
            self._extract_local_var_type_ref(
                node, content, file_path, refs, parent_symbol
            )
            # Recurse for initializer expressions
            for child in node.children:
                if child.type == "variable_declarator":
                    for vc in child.children:
                        self._traverse(vc, content, file_path, refs, scope_stack)
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
            self._extract_import(node, content, file_path, refs)
            return

        for child in node.children:
            self._traverse_imports(child, content, file_path, refs)

    # ------------------------------------------------------------------
    # Name extraction helpers
    # ------------------------------------------------------------------

    def _get_identifier(self, node: Any, content: str) -> str | None:
        """Return the first identifier child's text (class/method name)."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_content(child, content)
        return None

    def _dotted_name(self, node: Any, content: str) -> str | None:
        """Resolve an identifier or field_access chain to a dotted string."""
        if node.type == "identifier":
            return self.get_node_content(node, content)
        if node.type == "field_access":
            return self.get_node_content(node, content)
        if node.type == "scoped_identifier":
            return self.get_node_content(node, content)
        return None

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    def _extract_method_invocation_name(self, node: Any, content: str) -> str | None:
        """Extract the callable name from a ``method_invocation`` node.

        Handles:
        - ``foo()``                    → ``"foo"``
        - ``obj.method()``             → ``"obj.method"``
        - ``ClassName.staticMethod()`` → ``"ClassName.staticMethod"``
        """
        # tree-sitter-java method_invocation children:
        # [object.]name(argument_list)
        # The name is an identifier; the object (if present) precedes it.
        obj_part: str | None = None
        method_name: str | None = None

        for child in node.children:
            if child.type == "identifier":
                method_name = self.get_node_content(child, content)
            elif child.type in ("field_access", "scoped_identifier"):
                obj_part = self.get_node_content(child, content)
            elif child.type == "argument_list":
                break  # stop before args

        if method_name and obj_part:
            return f"{obj_part}.{method_name}"
        return method_name

    def _extract_object_creation_name(self, node: Any, content: str) -> str | None:
        """Extract the class name from an ``object_creation_expression`` node.

        Handles ``new ClassName()`` and ``new pkg.ClassName()``.
        """
        for child in node.children:
            if child.type == "type_identifier":
                return self.get_node_content(child, content)
            if child.type == "scoped_type_identifier":
                return self.get_node_content(child, content)
            if child.type == "generic_type":
                # new ArrayList<String>() — extract the outer type
                for gc in child.children:
                    if gc.type in ("type_identifier", "scoped_type_identifier"):
                        return self.get_node_content(gc, content)
        return None

    # ------------------------------------------------------------------
    # Inheritance extraction
    # ------------------------------------------------------------------

    def _extract_inheritance(
        self,
        class_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract extends/implements from a class/interface declaration.

        Handles:
        - ``class Foo extends Bar``
        - ``class Foo implements Baz, Qux``
        - ``interface Foo extends Bar``
        """
        for child in class_node.children:
            # superclass node: extends clause for classes
            if child.type == "superclass":
                self._collect_type_names_as_inheritance(
                    child, content, file_path, refs, parent_symbol
                )
            # super_interfaces node: implements clause for classes
            elif child.type == "super_interfaces":
                self._collect_type_names_as_inheritance(
                    child, content, file_path, refs, parent_symbol
                )
            # extends_interfaces node: extends clause for interfaces
            elif child.type == "extends_interfaces":
                self._collect_type_names_as_inheritance(
                    child, content, file_path, refs, parent_symbol
                )

    def _collect_type_names_as_inheritance(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Recursively collect type_identifier nodes as inheritance refs."""
        if node.type in ("type_identifier", "scoped_type_identifier"):
            name = self.get_node_content(node, content)
            if name:
                line, _ = self.get_line_numbers(node)
                refs.append(
                    SymbolReference(
                        name=name,
                        ref_type="inheritance",
                        file_path=file_path,
                        line=line,
                        parent_symbol=parent_symbol,
                    )
                )
            return
        if node.type == "generic_type":
            # e.g. Comparable<Foo> — extract the outer type
            for gc in node.children:
                if gc.type in ("type_identifier", "scoped_type_identifier"):
                    name = self.get_node_content(gc, content)
                    if name:
                        line, _ = self.get_line_numbers(gc)
                        refs.append(
                            SymbolReference(
                                name=name,
                                ref_type="inheritance",
                                file_path=file_path,
                                line=line,
                                parent_symbol=parent_symbol,
                            )
                        )
                    return
            return
        for child in node.children:
            self._collect_type_names_as_inheritance(
                child, content, file_path, refs, parent_symbol
            )

    # ------------------------------------------------------------------
    # Method type references (params + return type)
    # ------------------------------------------------------------------

    def _extract_method_type_refs(
        self,
        method_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type references from method parameters and return type."""
        for child in method_node.children:
            # Return type — appears as a type_identifier or generic_type before the method name
            if child.type in ("type_identifier", "scoped_type_identifier"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="type_annotation",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
            elif child.type == "generic_type":
                self._extract_generic_type_ref(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "formal_parameters":
                self._extract_param_type_refs(
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
        """Extract type references from formal_parameters."""
        for child in params_node.children:
            if child.type == "formal_parameter":
                self._extract_formal_param_type(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "spread_parameter":
                self._extract_formal_param_type(
                    child, content, file_path, refs, parent_symbol
                )

    def _extract_formal_param_type(
        self,
        param_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the type from a single formal_parameter or spread_parameter."""
        for child in param_node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="type_annotation",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
            elif child.type == "generic_type":
                self._extract_generic_type_ref(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "array_type":
                self._extract_array_type_ref(
                    child, content, file_path, refs, parent_symbol
                )

    # ------------------------------------------------------------------
    # Field and local variable type references
    # ------------------------------------------------------------------

    def _extract_field_type_ref(
        self,
        field_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the type from a field_declaration."""
        for child in field_node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="type_annotation",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
            elif child.type == "generic_type":
                self._extract_generic_type_ref(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "array_type":
                self._extract_array_type_ref(
                    child, content, file_path, refs, parent_symbol
                )

    def _extract_local_var_type_ref(
        self,
        var_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the type from a local_variable_declaration."""
        for child in var_node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="type_annotation",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
            elif child.type == "generic_type":
                self._extract_generic_type_ref(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "array_type":
                self._extract_array_type_ref(
                    child, content, file_path, refs, parent_symbol
                )

    # ------------------------------------------------------------------
    # Generic and array type helpers
    # ------------------------------------------------------------------

    def _extract_generic_type_ref(
        self,
        generic_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the outer type name from a generic_type node (e.g. List<String>)."""
        for child in generic_node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
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

    def _extract_array_type_ref(
        self,
        array_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the element type from an array_type node (e.g. String[])."""
        for child in array_node.children:
            if child.type in ("type_identifier", "scoped_type_identifier"):
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
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
            if child.type == "generic_type":
                self._extract_generic_type_ref(
                    child, content, file_path, refs, parent_symbol
                )
                return

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_import(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
    ) -> None:
        """Extract import references from an import_declaration node.

        Handles:
        - ``import java.util.List;``
        - ``import static java.lang.Math.PI;``
        - ``import java.util.*;`` (wildcard)
        """
        line, _ = self.get_line_numbers(node)
        # The import path is typically a scoped_identifier or identifier child
        # We extract the full text between 'import' (and optional 'static') and ';'
        import_path = self._extract_import_path(node, content)
        if import_path:
            refs.append(
                SymbolReference(
                    name=import_path,
                    ref_type="import",
                    file_path=file_path,
                    line=line,
                    parent_symbol=None,
                )
            )

    def _extract_import_path(self, node: Any, content: str) -> str | None:
        """Extract the full import path from an import_declaration node.

        Walks children to find scoped_identifier, identifier, or asterisk nodes
        and assembles the full import path.
        """
        for child in node.children:
            if child.type == "scoped_identifier":
                return self.get_node_content(child, content)
            if child.type == "identifier":
                return self.get_node_content(child, content)
            # Wildcard import: import java.util.*
            # tree-sitter-java represents this as scoped_identifier with asterisk
            if child.type == "asterisk":
                # The scoped part should have been found already; fallback to text
                pass
        # Fallback: extract from full text
        full_text = self.get_node_content(node, content).strip().rstrip(";").strip()
        # Remove 'import' keyword and optional 'static'
        if full_text.startswith("import"):
            full_text = full_text[len("import") :].strip()
        if full_text.startswith("static"):
            full_text = full_text[len("static") :].strip()
        return full_text if full_text else None
