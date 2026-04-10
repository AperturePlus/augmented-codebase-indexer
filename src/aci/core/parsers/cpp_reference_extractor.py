"""
C++ reference extractor.

Extracts symbol references (calls, includes, type annotations, inheritance)
from C++ source code using tree-sitter, complementing the CppParser which
extracts symbol definitions.
"""

from typing import Any

from aci.core.parsers.base import SymbolReference
from aci.core.parsers.reference_extractor import ReferenceExtractorInterface


class CppReferenceExtractor(ReferenceExtractorInterface):
    """Extract symbol references from C++ tree-sitter AST."""

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
        """Extract include directives from the AST."""
        refs: list[SymbolReference] = []
        self._traverse_includes(root_node, content, file_path, refs)
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

        # --- namespace_definition ---
        if node.type == "namespace_definition":
            ns_name = self._get_identifier(node, content)
            new_scope = [*scope_stack, ns_name] if ns_name else scope_stack
            for child in node.children:
                if child.type == "declaration_list":
                    for decl_child in child.children:
                        self._traverse(decl_child, content, file_path, refs, new_scope)
            return

        # --- class_specifier / struct_specifier ---
        if node.type in ("class_specifier", "struct_specifier"):
            class_name = self._get_class_name(node, content)
            if class_name:
                self._extract_inheritance(node, content, file_path, refs, parent_symbol)
                new_scope = [*scope_stack, class_name]
                for child in node.children:
                    if child.type == "field_declaration_list":
                        for body_child in child.children:
                            self._traverse(
                                body_child, content, file_path, refs, new_scope
                            )
            return

        # --- function_definition ---
        if node.type == "function_definition":
            func_name = self._get_function_name(node, content)
            if func_name:
                self._extract_function_type_refs(
                    node, content, file_path, refs, parent_symbol
                )
                new_scope = [*scope_stack, func_name]
                for child in node.children:
                    if child.type == "compound_statement":
                        for stmt in child.children:
                            self._traverse(
                                stmt, content, file_path, refs, new_scope
                            )
            return

        # --- function_declaration (prototype) ---
        if node.type in ("function_declaration", "field_declaration"):
            self._extract_function_type_refs(
                node, content, file_path, refs, parent_symbol
            )
            return

        # --- call_expression: foo(), obj.method(), obj->method(), ns::func() ---
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

        # --- new_expression: new ClassName() ---
        if node.type == "new_expression":
            name = self._extract_new_type_name(node, content)
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

        # --- declaration: variable declarations with types ---
        if node.type == "declaration":
            self._extract_declaration_type_ref(
                node, content, file_path, refs, parent_symbol
            )
            # Recurse into init expressions for calls
            for child in node.children:
                if child.type == "init_declarator":
                    for ic in child.children:
                        self._traverse(ic, content, file_path, refs, scope_stack)
            return

        # Default: recurse into children
        for child in node.children:
            self._traverse(child, content, file_path, refs, scope_stack)

    # ------------------------------------------------------------------
    # Internal traversal — includes only
    # ------------------------------------------------------------------

    def _traverse_includes(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
    ) -> None:
        """Walk the AST collecting only #include directives."""
        if node.type == "preproc_include":
            self._extract_include(node, content, file_path, refs)
            return

        for child in node.children:
            self._traverse_includes(child, content, file_path, refs)

    # ------------------------------------------------------------------
    # Name extraction helpers
    # ------------------------------------------------------------------

    def _get_identifier(self, node: Any, content: str) -> str | None:
        """Return the first identifier child's text."""
        for child in node.children:
            if child.type in ("identifier", "namespace_identifier"):
                return self.get_node_content(child, content)
        return None

    def _get_class_name(self, node: Any, content: str) -> str | None:
        """Return the class/struct name from a class_specifier or struct_specifier."""
        for child in node.children:
            if child.type == "type_identifier":
                return self.get_node_content(child, content)
        return None

    def _get_function_name(self, node: Any, content: str) -> str | None:
        """Extract the function name from a function_definition node."""
        for child in node.children:
            if child.type == "function_declarator":
                return self._get_declarator_name(child, content)
        return None

    def _get_declarator_name(self, node: Any, content: str) -> str | None:
        """Extract the name from a function_declarator or nested declarator."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_content(child, content)
            if child.type == "field_identifier":
                return self.get_node_content(child, content)
            if child.type == "qualified_identifier":
                return self.get_node_content(child, content)
            if child.type == "destructor_name":
                return self.get_node_content(child, content)
        return None

    def _resolve_name(self, node: Any, content: str) -> str | None:
        """Resolve an identifier, field_access, qualified_identifier, or scoped name."""
        if node.type in ("identifier", "field_identifier"):
            return self.get_node_content(node, content)
        if node.type == "qualified_identifier":
            return self.get_node_content(node, content)
        if node.type == "field_expression":
            # obj.method or obj->method
            return self.get_node_content(node, content)
        if node.type == "template_function":
            # func<T>() — extract the function name part
            for child in node.children:
                if child.type in ("identifier", "qualified_identifier", "field_identifier"):
                    return self.get_node_content(child, content)
        return None

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    def _extract_call_name(self, node: Any, content: str) -> str | None:
        """Extract the callable name from a ``call_expression`` node.

        Handles:
        - ``foo()``              → ``"foo"``
        - ``obj.method()``       → ``"obj.method"``
        - ``obj->method()``      → ``"obj->method"``
        - ``ns::func()``         → ``"ns::func"``
        - ``func<T>()``          → ``"func"``
        """
        if not node.children:
            return None
        func_node = node.children[0]
        return self._resolve_name(func_node, content)

    def _extract_new_type_name(self, node: Any, content: str) -> str | None:
        """Extract the type name from a ``new_expression`` node.

        Handles ``new ClassName()`` and ``new ns::ClassName()``.
        """
        for child in node.children:
            if child.type == "type_identifier":
                return self.get_node_content(child, content)
            if child.type == "qualified_identifier":
                return self.get_node_content(child, content)
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
        """Extract base classes from a class/struct specifier.

        Handles:
        - ``class Foo : public Bar, private Baz``
        - ``struct Foo : Bar``
        """
        for child in class_node.children:
            if child.type == "base_class_clause":
                self._extract_base_classes(
                    child, content, file_path, refs, parent_symbol
                )

    def _extract_base_classes(
        self,
        base_clause: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract each base class type from a base_class_clause node."""
        for child in base_clause.children:
            if child.type == "type_identifier":
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="inheritance",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
            elif child.type == "qualified_identifier":
                name = self.get_node_content(child, content)
                if name:
                    line, _ = self.get_line_numbers(child)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="inheritance",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )
            elif child.type == "template_type":
                # e.g. public Base<int> — extract the outer type
                for tc in child.children:
                    if tc.type in ("type_identifier", "qualified_identifier"):
                        name = self.get_node_content(tc, content)
                        if name:
                            line, _ = self.get_line_numbers(tc)
                            refs.append(
                                SymbolReference(
                                    name=name,
                                    ref_type="inheritance",
                                    file_path=file_path,
                                    line=line,
                                    parent_symbol=parent_symbol,
                                )
                            )
                        break

    # ------------------------------------------------------------------
    # Function type references (params + return type)
    # ------------------------------------------------------------------

    def _extract_function_type_refs(
        self,
        func_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type references from function return type and parameters."""
        for child in func_node.children:
            # Return type identifiers
            if child.type == "type_identifier":
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
            elif child.type == "qualified_identifier":
                # Could be a return type like ns::Type
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
            elif child.type == "template_type":
                self._extract_template_type_ref(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "function_declarator":
                self._extract_param_type_refs(
                    child, content, file_path, refs, parent_symbol
                )

    def _extract_param_type_refs(
        self,
        declarator_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract type references from function parameters."""
        for child in declarator_node.children:
            if child.type == "parameter_list":
                for param in child.children:
                    if param.type in ("parameter_declaration", "optional_parameter_declaration"):
                        self._extract_param_declaration_type(
                            param, content, file_path, refs, parent_symbol
                        )

    def _extract_param_declaration_type(
        self,
        param_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the type from a single parameter_declaration."""
        for child in param_node.children:
            if child.type == "type_identifier":
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
            elif child.type == "qualified_identifier":
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
            elif child.type == "template_type":
                self._extract_template_type_ref(
                    child, content, file_path, refs, parent_symbol
                )

    # ------------------------------------------------------------------
    # Variable declaration type references
    # ------------------------------------------------------------------

    def _extract_declaration_type_ref(
        self,
        decl_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the type from a declaration node (local variable)."""
        for child in decl_node.children:
            if child.type == "type_identifier":
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
            elif child.type == "qualified_identifier":
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
            elif child.type == "template_type":
                self._extract_template_type_ref(
                    child, content, file_path, refs, parent_symbol
                )

    # ------------------------------------------------------------------
    # Template type helper
    # ------------------------------------------------------------------

    def _extract_template_type_ref(
        self,
        template_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract the outer type name from a template_type node (e.g. vector<int>)."""
        for child in template_node.children:
            if child.type in ("type_identifier", "qualified_identifier"):
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

    # ------------------------------------------------------------------
    # Include extraction
    # ------------------------------------------------------------------

    def _extract_include(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
    ) -> None:
        """Extract include path from a ``preproc_include`` node.

        Handles:
        - ``#include <iostream>``
        - ``#include "myheader.h"``
        """
        line, _ = self.get_line_numbers(node)
        include_path: str | None = None

        for child in node.children:
            if child.type == "system_lib_string":
                # <iostream> — strip angle brackets
                raw = self.get_node_content(child, content)
                include_path = raw.strip("<>")
            elif child.type == "string_literal":
                # "myheader.h" — strip quotes
                raw = self.get_node_content(child, content)
                include_path = raw.strip('"')

        if include_path:
            refs.append(
                SymbolReference(
                    name=include_path,
                    ref_type="import",
                    file_path=file_path,
                    line=line,
                    parent_symbol=None,
                )
            )
