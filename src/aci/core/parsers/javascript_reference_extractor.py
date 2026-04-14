"""
JavaScript/TypeScript reference extractor.

Extracts symbol references (calls, imports, type annotations, inheritance)
from JavaScript and TypeScript source code using tree-sitter, complementing
the JavaScriptParser which extracts symbol definitions.
"""

from typing import Any

from aci.core.parsers.base import SymbolReference
from aci.core.parsers.reference_extractor import ReferenceExtractorInterface


class JavaScriptReferenceExtractor(ReferenceExtractorInterface):
    """Extract symbol references from JavaScript/TypeScript tree-sitter AST."""

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

        # --- class_declaration ---
        if node.type == "class_declaration":
            class_name = self._get_identifier(node, content)
            if class_name:
                self._extract_inheritance(node, content, file_path, refs, parent_symbol)
                new_scope = [*scope_stack, class_name]
                for child in node.children:
                    if child.type == "class_body":
                        for body_child in child.children:
                            self._traverse(
                                body_child, content, file_path, refs, new_scope
                            )
            return

        # --- method_definition (inside class body) ---
        if node.type == "method_definition":
            method_name = self._get_property_identifier(node, content)
            if method_name:
                self._extract_ts_annotations(
                    node, content, file_path, refs, parent_symbol
                )
                new_scope = [*scope_stack, method_name]
                for child in node.children:
                    if child.type == "statement_block":
                        for block_child in child.children:
                            self._traverse(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        # --- function_declaration ---
        if node.type == "function_declaration":
            func_name = self._get_identifier(node, content)
            if func_name:
                self._extract_ts_annotations(
                    node, content, file_path, refs, parent_symbol
                )
                new_scope = [*scope_stack, func_name]
                for child in node.children:
                    if child.type == "statement_block":
                        for block_child in child.children:
                            self._traverse(
                                block_child, content, file_path, refs, new_scope
                            )
            return

        # --- arrow function / function expression in variable declaration ---
        if node.type in ("lexical_declaration", "variable_declaration"):
            var_name = self._get_variable_func_name(node)
            if var_name:
                name_text = self.get_node_content(var_name, content)
                new_scope = [*scope_stack, name_text]
                for child in node.children:
                    if child.type == "variable_declarator":
                        for decl_child in child.children:
                            if decl_child.type in (
                                "arrow_function",
                                "function_expression",
                            ):
                                for fn_child in decl_child.children:
                                    if fn_child.type == "statement_block":
                                        for block_child in fn_child.children:
                                            self._traverse(
                                                block_child,
                                                content,
                                                file_path,
                                                refs,
                                                new_scope,
                                            )
                return
            # Not a function variable — fall through to recurse children
            for child in node.children:
                self._traverse(child, content, file_path, refs, scope_stack)
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
            # Recurse into arguments for nested calls
            for child in node.children:
                if child.type == "arguments":
                    for arg in child.children:
                        self._traverse(arg, content, file_path, refs, scope_stack)
            return

        # --- new_expression (constructor calls) ---
        if node.type == "new_expression":
            name = self._extract_new_name(node, content)
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
                if child.type == "arguments":
                    for arg in child.children:
                        self._traverse(arg, content, file_path, refs, scope_stack)
            return

        # --- type_annotation (TypeScript) ---
        if node.type == "type_annotation":
            type_name = self._extract_type_name(node, content)
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
        scope_stack: list[str],
    ) -> None:
        """Walk the AST collecting only import references."""
        parent_symbol = ".".join(scope_stack) if scope_stack else None

        # ES6 import statement
        if node.type == "import_statement":
            self._extract_es6_import(node, content, file_path, refs, parent_symbol)
            return

        # CommonJS require() — handled as call_expression with "require" callee
        if node.type == "call_expression":
            callee = self._first_child_of_type(node, "identifier")
            if callee and self.get_node_content(callee, content) == "require":
                self._extract_require_import(
                    node, content, file_path, refs, parent_symbol
                )
                return

        # Track scope for parent_symbol on imports inside functions/classes
        if node.type == "class_declaration":
            class_name = self._get_identifier(node, content)
            if class_name:
                new_scope = [*scope_stack, class_name]
                for child in node.children:
                    if child.type == "class_body":
                        for body_child in child.children:
                            self._traverse_imports(
                                body_child, content, file_path, refs, new_scope
                            )
            return

        if node.type in ("function_declaration", "method_definition"):
            func_name = (
                self._get_identifier(node, content)
                if node.type == "function_declaration"
                else self._get_property_identifier(node, content)
            )
            if func_name:
                new_scope = [*scope_stack, func_name]
                for child in node.children:
                    if child.type == "statement_block":
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
        """Return the first ``identifier`` child's text (class/function name)."""
        for child in node.children:
            if child.type == "identifier":
                return self.get_node_content(child, content)
        return None

    def _get_property_identifier(self, node: Any, content: str) -> str | None:
        """Return the first ``property_identifier`` child's text (method name)."""
        for child in node.children:
            if child.type == "property_identifier":
                return self.get_node_content(child, content)
        return None

    def _get_variable_func_name(self, node: Any) -> Any | None:
        """Return the identifier node for a variable-declared function/arrow.

        Looks for patterns like ``const foo = () => {}`` or
        ``const foo = function() {}``.
        """
        for child in node.children:
            if child.type == "variable_declarator":
                has_func = False
                ident_node = None
                for decl_child in child.children:
                    if decl_child.type == "identifier":
                        ident_node = decl_child
                    elif decl_child.type in ("arrow_function", "function_expression"):
                        has_func = True
                if has_func and ident_node is not None:
                    return ident_node
        return None

    def _first_child_of_type(self, node: Any, child_type: str) -> Any | None:
        """Return the first child matching *child_type*, or ``None``."""
        for child in node.children:
            if child.type == child_type:
                return child
        return None

    def _dotted_name(self, node: Any, content: str) -> str | None:
        """Resolve an identifier or member expression to a dotted string."""
        if node.type == "identifier":
            return self.get_node_content(node, content)
        if node.type == "member_expression":
            return self.get_node_content(node, content)
        return None

    def _extract_call_name(self, node: Any, content: str) -> str | None:
        """Extract the callable name from a ``call_expression`` node.

        Handles ``foo()``, ``obj.method()``, ``a.b.c()`` patterns.
        """
        if not node.children:
            return None
        func_node = node.children[0]
        return self._dotted_name(func_node, content)

    def _extract_new_name(self, node: Any, content: str) -> str | None:
        """Extract the class name from a ``new_expression`` node.

        Handles ``new Foo()`` and ``new a.B()`` patterns.
        """
        for child in node.children:
            if child.type in ("identifier", "member_expression"):
                return self._dotted_name(child, content)
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
        """Extract base classes from a ``class_declaration`` node.

        Handles ``class Foo extends Bar`` and TypeScript
        ``class Foo extends Bar implements Baz``.
        """
        for child in class_node.children:
            # tree-sitter-javascript: class_heritage contains the extends clause
            if child.type == "class_heritage":
                for heritage_child in child.children:
                    name = self._dotted_name(heritage_child, content)
                    if name:
                        line, _ = self.get_line_numbers(heritage_child)
                        refs.append(
                            SymbolReference(
                                name=name,
                                ref_type="inheritance",
                                file_path=file_path,
                                line=line,
                                parent_symbol=parent_symbol,
                            )
                        )

    # ------------------------------------------------------------------
    # Type annotation extraction (TypeScript / JSDoc best-effort)
    # ------------------------------------------------------------------

    def _extract_type_name(self, node: Any, content: str) -> str | None:
        """Extract the type name from a ``type_annotation`` node.

        Looks for the first ``type_identifier`` or ``identifier`` child.
        Falls back to the full text content for complex types.
        """
        for child in node.children:
            if child.type in ("type_identifier", "identifier"):
                return self.get_node_content(child, content)
            if child.type == "generic_type":
                # e.g. Array<string> — extract the outer type name
                for gc in child.children:
                    if gc.type in ("type_identifier", "identifier"):
                        return self.get_node_content(gc, content)
        return None

    def _extract_ts_annotations(
        self,
        func_node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract TypeScript type annotations from function/method parameters and return type."""
        for child in func_node.children:
            if child.type == "formal_parameters":
                self._extract_param_annotations(
                    child, content, file_path, refs, parent_symbol
                )
            elif child.type == "type_annotation":
                # Return type annotation
                type_name = self._extract_type_name(child, content)
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
            # Walk each parameter looking for type_annotation children
            self._collect_type_annotations(param, content, file_path, refs, parent_symbol)

    def _collect_type_annotations(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Recursively collect type_annotation nodes from a parameter subtree."""
        if node.type == "type_annotation":
            type_name = self._extract_type_name(node, content)
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
            self._collect_type_annotations(
                child, content, file_path, refs, parent_symbol
            )

    # ------------------------------------------------------------------
    # Import extraction helpers
    # ------------------------------------------------------------------

    def _extract_es6_import(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract references from ES6 import statements.

        Handles:
        - ``import foo from 'module'``
        - ``import { foo, bar } from 'module'``
        - ``import * as foo from 'module'``
        - ``import 'module'`` (side-effect import)
        """
        line, _ = self.get_line_numbers(node)

        # Extract the module source string
        module_name = self._extract_string_value(node)
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

        # Extract named imports from import_clause
        for child in node.children:
            if child.type == "import_clause":
                self._extract_import_clause_names(
                    child, content, file_path, line, refs, parent_symbol
                )

    def _extract_import_clause_names(
        self,
        clause_node: Any,
        content: str,
        file_path: str,
        line: int,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract imported names from an ``import_clause`` node."""
        for child in clause_node.children:
            # Default import: import foo from '...'
            if child.type == "identifier":
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
            # Named imports: import { foo, bar as baz } from '...'
            elif child.type == "named_imports":
                for spec in child.children:
                    if spec.type == "import_specifier":
                        # Use the first identifier (the original name)
                        ident = self._first_child_of_type(spec, "identifier")
                        if ident:
                            name = self.get_node_content(ident, content)
                            refs.append(
                                SymbolReference(
                                    name=name,
                                    ref_type="import",
                                    file_path=file_path,
                                    line=line,
                                    parent_symbol=parent_symbol,
                                )
                            )
            # Namespace import: import * as foo from '...'
            elif child.type == "namespace_import":
                ident = self._first_child_of_type(child, "identifier")
                if ident:
                    name = self.get_node_content(ident, content)
                    refs.append(
                        SymbolReference(
                            name=name,
                            ref_type="import",
                            file_path=file_path,
                            line=line,
                            parent_symbol=parent_symbol,
                        )
                    )

    def _extract_require_import(
        self,
        node: Any,
        content: str,
        file_path: str,
        refs: list[SymbolReference],
        parent_symbol: str | None,
    ) -> None:
        """Extract references from CommonJS ``require('module')`` calls."""
        line, _ = self.get_line_numbers(node)
        # Extract the module string from the arguments
        args = self._first_child_of_type(node, "arguments")
        if args:
            module_name = self._extract_string_value(args)
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

    def _extract_string_value(self, node: Any) -> str | None:
        """Extract the text of the first ``string`` child, stripping quotes."""
        for child in node.children:
            if child.type == "string":
                raw = child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                # Strip surrounding quotes (single, double, or backtick)
                if len(raw) >= 2 and raw[0] in ("'", '"', "`"):
                    return raw[1:-1]
                return raw
        return None
