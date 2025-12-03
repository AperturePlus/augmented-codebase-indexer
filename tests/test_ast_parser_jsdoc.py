"""
Unit tests for JSDoc extraction in AST parser.

Tests JavaScript/TypeScript docstring extraction functionality.
Requirements: 1.1, 1.2, 1.3, 1.4
"""

import pytest

from aci.core.ast_parser import TreeSitterParser


class TestJSDocExtraction:
    """Test JSDoc extraction from JavaScript/TypeScript code."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = TreeSitterParser()

    def test_function_with_single_line_jsdoc(self):
        """Test extracting single-line JSDoc from function."""
        code = '''/** Adds two numbers */
function add(a, b) {
    return a + b;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "add"
        assert nodes[0].node_type == "function"
        assert nodes[0].docstring is not None
        assert "Adds two numbers" in nodes[0].docstring

    def test_function_with_multiline_jsdoc(self):
        """Test extracting multi-line JSDoc with tags."""
        code = '''/**
 * Authenticates a user with credentials.
 * @param {string} username - The user's login name
 * @param {string} password - The user's password
 * @returns {Promise<string>} JWT token
 */
function authenticate(username, password) {
    return login(username, password);
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "authenticate"
        assert nodes[0].docstring is not None
        assert "Authenticates a user" in nodes[0].docstring
        assert "@param" in nodes[0].docstring
        assert "@returns" in nodes[0].docstring

    def test_class_with_jsdoc(self):
        """Test extracting JSDoc from class declaration."""
        code = '''/**
 * Represents a user in the system.
 * @class
 */
class User {
    constructor(name) {
        this.name = name;
    }
}'''
        nodes = self.parser.parse(code, "javascript")
        
        # Should have class and constructor method
        class_nodes = [n for n in nodes if n.node_type == "class"]
        assert len(class_nodes) == 1
        assert class_nodes[0].name == "User"
        assert class_nodes[0].docstring is not None
        assert "Represents a user" in class_nodes[0].docstring

    def test_method_with_jsdoc(self):
        """Test extracting JSDoc from class method."""
        code = '''class Calculator {
    /**
     * Multiplies two numbers.
     * @param {number} a - First number
     * @param {number} b - Second number
     * @returns {number} Product
     */
    multiply(a, b) {
        return a * b;
    }
}'''
        nodes = self.parser.parse(code, "javascript")
        
        method_nodes = [n for n in nodes if n.node_type == "method"]
        assert len(method_nodes) >= 1
        
        multiply = next((n for n in method_nodes if n.name == "multiply"), None)
        assert multiply is not None
        assert multiply.docstring is not None
        assert "Multiplies two numbers" in multiply.docstring

    def test_arrow_function_with_jsdoc(self):
        """Test extracting JSDoc from arrow function."""
        code = '''/**
 * Calculates the square of a number.
 * @param {number} x - Input value
 * @returns {number} Square of x
 */
const square = (x) => x * x;'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "square"
        assert nodes[0].docstring is not None
        assert "Calculates the square" in nodes[0].docstring

    def test_function_without_jsdoc(self):
        """Test function without JSDoc has None docstring."""
        code = '''function noDoc() {
    return 42;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "noDoc"
        assert nodes[0].docstring is None

    def test_jsdoc_with_description_tag(self):
        """Test JSDoc with @description tag."""
        code = '''/**
 * @description Formats a date string.
 * @param {Date} date - Date to format
 * @returns {string} Formatted date
 */
function formatDate(date) {
    return date.toISOString();
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "@description" in nodes[0].docstring

    def test_jsdoc_with_throws_tag(self):
        """Test JSDoc with @throws tag."""
        code = '''/**
 * Divides two numbers.
 * @param {number} a - Dividend
 * @param {number} b - Divisor
 * @returns {number} Quotient
 * @throws {Error} If divisor is zero
 */
function divide(a, b) {
    if (b === 0) throw new Error("Division by zero");
    return a / b;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "@throws" in nodes[0].docstring

    def test_jsdoc_with_example_tag(self):
        """Test JSDoc with @example tag."""
        code = '''/**
 * Greets a person.
 * @param {string} name - Person's name
 * @returns {string} Greeting message
 * @example
 * greet("World") // returns "Hello, World!"
 */
function greet(name) {
    return `Hello, ${name}!`;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "@example" in nodes[0].docstring

    def test_typescript_function_with_jsdoc(self):
        """Test JSDoc extraction from TypeScript-like code (JS parser)."""
        # Note: TypeScript uses same parser as JavaScript
        # TypeScript-specific syntax may not parse correctly
        code = '''/**
 * Fetches user data from API.
 * @param {number} userId - User ID
 * @returns {Promise} User object
 */
async function fetchUser(userId) {
    return await api.get("/users/" + userId);
}'''
        nodes = self.parser.parse(code, "typescript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "fetchUser"
        assert nodes[0].docstring is not None
        assert "Fetches user data" in nodes[0].docstring

    def test_multiple_functions_with_jsdoc(self):
        """Test multiple functions each with their own JSDoc."""
        code = '''/**
 * Adds numbers.
 */
function add(a, b) {
    return a + b;
}

/**
 * Subtracts numbers.
 */
function subtract(a, b) {
    return a - b;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 2
        
        add_node = next((n for n in nodes if n.name == "add"), None)
        sub_node = next((n for n in nodes if n.name == "subtract"), None)
        
        assert add_node is not None
        assert add_node.docstring is not None
        assert "Adds numbers" in add_node.docstring
        
        assert sub_node is not None
        assert sub_node.docstring is not None
        assert "Subtracts numbers" in sub_node.docstring

    def test_jsdoc_with_unicode(self):
        """Test JSDoc with Unicode characters."""
        code = """/**
 * Handles user request with emoji
 * @param {string} request - Request content
 * @returns {string} Response
 */
function handleRequest(request) {
    return process(request);
}"""
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "Handles user request" in nodes[0].docstring

    def test_regular_comment_not_extracted(self):
        """Test that regular comments (not JSDoc) are not extracted."""
        code = '''// This is a regular comment
function regularComment() {
    return 1;
}

/* This is a block comment */
function blockComment() {
    return 2;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 2
        for node in nodes:
            assert node.docstring is None


class TestJSDocExportSyntax:
    """Test JSDoc extraction with export syntax."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = TreeSitterParser()

    def test_export_function_with_jsdoc(self):
        """Test JSDoc extraction from exported function."""
        code = '''/**
 * Adds two numbers together.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} Sum
 */
export function add(a, b) {
    return a + b;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "add"
        assert nodes[0].docstring is not None
        assert "Adds two numbers" in nodes[0].docstring

    def test_export_default_function_with_jsdoc(self):
        """Test JSDoc extraction from export default function."""
        code = '''/**
 * Main entry point.
 * @returns {void}
 */
export default function main() {
    console.log("Hello");
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "main"
        assert nodes[0].docstring is not None
        assert "Main entry point" in nodes[0].docstring

    def test_export_async_function_with_jsdoc(self):
        """Test JSDoc extraction from exported async function."""
        code = '''/**
 * Fetches data from API.
 * @param {string} url - API endpoint
 * @returns {Promise<object>} Response data
 */
export async function fetchData(url) {
    return await fetch(url);
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "fetchData"
        assert nodes[0].docstring is not None
        assert "Fetches data" in nodes[0].docstring

    def test_export_class_with_jsdoc(self):
        """Test JSDoc extraction from exported class."""
        code = '''/**
 * User service for managing users.
 * @class
 */
export class UserService {
    /**
     * Creates a new user.
     * @param {string} name - User name
     */
    createUser(name) {
        return { name };
    }
}'''
        nodes = self.parser.parse(code, "javascript")
        
        class_nodes = [n for n in nodes if n.node_type == "class"]
        assert len(class_nodes) == 1
        assert class_nodes[0].name == "UserService"
        assert class_nodes[0].docstring is not None
        assert "User service" in class_nodes[0].docstring

    def test_export_const_arrow_function_with_jsdoc(self):
        """Test JSDoc extraction from exported const arrow function."""
        code = '''/**
 * Squares a number.
 * @param {number} x - Input
 * @returns {number} Square
 */
export const square = (x) => x * x;'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "square"
        assert nodes[0].docstring is not None
        assert "Squares a number" in nodes[0].docstring


class TestJSDocEdgeCases:
    """Test JSDoc extraction edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = TreeSitterParser()

    def test_jsdoc_with_blank_line_not_extracted(self):
        """Test that JSDoc with blank line between is not extracted."""
        code = '''/**
 * This should not be extracted.
 */

function separated() {
    return 1;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "separated"
        # Should NOT have docstring due to blank line
        assert nodes[0].docstring is None

    def test_license_header_not_extracted(self):
        """Test that file-level license comments are not extracted."""
        code = '''/**
 * Copyright 2024 Example Corp.
 * Licensed under MIT License.
 */

/**
 * Adds two numbers.
 */
function add(a, b) {
    return a + b;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].name == "add"
        assert nodes[0].docstring is not None
        # Should have the function's JSDoc, not the license
        assert "Adds two numbers" in nodes[0].docstring
        assert "Copyright" not in nodes[0].docstring

    def test_multiple_jsdoc_takes_closest(self):
        """Test that when multiple JSDoc exist, closest one is taken."""
        code = '''/**
 * First comment - should be ignored.
 */
/**
 * Second comment - should be extracted.
 */
function foo() {
    return 1;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "Second comment" in nodes[0].docstring
        assert "First comment" not in nodes[0].docstring

    def test_empty_jsdoc(self):
        """Test handling of empty JSDoc comment."""
        code = '''/** */
function empty() {
    return 1;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        # Empty JSDoc should still be extracted
        assert nodes[0].docstring is not None

    def test_jsdoc_with_only_whitespace(self):
        """Test handling of JSDoc with only whitespace."""
        code = '''/**
 *
 */
function whitespace() {
    return 1;
}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
