
import unittest
import asyncio
from pathlib import Path
import shutil
import tempfile
import os
from typing import List, Tuple

# Import necessary modules from the project
from aci.core.ast_parser import TreeSitterParser
from aci.core.chunker import Chunker
from aci.core.file_scanner import FileScanner, ScannedFile
from aci.core.tokenizer import TokenizerInterface

# Mock Tokenizer for testing
class MockTokenizer(TokenizerInterface):
    def encode(self, text: str) -> List[int]:
        return [1] * len(text.split())  # Simple mock
    
    def decode(self, tokens: List[int]) -> str:
        return " ".join(["token"] * len(tokens))
    
    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text:
            return ""
        
        if self.count_tokens(text) <= max_tokens:
            return text
            
        lines = text.split('\n')
        result = []
        current_count = 0
        
        for line in lines:
            count = self.count_tokens(line)
            if current_count + count > max_tokens:
                break
            result.append(line)
            current_count += count
            
        return "\n".join(result)

class TestCommentExtractionIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.parser = TreeSitterParser()
        self.tokenizer = MockTokenizer()
        self.chunker = Chunker(tokenizer=self.tokenizer)
        
        # Create complex test files with comments
        self.create_test_files()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        # 1. JavaScript with JSDoc
        js_content = """
/**
 * Processes a file by reading its content and applying a transformation.
 * This function handles text encoding and error logging.
 * @param {string} filePath - The path to the file
 * @returns {Promise<string>} The processed content
 */
async function processFile(filePath) {
    const fs = require('fs');
    // Read file
    const content = await fs.promises.readFile(filePath, 'utf-8');
    return content.toUpperCase();
}

/**
 * Calculates the retry delay based on exponential backoff.
 * @param {number} attempt - The current attempt number
 */
const calculateRetry = (attempt) => {
    return Math.pow(2, attempt) * 1000;
}

class DataManager {
    /**
     * Saves data to the persistent storage.
     * Warning: This operation is blocking.
     */
    saveToStorage(data) {
        console.log("Saving...");
    }
}
"""
        (self.test_dir / "complex_js.js").write_text(js_content, encoding="utf-8")

        # 2. Go with GoDoc
        go_content = """
package main

import "fmt"

// User represents a system user.
type User struct {
    Name string
    ID   int
}

// AuthenticateUser verifies the user's credentials against the database.
// It returns a session token if successful, or an error otherwise.
//
// Security: This function uses constant-time comparison.
func AuthenticateUser(username, password string) (string, error) {
    return "token", nil
}

/*
   ConnectDatabase establishes a connection pool to the SQL database.
   It retries automatically on connection failure.
*/
func ConnectDatabase(url string) error {
    fmt.Println("Connecting...")
    return nil
}

// Helper function without doc comment
func helper() {}
"""
        (self.test_dir / "complex_go.go").write_text(go_content, encoding="utf-8")

    def test_javascript_comment_extraction(self):
        """Test JSDoc extraction and indexing in JavaScript."""
        file_path = self.test_dir / "complex_js.js"
        content = file_path.read_text(encoding="utf-8")
        
        # 1. Parse AST
        ast_nodes = self.parser.parse(content, "javascript")
        
        # Verify AST extraction
        self.assertTrue(len(ast_nodes) > 0, "Should extract AST nodes from JS")
        
        # Find specific nodes
        process_func = next((n for n in ast_nodes if n.name == "processFile"), None)
        retry_func = next((n for n in ast_nodes if n.name == "calculateRetry"), None)
        save_method = next((n for n in ast_nodes if n.name == "saveToStorage"), None)

        # Verify Docstrings in AST
        self.assertIsNotNone(process_func)
        self.assertIn("Processes a file", process_func.docstring)
        self.assertIn("@param", process_func.docstring)

        self.assertIsNotNone(retry_func)
        self.assertIn("Calculates the retry delay", retry_func.docstring)

        self.assertIsNotNone(save_method)
        self.assertIn("Saves data to the persistent storage", save_method.docstring)

        # 2. Chunking & Content Integration
        scanned_file = ScannedFile(
            path=file_path,
            content=content,
            language="javascript",
            size_bytes=len(content),
            modified_time=0,
            content_hash="abc"
        )
        
        result = self.chunker.chunk(scanned_file, ast_nodes)
        chunks = result.chunks
        
        # Verify Chunks contain comments in CONTENT (not just metadata)
        process_chunk = next((c for c in chunks if c.metadata.get("function_name") == "processFile"), None)
        self.assertIsNotNone(process_chunk)
        # The chunk content should START with the docstring (or contain it prominently)
        self.assertIn("Processes a file", process_chunk.content)
        self.assertIn("async function processFile", process_chunk.content)
        
        print("\n[PASS] JavaScript Comment Extraction Verified")
        print(f"  - Found function 'processFile' with docstring length: {len(process_func.docstring)}")
        print(f"  - Chunk content preview: {process_chunk.content[:50]}...")

    def test_go_comment_extraction(self):
        """Test GoDoc extraction and indexing in Go."""
        file_path = self.test_dir / "complex_go.go"
        content = file_path.read_text(encoding="utf-8")
        
        # 1. Parse AST
        ast_nodes = self.parser.parse(content, "go")
        
        # Find nodes
        auth_func = next((n for n in ast_nodes if n.name == "AuthenticateUser"), None)
        connect_func = next((n for n in ast_nodes if n.name == "ConnectDatabase"), None)
        user_struct = next((n for n in ast_nodes if n.name == "User"), None)

        # Verify Docstrings
        self.assertIsNotNone(auth_func)
        self.assertIn("verifies the user's credentials", auth_func.docstring)
        self.assertIn("Security:", auth_func.docstring)

        self.assertIsNotNone(connect_func)
        self.assertIn("establishes a connection pool", connect_func.docstring)
        
        self.assertIsNotNone(user_struct)
        self.assertIn("represents a system user", user_struct.docstring)

        # 2. Chunking
        scanned_file = ScannedFile(
            path=file_path,
            content=content,
            language="go",
            size_bytes=len(content),
            modified_time=0,
            content_hash="xyz"
        )
        result = self.chunker.chunk(scanned_file, ast_nodes)
        chunks = result.chunks
        
        auth_chunk = next((c for c in chunks if c.metadata.get("function_name") == "AuthenticateUser"), None)
        self.assertIsNotNone(auth_chunk)
        self.assertIn("verifies the user's credentials", auth_chunk.content)
        
        print("\n[PASS] Go Comment Extraction Verified")
        print(f"  - Found function 'AuthenticateUser' with docstring length: {len(auth_func.docstring)}")

    def test_search_relevance_simulation(self):
        """
        Simulate a search query scenario.
        Query: "file processing error handling"
        Expected: Match 'processFile' function in JS because of its docstring.
        """
        # This test simulates what the semantic search would 'see' in the vector
        # We can't run full vector search here without embedding model, 
        # but we can verify the information is present in the text to be embedded.
        
        file_path = self.test_dir / "complex_js.js"
        content = file_path.read_text(encoding="utf-8")
        ast_nodes = self.parser.parse(content, "javascript")
        scanned_file = ScannedFile(path=file_path, content=content, language="javascript", size_bytes=len(content), modified_time=0, content_hash="abc")
        result = self.chunker.chunk(scanned_file, ast_nodes)
        chunks = result.chunks
        
        process_chunk = next((c for c in chunks if c.metadata.get("function_name") == "processFile"), None)
        
        # The query keywords are in the docstring, not the code variable names (except 'process')
        # "text encoding" is in docstring
        # "error logging" is in docstring
        
        term_in_doc = "error logging"
        term_in_code = "fs.promises.readFile"
        
        self.assertIn(term_in_doc, process_chunk.content)
        self.assertIn(term_in_code, process_chunk.content)
        
        print("\n[PASS] Search Relevance Simulation Verified")
        print(f"  - Chunk contains semantic term '{term_in_doc}' from docstring")
        print(f"  - Chunk contains code term '{term_in_code}' from source")

if __name__ == "__main__":
    unittest.main()
