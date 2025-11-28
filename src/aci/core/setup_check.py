#!/usr/bin/env python3
"""
Tree-sitter Environment Check Script

This script verifies that Tree-sitter and all required language packs
are properly installed and can be loaded.
"""

import sys


def check_tree_sitter_environment() -> bool:
    """
    Check Tree-sitter environment and report status.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("Tree-sitter Environment Check")
    print("=" * 60)

    all_ok = True

    # Check tree-sitter core
    print("\n[1] Checking tree-sitter core library...")
    try:
        import tree_sitter

        print(
            f"    ✓ tree-sitter version: {tree_sitter.__version__ if hasattr(tree_sitter, '__version__') else 'installed'}"
        )
    except ImportError as e:
        print(f"    ✗ tree-sitter not installed: {e}")
        all_ok = False

    # Check language packs
    language_packs = [
        ("tree_sitter_python", "Python"),
        ("tree_sitter_javascript", "JavaScript/TypeScript"),
        ("tree_sitter_go", "Go"),
    ]

    print("\n[2] Checking language packs...")
    for module_name, display_name in language_packs:
        try:
            module = __import__(module_name)
            # Try to get the language function
            if hasattr(module, "language"):
                lang_ptr = module.language()
                print(f"    ✓ {display_name} ({module_name}): loaded successfully")
            else:
                print(f"    ? {display_name} ({module_name}): installed but no language() function")
                all_ok = False
        except ImportError as e:
            print(f"    ✗ {display_name} ({module_name}): not installed - {e}")
            all_ok = False
        except Exception as e:
            print(f"    ✗ {display_name} ({module_name}): error loading - {e}")
            all_ok = False

    # Test parsing capability
    print("\n[3] Testing parsing capability...")
    try:
        from aci.core.ast_parser import TreeSitterParser, check_tree_sitter_setup

        status = check_tree_sitter_setup()
        for lang, available in status.items():
            if available:
                print(f"    ✓ {lang}: parser ready")
            else:
                print(f"    ✗ {lang}: parser not available")
                all_ok = False

    except ImportError as e:
        print(f"    ✗ Could not import ACI parser module: {e}")
        all_ok = False
    except Exception as e:
        print(f"    ✗ Error testing parser: {e}")
        all_ok = False

    # Test actual parsing
    print("\n[4] Testing actual code parsing...")
    try:
        from aci.core.ast_parser import TreeSitterParser

        parser = TreeSitterParser()

        # Test Python
        python_code = '''
def hello():
    """Say hello."""
    print("Hello, World!")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
'''
        nodes = parser.parse(python_code, "python")
        if len(nodes) >= 3:  # function, class, method
            print(f"    ✓ Python parsing: found {len(nodes)} nodes")
        else:
            print(f"    ? Python parsing: found {len(nodes)} nodes (expected >= 3)")

        # Test JavaScript
        js_code = """
function greet(name) {
    return "Hello, " + name;
}

const sayHi = () => {
    console.log("Hi!");
};

class Greeter {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return "Hello, " + this.name;
    }
}
"""
        nodes = parser.parse(js_code, "javascript")
        if len(nodes) >= 4:  # function, arrow function, class, methods
            print(f"    ✓ JavaScript parsing: found {len(nodes)} nodes")
        else:
            print(f"    ? JavaScript parsing: found {len(nodes)} nodes (expected >= 4)")

        # Test Go
        go_code = """
package main

func hello() {
    fmt.Println("Hello!")
}

type Greeter struct {
    Name string
}

func (g *Greeter) Greet() string {
    return "Hello, " + g.Name
}
"""
        nodes = parser.parse(go_code, "go")
        if len(nodes) >= 3:  # function, struct, method
            print(f"    ✓ Go parsing: found {len(nodes)} nodes")
        else:
            print(f"    ? Go parsing: found {len(nodes)} nodes (expected >= 3)")

    except Exception as e:
        print(f"    ✗ Error during parsing test: {e}")
        all_ok = False

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! Tree-sitter environment is ready.")
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nTo install missing packages, run:")
        print("  uv add tree-sitter tree-sitter-python tree-sitter-javascript tree-sitter-go")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = check_tree_sitter_environment()
    sys.exit(0 if success else 1)
