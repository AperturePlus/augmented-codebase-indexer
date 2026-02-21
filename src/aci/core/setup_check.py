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

    all_ok = True

    # Check tree-sitter core
    try:
        import tree_sitter  # noqa: F401

    except ImportError:
        all_ok = False

    # Check language packs
    language_packs = [
        ("tree_sitter_python", "Python"),
        ("tree_sitter_javascript", "JavaScript/TypeScript"),
        ("tree_sitter_go", "Go"),
    ]

    for module_name, _display_name in language_packs:
        try:
            module = __import__(module_name)
            # Try to get the language function
            if hasattr(module, "language"):
                module.language()
            else:
                all_ok = False
        except ImportError:
            all_ok = False
        except Exception:
            all_ok = False

    # Test parsing capability
    try:
        from aci.core.ast_parser import TreeSitterParser, check_tree_sitter_setup

        status = check_tree_sitter_setup()
        for _lang, available in status.items():
            if available:
                pass
            else:
                all_ok = False

    except ImportError:
        all_ok = False
    except Exception:
        all_ok = False

    # Test actual parsing
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
            pass
        else:
            pass

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
            pass
        else:
            pass

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
            pass
        else:
            pass

    except Exception:
        all_ok = False

    # Summary
    if all_ok:
        pass
    else:
        pass

    return all_ok


if __name__ == "__main__":
    success = check_tree_sitter_environment()
    sys.exit(0 if success else 1)
