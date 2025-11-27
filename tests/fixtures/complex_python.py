"""
Complex Python code fixture for AST parser testing.

This file contains various Python constructs to test the AST parser's
ability to handle real-world code patterns.
"""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Decorators and decorated functions
# =============================================================================

def simple_decorator(func):
    """A simple decorator."""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def decorator_with_args(arg1, arg2="default"):
    """A decorator factory with arguments."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


@simple_decorator
def decorated_function():
    """A function with a simple decorator."""
    pass


@decorator_with_args("value1", arg2="value2")
def function_with_parameterized_decorator(x: int, y: int) -> int:
    """A function with a parameterized decorator."""
    return x + y


@simple_decorator
@decorator_with_args("nested")
def multi_decorated_function():
    """A function with multiple decorators."""
    pass


# =============================================================================
# Classes with inheritance and complex structures
# =============================================================================

class BaseClass(ABC):
    """An abstract base class."""
    
    class_attribute: str = "base"
    
    def __init__(self, value: int):
        """Initialize the base class."""
        self._value = value
    
    @property
    def value(self) -> int:
        """Get the value."""
        return self._value
    
    @value.setter
    def value(self, new_value: int) -> None:
        """Set the value."""
        self._value = new_value
    
    @abstractmethod
    def abstract_method(self) -> str:
        """An abstract method that must be implemented."""
        pass
    
    @classmethod
    def class_method(cls) -> str:
        """A class method."""
        return cls.class_attribute
    
    @staticmethod
    def static_method(x: int, y: int) -> int:
        """A static method."""
        return x + y


class DerivedClass(BaseClass):
    """A class that inherits from BaseClass."""
    
    class_attribute = "derived"
    
    def __init__(self, value: int, name: str):
        """Initialize the derived class."""
        super().__init__(value)
        self.name = name
    
    def abstract_method(self) -> str:
        """Implementation of the abstract method."""
        return f"{self.name}: {self.value}"
    
    def method_with_complex_logic(self, items: List[int]) -> Dict[str, Any]:
        """A method with complex logic."""
        result = {
            "sum": sum(items),
            "count": len(items),
            "average": sum(items) / len(items) if items else 0,
        }
        
        # Nested comprehension
        filtered = [x for x in items if x > 0]
        
        # Conditional logic
        if filtered:
            result["max"] = max(filtered)
            result["min"] = min(filtered)
        else:
            result["max"] = None
            result["min"] = None
        
        return result


class MultipleInheritance(BaseClass, dict):
    """A class with multiple inheritance."""
    
    def __init__(self, value: int):
        BaseClass.__init__(self, value)
        dict.__init__(self)
    
    def abstract_method(self) -> str:
        return str(self.value)


# =============================================================================
# Nested classes
# =============================================================================

class OuterClass:
    """A class containing nested classes."""
    
    class InnerClass:
        """A nested class."""
        
        def inner_method(self):
            """A method in the inner class."""
            return "inner"
        
        class DeeplyNestedClass:
            """A deeply nested class."""
            
            def deeply_nested_method(self):
                """A method in the deeply nested class."""
                return "deeply nested"
    
    def outer_method(self):
        """A method in the outer class."""
        return self.InnerClass()


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class SimpleDataclass:
    """A simple dataclass."""
    name: str
    value: int
    optional: Optional[str] = None


@dataclass
class DataclassWithMethods:
    """A dataclass with custom methods."""
    x: int
    y: int
    
    def distance_from_origin(self) -> float:
        """Calculate distance from origin."""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.x < 0 or self.y < 0:
            raise ValueError("Coordinates must be non-negative")


# =============================================================================
# Async functions
# =============================================================================

async def async_function(url: str) -> str:
    """An async function."""
    return f"fetched: {url}"


async def async_function_with_await():
    """An async function that uses await."""
    result = await async_function("http://example.com")
    return result


class AsyncClass:
    """A class with async methods."""
    
    async def async_method(self) -> str:
        """An async method."""
        return "async result"
    
    async def async_context_manager(self):
        """An async method using context manager."""
        async with self:
            pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass


# =============================================================================
# Generator functions
# =============================================================================

def generator_function(n: int):
    """A generator function."""
    for i in range(n):
        yield i


def generator_with_send():
    """A generator that accepts sent values."""
    value = yield "start"
    while True:
        value = yield f"received: {value}"


# =============================================================================
# Lambda and closures
# =============================================================================

def function_returning_lambda():
    """A function that returns a lambda."""
    return lambda x, y: x + y


def closure_example(multiplier: int):
    """A function demonstrating closures."""
    def inner(value: int) -> int:
        return value * multiplier
    return inner


# =============================================================================
# Complex function signatures
# =============================================================================

def function_with_complex_signature(
    pos_only: int,
    /,
    regular: str,
    *args: int,
    keyword_only: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """A function with a complex signature using all parameter types."""
    return {
        "pos_only": pos_only,
        "regular": regular,
        "args": args,
        "keyword_only": keyword_only,
        "kwargs": kwargs,
    }


def function_with_type_hints(
    items: List[Dict[str, int]],
    callback: Optional[callable] = None,
) -> List[int]:
    """A function with complex type hints."""
    results = []
    for item in items:
        value = sum(item.values())
        if callback:
            value = callback(value)
        results.append(value)
    return results
