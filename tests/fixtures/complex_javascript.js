/**
 * Complex JavaScript code fixture for AST parser testing.
 * 
 * This file contains various JavaScript constructs to test the AST parser's
 * ability to handle real-world code patterns.
 */

// =============================================================================
// Function declarations and expressions
// =============================================================================

function simpleFunction() {
    return "simple";
}

function functionWithParams(a, b, c = 10) {
    return a + b + c;
}

function functionWithRestParams(...args) {
    return args.reduce((sum, val) => sum + val, 0);
}

function functionWithDestructuring({ name, value }, [first, ...rest]) {
    return { name, value, first, rest };
}

// Arrow functions
const arrowFunction = () => "arrow";

const arrowWithParams = (x, y) => x + y;

const arrowWithBody = (items) => {
    const filtered = items.filter(x => x > 0);
    const mapped = filtered.map(x => x * 2);
    return mapped;
};

const arrowReturningObject = (name, value) => ({ name, value });

// Immediately Invoked Function Expression (IIFE)
const iife = (function() {
    const privateVar = "private";
    return {
        getPrivate: () => privateVar
    };
})();

// =============================================================================
// Classes
// =============================================================================

class BaseClass {
    static staticProperty = "static";
    instanceProperty = "instance";
    #privateField = "private";
    
    constructor(value) {
        this.value = value;
    }
    
    get computedValue() {
        return this.value * 2;
    }
    
    set computedValue(newValue) {
        this.value = newValue / 2;
    }
    
    instanceMethod() {
        return this.value;
    }
    
    static staticMethod() {
        return BaseClass.staticProperty;
    }
    
    #privateMethod() {
        return this.#privateField;
    }
    
    async asyncMethod() {
        return await Promise.resolve(this.value);
    }
    
    *generatorMethod() {
        yield this.value;
        yield this.value * 2;
    }
}

class DerivedClass extends BaseClass {
    constructor(value, name) {
        super(value);
        this.name = name;
    }
    
    instanceMethod() {
        return `${this.name}: ${super.instanceMethod()}`;
    }
    
    derivedOnlyMethod() {
        return this.name;
    }
}

// Class expression
const ClassExpression = class {
    constructor(x) {
        this.x = x;
    }
    
    getX() {
        return this.x;
    }
};

// =============================================================================
// Async/Await
// =============================================================================

async function asyncFunction() {
    return "async result";
}

async function asyncWithAwait() {
    const result = await asyncFunction();
    return result;
}

async function asyncWithTryCatch() {
    try {
        const result = await asyncFunction();
        return result;
    } catch (error) {
        console.error(error);
        throw error;
    } finally {
        console.log("cleanup");
    }
}

async function asyncWithMultipleAwaits() {
    const [a, b, c] = await Promise.all([
        asyncFunction(),
        asyncFunction(),
        asyncFunction()
    ]);
    return { a, b, c };
}

// =============================================================================
// Generators
// =============================================================================

function* generatorFunction() {
    yield 1;
    yield 2;
    yield 3;
}

function* generatorWithReturn() {
    yield "first";
    return "done";
}

async function* asyncGenerator() {
    yield await Promise.resolve(1);
    yield await Promise.resolve(2);
}

// =============================================================================
// Higher-order functions and closures
// =============================================================================

function higherOrderFunction(callback) {
    return function(value) {
        return callback(value);
    };
}

function closureExample(multiplier) {
    return function(value) {
        return value * multiplier;
    };
}

function curryFunction(a) {
    return function(b) {
        return function(c) {
            return a + b + c;
        };
    };
}

// =============================================================================
// Object methods and computed properties
// =============================================================================

const objectWithMethods = {
    property: "value",
    
    method() {
        return this.property;
    },
    
    arrowMethod: () => "arrow in object",
    
    async asyncMethod() {
        return await Promise.resolve("async");
    },
    
    *generatorMethod() {
        yield this.property;
    },
    
    get computed() {
        return this.property.toUpperCase();
    },
    
    set computed(value) {
        this.property = value.toLowerCase();
    },
    
    ["computed" + "Key"]: "computed key value",
    
    [Symbol.iterator]: function*() {
        yield this.property;
    }
};

// =============================================================================
// Complex class with all features
// =============================================================================

class ComplexClass extends BaseClass {
    static #privateStaticField = "private static";
    static publicStaticField = "public static";
    
    #privateInstanceField;
    publicInstanceField;
    
    constructor(value, options = {}) {
        super(value);
        this.#privateInstanceField = options.private || "default";
        this.publicInstanceField = options.public || "default";
    }
    
    static get privateStatic() {
        return ComplexClass.#privateStaticField;
    }
    
    static set privateStatic(value) {
        ComplexClass.#privateStaticField = value;
    }
    
    static async staticAsyncMethod() {
        return await Promise.resolve("static async");
    }
    
    static *staticGeneratorMethod() {
        yield "static generator";
    }
    
    get privateField() {
        return this.#privateInstanceField;
    }
    
    set privateField(value) {
        this.#privateInstanceField = value;
    }
    
    #privateMethod() {
        return this.#privateInstanceField;
    }
    
    publicMethod() {
        return this.#privateMethod();
    }
    
    async asyncMethod() {
        const base = await super.asyncMethod();
        return `${base} + complex`;
    }
    
    *generatorMethod() {
        yield* super.generatorMethod();
        yield this.publicInstanceField;
    }
    
    [Symbol.toStringTag]() {
        return "ComplexClass";
    }
}

// =============================================================================
// Module pattern (for testing function extraction)
// =============================================================================

const modulePattern = (function() {
    // Private variables
    let privateCounter = 0;
    
    // Private function
    function privateIncrement() {
        privateCounter++;
    }
    
    // Public API
    return {
        increment: function() {
            privateIncrement();
        },
        
        getCount: function() {
            return privateCounter;
        },
        
        reset: function() {
            privateCounter = 0;
        }
    };
})();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        simpleFunction,
        BaseClass,
        DerivedClass,
        ComplexClass,
        asyncFunction,
        generatorFunction
    };
}
