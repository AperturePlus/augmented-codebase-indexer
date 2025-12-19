// Package fixtures provides complex Go code for AST parser testing.
//
// This file contains various Go constructs to test the AST parser's
// ability to handle real-world code patterns.
package fixtures

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

// =============================================================================
// Basic structs and methods
// =============================================================================

// SimpleStruct is a basic struct with fields.
type SimpleStruct struct {
	Name  string
	Value int
}

// GetName returns the name field.
func (s SimpleStruct) GetName() string {
	return s.Name
}

// SetName sets the name field (pointer receiver).
func (s *SimpleStruct) SetName(name string) {
	s.Name = name
}

// =============================================================================
// Embedded structs and interfaces
// =============================================================================

// BaseInterface defines basic operations.
type BaseInterface interface {
	Process(event string) error
	GetID() string
}

// ExtendedInterface embeds BaseInterface and adds more methods.
type ExtendedInterface interface {
	BaseInterface
	Validate() bool
	io.Reader
	io.Writer
}

// EmbeddedStruct demonstrates struct embedding.
type EmbeddedStruct struct {
	SimpleStruct    // Embedded struct
	sync.Mutex      // Embedded from standard library
	AdditionalField int
}

// Process implements BaseInterface.
func (e *EmbeddedStruct) Process(event string) error {
	e.Lock()
	defer e.Unlock()
	e.AdditionalField++
	return nil
}

// GetID implements BaseInterface.
func (e *EmbeddedStruct) GetID() string {
	return fmt.Sprintf("%s-%d", e.Name, e.Value)
}

// =============================================================================
// Generic types and functions (Go 1.18+)
// =============================================================================

// GenericStruct is a generic struct with type parameter.
type GenericStruct[T any] struct {
	Value T
	Items []T
}

// GetValue returns the value.
func (g GenericStruct[T]) GetValue() T {
	return g.Value
}

// SetValue sets the value.
func (g *GenericStruct[T]) SetValue(v T) {
	g.Value = v
}

// AddItem adds an item to the slice.
func (g *GenericStruct[T]) AddItem(item T) {
	g.Items = append(g.Items, item)
}

// GenericMap is a generic map wrapper.
type GenericMap[K comparable, V any] struct {
	data map[K]V
	mu   sync.RWMutex
}

// Get retrieves a value from the map.
func (m *GenericMap[K, V]) Get(key K) (V, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.data[key]
	return v, ok
}

// Set stores a value in the map.
func (m *GenericMap[K, V]) Set(key K, value V) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.data == nil {
		m.data = make(map[K]V)
	}
	m.data[key] = value
}

// GenericFunction is a generic function.
func GenericFunction[T any](items []T, predicate func(T) bool) []T {
	result := make([]T, 0)
	for _, item := range items {
		if predicate(item) {
			result = append(result, item)
		}
	}
	return result
}

// GenericConstrainedFunction uses type constraints.
func GenericConstrainedFunction[T comparable](a, b T) bool {
	return a == b
}

// =============================================================================
// Complex struct with multiple methods
// =============================================================================

// ComplexStruct demonstrates various method patterns.
type ComplexStruct struct {
	id       string
	data     map[string]interface{}
	mu       sync.RWMutex
	ctx      context.Context
	cancel   context.CancelFunc
	handlers []func(string) error
}

// NewComplexStruct creates a new ComplexStruct.
func NewComplexStruct(id string) *ComplexStruct {
	ctx, cancel := context.WithCancel(context.Background())
	return &ComplexStruct{
		id:       id,
		data:     make(map[string]interface{}),
		ctx:      ctx,
		cancel:   cancel,
		handlers: make([]func(string) error, 0),
	}
}

// GetID returns the ID.
func (c *ComplexStruct) GetID() string {
	return c.id
}

// SetData sets a key-value pair.
func (c *ComplexStruct) SetData(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[key] = value
}

// GetData retrieves a value by key.
func (c *ComplexStruct) GetData(key string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.data[key]
	return v, ok
}

// AddHandler adds an event handler.
func (c *ComplexStruct) AddHandler(h func(string) error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.handlers = append(c.handlers, h)
}

// Process processes an event through all handlers.
func (c *ComplexStruct) Process(event string) error {
	c.mu.RLock()
	handlers := make([]func(string) error, len(c.handlers))
	copy(handlers, c.handlers)
	c.mu.RUnlock()

	for _, h := range handlers {
		select {
		case <-c.ctx.Done():
			return c.ctx.Err()
		default:
			if err := h(event); err != nil {
				return fmt.Errorf("handler error: %w", err)
			}
		}
	}
	return nil
}

// Close cancels the context and cleans up.
func (c *ComplexStruct) Close() error {
	c.cancel()
	return nil
}

// =============================================================================
// Interface implementations
// =============================================================================

// Ensure ComplexStruct implements interfaces.
var (
	_ BaseInterface = (*ComplexStruct)(nil)
	_ io.Closer     = (*ComplexStruct)(nil)
)

// =============================================================================
// Error types
// =============================================================================

// CustomError is a custom error type.
type CustomError struct {
	Code    int
	Message string
	Cause   error
}

// Error implements the error interface.
func (e *CustomError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("[%d] %s: %v", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("[%d] %s", e.Code, e.Message)
}

// Unwrap returns the underlying error.
func (e *CustomError) Unwrap() error {
	return e.Cause
}

// Is checks if the error matches a target.
func (e *CustomError) Is(target error) bool {
	t, ok := target.(*CustomError)
	if !ok {
		return false
	}
	return e.Code == t.Code
}

// =============================================================================
// Functional options pattern
// =============================================================================

// Option is a functional option for configuring Server.
type Option func(*Server)

// Server is configured using functional options.
type Server struct {
	host    string
	port    int
	timeout int
	logger  io.Writer
}

// WithHost sets the host.
func WithHost(host string) Option {
	return func(s *Server) {
		s.host = host
	}
}

// WithPort sets the port.
func WithPort(port int) Option {
	return func(s *Server) {
		s.port = port
	}
}

// WithTimeout sets the timeout.
func WithTimeout(timeout int) Option {
	return func(s *Server) {
		s.timeout = timeout
	}
}

// WithLogger sets the logger.
func WithLogger(w io.Writer) Option {
	return func(s *Server) {
		s.logger = w
	}
}

// NewServer creates a new Server with options.
func NewServer(opts ...Option) *Server {
	s := &Server{
		host:    "localhost",
		port:    8080,
		timeout: 30,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Start starts the server.
func (s *Server) Start() error {
	if s.port <= 0 {
		return errors.New("invalid port")
	}
	return nil
}

// =============================================================================
// Standalone functions
// =============================================================================

// SimpleFunction is a simple function.
func SimpleFunction() string {
	return "simple"
}

// FunctionWithParams has multiple parameters.
func FunctionWithParams(a int, b string, c bool) (int, error) {
	if !c {
		return 0, errors.New("c is false")
	}
	return a + len(b), nil
}

// FunctionWithVariadic has variadic parameters.
func FunctionWithVariadic(prefix string, values ...int) []string {
	result := make([]string, len(values))
	for i, v := range values {
		result[i] = fmt.Sprintf("%s%d", prefix, v)
	}
	return result
}

// FunctionWithNamedReturns uses named return values.
func FunctionWithNamedReturns(input string) (result string, err error) {
	if input == "" {
		err = errors.New("empty input")
		return
	}
	result = "processed: " + input
	return
}

// FunctionWithDefer demonstrates defer usage.
func FunctionWithDefer(r io.ReadCloser) ([]byte, error) {
	defer r.Close()
	return io.ReadAll(r)
}

// FunctionWithPanic demonstrates panic and recover.
func FunctionWithPanic() (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("recovered: %v", r)
		}
	}()
	panic("intentional panic")
}

// FunctionWithGoroutine demonstrates goroutine usage.
func FunctionWithGoroutine(ch chan<- int, values []int) {
	var wg sync.WaitGroup
	for _, v := range values {
		wg.Add(1)
		go func(val int) {
			defer wg.Done()
			ch <- val * 2
		}(v)
	}
	go func() {
		wg.Wait()
		close(ch)
	}()
}

// FunctionWithSelect demonstrates select statement.
func FunctionWithSelect(ctx context.Context, ch <-chan int) (int, error) {
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	case v, ok := <-ch:
		if !ok {
			return 0, errors.New("channel closed")
		}
		return v, nil
	}
}
