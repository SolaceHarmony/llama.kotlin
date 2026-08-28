# Implementation Patterns Documentation

This document outlines the key implementation patterns used throughout the llama.kotlin project to ensure consistency, maintainability, and adherence to the DRY (Don't Repeat Yourself) principle.

## Architecture Overview

The llama.kotlin project follows a modular, utility-based architecture that emphasizes:

- **Centralized Utilities**: Common functionality consolidated into reusable utility objects
- **Type Safety**: Leveraging Kotlin's type system for compile-time safety
- **Memory Efficiency**: ByteArray-based storage with optimized access patterns
- **Documentation Standards**: Comprehensive KDoc following Kotlin conventions

## Core Utility Patterns

### 1. GGMLUtilities Object

Located in `src/nativeMain/kotlin/io.github.kotlinmania.llama.ore/GGMLUtilities.kt`

**Purpose**: Centralized utilities for formatting, display, and common operations.

**Key Components**:
- `formatDouble(x: Double)`: Consistent 2-decimal place formatting
- `formatSpeedup(speedup: Double)`: Standardized speedup display (e.g., "2.45x")
- `createSectionHeader()`: Reusable demo section formatting
- `createBulletPoint()`: Consistent bullet point formatting

**Usage Pattern**:
```kotlin
// Instead of custom formatting in each file
val speedup = 2.456789
println("Speedup: ${kotlin.math.round(speedup * 100.0) / 100.0}x")

// Use centralized utility
println("Speedup: ${GGMLUtilities.formatSpeedup(speedup)}")
```

### 2. ByteArrayExtensions Object

Located in `src/nativeMain/kotlin/io.github.kotlinmania.llama.ore/GGMLUtilities.kt`

**Purpose**: Centralized little-endian ByteArray operations.

**Key Components**:
- `getIntLe(offset: Int)`: Read 32-bit integer
- `getFloatLe(offset: Int)`: Read 32-bit float
- `setIntLe(offset: Int, value: Int)`: Write 32-bit integer
- `setFloatLe(offset: Int, value: Float)`: Write 32-bit float
- Similar methods for Short and Long types

**Usage Pattern**:
```kotlin
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe

// Consistent little-endian operations across all files
val value = buffer.getFloatLe(offset)
buffer.setFloatLe(offset, newValue)
```

### 3. GGMLTensorUtils Object

Located in `src/nativeMain/kotlin/io.github.kotlinmania.llama.ore/GGMLTensorUtils.kt`

**Purpose**: Tensor-specific utility functions and calculations.

**Key Components**:
- `calculateTotalSize(ne: LongArray)`: Tensor element count calculation
- `calculateContiguousStrides()`: Memory layout stride calculation
- Validation and bounds checking utilities

**Usage Pattern**:
```kotlin
// Instead of inline calculations
var totalSize = 1L
for (i in 0 until GGML_MAX_DIMS) {
    if (i < ne.size && ne[i] > 0) totalSize *= ne[i]
}

// Use centralized utility
val totalSize = GGMLTensorUtils.calculateTotalSize(ne)
```

### 4. DemoTextUtilities Object

Located in `src/nativeMain/kotlin/io.github.kotlinmania.llama.ore/GGMLUtilities.kt`

**Purpose**: Reusable components for demo output and documentation.

**Key Components**:
- `createFeatureSection()`: Complete feature section with header and bullets
- `FeatureLists`: Predefined feature lists for common categories
- Consistent demo formatting across examples

**Usage Pattern**:
```kotlin
// Instead of repetitive appendLine calls
appendLine("✅ All K-Quantization formats implemented:")
appendLine("  • Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K")
appendLine("  • Quantization and dequantization functions")

// Use centralized utility
append(DemoTextUtilities.createFeatureSection(
    sectionTitle = "K-Quantization Support",
    emoji = "📊",
    features = DemoTextUtilities.FeatureLists.kQuantizationFeatures
))
```

## Documentation Standards

### KDoc Guidelines

All public APIs should include comprehensive KDoc documentation following these patterns:

```kotlin
/**
 * Brief description of the function's purpose.
 * 
 * Detailed explanation of the function's behavior, including:
 * - Algorithm description
 * - Performance characteristics
 * - Usage context
 * 
 * @param paramName Description of parameter including type constraints
 * @param anotherParam Description with valid value ranges or formats
 * @return Description of return value and its format
 * @throws ExceptionType Conditions under which exceptions are thrown
 * @see RelatedClass For related functionality
 * @since 0.1.0 (if applicable)
 */
```

### Code Organization Patterns

1. **File Structure**:
   - Package declaration
   - Imports (grouped: stdlib, project, third-party)
   - File-level KDoc
   - Constants
   - Main implementation
   - Helper functions

2. **Class Structure**:
   - Class KDoc
   - Primary constructor
   - Properties
   - Public methods
   - Private methods
   - Companion object (if needed)

3. **Function Organization**:
   - Public functions first
   - Internal functions second
   - Private functions last
   - Group related functions together

## Memory Management Patterns

### ByteArray-Based Storage

All tensor data uses ByteArray storage with typed accessors:

```kotlin
// Pattern for tensor data access
class GGMLTensor {
    fun getFloat(allocator: GGMLGraphAllocator, vararg indices: Int): Float {
        val buffer = allocator.buffers[bufferId] ?: throw IllegalStateException("Buffer not found")
        val offset = calculateOffset(indices)
        return buffer.getFloatLe(offset)
    }
}
```

### Destination-Based Operations

All compute operations write results directly to pre-allocated tensors:

```kotlin
// Pattern for compute operations
fun computeAdd(allocator: GGMLGraphAllocator, context: GGMLContext,
               src0: GGMLTensor, src1: GGMLTensor, dst: GGMLTensor) {
    // Validation
    require(src0.type == src1.type) { "Type mismatch" }
    
    // Direct computation into destination
    for (i in 0 until totalElements) {
        val result = src0.getFloat(allocator, i) + src1.getFloat(allocator, i)
        dst.setFloat(allocator, result, i)
    }
}
```

## Error Handling Patterns

### Validation and Error Messages

```kotlin
// Pattern for parameter validation
require(condition) { "Descriptive error message with context" }
check(state) { "State validation message" }

// Pattern for detailed error context
require(offset + 4 <= size) { 
    "Insufficient buffer size: need ${offset + 4} bytes, have $size" 
}
```

### Exception Types

- `IllegalArgumentException`: Invalid parameters
- `IllegalStateException`: Invalid object state
- `IndexOutOfBoundsException`: Array/buffer access violations
- `NotImplementedError`: Placeholder implementations

## Testing Patterns

### Test Organization

```kotlin
class FeatureTest {
    @Test
    fun testBasicFunctionality() {
        // Arrange
        val input = createTestInput()
        
        // Act
        val result = performOperation(input)
        
        // Assert
        assertEquals(expected, result)
        assertTrue(validationCondition)
    }
    
    @Test
    fun testErrorConditions() {
        assertFailsWith<IllegalArgumentException> {
            performOperationWithInvalidInput()
        }
    }
}
```

### Test Utilities

Use `GGMLTestUtils` for common test patterns:
- `createStandardTestTensor()`: Standard tensor creation
- `validateTensorDimensions()`: Dimension validation
- `TensorComparison.tensorsStructurallyEqual()`: Structural comparison

## Performance Patterns

### Benchmarking

Use centralized benchmarking utilities:

```kotlin
val start = TimeSource.Monotonic.markNow()
performOperation()
val duration = start.elapsedNow()

// Use centralized formatting
println("Duration: ${GGMLUtilities.formatDouble(duration.inWholeMilliseconds.toDouble())}ms")
```

### Memory Optimization

- Prefer in-place operations when possible
- Use destination-based patterns to minimize allocations
- Leverage ByteArray pooling through allocators

## Migration and Compatibility

### Deprecation Pattern

When replacing functionality:

```kotlin
@Deprecated(
    message = "Use NewFunction() for better performance",
    replaceWith = ReplaceWith("NewFunction(param)")
)
fun oldFunction(param: Type): ReturnType {
    return NewFunction(param)
}
```

### Version Compatibility

- Mark breaking changes clearly in documentation
- Provide migration guides for major pattern changes
- Use semantic versioning for API changes

## Best Practices Summary

1. **Always use centralized utilities** instead of implementing common functionality inline
2. **Follow KDoc standards** for all public APIs
3. **Use destination-based patterns** for compute operations
4. **Leverage type safety** with proper validation and error messages
5. **Maintain consistency** in formatting, naming, and structure
6. **Document patterns** when establishing new architectural approaches
7. **Test thoroughly** using established test utilities and patterns

This documentation should be updated as new patterns emerge and existing patterns evolve.
