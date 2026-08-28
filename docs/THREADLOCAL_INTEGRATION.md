# ThreadLocal Integration Guide

This document describes the thread-local functionality added to llama.kotlin using `threadlocal-kotlin:0.3.1`.

## Overview

The llama.kotlin project now includes three thread-local wrappers that provide per-thread isolated state:

1. **GGMLThreadLocalContext** - Per-thread compute parameters for GGML operations
2. **ThreadLocalCFloatTrace** - Per-thread float operation tracing
3. **ThreadLocalKMalloc** - Per-thread memory allocation (for future use)

## Why ThreadLocal?

Previously, the codebase had to remove ThreadLocal usage because standard Java's `ThreadLocal` is not available in Kotlin Multiplatform's common code. The `threadlocal-kotlin:0.3.1` library solves this by providing a multiplatform-compatible ThreadLocal implementation.

### Supported Platforms

threadlocal-kotlin works across all llama.kotlin targets:
- JVM (Java 8+)
- macOS arm64
- Linux x64
- Windows mingw-x64
- JS (browser + Node.js)
- Wasm-JS
- Android (API 24+)

## Dependency

The threadlocal-kotlin library has been added to `build.gradle.kts`:

```kotlin
commonMain.dependencies {
    implementation("io.github.kotlinmania.llama.hreadlocal-kotlin:0.3.1")
}
```

## Components

### 1. GGMLThreadLocalContext

Located: `src/commonMain/kotlin/io.github.kotlinmania.llama.ore/GGMLThreadLocalContext.kt`

Provides thread-local storage for GGML compute parameters, eliminating the need to pass `GGMLComputeParams` through deep call stacks.

**Key Features:**
- Each thread has isolated compute parameters
- Automatic cleanup with `withParams` scope
- Helper methods to get thread ID and thread count
- No contention between threads

**Usage Example:**

```kotlin
// Set params for current thread
GGMLThreadLocalContext.setCurrentParams(
    GGMLComputeParams(ith = 0, nth = 4)
)

// Access from nested functions
val threadId = GGMLThreadLocalContext.getCurrentThreadId()

// Clean up
GGMLThreadLocalContext.clearCurrentParams()

// Or use scoped access
GGMLThreadLocalContext.withParams(params) {
    // Params available here
    doComputation()
}
// Params automatically cleared
```

**API:**
- `setCurrentParams(params)` - Set params for current thread
- `getCurrentParams()` - Get params or null
- `getCurrentParamsOr(default)` - Get params or default
- `requireCurrentParams()` - Get params or throw
- `clearCurrentParams()` - Remove params
- `withParams(params, block)` - Scoped params
- `getCurrentThreadId()` - Get thread ID
- `getTotalThreads()` - Get thread count
- `hasCurrentParams()` - Check if set

### 2. ThreadLocalCFloatTrace

Located: `external/klangnative/src/commonMain/kotlin/io.github.kotlinmania.llama.lang/bitwise/ThreadLocalCFloatTrace.kt`

Provides thread-local tracing of floating-point operations without contention.

**Key Features:**
- Each thread has independent trace buffer
- Enable/disable per thread
- Scoped tracing with automatic cleanup
- Statistics on traced operations

**Usage Example:**

```kotlin
// Enable tracing for current thread
ThreadLocalCFloatTrace.enable()

// Log operations
ThreadLocalCFloatTrace.log("+", 1.0f, 2.0f, 3.0f)

// Get traces
val entries = ThreadLocalCFloatTrace.getEntries()

// Clean up
ThreadLocalCFloatTrace.reset()
ThreadLocalCFloatTrace.disable()

// Or use scoped tracing
val (result, traces) = ThreadLocalCFloatTrace.withTracing {
    // Code to trace
    doFloatMath()
}
// Tracing automatically disabled and reset
```

**API:**
- `enable()` - Enable for current thread
- `disable()` - Disable for current thread
- `isEnabled()` - Check if enabled
- `log(operation, lhs, rhs, result)` - Log operation
- `getEntries()` - Get all traces
- `reset()` - Clear trace buffer
- `withTracing(block)` - Scoped tracing
- `getStats()` - Get statistics

### 3. ThreadLocalKMalloc

Located: `external/klangnative/src/commonMain/kotlin/io.github.kotlinmania.llama.lang/mem/ThreadLocalKMalloc.kt`

Provides thread-local memory allocation arenas. **Note:** This is currently a wrapper around the global `KMalloc` object and will be fully functional once `KMalloc` is refactored into instances.

**Key Features:**
- Per-thread allocation arenas (conceptual)
- No cross-thread pointer sharing
- Future: true per-thread heaps

**Usage Example:**

```kotlin
// Allocate in current thread
val ptr = ThreadLocalKMalloc.malloc(256)

// Use memory
// ...

// Free in same thread
ThreadLocalKMalloc.free(ptr)

// Get stats
val stats = ThreadLocalKMalloc.stats()
println("Allocated: ${stats.currentUsed} bytes")
```

**API:**
- `malloc(size)` - Allocate memory
- `calloc(count, size)` - Allocate zeroed memory
- `free(ptr)` - Free memory
- `realloc(ptr, newSize)` - Resize allocation
- `stats()` - Get arena statistics
- `setDefaultArenaSize(size)` - Configure arena size

## Testing

Tests have been added to verify the thread-local functionality:

- `src/commonTest/kotlin/io.github.kotlinmania.llama.ore/GGMLThreadLocalContextTest.kt` - Tests for GGML context
- `external/klangnative/src/commonTest/kotlin/io.github.kotlinmania.llama.lang/bitwise/ThreadLocalCFloatTraceTest.kt` - Tests for float tracing

Run tests with:
```bash
./gradlew jvmTest
./gradlew allTests  # All platforms
```

## Migration Guide

### For GGML Operations

**Before (explicit parameter passing):**
```kotlin
fun computeKernel(params: GGMLComputeParams, tensor: GGMLTensor) {
    val ith = params.ith
    // ...
    helperFunction(params, data)
}

fun helperFunction(params: GGMLComputeParams, data: Data) {
    // Use params.ith
}
```

**After (thread-local access):**
```kotlin
fun computeKernel(tensor: GGMLTensor) {
    val ith = GGMLThreadLocalContext.getCurrentThreadId()
    // ...
    helperFunction(data)
}

fun helperFunction(data: Data) {
    val params = GGMLThreadLocalContext.requireCurrentParams()
    // Use params.ith
}
```

### For Float Tracing

**Before (global CFloatTrace):**
```kotlin
// Not thread-safe
CFloatTrace.enable()
// All threads write to same buffer
```

**After (thread-local tracing):**
```kotlin
// Thread-safe
ThreadLocalCFloatTrace.enable()
// Each thread has independent buffer
```

## Implementation Notes

### How ThreadLocal Works

The `threadlocal-kotlin` library provides true thread-local storage:

- **JVM**: Uses `java.lang.ThreadLocal`
- **Native**: Uses `pthread_key_t` (POSIX) or Windows TLS
- **JS**: Uses `Map` keyed by coroutine context
- **Wasm**: Uses linear memory slots

### Memory Considerations

- Each thread allocates its own thread-local storage
- For `GGMLComputeParams`: ~64 bytes per thread (negligible)
- For `ThreadLocalCFloatTrace`: Grows with trace buffer (unbounded until reset)
- For `ThreadLocalKMalloc`: Future per-thread arena (configurable size)

### Thread Safety

All three components are inherently thread-safe because each thread has isolated state. No locks or synchronization needed.

## Future Enhancements

1. **KMalloc Instance Refactoring**
   - Make `KMalloc` instance-based instead of singleton
   - Enable true per-thread heaps in `ThreadLocalKMalloc`

2. **Context Stacking**
   - Support nested contexts with stack-based scoping
   - Useful for recursive computations

3. **Automatic Context Propagation**
   - Propagate context to child threads/coroutines
   - Simplify multi-threaded code

4. **Debug Mode**
   - Track context lifetime and detect leaks
   - Warn on cross-thread access attempts

## References

- [threadlocal-kotlin GitHub](https://github.com/KotlinMania/threadlocal-kotlin)
- [threadlocal-kotlin on Maven Central](https://central.sonatype.com/artifact/io.github.kotlinmania.llama.hreadlocal-kotlin)
- KDoc in source files for detailed API documentation

## Related Issues

This integration addresses:
- Thread-safety concerns with `KMalloc` (mentioned in `KMalloc.kt:104`)
- Multi-threaded tracing needs (mentioned in `CFloatTrace.kt:272`)
- Parameter passing verbosity in GGML operations

---

**Version**: 0.3.1
**Author**: llama.kotlin team
**Date**: May 2026
