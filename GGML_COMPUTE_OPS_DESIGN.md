# GGML Compute Operations Design Document

## Overview

This document outlines the design for implementing the actual computation functionality for tensor operations in the Kotlin Native port of llama.cpp. It focuses on the implementation details for the GGMLComputeOps.kt file, which contains the core computation functions for tensor operations.

## ⚠️ Major Architectural Update

**This design document has been updated to reflect a major architectural refactor completed in the codebase.** All compute operations have been transformed from memory-allocating functions to destination-based operations that write directly into pre-allocated tensors managed by the graph allocator.

### Key Changes:
- **Function Signatures**: All functions now require a destination tensor parameter
- **Memory Management**: Operations write directly to allocator-managed buffers
- **Performance**: Eliminated redundant memory allocations
- **Architecture**: Aligned with GGML patterns for memory reuse and optimization

See `COMPUTE_OPERATIONS_REFACTOR_SUMMARY.md` for complete details of the implementation.

**Note**: While some legacy function examples may remain in this document for reference, all current implementations follow the destination-based pattern shown in the examples above and detailed in the refactor summary.

## Purpose

The GGMLComputeOps.kt file provides the actual computation functionality for tensor operations using a **destination-based architecture**. Operations write results directly into pre-allocated destination tensors rather than creating new tensors, enabling efficient memory management and graph optimization.

## Function Specifications (Updated Architecture)

### Destination-Based Operation Pattern

All compute operations now follow this pattern:

```kotlin
fun computeOperation(
    graphAllocator: GGMLGraphAllocator, 
    context: GGMLContext, 
    inputTensors: ..., 
    dst: GGMLTensor
) {
    // 1. Validate destination tensor dimensions and type
    // 2. Perform computation using input tensor accessors
    // 3. Write results directly to destination using allocator-managed memory
}
```

### Element-wise Operations

#### computeAdd (Updated)

```kotlin
/**
 * Adds two tensors element-wise, writing results to destination tensor.
 *
 * @param graphAllocator The graph allocator managing tensor memory
 * @param context The computation context
 * @param a First input tensor
 * @param b Second input tensor  
 * @param dst Pre-allocated destination tensor for results
 */
fun computeAdd(
    graphAllocator: GGMLGraphAllocator, 
    context: GGMLContext, 
    a: GGMLTensor, 
    b: GGMLTensor, 
    dst: GGMLTensor
) {
    // Validate destination tensor
    require(dst.ne.contentEquals(a.ne)) { "Destination dimensions must match input" }
    require(dst.type == a.type) { "Destination type must match input" }

    val totalSize = calculateTotalSize(a.ne)
    
    // Perform the addition based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            // Use tensor accessors to write to allocator-managed buffer
            for (i in 0 until totalSize) {
                val aVal = a.getFloat(graphAllocator, i)
                val bVal = b.getFloat(graphAllocator, i) 
                dst.setFloat(graphAllocator, aVal + bVal, i)
            }
        }
        GGMLType.I32 -> {
            for (i in 0 until totalSize) {
                val aVal = a.getInt(graphAllocator, i)
                val bVal = b.getInt(graphAllocator, i)
                dst.setInt(graphAllocator, aVal + bVal, i) 
            }
        }
        // Handle other types...
    }
}
```

#### computeMul (Updated)

```kotlin
/**
 * Multiplies two tensors element-wise, writing results to destination tensor.
 *
 * @param graphAllocator The graph allocator managing tensor memory
 * @param context The computation context
 * @param a First input tensor
 * @param b Second input tensor  
 * @param dst Pre-allocated destination tensor for results
 */
fun computeMul(
    graphAllocator: GGMLGraphAllocator, 
    context: GGMLContext, 
    a: GGMLTensor, 
    b: GGMLTensor, 
    dst: GGMLTensor
) {
    // Validate destination tensor
    require(dst.ne.contentEquals(a.ne)) { "Destination dimensions must match input" }
    require(dst.type == a.type) { "Destination type must match input" }

    val totalSize = calculateTotalSize(a.ne)
    
    // Perform multiplication based on tensor type
    when (a.type) {
        GGMLType.F32 -> {
            for (i in 0 until totalSize) {
                val aVal = a.getFloat(graphAllocator, i)
                val bVal = b.getFloat(graphAllocator, i) 
                dst.setFloat(graphAllocator, aVal * bVal, i)
            }
        }
        GGMLType.I32 -> {
            for (i in 0 until totalSize) {
                val aVal = a.getInt(graphAllocator, i)
                val bVal = b.getInt(graphAllocator, i)
                dst.setInt(graphAllocator, aVal * bVal, i)
            }
        }
        // Handle other types...
    }
}
```

### Matrix Operations

#### computeMatMul (Updated)

```kotlin
/**
 * Performs matrix multiplication of two tensors.
 *
 * @param context The GGML context
 * @param a The first tensor (m x n)
 * @param b The second tensor (n x p)
 * @return The result tensor (m x p)
 */
fun computeMatMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    // Check that the tensors have compatible dimensions for matrix multiplication
    // a: m x n, b: n x p, result: m x p
    val m = a.ne[0]
    val n = a.ne[1]
    val p = b.ne[1]

    if (b.ne[0] != n) {
        throw IllegalArgumentException("Incompatible dimensions for matrix multiplication")
    }

    // Create a new tensor for the result
    val result = GGMLTensor(type = a.type)
    result.ne[0] = m
    result.ne[1] = p
    for (i in 2 until GGML_MAX_DIMS) {
        result.ne[i] = 1
    }

    // Set strides based on the data type
    val typeSize = when (a.type) {
        GGMLType.F32 -> 4u
        GGMLType.F16 -> 2u
        GGMLType.I8 -> 1u
        GGMLType.I16 -> 2u
        GGMLType.I32 -> 4u
        GGMLType.I64 -> 8u
        else -> 1u // Default for quantized types
    }

    result.nb[0] = typeSize
    result.nb[1] = result.nb[0] * result.ne[0].toULong()
    for (i in 2 until GGML_MAX_DIMS) {
        result.nb[i] = result.nb[i-1] * result.ne[i-1].toULong()
    }

    // Calculate total size
    val totalSize = (m * p).toInt()

    // Perform the matrix multiplication based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val resultData = FloatArray(totalSize)

            for (i in 0 until m.toInt()) {
                for (j in 0 until p.toInt()) {
                    var sum = 0.0f
                    for (k in 0 until n.toInt()) {
                        sum += aData[i * n.toInt() + k] * bData[k * p.toInt() + j]
                    }
                    resultData[i * p.toInt() + j] = sum
                }
            }

            result.data = resultData
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}
```

### Activation Functions

#### computeRelu

```kotlin
/**
 * Applies the ReLU activation function to a tensor.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor
 */
fun computeRelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Apply the ReLU function based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val aData = a.data as FloatArray
            val resultData = FloatArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = if (aData[i] > 0.0f) aData[i] else 0.0f
            }

            result.data = resultData
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}
```

#### computeGelu

```kotlin
/**
 * Applies the GELU activation function to a tensor.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor
 */
fun computeGelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Apply the GELU function based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val aData = a.data as FloatArray
            val resultData = FloatArray(totalSize)

            for (i in 0 until totalSize) {
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                val x = aData[i]
                resultData[i] = x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x)))
            }

            result.data = resultData
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}
```

## Integration with Existing Code (Updated)

The functions in GGMLComputeOps.kt are called from the corresponding functions in GGMLOps.kt with the destination-based architecture. The calling functions must pre-allocate destination tensors using the graph allocator.

```kotlin
// In GGMLOps.kt (Updated approach)
fun add(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    // Set up the operation in the computation graph
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.ADD
    result.src[0] = a
    result.src[1] = b
    
    // Pre-allocate destination tensor using graph allocator
    val dst = context.graphAllocator.allocateTensor(a.type, a.ne)

    // If immediate computation is required, call the compute function
    if (context.computeImmediately) {
        computeAdd(context.graphAllocator, context, a, b, dst)
        return dst
    }

    return result
}
```

**Key Changes in Integration:**
- Destination tensor must be pre-allocated using `context.graphAllocator.allocateTensor()`
- Compute functions called with additional `graphAllocator` and `dst` parameters
- No return value from compute functions - results written to destination
- Graph allocator manages all memory allocation and reuse

## Optimization Strategies

### SIMD Vectorization

For CPU backends, SIMD (Single Instruction, Multiple Data) vectorization can be used to accelerate tensor operations. This involves:
1. Using SIMD intrinsics for the target architecture (ARM NEON, x86 AVX, etc.)
2. Processing multiple elements in parallel

Example for F32 addition with SIMD:

```kotlin
// Pseudocode for SIMD-accelerated addition
fun computeAddSimd(a: FloatArray, b: FloatArray, result: FloatArray, size: Int) {
    // Process 4 elements at a time using SIMD
    val simdSize = size / 4 * 4
    for (i in 0 until simdSize step 4) {
        // Load 4 elements from a and b
        val aVec = loadFloat4(a, i)
        val bVec = loadFloat4(b, i)

        // Add the vectors
        val resultVec = addFloat4(aVec, bVec)

        // Store the result
        storeFloat4(result, i, resultVec)
    }

    // Process remaining elements
    for (i in simdSize until size) {
        result[i] = a[i] + b[i]
    }
}
```

### Multi-threading

For large tensors, multi-threading can be used to parallelize tensor operations. This involves:
1. Dividing the tensor into chunks
2. Processing each chunk in a separate thread
3. Synchronizing the results

Example for F32 addition with multi-threading:

```kotlin
// Pseudocode for multi-threaded addition
fun computeAddMultiThreaded(a: FloatArray, b: FloatArray, result: FloatArray, size: Int, numThreads: Int) {
    val chunkSize = size / numThreads

    // Create and start threads
    val threads = Array(numThreads) { threadId ->
        Thread {
            val start = threadId * chunkSize
            val end = if (threadId == numThreads - 1) size else (threadId + 1) * chunkSize

            for (i in start until end) {
                result[i] = a[i] + b[i]
            }
        }
    }

    // Start all threads
    threads.forEach { it.start() }

    // Wait for all threads to complete
    threads.forEach { it.join() }
}
```

### Memory Access Patterns

Optimizing memory access patterns can improve cache utilization and reduce memory bandwidth requirements. This involves:
1. Tiling matrix multiplication to improve cache locality
2. Using memory-efficient algorithms for large tensors

Example for tiled matrix multiplication:

```kotlin
// Pseudocode for tiled matrix multiplication
fun computeMatMulTiled(a: FloatArray, b: FloatArray, result: FloatArray, m: Int, n: Int, p: Int) {
    val tileSize = 32 // Choose an appropriate tile size based on cache size

    // Zero the result array
    for (i in 0 until m * p) {
        result[i] = 0.0f
    }

    // Tile the computation
    for (i0 in 0 until m step tileSize) {
        val iEnd = minOf(i0 + tileSize, m)
        for (j0 in 0 until p step tileSize) {
            val jEnd = minOf(j0 + tileSize, p)
            for (k0 in 0 until n step tileSize) {
                val kEnd = minOf(k0 + tileSize, n)

                // Compute on the current tile
                for (i in i0 until iEnd) {
                    for (j in j0 until jEnd) {
                        var sum = result[i * p + j]
                        for (k in k0 until kEnd) {
                            sum += a[i * n + k] * b[k * p + j]
                        }
                        result[i * p + j] = sum
                    }
                }
            }
        }
    }
}
```

## Conclusion

This design document outlines the implementation details for the GGMLComputeOps.kt file, which will contain the actual computation functionality for tensor operations in the Kotlin Native port of llama.cpp. It provides a roadmap for future implementation without risking breaking the existing code.

The implementation focuses on the F32 data type as a starting point, with placeholders for other data types to be implemented later. It also includes optimization strategies for SIMD vectorization, multi-threading, and memory access patterns.
