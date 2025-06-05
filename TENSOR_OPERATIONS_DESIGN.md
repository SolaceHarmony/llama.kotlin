# Tensor Operations Design Document

## Overview

This document outlines the design for implementing tensor operations in the Kotlin Native port of llama.cpp. It focuses on the actual computation functionality for tensor operations, which is one of the next steps identified in the KOTLIN_PORT_STATUS.md file.

## Tensor Data Structure

The tensor data structure is defined in GGMLTypes.kt as follows:

```kotlin
class GGMLTensor(
    var type: GGMLType = GGMLType.F32,
    var buffer: Any? = null,
    var ne: LongArray = LongArray(GGML_MAX_DIMS) { 0L },
    var nb: ULongArray = ULongArray(GGML_MAX_DIMS) { 0u },
    var op: GGMLOp = GGMLOp.NONE,
    var opParams: IntArray = IntArray(GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { 0 },
    var flags: Int = 0,
    var grad: GGMLTensor? = null,
    var src: Array<GGMLTensor?> = Array(GGML_MAX_SRC) { null },
    var viewSrc: GGMLTensor? = null,
    var viewOffs: ULong = 0u,
    var data: Any? = null,
    var name: String = ""
)
```

The key properties for tensor operations are:
- `type`: The data type of the tensor (F32, F16, quantized types, etc.)
- `ne`: The dimensions of the tensor (number of elements in each dimension)
- `nb`: The strides of the tensor (number of bytes between elements in each dimension)
- `data`: The actual data of the tensor, which can be of different types depending on `type`

## Tensor Creation

### Approach

Tensor creation involves:
1. Allocating a new GGMLTensor object
2. Setting the dimensions and strides
3. Allocating memory for the tensor data
4. Initializing the tensor data (optional)

### Pseudocode

```
function createTensor(context, type, dimensions):
    tensor = new GGMLTensor(type = type)

    // Set dimensions
    for i in 0 to dimensions.length - 1:
        tensor.ne[i] = dimensions[i]

    // Set strides based on the data type
    typeSize = getSizeForType(type)
    tensor.nb[0] = typeSize
    for i in 1 to GGML_MAX_DIMS - 1:
        tensor.nb[i] = tensor.nb[i-1] * tensor.ne[i-1]

    // Allocate memory
    if context.memBuffer != null and not context.noAlloc:
        totalSize = calculateTotalSize(tensor.ne)
        tensor.data = allocateMemory(type, totalSize)

    return tensor
```

## Element-wise Operations

### Approach

Element-wise operations (add, multiply, etc.) involve:
1. Checking that the tensors have compatible dimensions
2. Creating a new tensor for the result (or using an existing one)
3. Performing the operation element by element

### Pseudocode for Addition

```
function add(context, a, b):
    // Check that the tensors have compatible dimensions
    for i in 0 to GGML_MAX_DIMS - 1:
        if a.ne[i] != b.ne[i]:
            throw IncompatibleDimensionsException

    // Create a new tensor for the result
    result = createTensor(context, a.type, a.ne)

    // Perform the addition
    if a.type == GGMLType.F32:
        aData = a.data as FloatArray
        bData = b.data as FloatArray
        resultData = result.data as FloatArray

        for i in 0 to aData.size - 1:
            resultData[i] = aData[i] + bData[i]
    else:
        // Handle other data types

    return result
```

### Pseudocode for Multiplication

```
function mul(context, a, b):
    // Check that the tensors have compatible dimensions
    for i in 0 to GGML_MAX_DIMS - 1:
        if a.ne[i] != b.ne[i]:
            throw IncompatibleDimensionsException

    // Create a new tensor for the result
    result = createTensor(context, a.type, a.ne)

    // Perform the multiplication
    if a.type == GGMLType.F32:
        aData = a.data as FloatArray
        bData = b.data as FloatArray
        resultData = result.data as FloatArray

        for i in 0 to aData.size - 1:
            resultData[i] = aData[i] * bData[i]
    else:
        // Handle other data types

    return result
```

## Matrix Multiplication

### Approach

Matrix multiplication involves:
1. Checking that the tensors have compatible dimensions for matrix multiplication
2. Creating a new tensor for the result
3. Performing the matrix multiplication

### Pseudocode

```
function matMul(context, a, b):
    // Check that the tensors have compatible dimensions for matrix multiplication
    // a: m x n, b: n x p, result: m x p
    m = a.ne[0]
    n = a.ne[1]
    p = b.ne[1]

    if b.ne[0] != n:
        throw IncompatibleDimensionsException

    // Create a new tensor for the result
    resultDimensions = [m, p]
    result = createTensor(context, a.type, resultDimensions)

    // Perform the matrix multiplication
    if a.type == GGMLType.F32:
        aData = a.data as FloatArray
        bData = b.data as FloatArray
        resultData = result.data as FloatArray

        for i in 0 to m - 1:
            for j in 0 to p - 1:
                sum = 0.0f
                for k in 0 to n - 1:
                    sum += aData[i * n + k] * bData[k * p + j]
                resultData[i * p + j] = sum
    else:
        // Handle other data types

    return result
```

## Activation Functions

### Approach

Activation functions (ReLU, GELU, etc.) involve:
1. Creating a new tensor for the result (or using an existing one)
2. Applying the activation function element by element

### Pseudocode for ReLU

```
function relu(context, a):
    // Create a new tensor for the result
    result = createTensor(context, a.type, a.ne)

    // Apply the ReLU function
    if a.type == GGMLType.F32:
        aData = a.data as FloatArray
        resultData = result.data as FloatArray

        for i in 0 to aData.size - 1:
            resultData[i] = max(0.0f, aData[i])
    else:
        // Handle other data types

    return result
```

### Pseudocode for GELU

```
function gelu(context, a):
    // Create a new tensor for the result
    result = createTensor(context, a.type, a.ne)

    // Apply the GELU function
    if a.type == GGMLType.F32:
        aData = a.data as FloatArray
        resultData = result.data as FloatArray

        for i in 0 to aData.size - 1:
            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            x = aData[i]
            resultData[i] = x * 0.5f * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)))
    else:
        // Handle other data types

    return result
```

## Computation Graph Execution

### Approach

Computation graph execution involves:
1. Topologically sorting the nodes in the graph
2. Executing the operations in order
3. Managing memory for intermediate results

### Pseudocode

```
function executeGraph(context, graph):
    // Topologically sort the nodes
    sortedNodes = topologicalSort(graph)

    // Execute the operations in order
    for node in sortedNodes:
        if node.op == GGMLOp.ADD:
            node.data = add(context, node.src[0], node.src[1])
        else if node.op == GGMLOp.MUL:
            node.data = mul(context, node.src[0], node.src[1])
        else if node.op == GGMLOp.MUL_MAT:
            node.data = matMul(context, node.src[0], node.src[1])
        else if node.op == GGMLOp.RELU:
            node.data = relu(context, node.src[0])
        else if node.op == GGMLOp.GELU:
            node.data = gelu(context, node.src[0])
        // Handle other operations

    // Return the result of the last node
    return sortedNodes.last()
```

## Memory Management

### Approach

Memory management involves:
1. Allocating memory for tensors
2. Freeing memory when tensors are no longer needed
3. Reusing memory for intermediate results when possible

### Pseudocode

```
function allocateMemory(type, size):
    if type == GGMLType.F32:
        return FloatArray(size) { 0.0f }
    else if type == GGMLType.F16:
        return ShortArray(size) { 0 }
    else if type == GGMLType.I8:
        return ByteArray(size) { 0 }
    else if type == GGMLType.I32:
        return IntArray(size) { 0 }
    else:
        // Handle other data types
```

## Optimization Strategies

### SIMD Vectorization

For CPU backends, SIMD (Single Instruction, Multiple Data) vectorization can be used to accelerate tensor operations. This involves:
1. Using SIMD intrinsics for the target architecture (ARM NEON, x86 AVX, etc.)
2. Processing multiple elements in parallel

### Multi-threading

For large tensors, multi-threading can be used to parallelize tensor operations. This involves:
1. Dividing the tensor into chunks
2. Processing each chunk in a separate thread
3. Synchronizing the results

### Memory Access Patterns

Optimizing memory access patterns can improve cache utilization and reduce memory bandwidth requirements. This involves:
1. Tiling matrix multiplication to improve cache locality
2. Using memory-efficient algorithms for large tensors

## Implementation Strategy

To implement the tensor operations described in this document, we will create a new file called `GGMLComputeOps.kt` that will contain the actual computation functionality for tensor operations. This separation allows for cleaner code organization and easier maintenance.

### GGMLComputeOps.kt

The `GGMLComputeOps.kt` file will contain the following functions:

1. Utility Functions:
   - `calculateTotalSize`: Calculates the total size of a tensor based on its dimensions
   - `allocateMemory`: Allocates memory for a tensor based on its type and size

2. Element-wise Operations:
   - `computeAdd`: Adds two tensors element-wise
   - `computeMul`: Multiplies two tensors element-wise

3. Matrix Operations:
   - `computeMatMul`: Performs matrix multiplication of two tensors

4. Activation Functions:
   - `computeRelu`: Applies the ReLU activation function to a tensor
   - `computeGelu`: Applies the GELU activation function to a tensor

### Integration with Existing Code

The functions in `GGMLComputeOps.kt` will be called from the corresponding functions in `GGMLOps.kt`. For example, the `add` function in `GGMLOps.kt` will call `computeAdd` from `GGMLComputeOps.kt` to perform the actual computation.

```kotlin
// In GGMLOps.kt
fun add(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    // Set up the operation in the computation graph
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.ADD
    result.src[0] = a
    result.src[1] = b

    // If immediate computation is required, call the compute function
    if (context.computeImmediately) {
        return computeAdd(context, a, b)
    }

    return result
}
```

## Conclusion

This design document outlines the approach for implementing tensor operations in the Kotlin Native port of llama.cpp. It provides a roadmap for future implementation, focusing on the actual computation functionality for tensor operations, which is one of the next steps identified in the KOTLIN_PORT_STATUS.md file.

The implementation strategy involves creating a new file called `GGMLComputeOps.kt` that will contain the actual computation functionality for tensor operations, separate from the tensor creation and management functions in `GGMLOps.kt`. This separation allows for cleaner code organization and easier maintenance.
