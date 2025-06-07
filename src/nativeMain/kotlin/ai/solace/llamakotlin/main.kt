package ai.solace.llamakotlin

import ai.solace.llamakotlin.core.*

fun main() {
    kotlin.io.println("LLaMA Kotlin Native - Starting up")
    kotlin.io.println("This is a Kotlin Native port of llama.cpp")

    // Demonstrate the use of the computation graph functionality
    demonstrateComputationGraph()

    // Demonstrate the use of the memory allocation functionality
    demonstrateMemoryAllocation()

    // Demonstrate the use of the optimized tensor operations
    demonstrateOptimizedTensorOps()

}

/**
 * Demonstrates the use of the optimized tensor operations.
 */
fun demonstrateOptimizedTensorOps() {
    kotlin.io.println("\nDemonstrating optimized tensor operations:")

    // Create a context
    val context = GGMLContext(
        memSize = (16 * 1024 * 1024).toULong(), // 16 MB
        memBuffer = null,
        memBufferOwned = false,
        noAlloc = false,
        computeImmediately = true // Important: set to true to compute immediately
    )

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 4, 4)
    val b = createTensor2D(context, GGMLType.F32, 4, 4)

    // Initialize tensor data
    val aData = a.data as FloatArray
    val bData = b.data as FloatArray
    for (i in 0 until 16) {
        aData[i] = i.toFloat()
        bData[i] = (i + 1).toFloat()
    }

    kotlin.io.println("Tensor a: [${aData.take(16).joinToString()}]")
    kotlin.io.println("Tensor b: [${bData.take(16).joinToString()}]")

    // Test optimized add operation
    val c = computeAdd(context, a, b)
    val cData = c.data as FloatArray
    kotlin.io.println("a + b: [${cData.take(16).joinToString()}]")

    // Test optimized mul operation
    val d = computeMul(context, a, b)
    val dData = d.data as FloatArray
    kotlin.io.println("a * b: [${dData.take(16).joinToString()}]")

    // Test optimized matMul operation
    val e = computeMatMul(context, a, b)
    val eData = e.data as FloatArray
    kotlin.io.println("a @ b: [${eData.take(16).joinToString()}]")
}

/**
 * Demonstrates the use of the computation graph functionality.
 */
fun demonstrateComputationGraph() {
    kotlin.io.println("\nDemonstrating computation graph functionality:")

    // Create a context
    val context = GGMLContext(
        memSize = (16 * 1024 * 1024).toULong(), // 16 MB
        memBuffer = null,
        memBufferOwned = false,
        noAlloc = false,
        computeImmediately = false // Important: set to false to use the computation graph
    )

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 2, 2)
    val b = createTensor2D(context, GGMLType.F32, 2, 2)

    // Initialize tensor data
    (a.data as FloatArray)[0] = 1.0f
    (a.data as FloatArray)[1] = 2.0f
    (a.data as FloatArray)[2] = 3.0f
    (a.data as FloatArray)[3] = 4.0f

    (b.data as FloatArray)[0] = 5.0f
    (b.data as FloatArray)[1] = 6.0f
    (b.data as FloatArray)[2] = 7.0f
    (b.data as FloatArray)[3] = 8.0f

    kotlin.io.println("Tensor a: [${(a.data as FloatArray).joinToString()}]")
    kotlin.io.println("Tensor b: [${(b.data as FloatArray).joinToString()}]")

    // Create operations
    val c = add(context, a, b) // c = a + b
    val d = mul(context, a, b) // d = a * b
    val e = matMul(context, a, b) // e = a @ b

    // Create a computation graph
    val graph = createGraph(100) // Maximum 100 nodes

    // Build the graph for all operations
    buildForward(graph, c)
    buildForward(graph, d)
    buildForward(graph, e)

    kotlin.io.println("Graph built with ${graph.nNodes} nodes and ${graph.nLeafs} leaf nodes")

    // Execute the graph
    executeGraph(context, graph)

    // Print the results
    kotlin.io.println("c = a + b: [${(c.data as FloatArray).joinToString()}]")
    kotlin.io.println("d = a * b: [${(d.data as FloatArray).joinToString()}]")
    kotlin.io.println("e = a @ b: [${(e.data as FloatArray).joinToString()}]")
}

/**
 * Demonstrates the use of the memory allocation functionality.
 */
fun demonstrateMemoryAllocation() {
    kotlin.io.println("\nDemonstrating memory allocation functionality:")

    // Create a tensor allocator
    val tensorAllocator = GGMLTensorAllocator()

    // Create a tensor
    val tensor = GGMLTensor(type = GGMLType.F32)
    tensor.ne[0] = 2
    tensor.ne[1] = 3

    // Allocate memory for the tensor
    tensorAllocator.allocate(tensor)

    // Initialize tensor data
    val data = tensor.data as FloatArray
    for (i in data.indices) {
        data[i] = i.toFloat()
    }

    kotlin.io.println("Tensor data: [${data.joinToString()}]")

    // Create a dynamic tensor allocator
    val dynTensorAllocator = GGMLDynTensorAllocator()

    // Create tensors
    val tensor1 = GGMLTensor(type = GGMLType.F32)
    tensor1.ne[0] = 2
    tensor1.ne[1] = 2

    val tensor2 = GGMLTensor(type = GGMLType.F32)
    tensor2.ne[0] = 3
    tensor2.ne[1] = 3

    // Calculate tensor sizes
    val size1 = 4UL // 2x2
    val size2 = 9UL // 3x3

    // Allocate memory for the tensors
    val offset1 = dynTensorAllocator.allocate(size1, tensor1)
    val offset2 = dynTensorAllocator.allocate(size2, tensor2)

    kotlin.io.println("Tensor 1 allocated at offset $offset1 with size $size1")
    kotlin.io.println("Tensor 2 allocated at offset $offset2 with size $size2")

    // Free memory for tensor1
    dynTensorAllocator.freeTensor(offset1, size1, tensor1)
    kotlin.io.println("Tensor 1 freed")

    // Allocate memory for a new tensor
    val tensor3 = GGMLTensor(type = GGMLType.F32)
    tensor3.ne[0] = 2
    tensor3.ne[1] = 1

    val size3 = 2UL // 2x1
    val offset3 = dynTensorAllocator.allocate(size3, tensor3)

    kotlin.io.println("Tensor 3 allocated at offset $offset3 with size $size3")

    // Create a graph allocator
    val graphAllocator = GGMLGraphAllocator()

    // Create a computation graph
    val graph = createGraph(10)

    // Create tensors for the graph
    val a = GGMLTensor(type = GGMLType.F32)
    a.ne[0] = 2
    a.ne[1] = 2

    val b = GGMLTensor(type = GGMLType.F32)
    b.ne[0] = 2
    b.ne[1] = 2

    // Set up the graph
    graph.leafs[0] = a
    graph.leafs[1] = b
    graph.nLeafs = 2

    // Create an operation node
    val c = GGMLTensor(type = GGMLType.F32)
    c.ne[0] = 2
    c.ne[1] = 2
    c.op = GGMLOp.ADD
    c.src[0] = a
    c.src[1] = b

    graph.nodes[0] = c
    graph.nNodes = 1

    // Allocate memory for the graph
    val success = graphAllocator.allocateGraph(graph)

    kotlin.io.println("Graph allocation ${if (success) "succeeded" else "failed"}")
    kotlin.io.println("Buffer size: ${graphAllocator.getBufferSize(0)}")
}

