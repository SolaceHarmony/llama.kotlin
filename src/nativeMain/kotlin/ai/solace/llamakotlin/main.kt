package ai.solace.llamakotlin

import ai.solace.llamakotlin.core.*
import ai.solace.llamakotlin.examples.runComprehensiveDemo

fun main() {
    kotlin.io.println("🦙 LLaMA Kotlin Native - Advanced Capabilities Demonstration")
    kotlin.io.println("This is a comprehensive Kotlin Native port of llama.cpp")
    kotlin.io.println("Showcasing K-Quantization, GGUF loading, tensor operations, and more!")
    kotlin.io.println()

    // Run the comprehensive integration demo
    val demoResult = runComprehensiveDemo()
    kotlin.io.println(demoResult)
    
    kotlin.io.println()
    kotlin.io.println("=" .repeat(60))
    kotlin.io.println("🎯 Additional Core Functionality Demos")
    kotlin.io.println("=" .repeat(60))

    // Demonstrate the use of the computation graph functionality
    demonstrateComputationGraph()

    // Demonstrate the use of the memory allocation functionality
    demonstrateMemoryAllocation()

    // Demonstrate the use of the optimized tensor operations
    demonstrateOptimizedTensorOps()

    // Demonstrate the new graph optimization and scheduling features
    demonstrateGraphOptimization()
    
    kotlin.io.println()
    kotlin.io.println("🚀 All demonstrations completed! The Kotlin llama.cpp port is comprehensive and functional.")
}

/**
 * Demonstrates the use of the optimized tensor operations.
 */
fun demonstrateOptimizedTensorOps() {
    kotlin.io.println("\nDemonstrating optimized tensor operations:")

    // Build via graph and run with allocator/backends to respect dst-arg API
    val context = GGMLContext(computeImmediately = false)
    val graph = createGraph(10)
    val allocator = graph.allocator ?: GGMLGraphAllocator().also { graph.allocator = it }

    // Create tensors directly and mark as leafs
    val a = GGMLTensor(type = GGMLType.F32).apply { ne[0] = 4; ne[1] = 4; data = FloatArray(16) { it.toFloat() } }
    val b = GGMLTensor(type = GGMLType.F32).apply { ne[0] = 4; ne[1] = 4; data = FloatArray(16) { (it + 1).toFloat() } }

    graph.leafs[0] = a; graph.leafs[1] = b; graph.nLeafs = 2

    val addNode = add(context, a, b)
    val mulNode = mul(context, a, b)
    val mmNode = matMul(context, a, b)

    graph.nodes[0] = addNode
    graph.nodes[1] = mulNode
    graph.nodes[2] = mmNode
    graph.nNodes = 3

    // Allocate and execute
    allocator.allocateGraph(graph)
    computeGraphWithBackend(graph, context)

    kotlin.io.println("Tensor a: [${(a.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("Tensor b: [${(b.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("a + b: [${(addNode.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("a * b: [${(mulNode.data as FloatArray).take(16).joinToString()}]")
    kotlin.io.println("a @ b: [${(mmNode.data as FloatArray).take(16).joinToString()}]")
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

    // Execute the graph (allocates internally and uses backend/CPU compute)
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

/**
 * Demonstrates the new graph optimization and scheduling functionality.
 */
fun demonstrateGraphOptimization() {
    kotlin.io.println("\n=== Graph Optimization and Scheduling Demo ===")
    
    try {
        // Run the optimization example
        runOptimizationExample()
        
        kotlin.io.println("\n--- Individual Pass Demonstration ---")
        demonstrateOptimizationPasses()
        
        kotlin.io.println("\n--- Performance Benchmarks ---")
        runPerformanceBenchmarks()
        
        kotlin.io.println("\n✓ Graph optimization and scheduling demonstration completed successfully")
        
    } catch (e: Exception) {
        kotlin.io.println("✗ Error during optimization demo: ${e.message}")
        // Don't crash the main demo, just log the error
    }
}

