package ai.solace.llamakotlin.core

/**
 * Example demonstrating graph optimization and scheduling functionality
 */

/**
 * Example function that creates and executes an optimized computation graph
 */
fun runOptimizationExample() {
    println("=== GGML Graph Optimization and Scheduling Example ===")
    
    // Initialize context and allocator
    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()
    
    // Create backend manager (CPU available by default)
    val backendManager = GGMLBackendManager()
    val cpuBackend = backendManager.getFallbackBackend()
    println("✓ Backend initialized: primary=${backendManager.getPrimaryBackend()?.getName()} fallback=${cpuBackend?.getName()}")
    
    // Create graph with redundant operations for optimization demonstration
    val graph = createGraph(10)
    
    // Create input tensors
    println("\n--- Creating Input Tensors ---")
    val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
    val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
    val c = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
    
    a.data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
    b.data = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
    c.data = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
    
    println("Input A: [${(a.data as FloatArray).joinToString(", ")}]")
    println("Input B: [${(b.data as FloatArray).joinToString(", ")}]")
    println("Input C: [${(c.data as FloatArray).joinToString(", ")}]")
    
    // Create computation: result = ((a + b) * c) + (a + b) [redundant a+b]
    println("\n--- Building Computation Graph ---")
    val add1 = add(context, a, b)  // First a + b
    val add2 = add(context, a, b)  // Redundant a + b (should be optimized away)
    val mul1 = mul(context, add1, c)  // (a + b) * c
    val deadOp = mul(context, a, c)   // Dead operation (not used in final result)
    val finalResult = add(context, mul1, add2)  // Final result using both adds
    
    // Mark final result as output
    finalResult.flags = finalResult.flags or GGML_TENSOR_FLAG_OUTPUT
    
    // Add nodes to graph
    graph.nodes[0] = add1
    graph.nodes[1] = add2
    graph.nodes[2] = mul1
    graph.nodes[3] = deadOp
    graph.nodes[4] = finalResult
    graph.nNodes = 5
    
    graph.leafs[0] = finalResult
    graph.nLeafs = 1
    
    println("Original graph has ${graph.nNodes} operations")
    println("Operations: ADD, ADD (redundant), MUL, MUL (dead), ADD")
    
    // Create optimizer and scheduler
    val optimizer = GGMLGraphOptimizer()
    optimizer.setMaxIterations(3)
    
    val scheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.PARALLEL)
    scheduler.setMaxWorkers(2)
    
    // Show scheduler stats
    val stats = scheduler.getStats()
    println("\n--- Scheduler Configuration ---")
    println("Strategy: ${stats.strategy}")
    println("Max Workers: ${stats.maxWorkers}")
    println("Available Backends: ${stats.availableBackends.joinToString(", ")}")
    
    // Execute with optimization and scheduling
    println("\n--- Executing Optimized Graph ---")
    val status = executeOptimizedGraph(graph, context, optimizer, scheduler)
    
    when (status) {
        GGMLBackendStatus.SUCCESS -> {
            println("✓ Graph execution completed successfully")
            println("Optimized graph has ${graph.nNodes} operations")
            
            // Show result
            val resultData = finalResult.data as FloatArray
            println("Final result: [${resultData.joinToString(", ")}]")
            
            // Verify the computation: ((a + b) * c) + (a + b) = 2 * (a + b) * c
            // For our inputs: 2 * ([1,2,3,4] + [2,3,4,5]) * [1,1,1,1] = 2 * [3,5,7,9] = [6,10,14,18]
            val expected = floatArrayOf(6.0f, 10.0f, 14.0f, 18.0f)
            val isCorrect = resultData.zip(expected).all { (actual, exp) -> 
                kotlin.math.abs(actual - exp) < 0.001f
            }
            
            if (isCorrect) {
                println("✓ Computation result is correct!")
            } else {
                println("✗ Computation result is incorrect")
                println("Expected: [${expected.joinToString(", ")}]")
            }
        }
        GGMLBackendStatus.FAILED -> println("✗ Graph execution failed")
        GGMLBackendStatus.ALLOC_FAILED -> println("✗ Graph execution failed due to allocation error")
        GGMLBackendStatus.NOT_SUPPORTED -> println("✗ Graph execution not supported by backend")
    }
    
    // Cleanup
    backendManager.cleanup()
    println("\n--- Cleanup Complete ---")
}

/**
 * Demonstration of individual optimization passes
 */
fun demonstrateOptimizationPasses() {
    println("\n=== Individual Optimization Pass Demonstration ===")
    
    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()
    
    // Test Dead Code Elimination
    println("\n--- Dead Code Elimination Pass ---")
    val graph1 = createTestGraphWithDeadCode(context, allocator)
    println("Before: ${graph1.nNodes} nodes")
    
    val deadCodePass = DeadCodeEliminationPass()
    val modified1 = deadCodePass.apply(graph1, context)
    println("After: ${graph1.nNodes} nodes")
    println("Modified: $modified1")
    
    // Test Redundant Operation Removal
    println("\n--- Redundant Operation Removal Pass ---")
    val graph2 = createTestGraphWithRedundantOps(context, allocator)
    println("Before: ${graph2.nNodes} nodes")
    
    val redundantPass = RedundantOpRemovalPass()
    val modified2 = redundantPass.apply(graph2, context)
    println("After: ${graph2.nNodes} nodes")
    println("Modified: $modified2")
    
    // Test Constant Folding
    println("\n--- Constant Folding Pass ---")
    val graph3 = createTestGraphWithConstants(context, allocator)
    val nodeWithConstantOp = graph3.nodes[0]!!
    println("Before: operation ${nodeWithConstantOp.op}")
    
    val constantPass = ConstantFoldingPass()
    val modified3 = constantPass.apply(graph3, context)
    println("After: operation ${nodeWithConstantOp.op}")
    println("Modified: $modified3")
    
    if (modified3) {
        println("Constant value computed: ${(nodeWithConstantOp.data as? FloatArray)?.contentToString()}")
    }
}

private fun createTestGraphWithDeadCode(context: GGMLContext, allocator: GGMLGraphAllocator): GGMLCGraph {
    val graph = createGraph(5)
    
    val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
    val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
    a.data = floatArrayOf(1.0f, 2.0f)
    b.data = floatArrayOf(3.0f, 4.0f)
    
    val used = add(context, a, b)
    val dead = mul(context, a, b)  // Dead - not used anywhere
    val output = mul(context, used, a)
    
    output.flags = output.flags or GGML_TENSOR_FLAG_OUTPUT
    
    graph.nodes[0] = used
    graph.nodes[1] = dead
    graph.nodes[2] = output
    graph.nNodes = 3
    
    graph.leafs[0] = output
    graph.nLeafs = 1
    
    return graph
}

private fun createTestGraphWithRedundantOps(context: GGMLContext, allocator: GGMLGraphAllocator): GGMLCGraph {
    val graph = createGraph(5)
    
    val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
    val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
    a.data = floatArrayOf(1.0f, 2.0f)
    b.data = floatArrayOf(3.0f, 4.0f)
    
    val add1 = add(context, a, b)
    val add2 = add(context, a, b)  // Redundant
    val result1 = mul(context, add1, a)
    val result2 = mul(context, add2, a)  // Uses redundant op
    
    result1.flags = result1.flags or GGML_TENSOR_FLAG_OUTPUT
    result2.flags = result2.flags or GGML_TENSOR_FLAG_OUTPUT
    
    graph.nodes[0] = add1
    graph.nodes[1] = add2
    graph.nodes[2] = result1
    graph.nodes[3] = result2
    graph.nNodes = 4
    
    return graph
}

private fun createTestGraphWithConstants(context: GGMLContext, allocator: GGMLGraphAllocator): GGMLCGraph {
    val graph = createGraph(3)
    
    val const1 = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
    const1.data = floatArrayOf(5.0f, 10.0f)
    const1.op = GGMLOp.NONE  // Mark as constant
    
    val const2 = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
    const2.data = floatArrayOf(2.0f, 3.0f)
    const2.op = GGMLOp.NONE  // Mark as constant
    
    val addConst = add(context, const1, const2)  // Should be folded to constant
    
    graph.nodes[0] = addConst
    graph.nNodes = 1
    
    return graph
}

// Helper to run optimization then schedule execution using unified APIs
fun executeOptimizedGraph(
    graph: GGMLCGraph,
    context: GGMLContext,
    optimizer: GGMLGraphOptimizer,
    scheduler: GGMLScheduler
): GGMLBackendStatus {
    optimizer.optimize(graph, context)
    return scheduler.execute(graph, context)
}