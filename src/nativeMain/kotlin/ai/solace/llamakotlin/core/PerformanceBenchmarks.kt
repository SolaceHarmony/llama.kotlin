package ai.solace.llamakotlin.core

/**
 * Performance benchmarks for graph optimization and scheduling
 */

data class BenchmarkResult(
    val operationName: String,
    val unoptimizedTimeMs: Long,
    val optimizedTimeMs: Long,
    val speedupRatio: Double,
    val memoryReduction: Int = 0
)

/**
 * Benchmark graph optimization performance
 */
fun benchmarkGraphOptimization(): List<BenchmarkResult> {
    println("=== Graph Optimization Performance Benchmark ===")
    
    val results = mutableListOf<BenchmarkResult>()
    
    // Benchmark 1: Large graph with redundant operations
    results.add(benchmarkRedundantOperations())
    
    // Benchmark 2: Graph with dead code
    results.add(benchmarkDeadCodeElimination())
    
    // Benchmark 3: Graph with constants
    results.add(benchmarkConstantFolding())
    
    // Benchmark 4: Multi-threaded vs single-threaded execution
    results.add(benchmarkParallelExecution())
    
    return results
}

private fun benchmarkRedundantOperations(): BenchmarkResult {
    println("\n--- Benchmarking Redundant Operation Removal ---")
    
    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()
    
    // Create graph with many redundant operations
    val graph = createGraph(100)
    val inputs = mutableListOf<GGMLTensor>()
    
    // Create input tensors
    for (i in 0 until 10) {
        val input = allocator.allocateTensor(GGMLType.F32, longArrayOf(100, 1, 1, 1))
        val data = FloatArray(100) { it.toFloat() }
        input.data = data
        inputs.add(input)
    }
    
    // Create many redundant ADD operations
    var nodeCount = 0
    for (i in 0 until 10) {
        for (j in i + 1 until 10) {
            // Create 5 identical operations for each pair
            repeat(5) {
                val addOp = add(context, inputs[i], inputs[j])
                graph.nodes[nodeCount] = addOp
                nodeCount++
            }
        }
    }
    graph.nNodes = nodeCount
    
    val finalResult = graph.nodes[nodeCount - 1]!!
    finalResult.flags = finalResult.flags or GGML_TENSOR_FLAG_OUTPUT
    graph.leafs[0] = finalResult
    graph.nLeafs = 1
    
    println("Created graph with ${graph.nNodes} operations")
    
    // Benchmark unoptimized execution
    val unoptimizedGraph = duplicateGraph(graph)
    val unoptimizedStart = getCurrentTimeMs()
    
    // Create backend for execution
    val backendManager = GGMLBackendManager()
    
    val scheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.SEQUENTIAL)
    scheduler.execute(unoptimizedGraph, context)
    
    val unoptimizedTime = getCurrentTimeMs() - unoptimizedStart
    
    // Benchmark optimized execution
    val optimizer = GGMLGraphOptimizer()
    val optimizedStart = getCurrentTimeMs()
    
    val optimizationResult = optimizer.optimize(graph, context)
    scheduler.execute(graph, context)
    
    val optimizedTime = getCurrentTimeMs() - optimizedStart
    
    val speedup = if (optimizedTime > 0) unoptimizedTime.toDouble() / optimizedTime.toDouble() else 1.0
    val memoryReduction = unoptimizedGraph.nNodes - graph.nNodes
    
    println("Unoptimized: ${unoptimizedTime}ms with ${unoptimizedGraph.nNodes} operations")
    println("Optimized: ${optimizedTime}ms with ${graph.nNodes} operations")
    println("Speedup: ${format2(speedup)}x")
    println("Memory reduction: $memoryReduction operations")
    
    backendManager.cleanup()
    
    return BenchmarkResult(
        "Redundant Operations",
        unoptimizedTime,
        optimizedTime,
        speedup,
        memoryReduction
    )
}

private fun benchmarkDeadCodeElimination(): BenchmarkResult {
    println("\n--- Benchmarking Dead Code Elimination ---")
    
    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()
    
    val graph = createGraph(50)
    
    // Create inputs
    val input1 = allocator.allocateTensor(GGMLType.F32, longArrayOf(50, 1, 1, 1))
    val input2 = allocator.allocateTensor(GGMLType.F32, longArrayOf(50, 1, 1, 1))
    input1.data = FloatArray(50) { it.toFloat() }
    input2.data = FloatArray(50) { (it + 1).toFloat() }
    
    // Create a chain of operations where only some are used
    var nodeCount = 0
    val usedOp = add(context, input1, input2)
    graph.nodes[nodeCount++] = usedOp
    
    // Create many dead operations
    for (i in 0 until 20) {
        val deadOp = mul(context, input1, input2)
        graph.nodes[nodeCount++] = deadOp
    }
    
    // Final result uses only the first operation
    val result = mul(context, usedOp, input1)
    result.flags = result.flags or GGML_TENSOR_FLAG_OUTPUT
    graph.nodes[nodeCount++] = result
    graph.nNodes = nodeCount
    
    graph.leafs[0] = result
    graph.nLeafs = 1
    
    // Create backend
    val backendManager = GGMLBackendManager()
    val scheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.SEQUENTIAL)
    
    // Benchmark unoptimized
    val unoptimizedGraph = duplicateGraph(graph)
    val unoptimizedStart = getCurrentTimeMs()
    scheduler.execute(unoptimizedGraph, context)
    val unoptimizedTime = getCurrentTimeMs() - unoptimizedStart
    
    // Benchmark optimized
    val optimizer = GGMLGraphOptimizer()
    val optimizedStart = getCurrentTimeMs()
    optimizer.optimize(graph, context)
    scheduler.execute(graph, context)
    val optimizedTime = getCurrentTimeMs() - optimizedStart
    
    val speedup = if (optimizedTime > 0) unoptimizedTime.toDouble() / optimizedTime.toDouble() else 1.0
    val memoryReduction = unoptimizedGraph.nNodes - graph.nNodes
    
    println("Unoptimized: ${unoptimizedTime}ms with ${unoptimizedGraph.nNodes} operations")
    println("Optimized: ${optimizedTime}ms with ${graph.nNodes} operations")
    println("Memory reduction: $memoryReduction operations")
    
    backendManager.cleanup()
    
    return BenchmarkResult("Dead Code Elimination", unoptimizedTime, optimizedTime, speedup, memoryReduction)
}

private fun benchmarkConstantFolding(): BenchmarkResult {
    println("\n--- Benchmarking Constant Folding ---")
    
    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()
    
    val graph = createGraph(30)
    
    // Create many constants
    val constants = mutableListOf<GGMLTensor>()
    for (i in 0 until 10) {
        val constant = allocator.allocateTensor(GGMLType.F32, longArrayOf(10, 1, 1, 1))
        constant.data = FloatArray(10) { it.toFloat() + i }
        constant.op = GGMLOp.NONE
        constants.add(constant)
    }
    
    // Create operations on constants that can be folded
    var nodeCount = 0
    for (i in 0 until constants.size - 1) {
        val addOp = add(context, constants[i], constants[i + 1])
        graph.nodes[nodeCount++] = addOp
    }
    graph.nNodes = nodeCount
    
    // Create backend
    val backendManager = GGMLBackendManager()
    val scheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.SEQUENTIAL)
    
    // Benchmark unoptimized
    val unoptimizedGraph = duplicateGraph(graph)
    val unoptimizedStart = getCurrentTimeMs()
    scheduler.execute(unoptimizedGraph, context)
    val unoptimizedTime = getCurrentTimeMs() - unoptimizedStart
    
    // Count operations that should be folded
    val originalOpCount = graph.nodes.take(graph.nNodes).count { it?.op != GGMLOp.NONE }
    
    // Benchmark optimized
    val optimizer = GGMLGraphOptimizer()
    val optimizedStart = getCurrentTimeMs()
    optimizer.optimize(graph, context)
    scheduler.execute(graph, context)
    val optimizedTime = getCurrentTimeMs() - optimizedStart
    
    val foldedOpCount = graph.nodes.take(graph.nNodes).count { it?.op != GGMLOp.NONE }
    val operationReduction = originalOpCount - foldedOpCount
    
    val speedup = if (optimizedTime > 0) unoptimizedTime.toDouble() / optimizedTime.toDouble() else 1.0
    
    println("Original operations: $originalOpCount")
    println("After folding: $foldedOpCount")
    println("Operations folded: $operationReduction")
    
    backendManager.cleanup()
    
    return BenchmarkResult("Constant Folding", unoptimizedTime, optimizedTime, speedup, operationReduction)
}

private fun benchmarkParallelExecution(): BenchmarkResult {
    println("\n--- Benchmarking Parallel vs Sequential Execution ---")
    
    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()
    
    // Create a graph with independent operations that can be parallelized
    val graph = createGraph(20)
    
    val inputs = mutableListOf<GGMLTensor>()
    for (i in 0 until 10) {
        val input = allocator.allocateTensor(GGMLType.F32, longArrayOf(100, 1, 1, 1))
        input.data = FloatArray(100) { it.toFloat() + i }
        inputs.add(input)
    }
    
    // Create independent operations
    var nodeCount = 0
    for (i in 0 until inputs.size - 1) {
        val op = add(context, inputs[i], inputs[i + 1])
        op.flags = op.flags or GGML_TENSOR_FLAG_OUTPUT
        graph.nodes[nodeCount++] = op
        graph.leafs[i] = op
    }
    graph.nNodes = nodeCount
    graph.nLeafs = nodeCount
    
    val backendManager = GGMLBackendManager()
    
    // Benchmark sequential execution
    val sequentialScheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.SEQUENTIAL)
    val sequentialGraph = duplicateGraph(graph)
    
    val sequentialStart = getCurrentTimeMs()
    sequentialScheduler.execute(sequentialGraph, context)
    val sequentialTime = getCurrentTimeMs() - sequentialStart
    
    // Benchmark parallel execution
    val parallelScheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.PARALLEL)
    parallelScheduler.setMaxWorkers(4)
    
    val parallelStart = getCurrentTimeMs()
    parallelScheduler.execute(graph, context)
    val parallelTime = getCurrentTimeMs() - parallelStart
    
    val speedup = if (parallelTime > 0) sequentialTime.toDouble() / parallelTime.toDouble() else 1.0
    
    println("Sequential: ${sequentialTime}ms")
    println("Parallel: ${parallelTime}ms")
    println("Speedup: ${format2(speedup)}x")
    
    backendManager.cleanup()
    
    return BenchmarkResult("Parallel Execution", sequentialTime, parallelTime, speedup)
}

// Helper functions

@Suppress("DEPRECATION")
private fun getCurrentTimeMs(): Long {
    return kotlin.system.getTimeNanos() / 1_000_000
}

private fun format2(x: Double): String {
    val v = kotlin.math.round(x * 100.0) / 100.0
    return v.toString()
}

private fun duplicateGraph(original: GGMLCGraph): GGMLCGraph {
    val duplicate = createGraph(original.size)
    
    for (i in 0 until original.nNodes) {
        duplicate.nodes[i] = original.nodes[i]
    }
    duplicate.nNodes = original.nNodes
    
    for (i in 0 until original.nLeafs) {
        duplicate.leafs[i] = original.leafs[i]
    }
    duplicate.nLeafs = original.nLeafs
    
    return duplicate
}

/**
 * Run all benchmarks and display results
 */
fun runPerformanceBenchmarks() {
    val results = benchmarkGraphOptimization()
    
    println("\n=== Performance Benchmark Summary ===")
    results.forEach { result ->
        println("${result.operationName}:")
        println("  Speedup: ${format2(result.speedupRatio)}x")
        if (result.memoryReduction > 0) {
            println("  Memory reduction: ${result.memoryReduction} operations")
        }
        println()
    }
    
    val avgSpeedup = results.map { it.speedupRatio }.average()
    println("Average speedup: ${format2(avgSpeedup)}x")
}