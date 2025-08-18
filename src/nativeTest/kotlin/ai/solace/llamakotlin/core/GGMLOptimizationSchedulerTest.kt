package ai.solace.llamakotlin.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertNotNull

/**
 * Tests for graph optimization passes and scheduler functionality
 */
class GGMLOptimizationSchedulerTest {
    
    @Test
    fun testDeadCodeEliminationPass() {
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        // Create a graph with some dead code
        val graph = createGraph(10)
        
        // Create input tensors
        val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        a.data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        b.data = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
        
        // Create operations
        val add1 = add(context, a, b)  // Used
        val add2 = add(context, a, b)  // Dead - same as add1 but not used
        val result = mul(context, add1, a)  // Uses add1, so add1 is not dead
        
        // Mark result as output
        result.flags = result.flags or GGML_TENSOR_FLAG_OUTPUT
        
        // Add to graph
        graph.nodes[0] = add1
        graph.nodes[1] = add2  // This should be eliminated
        graph.nodes[2] = result
        graph.nNodes = 3
        
        graph.leafs[0] = result
        graph.nLeafs = 1
        
        val pass = DeadCodeEliminationPass()
        val wasModified = pass.apply(graph, context)
        
        assertTrue(wasModified, "Dead code elimination should modify the graph")
        assertEquals(2, graph.nNodes, "Dead node should be removed")
        
        // Verify remaining nodes are the ones we expect
        val remainingOps = mutableSetOf<GGMLOp>()
        for (i in 0 until graph.nNodes) {
            graph.nodes[i]?.let { remainingOps.add(it.op) }
        }
        assertTrue(remainingOps.contains(GGMLOp.ADD), "ADD operation should remain")
        assertTrue(remainingOps.contains(GGMLOp.MUL), "MUL operation should remain")
    }
    
    @Test
    fun testRedundantOpRemovalPass() {
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        val graph = createGraph(10)
        
        // Create input tensors
        val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        a.data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        b.data = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
        
        // Create identical operations
        val add1 = add(context, a, b)
        val add2 = add(context, a, b)  // Redundant - identical to add1
        val result1 = mul(context, add1, a)
        val result2 = mul(context, add2, a)  // Uses redundant add2
        
        // Mark results as outputs
        result1.flags = result1.flags or GGML_TENSOR_FLAG_OUTPUT
        result2.flags = result2.flags or GGML_TENSOR_FLAG_OUTPUT
        
        // Add to graph
        graph.nodes[0] = add1
        graph.nodes[1] = add2
        graph.nodes[2] = result1
        graph.nodes[3] = result2
        graph.nNodes = 4
        
        val pass = RedundantOpRemovalPass()
        val wasModified = pass.apply(graph, context)
        
        assertTrue(wasModified, "Redundant operation removal should modify the graph")
        assertTrue(graph.nNodes < 4, "Redundant operations should be removed")
    }
    
    @Test
    fun testConstantFoldingPass() {
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        val graph = createGraph(5)
        
        // Create constant tensors (tensors with data but no operation)
        val const1 = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
        const1.data = floatArrayOf(3.0f, 4.0f)
        const1.op = GGMLOp.NONE
        
        val const2 = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
        const2.data = floatArrayOf(2.0f, 5.0f)
        const2.op = GGMLOp.NONE
        
        // Create operation on constants
        val addConst = add(context, const1, const2)
        
        // Add to graph
        graph.nodes[0] = addConst
        graph.nNodes = 1
        
        val pass = ConstantFoldingPass()
        val wasModified = pass.apply(graph, context)
        
        assertTrue(wasModified, "Constant folding should modify the graph")
        assertEquals(GGMLOp.NONE, addConst.op, "Folded operation should become NONE")
        assertNotNull(addConst.data, "Folded operation should have computed data")
    }
    
    @Test
    fun testGraphOptimizerIntegration() {
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        val graph = createGraph(10)
        
        // Create a graph with multiple optimization opportunities
        val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        a.data = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        b.data = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
        
        val add1 = add(context, a, b)
        val add2 = add(context, a, b)  // Redundant
        val deadOp = mul(context, a, b)  // Dead - not used
        val result = mul(context, add1, a)
        
        result.flags = result.flags or GGML_TENSOR_FLAG_OUTPUT
        
        graph.nodes[0] = add1
        graph.nodes[1] = add2
        graph.nodes[2] = deadOp
        graph.nodes[3] = result
        graph.nNodes = 4
        
        graph.leafs[0] = result
        graph.nLeafs = 1
        
        val optimizer = GGMLGraphOptimizer()
        val optimizationResult = optimizer.optimize(graph, context)
        
        assertTrue(optimizationResult.iterations > 0, "Optimizer should run at least one iteration")
        assertTrue(graph.nNodes < 4, "Graph should be optimized")
    }
    
    @Test
    fun testCpuBackend() {
        val backend = GGMLCpuBackend(threadCount = 2)
        assertTrue(backend.initialize(), "CPU backend should initialize successfully")
        
        assertEquals(GGMLBackendType.CPU, backend.type)
        assertEquals(2, backend.getThreadCount())
        assertTrue(backend.getCapabilities() > 0)
        
        // Test operation support
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        val tensor = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        tensor.op = GGMLOp.ADD
        
        assertTrue(backend.supportsOperation(tensor), "CPU backend should support ADD operation")
        
        backend.cleanup()
    }
    
    @Test
    fun testBackendManager() {
        val manager = GGMLBackendManager()
        val cpuBackend = GGMLCpuBackend()
        
        assertTrue(manager.registerBackend(cpuBackend), "Backend registration should succeed")
        assertEquals(cpuBackend, manager.getPrimaryBackend())
        assertEquals(1, manager.getBackends().size)
        
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        val tensor = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 1, 1, 1))
        tensor.op = GGMLOp.ADD
        
        val bestBackend = manager.getBestBackend(tensor)
        assertNotNull(bestBackend)
        assertEquals(GGMLBackendType.CPU, bestBackend.type)
        
        manager.cleanup()
    }
    
    @Test
    fun testSchedulerSequential() {
        val backendManager = GGMLBackendManager()
        val cpuBackend = GGMLCpuBackend()
        backendManager.registerBackend(cpuBackend)
        
        val scheduler = GGMLScheduler(
            backendManager, 
            GGMLSchedulingStrategy.SEQUENTIAL
        )
        
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        val graph = createGraph(5)
        
        // Create simple computation
        val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
        val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
        a.data = floatArrayOf(1.0f, 2.0f)
        b.data = floatArrayOf(3.0f, 4.0f)
        
        val result = add(context, a, b)
        
        graph.nodes[0] = result
        graph.nNodes = 1
        
        val status = scheduler.execute(graph, context)
        assertEquals(GGMLBackendStatus.SUCCESS, status)
        
        backendManager.cleanup()
    }
    
    @Test
    fun testDependencyTracker() {
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        val graph = createGraph(5)
        
        // Create computation with dependencies: c = (a + b) * a
        val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
        val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(2, 1, 1, 1))
        a.data = floatArrayOf(1.0f, 2.0f)
        b.data = floatArrayOf(3.0f, 4.0f)
        
        val add = add(context, a, b)
        val result = mul(context, add, a)
        
        graph.nodes[0] = add
        graph.nodes[1] = result
        graph.nNodes = 2
        
        val tracker = GGMLDependencyTracker()
        tracker.buildDependencies(graph)
        
        // Initially, only 'add' should be ready (no dependencies)
        var readyNodes = tracker.getReadyNodes()
        assertEquals(1, readyNodes.size)
        assertEquals(GGMLOp.ADD, readyNodes[0].op)
        
        // After marking 'add' complete, 'result' should be ready
        tracker.markCompleted(add)
        readyNodes = tracker.getReadyNodes()
        assertEquals(1, readyNodes.size)
        assertEquals(GGMLOp.MUL, readyNodes[0].op)
        
        // After marking 'result' complete, no nodes should be ready
        tracker.markCompleted(result)
        assertTrue(tracker.isComplete())
        assertTrue(tracker.getReadyNodes().isEmpty())
    }
    
    @Test
    fun testOptimizedGraphExecution() {
        val context = GGMLContext()
        val allocator = GGMLGraphAllocator()
        
        // Set up backend manager
        val backendManager = GGMLBackendManager()
        val cpuBackend = GGMLCpuBackend()
        backendManager.registerBackend(cpuBackend)
        
        // Create optimizer and scheduler
        val optimizer = GGMLGraphOptimizer()
        val scheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.SEQUENTIAL)
        
        // Create test graph
        val graph = createGraph(5)
        
        val a = allocator.allocateTensor(GGMLType.F32, longArrayOf(3, 1, 1, 1))
        val b = allocator.allocateTensor(GGMLType.F32, longArrayOf(3, 1, 1, 1))
        a.data = floatArrayOf(1.0f, 2.0f, 3.0f)
        b.data = floatArrayOf(4.0f, 5.0f, 6.0f)
        
        val add1 = add(context, a, b)
        val add2 = add(context, a, b)  // Redundant
        val result = mul(context, add1, b)
        
        result.flags = result.flags or GGML_TENSOR_FLAG_OUTPUT
        
        graph.nodes[0] = add1
        graph.nodes[1] = add2
        graph.nodes[2] = result
        graph.nNodes = 3
        
        // Execute with optimization and scheduling
        val status = executeOptimizedGraph(graph, context, optimizer, scheduler)
        assertEquals(GGMLBackendStatus.SUCCESS, status)
        
        // Verify the result was computed correctly
        assertNotNull(result.data)
        
        backendManager.cleanup()
    }
}