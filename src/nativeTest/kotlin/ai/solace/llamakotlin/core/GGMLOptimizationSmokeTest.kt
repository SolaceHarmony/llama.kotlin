package ai.solace.llamakotlin.core

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * Basic smoke tests for the new optimization and scheduling functionality
 */
class GGMLOptimizationSmokeTest {
    
    @Test
    fun testBackendBasicFunctionality() {
        val backend = GGMLCpuBackend(threadCount = 1)
        
        assertTrue(backend.initialize(), "Backend should initialize")
        assertTrue(backend.isReady(), "Backend should be ready after initialization")
        assertEquals(1, backend.getThreadCount(), "Thread count should match")
        
        backend.cleanup()
    }
    
    @Test
    fun testBackendManager() {
        val manager = GGMLBackendManager()
        val backend = GGMLCpuBackend()
        
        assertTrue(manager.registerBackend(backend), "Backend registration should succeed")
        assertNotNull(manager.getPrimaryBackend(), "Primary backend should be set")
        assertEquals(1, manager.getBackends().size, "Should have one backend")
        
        manager.cleanup()
    }
    
    @Test
    fun testGraphOptimizer() {
        val context = GGMLContext()
        val optimizer = GGMLGraphOptimizer()
        val graph = createGraph(1)
        
        // Test with empty graph
        val result = optimizer.optimize(graph, context)
        assertEquals(0, result.iterations, "Empty graph should not require optimization")
    }
    
    @Test
    fun testSchedulerCreation() {
        val backendManager = GGMLBackendManager()
        val backend = GGMLCpuBackend()
        backendManager.registerBackend(backend)
        
        val scheduler = GGMLScheduler(backendManager, GGMLSchedulingStrategy.SEQUENTIAL)
        scheduler.setMaxWorkers(2)
        
        val stats = scheduler.getStats()
        assertEquals(2, stats.maxWorkers)
        assertEquals(GGMLSchedulingStrategy.SEQUENTIAL, stats.strategy)
        
        backendManager.cleanup()
    }
    
    @Test
    fun testDependencyTrackerBasic() {
        val tracker = GGMLDependencyTracker()
        val graph = createGraph(1)
        
        // Test with empty graph
        tracker.buildDependencies(graph)
        assertTrue(tracker.getReadyNodes().isEmpty(), "Empty graph should have no ready nodes")
        assertTrue(tracker.isComplete(), "Empty graph should be complete")
    }
    
    @Test
    fun testOptimizationPassNames() {
        val passes = listOf(
            DeadCodeEliminationPass(),
            RedundantOpRemovalPass(),
            ConstantFoldingPass(),
            MemoryOptimizationPass()
        )
        
        val expectedNames = setOf(
            "DeadCodeElimination",
            "RedundantOpRemoval", 
            "ConstantFolding",
            "MemoryOptimization"
        )
        
        val actualNames = passes.map { it.getName() }.toSet()
        assertEquals(expectedNames, actualNames, "Pass names should match expected")
    }
}