package ai.solace.llamakotlin.core

import kotlin.test.*

/**
 * Integration test for backend abstraction with hybrid execution.
 * Tests the complete flow from graph creation to execution across different backends.
 */
class GGMLBackendIntegrationTest {

    @Test
    fun testEndToEndCpuExecution() {
        println("Testing end-to-end CPU execution...")
        
        // Create CPU backend and graph
        val backend = GGMLCpuBackend()
        val graph = createGraph(10, backend)
        
        // Verify graph setup
        assertNotNull(graph.allocator)
        assertEquals(backend, graph.allocator!!.backend)
        assertEquals(10, graph.size)
        
        // Test computation (empty graph should succeed)
        val status = backend.graphCompute(graph)
        assertEquals(GGMLStatus.SUCCESS, status)
        
        backend.free()
    }
    
    @Test
    fun testHybridExecutionStrategies() {
        println("Testing hybrid execution strategies...")
        
        val manager = GGMLBackendManager()
        
        // Test different strategies
        val strategies = listOf(
            GGMLBackendSelectionStrategy.CPU_ONLY,
            GGMLBackendSelectionStrategy.AUTO
        )
        
        strategies.forEach { strategy ->
            val graph = manager.createGraphWithBackend(5, strategy)
            assertNotNull(graph.allocator)
            assertNotNull(graph.allocator!!.backend)
            
            val status = manager.computeGraphHybrid(graph)
            assertEquals(GGMLStatus.SUCCESS, status)
        }
        
        manager.cleanup()
    }
    
    @Test
    fun testBackendSelectionLogic() {
        println("Testing backend selection logic...")
        
        val manager = GGMLBackendManager()
        
        // Small operation - should prefer CPU
        val smallTensor = GGMLTensor(op = GGMLOp.ADD)
        smallTensor.ne[0] = 10L
        val smallBackend = manager.selectBackend(smallTensor)
        assertNotNull(smallBackend)
        assertEquals("CPU", smallBackend.getName())
        
        // Large matrix multiplication - might prefer Metal if available, else CPU
        val largeTensor = GGMLTensor(op = GGMLOp.MUL_MAT)
        largeTensor.ne[0] = 1000L
        largeTensor.ne[1] = 1000L
        val largeBackend = manager.selectBackend(largeTensor)
        assertNotNull(largeBackend)
        // Should be CPU since Metal is stub
        assertEquals("CPU", largeBackend.getName())
        
        manager.cleanup()
    }
    
    @Test
    fun testBufferDataIntegrity() {
        println("Testing buffer data integrity...")
        
        val backend = GGMLCpuBackend()
        val buffer = backend.allocBuffer(64u) as? GGMLCpuBuffer
        assertNotNull(buffer)
        
        // Create a simple tensor
        val tensor = GGMLTensor(type = GGMLType.F32)
        tensor.ne[0] = 4L
        tensor.nb[0] = 4u  // F32 size
        tensor.dataOffset = 0u
        buffer.initTensor(tensor)
        
        // Test data round-trip
        val originalData = ByteArray(16) { i -> (i * 3).toByte() }
        buffer.setTensor(tensor, originalData, 0u, 16u)
        
        val retrievedData = ByteArray(16)
        buffer.getTensor(tensor, retrievedData, 0u, 16u)
        
        assertContentEquals(originalData, retrievedData, "Data should be preserved through buffer operations")
        
        // Test buffer clearing
        buffer.clear(255u)
        val data = buffer.getBase() as ByteArray
        assertEquals(255.toByte(), data[0], "Buffer should be cleared to specified value")
        
        buffer.free()
        backend.free()
    }
    
    @Test
    fun testBackendCompatibility() {
        println("Testing backend compatibility...")
        
        val cpuBackend = GGMLCpuBackend()
        val metalBackend = GGMLMetalBackend()
        
        // Test buffer type compatibility
        val cpuBufferType = cpuBackend.getDefaultBufferType()
        val metalBufferType = metalBackend.getDefaultBufferType()
        
        // CPU should support host buffer types
        assertTrue(cpuBackend.supportsBufferType(cpuBufferType))
        
        // Metal buffer type should not be supported by CPU (non-host)
        assertFalse(cpuBackend.supportsBufferType(metalBufferType))
        
        // Test operation support
        val supportedOps = listOf(GGMLOp.ADD, GGMLOp.MUL, GGMLOp.MUL_MAT)
        val unsupportedOps = listOf(GGMLOp.NONE, GGMLOp.COUNT)
        
        supportedOps.forEach { op ->
            val tensor = GGMLTensor(op = op)
            assertTrue(cpuBackend.supportsOp(tensor), "CPU should support $op")
            // Metal stub should not support operations
            assertFalse(metalBackend.supportsOp(tensor), "Metal stub should not support $op")
        }
        
        unsupportedOps.forEach { op ->
            val tensor = GGMLTensor(op = op)
            assertFalse(cpuBackend.supportsOp(tensor), "CPU should not support $op")
            assertFalse(metalBackend.supportsOp(tensor), "Metal should not support $op")
        }
        
        cpuBackend.free()
        metalBackend.free()
    }
    
    @Test
    fun testGlobalBackendManager() {
        println("Testing global backend manager...")
        
        // Test convenience functions
        val graph = createGraphWithGlobalBackend(20, GGMLBackendSelectionStrategy.CPU_ONLY)
        assertNotNull(graph.allocator)
        assertNotNull(graph.allocator!!.backend)
        assertEquals("CPU", graph.allocator!!.backend!!.getName())
        
        // Test hybrid computation
        val status = computeGraphHybrid(graph)
        assertEquals(GGMLStatus.SUCCESS, status)
        
        // Test backend info
        val info = globalBackendManager.getBackendInfo()
        assertNotNull(info["availableBackends"])
        assertNotNull(info["primaryBackend"])
        assertNotNull(info["fallbackBackend"])
        
        @Suppress("UNCHECKED_CAST")
        val backends = info["availableBackends"] as List<String>
        assertTrue(backends.contains("CPU"), "Should have CPU backend")
    }
    
    @Test
    fun testGraphAllocatorBackendIntegration() {
        println("Testing graph allocator backend integration...")
        
        val backend = GGMLCpuBackend()
        val allocator = GGMLGraphAllocator(backend)
        
        // Verify backend integration
        assertEquals(backend, allocator.backend)
        assertTrue(allocator.backendBuffers.isNotEmpty())
        assertNotNull(allocator.backendBuffers[0])
        
        // Test buffer access
        val buffer = allocator.backendBuffers[0]
        assertTrue(buffer is GGMLCpuBuffer)
        assertTrue(buffer.getType().isHost())
        
        // Test fallback behavior
        val legacyAllocator = GGMLGraphAllocator()
        assertNull(legacyAllocator.backend)
        assertNotNull(legacyAllocator.buffers[0]) // Should still have ByteArray
        
        backend.free()
    }
    
    @Test
    fun testErrorHandling() {
        println("Testing error handling...")
        
        val backend = GGMLCpuBackend()
        
        // Test zero buffer allocation
        val zeroBuffer = backend.allocBuffer(0u)
        assertNull(zeroBuffer, "Zero size allocation should return null")
        
        // Test empty graph computation
        val graph = createGraph(0, backend)
        val status = backend.graphCompute(graph)
        assertEquals(GGMLStatus.SUCCESS, status, "Empty graph should succeed")
        
        // Test Metal stub operations
        val metalBackend = GGMLMetalBackend()
        val metalGraph = createGraph(5)
        val metalStatus = metalBackend.graphCompute(metalGraph)
        assertEquals(GGMLStatus.FAILED, metalStatus, "Metal stub should fail")
        
        backend.free()
        metalBackend.free()
    }
}