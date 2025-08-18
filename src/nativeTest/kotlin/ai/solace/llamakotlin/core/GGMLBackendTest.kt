package ai.solace.llamakotlin.core

import kotlin.test.*

/**
 * Test suite for the GGML backend abstraction system.
 * 
 * Tests the backend interfaces, CPU and Metal backend implementations,
 * buffer management, and integration with the graph allocator.
 */
class GGMLBackendTest {
    
    @Test
    fun testBackendRegistryInitialization() {
        // Test that the registry initializes properly
        val count = GGMLBackendRegistry.getCount()
        assertTrue(count >= 2, "Should have at least CPU and Metal backends registered")
        
        val cpuIndex = GGMLBackendRegistry.findByName("CPU")
        assertNotNull(cpuIndex, "CPU backend should be registered")
        
        val metalIndex = GGMLBackendRegistry.findByName("Metal")
        assertNotNull(metalIndex, "Metal backend should be registered")
        
        val cpuName = GGMLBackendRegistry.getName(cpuIndex!!)
        assertEquals("CPU", cpuName, "CPU backend name should match")
        
        val metalName = GGMLBackendRegistry.getName(metalIndex!!)
        assertEquals("Metal", metalName, "Metal backend name should match")
    }
    
    @Test
    fun testCpuBackendCreation() {
        // Test CPU backend creation
        val cpuIndex = GGMLBackendRegistry.findByName("CPU")
        assertNotNull(cpuIndex)
        
        val backend = GGMLBackendRegistry.initBackend(cpuIndex!!)
        assertNotNull(backend, "CPU backend should be created")
        assertTrue(backend is GGMLCpuBackend, "Should be a CPU backend instance")
        
        assertEquals("CPU", backend.getName())
        assertTrue(backend.getGuid().isNotEmpty(), "Backend should have a GUID")
        
        // Test buffer type
        val bufferType = backend.getDefaultBufferType()
        assertTrue(bufferType is GGMLCpuBufferType)
        assertTrue(bufferType.isHost(), "CPU buffer type should be host accessible")
        assertEquals("CPU", bufferType.getName())
        assertTrue(bufferType.getAlignment() > 0u, "Should have non-zero alignment")
    }
    
    @Test
    fun testMetalBackendCreation() {
        // Test Metal backend creation (stub)
        val metalIndex = GGMLBackendRegistry.findByName("Metal")
        assertNotNull(metalIndex)
        
        val backend = GGMLBackendRegistry.initBackend(metalIndex!!)
        assertNotNull(backend, "Metal backend should be created")
        assertTrue(backend is GGMLMetalBackend, "Should be a Metal backend instance")
        
        assertEquals("Metal", backend.getName())
        assertTrue(backend.getGuid().isNotEmpty(), "Backend should have a GUID")
        
        // Test buffer type
        val bufferType = backend.getDefaultBufferType()
        assertTrue(bufferType is GGMLMetalBufferType)
        assertFalse(bufferType.isHost(), "Metal buffer type should not be host accessible")
        assertEquals("Metal", bufferType.getName())
        assertTrue(bufferType.getAlignment() > 0u, "Should have non-zero alignment")
    }
    
    @Test
    fun testCpuBufferAllocation() {
        val backend = GGMLCpuBackend()
        val bufferType = backend.getDefaultBufferType()
        
        // Test successful allocation
        val buffer = bufferType.allocBuffer(1024u)
        assertNotNull(buffer, "Buffer allocation should succeed")
        assertTrue(buffer is GGMLCpuBuffer)
        
        assertEquals(1024u, buffer.getSize())
        assertEquals("CPU", buffer.getName())
        assertNotNull(buffer.getBase(), "Buffer should have base data")
        assertTrue(buffer.getBase() is ByteArray, "CPU buffer base should be ByteArray")
        
        // Test zero size allocation
        val zeroBuffer = bufferType.allocBuffer(0u)
        assertNull(zeroBuffer, "Zero size allocation should return null")
        
        // Cleanup
        buffer.free()
    }
    
    @Test
    fun testMetalBufferAllocation() {
        val backend = GGMLMetalBackend()
        val bufferType = backend.getDefaultBufferType()
        
        // Test allocation (should fail since it's a stub)
        val buffer = bufferType.allocBuffer(1024u)
        assertNull(buffer, "Metal buffer allocation should return null (stub implementation)")
    }
    
    @Test
    fun testCpuBufferOperations() {
        val bufferType = GGMLCpuBufferType()
        val buffer = bufferType.allocBuffer(1024u) as GGMLCpuBuffer
        
        // Test clear operation
        buffer.clear(42u)
        val data = buffer.getBase() as ByteArray
        assertEquals(42, data[0].toUByte().toInt())
        assertEquals(42, data[100].toUByte().toInt())
        
        // Test tensor operations
        val tensor = GGMLTensor(type = GGMLType.F32)
        tensor.ne[0] = 10L
        tensor.ne[1] = 1L
        tensor.nb[0] = 4u // F32 size
        tensor.nb[1] = 40u // 10 * F32 size
        tensor.dataOffset = 0u
        
        buffer.initTensor(tensor)
        assertEquals(buffer, tensor.buffer)
        
        // Test data operations
        val testData = ByteArray(40) { (it % 256).toByte() }
        buffer.setTensor(tensor, testData, 0u, 40u)
        
        val retrievedData = ByteArray(40)
        buffer.getTensor(tensor, retrievedData, 0u, 40u)
        
        assertContentEquals(testData, retrievedData)
        
        buffer.free()
    }
    
    @Test
    fun testBackendOperationSupport() {
        val cpuBackend = GGMLCpuBackend()
        
        // Test operation support
        val addTensor = GGMLTensor(op = GGMLOp.ADD)
        assertTrue(cpuBackend.supportsOp(addTensor), "CPU should support ADD operation")
        
        val mulMatTensor = GGMLTensor(op = GGMLOp.MUL_MAT)
        assertTrue(cpuBackend.supportsOp(mulMatTensor), "CPU should support MUL_MAT operation")
        
        val noneTensor = GGMLTensor(op = GGMLOp.NONE)
        assertFalse(cpuBackend.supportsOp(noneTensor), "CPU should not support NONE operation")
        
        // Test buffer type support
        val cpuBufferType = cpuBackend.getDefaultBufferType()
        assertTrue(cpuBackend.supportsBufferType(cpuBufferType), "CPU should support its own buffer type")
        
        val metalBackend = GGMLMetalBackend()
        val metalBufferType = metalBackend.getDefaultBufferType()
        
        // CPU should support Metal buffer type only if it's host accessible (it's not)
        assertFalse(cpuBackend.supportsBufferType(metalBufferType), "CPU should not support Metal buffer type")
    }
    
    @Test
    fun testGraphAllocatorWithBackend() {
        val backend = GGMLCpuBackend()
        val allocator = GGMLGraphAllocator(backend)
        
        // Verify backend integration
        assertEquals(backend, allocator.backend)
        assertTrue(allocator.backendBuffers.isNotEmpty())
        assertNotNull(allocator.backendBuffers[0], "Should have a backend buffer")
        assertTrue(allocator.backendBuffers[0] is GGMLCpuBuffer)
        
        // Test without backend (legacy)
        val legacyAllocator = GGMLGraphAllocator()
        assertNull(legacyAllocator.backend)
        assertTrue(legacyAllocator.backendBuffers.isNotEmpty())
        assertNull(legacyAllocator.backendBuffers[0], "Should not have backend buffer")
    }
    
    @Test
    fun testCpuGraphComputation() {
        val backend = GGMLCpuBackend()
        val allocator = GGMLGraphAllocator(backend)
        val graph = GGMLCGraph(allocator = allocator)
        
        // Test graph computation (will fail due to empty graph but should not crash)
        val result = backend.graphCompute(graph)
        assertEquals(GGMLStatus.SUCCESS, result, "Empty graph computation should succeed")
    }
    
    @Test
    fun testMetalGraphComputation() {
        val backend = GGMLMetalBackend()
        val graph = GGMLCGraph()
        
        // Test graph computation (should fail since it's a stub)
        val result = backend.graphCompute(graph)
        assertEquals(GGMLStatus.FAILED, result, "Metal graph computation should fail (stub)")
    }
    
    @Test
    fun testBackendBufferCompatibility() {
        val cpuBackend = GGMLCpuBackend()
        val cpuBufferType = cpuBackend.getDefaultBufferType()
        
        val metalBackend = GGMLMetalBackend()
        val metalBufferType = metalBackend.getDefaultBufferType()
        
        // Test buffer type compatibility
        assertTrue(cpuBufferType.isHost())
        assertFalse(metalBufferType.isHost())
        
        // Test alignment requirements
        assertTrue(cpuBufferType.getAlignment() >= 16u, "CPU should have at least 16-byte alignment")
        assertTrue(metalBufferType.getAlignment() >= 16u, "Metal should have at least 16-byte alignment")
        
        // Test max sizes
        assertTrue(cpuBufferType.getMaxSize() > 0u, "CPU should have positive max size")
        assertTrue(metalBufferType.getMaxSize() > 0u, "Metal should have positive max size")
    }
    
    @Test
    fun testTensorBufferIntegration() {
        val backend = GGMLCpuBackend()
        val buffer = backend.allocBuffer(1024u)
        assertNotNull(buffer)
        
        val tensor = GGMLTensor(type = GGMLType.F32)
        tensor.buffer = buffer
        
        // Verify integration
        assertEquals(buffer, tensor.buffer)
        assertTrue(tensor.buffer is GGMLCpuBuffer)
        
        // Test backend-specific operations
        assertTrue(backend.supportsOp(tensor))
        assertTrue(backend.supportsBufferType(buffer.getType()))
        
        buffer.free()
    }
}