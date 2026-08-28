package io.github.kotlinmania.llama.core

import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertContentEquals
import kotlin.test.assertFalse

class GGMLBackendTest {
    @BeforeTest
    fun ensureRegistryInitialized() {
        io.github.kotlinmania.llama.ore.GGMLBackendRegistry.init()
    }

    @Test
    fun cpuBackendIsRegistered() {
        val index = io.github.kotlinmania.llama.ore.GGMLBackendRegistry.findByName("CPU")
        assertNotNull(index, "CPU backend should be registered")
        assertEquals("CPU", io.github.kotlinmania.llama.ore.GGMLBackendRegistry.getName(index))
        assertEquals(1, io.github.kotlinmania.llama.ore.GGMLBackendRegistry.getCount(), "Only CPU backend expected")
    }

    @Test
    fun cpuBackendCreationAndBufferType() {
        val cpuIndex = io.github.kotlinmania.llama.ore.GGMLBackendRegistry.findByName("CPU")
        assertNotNull(cpuIndex)

        val backend = io.github.kotlinmania.llama.ore.GGMLBackendRegistry.initBackend(cpuIndex)
        assertTrue(backend is io.github.kotlinmania.llama.ore.GGMLCpuBackend)
        assertEquals("CPU", backend.getName())
        assertTrue(backend.getGuid().isNotEmpty())

        val bufferType = backend.getDefaultBufferType()
        assertTrue(bufferType is io.github.kotlinmania.llama.ore.GGMLCpuBufferType)
        assertTrue(bufferType.isHost())
        assertEquals("CPU", bufferType.getName())
        assertTrue(bufferType.getAlignment() > 0u)
    }

    @Test
    fun cpuBufferLifecycle() {
        val backend = io.github.kotlinmania.llama.ore.GGMLCpuBackend()
        val bufferType = backend.getDefaultBufferType()

        val buffer = bufferType.allocBuffer(1024u)
        assertNotNull(buffer)
        assertTrue(buffer is io.github.kotlinmania.llama.ore.GGMLCpuBuffer)
        assertEquals(1024u, buffer.getSize())
        assertTrue(buffer.getBase() is ByteArray)

        val zero = bufferType.allocBuffer(0u)
        assertNull(zero, "Zero-sized allocation should return null")

        buffer.free()
        backend.free()
    }

    @Test
    fun cpuBufferReadWriteRoundTrip() {
        val buffer = io.github.kotlinmania.llama.ore.GGMLCpuBufferType()
            .allocBuffer(1024u) as io.github.kotlinmania.llama.ore.GGMLCpuBuffer

        val tensor =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        tensor.ne[0] = 10L
        tensor.nb[0] = 4u
        tensor.dataOffset = 0u
        buffer.initTensor(tensor)

        val payload = ByteArray(40) { (it * 3).toByte() }
        buffer.setTensor(tensor, payload, 0u, payload.size.toULong())

        val roundTrip = ByteArray(payload.size)
        buffer.getTensor(tensor, roundTrip, 0u, roundTrip.size.toULong())
        assertContentEquals(payload, roundTrip)

        buffer.clear(0u)
        val cleared = buffer.getBase() as ByteArray
        assertTrue(cleared.all { it == 0.toByte() })

        buffer.free()
    }

    @Test
    fun cpuBackendSupportsExpectedOps() {
        val backend = io.github.kotlinmania.llama.ore.GGMLCpuBackend()

        val supportedOps = listOf(io.github.kotlinmania.llama.ore.GGMLOp.ADD, io.github.kotlinmania.llama.ore.GGMLOp.MUL, io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT)
        supportedOps.forEach { op ->
            val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(op = op)
            assertTrue(backend.supportsOp(tensor), "CPU should support $op")
        }

        val unsupported =
            io.github.kotlinmania.llama.ore.GGMLTensor(op = io.github.kotlinmania.llama.ore.GGMLOp.NONE)
        assertFalse(backend.supportsOp(unsupported))

        val bufferType = backend.getDefaultBufferType()
        assertTrue(backend.supportsBufferType(bufferType))

        backend.free()
    }

    @Test
    fun graphAllocatorBindsCpuBackendBuffers() {
        val backend = io.github.kotlinmania.llama.ore.GGMLCpuBackend()
        val allocator = io.github.kotlinmania.llama.ore.GGMLGraphAllocator(backend)

        assertEquals(backend, allocator.backend)
        assertTrue(allocator.backendBuffers.first() is io.github.kotlinmania.llama.ore.GGMLCpuBuffer)

        val graph = io.github.kotlinmania.llama.ore.GGMLCGraph(allocator = allocator)
        val status = backend.graphCompute(graph)
        assertEquals(io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS, status)

        backend.free()
    }
}
