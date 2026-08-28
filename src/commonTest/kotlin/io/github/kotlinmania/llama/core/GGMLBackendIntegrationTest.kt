package io.github.kotlinmania.llama.core

import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class GGMLBackendIntegrationTest {

    @BeforeTest
    fun setUp() {
        io.github.kotlinmania.llama.ore.GGMLBackendRegistry.init()
    }

    @Test
    fun endToEndCpuExecution() {
        val backend = io.github.kotlinmania.llama.ore.GGMLCpuBackend()
        val graph = io.github.kotlinmania.llama.ore.createGraph(5, backend)
        val status = backend.graphCompute(graph)
        assertEquals(io.github.kotlinmania.llama.ore.GGMLStatus.SUCCESS, status)
        backend.free()
    }

    @Test
    fun bufferDataIntegrity() {
        val backend = io.github.kotlinmania.llama.ore.GGMLCpuBackend()
        val buffer = backend.allocBuffer(64u) as io.github.kotlinmania.llama.ore.GGMLCpuBuffer

        val tensor =
            io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
        tensor.ne[0] = 4L
        tensor.nb[0] = 4u
        tensor.dataOffset = 0u
        buffer.initTensor(tensor)

        val payload = ByteArray(16) { it.toByte() }
        buffer.setTensor(tensor, payload, 0u, payload.size.toULong())

        val roundTrip = ByteArray(payload.size)
        buffer.getTensor(tensor, roundTrip, 0u, roundTrip.size.toULong())
        assertContentEquals(payload, roundTrip)

        buffer.free()
        backend.free()
    }
}
