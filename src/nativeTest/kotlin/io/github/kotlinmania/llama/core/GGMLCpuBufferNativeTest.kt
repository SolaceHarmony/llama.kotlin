@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.ore

import kotlinx.cinterop.toLong
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertContentEquals

/**
 * Tests for [GGMLCpuBufferTypeNative], [GGMLCpuBufferNative], and
 * [GGMLCpuBufferFromPtrNative].
 *
 * Validates that the native CPU buffer vtable matches the C original:
 * - `ggml_backend_cpu_buffer_type_alloc_buffer`  (ggml-backend.cpp:2305–2313)
 * - `ggml_backend_cpu_buffer_get_base`            (ggml-backend.cpp:2213–2223)
 * - `ggml_backend_cpu_buffer_set_tensor`          (ggml-backend.cpp:2237–2241)
 * - `ggml_backend_cpu_buffer_get_tensor`          (ggml-backend.cpp:2244–2248)
 * - `ggml_backend_cpu_buffer_cpy_tensor`          (ggml-backend.cpp:2251–2260)
 * - `ggml_backend_cpu_buffer_clear`               (ggml-backend.cpp:2262–2265)
 * - `ggml_backend_cpu_buffer_memset_tensor`       (ggml-backend.cpp:2230–2234)
 * - `ggml_backend_cpu_buffer_free_buffer`         (ggml-backend.cpp:2225–2228)
 * - `ggml_backend_cpu_buffer_from_ptr_i`          (ggml-backend.cpp:2281–2293)
 */
class GGMLCpuBufferNativeTest {

    // ========================================================================
    // GGMLCpuBufferTypeNative
    // ========================================================================

    @Test
    fun bufferTypeNameIsCPU() {
        val buft = GGMLCpuBufferTypeNative()
        assertEquals("CPU", buft.getName())
    }

    @Test
    fun bufferTypeAlignmentIs32() {
        // C: TENSOR_ALIGNMENT is 32 (ggml-impl.h line 44).
        val buft = GGMLCpuBufferTypeNative()
        assertEquals(32u, buft.getAlignment())
    }

    @Test
    fun bufferTypeIsHost() {
        val buft = GGMLCpuBufferTypeNative()
        assertTrue(buft.isHost(), "CPU buffer type must be a host buffer")
    }

    @Test
    fun bufferTypeAllocBufferReturnsNonNullForPositiveSize() {
        val buft = GGMLCpuBufferTypeNative()
        val buf = buft.allocBuffer(1024u)
        assertNotNull(buf)
        assertTrue(buf is GGMLCpuBufferNative, "Must return native buffer")
        assertEquals(1024u, buf.getSize())
        buf.free()
    }

    @Test
    fun bufferTypeAllocBufferReturnsNullForZeroSize() {
        // C: alloc_buffer with size=0 returns NULL.
        val buft = GGMLCpuBufferTypeNative()
        assertNull(buft.allocBuffer(0u), "Zero-size allocation must return null")
    }

    @Test
    fun bufferTypeMaxSizeIsULongMax() {
        val buft = GGMLCpuBufferTypeNative()
        assertEquals(ULong.MAX_VALUE, buft.getMaxSize())
    }

    // ========================================================================
    // GGMLCpuBufferNative — base and metadata
    // ========================================================================

    private fun allocBuffer(size: ULong = 4096u): GGMLCpuBufferNative {
        val buft = GGMLCpuBufferTypeNative()
        return buft.allocBuffer(size)!! as GGMLCpuBufferNative
    }

    @Test
    fun bufferGetBaseReturnsNativeAlignedBuffer() {
        val buf = allocBuffer()
        val base = buf.getBase()
        assertTrue(base is NativeAlignedBuffer, "getBase must return NativeAlignedBuffer")
        buf.free()
    }

    @Test
    fun bufferGetSizeMatchesAllocated() {
        val buf = allocBuffer(2048u)
        assertEquals(2048u, buf.getSize())
        buf.free()
    }

    @Test
    fun bufferNameIsCPU() {
        val buf = allocBuffer()
        assertEquals("CPU", buf.getName())
        buf.free()
    }

    @Test
    fun bufferGetTypeReturnsCpuBufferType() {
        val buf = allocBuffer()
        assertTrue(buf.getType() is GGMLCpuBufferTypeNative)
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — initTensor
    // ========================================================================

    private fun makeTensor(
        type: GGMLType = GGMLType.F32,
        ne0: Long = 16,
        dataOffset: ULong = 0u
    ): GGMLTensor {
        val t = GGMLTensor(type = type)
        // C convention: unused dims are 1, not 0 (mirrors ggml_new_tensor_1d)
        t.ne[0] = ne0
        t.ne[1] = 1
        t.ne[2] = 1
        t.ne[3] = 1
        // Set strides matching C layout
        val bs = type.byteSize.toUInt().toULong()
        t.nb[0] = bs
        t.nb[1] = bs * ne0.toULong()
        t.nb[2] = t.nb[1]
        t.nb[3] = t.nb[2]
        t.dataOffset = dataOffset
        return t
    }

    @Test
    fun initTensorSetsTensorBufferAndData() {
        val buf = allocBuffer()
        val tensor = makeTensor()

        val status = buf.initTensor(tensor)
        assertEquals(GGMLStatus.SUCCESS, status)
        assertEquals(buf, tensor.buffer, "initTensor must set tensor.buffer")
        assertTrue(tensor.data is NativeAlignedBuffer, "initTensor must set tensor.data to NativeAlignedBuffer")
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — setTensor / getTensor roundtrip
    // ========================================================================

    @Test
    fun setGetTensorRoundTrip() {
        val buf = allocBuffer()
        val tensor = makeTensor(ne0 = 10)
        tensor.dataOffset = 0u
        buf.initTensor(tensor)

        // 10 F32 elements = 40 bytes
        val payload = ByteArray(40) { (it * 3 + 7).toByte() }
        buf.setTensor(tensor, payload, 0u, 40u)

        val result = ByteArray(40)
        buf.getTensor(tensor, result, 0u, 40u)
        assertContentEquals(payload, result, "set/get tensor roundtrip must be exact")
        buf.free()
    }

    @Test
    fun setGetTensorWithOffset() {
        // Validates that tensor.dataOffset is correctly applied.
        // In C: memcpy((char *)tensor->data + offset, data, size)
        // where tensor->data = buffer_get_base() + tensor->view_offs
        val buf = allocBuffer(8192u)
        val tensor = makeTensor(ne0 = 8)
        tensor.dataOffset = 256u // tensor data starts 256 bytes into the buffer
        buf.initTensor(tensor)

        val payload = ByteArray(32) { i -> (i + 1).toByte() }
        buf.setTensor(tensor, payload, 0u, 32u)

        val result = ByteArray(32)
        buf.getTensor(tensor, result, 0u, 32u)
        assertContentEquals(payload, result)
        buf.free()
    }

    @Test
    fun setGetTensorPartialOffset() {
        // Write to a sub-range of the tensor using the offset parameter.
        val buf = allocBuffer()
        val tensor = makeTensor(ne0 = 16)
        tensor.dataOffset = 0u
        buf.initTensor(tensor)

        // Zero the whole tensor area
        buf.clear(0u)

        // Write 8 bytes at tensor offset 16 (i.e., starting at the 5th float)
        val patch = ByteArray(8) { 0xFF.toByte() }
        buf.setTensor(tensor, patch, 16u, 8u)

        // Read the full 64 bytes
        val full = ByteArray(64)
        buf.getTensor(tensor, full, 0u, 64u)

        // Bytes 0..15 should be zero
        for (i in 0 until 16) {
            assertEquals(0.toByte(), full[i], "Byte $i before patch should be 0")
        }
        // Bytes 16..23 should be 0xFF
        for (i in 16 until 24) {
            assertEquals(0xFF.toByte(), full[i], "Byte $i in patch should be 0xFF")
        }
        // Bytes 24..63 should be zero
        for (i in 24 until 64) {
            assertEquals(0.toByte(), full[i], "Byte $i after patch should be 0")
        }
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — copyTensor
    // ========================================================================

    @Test
    fun copyTensorBetweenNativeBuffers() {
        val bufA = allocBuffer()
        val bufB = allocBuffer()

        val src = makeTensor(ne0 = 8)
        src.dataOffset = 0u
        bufA.initTensor(src)

        val dst = makeTensor(ne0 = 8)
        dst.dataOffset = 0u
        bufB.initTensor(dst)

        // Write pattern to source
        val pattern = ByteArray(32) { (it * 5).toByte() }
        bufA.setTensor(src, pattern, 0u, 32u)

        // Verify the write worked by reading back from bufA
        val readback = ByteArray(32)
        bufA.getTensor(src, readback, 0u, 32u)
        assertContentEquals(pattern, readback, "setTensor/getTensor on bufA should round-trip")

        // Copy src → dst
        val ok = bufB.copyTensor(src, dst)
        assertTrue(ok, "copyTensor between two host buffers must succeed")

        // Verify dst contains the same data
        val result = ByteArray(32)
        bufB.getTensor(dst, result, 0u, 32u)
        assertContentEquals(pattern, result)

        bufB.free()
        bufA.free()
    }

    @Test
    fun copyTensorFromNullBufferReturnsFalse() {
        val buf = allocBuffer()
        val src = makeTensor()
        // src.buffer is null (not init'd)
        val dst = makeTensor()
        buf.initTensor(dst)

        assertFalse(buf.copyTensor(src, dst), "copyTensor with null src buffer must return false")
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — clear
    // ========================================================================

    @Test
    fun clearSetsAllBytesToValue() {
        val buf = allocBuffer(128u)

        // Fill with non-zero, then clear to 0xCD
        buf.nativeBuffer.fill(0xFFu)
        buf.clear(0xCDu)

        val out = ByteArray(128)
        buf.nativeBuffer.copyToByteArray(out, 0, 0, 128)
        assertTrue(out.all { it == 0xCD.toByte() }, "clear(0xCD) must set all bytes")
        buf.free()
    }

    @Test
    fun clearToZero() {
        val buf = allocBuffer(64u)
        buf.nativeBuffer.fill(0xFFu)
        buf.clear(0u)

        val out = ByteArray(64)
        buf.nativeBuffer.copyToByteArray(out, 0, 0, 64)
        assertTrue(out.all { it == 0.toByte() }, "clear(0) must zero the buffer")
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — memsetTensor
    // ========================================================================

    @Test
    fun memsetTensorSetsRegion() {
        val buf = allocBuffer(256u)
        buf.clear(0u)

        val tensor = makeTensor(ne0 = 32)
        tensor.dataOffset = 64u
        buf.initTensor(tensor)

        // memset 16 bytes starting at tensor-relative offset 8
        buf.memsetTensor(tensor, 0xAAu, offset = 8, size = 16)

        val full = ByteArray(256)
        buf.nativeBuffer.copyToByteArray(full, 0, 0, 256)

        // The absolute region [64+8, 64+8+16) = [72, 88) should be 0xAA
        for (i in 0 until 256) {
            val expected = if (i in 72 until 88) 0xAA.toByte() else 0.toByte()
            assertEquals(expected, full[i], "Byte at abs offset $i")
        }
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — getTensorPointer
    // ========================================================================

    @Test
    fun getTensorPointerOffsetsCorrectly() {
        val buf = allocBuffer(1024u)
        val tensor = makeTensor()
        tensor.dataOffset = 128u
        buf.initTensor(tensor)

        val ptr = buf.getTensorPointer(tensor)
        val baseAddr = buf.nativeBuffer.getAlignedBase().toLong()
        val ptrAddr = ptr.toLong()

        assertEquals(
            baseAddr + 128, ptrAddr,
            "getTensorPointer must return base + dataOffset"
        )
        buf.free()
    }

    // ========================================================================
    // GGMLCpuBufferNative — free
    // ========================================================================

    @Test
    fun freeDoesNotCrash() {
        val buf = allocBuffer()
        buf.clear(0x42u) // use the buffer
        buf.free()
        // If we reach here, free didn't crash.
    }

    // ========================================================================
    // GGMLCpuBufferFromPtrNative — external pointer wrapper
    // ========================================================================

    @Test
    fun fromPtrBufferNameIsCPU() {
        val nab = NativeAlignedBuffer.alloc(256)!!
        val buf = GGMLCpuBufferFromPtrNative(nab, 256u)
        assertEquals("CPU", buf.getName())
        // free is a no-op on from-ptr buffers; clean up the actual allocation ourselves
        buf.free()
        nab.free()
    }

    @Test
    fun fromPtrBufferGetBaseReturnsNativeAlignedBuffer() {
        val nab = NativeAlignedBuffer.alloc(256)!!
        val buf = GGMLCpuBufferFromPtrNative(nab, 256u)
        assertTrue(buf.getBase() is NativeAlignedBuffer)
        buf.free()
        nab.free()
    }

    @Test
    fun fromPtrBufferFreeDoesNotReleaseMemory() {
        val nab = NativeAlignedBuffer.alloc(256)!!
        val buf = GGMLCpuBufferFromPtrNative(nab, 256u)

        // Write data before "freeing" the wrapper
        nab.setByte(0, 42)

        buf.free() // should be a no-op

        // Verify the memory is still accessible (not freed)
        assertEquals(42.toByte(), nab.getByte(0))

        nab.free() // actually free
    }

    @Test
    fun fromPtrSetGetTensorRoundTrip() {
        val nab = NativeAlignedBuffer.alloc(512)!!
        val buf = GGMLCpuBufferFromPtrNative(nab, 512u)

        val tensor = makeTensor(ne0 = 4)
        tensor.dataOffset = 0u
        buf.initTensor(tensor)

        val payload = ByteArray(16) { (it + 10).toByte() }
        buf.setTensor(tensor, payload, 0u, 16u)

        val result = ByteArray(16)
        buf.getTensor(tensor, result, 0u, 16u)
        assertContentEquals(payload, result)

        buf.free()
        nab.free()
    }

    @Test
    fun fromPtrClearWorks() {
        val nab = NativeAlignedBuffer.alloc(64)!!
        nab.fill(0xFFu)

        val buf = GGMLCpuBufferFromPtrNative(nab, 64u)
        buf.clear(0u)

        val out = ByteArray(64)
        nab.copyToByteArray(out, 0, 0, 64)
        assertTrue(out.all { it == 0.toByte() })

        buf.free()
        nab.free()
    }

    // ========================================================================
    // expect/actual factory — ggml_aligned_malloc / ggml_aligned_free
    // ========================================================================

    @Test
    fun ggmlAlignedMallocReturnsNativeAlignedBuffer() {
        val ptr = ggml_aligned_malloc(512)
        assertNotNull(ptr)
        assertTrue(ptr is NativeAlignedBuffer)

        val nab = ptr
        assertEquals(512L, nab.sizeBytes)

        // Verify alignment
        assertEquals(0L, nab.pointer.toLong() % NativeAlignedBuffer.MALLOC_ALIGNMENT)

        ggml_aligned_free(ptr, 512)
    }

    @Test
    fun ggmlAlignedFreeHandlesNull() {
        // Must not crash on null — mirrors C behavior where free(NULL) is a no-op.
        ggml_aligned_free(null, 0)
    }

    @Test
    fun ggmlAlignedFreeHandlesNonBuffer() {
        // If passed something that isn't a NativeAlignedBuffer, it's a no-op.
        ggml_aligned_free("not a buffer", 0)
    }

    @Test
    fun createDefaultCpuBufferTypeReturnsNativeType() {
        val buft = createDefaultCpuBufferType()
        assertTrue(buft is GGMLCpuBufferTypeNative)
        assertEquals("CPU", buft.getName())
        assertEquals(32u, buft.getAlignment())
    }
}
