@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.platform.nativeAlignedAlloc
import io.github.kotlinmania.llama.platform.nativeAlignedFree
import kotlinx.cinterop.toLong
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for [NativeAlignedBuffer].
 *
 * Validates that the Kotlin/Native memory wrapper behaves identically to the C
 * original (`ggml_aligned_malloc` / `ggml_aligned_free` in ggml.c lines 326–397,
 * and the CPU buffer vtable in ggml-backend.cpp lines 2211–2293).
 *
 * Every test allocates and frees its own buffer to avoid leaks.
 */
class NativeAlignedBufferTest {

    // ========================================================================
    // Allocation and alignment
    // ========================================================================

    @Test
    fun allocReturnsNonNullForPositiveSize() {
        val buf = NativeAlignedBuffer.alloc(256)
        assertNotNull(buf, "alloc(256) must not return null")
        buf.free()
    }

    @Test
    fun allocReturnsNullForZeroSize() {
        // C: ggml_aligned_malloc(0) prints a warning and returns NULL.
        val buf = NativeAlignedBuffer.alloc(0)
        assertNull(buf, "alloc(0) must return null")
    }

    @Test
    fun allocatedPointerIs64ByteAligned() {
        // C: posix_memalign guarantees alignment to the requested boundary (64).
        val buf = NativeAlignedBuffer.alloc(4096)!!
        val addr = buf.pointer.toLong()
        assertEquals(
            0L, addr % NativeAlignedBuffer.MALLOC_ALIGNMENT,
            "Pointer 0x${addr.toString(16)} must be 64-byte aligned"
        )
        buf.free()
    }

    @Test
    fun sizeBytesMatchesRequestedSize() {
        val requested = 1024L
        val buf = NativeAlignedBuffer.alloc(requested)!!
        assertEquals(requested, buf.sizeBytes, "sizeBytes must equal the requested allocation size")
        buf.free()
    }

    @Test
    fun allocWithCustomAlignmentIsHonored() {
        // Verify the alignment parameter overrides the default.
        val alignment = 128
        val buf = NativeAlignedBuffer.alloc(512, alignment)!!
        val addr = buf.pointer.toLong()
        assertEquals(
            0L, addr % alignment,
            "Pointer 0x${addr.toString(16)} must be $alignment-byte aligned"
        )
        buf.free()
    }

    @Test
    fun multipleAllocationsReturnDistinctPointers() {
        val a = NativeAlignedBuffer.alloc(256)!!
        val b = NativeAlignedBuffer.alloc(256)!!
        assertTrue(
            a.pointer.toLong() != b.pointer.toLong(),
            "Two allocations must return distinct pointers"
        )
        b.free()
        a.free()
    }

    // ========================================================================
    // getAlignedBase — TENSOR_ALIGNMENT padding
    // ========================================================================

    @Test
    fun getAlignedBaseReturnsAlignedPointer() {
        // Because MALLOC_ALIGNMENT (64) >= TENSOR_ALIGNMENT (32), the base
        // pointer is already tensor-aligned.  getAlignedBase should return
        // the same pointer.
        val buf = NativeAlignedBuffer.alloc(4096)!!
        val baseAddr = buf.getAlignedBase().toLong()
        assertEquals(
            0L, baseAddr % NativeAlignedBuffer.TENSOR_ALIGNMENT,
            "getAlignedBase must be 32-byte aligned"
        )
        // Since 64 is a multiple of 32, base should equal the original pointer.
        assertEquals(buf.pointer.toLong(), baseAddr, "64-aligned pointer is already 32-aligned")
        buf.free()
    }

    // ========================================================================
    // Typed element access — roundtrip correctness
    // ========================================================================

    @Test
    fun byteRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(16)!!
        buf.setByte(0, 42)
        buf.setByte(7, -1)
        buf.setByte(15, Byte.MAX_VALUE)
        assertEquals(42.toByte(), buf.getByte(0))
        assertEquals((-1).toByte(), buf.getByte(7))
        assertEquals(Byte.MAX_VALUE, buf.getByte(15))
        buf.free()
    }

    @Test
    fun ubyteRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(16)!!
        buf.setUByte(0, 0u)
        buf.setUByte(1, 255u)
        buf.setUByte(2, 128u)
        assertEquals(0u.toUByte(), buf.getUByte(0))
        assertEquals(255u.toUByte(), buf.getUByte(1))
        assertEquals(128u.toUByte(), buf.getUByte(2))
        buf.free()
    }

    @Test
    fun floatRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(32)!!
        val values = floatArrayOf(0.0f, 1.0f, -1.0f, Float.MAX_VALUE, Float.MIN_VALUE, 3.14159f)
        for ((i, v) in values.withIndex()) {
            buf.setFloat((i * 4).toLong(), v)
        }
        for ((i, v) in values.withIndex()) {
            assertEquals(v, buf.getFloat((i * 4).toLong()), "Float at offset ${i * 4}")
        }
        buf.free()
    }

    @Test
    fun floatSpecialValuesPreserved() {
        val buf = NativeAlignedBuffer.alloc(16)!!
        buf.setFloat(0, Float.NaN)
        buf.setFloat(4, Float.POSITIVE_INFINITY)
        buf.setFloat(8, Float.NEGATIVE_INFINITY)
        assertTrue(buf.getFloat(0).isNaN(), "NaN must roundtrip")
        assertEquals(Float.POSITIVE_INFINITY, buf.getFloat(4))
        assertEquals(Float.NEGATIVE_INFINITY, buf.getFloat(8))
        buf.free()
    }

    @Test
    fun intRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(16)!!
        buf.setInt(0, 0)
        buf.setInt(4, Int.MAX_VALUE)
        buf.setInt(8, Int.MIN_VALUE)
        buf.setInt(12, -42)
        assertEquals(0, buf.getInt(0))
        assertEquals(Int.MAX_VALUE, buf.getInt(4))
        assertEquals(Int.MIN_VALUE, buf.getInt(8))
        assertEquals(-42, buf.getInt(12))
        buf.free()
    }

    @Test
    fun shortRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(8)!!
        buf.setShort(0, 0)
        buf.setShort(2, Short.MAX_VALUE)
        buf.setShort(4, Short.MIN_VALUE)
        buf.setShort(6, -1)
        assertEquals(0.toShort(), buf.getShort(0))
        assertEquals(Short.MAX_VALUE, buf.getShort(2))
        assertEquals(Short.MIN_VALUE, buf.getShort(4))
        assertEquals((-1).toShort(), buf.getShort(6))
        buf.free()
    }

    @Test
    fun longRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(24)!!
        buf.setLong(0, 0L)
        buf.setLong(8, Long.MAX_VALUE)
        buf.setLong(16, Long.MIN_VALUE)
        assertEquals(0L, buf.getLong(0))
        assertEquals(Long.MAX_VALUE, buf.getLong(8))
        assertEquals(Long.MIN_VALUE, buf.getLong(16))
        buf.free()
    }

    @Test
    fun mixedTypesAtDifferentOffsets() {
        // Verify that writing different types into the same buffer at non-overlapping
        // offsets does not corrupt adjacent data.
        val buf = NativeAlignedBuffer.alloc(64)!!
        buf.setByte(0, 0xAB.toByte())       // offset 0:  1 byte
        buf.setShort(2, 0x1234)              // offset 2:  2 bytes
        buf.setInt(4, 0x0BADF00D)            // offset 4:  4 bytes
        buf.setFloat(8, 2.71828f)            // offset 8:  4 bytes
        buf.setLong(16, 0x0102030405060708L) // offset 16: 8 bytes

        assertEquals(0xAB.toByte(), buf.getByte(0))
        assertEquals(0x1234.toShort(), buf.getShort(2))
        assertEquals(0x0BADF00D, buf.getInt(4))
        assertEquals(2.71828f, buf.getFloat(8))
        assertEquals(0x0102030405060708L, buf.getLong(16))
        buf.free()
    }

    // ========================================================================
    // Bulk copy — ByteArray ↔ NativeAlignedBuffer
    // ========================================================================

    @Test
    fun copyFromByteArrayRoundTrip() {
        val buf = NativeAlignedBuffer.alloc(256)!!
        val src = ByteArray(100) { (it and 0xFF).toByte() }

        buf.copyFromByteArray(src, srcOffset = 0, dstOffset = 0, length = src.size)

        val dst = ByteArray(100)
        buf.copyToByteArray(dst, srcOffset = 0, dstOffset = 0, length = dst.size)
        assertContentEquals(src, dst, "Bulk copy roundtrip must be exact")
        buf.free()
    }

    @Test
    fun copyFromByteArrayWithOffsets() {
        val buf = NativeAlignedBuffer.alloc(256)!!
        buf.fill(0u) // zero the buffer

        val src = byteArrayOf(10, 20, 30, 40, 50, 60, 70, 80)
        // Copy bytes [2..5] from src into buffer at offset 100
        buf.copyFromByteArray(src, srcOffset = 2, dstOffset = 100, length = 4)

        val dst = ByteArray(4)
        buf.copyToByteArray(dst, srcOffset = 100, dstOffset = 0, length = 4)
        assertContentEquals(byteArrayOf(30, 40, 50, 60), dst)
        buf.free()
    }

    @Test
    fun copyBetweenNativeBuffers() {
        val a = NativeAlignedBuffer.alloc(128)!!
        val b = NativeAlignedBuffer.alloc(128)!!

        // Write a known pattern into buffer a
        val pattern = ByteArray(64) { (it * 7).toByte() }
        a.copyFromByteArray(pattern, 0, 0, pattern.size)

        // Copy from a into b at offset 32
        b.copyFrom(a, srcOffset = 0, dstOffset = 32, length = 64)

        // Verify by reading b back to a ByteArray
        val result = ByteArray(64)
        b.copyToByteArray(result, srcOffset = 32, dstOffset = 0, length = 64)
        assertContentEquals(pattern, result, "Native-to-native copy must be exact")

        b.free()
        a.free()
    }

    // ========================================================================
    // Fill and memset
    // ========================================================================

    @Test
    fun fillSetsEntireBuffer() {
        val buf = NativeAlignedBuffer.alloc(64)!!
        buf.fill(0xABu)

        val out = ByteArray(64)
        buf.copyToByteArray(out, 0, 0, 64)
        assertTrue(out.all { it == 0xAB.toByte() }, "fill must set all bytes to 0xAB")
        buf.free()
    }

    @Test
    fun fillWithZeroClearsBuffer() {
        val buf = NativeAlignedBuffer.alloc(64)!!
        // Write non-zero data first
        buf.fill(0xFFu)
        // Clear
        buf.fill(0u)

        val out = ByteArray(64)
        buf.copyToByteArray(out, 0, 0, 64)
        assertTrue(out.all { it == 0.toByte() }, "fill(0) must zero the buffer")
        buf.free()
    }

    @Test
    fun memsetRegionIsTargeted() {
        val buf = NativeAlignedBuffer.alloc(64)!!
        buf.fill(0u) // start with zeros

        // Set bytes [16..31] to 0xFF
        buf.memsetRegion(0xFFu, offset = 16, length = 16)

        val out = ByteArray(64)
        buf.copyToByteArray(out, 0, 0, 64)

        // Bytes before the region are still zero
        for (i in 0 until 16) {
            assertEquals(0.toByte(), out[i], "Byte $i before memset region should be 0")
        }
        // Bytes in the region are 0xFF
        for (i in 16 until 32) {
            assertEquals(0xFF.toByte(), out[i], "Byte $i in memset region should be 0xFF")
        }
        // Bytes after the region are still zero
        for (i in 32 until 64) {
            assertEquals(0.toByte(), out[i], "Byte $i after memset region should be 0")
        }
        buf.free()
    }

    // ========================================================================
    // fromPointer — externally owned memory
    // ========================================================================

    @Test
    fun fromPointerWrapsExternalMemory() {
        // Allocate memory externally, wrap it, verify access.
        val size = 256L
        val rawPtr = nativeAlignedAlloc(size, NativeAlignedBuffer.MALLOC_ALIGNMENT)
        assertNotNull(rawPtr, "native aligned allocation must succeed")
        val buf = NativeAlignedBuffer.fromPointer(rawPtr, size)

        // Write through the wrapper
        buf.setByte(0, 99)
        buf.setFloat(4, 1.5f)
        assertEquals(99.toByte(), buf.getByte(0))
        assertEquals(1.5f, buf.getFloat(4))

        // free does NOT release because the pointer is externally owned.
        buf.free()

        // The pointer is still valid (we own it), write again to prove no crash.
        buf.setByte(0, 1)
        assertEquals(1.toByte(), buf.getByte(0))

        nativeAlignedFree(rawPtr)
    }

    // ========================================================================
    // Lifecycle — free
    // ========================================================================

    @Test
    fun freeOnOwnedBufferDoesNotCrash() {
        val buf = NativeAlignedBuffer.alloc(128)!!
        // Write some data so the buffer is "used"
        buf.setInt(0, 42)
        buf.free()
        // If we get here, free() did not crash.
    }

    @Test
    fun toStringContainsSizeAndAddress() {
        val buf = NativeAlignedBuffer.alloc(512)!!
        val str = buf.toString()
        assertTrue(str.contains("512"), "toString must include size")
        assertTrue(str.contains("NativeAlignedBuffer"), "toString must include class name")
        buf.free()
    }

    // ========================================================================
    // Large allocation — validates the allocator handles realistic sizes
    // ========================================================================

    @Test
    fun allocLargeBufferSucceeds() {
        // 16 MB — typical tensor buffer size for small model layers
        val sixteenMB = 16L * 1024 * 1024
        val buf = NativeAlignedBuffer.alloc(sixteenMB)
        assertNotNull(buf, "16 MB allocation must succeed")
        assertEquals(sixteenMB, buf.sizeBytes)

        // Verify alignment still holds
        assertEquals(0L, buf.pointer.toLong() % NativeAlignedBuffer.MALLOC_ALIGNMENT)

        // Smoke-test: write and read at the boundaries
        buf.setFloat(0, 1.0f)
        buf.setFloat(sixteenMB - 4, 2.0f)
        assertEquals(1.0f, buf.getFloat(0))
        assertEquals(2.0f, buf.getFloat(sixteenMB - 4))

        buf.free()
    }

    // ========================================================================
    // Stress — sequential float array pattern
    // ========================================================================

    @Test
    fun floatArrayPatternIntegrity() {
        // Write 1024 floats, read them all back, verify each value.
        // This validates that stride arithmetic is correct for sequential access.
        val count = 1024
        val buf = NativeAlignedBuffer.alloc(count.toLong() * 4)!!

        for (i in 0 until count) {
            buf.setFloat((i * 4).toLong(), i.toFloat())
        }
        for (i in 0 until count) {
            assertEquals(
                i.toFloat(), buf.getFloat((i * 4).toLong()),
                "Float at index $i"
            )
        }
        buf.free()
    }
}
