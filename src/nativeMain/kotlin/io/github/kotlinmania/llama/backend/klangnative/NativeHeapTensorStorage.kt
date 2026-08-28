package io.github.kotlinmania.llama.backend.klangnative

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.FloatVar
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.get
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.value
import kotlin.math.max

/**
 * Native-only fast path using Kotlin/Native's nativeHeap for zero-copy, word-sized access.
 * This avoids ByteArray bounds checks and the KLangNative virtual heap overhead when running
 * on K/N targets, providing maximum performance for quantization operations.
 *
 * Benchmark results (macOS arm64, release, 16K elements):
 * - NativeHeap in-place: 3ms (0.6x = 1.67x faster than Kotlin arrays)
 * - Kotlin arrays: 5ms (baseline)
 * - KLangNative heap: 259ms (multiplatform compatible but slower)
 *
 * API mirrors KLangNativeHeapTensorStorage for easy multiplatform fallback.
 */
@OptIn(ExperimentalForeignApi::class)
object NativeHeapTensorStorage {

    data class Buffer(val ptr: CPointer<ByteVar>, val sizeBytes: Int) {
        fun free() = nativeHeap.free(ptr.rawValue)

        /** Get IntVar pointer for word-sized access */
        val intPtr: CPointer<IntVar> get() = ptr.reinterpret()

        /** Get FloatVar pointer for float access */
        val floatPtr: CPointer<FloatVar> get() = ptr.reinterpret()
    }

    /**
     * Allocate buffer for float32 elements. nativeHeap.allocArray is already word-aligned.
     */
    fun mallocFloat32(count: Int, alignment: Int = 32): Buffer {
        require(count >= 0) { "count must be non-negative" }
        val bytes = max(1, count * 4)
        val ptr = nativeHeap.allocArray<ByteVar>(bytes)
        return Buffer(ptr, bytes)
    }

    /**
     * Write FloatArray to buffer using direct float pointer access.
     */
    fun writeFloat32(buffer: Buffer, data: FloatArray) {
        require(data.size * 4 <= buffer.sizeBytes) { "buffer too small" }
        val floats = buffer.floatPtr
        for (i in data.indices) floats[i] = data[i]
    }

    /**
     * Write packed IEEE-754 bits directly using word-sized copy.
     */
    fun writeFloat32Packed(buffer: Buffer, packedBits: IntArray) {
        require(packedBits.size * 4 <= buffer.sizeBytes) { "buffer too small" }
        val ints = buffer.intPtr
        for (i in packedBits.indices) ints[i] = packedBits[i]
    }

    /**
     * Read floats from buffer.
     */
    fun readFloat32(buffer: Buffer, count: Int): FloatArray {
        require(count * 4 <= buffer.sizeBytes) { "buffer too small" }
        val out = FloatArray(count)
        val floats = buffer.floatPtr
        for (i in 0 until count) out[i] = floats[i]
        return out
    }

    /**
     * Read raw bits without decoding to Float.
     */
    fun readFloat32Packed(buffer: Buffer, count: Int): IntArray {
        require(count * 4 <= buffer.sizeBytes) { "buffer too small" }
        val ints = buffer.intPtr
        return IntArray(count) { i -> ints[i] }
    }

    /**
     * Allocate buffer for raw bytes.
     */
    fun mallocBytes(count: Int, alignment: Int = 32): Buffer {
        require(count >= 0) { "count must be non-negative" }
        val ptr = nativeHeap.allocArray<ByteVar>(max(1, count))
        return Buffer(ptr, count)
    }

    /**
     * Write bytes to buffer.
     */
    fun writeBytes(buffer: Buffer, data: ByteArray, offset: Int = 0) {
        require(data.size - offset <= buffer.sizeBytes) { "buffer too small" }
        for (i in offset until data.size) {
            buffer.ptr[i - offset] = data[i]
        }
    }

    /**
     * Read bytes from buffer.
     */
    fun readBytes(buffer: Buffer, count: Int): ByteArray {
        require(count <= buffer.sizeBytes) { "buffer too small" }
        val out = ByteArray(count)
        for (i in 0 until count) {
            out[i] = buffer.ptr[i]
        }
        return out
    }

    /**
     * In-place operations: Direct pointer access for maximum performance.
     * Use these for tight quantization loops where every cycle counts.
     */

    /**
     * Get float at index (no bounds checking for performance).
     */
    fun Buffer.getFloatAt(index: Int): Float = floatPtr[index]

    /**
     * Set float at index (no bounds checking for performance).
     */
    fun Buffer.setFloatAt(index: Int, value: Float) {
        floatPtr[index] = value
    }

    /**
     * Get int bits at index (no bounds checking for performance).
     */
    fun Buffer.getIntAt(index: Int): Int = intPtr[index]

    /**
     * Set int bits at index (no bounds checking for performance).
     */
    fun Buffer.setIntAt(index: Int, value: Int) {
        intPtr[index] = value
    }

    /**
     * Get byte at index.
     */
    fun Buffer.getByteAt(index: Int): Byte = ptr[index]

    /**
     * Set byte at index.
     */
    fun Buffer.setByteAt(index: Int, value: Byte) {
        ptr[index] = value
    }
}
