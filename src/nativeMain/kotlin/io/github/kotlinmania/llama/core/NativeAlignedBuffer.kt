// port-lint: source ggml/src/ggml.c (ggml_aligned_malloc / ggml_aligned_free)
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.platform.nativeAlignedAlloc
import io.github.kotlinmania.llama.platform.nativeAlignedFree
import io.github.kotlinmania.llama.platform.nativeMemcpy
import io.github.kotlinmania.llama.platform.nativeMemset
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.FloatVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.LongVar
import kotlinx.cinterop.ShortVar
import kotlinx.cinterop.UByteVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.get
import kotlinx.cinterop.plus
import kotlinx.cinterop.pointed
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.set
import kotlinx.cinterop.toLong
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value

/**
 * Native aligned memory buffer.
 *
 * Wraps a pointer returned by `posix_memalign` with GGML's required alignment.
 * This is the Kotlin/Native equivalent of the `void *` context in
 * `ggml_backend_cpu_buffer` (ggml-backend.cpp line 2305–2313).
 *
 * The C code (ggml.c line 326–397) uses:
 * - macOS: `vm_allocate` (page-aligned, via Mach VM)
 * - Linux: `posix_memalign` with 64-byte alignment
 * - `TENSOR_ALIGNMENT` (32) is used separately for `get_base` padding
 *
 * We use `posix_memalign` on all POSIX targets — it is available on macOS and Linux,
 * and the 64-byte alignment exceeds TENSOR_ALIGNMENT's 32-byte requirement.
 *
 * @property pointer Raw pointer to the allocated memory region.
 * @property sizeBytes Number of usable bytes in the buffer.
 * @property ownsMemory When `true`, [free] will release the memory.
 *   Set to `false` for buffers wrapping externally-owned pointers
 *   (mirrors `ggml_backend_cpu_buffer_from_ptr_i` where `free_buffer = NULL`).
 */
class NativeAlignedBuffer private constructor(
    val pointer: CPointer<ByteVar>,
    val sizeBytes: Long,
    private val ownsMemory: Boolean = true
) {

    companion object {
        /** Allocation alignment — matches `alignment` in `ggml_aligned_malloc` (ggml.c line 330). */
        const val MALLOC_ALIGNMENT = 64

        /** Tensor data alignment — matches `TENSOR_ALIGNMENT` in ggml-impl.h line 44. */
        const val TENSOR_ALIGNMENT = 32

        /**
         * Allocate [size] bytes of aligned memory.
         *
         * Mirrors `ggml_aligned_malloc` (ggml.c line 326–379).
         * Returns `null` when [size] is 0 or allocation fails.
         */
        fun alloc(size: Long, alignment: Int = MALLOC_ALIGNMENT): NativeAlignedBuffer? {
            if (size == 0L) {
                println("NativeAlignedBuffer.alloc: behavior may be unexpected when allocating 0 bytes")
                return null
            }
            val ptr = nativeAlignedAlloc(size, alignment)
            if (ptr == null) {
                val desc = when {
                    alignment <= 0 || alignment and (alignment - 1) != 0 -> "invalid alignment value"
                    size < 0 -> "invalid allocation size"
                    else -> "insufficient memory"
                }
                println(
                    "NativeAlignedBuffer.alloc: $desc " +
                        "(attempted to allocate ${size / (1024.0 * 1024.0)} MB)"
                )
                return null
            }
            return NativeAlignedBuffer(ptr, size)
        }

        /**
         * Wrap an externally-owned pointer (does not free on [free]).
         *
         * Mirrors `ggml_backend_cpu_buffer_from_ptr_i` (ggml-backend.cpp line 2281–2293)
         * where `free_buffer = NULL`.
         */
        fun fromPointer(pointer: CPointer<ByteVar>, size: Long): NativeAlignedBuffer {
            return NativeAlignedBuffer(pointer, size, ownsMemory = false)
        }
    }

    // ========================================================================
    // Base pointer alignment  (ggml_backend_cpu_buffer_get_base)
    // ========================================================================

    /**
     * Returns the aligned base pointer, padded up to [TENSOR_ALIGNMENT].
     *
     * Mirrors `ggml_backend_cpu_buffer_get_base` (ggml-backend.cpp line 2213–2223):
     * ```c
     * uintptr_t data = (uintptr_t)buffer->context;
     * if (data % TENSOR_ALIGNMENT != 0) {
     *     data = GGML_PAD(data, TENSOR_ALIGNMENT);
     * }
     * return (void *)data;
     * ```
     *
     * Because we allocate with 64-byte alignment (≥ 32), this will almost
     * always return the same pointer.  The check is included for correctness.
     */
    fun getAlignedBase(): CPointer<ByteVar> {
        val addr = pointer.toLong()
        if (addr % TENSOR_ALIGNMENT != 0L) {
            val aligned = (addr + TENSOR_ALIGNMENT - 1) and (TENSOR_ALIGNMENT - 1).toLong().inv()
            return (pointer + (aligned - addr))!!
        }
        return pointer
    }

    // ========================================================================
    // Typed element access  (pointer arithmetic, matching C's direct casts)
    // ========================================================================

    fun getByte(offset: Long): Byte = pointer[offset]
    fun setByte(offset: Long, value: Byte) { pointer[offset] = value }

    fun getUByte(offset: Long): UByte = pointer[offset].toUByte()
    fun setUByte(offset: Long, value: UByte) { pointer[offset] = value.toByte() }

    fun getFloat(offset: Long): Float {
        return (pointer + offset)!!.reinterpret<FloatVar>().pointed.value
    }

    fun setFloat(offset: Long, value: Float) {
        (pointer + offset)!!.reinterpret<FloatVar>().pointed.value = value
    }

    fun getInt(offset: Long): Int {
        return (pointer + offset)!!.reinterpret<IntVar>().pointed.value
    }

    fun setInt(offset: Long, value: Int) {
        (pointer + offset)!!.reinterpret<IntVar>().pointed.value = value
    }

    fun getShort(offset: Long): Short {
        return (pointer + offset)!!.reinterpret<ShortVar>().pointed.value
    }

    fun setShort(offset: Long, value: Short) {
        (pointer + offset)!!.reinterpret<ShortVar>().pointed.value = value
    }

    fun getLong(offset: Long): Long {
        return (pointer + offset)!!.reinterpret<LongVar>().pointed.value
    }

    fun setLong(offset: Long, value: Long) {
        (pointer + offset)!!.reinterpret<LongVar>().pointed.value = value
    }

    // ========================================================================
    // Bulk copy operations  (memcpy-based, matching C's set_tensor / get_tensor)
    // ========================================================================

    /**
     * Copy from a Kotlin [ByteArray] into this native buffer.
     *
     * Mirrors the host → device direction of `ggml_backend_cpu_buffer_set_tensor`
     * (ggml-backend.cpp line 2237–2241):
     * ```c
     * memcpy((char *)tensor->data + offset, data, size);
     * ```
     */
    fun copyFromByteArray(src: ByteArray, srcOffset: Int, dstOffset: Long, length: Int) {
        src.usePinned { pinned ->
            nativeMemcpy(
                (pointer + dstOffset)!!,
                pinned.addressOf(srcOffset),
                length.toLong(),
            )
        }
    }

    /**
     * Copy from this native buffer into a Kotlin [ByteArray].
     *
     * Mirrors the device → host direction of `ggml_backend_cpu_buffer_get_tensor`
     * (ggml-backend.cpp line 2244–2248):
     * ```c
     * memcpy(data, (const char *)tensor->data + offset, size);
     * ```
     */
    fun copyToByteArray(dst: ByteArray, srcOffset: Long, dstOffset: Int, length: Int) {
        dst.usePinned { pinned ->
            nativeMemcpy(
                pinned.addressOf(dstOffset),
                (pointer + srcOffset)!!,
                length.toLong(),
            )
        }
    }

    /**
     * Copy between two native buffers.
     *
     * Mirrors `ggml_backend_cpu_buffer_cpy_tensor` (ggml-backend.cpp line 2251–2260):
     * ```c
     * memcpy(dst->data, src->data, ggml_nbytes(src));
     * ```
     */
    fun copyFrom(src: NativeAlignedBuffer, srcOffset: Long, dstOffset: Long, length: Long) {
        nativeMemcpy(
            (pointer + dstOffset)!!,
            (src.pointer + srcOffset)!!,
            length,
        )
    }

    // ========================================================================
    // Fill / clear  (memset)
    // ========================================================================

    /**
     * Fill a region with a byte value.
     *
     * Mirrors `ggml_backend_cpu_buffer_clear` (ggml-backend.cpp line 2262–2265):
     * ```c
     * memset(buffer->context, value, buffer->size);
     * ```
     */
    fun fill(value: UByte, offset: Long = 0, length: Long = sizeBytes) {
        nativeMemset((pointer + offset)!!, value.toInt(), length)
    }

    /**
     * Memset a region starting at [offset].
     *
     * Mirrors `ggml_backend_cpu_buffer_memset_tensor` (ggml-backend.cpp line 2230–2234):
     * ```c
     * memset((char *)tensor->data + offset, value, size);
     * ```
     */
    fun memsetRegion(value: UByte, offset: Long, length: Long) {
        nativeMemset((pointer + offset)!!, value.toInt(), length)
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /**
     * Free the allocated memory.
     *
     * Mirrors `ggml_aligned_free` (ggml.c line 382–397).
     * For non-owned pointers (`ownsMemory == false`), this is a no-op —
     * matching `ggml_backend_cpu_buffer_from_ptr_i` where `free_buffer = NULL`.
     */
    fun free() {
        if (ownsMemory) {
            nativeAlignedFree(pointer)
        }
    }

    override fun toString(): String =
        "NativeAlignedBuffer(ptr=${pointer.toLong().toString(16)}, size=$sizeBytes, owns=$ownsMemory)"
}
