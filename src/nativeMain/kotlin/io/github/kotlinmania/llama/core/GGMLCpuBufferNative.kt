// port-lint: source ggml/src/ggml-backend.cpp (CPU buffer vtable, lines 2211–2293)
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.ore

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.convert
import kotlinx.cinterop.plus
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import platform.posix.memcpy

/**
 * Native CPU backend buffer type using aligned native memory.
 *
 * Mirrors the vtable wired into `ggml_backend_cpu_buffer_type()` (ggml-backend.cpp
 * line 2321–2343) and the allocation function
 * `ggml_backend_cpu_buffer_type_alloc_buffer` (line 2305–2313).
 *
 * Unlike the commonMain [GGMLCpuBufferType] which allocates a [ByteArray],
 * this type allocates memory through `posix_memalign` via [NativeAlignedBuffer],
 * providing true 64-byte-aligned native memory matching the C implementation.
 */
class GGMLCpuBufferTypeNative : GGMLBackendBufferType {

    override fun getName(): String = "CPU"

    /**
     * Allocate a buffer of [size] bytes using aligned native memory.
     *
     * Mirrors `ggml_backend_cpu_buffer_type_alloc_buffer` (ggml-backend.cpp line 2305–2313):
     * ```c
     * void * data = ggml_aligned_malloc(size);
     * if (data == NULL) { ... return NULL; }
     * return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
     * ```
     *
     * The C code stores the raw `void *` from `ggml_aligned_malloc` as `buffer->context`.
     * Here, [NativeAlignedBuffer] wraps that pointer with lifetime management.
     */
    override fun allocBuffer(size: ULong): GGMLBackendBuffer? {
        if (size == 0uL) return null

        val nativeBuffer = NativeAlignedBuffer.alloc(size.toLong())
        if (nativeBuffer == null) {
            println("GGMLCpuBufferTypeNative: failed to allocate buffer of size $size")
            return null
        }
        return GGMLCpuBufferNative(this, nativeBuffer, size)
    }

    /** 32-byte alignment for tensor data within the buffer. */
    override fun getAlignment(): UInt = NativeAlignedBuffer.TENSOR_ALIGNMENT.toUInt()

    override fun getMaxSize(): ULong = ULong.MAX_VALUE

    override fun isHost(): Boolean = true
}

/**
 * Native CPU backend buffer backed by [NativeAlignedBuffer].
 *
 * Mirrors the `ggml_backend_cpu_buffer_i` vtable (ggml-backend.cpp line 2267–2279).
 * Each method corresponds 1:1 to a C callback in the vtable.
 *
 * In the C code, `buffer->context` IS the raw `void *` pointer returned by
 * `ggml_aligned_malloc`.  Here, [nativeBuffer] holds that pointer plus metadata.
 *
 * Tensor data pointers (`tensor->data`) in C point directly into this buffer:
 * ```c
 * tensor->data = (char *)buffer_get_base() + tensor->view_offs;
 * ```
 * The Kotlin equivalent stores the [NativeAlignedBuffer] (or a sub-pointer) as
 * `tensor.data`, and operations use `tensor.dataOffset` for addressing.
 *
 * @property bufferType The buffer type that created this buffer.
 * @property nativeBuffer The underlying aligned native memory.
 * @property size Logical size of the buffer (may be ≤ nativeBuffer.sizeBytes due to alignment padding).
 */
class GGMLCpuBufferNative(
    private val bufferType: GGMLCpuBufferTypeNative,
    val nativeBuffer: NativeAlignedBuffer,
    private val size: ULong
) : GGMLBackendBuffer {

    override fun getType(): GGMLBackendBufferType = bufferType

    override fun getName(): String = "CPU"

    /**
     * Returns the aligned base pointer.
     *
     * Mirrors `ggml_backend_cpu_buffer_get_base` (ggml-backend.cpp line 2213–2223).
     * Returns the [NativeAlignedBuffer] itself — callers that need the raw
     * `CPointer<ByteVar>` can access [NativeAlignedBuffer.getAlignedBase].
     */
    override fun getBase(): Any = nativeBuffer

    override fun getSize(): ULong = size

    /**
     * Free the buffer's native memory.
     *
     * Mirrors `ggml_backend_cpu_buffer_free_buffer` (ggml-backend.cpp line 2225–2228):
     * ```c
     * ggml_aligned_free(buffer->context, buffer->size);
     * ```
     */
    override fun free() {
        nativeBuffer.free()
    }

    override fun initTensor(tensor: GGMLTensor): GGMLStatus {
        tensor.buffer = this
        // In C, init_tensor is NULL for CPU buffers — no special init needed.
        // We set tensor.data to the NativeAlignedBuffer so compute ops can reach it.
        tensor.data = nativeBuffer
        return GGMLStatus.SUCCESS
    }

    /**
     * Copy data from a host [ByteArray] into a tensor's region of the buffer.
     *
     * Mirrors `ggml_backend_cpu_buffer_set_tensor` (ggml-backend.cpp line 2237–2241):
     * ```c
     * memcpy((char *)tensor->data + offset, data, size);
     * ```
     *
     * Note the C code indexes from `tensor->data`, which is already
     * `buffer_base + tensor->view_offs`.  Here we use `tensor.dataOffset + offset`
     * relative to the buffer's base pointer.
     */
    override fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val dstOffset = tensor.dataOffset.toLong() + offset.toLong()
        nativeBuffer.copyFromByteArray(data, 0, dstOffset, size.toInt())
    }

    /**
     * Copy data from a tensor's region of the buffer into a host [ByteArray].
     *
     * Mirrors `ggml_backend_cpu_buffer_get_tensor` (ggml-backend.cpp line 2244–2248):
     * ```c
     * memcpy(data, (const char *)tensor->data + offset, size);
     * ```
     */
    override fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val srcOffset = tensor.dataOffset.toLong() + offset.toLong()
        nativeBuffer.copyToByteArray(data, srcOffset, 0, size.toInt())
    }

    /**
     * Copy tensor data between buffers.
     *
     * Mirrors `ggml_backend_cpu_buffer_cpy_tensor` (ggml-backend.cpp line 2251–2260):
     * ```c
     * if (ggml_backend_buffer_is_host(src->buffer)) {
     *     memcpy(dst->data, src->data, ggml_nbytes(src));
     *     return true;
     * }
     * return false;
     * ```
     */
    override fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean {
        val srcBuffer = src.buffer ?: return false
        if (!srcBuffer.getType().isHost()) return false

        val nbytes = src.nBytes()

        val srcBase = srcBuffer.getBase()
        val dstOffset = dst.dataOffset.toLong()
        val srcOffset = src.dataOffset.toLong()

        when (srcBase) {
            is NativeAlignedBuffer -> {
                // Native-to-native copy via memcpy
                nativeBuffer.copyFrom(srcBase, srcOffset, dstOffset, nbytes)
            }
            is ByteArray -> {
                // ByteArray source → native destination
                nativeBuffer.copyFromByteArray(srcBase, srcOffset.toInt(), dstOffset, nbytes.toInt())
            }
            else -> return false
        }
        return true
    }

    /**
     * Clear the entire buffer with [value].
     *
     * Mirrors `ggml_backend_cpu_buffer_clear` (ggml-backend.cpp line 2262–2265):
     * ```c
     * memset(buffer->context, value, buffer->size);
     * ```
     */
    override fun clear(value: UByte) {
        nativeBuffer.fill(value, 0, nativeBuffer.sizeBytes)
    }

    /**
     * Memset a region of a tensor's data.
     *
     * Mirrors `ggml_backend_cpu_buffer_memset_tensor` (ggml-backend.cpp line 2230–2234):
     * ```c
     * memset((char *)tensor->data + offset, value, size);
     * ```
     */
    fun memsetTensor(tensor: GGMLTensor, value: UByte, offset: Long, size: Long) {
        val absOffset = tensor.dataOffset.toLong() + offset
        nativeBuffer.memsetRegion(value, absOffset, size)
    }

    /**
     * Get the raw native pointer for a tensor's data region.
     *
     * This is the Kotlin equivalent of C's `tensor->data` which is a `void *`
     * pointing into the buffer.  Useful for compute ops that need direct pointer access.
     */
    fun getTensorPointer(tensor: GGMLTensor): CPointer<ByteVar> {
        val base = nativeBuffer.getAlignedBase()
        return (base + tensor.dataOffset.toLong())!!
    }
}

/**
 * Native CPU buffer wrapping an externally-owned pointer.
 *
 * Mirrors `ggml_backend_cpu_buffer_from_ptr_i` (ggml-backend.cpp line 2281–2293)
 * where `free_buffer = NULL` — the pointer is not owned by the buffer.
 *
 * Used by `ggml_backend_cpu_device_buffer_from_host_ptr` (line 416–421).
 */
class GGMLCpuBufferFromPtrNative(
    private val nativeBuffer: NativeAlignedBuffer,
    private val size: ULong
) : GGMLBackendBuffer {
    private val buft = GGMLCpuBufferTypeNative()

    override fun getType(): GGMLBackendBufferType = buft
    override fun getName(): String = "CPU"
    override fun getBase(): Any = nativeBuffer
    override fun getSize(): ULong = size

    // Does not free — pointer is externally owned.
    override fun free() {}

    override fun initTensor(tensor: GGMLTensor): GGMLStatus {
        tensor.buffer = this
        tensor.data = nativeBuffer
        return GGMLStatus.SUCCESS
    }

    override fun setTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val dstOffset = tensor.dataOffset.toLong() + offset.toLong()
        nativeBuffer.copyFromByteArray(data, 0, dstOffset, size.toInt())
    }

    override fun getTensor(tensor: GGMLTensor, data: ByteArray, offset: ULong, size: ULong) {
        val srcOffset = tensor.dataOffset.toLong() + offset.toLong()
        nativeBuffer.copyToByteArray(data, srcOffset, 0, size.toInt())
    }

    override fun copyTensor(src: GGMLTensor, dst: GGMLTensor): Boolean {
        val srcBuffer = src.buffer ?: return false
        if (!srcBuffer.getType().isHost()) return false

        val nbytes = src.nBytes()
        val srcBase = srcBuffer.getBase()
        val dstOffset = dst.dataOffset.toLong()
        val srcOffset = src.dataOffset.toLong()

        when (srcBase) {
            is NativeAlignedBuffer -> {
                nativeBuffer.copyFrom(srcBase, srcOffset, dstOffset, nbytes)
            }
            is ByteArray -> {
                nativeBuffer.copyFromByteArray(srcBase, srcOffset.toInt(), dstOffset, nbytes.toInt())
            }
            else -> return false
        }
        return true
    }

    override fun clear(value: UByte) {
        nativeBuffer.fill(value, 0, nativeBuffer.sizeBytes)
    }
}
