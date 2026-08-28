// port-lint: source ggml/src/ggml.c (ggml_aligned_malloc / ggml_aligned_free)
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.ore

/**
 * Native actual for [ggml_aligned_malloc].
 *
 * Delegates to [NativeAlignedBuffer.alloc] which calls `posix_memalign` with
 * 64-byte alignment, matching the C implementation (ggml.c line 326–379).
 *
 * Returns a [NativeAlignedBuffer] (or `null` if allocation fails / size is 0).
 */
actual fun ggml_aligned_malloc(size: Long): Any? {
    return NativeAlignedBuffer.alloc(size)
}

/**
 * Native actual for [ggml_aligned_free].
 *
 * Mirrors `ggml_aligned_free` (ggml.c line 382–397).
 * Calls `platform.posix.free` on the underlying pointer.
 */
actual fun ggml_aligned_free(ptr: Any?, size: Long) {
    (ptr as? NativeAlignedBuffer)?.free()
}

/**
 * Native actual for [createDefaultCpuBufferType].
 *
 * Returns [GGMLCpuBufferTypeNative] which allocates aligned native memory
 * via `posix_memalign`, matching the C CPU buffer type
 * (ggml-backend.cpp line 2305–2313).
 */
actual fun createDefaultCpuBufferType(): GGMLBackendBufferType {
    return GGMLCpuBufferTypeNative()
}
