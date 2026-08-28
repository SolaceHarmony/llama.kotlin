package io.github.kotlinmania.llama.ore

/**
 * JS actual for [io.github.kotlinmania.llama.ore.ggml_aligned_malloc].
 *
 * Falls back to [ByteArray] — JavaScript does not expose aligned memory allocation.
 */
actual fun ggml_aligned_malloc(size: Long): Any? {
    if (size == 0L) return null
    return ByteArray(size.toInt())
}

/**
 * JS actual for [io.github.kotlinmania.llama.ore.ggml_aligned_free].
 *
 * No-op: JavaScript garbage collector handles deallocation.
 */
actual fun ggml_aligned_free(ptr: Any?, size: Long) {
    // No-op: GC handles ByteArray deallocation
}

/**
 * JS actual for [io.github.kotlinmania.llama.ore.createDefaultCpuBufferType].
 *
 * Returns the ByteArray-backed [io.github.kotlinmania.llama.ore.GGMLCpuBufferType].
 */
actual fun createDefaultCpuBufferType(): io.github.kotlinmania.llama.ore.GGMLBackendBufferType {
    return io.github.kotlinmania.llama.ore.GGMLCpuBufferType()
}
