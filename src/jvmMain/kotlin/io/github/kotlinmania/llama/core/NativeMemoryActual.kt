package io.github.kotlinmania.llama.ore

/**
 * JVM actual for [io.github.kotlinmania.llama.ore.ggml_aligned_malloc].
 *
 * Falls back to [ByteArray] — JVM does not expose aligned memory allocation.
 * The GC manages the lifecycle.
 */
actual fun ggml_aligned_malloc(size: Long): Any? {
    if (size == 0L) return null
    return ByteArray(size.toInt())
}

/**
 * JVM actual for [ggml_aligned_free].
 *
 * No-op: JVM garbage collector handles deallocation.
 */
actual fun ggml_aligned_free(ptr: Any?, size: Long) {
    // No-op: GC handles ByteArray deallocation
}

/**
 * JVM actual for [createDefaultCpuBufferType].
 *
 * Returns the ByteArray-backed [GGMLCpuBufferType].
 */
actual fun createDefaultCpuBufferType(): GGMLBackendBufferType {
    return GGMLCpuBufferType()
}
