package io.github.kotlinmania.llama.ore

actual fun ggml_aligned_malloc(size: Long): Any? {
    if (size == 0L) return null
    return ByteArray(size.toInt())
}

actual fun ggml_aligned_free(ptr: Any?, size: Long) {
    // Android/JVM garbage collection handles ByteArray deallocation.
}

actual fun createDefaultCpuBufferType(): GGMLBackendBufferType {
    return GGMLCpuBufferType()
}
