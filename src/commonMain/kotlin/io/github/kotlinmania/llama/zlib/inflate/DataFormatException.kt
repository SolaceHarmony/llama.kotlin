package io.github.kotlinmania.llama.lib.inflate

/**
 * Signals invalid or corrupt DEFLATE/zlib data encountered during decoding.
 */
class DataFormatException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause)
