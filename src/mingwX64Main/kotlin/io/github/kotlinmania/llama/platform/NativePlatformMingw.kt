@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.platform

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.nativeHeap
import kotlinx.cinterop.ptr
import kotlinx.cinterop.rawValue
import kotlinx.cinterop.usePinned
import platform.posix.SEEK_SET
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fstat
import platform.posix.fwrite
import platform.posix.lseek
import platform.posix.memcpy
import platform.posix.memset
import platform.posix.read
import platform.posix.stat
import platform.posix.write

internal actual fun nativeAlignedAlloc(
    sizeBytes: Long,
    alignmentBytes: Int,
): CPointer<ByteVar>? {
    if (alignmentBytes <= 0 || sizeBytes <= 0L || sizeBytes > Int.MAX_VALUE) return null
    return nativeHeap.allocArray(sizeBytes.toInt())
}

internal actual fun nativeAlignedFree(pointer: CPointer<ByteVar>) {
    nativeHeap.free(pointer.rawValue)
}

internal actual fun nativeMemcpy(
    destination: COpaquePointer?,
    source: COpaquePointer?,
    sizeBytes: Long,
) {
    memcpy(destination, source, sizeBytes.convert())
}

internal actual fun nativeMemset(
    destination: COpaquePointer?,
    value: Int,
    sizeBytes: Long,
) {
    memset(destination, value, sizeBytes.convert())
}

internal actual fun nativeFileSize(fd: Int): Long =
    memScoped {
        val st = alloc<stat>()
        val size: Number = st.st_size
        if (fstat(fd, st.ptr) == 0) size.toLong() else -1L
    }

internal actual fun nativeSeek(
    fd: Int,
    offset: Long,
): Long {
    val result: Number = lseek(fd, offset.convert(), SEEK_SET)
    return result.toLong()
}

internal actual fun nativeRead(
    fd: Int,
    destination: COpaquePointer?,
    sizeBytes: Long,
): Long {
    val result: Number = read(fd, destination, sizeBytes.convert())
    return result.toLong()
}

internal actual fun nativeWrite(
    fd: Int,
    source: COpaquePointer?,
    sizeBytes: Long,
): Long {
    val result: Number = write(fd, source, sizeBytes.convert())
    return result.toLong()
}

internal actual fun nativePageSize(): Long = 4096L

internal actual fun nativePhysicalPages(): Long = 0L

internal actual val nativeMmapSupported: Boolean = false

internal actual val nativeMlockSupported: Boolean = false

internal actual fun nativeMmapReadOnly(
    fd: Int,
    sizeBytes: Long,
): COpaquePointer? = null

internal actual fun nativeMunmap(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int = 0

internal actual fun nativeMlock(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int = -1

internal actual fun nativeMunlock(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int = 0

internal actual fun nativeWriteBytesToFile(
    bytes: ByteArray,
    path: String,
) {
    val file = fopen(path, "wb") ?: error("Failed to open file for writing: $path")
    try {
        if (bytes.isNotEmpty()) {
            bytes.usePinned { pinned ->
                fwrite(pinned.addressOf(0), 1.convert(), bytes.size.convert(), file)
            }
        }
    } finally {
        fclose(file)
    }
}
