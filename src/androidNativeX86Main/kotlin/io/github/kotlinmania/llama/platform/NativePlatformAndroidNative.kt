@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.platform

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.convert
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.posix.MAP_FAILED
import platform.posix.MAP_PRIVATE
import platform.posix.PROT_READ
import platform.posix.SEEK_SET
import platform.posix._SC_PAGESIZE
import platform.posix._SC_PHYS_PAGES
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fstat
import platform.posix.free
import platform.posix.fwrite
import platform.posix.lseek
import platform.posix.memcpy
import platform.posix.memset
import platform.posix.mlock
import platform.posix.mmap
import platform.posix.munlock
import platform.posix.munmap
import platform.posix.posix_memalign
import platform.posix.read
import platform.posix.stat
import platform.posix.sysconf
import platform.posix.write

internal actual fun nativeAlignedAlloc(
    sizeBytes: Long,
    alignmentBytes: Int,
): CPointer<ByteVar>? {
    if (sizeBytes <= 0L) return null
    return memScoped {
        val ptrVar = alloc<COpaquePointerVar>()
        val result = posix_memalign(ptrVar.ptr, alignmentBytes.convert(), sizeBytes.convert())
        if (result == 0) ptrVar.value?.reinterpret<ByteVar>() else null
    }
}

internal actual fun nativeAlignedFree(pointer: CPointer<ByteVar>) {
    free(pointer)
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
    val result: Number = lseek(fd, offset.toULong().convert(), SEEK_SET)
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

internal actual fun nativePageSize(): Long {
    val result: Number = sysconf(_SC_PAGESIZE)
    return result.toLong()
}

internal actual fun nativePhysicalPages(): Long {
    val result: Number = sysconf(_SC_PHYS_PAGES)
    return result.toLong()
}

internal actual val nativeMmapSupported: Boolean = true

internal actual val nativeMlockSupported: Boolean = true

internal actual fun nativeMmapReadOnly(
    fd: Int,
    sizeBytes: Long,
): COpaquePointer? {
    val pointer = mmap(null, sizeBytes.convert(), PROT_READ, MAP_PRIVATE, fd, 0)
    return if (pointer == MAP_FAILED) null else pointer
}

internal actual fun nativeMunmap(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int = munmap(pointer, sizeBytes.convert())

internal actual fun nativeMlock(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int = mlock(pointer, sizeBytes.convert())

internal actual fun nativeMunlock(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int = munlock(pointer, sizeBytes.convert())

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
