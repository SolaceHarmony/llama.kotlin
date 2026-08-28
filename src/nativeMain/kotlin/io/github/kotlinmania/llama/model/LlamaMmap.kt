// port-lint: source llama.cpp/src/llama-mmap.h
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.platform.nativeFileSize
import io.github.kotlinmania.llama.platform.nativeMlock
import io.github.kotlinmania.llama.platform.nativeMlockSupported
import io.github.kotlinmania.llama.platform.nativeMmapReadOnly
import io.github.kotlinmania.llama.platform.nativeMmapSupported
import io.github.kotlinmania.llama.platform.nativeMunlock
import io.github.kotlinmania.llama.platform.nativeMunmap
import io.github.kotlinmania.llama.platform.nativePageSize
import io.github.kotlinmania.llama.platform.nativeRead
import io.github.kotlinmania.llama.platform.nativeSeek
import io.github.kotlinmania.llama.platform.nativeWrite
import kotlinx.cinterop.*
import platform.posix.*

// =============================================================================
// LlamaMmap – memory-mapped file types for model loading
// Ported from: llama-mmap.h
// =============================================================================

/**
 * Abstraction over a file handle with positioned read/write operations.
 *
 * Port of `llama_file` from `llama-mmap.h`. Uses POSIX I/O on Kotlin/Native.
 *
 * @property path  The filesystem path that was opened.
 */
class LlamaFile private constructor(val path: String, private var fd: Int) {

    private var position: Long = 0
    private var fileSize: Long = computeFileSize()

    private fun computeFileSize(): Long = nativeFileSize(fd)

    /** Current read/write position in bytes. */
    fun tell(): Long = position

    /** Total size of the file in bytes. */
    fun size(): Long = fileSize

    /** Return the underlying file descriptor (needed for mmap). */
    fun fileno(): Int = fd

    /** Seek to an absolute [offset]. */
    fun seek(offset: Long) {
        val result = nativeSeek(fd, offset)
        if (result == -1L) {
            throw IllegalStateException("lseek failed for '$path': ${strerror(errno)?.toKString()}")
        }
        position = offset
    }

    /**
     * Read exactly [len] bytes into [dst] starting at [dstOffset].
     *
     * @throws IllegalStateException if fewer bytes are available.
     */
    fun readRaw(dst: ByteArray, len: Int, dstOffset: Int = 0) {
        var totalRead = 0
        dst.usePinned { pinned ->
            while (totalRead < len) {
                val n = nativeRead(
                    fd,
                    pinned.addressOf(dstOffset + totalRead),
                    (len - totalRead).toLong(),
                )
                if (n <= 0) {
                    throw IllegalStateException(
                        "read failed for '$path': expected $len bytes, got $totalRead" +
                            " (errno: ${strerror(errno)?.toKString()})"
                    )
                }
                totalRead += n.toInt()
            }
        }
        position += totalRead
    }

    /** Read a little-endian 32-bit unsigned integer. */
    fun readU32(): UInt {
        val buf = ByteArray(4)
        readRaw(buf, 4)
        return (buf[0].toUByte().toUInt()) or
            (buf[1].toUByte().toUInt() shl 8) or
            (buf[2].toUByte().toUInt() shl 16) or
            (buf[3].toUByte().toUInt() shl 24)
    }

    /** Write [len] bytes from [src] starting at [srcOffset]. */
    fun writeRaw(src: ByteArray, len: Int, srcOffset: Int = 0) {
        var totalWritten = 0
        src.usePinned { pinned ->
            while (totalWritten < len) {
                val n = nativeWrite(
                    fd,
                    pinned.addressOf(srcOffset + totalWritten),
                    (len - totalWritten).toLong(),
                )
                if (n <= 0) {
                    throw IllegalStateException(
                        "write failed for '$path': expected $len bytes, wrote $totalWritten" +
                            " (errno: ${strerror(errno)?.toKString()})"
                    )
                }
                totalWritten += n.toInt()
            }
        }
        position += totalWritten
    }

    /** Write a little-endian 32-bit unsigned integer. */
    fun writeU32(value: UInt) {
        val buf = ByteArray(4)
        buf[0] = (value and 0xFFu).toByte()
        buf[1] = ((value shr 8) and 0xFFu).toByte()
        buf[2] = ((value shr 16) and 0xFFu).toByte()
        buf[3] = ((value shr 24) and 0xFFu).toByte()
        writeRaw(buf, 4)
    }

    /** Alignment used for direct-I/O reads, or 1 for normal I/O. */
    // port-lint: source llama.cpp/src/llama-mmap.cpp llama_file::read_alignment
    fun readAlignment(): Long = 1L  // No O_DIRECT on macOS

    /** Whether this file was opened with direct I/O (O_DIRECT). */
    // port-lint: source llama.cpp/src/llama-mmap.cpp llama_file::has_direct_io
    fun hasDirectIo(): Boolean = false

    // port-lint: source llama.cpp/src/llama-mmap.cpp llama_file::read_raw_unsafe
    /**
     * Read without error checking — in C++ this skips validation.
     * In Kotlin/Native we delegate to [readRaw] since safety is always on.
     */
    fun readRawUnsafe(dst: ByteArray, len: Int, dstOffset: Int = 0) {
        readRaw(dst, len, dstOffset)
    }

    /** Close the file descriptor. */
    fun close() {
        if (fd >= 0) {
            platform.posix.close(fd)
            fd = -1
        }
    }

    companion object {
        /**
         * Open a file at [path] with the given [mode] (e.g. "rb", "wb").
         *
         * @param useDirectIo  Attempt O_DIRECT for bypassing the page cache (ignored on macOS).
         */
        fun open(path: String, mode: String, useDirectIo: Boolean = false): LlamaFile {
            val flags = when {
                mode.startsWith("w") -> O_WRONLY or O_CREAT or O_TRUNC
                else -> O_RDONLY
            }
            val fd = platform.posix.open(path, flags, S_IRUSR or S_IWUSR or S_IRGRP or S_IROTH)
            if (fd < 0) {
                throw IllegalStateException("Failed to open '$path': ${strerror(errno)?.toKString()}")
            }
            return LlamaFile(path, fd)
        }
    }
}

/**
 * Memory-mapped region of a [LlamaFile].
 *
 * Port of `llama_mmap` from `llama-mmap.h`. Uses platform mmap(2) underneath.
 *
 * @property file  The file being mapped.
 */
class LlamaMmap private constructor(
    val file: LlamaFile,
) {
    private var mappedSize: Long = 0
    private var mappedPtr: COpaquePointer? = null

    /** Size of the mapped region in bytes. */
    fun size(): Long = mappedSize

    /** Return the base address of the mapped region. */
    fun getAddr(): COpaquePointer? = mappedPtr

    /**
     * Unmap a fragment of the region between byte offsets [first] and [last].
     *
     * This can release physical pages back to the OS for memory savings.
     * Offsets are page-aligned internally.
     */
    fun unmapFragment(first: Long, last: Long) {
        val pageSize = nativePageSize()
        if (pageSize <= 0 || first >= last) return

        // Align first up to page boundary, last down to page boundary
        val alignedFirst = ((first + pageSize - 1) / pageSize) * pageSize
        val alignedLast = (last / pageSize) * pageSize

        if (alignedFirst >= alignedLast) return

        val basePtr = mappedPtr ?: return
        val fragmentPtr = interpretCPointer<ByteVar>(basePtr.rawValue + alignedFirst) ?: return
        nativeMunmap(fragmentPtr, alignedLast - alignedFirst)
    }

    /** Unmap the entire region. */
    fun close() {
        val ptr = mappedPtr
        if (ptr != null && mappedSize > 0) {
            nativeMunmap(ptr, mappedSize)
            mappedPtr = null
            mappedSize = 0
        }
    }

    companion object {
        /** Whether memory-mapping is supported on the current platform. */
        val SUPPORTED: Boolean = nativeMmapSupported

        /**
         * Create a new memory-mapping of [file].
         *
         * @param file      The file to map.
         * @param prefetch  Number of bytes to prefetch (hint to the OS).
         * @param numa      Whether to use NUMA-aware allocation (ignored on macOS).
         */
        fun create(file: LlamaFile, prefetch: Long = -1, numa: Boolean = false): LlamaMmap {
            val size = file.size()
            if (size == 0L) {
                throw IllegalStateException("Cannot mmap empty file '${file.path}'")
            }

            val ptr = nativeMmapReadOnly(file.fileno(), size)
            if (ptr == null) {
                throw IllegalStateException(
                    "mmap failed for '${file.path}': ${strerror(errno)?.toKString()}"
                )
            }

            val mmap = LlamaMmap(file)
            mmap.mappedPtr = ptr
            mmap.mappedSize = size
            return mmap
        }
    }
}

/**
 * Pin physical pages in memory to prevent swapping.
 *
 * Port of `llama_mlock` from `llama-mmap.h`. Uses mlock(2) on supported platforms.
 */
class LlamaMlock {

    private var basePtr: COpaquePointer? = null
    private var lockedSize: Long = 0

    /**
     * Set the base address to lock.
     *
     * Must be called before [growTo].
     */
    fun init(addr: COpaquePointer?) {
        basePtr = addr
        lockedSize = 0
    }

    /**
     * Grow the locked region to [targetSize] bytes.
     *
     * Pages between the current locked size and [targetSize] are locked.
     */
    fun growTo(targetSize: Long) {
        if (targetSize <= lockedSize) return
        val ptr = basePtr ?: throw IllegalStateException("LlamaMlock not initialized — call init() first")

        val lockFrom = interpretCPointer<ByteVar>(ptr.rawValue + lockedSize) ?: return
        val lockLen = targetSize - lockedSize

        val result = nativeMlock(lockFrom, lockLen)
        if (result != 0) {
            // mlock can fail due to resource limits — log but don't throw
            // (matches llama.cpp behavior which warns but continues)
        }
        lockedSize = targetSize
    }

    /** Unlock all locked pages. */
    fun close() {
        val ptr = basePtr
        if (ptr != null && lockedSize > 0) {
            nativeMunlock(ptr, lockedSize)
            lockedSize = 0
        }
    }

    companion object {
        /** Whether memory locking is supported on the current platform. */
        val SUPPORTED: Boolean = nativeMlockSupported
    }
}

/**
 * Return the platform maximum path length.
 *
 * Port of `llama_path_max()` from `llama-mmap.h`.
 */
fun llamaPathMax(): Long = 4096  // PATH_MAX on most POSIX systems
