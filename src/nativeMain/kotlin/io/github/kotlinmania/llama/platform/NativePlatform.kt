@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.llama.platform

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.CPointer

internal expect fun nativeAlignedAlloc(
    sizeBytes: Long,
    alignmentBytes: Int,
): CPointer<ByteVar>?

internal expect fun nativeAlignedFree(pointer: CPointer<ByteVar>)

internal expect fun nativeMemcpy(
    destination: COpaquePointer?,
    source: COpaquePointer?,
    sizeBytes: Long,
)

internal expect fun nativeMemset(
    destination: COpaquePointer?,
    value: Int,
    sizeBytes: Long,
)

internal expect fun nativeFileSize(fd: Int): Long

internal expect fun nativeSeek(
    fd: Int,
    offset: Long,
): Long

internal expect fun nativeRead(
    fd: Int,
    destination: COpaquePointer?,
    sizeBytes: Long,
): Long

internal expect fun nativeWrite(
    fd: Int,
    source: COpaquePointer?,
    sizeBytes: Long,
): Long

internal expect fun nativePageSize(): Long

internal expect fun nativePhysicalPages(): Long

internal expect val nativeMmapSupported: Boolean

internal expect val nativeMlockSupported: Boolean

internal expect fun nativeMmapReadOnly(
    fd: Int,
    sizeBytes: Long,
): COpaquePointer?

internal expect fun nativeMunmap(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int

internal expect fun nativeMlock(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int

internal expect fun nativeMunlock(
    pointer: COpaquePointer?,
    sizeBytes: Long,
): Int

internal expect fun nativeWriteBytesToFile(
    bytes: ByteArray,
    path: String,
)
