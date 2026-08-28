// port-lint: source llama.cpp/src/llama-io.h
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// LlamaIO – serialization / deserialization interfaces
// Ported from: llama-io.h
// =============================================================================

/**
 * Interface for writing serialized state data.
 *
 * Implementations may write to a byte buffer, a file, or a network stream.
 *
 * Port of `llama_io_write_i` from `llama-io.h`.
 */
interface LlamaIoWrite {

    /**
     * Write raw bytes from [src].
     *
     * @param src   Source byte array.
     * @param size  Number of bytes to write (starting from index 0).
     */
    fun write(src: ByteArray, size: Int = src.size)

    /**
     * Write the data of a [tensor], starting at byte [offset] for [size] bytes.
     *
     * @param tensor  The tensor whose backing data should be written.
     * @param offset  Byte offset into the tensor's data buffer.
     * @param size    Number of bytes to write.
     */
    fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int)

    /** Total number of bytes written so far. */
    fun nBytes(): Long

    /**
     * Convenience: write a [String] as a length-prefixed sequence of bytes.
     *
     * Port of `llama_io_write_i::write_string()`.
     */
    fun writeString(str: String) {
        val bytes = str.encodeToByteArray()
        val lenBytes = ByteArray(4)
        lenBytes[0] = (bytes.size and 0xFF).toByte()
        lenBytes[1] = ((bytes.size shr 8) and 0xFF).toByte()
        lenBytes[2] = ((bytes.size shr 16) and 0xFF).toByte()
        lenBytes[3] = ((bytes.size shr 24) and 0xFF).toByte()
        write(lenBytes)
        write(bytes)
    }
}

/**
 * Interface for reading serialized state data.
 *
 * Port of `llama_io_read_i` from `llama-io.h`.
 */
interface LlamaIoRead {

    /**
     * Read [size] bytes and return them as a new [ByteArray].
     *
     * @param size  Number of bytes to read.
     * @return A byte array of exactly [size] bytes.
     */
    fun read(size: Int): ByteArray

    /**
     * Read [size] bytes into the provided [dst] buffer.
     *
     * @param dst   Destination byte array (must be at least [size] bytes).
     * @param size  Number of bytes to read.
     */
    fun readTo(dst: ByteArray, size: Int)

    /** Total number of bytes read so far. */
    fun nBytes(): Long

    /**
     * Convenience: read a length-prefixed [String].
     *
     * Port of `llama_io_read_i::read_string()`.
     */
    fun readString(): String {
        val lenBytes = read(4)
        val len = (lenBytes[0].toInt() and 0xFF) or
                ((lenBytes[1].toInt() and 0xFF) shl 8) or
                ((lenBytes[2].toInt() and 0xFF) shl 16) or
                ((lenBytes[3].toInt() and 0xFF) shl 24)
        val strBytes = read(len)
        return strBytes.decodeToString()
    }
}

// =============================================================================
// Concrete implementations – byte-array-backed I/O
// =============================================================================

/**
 * [LlamaIoWrite] backed by a growable [ByteArray].
 *
 * Useful for serialising state into memory before flushing to disk or network.
 */
class ByteArrayIoWriter(initialCapacity: Int = 4096) : LlamaIoWrite {
    private var buffer = ByteArray(initialCapacity)
    private var position = 0

    private fun ensureCapacity(needed: Int) {
        val required = position + needed
        if (required <= buffer.size) return
        var newCap = buffer.size * 2
        while (newCap < required) newCap *= 2
        buffer = buffer.copyOf(newCap)
    }

    override fun write(src: ByteArray, size: Int) {
        ensureCapacity(size)
        src.copyInto(buffer, position, 0, size)
        position += size
    }

    override fun writeTensor(tensor: GGMLTensor, offset: Int, size: Int) {
        val data = tensor.data as? ByteArray
            ?: error("writeTensor: tensor '${tensor.name}' has no ByteArray data")
        ensureCapacity(size)
        data.copyInto(buffer, position, offset, offset + size)
        position += size
    }

    override fun nBytes(): Long = position.toLong()

    /** Return the written bytes as a trimmed [ByteArray]. */
    fun toByteArray(): ByteArray = buffer.copyOf(position)
}

/**
 * [LlamaIoRead] backed by a [ByteArray].
 *
 * Useful for deserialising state that was previously written by
 * [ByteArrayIoWriter] or loaded from a file.
 */
class ByteArrayIoReader(private val data: ByteArray) : LlamaIoRead {
    private var position = 0

    override fun read(size: Int): ByteArray {
        require(position + size <= data.size) {
            "ByteArrayIoReader: cannot read $size bytes at position $position " +
                    "(buffer size: ${data.size})"
        }
        val result = data.copyOfRange(position, position + size)
        position += size
        return result
    }

    override fun readTo(dst: ByteArray, size: Int) {
        require(position + size <= data.size) {
            "ByteArrayIoReader: cannot read $size bytes at position $position " +
                    "(buffer size: ${data.size})"
        }
        data.copyInto(dst, 0, position, position + size)
        position += size
    }

    override fun nBytes(): Long = position.toLong()

    /** Number of bytes remaining in the buffer. */
    fun remaining(): Int = data.size - position
}

// =============================================================================
// Typed read/write helpers
// =============================================================================

/** Write a little-endian UInt (4 bytes). */
fun LlamaIoWrite.writeU32(value: UInt) {
    val bytes = ByteArray(4)
    bytes[0] = (value.toInt() and 0xFF).toByte()
    bytes[1] = ((value.toInt() shr 8) and 0xFF).toByte()
    bytes[2] = ((value.toInt() shr 16) and 0xFF).toByte()
    bytes[3] = ((value.toInt() shr 24) and 0xFF).toByte()
    write(bytes)
}

/** Write a little-endian Int (4 bytes). */
fun LlamaIoWrite.writeI32(value: Int) = writeU32(value.toUInt())

/** Read a little-endian UInt (4 bytes). */
fun LlamaIoRead.readU32(): UInt {
    val bytes = read(4)
    return ((bytes[0].toInt() and 0xFF) or
            ((bytes[1].toInt() and 0xFF) shl 8) or
            ((bytes[2].toInt() and 0xFF) shl 16) or
            ((bytes[3].toInt() and 0xFF) shl 24)).toUInt()
}

/** Read a little-endian Int (4 bytes). */
fun LlamaIoRead.readI32(): Int = readU32().toInt()

/** Write a little-endian Float (4 bytes, IEEE 754). */
fun LlamaIoWrite.writeF32(value: Float) = writeI32(value.toRawBits())

/** Read a little-endian Float (4 bytes, IEEE 754). */
fun LlamaIoRead.readF32(): Float = Float.fromBits(readI32())
