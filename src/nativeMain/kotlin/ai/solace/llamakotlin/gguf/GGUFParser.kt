package ai.solace.llamakotlin.gguf

import ai.solace.llamakotlin.core.GGMLType
import ai.solace.llamakotlin.core.getLongLe
import ai.solace.llamakotlin.core.getIntLe
import ai.solace.llamakotlin.core.getShortLe
import ai.solace.llamakotlin.core.getFloatLe
import kotlin.math.pow

/**
 * GGUF file parser for reading binary model files
 */
class GGUFParser(private val data: ByteArray) {
    private var position = 0

    /**
     * Parse a GGUF file from the binary data
     */
    fun parse(): GGUFContext {
        // Read header
        val magic = readString(4)
        if (magic != GGUFConstants.MAGIC) {
            throw IllegalArgumentException("Invalid GGUF magic: $magic")
        }

        val version = readUInt32()
        val tensorCount = readUInt64()
        val metadataKvCount = readUInt64()

        // Read metadata key-value pairs
        val metadata = mutableMapOf<String, GGUFKeyValue>()
        repeat(metadataKvCount.toInt()) {
            val kv = readKeyValue()
            metadata[kv.key] = kv
        }

        // Read tensor information
        val tensors = mutableListOf<GGUFTensorInfo>()
        repeat(tensorCount.toInt()) {
            val tensorInfo = readTensorInfo()
            tensors.add(tensorInfo)
        }

        // Calculate alignment and data offset
        val alignment = metadata["general.alignment"]?.value as? Long ?: GGUFConstants.DEFAULT_ALIGNMENT
        val dataOffset = alignOffset(position.toLong(), alignment)

        return GGUFContext(
            version = version,
            metadata = metadata,
            tensors = tensors,
            dataOffset = dataOffset,
            alignment = alignment,
            rawData = data
        )
    }

    private fun readKeyValue(): GGUFKeyValue {
        val keyLength = readUInt64()
        val key = readString(keyLength.toInt())
        val typeValue = readUInt32()
        val type = GGUFType.fromValue(typeValue.toInt())
        
        val value = when (type) {
            GGUFType.UINT8 -> readUInt8()
            GGUFType.INT8 -> readInt8()
            GGUFType.UINT16 -> readUInt16()
            GGUFType.INT16 -> readInt16()
            GGUFType.UINT32 -> readUInt32()
            GGUFType.INT32 -> readInt32()
            GGUFType.FLOAT32 -> readFloat32()
            GGUFType.BOOL -> readBool()
            GGUFType.STRING -> {
                val strLength = readUInt64()
                readString(strLength.toInt())
            }
            GGUFType.ARRAY -> readArray()
            GGUFType.UINT64 -> readUInt64()
            GGUFType.INT64 -> readInt64()
            GGUFType.FLOAT64 -> readFloat64()
        }

        return GGUFKeyValue(key, type, value)
    }

    private fun readTensorInfo(): GGUFTensorInfo {
        val nameLength = readUInt64()
        val name = readString(nameLength.toInt())
        
        val nDims = readUInt32()
        val dimensions = IntArray(nDims.toInt()) { readUInt64().toInt() }
        
        val typeValue = readUInt32()
        val type = GGMLType.fromValue(typeValue.toInt())
            ?: throw IllegalArgumentException("Unknown tensor type: $typeValue")
        
        val offset = readUInt64()
        
        return GGUFTensorInfo(name, dimensions, type, offset)
    }

    private fun readArray(): List<Any> {
        val arrayType = GGUFType.fromValue(readUInt32().toInt())
        val arrayLength = readUInt64()
        
        return (0 until arrayLength.toInt()).map {
            when (arrayType) {
                GGUFType.UINT8 -> readUInt8()
                GGUFType.INT8 -> readInt8()
                GGUFType.UINT16 -> readUInt16()
                GGUFType.INT16 -> readInt16()
                GGUFType.UINT32 -> readUInt32()
                GGUFType.INT32 -> readInt32()
                GGUFType.FLOAT32 -> readFloat32()
                GGUFType.BOOL -> readBool()
                GGUFType.STRING -> {
                    val strLength = readUInt64()
                    readString(strLength.toInt())
                }
                GGUFType.UINT64 -> readUInt64()
                GGUFType.INT64 -> readInt64()
                GGUFType.FLOAT64 -> readFloat64()
                GGUFType.ARRAY -> throw IllegalArgumentException("Nested arrays not supported")
            }
        }
    }

    private fun readUInt8(): UByte {
        if (position >= data.size) throw IndexOutOfBoundsException("End of data reached")
        return data[position++].toUByte()
    }

    private fun readInt8(): Byte {
        if (position >= data.size) throw IndexOutOfBoundsException("End of data reached")
        return data[position++]
    }

    private fun readUInt16(): UShort {
        val result = data.getShortLe(position).toUShort()
        position += 2
        return result
    }

    private fun readInt16(): Short {
        val result = data.getShortLe(position)
        position += 2
        return result
    }

    private fun readUInt32(): UInt {
        val result = data.getIntLe(position).toUInt()
        position += 4
        return result
    }

    private fun readInt32(): Int {
        val result = data.getIntLe(position)
        position += 4
        return result
    }

    private fun readFloat32(): Float {
        val result = data.getFloatLe(position)
        position += 4
        return result
    }

    private fun readUInt64(): ULong {
        val result = data.getLongLe(position).toULong()
        position += 8
        return result
    }

    private fun readInt64(): Long {
        val result = data.getLongLe(position)
        position += 8
        return result
    }

    private fun readFloat64(): Double {
        val result = Double.fromBits(data.getLongLe(position))
        position += 8
        return result
    }

    private fun readBool(): Boolean {
        return readUInt8() != 0.toUByte()
    }

    private fun readString(length: Int): String {
        if (position + length > data.size) {
            throw IndexOutOfBoundsException("String length $length exceeds available data")
        }
        val result = data.sliceArray(position until position + length).decodeToString()
        position += length
        return result
    }

    private fun alignOffset(offset: Long, alignment: Long): Long {
        return ((offset + alignment - 1) / alignment) * alignment
    }
}