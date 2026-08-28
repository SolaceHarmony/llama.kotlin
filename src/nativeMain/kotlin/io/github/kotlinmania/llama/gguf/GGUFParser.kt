// port-lint: source ggml/src/gguf.cpp
package io.github.kotlinmania.llama.gguf

import io.github.kotlinmania.llama.ore.GGMLType
import io.github.kotlinmania.llama.ore.GGML_MAX_DIMS
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe

// ---------------------------------------------------------------------------
// Padding helper – equivalent to C GGML_PAD(x, n)
// ---------------------------------------------------------------------------

/** Round [offset] up to the next multiple of [alignment]. */
private fun alignOffset(offset: Long, alignment: Long): Long {
    return ((offset + alignment - 1) / alignment) * alignment
}

// =====================================================================
// GGUFParser – byte-array-based reader (existing implementation, kept)
// =====================================================================

/**
 * GGUF file parser that operates on an in-memory [ByteArray].
 *
 * This is the *original* parser that reads the full file into a byte array
 * first.  For large model files prefer [GGUFFileReader] which streams from
 * disk and produces a [GGUFContextInternal].
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
        
        val offset = readUInt64().toLong()
        
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
}

// =====================================================================
// GGUFFileReader – streaming reader that produces GGUFContextInternal
// (mirrors gguf_init_from_file / gguf_reader from gguf.cpp)
// =====================================================================

/**
 * Reads a GGUF file from a [ByteArray] and produces a [GGUFContextInternal].
 *
 * This corresponds to the C++ `gguf_init_from_file_ptr` function.  It
 * validates the magic, version, and header, then reads KV pairs and tensor
 * info entries into the context.  Complex tensor-data loading is deferred
 * with `TODO`.
 */
object GGUFFileReader {
    /**
     * Create an empty [GGUFContextInternal] (mirrors `gguf_init_empty`).
     */
    fun initEmpty(): GGUFContextInternal = GGUFContextInternal()

    /**
     * Parse a GGUF file that has already been loaded into [bytes].
     *
     * This is the Kotlin equivalent of `gguf_init_from_file_ptr`.
     */
    fun initFromBytes(bytes: ByteArray, params: GGUFInitParams = GGUFInitParams()): GGUFContextInternal {
        val r = ByteArrayReader(bytes)
        val ctx = GGUFContextInternal()

        // -- magic --
        val magic = r.readBytes(4).decodeToString()
        require(magic == GGUFConstants.MAGIC) {
            "Invalid GGUF magic: '$magic', expected '${GGUFConstants.MAGIC}'"
        }

        // -- version --
        ctx.version = r.readUInt32()
        require(ctx.version != 0u) { "Bad GGUF version: 0" }
        require((ctx.version.toInt() and 0x0000FFFF) != 0) {
            "Endianness mismatch: version=${ctx.version}"
        }
        require(ctx.version != 1u) { "GGUFv1 is no longer supported" }
        require(ctx.version <= GGUFConstants.VERSION) {
            "GGUF version ${ctx.version} not supported (max ${GGUFConstants.VERSION})"
        }

        // -- tensor / kv counts --
        val nTensors = r.readInt64()
        require(nTensors >= 0) { "Invalid tensor count: $nTensors" }
        val nKv = r.readInt64()
        require(nKv >= 0) { "Invalid KV count: $nKv" }

        // -- read KV pairs --
        val seenKeys = mutableSetOf<String>()
        for (i in 0 until nKv) {
            val key = r.readGGUFString()
            require(key !in seenKeys) { "Duplicate KV key '$key' at index $i" }
            seenKeys.add(key)

            var type = GGUFType.fromValue(r.readInt32())
            var isArray = false
            var n = 1L
            if (type == GGUFType.ARRAY) {
                isArray = true
                type = GGUFType.fromValue(r.readInt32())
                n = r.readInt64().coerceAtLeast(0)
            }

            val kv = readKvValue(r, key, type, isArray, n)
            ctx.kv.add(kv)
        }

        // -- resolve alignment from metadata --
        val alignIdx = findKeyIndex(ctx, GGUFConstants.KEY_GENERAL_ALIGNMENT)
        if (alignIdx >= 0) {
            ctx.alignment = getValU32(ctx, alignIdx.toLong()).toLong()
        }
        require(ctx.alignment > 0 && (ctx.alignment and (ctx.alignment - 1)) == 0L) {
            "Alignment ${ctx.alignment} is not a power of 2"
        }

        // -- read tensor info --
        for (i in 0 until nTensors) {
            val entry = GGUFTensorInfoEntry()
            entry.name = r.readGGUFString()
            // duplicate name check
            for (j in 0 until ctx.info.size) {
                require(entry.name != ctx.info[j].name) {
                    "Duplicate tensor name '${entry.name}'"
                }
            }

            val nDims = r.readUInt32().toInt()
            require(nDims in 0..GGML_MAX_DIMS) {
                "Tensor '${entry.name}' has $nDims dims (max $GGML_MAX_DIMS)"
            }
            for (j in 0 until GGML_MAX_DIMS) {
                entry.ne[j] = if (j < nDims) r.readInt64() else 1L
                require(entry.ne[j] >= 0) {
                    "Tensor '${entry.name}' dim $j has negative size ${entry.ne[j]}"
                }
            }

            val typeVal = r.readInt32()
            entry.type = GGMLType.fromValue(typeVal)
                ?: throw IllegalArgumentException(
                    "Tensor '${entry.name}' has invalid type $typeVal"
                )

            // compute byte strides
            val typeSize = entry.type.byteSize.toLong()
            entry.nb[0] = typeSize
            entry.nb[1] = entry.nb[0] * entry.ne[0]
            for (j in 2 until GGML_MAX_DIMS) {
                entry.nb[j] = entry.nb[j - 1] * entry.ne[j - 1]
            }

            entry.offset = r.readUInt64()
            ctx.info.add(entry)
        }

        // -- compute data offset (aligned) and data size --
        ctx.dataOffset = alignOffset(r.position.toLong(), ctx.alignment)
        ctx.dataSize = 0L
        for (i in ctx.info.indices) {
            val ti = ctx.info[i]
            val padded = alignOffset(ti.numBytes(), ctx.alignment)
            ctx.dataSize += padded
        }

        // -- optionally load tensor data --
        if (params.loadData && !params.noAlloc) {
            val dataStart = ctx.dataOffset.toInt()
            val dataEnd = dataStart + ctx.dataSize.toInt()
            if (dataEnd <= bytes.size) {
                ctx.data = bytes.sliceArray(dataStart until dataEnd)
            }
        }

        return ctx
    }

    /**
     * Parse a GGUF file from the filesystem.
     *
     * Mirrors `gguf_init_from_file` in the C++ source.
     */
    fun initFromFile(path: String, params: GGUFInitParams = GGUFInitParams()): GGUFContextInternal {
        error("initFromFile not yet ported")
    }

    // -----------------------------------------------------------------
    // Internal: read a single KV value (scalar or array)
    // -----------------------------------------------------------------

    private fun readKvValue(
        r: ByteArrayReader,
        key: String,
        type: GGUFType,
        isArray: Boolean,
        n: Long
    ): GGUFKv {
        if (type == GGUFType.STRING) {
            val strings = mutableListOf<String>()
            val count = if (isArray) n else 1L
            for (j in 0 until count) {
                strings.add(r.readGGUFString())
            }
            return if (isArray) GGUFKv.ofStringArray(key, strings)
            else GGUFKv.ofString(key, strings[0])
        }

        val elemSize = type.typeSize
        require(elemSize > 0) { "Cannot read value of type $type" }
        val count = if (isArray) n else 1L
        val totalBytes = (count * elemSize).toInt()

        // For BOOL the wire format is int8 per element
        val raw = if (type == GGUFType.BOOL) {
            ByteArray(totalBytes).also { buf ->
                for (j in 0 until count) {
                    buf[j.toInt()] = r.readByte()
                }
            }
        } else {
            r.readBytes(totalBytes)
        }

        return if (isArray) GGUFKv(key, true, type, raw)
        else GGUFKv(key, false, type, raw)
    }
}

// =====================================================================
// ByteArrayReader – sequential little-endian reader
// =====================================================================

/**
 * Reads primitive values sequentially from a [ByteArray] in little-endian
 * byte order.  This is the Kotlin equivalent of the C++ `gguf_reader`
 * helper struct.
 */
internal class ByteArrayReader(private val data: ByteArray) {
    var position: Int = 0
        private set

    fun readByte(): Byte {
        check(position < data.size) { "End of data reached at offset $position" }
        return data[position++]
    }

    fun readBytes(n: Int): ByteArray {
        check(position + n <= data.size) {
            "Cannot read $n bytes at offset $position (size=${data.size})"
        }
        val result = data.copyOfRange(position, position + n)
        position += n
        return result
    }

    fun readUInt32(): UInt {
        val v = data.getIntLe(position).toUInt()
        position += 4
        return v
    }

    fun readInt32(): Int {
        val v = data.getIntLe(position)
        position += 4
        return v
    }

    fun readInt64(): Long {
        val v = data.getLongLe(position)
        position += 8
        return v
    }

    fun readUInt64(): ULong {
        val v = data.getLongLe(position).toULong()
        position += 8
        return v
    }

    /** Read a GGUF-encoded string: `uint64 length` followed by `length` UTF-8 bytes. */
    fun readGGUFString(): String {
        val len = readInt64()
        require(len in 0..GGUFConstants.MAX_STRING_LENGTH) {
            "String length $len out of range"
        }
        val bytes = readBytes(len.toInt())
        return bytes.decodeToString()
    }
}

// =====================================================================
// Accessor functions – mirrors the C gguf_get_* / gguf_find_* API
// =====================================================================

// These are top-level functions that operate on a GGUFContextInternal,
// matching the C function signatures.  Simple accessors are fully
// implemented; write / serialization helpers are deferred.

/** Return the file-format version (mirrors `gguf_get_version`). */
fun getVersion(ctx: GGUFContextInternal): UInt = ctx.version

/** Return the data-section byte alignment (mirrors `gguf_get_alignment`). */
fun getAlignment(ctx: GGUFContextInternal): Long = ctx.alignment

/** Return the byte offset of the tensor data section (mirrors `gguf_get_data_offset`). */
fun getDataOffset(ctx: GGUFContextInternal): Long = ctx.dataOffset

/** Number of KV pairs (mirrors `gguf_get_n_kv`). */
fun getNumKv(ctx: GGUFContextInternal): Long = ctx.kv.size.toLong()

/**
 * Find the index of a KV pair by [key], or −1 if not found.
 * Mirrors `gguf_find_key`.
 */
fun findKeyIndex(ctx: GGUFContextInternal, key: String): Int {
    return ctx.kv.indexOfFirst { it.key == key }
}

/** Return the key string at [keyId] (mirrors `gguf_get_key`). */
fun getKey(ctx: GGUFContextInternal, keyId: Long): String {
    require(keyId in 0 until ctx.kv.size) { "keyId $keyId out of range" }
    return ctx.kv[keyId.toInt()].key
}

/**
 * Return the [GGUFType] of the KV entry at [keyId].
 * For arrays this returns [GGUFType.ARRAY]; use [getArrType] for the
 * element type.  Mirrors `gguf_get_kv_type`.
 */
fun getKvType(ctx: GGUFContextInternal, keyId: Long): GGUFType {
    require(keyId in 0 until ctx.kv.size) { "keyId $keyId out of range" }
    val kv = ctx.kv[keyId.toInt()]
    return if (kv.isArray) GGUFType.ARRAY else kv.type
}

/** Element type of the array at [keyId] (mirrors `gguf_get_arr_type`). */
fun getArrType(ctx: GGUFContextInternal, keyId: Long): GGUFType {
    require(keyId in 0 until ctx.kv.size) { "keyId $keyId out of range" }
    val kv = ctx.kv[keyId.toInt()]
    require(kv.isArray) { "KV at $keyId is not an array" }
    return kv.type
}

/** Number of elements in the array at [keyId] (mirrors `gguf_get_arr_n`). */
fun getArrN(ctx: GGUFContextInternal, keyId: Long): Long {
    require(keyId in 0 until ctx.kv.size) { "keyId $keyId out of range" }
    return ctx.kv[keyId.toInt()].getNumElements()
}

// -- typed scalar getters (mirrors gguf_get_val_*) -------------------------

private fun requireScalar(ctx: GGUFContextInternal, keyId: Long): GGUFKv {
    require(keyId in 0 until ctx.kv.size) { "keyId $keyId out of range" }
    val kv = ctx.kv[keyId.toInt()]
    require(kv.getNumElements() == 1L) { "Expected scalar, got ${kv.getNumElements()} elements" }
    return kv
}

fun getValU8(ctx: GGUFContextInternal, keyId: Long): UByte {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.UINT8)
    return kv.data[0].toUByte()
}

fun getValI8(ctx: GGUFContextInternal, keyId: Long): Byte {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.INT8)
    return kv.data[0]
}

fun getValU16(ctx: GGUFContextInternal, keyId: Long): UShort {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.UINT16)
    return kv.data.getShortLe(0).toUShort()
}

fun getValI16(ctx: GGUFContextInternal, keyId: Long): Short {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.INT16)
    return kv.data.getShortLe(0)
}

fun getValU32(ctx: GGUFContextInternal, keyId: Long): UInt {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.UINT32)
    return kv.data.getIntLe(0).toUInt()
}

fun getValI32(ctx: GGUFContextInternal, keyId: Long): Int {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.INT32)
    return kv.data.getIntLe(0)
}

fun getValF32(ctx: GGUFContextInternal, keyId: Long): Float {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.FLOAT32)
    return kv.data.getFloatLe(0)
}

fun getValU64(ctx: GGUFContextInternal, keyId: Long): ULong {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.UINT64)
    return kv.data.getLongLe(0).toULong()
}

fun getValI64(ctx: GGUFContextInternal, keyId: Long): Long {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.INT64)
    return kv.data.getLongLe(0)
}

fun getValF64(ctx: GGUFContextInternal, keyId: Long): Double {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.FLOAT64)
    return Double.fromBits(kv.data.getLongLe(0))
}

fun getValBool(ctx: GGUFContextInternal, keyId: Long): Boolean {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.BOOL)
    return kv.data[0] != 0.toByte()
}

fun getValStr(ctx: GGUFContextInternal, keyId: Long): String {
    val kv = requireScalar(ctx, keyId)
    require(kv.type == GGUFType.STRING)
    return kv.dataString[0]
}

/** Raw bytes of a scalar numeric value (mirrors `gguf_get_val_data`). */
fun getValData(ctx: GGUFContextInternal, keyId: Long): ByteArray {
    val kv = requireScalar(ctx, keyId)
    require(kv.type != GGUFType.STRING)
    return kv.data
}

/** Raw bytes backing a non-string array (mirrors `gguf_get_arr_data`). */
fun getArrData(ctx: GGUFContextInternal, keyId: Long): ByteArray {
    require(keyId in 0 until ctx.kv.size)
    val kv = ctx.kv[keyId.toInt()]
    require(kv.type != GGUFType.STRING) { "Use getArrStr for string arrays" }
    return kv.data
}

/** Get the [i]-th string from a string array (mirrors `gguf_get_arr_str`). */
fun getArrStr(ctx: GGUFContextInternal, keyId: Long, i: Int): String {
    require(keyId in 0 until ctx.kv.size)
    val kv = ctx.kv[keyId.toInt()]
    require(kv.type == GGUFType.STRING) { "Not a string array" }
    return kv.dataString[i]
}

// -- tensor accessors (mirrors gguf_get_n_tensors, etc.) --------------------

/** Number of tensors in the context (mirrors `gguf_get_n_tensors`). */
fun getNumTensors(ctx: GGUFContextInternal): Long = ctx.info.size.toLong()

/**
 * Find tensor by [name], returning its index or −1 if not found.
 * Mirrors `gguf_find_tensor`.
 */
fun findTensorIndex(ctx: GGUFContextInternal, name: String): Int {
    return ctx.info.indexOfFirst { it.name == name }
}

/** Byte offset of tensor [tensorId] within the data section (mirrors `gguf_get_tensor_offset`). */
fun getTensorOffset(ctx: GGUFContextInternal, tensorId: Long): ULong {
    require(tensorId in 0 until ctx.info.size) { "tensorId $tensorId out of range" }
    return ctx.info[tensorId.toInt()].offset
}

/** Name of tensor [tensorId] (mirrors `gguf_get_tensor_name`). */
fun getTensorName(ctx: GGUFContextInternal, tensorId: Long): String {
    require(tensorId in 0 until ctx.info.size) { "tensorId $tensorId out of range" }
    return ctx.info[tensorId.toInt()].name
}

/** Data type of tensor [tensorId] (mirrors `gguf_get_tensor_type`). */
fun getTensorType(ctx: GGUFContextInternal, tensorId: Long): GGMLType {
    require(tensorId in 0 until ctx.info.size) { "tensorId $tensorId out of range" }
    return ctx.info[tensorId.toInt()].type
}

/** Byte size of tensor [tensorId] (mirrors `gguf_get_tensor_size`). */
fun getTensorSize(ctx: GGUFContextInternal, tensorId: Long): Long {
    require(tensorId in 0 until ctx.info.size) { "tensorId $tensorId out of range" }
    return ctx.info[tensorId.toInt()].numBytes()
}

// =====================================================================
// Mutator functions – mirrors gguf_set_val_*, gguf_remove_key, etc.
// =====================================================================

/**
 * Remove the KV entry with [key] if present.
 * Returns the former index or −1 if the key did not exist.
 * Mirrors `gguf_remove_key`.
 */
fun removeKey(ctx: GGUFContextInternal, key: String): Int {
    val idx = findKeyIndex(ctx, key)
    if (idx >= 0) ctx.kv.removeAt(idx)
    return idx
}

/**
 * Validate reserved keys.  Currently only `general.alignment` is
 * reserved and must be a power-of-two `u32`.
 */
private fun checkReservedKeys(key: String, type: GGUFType) {
    if (key == GGUFConstants.KEY_GENERAL_ALIGNMENT) {
        require(type == GGUFType.UINT32) {
            "${GGUFConstants.KEY_GENERAL_ALIGNMENT} must be type u32"
        }
    }
}

// -- typed scalar setters (mirrors gguf_set_val_*) -------------------------

fun setValU8(ctx: GGUFContextInternal, key: String, value: UByte) {
    checkReservedKeys(key, GGUFType.UINT8); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofU8(key, value))
}

fun setValI8(ctx: GGUFContextInternal, key: String, value: Byte) {
    checkReservedKeys(key, GGUFType.INT8); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofI8(key, value))
}

fun setValU16(ctx: GGUFContextInternal, key: String, value: UShort) {
    checkReservedKeys(key, GGUFType.UINT16); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofU16(key, value))
}

fun setValI16(ctx: GGUFContextInternal, key: String, value: Short) {
    checkReservedKeys(key, GGUFType.INT16); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofI16(key, value))
}

fun setValU32(ctx: GGUFContextInternal, key: String, value: UInt) {
    checkReservedKeys(key, GGUFType.UINT32); removeKey(ctx, key)
    if (key == GGUFConstants.KEY_GENERAL_ALIGNMENT) {
        require(value > 0u && (value and (value - 1u)) == 0u) {
            "${GGUFConstants.KEY_GENERAL_ALIGNMENT} must be a power of 2"
        }
    }
    ctx.kv.add(GGUFKv.ofU32(key, value))
}

fun setValI32(ctx: GGUFContextInternal, key: String, value: Int) {
    checkReservedKeys(key, GGUFType.INT32); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofI32(key, value))
}

fun setValF32(ctx: GGUFContextInternal, key: String, value: Float) {
    checkReservedKeys(key, GGUFType.FLOAT32); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofF32(key, value))
}

fun setValU64(ctx: GGUFContextInternal, key: String, value: ULong) {
    checkReservedKeys(key, GGUFType.UINT64); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofU64(key, value))
}

fun setValI64(ctx: GGUFContextInternal, key: String, value: Long) {
    checkReservedKeys(key, GGUFType.INT64); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofI64(key, value))
}

fun setValF64(ctx: GGUFContextInternal, key: String, value: Double) {
    checkReservedKeys(key, GGUFType.FLOAT64); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofF64(key, value))
}

fun setValBool(ctx: GGUFContextInternal, key: String, value: Boolean) {
    checkReservedKeys(key, GGUFType.BOOL); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofBool(key, value))
}

fun setValStr(ctx: GGUFContextInternal, key: String, value: String) {
    checkReservedKeys(key, GGUFType.STRING); removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofString(key, value))
}

/**
 * Set an array of raw (non-string) data (mirrors `gguf_set_arr_data`).
 *
 * @param type  element type of the array
 * @param data  raw little-endian bytes, length must be `n * type.typeSize`
 * @param n     number of elements
 */
fun setArrData(ctx: GGUFContextInternal, key: String, type: GGUFType, data: ByteArray, n: Long) {
    removeKey(ctx, key)
    val kv = GGUFKv.ofDataArray(key, GGUFType.INT8, data)
    kv.cast(type)
    ctx.kv.add(kv)
}

/** Set a string array (mirrors `gguf_set_arr_str`). */
fun setArrStr(ctx: GGUFContextInternal, key: String, values: List<String>) {
    removeKey(ctx, key)
    ctx.kv.add(GGUFKv.ofStringArray(key, values))
}

/**
 * Copy all KV pairs from [src] into [ctx], overwriting duplicates.
 * Mirrors `gguf_set_kv`.
 */
fun setKv(ctx: GGUFContextInternal, src: GGUFContextInternal) {
    for (kv in src.kv) {
        if (!kv.isArray) {
            when (kv.type) {
                GGUFType.UINT8   -> setValU8(ctx, kv.key, kv.data[0].toUByte())
                GGUFType.INT8    -> setValI8(ctx, kv.key, kv.data[0])
                GGUFType.UINT16  -> setValU16(ctx, kv.key, kv.data.getShortLe(0).toUShort())
                GGUFType.INT16   -> setValI16(ctx, kv.key, kv.data.getShortLe(0))
                GGUFType.UINT32  -> setValU32(ctx, kv.key, kv.data.getIntLe(0).toUInt())
                GGUFType.INT32   -> setValI32(ctx, kv.key, kv.data.getIntLe(0))
                GGUFType.FLOAT32 -> setValF32(ctx, kv.key, kv.data.getFloatLe(0))
                GGUFType.UINT64  -> setValU64(ctx, kv.key, kv.data.getLongLe(0).toULong())
                GGUFType.INT64   -> setValI64(ctx, kv.key, kv.data.getLongLe(0))
                GGUFType.FLOAT64 -> setValF64(ctx, kv.key, Double.fromBits(kv.data.getLongLe(0)))
                GGUFType.BOOL    -> setValBool(ctx, kv.key, kv.data[0] != 0.toByte())
                GGUFType.STRING  -> setValStr(ctx, kv.key, kv.dataString[0])
                GGUFType.ARRAY   -> error("Invalid scalar type ARRAY")
            }
        } else {
            if (kv.type == GGUFType.STRING) {
                setArrStr(ctx, kv.key, kv.dataString.toList())
            } else {
                setArrData(ctx, kv.key, kv.type, kv.data.copyOf(), kv.getNumElements())
            }
        }
    }
}

// =====================================================================
// Tensor mutation (mirrors gguf_add_tensor, gguf_set_tensor_*)
// =====================================================================

/**
 * Add a new tensor entry to the context.  The name must be unique.
 * Mirrors `gguf_add_tensor`.
 */
fun addTensor(
    ctx: GGUFContextInternal,
    name: String,
    ne: LongArray,
    type: GGMLType
) {
    require(findTensorIndex(ctx, name) == -1) { "Duplicate tensor name: $name" }
    val entry = GGUFTensorInfoEntry(name = name, type = type)
    for (i in ne.indices.take(GGML_MAX_DIMS)) entry.ne[i] = ne[i]

    // compute strides
    val typeSize = type.byteSize.toLong()
    entry.nb[0] = typeSize
    entry.nb[1] = entry.nb[0] * entry.ne[0]
    for (j in 2 until GGML_MAX_DIMS) {
        entry.nb[j] = entry.nb[j - 1] * entry.ne[j - 1]
    }

    entry.offset = if (ctx.info.isEmpty()) 0uL else {
        val prev = ctx.info.last()
        (prev.offset.toLong() + alignOffset(prev.numBytes(), ctx.alignment)).toULong()
    }
    ctx.info.add(entry)
}

/**
 * Change the data type of an existing tensor and recompute offsets for
 * all subsequent tensors.  Mirrors `gguf_set_tensor_type`.
 */
fun setTensorType(ctx: GGUFContextInternal, name: String, type: GGMLType) {
    val id = findTensorIndex(ctx, name)
    require(id >= 0) { "Tensor not found: $name" }
    val tensor = ctx.info[id]
    tensor.type = type

    val typeSize = type.byteSize.toLong()
    tensor.nb[0] = typeSize
    tensor.nb[1] = tensor.nb[0] * tensor.ne[0]
    for (j in 2 until GGML_MAX_DIMS) {
        tensor.nb[j] = tensor.nb[j - 1] * tensor.ne[j - 1]
    }

    // recalculate offsets for all subsequent tensors
    for (i in (id + 1) until ctx.info.size) {
        val prev = ctx.info[i - 1]
        ctx.info[i].offset =
            (prev.offset.toLong() + alignOffset(prev.numBytes(), ctx.alignment)).toULong()
    }
}

// =====================================================================
// Write / serialization (mirrors gguf_write_to_file, gguf_get_meta_*)
// =====================================================================

/**
 * Write the entire [ctx] to a binary GGUF byte array.
 *
 * When [onlyMeta] is `true` only the header, KV pairs, tensor info, and
 * padding are written (no tensor data).
 *
 * Mirrors `gguf_write_to_buf` / `gguf_write_out`.
 */
fun writeToByteArray(ctx: GGUFContextInternal, onlyMeta: Boolean = false): ByteArray {
    error("writeToByteArray not yet ported")
}

/**
 * Write the entire [ctx] to a file at [path].
 * Mirrors `gguf_write_to_file`.
 */
fun writeToFile(ctx: GGUFContextInternal, path: String, onlyMeta: Boolean = false): Boolean {
    error("writeToFile not yet ported")
}

/**
 * Calculate the size in bytes of the meta data (header + KV + tensor info +
 * padding).  Mirrors `gguf_get_meta_size`.
 */
fun getMetaSize(ctx: GGUFContextInternal): Long {
    error("getMetaSize not yet ported")
}

/**
 * Serialize only the meta data (header + KV + tensor info) into a [ByteArray].
 * Mirrors `gguf_get_meta_data`.
 */
fun getMetaData(ctx: GGUFContextInternal): ByteArray {
    error("getMetaData not yet ported")
}
