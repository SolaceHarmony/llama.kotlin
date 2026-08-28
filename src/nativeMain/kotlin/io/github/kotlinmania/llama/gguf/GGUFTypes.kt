// port-lint: source ggml/include/gguf.h
package io.github.kotlinmania.llama.gguf

import io.github.kotlinmania.llama.ore.GGMLType
import io.github.kotlinmania.llama.ore.GGML_MAX_DIMS

// ---------------------------------------------------------------------------
// GGUF file-format constants
// ---------------------------------------------------------------------------

/**
 * GGUF header constants matching the C defines in gguf.h.
 *
 * A GGUF file starts with the four-byte magic `"GGUF"`, followed by a version
 * number, tensor count, KV-pair count, then the KV and tensor-info sections.
 * All multi-byte integers are little-endian.
 */
object GGUFConstants {
    /** Four-byte magic at the start of every GGUF file. */
    const val MAGIC = "GGUF"

    /** Current GGUF format version this implementation supports (up to). */
    const val VERSION: UInt = 3u

    /** Metadata key whose `u32` value overrides the default alignment. */
    const val KEY_GENERAL_ALIGNMENT = "general.alignment"

    /** Default byte-alignment for the tensor-data section. */
    const val DEFAULT_ALIGNMENT: Long = 32L

    /** Safety cap: longest string we accept when reading a file (1 GiB). */
    const val MAX_STRING_LENGTH: Long = 1024L * 1024L * 1024L

    /** Safety cap: largest array element count we accept (1 Gi elements). */
    const val MAX_ARRAY_ELEMENTS: Long = 1024L * 1024L * 1024L
}

// ---------------------------------------------------------------------------
// GGUFType – value types that can appear in GGUF key-value metadata
// ---------------------------------------------------------------------------

/**
 * Types that can be stored as GGUF KV data.
 *
 * Mirrors the C enum `gguf_type`.  Every enum entry carries its wire value
 * ([value]), the byte size of a single scalar of that type ([typeSize], zero
 * for STRING and ARRAY which are variable-length), and a short display name
 * ([typeName]).
 */
enum class GGUFType(
    val value: Int,
    val typeSize: Int,
    val typeName: String
) {
    UINT8  (0,  1, "u8"),
    INT8   (1,  1, "i8"),
    UINT16 (2,  2, "u16"),
    INT16  (3,  2, "i16"),
    UINT32 (4,  4, "u32"),
    INT32  (5,  4, "i32"),
    FLOAT32(6,  4, "f32"),
    BOOL   (7,  1, "bool"),
    STRING (8,  0, "str"),
    ARRAY  (9,  0, "arr"),
    UINT64 (10, 8, "u64"),
    INT64  (11, 8, "i64"),
    FLOAT64(12, 8, "f64");

    companion object {
        /** Look up a [GGUFType] by its wire [value]; throws on unknown values. */
        fun fromValue(value: Int): GGUFType {
            return entries.find { it.value == value }
                ?: throw IllegalArgumentException("Unknown GGUF type: $value")
        }
    }
}

/** Port of `gguf_type_size` from gguf.cpp line 126. Returns 0 for STRING/ARRAY. */
fun gguf_type_size(type: GGUFType): Int = type.typeSize

// ---------------------------------------------------------------------------
// GGUFKv – internal key-value entry (mirrors C++ struct gguf_kv)
// ---------------------------------------------------------------------------

/**
 * A single key-value entry inside a [GGUFContextInternal].
 *
 * Scalar numeric values and bool arrays are stored in [data] as raw
 * little-endian bytes; string scalars and string arrays are stored in
 * [dataString].  The [isArray] flag distinguishes scalars from arrays.
 *
 * This is the Kotlin equivalent of the C++ `struct gguf_kv`.
 */
class GGUFKv(
    val key: String,
    val isArray: Boolean,
    var type: GGUFType,
    val data: ByteArray = ByteArray(0),
    val dataString: MutableList<String> = mutableListOf()
) {
    init {
        require(key.isNotEmpty()) { "GGUF KV key must not be empty" }
    }

    /** Number of logical elements (1 for scalars, n for arrays). */
    fun getNumElements(): Long {
        if (type == GGUFType.STRING) {
            val ne = dataString.size.toLong()
            require(isArray || ne == 1L)
            return ne
        }
        val ts = type.typeSize
        require(ts > 0) { "Cannot compute element count for type $type" }
        require(data.size % ts == 0)
        val ne = data.size.toLong() / ts
        require(isArray || ne == 1L)
        return ne
    }

    /** Re-interpret the raw [data] bytes as a different [GGUFType]. */
    fun cast(newType: GGUFType) {
        val newSize = newType.typeSize
        require(newSize > 0 && data.size % newSize == 0)
        type = newType
    }

    companion object {
        // -- Scalar constructors ------------------------------------------------

        fun ofU8(key: String, value: UByte): GGUFKv {
            val d = ByteArray(1); d[0] = value.toByte()
            return GGUFKv(key, false, GGUFType.UINT8, d)
        }
        fun ofI8(key: String, value: Byte): GGUFKv {
            val d = ByteArray(1); d[0] = value
            return GGUFKv(key, false, GGUFType.INT8, d)
        }
        fun ofU16(key: String, value: UShort): GGUFKv {
            val v = value.toInt()
            val d = byteArrayOf((v and 0xFF).toByte(), (v shr 8 and 0xFF).toByte())
            return GGUFKv(key, false, GGUFType.UINT16, d)
        }
        fun ofI16(key: String, value: Short): GGUFKv {
            val v = value.toInt()
            val d = byteArrayOf((v and 0xFF).toByte(), (v shr 8 and 0xFF).toByte())
            return GGUFKv(key, false, GGUFType.INT16, d)
        }
        fun ofU32(key: String, value: UInt): GGUFKv {
            val v = value.toInt()
            val d = ByteArray(4)
            d[0] = (v and 0xFF).toByte(); d[1] = (v shr 8 and 0xFF).toByte()
            d[2] = (v shr 16 and 0xFF).toByte(); d[3] = (v shr 24 and 0xFF).toByte()
            return GGUFKv(key, false, GGUFType.UINT32, d)
        }
        fun ofI32(key: String, value: Int): GGUFKv {
            val d = ByteArray(4)
            d[0] = (value and 0xFF).toByte(); d[1] = (value shr 8 and 0xFF).toByte()
            d[2] = (value shr 16 and 0xFF).toByte(); d[3] = (value shr 24 and 0xFF).toByte()
            return GGUFKv(key, false, GGUFType.INT32, d)
        }
        fun ofF32(key: String, value: Float): GGUFKv {
            val bits = value.toRawBits()
            val d = ByteArray(4)
            d[0] = (bits and 0xFF).toByte(); d[1] = (bits shr 8 and 0xFF).toByte()
            d[2] = (bits shr 16 and 0xFF).toByte(); d[3] = (bits shr 24 and 0xFF).toByte()
            return GGUFKv(key, false, GGUFType.FLOAT32, d)
        }
        fun ofU64(key: String, value: ULong): GGUFKv {
            val v = value.toLong()
            val d = ByteArray(8)
            for (i in 0..7) d[i] = (v shr (i * 8) and 0xFF).toByte()
            return GGUFKv(key, false, GGUFType.UINT64, d)
        }
        fun ofI64(key: String, value: Long): GGUFKv {
            val d = ByteArray(8)
            for (i in 0..7) d[i] = (value shr (i * 8) and 0xFF).toByte()
            return GGUFKv(key, false, GGUFType.INT64, d)
        }
        fun ofF64(key: String, value: Double): GGUFKv {
            val bits = value.toRawBits()
            val d = ByteArray(8)
            for (i in 0..7) d[i] = (bits shr (i * 8) and 0xFF).toByte()
            return GGUFKv(key, false, GGUFType.FLOAT64, d)
        }
        fun ofBool(key: String, value: Boolean): GGUFKv {
            val d = byteArrayOf(if (value) 1 else 0)
            return GGUFKv(key, false, GGUFType.BOOL, d)
        }
        fun ofString(key: String, value: String): GGUFKv {
            return GGUFKv(key, false, GGUFType.STRING, dataString = mutableListOf(value))
        }

        // -- Array constructors -------------------------------------------------

        fun ofStringArray(key: String, values: List<String>): GGUFKv {
            return GGUFKv(key, true, GGUFType.STRING, dataString = values.toMutableList())
        }

        fun ofDataArray(key: String, type: GGUFType, rawBytes: ByteArray): GGUFKv {
            return GGUFKv(key, true, type, rawBytes)
        }
    }
}

// ---------------------------------------------------------------------------
// GGUFTensorInfoEntry – internal tensor metadata (mirrors C++ gguf_tensor_info)
// ---------------------------------------------------------------------------

/**
 * Internal tensor metadata entry inside a [GGUFContextInternal].
 *
 * Mirrors the C++ `struct gguf_tensor_info`.  Holds the tensor's name,
 * element counts per dimension ([ne]), byte strides ([nb]), data type,
 * and the byte offset of this tensor's data within the data section.
 */
class GGUFTensorInfoEntry(
    var name: String = "",
    val ne: LongArray = LongArray(GGML_MAX_DIMS) { 1L },
    val nb: LongArray = LongArray(GGML_MAX_DIMS),
    var type: GGMLType = GGMLType.F32,
    var offset: ULong = 0uL
) {
    /** Number of dimensions that are greater than 1 (or at least 1). */
    fun nDims(): Int {
        for (i in GGML_MAX_DIMS - 1 downTo 1) {
            if (ne[i] > 1L) return i + 1
        }
        return 1
    }

    /** Total number of elements across all dimensions. */
    fun numElements(): Long = ne[0] * ne[1] * ne[2] * ne[3]

    /**
     * Total byte size of the tensor data.
     *
     * For unquantized types this is simply `numElements * typeSize`.
     * For quantized types [type.byteSize] already represents per-block size
     * and the block element count is encoded in the type's quant constant.
     * A more precise calculation may be added when block-size helpers land on
     * [GGMLType]; for now this gives a correct answer for all non-quantized
     * types and a usable upper-bound for quantized ones.
     */
    fun numBytes(): Long {
        val typeSize = type.byteSize.toLong()
        if (typeSize == 0L) return 0L
        return numElements() * typeSize
    }
}

// ---------------------------------------------------------------------------
// GGUFContextInternal – full mutable context (mirrors C++ struct gguf_context)
// ---------------------------------------------------------------------------

/**
 * Mutable GGUF context that mirrors the C++ `struct gguf_context`.
 *
 * This is the central data structure for reading, modifying, and writing
 * GGUF files.  It holds:
 * - [version]: the GGUF format version read from the file header.
 * - [kv]: an ordered list of key-value metadata entries.
 * - [info]: an ordered list of tensor metadata entries.
 * - [alignment]: byte alignment for the tensor data section.
 * - [dataOffset]: byte offset from the start of the file to the first byte of
 *   tensor data.
 * - [dataSize]: total size in bytes of the tensor data section.
 * - [data]: optional raw tensor data blob (only populated when the caller
 *   requests it during [GGUFParser.initFromFile]).
 */
class GGUFContextInternal(
    var version: UInt = GGUFConstants.VERSION,
    val kv: MutableList<GGUFKv> = mutableListOf(),
    val info: MutableList<GGUFTensorInfoEntry> = mutableListOf(),
    var alignment: Long = GGUFConstants.DEFAULT_ALIGNMENT,
    var dataOffset: Long = 0L,
    var dataSize: Long = 0L,
    var data: ByteArray? = null
)

// ---------------------------------------------------------------------------
// GGUFInitParams – options for gguf_init_from_file
// ---------------------------------------------------------------------------

/**
 * Parameters controlling how [GGUFParser.initFromFile] loads a GGUF file.
 *
 * @property noAlloc When `true`, tensor metadata is loaded but no tensor data
 *   is read into memory.
 * @property loadData When `true` **and** [noAlloc] is `false`, the parser reads
 *   the binary tensor blob and sets [GGUFContextInternal.data].
 */
data class GGUFInitParams(
    val noAlloc: Boolean = false,
    val loadData: Boolean = false
)

// ---------------------------------------------------------------------------
// Legacy data classes (kept for backward compatibility with existing code)
// ---------------------------------------------------------------------------

/**
 * GGUF key-value pair data structure (simple / legacy representation).
 */
data class GGUFKeyValue(
    val key: String,
    val type: GGUFType,
    val value: Any
)

/**
 * GGUF tensor information (simple / legacy representation).
 */
data class GGUFTensorInfo(
    val name: String,
    val dimensions: IntArray,
    val type: GGMLType,
    val offset: Long
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as GGUFTensorInfo

        if (name != other.name) return false
        if (!dimensions.contentEquals(other.dimensions)) return false
        if (type != other.type) return false
        if (offset != other.offset) return false

        return true
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + dimensions.contentHashCode()
        result = 31 * result + type.hashCode()
        result = 31 * result + offset.hashCode()
        return result
    }
}
