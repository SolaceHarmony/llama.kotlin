package ai.solace.llamakotlin.gguf

/**
 * GGUF data types enum matching the C++ enum gguf_type
 */
enum class GGUFType(val value: Int) {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    FLOAT32(6),
    BOOL(7),
    STRING(8),
    ARRAY(9),
    UINT64(10),
    INT64(11),
    FLOAT64(12);

    companion object {
        fun fromValue(value: Int): GGUFType {
            return entries.find { it.value == value }
                ?: throw IllegalArgumentException("Unknown GGUF type: $value")
        }
    }
}

/**
 * GGUF key-value pair data structure
 */
data class GGUFKeyValue(
    val key: String,
    val type: GGUFType,
    val value: Any
)

/**
 * GGUF tensor information
 */
data class GGUFTensorInfo(
    val name: String,
    val dimensions: IntArray,
    val type: ai.solace.llamakotlin.core.GGMLType,
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

/**
 * GGUF header constants
 */
object GGUFConstants {
    const val MAGIC = "GGUF"
    const val DEFAULT_ALIGNMENT = 32L
}