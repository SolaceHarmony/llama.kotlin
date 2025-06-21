package ai.solace.llamakotlin.core

import kotlin.native.concurrent.SharedImmutable
import kotlin.Short.Companion.SIZE_BYTES

// Numeric conversion functions (assuming they are in the same package or imported)
// import ai.solace.llamakotlin.core.halfToFloat // Not needed if in same file/package
// import ai.solace.llamakotlin.core.floatToHalf // Not needed if in same file/package

// Helper functions for Little Endian byte conversions
internal fun ByteArray.getIntLe(offset: Int): Int {
    if (offset + 3 >= size) throw IndexOutOfBoundsException("Not enough bytes to read an Int at offset $offset")
    return (this[offset].toInt() and 0xFF) or
            ((this[offset + 1].toInt() and 0xFF) shl 8) or
            ((this[offset + 2].toInt() and 0xFF) shl 16) or
            ((this[offset + 3].toInt() and 0xFF) shl 24)
}

internal fun ByteArray.getFloatLe(offset: Int): Float = Float.fromBits(this.getIntLe(offset))

internal fun ByteArray.setIntLe(offset: Int, value: Int) {
    if (offset + 3 >= size) throw IndexOutOfBoundsException("Not enough bytes to write an Int at offset $offset")
    this[offset] = (value and 0xFF).toByte()
    this[offset + 1] = ((value shr 8) and 0xFF).toByte()
    this[offset + 2] = ((value shr 16) and 0xFF).toByte()
    this[offset + 3] = ((value shr 24) and 0xFF).toByte()
}

internal fun ByteArray.setFloatLe(offset: Int, value: Float) = this.setIntLe(offset, value.toRawBits())

internal fun ByteArray.getShortLe(offset: Int): Short {
    if (offset + 1 >= size) throw IndexOutOfBoundsException("Not enough bytes to read a Short at offset $offset")
    return ((this[offset].toInt() and 0xFF) or
            ((this[offset + 1].toInt() and 0xFF) shl 8)).toShort()
}

internal fun ByteArray.setShortLe(offset: Int, value: Short) {
    if (offset + 1 >= size) throw IndexOutOfBoundsException("Not enough bytes to write a Short at offset $offset")
    this[offset] = (value.toInt() and 0xFF).toByte()
    this[offset + 1] = ((value.toInt() shr 8) and 0xFF).toByte()
}

internal fun ByteArray.getLongLe(offset: Int): Long {
    require(offset + Long.SIZE_BYTES <= size) { "Offset $offset + ${Long.SIZE_BYTES} > size $size" }
    var result = 0L
    for (i in 0 until Long.SIZE_BYTES) {
        result = result or ((this[offset + i].toLong() and 0xFF) shl (i * 8))
    }
    return result
}

internal fun ByteArray.setLongLe(offset: Int, value: Long) {
    require(offset + Long.SIZE_BYTES <= size) { "Offset $offset + ${Long.SIZE_BYTES} > size $size" }
    for (i in 0 until Long.SIZE_BYTES) {
        this[offset + i] = ((value shr (i * 8)) and 0xFF).toByte()
    }
}


/**
 * Kotlin Native port of GGML tensor library core data types.
 * This file contains the core data structures used in the GGML library.
 */

/**
 * Maximum number of dimensions in a tensor
 */
const val GGML_MAX_DIMS = 4

/**
 * Maximum number of source tensors for an operation
 */
const val GGML_MAX_SRC = 10

/**
 * Maximum number of operation parameters
 */
const val GGML_MAX_OP_PARAMS = 32

/**
 * Maximum name length for a tensor
 */
const val GGML_MAX_NAME = 64
const val GGML_TENSOR_FLAG_OUTPUT = 1 shl 0
internal const val QK8_0: Int = 32 // Block size for Q8_0 blocks
internal const val QK4_0: Int = 32 // Block size for Q4_0 blocks
internal const val QK4_1: Int = 32 // Block size for Q4_1 blocks (same number of elements as Q4_0)
internal const val QK_K: Int = 256 // Block size for K-quants (generic)
internal const val QK4_K: Int = QK_K // Alias for Q4_K block size, using the generic K-quant block size

// Constants for Q4_K type
internal const val SCALES_SIZE_Q4_K: Int = 12 // Bytes for scales and mins (packed)
internal const val QS_SIZE_Q4_K: Int = QK4_K / 2 // Bytes for quantized data (128 bytes)

// Constants for Q1_5_K type
internal const val QK1_5_K: Int = QK_K // Number of elements in a Q1_5_K block, typically same as other K-quants
internal const val BYTES_SCALE_Q1_5_K: Int = 2 // Size of scale (Float16)
internal const val PACKED_DATA_SIZE_Q1_5_K: Int = 52 // ceil(QK1_5_K / 5) = ceil(256 / 5) = 52 bytes for packed data


/**
 * Represents the type of quantization.
 */
enum class QuantizationType {
    NONE,           // No quantization / Float
    LEGACY_Q,       // Legacy quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
    K_QUANTS        // K-Quantization (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)
}

/**
 * Tensor data types
 */
@Suppress("UNUSED_PARAMETER") // For description in enum, if not used elsewhere
enum class GGMLType(
    val description: String,
    val byteSize: ULong,
    val quantizationType: QuantizationType = QuantizationType.NONE,
    val blckSize: Int = 1 // Number of elements in a block, 1 for non-block types
) {
    F32("float32", 4uL, QuantizationType.NONE),    // 32-bit float
    F16("float16", 2uL, QuantizationType.NONE),    // 16-bit float
    Q4_0("q4_0", 2uL + (QK4_0 / 2).toULong(), QuantizationType.LEGACY_Q, QK4_0),
    Q4_1("q4_1", (2uL * Short.SIZE_BYTES.toULong()) + (QK4_1 / 2).toULong(), QuantizationType.LEGACY_Q, QK4_1),
    Q5_0("q5_0", 0uL, QuantizationType.LEGACY_Q), // Placeholder, define actual size and blckSize if used
    Q5_1("q5_1", 0uL, QuantizationType.LEGACY_Q), // Placeholder
    Q8_0("q8_0", 2uL + QK8_0.toULong(), QuantizationType.LEGACY_Q, QK8_0),
    Q8_1("q8_1", 0uL, QuantizationType.LEGACY_Q), // Placeholder
    Q2_K("q2_k", 0uL, QuantizationType.K_QUANTS, QK_K), // Placeholder for byteSize
    Q3_K("q3_k", 0uL, QuantizationType.K_QUANTS, QK_K), // Placeholder for byteSize
    Q4_K("q4_k", (2uL + 2uL + SCALES_SIZE_Q4_K.toULong() + QS_SIZE_Q4_K.toULong()), QuantizationType.K_QUANTS, QK4_K),
    Q5_K("q5_k", 0uL, QuantizationType.K_QUANTS, QK_K), // Placeholder for byteSize
    Q6_K("q6_k", 0uL, QuantizationType.K_QUANTS, QK_K), // Placeholder for byteSize
    Q8_K("q8_k", 0uL, QuantizationType.K_QUANTS, QK_K), // Placeholder for byteSize (usually d + qs (QK_K))
    Q1_5_K("q1_5_k", (BYTES_SCALE_Q1_5_K + PACKED_DATA_SIZE_Q1_5_K).toULong(), QuantizationType.K_QUANTS, QK1_5_K),
    I8("int8", 1uL, QuantizationType.NONE),     // 8-bit integer
    I16("int16", 2uL, QuantizationType.NONE),    // 16-bit integer
    I32("int32", 4uL, QuantizationType.NONE),    // 32-bit integer
    I64("int64", 8uL, QuantizationType.NONE),    // 64-bit integer
    COUNT("count", 0uL, QuantizationType.NONE)   // Number of types (not a real data type)
}

/**
 * Data class representing a Q4_K quantization block.
 * This structure mirrors the C `block_q4_K` but uses Kotlin types.
 * Note: `d` and `dmin` are stored as Float, converted from FP16.
 * `scales` array is 12 bytes.
 * `qs` array is 128 bytes (256 nibbles).
 */
data class BlockQ4K(
    val d: Float,          // super-block scale
    val dmin: Float,       // super-block min
    val scales: ByteArray, // 12 bytes for sub-block scales/mins
    val qs: ByteArray      // 128 bytes for 256 4-bit quants
) {
    init {
        require(scales.size == SCALES_SIZE_Q4_K) { "Scales ByteArray must be of size $SCALES_SIZE_Q4_K" }
        require(qs.size == QS_SIZE_Q4_K) { "QS ByteArray must be of size $QS_SIZE_Q4_K" }
    }

    companion object {
        /**
         * Converts a raw ByteArray (representing a Q4_K block from memory) to a BlockQ4K object.
         * The input ByteArray is assumed to be exactly the size of a Q4_K block (144 bytes).
         */
        fun fromByteArray(blockBytes: ByteArray): BlockQ4K {
            require(blockBytes.size == GGMLType.Q4_K.byteSize.toInt()) {
                "Input ByteArray size must be ${GGMLType.Q4_K.byteSize} for a Q4_K block."
            }
            val d = halfToFloat(blockBytes.getShortLe(0))
            val dmin = halfToFloat(blockBytes.getShortLe(Short.SIZE_BYTES))
            val scales = blockBytes.copyOfRange(2 * Short.SIZE_BYTES, 2 * Short.SIZE_BYTES + SCALES_SIZE_Q4_K)
            val qs = blockBytes.copyOfRange(2 * Short.SIZE_BYTES + SCALES_SIZE_Q4_K, 2 * Short.SIZE_BYTES + SCALES_SIZE_Q4_K + QS_SIZE_Q4_K)
            return BlockQ4K(d, dmin, scales, qs)
        }
    }

    /**
     * Converts this BlockQ4K object to a raw ByteArray.
     */
    fun toByteArray(): ByteArray {
        val byteArray = ByteArray(GGMLType.Q4_K.byteSize.toInt())
        byteArray.setShortLe(0, floatToHalf(d))
        byteArray.setShortLe(Short.SIZE_BYTES, floatToHalf(dmin))
        scales.copyInto(byteArray, destinationOffset = 2 * Short.SIZE_BYTES)
        qs.copyInto(byteArray, destinationOffset = 2 * Short.SIZE_BYTES + SCALES_SIZE_Q4_K)
        return byteArray
    }
}

/**
 * Tensor operations
 */
enum class GGMLOp(val canBeInplace: Boolean = false) {
    NONE,
    DUP,
    ADD(true),
    SUB(true),
    MUL(true), // Element-wise multiplication
    DIV(true),
    SQR(true),
    SQRT(true),
    SUM, // Typically not inplace (reduces dimensions)
    MEAN, // Typically not inplace
    REPEAT,
    ABS(true),
    SGN(true),
    NEG(true),
    STEP(true),
    RELU(true),
    GELU(true),
    SILU(true),
    NORM(true), // LayerNorm, can be inplace if shapes match and specific handling
    RMS_NORM(true),
    MUL_MAT, // Matrix multiplication, typically not inplace
    SCALE(true),
    CPY, // Copy, not inplace by definition of creating a new tensor with copied data
    RESHAPE, // Reshape is a view, metadata change, not inplace on data buffer in the same way
    VIEW,    // View is a metadata change
    PERMUTE,
    TRANSPOSE,
    GET_ROWS,
    DIAG_MASK_INF(true),
    SOFT_MAX(true), // Can be made inplace
    ROPE(true),
    CONV_1D_1S,
    CONV_1D_2S,
    FLASH_ATTN,
    FLASH_FF,
    MAP_UNARY, // Depends on the specific unary op mapped
    MAP_BINARY, // Depends on the specific binary op mapped
    COUNT
}

/**
 * Computation graph evaluation order
 */
enum class GGMLCGraphEvalOrder {
    NONE,
    FORWARD,
    BACKWARD
}

/**
 * 16-bit brain floating point type
 */
data class GGMLBF16(val bits: UShort)

/**
 * Base object structure
 */
class GGMLObject(
    var offset: ULong = 0u,
    var size: ULong = 0u,
    var next: GGMLObject? = null
)

/**
 * Tensor data structure
 */
class GGMLTensor(
    var type: GGMLType = GGMLType.F32,
    var buffer: Any? = null,
    var ne: LongArray = LongArray(GGML_MAX_DIMS) { 0L },
    var nb: ULongArray = ULongArray(GGML_MAX_DIMS) { 0u },
    var op: GGMLOp = GGMLOp.NONE,
    var opParams: IntArray = IntArray(GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { 0 },
    var flags: Int = 0,
    var grad: GGMLTensor? = null,
    var src: Array<GGMLTensor?> = Array(GGML_MAX_SRC) { null },
    var viewSrc: GGMLTensor? = null,
    var viewOffs: ULong = 0u,
    var data: Any? = null,
    var name: String = "",
    var bufferId: Int = -1,
    var dataOffset: ULong = 0u
) {
    fun isOutput(): Boolean = (this.flags and GGML_TENSOR_FLAG_OUTPUT) != 0

    /**
     * Calculates the rank of the tensor (number of dimensions > 1).
     * Or 0 for a scalar that might have ne=[1,1,1,1] or ne=[].
     * Or 1 for a vector that might be ne=[N,1,1,1].
     */
    internal fun rank(): Int {
        if (ne.all { it <= 1L }) { // Covers scalars like [1,1,1,1] and true 0-rank like [] if ne can be empty
            return if (ne.any { it > 0L}) 1 else 0 // Treat ne=[1,1,1,1] as rank 1 for element calculation if needed
        }
        return ne.indexOfLast { it > 1L } + 1
    }

    /**
     * Calculates the total number of elements in the tensor.
     * For block-quantized types, this is the total number of fundamental elements (e.g., individual weights).
     */
    fun numElements(): Long {
        if (ne.isEmpty()) return 0L
        var count = 1L
        // Only multiply dimensions that are part of the tensor's actual rank
        // or all dimensions if it's a scalar represented by [1,1,1,1]
        val r = rank()
        if (r == 0 && ne.all { it <= 1L}) return 1L // Scalar, effectively 1 element
        if (r == 0 && ne.any { it == 0L}) return 0L // Not a valid tensor shape for elements typically

        for (i in 0 until r.coerceAtLeast(1)) { // Iterate at least once for scalars like ne=[N]
             if (ne[i] == 0L && r > 1) return 0L // Invalid dimension in a multi-dim tensor
             if (ne[i] > 0L) count *= ne[i]
        }
        return count
    }

    internal fun isValidZeroSizedTensor(): Boolean {
        // COUNT type is a valid zero-sized tensor (conceptual, no data).
        if (this.type == GGMLType.COUNT) {
            return true
        }
        // If any dimension (ne[i]) for the actual rank of the tensor is 0,
        // then the total number of elements is 0, making it a valid zero-sized tensor.
        // rank() can be 0 for an uninitialized tensor (ne all 0s or 1s but effectively no elements).
        // rank() can be 1 for ne=[N,1,1,1]. Loop from 0 until rank().
        val r = this.rank()
        if (r == 0 && this.ne.all { it <= 0L }) return true // An uninitialized or ne=[] tensor is zero-sized
        if (r == 0 && this.ne.any { it > 0L }) return false // A scalar like ne=[1,1,1,1] is not zero-sized

        for (i in 0 until r) { // Iterate up to actual rank
            if (this.ne[i] == 0L) {
                return true
            }
        }
        // If type.byteSize is 0 for a non-COUNT type, but numElements > 0,
        // it's an issue with type definition, not a valid zero-sized data tensor.
        // That case is handled by warnings elsewhere (e.g. stride calculation).
        // This function focuses on whether the *data itself* is zero-sized due to dimensions.
        return false
    }

    // Helper to calculate byte offset of an element given its indices
    private fun getElementByteOffset(vararg indices: Int): ULong {
        // The nb array in ggml stores the strides directly:
        // nb[0] = stride for dim 0 (e.g. type_size for contiguous)
        // nb[1] = stride for dim 1 (e.g. ne[0]*type_size for contiguous)
        // ...
        // For a tensor t, address of t(i0,i1,i2,i3) = t->data + i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3].
        // This is the interpretation this function will use.

        var finalOffset = 0uL
        // Number of actual dimensions in the tensor (where ne[d] > 1)
        // val rank = ne.count { it > 1L }
        // if (indices.size != rank && !(rank == 0 && indices.isEmpty())) {
        //    throw IllegalArgumentException("Number of indices (${indices.size}) must match tensor rank ($rank). Tensor shape: ${ne.joinToString()}. Indices: ${indices.joinToString()}")
        // }
        // The above rank check might be too strict if ne contains trailing 1s for lower rank tensors.
        // Example: A 2D tensor might have ne = [10, 20, 1, 1]. Rank is 2. indices.size should be 2.

        // Iterating up to indices.size assumes that the provided indices match the intended dimensions.
        for (d in indices.indices) {
            if (d >= GGML_MAX_DIMS) { // Should not happen if indices.size is checked against rank based on ne
                throw IllegalArgumentException("Dimension index $d exceeds GGML_MAX_DIMS.")
            }
            if (indices[d] < 0 || indices[d] >= ne[d]) {
                val shapeString = ne.joinToString(limit = GGML_MAX_DIMS)
                throw IllegalArgumentException("Index ${indices[d]} for dimension $d is out of bounds (0 to ${ne[d] - 1}) for tensor shape [$shapeString]")
            }
            finalOffset += indices[d].toULong() * nb[d]
        }
        return finalOffset
    }

    // Accessor methods for F32
    fun getFloat(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Float {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for F32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for F32 is out of bounds for buffer size ${buffer.size}")
        }
        return buffer.getFloatLe(finalByteOffset.toInt())
    }

    fun setFloat(graphAllocator: GGMLGraphAllocator, value: Float, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for F32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for F32 is out of bounds for buffer size ${buffer.size}")
        }
        buffer.setFloatLe(finalByteOffset.toInt(), value)
    }

    // Accessor methods for I32
    fun getInt(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Int {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for I32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for I32 is out of bounds for buffer size ${buffer.size}")
        }
        return buffer.getIntLe(finalByteOffset.toInt())
    }

    fun setInt(graphAllocator: GGMLGraphAllocator, value: Int, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for I32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for I32 is out of bounds for buffer size ${buffer.size}")
        }
        buffer.setIntLe(finalByteOffset.toInt(), value)
    }

    // Accessor methods for I16
    fun getShort(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Short {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 2u > buffer.size.toUInt()) { // Check for I16 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 2 bytes for I16 is out of bounds for buffer size ${buffer.size}")
        }
        return buffer.getShortLe(finalByteOffset.toInt())
    }

    fun setShort(graphAllocator: GGMLGraphAllocator, value: Short, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 2u > buffer.size.toUInt()) { // Check for I16 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 2 bytes for I16 is out of bounds for buffer size ${buffer.size}")
        }
        buffer.setShortLe(finalByteOffset.toInt(), value)
    }

    // Accessor methods for F16 (Half Float)
    fun getHalf(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Float {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        // Check bounds, considering Short.SIZE_BYTES (2 bytes for F16)
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Short at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - SIZE_BYTES})")
        }
        val shortBits = buffer.getShortLe(finalByteOffset.toInt())
        return halfToFloat(shortBits) // halfToFloat is in NumericConversions.kt, assumed imported or accessible
    }

    fun setHalf(graphAllocator: GGMLGraphAllocator, value: Float, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        // Check bounds, considering Short.SIZE_BYTES (2 bytes for F16)
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Short at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - SIZE_BYTES})")
        }
        val shortBits = floatToHalf(value) // floatToHalf is in NumericConversions.kt, assumed imported or accessible
        buffer.setShortLe(finalByteOffset.toInt(), shortBits)
    }

    // Accessor methods for I8 (Byte)
    fun getByte(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Byte {
        // require(type == GGMLType.I8) { "getByte() called on non-I8 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Byte at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}) for tensor $name")
        }
        return buffer[finalByteOffset.toInt()]
    }

    fun setByte(graphAllocator: GGMLGraphAllocator, value: Byte, vararg indices: Int) {
        // require(type == GGMLType.I8) { "setByte() called on non-I8 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Byte at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}) for tensor $name")
        }
        buffer[finalByteOffset.toInt()] = value
    }

    // Accessor methods for I64 (Long)
    fun getLong(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Long {
        // require(type == GGMLType.I64) { "getLong() called on non-I64 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Long.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Long at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Long.SIZE_BYTES}) for tensor $name")
        }
        return buffer.getLongLe(finalByteOffset.toInt()) // Uses new ByteArray extension
    }

    fun setLong(graphAllocator: GGMLGraphAllocator, value: Long, vararg indices: Int) {
        // require(type == GGMLType.I64) { "setLong() called on non-I64 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Long.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Long at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Long.SIZE_BYTES}) for tensor $name")
        }
        buffer.setLongLe(finalByteOffset.toInt(), value) // Uses new ByteArray extension
    }

    // --- Q8_0 Accessors ---

    /**
     * For Q8_0 and similar block-quantized types, calculates the number of blocks.
     * Assumes ne holds the number of fundamental elements (e.g., individual weights).
     */
    fun getNumBlocks(): Long {
        val totalElements = numElements()
        if (totalElements == 0L) return 0L

        val elementsPerBlock = when (type) {
            GGMLType.Q8_0 -> QK8_0.toLong()
            GGMLType.Q4_0 -> QK4_0.toLong()
            GGMLType.Q4_1 -> QK4_1.toLong()
            GGMLType.Q4_K -> QK4_K.toLong()
            // Future block types can be added here
            else -> {
                // Or throw IllegalArgumentException("getNumBlocks is only for block-quantized types")
                return 0L
            }
        }
        if (elementsPerBlock == 0L) return 0L // Avoid division by zero

        // Ensure that total elements are a multiple of block size, as per ggml constraints.
        // If not, it indicates an issue with tensor setup or understanding of its true dimensions.
        if (totalElements % elementsPerBlock != 0L) {
            // This is usually an error in ggml, as tensors are expected to be whole blocks.
            // However, some implementations might pad. For strictness, one might throw here.
            // For now, simple integer division, implying full blocks.
            println("Warning: Tensor ${name} of type ${type} has total elements $totalElements which is not perfectly divisible by block size $elementsPerBlock.")
        }
        return totalElements / elementsPerBlock
    }

    /**
     * Retrieves the F16 scale for a specific block in a Q8_0 quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @return The scale value as a Float.
     */
    fun getQ8_0BlockScale(graphAllocator: GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == GGMLType.Q8_0) { "Tensor type must be Q8_0 to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        // type.byteSize for Q8_0 is the size of one block (e.g., 34 bytes)
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated for tensor '$name'.")

        // The scale is the first F16 (2 bytes) in the block
        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + SHORT_SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q8_0 scale at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - SHORT_SIZE_BYTES}). DataOffset: $dataOffset, BlockByteOffset: $blockByteOffset.")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return halfToFloat(scaleBits)
    }

    /**
     * Retrieves a single quantized weight (Int8/Byte) from a specific block in a Q8_0 tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK8_0 - 1).
     * @return The quantized weight as a Byte.
     */
    fun getQ8_0Weight(graphAllocator: GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == GGMLType.Q8_0) { "Tensor type must be Q8_0 to get weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < QK8_0) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${QK8_0 -1}) for Q8_0 block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize // type.byteSize is block size for Q8_0
        val qsArrayBaseOffsetInBlock = 2uL // The F16 scale takes the first 2 bytes of the block
        val finalWeightByteOffset = dataOffset + blockByteOffset + qsArrayBaseOffsetInBlock + itemIndexInBlock.toULong()

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated for tensor '$name'.")

        if (finalWeightByteOffset.toInt() < 0 || finalWeightByteOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to read Q8_0 weight at offset ${finalWeightByteOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}). DataOffset: $dataOffset, BlockByteOffset: $blockByteOffset, ItemOffset: $qsArrayBaseOffsetInBlock + $itemIndexInBlock.")
        }
        return buffer[finalWeightByteOffset.toInt()]
    }

    // --- Q4_0 Accessors ---

    /**
     * Retrieves the F16 scale for a specific block in a Q4_0 quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @return The scale value as a Float.
     */
    fun getQ4_0BlockScale(graphAllocator: GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == GGMLType.Q4_0) { "Tensor type must be Q4_0 to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        // type.byteSize for Q4_0 is the size of one block (e.g., 18 bytes)
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        // The scale is the first F16 (2 bytes) in the block
        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + SHORT_SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_0 scale at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - SHORT_SIZE_BYTES}). DataOffset: $dataOffset, BlockByteOffset: $blockByteOffset.")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return halfToFloat(scaleBits)
    }

    /**
     * Retrieves a single 4-bit quantized weight (nibble) from a specific block in a Q4_0 tensor.
     * The returned Byte contains the raw 4-bit value (0-15).
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK4_0 - 1).
     * @return The quantized 4-bit weight as a Byte (value 0-15).
     */
    fun getQ4_0NibbleWeight(graphAllocator: GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == GGMLType.Q4_0) { "Tensor type must be Q4_0 to get nibble weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < QK4_0) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${QK4_0 -1}) for Q4_0 block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val qsArrayBaseOffsetInBlock = 2uL // The F16 scale takes the first 2 bytes
        val byteContainingNibbleIndex = itemIndexInBlock / 2 // Each byte stores two 4-bit nibbles

        val finalByteToReadOffset = dataOffset + blockByteOffset + qsArrayBaseOffsetInBlock + byteContainingNibbleIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToReadOffset.toInt() < 0 || finalByteToReadOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to read Q4_0 nibble weight byte at offset ${finalByteToReadOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}).")
        }
        val packedByte = buffer[finalByteToReadOffset.toInt()]

        val nibble = if (itemIndexInBlock % 2 == 0) {
            packedByte.toInt() and 0x0F // First item in the byte (lower 4 bits)
        } else {
            (packedByte.toInt() ushr 4) and 0x0F // Second item in the byte (upper 4 bits)
        }
        return nibble.toByte()
    }

    // --- Q4_1 Accessors ---

    /**
     * Retrieves the F16 scale ('d') for a specific block in a Q4_1 quantized tensor.
     */
    fun getQ4_1BlockScale(graphAllocator: GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == GGMLType.Q4_1) { "Tensor type must be Q4_1 to get block scale 'd'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize // type.byteSize for Q4_1 is 20 bytes
        val finalScaleByteOffset = dataOffset + blockByteOffset // Scale 'd' is the first F16

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_1 scale 'd' at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - SIZE_BYTES}).")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return halfToFloat(scaleBits)
    }

    /**
     * Retrieves the F16 min value ('m') for a specific block in a Q4_1 quantized tensor.
     */
    fun getQ4_1BlockMin(graphAllocator: GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == GGMLType.Q4_1) { "Tensor type must be Q4_1 to get block min 'm'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val minOffsetWithinBlock = SIZE_BYTES.toULong() // Min 'm' is the second F16 (after the scale 'd')
        val finalMinByteOffset = dataOffset + blockByteOffset + minOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalMinByteOffset.toInt() < 0 || finalMinByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_1 min 'm' at offset ${finalMinByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - SIZE_BYTES}).")
        }

        val minBits = buffer.getShortLe(finalMinByteOffset.toInt())
        return halfToFloat(minBits)
    }

    /**
     * Retrieves a single 4-bit quantized weight (nibble) from a specific block in a Q4_1 tensor.
     * The returned Byte contains the raw 4-bit value (0-15).
     */
    fun getQ4_1NibbleWeight(graphAllocator: GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == GGMLType.Q4_1) { "Tensor type must be Q4_1 to get nibble weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < QK4_1) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${QK4_1 -1}) for Q4_1 block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val qsBaseOffsetWithinBlock = (2 * SIZE_BYTES).toULong() // Weights start after two F16s (d and m)
        val byteContainingNibbleIndex = itemIndexInBlock / 2

        val finalByteToReadOffset = dataOffset + blockByteOffset + qsBaseOffsetWithinBlock + byteContainingNibbleIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToReadOffset.toInt() < 0 || finalByteToReadOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to read Q4_1 nibble weight byte at offset ${finalByteToReadOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds.")
        }
        val packedByte = buffer[finalByteToReadOffset.toInt()]

        val nibble = if (itemIndexInBlock % 2 == 0) {
            packedByte.toInt() and 0x0F // First item (lower 4 bits)
        } else {
            (packedByte.toInt() ushr 4) and 0x0F // Second item (upper 4 bits)
        }
        return nibble.toByte()
    }

    // --- Q4_K Accessors ---

    /**
     * Retrieves the overall F16 scale ('d') for a specific block in a Q4_K quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @return The overall scale value as a Float.
     */
    fun getQ4KOverallScale(graphAllocator: GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K to get overall scale 'd'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        // type.byteSize for Q4_K is the size of one block (144 bytes)
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset // Overall scale 'd' is the first F16

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_K overall scale 'd' at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}).")
        }
        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return halfToFloat(scaleBits)
    }

    /**
     * Sets the overall F16 scale ('d') for a specific block in a Q4_K quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param scale The overall scale value as a Float.
     */
    fun setQ4KOverallScale(graphAllocator: GGMLGraphAllocator, blockIndex: Int, scale: Float) {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K to set overall scale 'd'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Q4_K overall scale 'd' at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}).")
        }
        val scaleBits = floatToHalf(scale)
        buffer.setShortLe(finalScaleByteOffset.toInt(), scaleBits)
    }

    /**
     * Retrieves the overall F16 min value ('dmin') for a specific block in a Q4_K quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @return The overall min value as a Float.
     */
    fun getQ4KOverallMin(graphAllocator: GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K to get overall min 'dmin'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val minOffsetWithinBlock = Short.SIZE_BYTES.toULong() // Overall min 'dmin' is the second F16
        val finalMinByteOffset = dataOffset + blockByteOffset + minOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalMinByteOffset.toInt() < 0 || finalMinByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_K overall min 'dmin' at offset ${finalMinByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}).")
        }
        val minBits = buffer.getShortLe(finalMinByteOffset.toInt())
        return halfToFloat(minBits)
    }

    /**
     * Sets the overall F16 min value ('dmin') for a specific block in a Q4_K quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param min The overall min value as a Float.
     */
    fun setQ4KOverallMin(graphAllocator: GGMLGraphAllocator, blockIndex: Int, min: Float) {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K to set overall min 'dmin'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val minOffsetWithinBlock = Short.SIZE_BYTES.toULong()
        val finalMinByteOffset = dataOffset + blockByteOffset + minOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalMinByteOffset.toInt() < 0 || finalMinByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Q4_K overall min 'dmin' at offset ${finalMinByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}).")
        }
        val minBits = floatToHalf(min)
        buffer.setShortLe(finalMinByteOffset.toInt(), minBits)
    }

    // Helper to get the base offset of the 12-byte scales/mins array within a Q4_K block
    private fun getQ4KScalesArrayBaseOffset(blockIndex: Int): ULong {
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // d and dmin take 2 * Short.SIZE_BYTES
        return dataOffset + blockByteOffset + (2 * Short.SIZE_BYTES).toULong()
    }

    /**
     * Unpacks and returns the 6-bit quantized scale for a sub-block.
     * This returns the raw 6-bit value.
     */
    private fun getQ4KSubBlockScaleQuantized(graphAllocator: GGMLGraphAllocator, blockIndex: Int, subBlockIdx: Int): Byte {
        require(subBlockIdx in 0..7) { "subBlockIdx $subBlockIdx must be between 0 and 7." }

        val scalesBaseOffset = getQ4KScalesArrayBaseOffset(blockIndex)
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Buffer not found for $name")

        val scaleQuant: Int
        if (subBlockIdx < 4) { // Sub-blocks 0-3
            // ls_j is in lower 6 bits of scales_byte_array[j]
            val byteOffset = scalesBaseOffset + subBlockIdx.toULong()
            if (byteOffset.toInt() < 0 || byteOffset.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $byteOffset out of bounds for buffer size ${buffer.size}")
            scaleQuant = buffer[byteOffset.toInt()].toInt() and 0x3F
        } else { // Sub-blocks 4-7
            // ls_{j+4} is split: lower 4 bits in scales_byte_array[j+8], upper 2 bits in scales_byte_array[j]
            val j = subBlockIdx - 4
            val byteOffset1 = scalesBaseOffset + j.toULong() // scales_byte_array[j]
            val byteOffset2 = scalesBaseOffset + (j + 8).toULong() // scales_byte_array[j+8]

            if (byteOffset1.toInt() < 0 || byteOffset1.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $byteOffset1 out of bounds for buffer size ${buffer.size}")
            if (byteOffset2.toInt() < 0 || byteOffset2.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $byteOffset2 out of bounds for buffer size ${buffer.size}")

            val upper2bits = (buffer[byteOffset1.toInt()].toInt() ushr 6) and 0x03
            val lower4bits = buffer[byteOffset2.toInt()].toInt() and 0x0F
            scaleQuant = (upper2bits shl 4) or lower4bits
        }
        return scaleQuant.toByte()
    }

     /**
     * Unpacks and returns the 6-bit quantized min for a sub-block.
     * This returns the raw 6-bit value.
     */
    private fun getQ4KSubBlockMinQuantized(graphAllocator: GGMLGraphAllocator, blockIndex: Int, subBlockIdx: Int): Byte {
        require(subBlockIdx in 0..7) { "subBlockIdx $subBlockIdx must be between 0 and 7." }

        val scalesBaseOffset = getQ4KScalesArrayBaseOffset(blockIndex)
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Buffer not found for $name")

        val minQuant: Int
        if (subBlockIdx < 4) { // Sub-blocks 0-3
            // lm_j is in lower 6 bits of scales_byte_array[j+4]
            val byteOffset = scalesBaseOffset + (subBlockIdx + 4).toULong()
            if (byteOffset.toInt() < 0 || byteOffset.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $byteOffset out of bounds for buffer size ${buffer.size}")
            minQuant = buffer[byteOffset.toInt()].toInt() and 0x3F
        } else { // Sub-blocks 4-7
            // lm_{j+4} is split: lower 4 bits in scales_byte_array[j+8] (upper nibble), upper 2 bits in scales_byte_array[j+4]
            val j = subBlockIdx - 4
            val byteOffset1 = scalesBaseOffset + (j + 4).toULong() // scales_byte_array[j+4]
            val byteOffset2 = scalesBaseOffset + (j + 8).toULong() // scales_byte_array[j+8]

            if (byteOffset1.toInt() < 0 || byteOffset1.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $byteOffset1 out of bounds for buffer size ${buffer.size}")
            if (byteOffset2.toInt() < 0 || byteOffset2.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $byteOffset2 out of bounds for buffer size ${buffer.size}")

            val upper2bits = (buffer[byteOffset1.toInt()].toInt() ushr 6) and 0x03
            val lower4bits = (buffer[byteOffset2.toInt()].toInt() ushr 4) and 0x0F
            minQuant = (upper2bits shl 4) or lower4bits
        }
        return minQuant.toByte()
    }

    /**
     * Retrieves the dequantized scale for a sub-block in a Q4_K tensor.
     * Effective_scale = overall_scale * sub_block_scale_quantized.
     */
    fun getQ4KSubBlockScale(graphAllocator: GGMLGraphAllocator, blockIndex: Int, subBlockIdx: Int): Float {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K." }
        val overallScale = getQ4KOverallScale(graphAllocator, blockIndex)
        val scaleQuant = getQ4KSubBlockScaleQuantized(graphAllocator, blockIndex, subBlockIdx)
        return overallScale * (scaleQuant.toInt() and 0xFF) // scaleQuant is 0-63
    }

    /**
     * Retrieves the dequantized min for a sub-block in a Q4_K tensor.
     * Effective_min = overall_min * sub_block_min_quantized.
     */
    fun getQ4KSubBlockMin(graphAllocator: GGMLGraphAllocator, blockIndex: Int, subBlockIdx: Int): Float {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K." }
        val overallMin = getQ4KOverallMin(graphAllocator, blockIndex)
        val minQuant = getQ4KSubBlockMinQuantized(graphAllocator, blockIndex, subBlockIdx)
        // In llama.cpp, dmin is positive, and the actual min is negative.
        // If overallMin is stored as a positive value (like scales), then effective min is -(overallMin * minQuant)
        // However, the quantization formula is usually (val - min)/scale or val/scale + min.
        // Let's assume dmin is stored as is, and used as `d_eff * q - min_eff`.
        // If lm is 0-63 and dmin is positive, min_eff = dmin * lm.
        return overallMin * (minQuant.toInt() and 0xFF) // minQuant is 0-63
    }

    /**
     * Sets the 6-bit quantized scale and min for a sub-block in a Q4_K tensor.
     * @param scaleQuant Raw 6-bit scale (0-63).
     * @param minQuant Raw 6-bit min (0-63).
     */
    fun setQ4KSubBlockScaleAndMin(graphAllocator: GGMLGraphAllocator, blockIndex: Int, subBlockIdx: Int, scaleQuant: Byte, minQuant: Byte) {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K." }
        require(subBlockIdx in 0..7) { "subBlockIdx $subBlockIdx must be between 0 and 7." }
        require(scaleQuant >= 0 && scaleQuant < 64) { "scaleQuant $scaleQuant out of 6-bit range (0-63)." }
        require(minQuant >= 0 && minQuant < 64) { "minQuant $minQuant out of 6-bit range (0-63)." }

        val scalesBaseOffset = getQ4KScalesArrayBaseOffset(blockIndex)
        val buffer = graphAllocator.buffers[bufferId] ?: throw IllegalStateException("Buffer not found for $name")

        val sQ = scaleQuant.toInt() and 0x3F
        val mQ = minQuant.toInt() and 0x3F

        if (subBlockIdx < 4) { // Sub-blocks 0-3
            // Scale: ls_j is in lower 6 bits of scales_byte_array[j]
            val scaleByteOffset = scalesBaseOffset + subBlockIdx.toULong()
            if (scaleByteOffset.toInt() < 0 || scaleByteOffset.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $scaleByteOffset out of bounds for buffer size ${buffer.size}")
            var currentByte = buffer[scaleByteOffset.toInt()].toInt()
            currentByte = (currentByte and 0xC0) or sQ // Keep upper 2 bits, set lower 6
            buffer[scaleByteOffset.toInt()] = currentByte.toByte()

            // Min: lm_j is in lower 6 bits of scales_byte_array[j+4]
            val minByteOffset = scalesBaseOffset + (subBlockIdx + 4).toULong()
            if (minByteOffset.toInt() < 0 || minByteOffset.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $minByteOffset out of bounds for buffer size ${buffer.size}")
            currentByte = buffer[minByteOffset.toInt()].toInt()
            currentByte = (currentByte and 0xC0) or mQ // Keep upper 2 bits, set lower 6
            buffer[minByteOffset.toInt()] = currentByte.toByte()
        } else { // Sub-blocks 4-7
            val j = subBlockIdx - 4
            // Scale: ls_{j+4} is split. Lower 4 bits in scales_byte_array[j+8], upper 2 bits in scales_byte_array[j]
            val sUpper2 = (sQ ushr 4) and 0x03 // sQ_3, sQ_2
            val sLower4 = sQ and 0x0F          // sQ_1, sQ_0 (conventionally, this is how it's split in code)

            // Update upper 2 bits of scale in scales_byte_array[j]
            val scaleByteOffset1 = scalesBaseOffset + j.toULong()
            if (scaleByteOffset1.toInt() < 0 || scaleByteOffset1.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $scaleByteOffset1 out of bounds for buffer size ${buffer.size}")
            var currentByte1 = buffer[scaleByteOffset1.toInt()].toInt()
            currentByte1 = (currentByte1 and 0x3F) or (sUpper2 shl 6) // Keep lower 6 bits, set upper 2
            buffer[scaleByteOffset1.toInt()] = currentByte1.toByte()

            // Update lower 4 bits of scale in scales_byte_array[j+8]
            val commonByteOffset = scalesBaseOffset + (j + 8).toULong()
            if (commonByteOffset.toInt() < 0 || commonByteOffset.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $commonByteOffset out of bounds for buffer size ${buffer.size}")
            var commonByte = buffer[commonByteOffset.toInt()].toInt()
            commonByte = (commonByte and 0xF0) or sLower4 // Keep upper 4 bits, set lower 4
            // This commonByte will be updated again by min, so write it back after min part.

            // Min: lm_{j+4} is split. Lower 4 bits in scales_byte_array[j+8] (upper nibble), upper 2 bits in scales_byte_array[j+4]
            val mUpper2 = (mQ ushr 4) and 0x03
            val mLower4 = mQ and 0x0F

            // Update upper 2 bits of min in scales_byte_array[j+4]
            val minByteOffset1 = scalesBaseOffset + (j + 4).toULong()
            if (minByteOffset1.toInt() < 0 || minByteOffset1.toInt() >= buffer.size) throw IndexOutOfBoundsException("Offset $minByteOffset1 out of bounds for buffer size ${buffer.size}")
            var currentByte2 = buffer[minByteOffset1.toInt()].toInt()
            currentByte2 = (currentByte2 and 0x3F) or (mUpper2 shl 6) // Keep lower 6 bits, set upper 2
            buffer[minByteOffset1.toInt()] = currentByte2.toByte()

            // Update lower 4 bits of min in scales_byte_array[j+8] (upper nibble)
            commonByte = (commonByte and 0x0F) or (mLower4 shl 4) // Keep lower 4 bits (from scale), set upper 4 for min
            buffer[commonByteOffset.toInt()] = commonByte.toByte()
        }
    }


    /**
     * Retrieves a single 4-bit quantized weight (nibble) from a specific block in a Q4_K tensor.
     * The returned Byte contains the raw 4-bit value (0-15).
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK4_K - 1, i.e., 0-255).
     * @return The quantized 4-bit weight as a Byte (value 0-15).
     */
    fun getQ4KNibble(graphAllocator: GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K to get nibble weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < QK4_K) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${QK4_K -1}) for Q4_K block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // qs array starts after: d (F16), dmin (F16), scales array (12 bytes)
        val qsArrayBaseOffsetInBlock = (2 * Short.SIZE_BYTES + SCALES_SIZE_Q4_K).toULong()
        val byteContainingNibbleIndex = itemIndexInBlock / 2 // Each byte stores two 4-bit nibbles

        val finalByteToReadOffset = dataOffset + blockByteOffset + qsArrayBaseOffsetInBlock + byteContainingNibbleIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToReadOffset.toInt() < 0 || finalByteToReadOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to read Q4_K nibble weight byte at offset ${finalByteToReadOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds.")
        }
        val packedByte = buffer[finalByteToReadOffset.toInt()]

        val nibble = if (itemIndexInBlock % 2 == 0) {
            packedByte.toInt() and 0x0F // First item in the byte (lower 4 bits)
        } else {
            (packedByte.toInt() ushr 4) and 0x0F // Second item in the byte (upper 4 bits)
        }
        return nibble.toByte()
    }

    /**
     * Sets a single 4-bit quantized weight (nibble) in a specific block in a Q4_K tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK4_K - 1).
     * @param nibble The 4-bit weight to set (value 0-15).
     */
    fun setQ4KNibble(graphAllocator: GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int, nibble: Byte) {
        require(type == GGMLType.Q4_K) { "Tensor type must be Q4_K to set nibble weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < QK4_K) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${QK4_K -1}) for Q4_K block in tensor '$name'."}
        require(nibble >= 0 && nibble < 16) { "Nibble value $nibble must be between 0 and 15." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val qsArrayBaseOffsetInBlock = (2 * Short.SIZE_BYTES + SCALES_SIZE_Q4_K).toULong()
        val byteContainingNibbleIndex = itemIndexInBlock / 2

        val finalByteToModifyOffset = dataOffset + blockByteOffset + qsArrayBaseOffsetInBlock + byteContainingNibbleIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToModifyOffset.toInt() < 0 || finalByteToModifyOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to write Q4_K nibble weight byte at offset ${finalByteToModifyOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds.")
        }

        var currentByte = buffer[finalByteToModifyOffset.toInt()].toInt()
        if (itemIndexInBlock % 2 == 0) { // First item in the byte (lower 4 bits)
            currentByte = (currentByte and 0xF0) or (nibble.toInt() and 0x0F)
        } else { // Second item in the byte (upper 4 bits)
            currentByte = (currentByte and 0x0F) or ((nibble.toInt() and 0x0F) shl 4)
        }
        buffer[finalByteToModifyOffset.toInt()] = currentByte.toByte()
    }

    /**
     * Retrieves a raw byte array representing one block of data for this tensor.
     * This is primarily useful for block-quantized types.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @return A ByteArray containing the raw bytes of the specified block.
     */
    fun getBlockByteArray(graphAllocator: GGMLGraphAllocator, blockIndex: Int): ByteArray {
        require(type.blckSize > 1) { "getBlockByteArray is intended for block-quantized types. Tensor '$name' is type ${type.description} with blckSize ${type.blckSize}." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockSizeInBytes = type.byteSize.toInt() // For block types, byteSize is per block.
        if (blockSizeInBytes <= 0) {
            throw IllegalStateException("Cannot get block bytes for type ${type.description} as its type.byteSize is not positive ($blockSizeInBytes).")
        }

        val blockByteOffsetInTensorData = blockIndex.toULong() * type.byteSize
        val finalBlockReadOffset = dataOffset + blockByteOffsetInTensorData

        val buffer = graphAllocator.buffers[bufferId]
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated for tensor '$name'.")

        if (finalBlockReadOffset.toInt() < 0 || finalBlockReadOffset.toInt() + blockSizeInBytes > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read block $blockIndex at offset ${finalBlockReadOffset.toInt()} (size $blockSizeInBytes bytes) for tensor '$name' is out of buffer bounds (0-${buffer.size}). DataOffset: $dataOffset, BlockByteOffsetInTensor: $blockByteOffsetInTensorData.")
        }

        return buffer.copyOfRange(finalBlockReadOffset.toInt(), finalBlockReadOffset.toInt() + blockSizeInBytes)
    }
}

/**
 * Scratch buffer for temporary storage
 */
class GGMLScratch(
    var offs: ULong = 0u,
    var size: ULong = 0u,
    var data: Any? = null
)

/**
 * Context for GGML operations
 */
class GGMLContext(
    var memSize: ULong = 0u,
    var memBuffer: Any? = null,
    var memBufferOwned: Boolean = false,
    var noAlloc: Boolean = false,
    var noAllocSave: Boolean = false,
    var nObjects: Int = 0,
    var objectsBegin: GGMLObject? = null,
    var objectsEnd: GGMLObject? = null,
    var scratch: GGMLScratch = GGMLScratch(),
    var scratchSave: GGMLScratch = GGMLScratch(),
    var computeImmediately: Boolean = true,
    var graphAllocator: GGMLGraphAllocator? = null, // Added for graph execution
    var backend: ai.solace.llamakotlin.backends.Backend? = null // Added for graph execution
)

/**
 * Parameters for initializing the GGML context
 */
class GGMLInitParams(
    var memSize: ULong = 0u,
    var memBuffer: Any? = null,
    var noAlloc: Boolean = false
)

/**
 * Computation graph
 */
class GGMLCGraph(
    var size: Int = 0,
    var nNodes: Int = 0,
    var nLeafs: Int = 0,
    var nodes: Array<GGMLTensor?> = emptyArray(),
    var grads: Array<GGMLTensor?> = emptyArray(),
    var leafs: Array<GGMLTensor?> = emptyArray(),
    var visitedHashSet: Any? = null,
    var order: GGMLCGraphEvalOrder = GGMLCGraphEvalOrder.NONE
)

/**
 * Calculates the byte size of a tensor, considering its type and dimensions.
 * For block-quantized types, this calculates the size based on blocks.
 * For regular types, it's num_elements * type_byte_size.
 * For COUNT type or tensors with 0 elements, it's 0.
 */
internal fun calculateTensorByteSize(tensor: GGMLTensor): ULong {
    val numElements = tensor.numElements().toULong()

    // If a tensor has zero elements (e.g. ne = [0, ...]), its byte size is 0.
    // Also, GGMLType.COUNT is defined with byteSize = 0uL, so it also results in 0.
    if (numElements == 0uL) {
        return 0uL
    }
    // For types like GGMLType.COUNT, type.byteSize is 0, which correctly yields 0.
    if (tensor.type.byteSize == 0uL && tensor.type != GGMLType.COUNT) {
        // This case indicates an issue with a new/custom type definition if it has elements but no byteSize.
        // Standard block types have non-zero byteSize (representing block size).
        println("Warning: Tensor ${tensor.name} of type ${tensor.type} has $numElements elements but type.byteSize is 0. Effective byte size will be 0.")
        return 0uL
    }


    return when (tensor.type) {
        // Explicitly list block-quantized types. Their type.byteSize is "bytes per block".
        // Add Q1_5_K and Q4_K to this list.
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q8_0, GGMLType.Q1_5_K, GGMLType.Q4_K -> {
            // These constants should be defined in GGMLTypes.kt or accessible.
            val elementsPerBlock = when(tensor.type) {
                GGMLType.Q4_0 -> QK4_0.toULong()
                GGMLType.Q4_1 -> QK4_1.toULong()
                GGMLType.Q8_0 -> QK8_0.toULong()
                GGMLType.Q4_K -> QK4_K.toULong()
                GGMLType.Q1_5_K -> QK1_5_K.toULong() // Added Q1_5_K
                // Add other K-quants or block types here if they have a similar structure
                // For example: GGMLType.Q2_K -> QK_K.toULong() (if QK_K is its block size)
                // For Q2_K, Q3_K, Q5_K, Q6_K, Q8_K, their byteSize is currently 0uL.
                // If their byteSize is updated to represent block size, they should be added here.
                // For now, they will fall into the 'else' for byte size calculation based on numElements * type.byteSize.
                else -> {
                    // This path should ideally not be reached if the outer 'when' is exhaustive for block types
                    // with non-zero type.byteSize intended as block size.
                    println("Warning: Unhandled block-quantized type ${tensor.type} in calculateTensorByteSize or type.byteSize is 0. Assuming elementsPerBlock = 1 for safety, but this is likely an error for block types.")
                    1uL
                }
            }

            if (elementsPerBlock == 0uL) { // Should not happen for valid block types
                println("Error: Tensor ${tensor.name} type ${tensor.type} has elementsPerBlock = 0. Byte size calculation invalid.")
                return 0uL
            }

            // GGML requires the number of elements to be a multiple of the block size for quantized types.
            if (numElements % elementsPerBlock != 0uL) {
                 println("Warning: Tensor ${tensor.name} of type ${tensor.type} has $numElements elements, which is not perfectly divisible by block size $elementsPerBlock. Byte size calculation might be incorrect if padding or specific handling is expected.")
                 // Depending on strictness, one might throw an error or adjust numElements to be block-aligned.
                 // For now, proceed with integer division, which implies only full blocks are counted.
            }
            // Calculate size based on full blocks. tensor.type.byteSize for these types is defined as "bytes per block".
            (numElements / elementsPerBlock) * tensor.type.byteSize
        }
        // For non-block types (F32, F16, I32, I16, I8, I64), type.byteSize is the size of one element.
        // For GGMLType.COUNT, byteSize is 0, so this correctly results in 0.
        else -> numElements * tensor.type.byteSize
    }
}
