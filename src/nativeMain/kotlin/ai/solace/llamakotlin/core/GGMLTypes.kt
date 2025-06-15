package ai.solace.llamakotlin.core

import kotlin.native.concurrent.SharedImmutable

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

/**
 * Tensor data types
 */
@Suppress("UNUSED_PARAMETER") // For description in enum, if not used elsewhere
enum class GGMLType(val description: String, val byteSize: ULong) {
    F32("float32", 4uL),    // 32-bit float
    F16("float16", 2uL),    // 16-bit float
    // For quantized types, byteSize here represents the size of the fundamental element IF applicable for simple stride calculations.
    // Actual memory per element for quantized types is fractional and depends on block size.
    // Using 0uL as a placeholder signifies that direct byteSize-based stride calculation isn't straightforward.
    // The ggml library itself has type_size and block_size fields and functions like ggml_type_size() / ggml_blck_size().
    // For now, these are placeholders. The stride logic will primarily rely on non-zero byteSize for unquantized types.
    Q4_0("q4_0", 0uL),   // 4-bit quantized
    Q4_1("q4_1", 0uL),   // 4-bit quantized with different scaling
    Q5_0("q5_0", 0uL),   // 5-bit quantized
    Q5_1("q5_1", 0uL),   // 5-bit quantized with different scaling
    Q8_0("q8_0", 0uL),   // 8-bit quantized (often 1 byte per element before block structure)
    Q8_1("q8_1", 0uL),   // 8-bit quantized with different scaling
    Q2_K("q2_k", 0uL),   // 2-bit quantized for K-quants
    Q3_K("q3_k", 0uL),   // 3-bit quantized for K-quants
    Q4_K("q4_k", 0uL),   // 4-bit quantized for K-quants
    Q5_K("q5_k", 0uL),   // 5-bit quantized for K-quants
    Q6_K("q6_k", 0uL),   // 6-bit quantized for K-quants
    Q8_K("q8_k", 0uL),   // 8-bit quantized for K-quants (potentially 1 byte + K-scale factor)
    Q1_5_K("q1_5_k", 0uL), // 1.5-bit quantized for K-quants (ternary: -1, 0, 1) - size is complex
    I8("int8", 1uL),     // 8-bit integer
    I16("int16", 2uL),    // 16-bit integer
    I32("int32", 4uL),    // 32-bit integer
    I64("int64", 8uL),    // 64-bit integer
    COUNT("count", 0uL)   // Number of types (not a real data type)
}

/**
 * Tensor operations
 */
enum class GGMLOp {
    NONE,
    DUP,
    ADD,
    SUB,
    MUL,
    DIV,
    SQR,
    SQRT,
    SUM,
    MEAN,
    REPEAT,
    ABS,
    SGN,
    NEG,
    STEP,
    RELU,
    GELU,
    SILU,
    NORM,
    RMS_NORM,
    MUL_MAT,
    SCALE,
    CPY,
    RESHAPE,
    VIEW,
    PERMUTE,
    TRANSPOSE,
    GET_ROWS,
    DIAG_MASK_INF,
    SOFT_MAX,
    ROPE,
    CONV_1D_1S,
    CONV_1D_2S,
    FLASH_ATTN,
    FLASH_FF,
    MAP_UNARY,
    MAP_BINARY,
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

    // Placeholder Accessor methods for F16 (Half Float)
    fun getHalf(graphAllocator: GGMLGraphAllocator, vararg indices: Int): Nothing {
        throw NotImplementedError("F16 (Half) get access is not yet implemented.")
    }

    fun setHalf(graphAllocator: GGMLGraphAllocator, value: Any, vararg indices: Int) {
        throw NotImplementedError("F16 (Half) set access is not yet implemented.")
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
    var computeImmediately: Boolean = true
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
