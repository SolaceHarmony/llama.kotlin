// port-lint: source ggml/include/ggml.h
package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe
import io.github.kotlinmania.llama.ore.insertBits
import io.github.kotlinmania.llama.ore.readBits
import io.github.kotlinmania.llama.ore.toUnsignedInt
import io.github.kotlinmania.llama.ore.withBits
import kotlin.Short.Companion.SIZE_BYTES

/**
 * Kotlin Native port of GGML tensor library core data types.
 * 
 * This file contains the fundamental data structures and type definitions used throughout
 * the GGML tensor computation library. It provides:
 * 
 * - Tensor data type definitions (F32, F16, quantized types, etc.)
 * - Core tensor structure (GGMLTensor) with dimension and memory layout information
 * - Operation type enumeration for computational graphs
 * - Memory management utilities and buffer access patterns
 * - Type-safe accessor methods for different data formats
 * 
 * The implementation follows Kotlin/Native memory management patterns while maintaining
 * compatibility with the original GGML C++ library design principles.
 */

/**
 * Maximum number of dimensions supported in a tensor.
 * GGML uses a fixed-size tensor structure with up to 4 dimensions
 * following NumPy-style dimension ordering.
 */
const val GGML_MAX_DIMS = 4

/**
 * Maximum number of source tensors for a single operation.
 * Used for operations that can take multiple input tensors.
 */
const val GGML_MAX_SRC = 10

private fun Byte.toUnsignedInt(): Int = toInt() and 0xFF
private fun Byte.readBits(offset: Int, width: Int): Int {
    val mask = if (width >= 32) -1 else (1 shl width) - 1
    return (toUnsignedInt() ushr offset) and mask
}
private fun Byte.withBits(value: Int, offset: Int, width: Int): Byte {
    val fieldMask = if (width >= 32) -1 else (1 shl width) - 1
    val mask = fieldMask shl offset
    val cleared = toUnsignedInt() and mask.inv()
    val writeVal = (value and fieldMask) shl offset
    return (cleared or (writeVal and mask)).toByte()
}
private fun Int.insertBits(value: Int, offset: Int, width: Int): Int {
    val fieldMask = if (width >= 32) -1 else (1 shl width) - 1
    val mask = fieldMask shl offset
    val cleared = this and mask.inv()
    val writeVal = (value and fieldMask) shl offset
    return cleared or (writeVal and mask)
}

/**
 * Maximum number of operation parameters.
 * Used for storing operation-specific configuration data.
 */
const val GGML_MAX_OP_PARAMS = 32

/**
 * Maximum name length for a tensor.
 * Used for debugging and graph visualization purposes.
 */
const val GGML_MAX_NAME = 64

/**
 * Tensor flag indicating this tensor is an output of the computation graph.
 * Used for gradient computation and memory management optimization.
 */
const val GGML_TENSOR_FLAG_OUTPUT = 1 shl 0

// Quantization block size constants
/** Block size for Q8_0 quantization (32 elements per block). */
internal const val QK8_0: Int = 32
/** Block size for Q4_0 quantization (32 elements per block). */
internal const val QK4_0: Int = 32
/** Block size for Q4_1 quantization (32 elements per block, same as Q4_0). */
internal const val QK4_1: Int = 32

// BitNet 1.58 constants  
/** Block size for BitNet 1.58 quantization (32 ternary values per block). */
internal const val QK_BITNET_1_58: Int = 32

// K-Quant constants
/** Super-block size for K-quantization schemes (256 elements per super-block). */
internal const val QK_K: Int = 256
/** Number of scale bytes for Q4_K and Q5_K quantization schemes. */
internal const val K_SCALE_SIZE: Int = 12

/**
 * Enumeration of supported tensor data types in the GGML system.
 * 
 * This enum defines all supported data formats for tensor storage and computation,
 * including:
 * - Standard floating-point types (F32, F16)
 * - Integer types (I32, I64, etc.)
 * - Quantized types for memory-efficient inference
 * - Specialized quantization schemes (K-quants, BitNet)
 * 
 * Each type includes its description and memory size per element in bytes.
 * The byteSize is used for memory allocation and stride calculations.
 * 
 * @param description Human-readable description of the data type
 * @param byteSize Size of one element in bytes (0 for variable-size types)
 */
enum class GGMLType(val description: String, val byteSize: ULong) {
    /** 32-bit IEEE 754 floating-point type. */
    F32("float32", 4uL),
    F16("float16", 2uL),    // 16-bit float
    // For quantized types, byteSize here represents the size of the fundamental element IF applicable for simple stride calculations.
    // Actual memory per element for quantized types is fractional and depends on block size.
    // Using 0uL as a skeleton signifies that direct byteSize-based stride calculation isn't straightforward.
    // The ggml library itself has type_size and block_size fields and functions like ggml_type_size() / ggml_blck_size().
    // For now, these are templates. The stride logic will primarily rely on non-zero byteSize for unquantized types.
    // Q4_0 byteSize is per block: sizeof(F16 scale) + (QK4_0/2) * sizeof(I8 weights_packed)
    Q4_0("q4_0", 2uL + (io.github.kotlinmania.llama.ore.QK4_0 / 2).toULong()),   // 4-bit quantized, 18 bytes per block (2 + 32/2*1)
    // Q4_1 byteSize is per block: 2 * sizeof(F16 scale/min) + (QK4_1/2) * sizeof(I8 weights_packed)
    Q4_1("q4_1", (2uL * Short.SIZE_BYTES.toULong()) + (io.github.kotlinmania.llama.ore.QK4_1 / 2).toULong()),   // 4-bit quantized: 2*F16 (scale d, min m) + QK4_1/2 bytes for packed weights = 4 + 16 = 20 bytes per block
    Q5_0("q5_0", 0uL),   // 5-bit quantized
    Q5_1("q5_1", 0uL),   // 5-bit quantized with different scaling
    // Q8_0 byteSize is per block: sizeof(Float16 for scale) + QK8_0 * sizeof(Int8 for weights)
    Q8_0("q8_0", 2uL + io.github.kotlinmania.llama.ore.QK8_0.toULong()),   // 8-bit quantized, 34 bytes per block (2 + 32*1)
    Q8_1("q8_1", 0uL),   // 8-bit quantized with different scaling
    // K-Quant types with correct block sizes (based on ggml-common.h)
    Q2_K("q2_k", (2uL * Short.SIZE_BYTES.toULong()) + (io.github.kotlinmania.llama.ore.QK_K / 16).toULong() + (io.github.kotlinmania.llama.ore.QK_K / 4).toULong()),   // 2*F16 + QK_K/16 + QK_K/4 = 4 + 16 + 64 = 84 bytes
    Q3_K("q3_k", Short.SIZE_BYTES.toULong() + (io.github.kotlinmania.llama.ore.QK_K / 4).toULong() + (io.github.kotlinmania.llama.ore.QK_K / 8).toULong() + 12uL),   // F16 + QK_K/4 + QK_K/8 + 12 = 2 + 64 + 32 + 12 = 110 bytes
    Q4_K("q4_k", (2uL * Short.SIZE_BYTES.toULong()) + io.github.kotlinmania.llama.ore.K_SCALE_SIZE.toULong() + (io.github.kotlinmania.llama.ore.QK_K / 2).toULong()),   // 2*F16 + K_SCALE_SIZE + QK_K/2 = 4 + 12 + 128 = 144 bytes
    Q5_K("q5_k", (2uL * Short.SIZE_BYTES.toULong()) + io.github.kotlinmania.llama.ore.K_SCALE_SIZE.toULong() + (io.github.kotlinmania.llama.ore.QK_K / 8).toULong() + (io.github.kotlinmania.llama.ore.QK_K / 2).toULong()),   // 2*F16 + K_SCALE_SIZE + QK_K/8 + QK_K/2 = 4 + 12 + 32 + 128 = 176 bytes
    Q6_K("q6_k", Short.SIZE_BYTES.toULong() + (io.github.kotlinmania.llama.ore.QK_K / 16).toULong() + ((3 * io.github.kotlinmania.llama.ore.QK_K) / 4).toULong()),   // F16 + QK_K/16 + 3*QK_K/4 = 2 + 16 + 192 = 210 bytes
    Q8_K("q8_k", 4uL + io.github.kotlinmania.llama.ore.QK_K.toULong() + ((io.github.kotlinmania.llama.ore.QK_K / 16) * 2).toULong()),   // (15) F32 + QK_K + QK_K/16*sizeof(int16_t)
    IQ2_XXS("iq2_xxs", 0uL), // (16)
    IQ2_XS("iq2_xs", 0uL),   // (17)
    IQ3_XXS("iq3_xxs", 0uL), // (18)
    IQ1_S("iq1_s", 0uL),     // (19)
    IQ4_NL("iq4_nl", 0uL),   // (20)
    IQ3_S("iq3_s", 0uL),     // (21)
    IQ2_S("iq2_s", 0uL),     // (22)
    IQ4_XS("iq4_xs", 0uL),   // (23)
    I8("int8", 1uL),         // (24) 8-bit integer
    I16("int16", 2uL),       // (25) 16-bit integer
    I32("int32", 4uL),       // (26) 32-bit integer
    I64("int64", 8uL),       // (27) 64-bit integer
    F64("float64", 8uL),     // (28) double precision
    IQ1_M("iq1_m", 0uL),     // (29)
    BF16("bf16", 2uL),       // (30) Brain float 16
    // 31-33 removed (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8)
    TQ1_0("tq1_0", 0uL),     // (34) Ternary quantization
    TQ2_0("tq2_0", 0uL),     // (35) Ternary quantization v2
    // 36-38 removed (IQ4_NL_4_4, IQ4_NL_4_8, IQ4_NL_8_8)
    MXFP4("mxfp4", 17uL),    // (39) MXFP4: 1 + QK_MXFP4/2 = 1 + 16 = 17 bytes
    NVFP4("nvfp4", 36uL),    // (40) NVFP4: 4 + QK_NVFP4/2 = 4 + 32 = 36 bytes
    Q1_0("q1_0", 18uL),      // (41) Q1_0: 2 + QK1_0/8 = 2 + 16 = 18 bytes
    // Custom extension types (not in C++ ggml)
    BITNET_1_58("bitnet_1_58", Short.SIZE_BYTES.toULong() + 8uL),
    Q1_5_K("q1_5_k", 0uL),
    COUNT("count", 0uL);   // sentinel — not a real data type

    /**
     * Get size in bytes as Int (convenience property)
     */
    val sizeBytes: Int
        get() = byteSize.toInt()

    companion object {
        /**
         * Create GGMLType from integer value based on C++ enum order
         */
        fun fromValue(value: Int): GGMLType? {
            return when (value) {
                0 -> F32
                1 -> F16
                2 -> Q4_0
                3 -> Q4_1
                // 4, 5 removed (Q4_2, Q4_3)
                6 -> Q5_0
                7 -> Q5_1
                8 -> Q8_0
                9 -> Q8_1
                10 -> Q2_K
                11 -> Q3_K
                12 -> Q4_K
                13 -> Q5_K
                14 -> Q6_K
                15 -> Q8_K
                16 -> IQ2_XXS
                17 -> IQ2_XS
                18 -> IQ3_XXS
                19 -> IQ1_S
                20 -> IQ4_NL
                21 -> IQ3_S
                22 -> IQ2_S
                23 -> IQ4_XS
                24 -> I8
                25 -> I16
                26 -> I32
                27 -> I64
                28 -> F64
                29 -> IQ1_M
                30 -> BF16
                // 31-33 removed
                34 -> TQ1_0
                35 -> TQ2_0
                // 36-38 removed
                39 -> MXFP4
                40 -> NVFP4
                41 -> Q1_0
                else -> null
            }
        }
    }
}

/**
 * Tensor operations
 */
enum class GGMLOp(val canBeInplace: Boolean = false) {
    NONE,
    DUP,
    ADD(true),
    ADD_ID(true),
    ADD1(true),
    SUB(true),
    MUL(true), // Element-wise multiplication
    DIV(true),
    SQR(true),
    SQRT(true),
    SUM, // Typically not inplace (reduces dimensions)
    MEAN, // Typically not inplace
    REPEAT,
    REPEAT_BACK,
    ABS(true),
    SGN(true),
    NEG(true),
    STEP(true),
    RELU(true),
    SIN(true),
    COS(true),
    LOG(true),
    GELU(true),
    GELU_QUICK(true),
    SILU(true),
    SILU_BACK(true),
    NORM(true),
    L2_NORM(true),
    RMS_NORM(true),
    RMS_NORM_BACK(true),
    MUL_MAT,    // Matrix multiplication, typically not inplace
    MUL_MAT_ID, // Matrix multiplication by expert id (MoE)
    SCALE(true),
    CPY, // Copy, not inplace by definition of creating a new tensor with copied data
    RESHAPE, // Reshape is a view, metadata change, not inplace on data buffer in the same way
    VIEW,    // View is a metadata change
    PERMUTE,
    TRANSPOSE,
    GET_ROWS,
    DIAG_MASK_INF(true),
    SOFT_MAX(true), // Can be made inplace
    SOFT_MAX_BACK(true),
    ROPE(true),
    ROPE_BACK,
    CLAMP(true),
    CONV_TRANSPOSE_1D,
    IM2COL,
    IM2COL_BACK,
    IM2COL_3D,
    CONV_2D,
    CONV_3D,
    CONV_2D_DW,
    CONV_TRANSPOSE_2D,
    POOL_1D,
    POOL_2D,
    POOL_2D_BACK,
    UPSCALE,
    PAD,
    PAD_REFLECT_1D,
    ROLL,
    ARANGE,
    TIMESTEP_EMBEDDING,
    ARGSORT,
    TOP_K,
    LEAKY_RELU(true),
    TRI,
    FILL(true),
    FLASH_ATTN_EXT,
    FLASH_ATTN_BACK,
    SSM_CONV,
    SSM_SCAN,
    WIN_PART,
    WIN_UNPART,
    GET_REL_POS,
    RWKV_WKV6,
    RWKV_WKV7,
    GATED_LINEAR_ATTN,
    GATED_DELTA_NET,
    UNARY,
    GLU,
    MAP_CUSTOM1,
    MAP_CUSTOM2,
    MAP_CUSTOM3,
    CUSTOM_4D,
    CROSS_ENTROPY_LOSS,
    CROSS_ENTROPY_LOSS_BACK,
    OPT_STEP_ADAMW,
    OPT_STEP_SGD,
    ACC(true),
    SET(true),
    CONT,
    CONCAT,
    SUM_ROWS,
    ARGMAX,
    CUMSUM,
    COUNT_EQUAL,
    GET_ROWS_BACK,
    SET_ROWS,
    DIAG,
    DIAG_MASK_ZERO(true),
    OUT_PROD,
    GROUP_NORM(true),
    // kept for compatibility with existing code
    CONV_1D_1S,
    CONV_1D_2S,
    FLASH_ATTN,
    FLASH_FF,
    MAP_UNARY,
    MAP_BINARY,
    EXP(true),
    SIGMOID(true),
    TANH(true),
    HARDSWISH(true),
    HARDSIGMOID(true),
    GELU_ERF(true),
    SILU_BACK_OP(true),
    SOFTPLUS(true),
    ELU(true),
    CEIL(true),
    FLOOR(true),
    ROUND(true),
    TRUNC(true),
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
    var type: io.github.kotlinmania.llama.ore.GGMLObjectType = io.github.kotlinmania.llama.ore.GGMLObjectType.TENSOR,
    var offs: Long = 0L,
    var offset: ULong = 0u,
    var size: ULong = 0u,
    var next: GGMLObject? = null,
    var tensor: io.github.kotlinmania.llama.ore.GGMLTensor? = null
)

/**
 * Tensor data structure
 */
class GGMLTensor(
    var type: io.github.kotlinmania.llama.ore.GGMLType = io.github.kotlinmania.llama.ore.GGMLType.F32,
    var buffer: io.github.kotlinmania.llama.ore.GGMLBackendBuffer? = null,
    var ne: LongArray = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 0L },
    var nb: ULongArray = ULongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 0u },
    var op: io.github.kotlinmania.llama.ore.GGMLOp = io.github.kotlinmania.llama.ore.GGMLOp.NONE,
    var opParams: IntArray = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { 0 },
    var flags: Int = 0,
    var grad: GGMLTensor? = null,
    var src: Array<GGMLTensor?> = Array(io.github.kotlinmania.llama.ore.GGML_MAX_SRC) { null },
    var viewSrc: GGMLTensor? = null,
    var viewOffs: ULong = 0u,
    var data: Any? = null,
    var name: String = "",
    var bufferId: Int = -1,
    var dataOffset: ULong = 0u,
    var parentObject: io.github.kotlinmania.llama.ore.GGMLObject? = null
) {
    var offset: ULong
        get() = dataOffset
        set(value) { dataOffset = value }

    fun isOutput(): Boolean = (this.flags and io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_OUTPUT) != 0

    /**
     * Total number of bytes used by this tensor's data.
     *
     * Port of `ggml_nbytes()` — computes `max(1, ne[3]) * nb[3]` which gives
     * the tight byte count for the entire tensor, including padding implied by
     * the stride array.
     */
    fun nBytes(): Long {
        // If strides are properly set, use ne[3]*nb[3]
        if (nb.size > 3 && nb[3] > 0u) {
            val ne3 = if (ne.size > 3) maxOf(1L, ne[3]) else 1L
            return ne3 * nb[3].toLong()
        }
        // Fallback: compute from element count and type byte size
        val nelem = numElements()
        val bs = type.byteSize.toLong()
        return if (bs > 0) nelem * bs else 0L
    }

    /**
     * Compute the number of bytes for a "row" of elements of this tensor's type.
     *
     * Port of `ggml_row_size(type, ne)` — for non-quantized types this is
     * simply `typeByteSize * ne`. For block-quantized types this accounts for
     * the block packing ratio.
     *
     * @param nElements Number of elements in the row.
     */
    fun rowSizeBytes(nElements: Int): Long {
        val bs = type.byteSize.toLong()
        return if (bs > 0) nElements.toLong() * bs else 0L
    }

    /**
     * Copy [length] bytes from [src] into this tensor's data at byte offset [offset].
     *
     * Port of `memcpy(tensor->data + offset, src, length)`.
     * Assumes [data] is a [ByteArray].
     */
    fun setData(src: ByteArray, offset: Int, length: Int) {
        val dst = data as? ByteArray
            ?: throw IllegalStateException("setData: tensor data is not a ByteArray")
        src.copyInto(dst, destinationOffset = offset, startIndex = 0, endIndex = length)
    }

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
        if (this.type == io.github.kotlinmania.llama.ore.GGMLType.COUNT) {
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
        // The above rank check might be too strict if ne contains trailing 1s for lower rank tensors.
        // Example: A 2D tensor might have ne = [10, 20, 1, 1]. Rank is 2. indices.size should be 2.

        // Iterating up to indices.size assumes that the provided indices match the intended dimensions.
        for (d in indices.indices) {
            if (d >= io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { // Should not happen if indices.size is checked against rank based on ne
                throw IllegalArgumentException("Dimension index $d exceeds GGML_MAX_DIMS.")
            }
            if (indices[d] < 0 || indices[d] >= ne[d]) {
                val shapeString = ne.joinToString(limit = io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                throw IllegalArgumentException("Index ${indices[d]} for dimension $d is out of bounds (0 to ${ne[d] - 1}) for tensor shape [$shapeString]")
            }
            finalOffset += indices[d].toULong() * nb[d]
        }
        return finalOffset
    }

    // Accessor methods for F32
    fun getFloat(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, vararg indices: Int): Float {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for F32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for F32 is out of bounds for buffer size ${buffer.size}")
        }
        return buffer.getFloatLe(finalByteOffset.toInt())
    }

    fun setFloat(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, value: Float, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for F32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for F32 is out of bounds for buffer size ${buffer.size}")
        }
        buffer.setFloatLe(finalByteOffset.toInt(), value)
    }

    // Accessor methods for I32
    fun getInt(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, vararg indices: Int): Int {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for I32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for I32 is out of bounds for buffer size ${buffer.size}")
        }
        return buffer.getIntLe(finalByteOffset.toInt())
    }

    fun setInt(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, value: Int, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 4u > buffer.size.toUInt()) { // Check for I32 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 4 bytes for I32 is out of bounds for buffer size ${buffer.size}")
        }
        buffer.setIntLe(finalByteOffset.toInt(), value)
    }

    // Accessor methods for I16
    fun getShort(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, vararg indices: Int): Short {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 2u > buffer.size.toUInt()) { // Check for I16 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 2 bytes for I16 is out of bounds for buffer size ${buffer.size}")
        }
        return buffer.getShortLe(finalByteOffset.toInt())
    }

    fun setShort(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, value: Short, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset + 2u > buffer.size.toUInt()) { // Check for I16 size
            throw IndexOutOfBoundsException("Calculated offset $finalByteOffset + 2 bytes for I16 is out of bounds for buffer size ${buffer.size}")
        }
        buffer.setShortLe(finalByteOffset.toInt(), value)
    }

    // Accessor methods for F16 (Half Float)
    fun getHalf(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, vararg indices: Int): Float {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        // Check bounds, considering Short.SIZE_BYTES (2 bytes for F16)
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Short at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - SIZE_BYTES})")
        }
        val shortBits = buffer.getShortLe(finalByteOffset.toInt())
        return io.github.kotlinmania.llama.ore.halfToFloat(shortBits) // halfToFloat is in NumericConversions.kt, assumed imported or accessible
    }

    fun setHalf(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, value: Float, vararg indices: Int) {
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated.")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        // Check bounds, considering Short.SIZE_BYTES (2 bytes for F16)
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Short at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - SIZE_BYTES})")
        }
        val shortBits =
            io.github.kotlinmania.llama.ore.floatToHalf(value) // floatToHalf is in NumericConversions.kt, assumed imported or accessible
        buffer.setShortLe(finalByteOffset.toInt(), shortBits)
    }

    // Accessor methods for I8 (Byte)
    fun getByte(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, vararg indices: Int): Byte {
        // require(type == GGMLType.I8) { "getByte() called on non-I8 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Byte at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}) for tensor $name")
        }
        return buffer[finalByteOffset.toInt()]
    }

    fun setByte(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, value: Byte, vararg indices: Int) {
        // require(type == GGMLType.I8) { "setByte() called on non-I8 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to write Byte at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}) for tensor $name")
        }
        buffer[finalByteOffset.toInt()] = value
    }

    // Accessor methods for I64 (Long)
    fun getLong(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, vararg indices: Int): Long {
        // require(type == GGMLType.I64) { "getLong() called on non-I64 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        val elementByteOffset = getElementByteOffset(*indices)
        val finalByteOffset = dataOffset + elementByteOffset
        if (finalByteOffset.toInt() < 0 || finalByteOffset.toInt() + Long.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Long at offset ${finalByteOffset.toInt()} (tensor offset $dataOffset + element offset $elementByteOffset) is out of buffer bounds (0-${buffer.size - Long.SIZE_BYTES}) for tensor $name")
        }
        return buffer.getLongLe(finalByteOffset.toInt()) // Uses new ByteArray extension
    }

    fun setLong(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, value: Long, vararg indices: Int) {
        // require(type == GGMLType.I64) { "setLong() called on non-I64 tensor: $type" }
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
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
            io.github.kotlinmania.llama.ore.GGMLType.Q8_0 -> io.github.kotlinmania.llama.ore.QK8_0.toLong()
            io.github.kotlinmania.llama.ore.GGMLType.Q4_0 -> io.github.kotlinmania.llama.ore.QK4_0.toLong()
            io.github.kotlinmania.llama.ore.GGMLType.Q4_1 -> io.github.kotlinmania.llama.ore.QK4_1.toLong()
            // BitNet 1.58 quantization
            io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> io.github.kotlinmania.llama.ore.QK_BITNET_1_58.toLong()
            // K-Quant types
            io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K -> io.github.kotlinmania.llama.ore.QK_K.toLong()
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
    fun getQ8_0BlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "Tensor type must be Q8_0 to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        // type.byteSize for Q8_0 is the size of one block (e.g., 34 bytes)
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId. Ensure graphAllocator.buffers is populated for tensor '$name'.")

        // The scale is the first F16 (2 bytes) in the block
        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q8_0 scale at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}). DataOffset: $dataOffset, BlockByteOffset: $blockByteOffset.")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return io.github.kotlinmania.llama.ore.halfToFloat(scaleBits)
    }

    /**
     * Retrieves a single quantized weight (Int8/Byte) from a specific block in a Q8_0 tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK8_0 - 1).
     * @return The quantized weight as a Byte.
     */
    fun getQ8_0Weight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "Tensor type must be Q8_0 to get weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < io.github.kotlinmania.llama.ore.QK8_0) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${io.github.kotlinmania.llama.ore.QK8_0 -1}) for Q8_0 block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize // type.byteSize is block size for Q8_0
        val qsArrayBaseOffsetInBlock = 2uL // The F16 scale takes the first 2 bytes of the block
        val finalWeightByteOffset = dataOffset + blockByteOffset + qsArrayBaseOffsetInBlock + itemIndexInBlock.toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
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
    fun getQ4_0BlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0) { "Tensor type must be Q4_0 to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        // type.byteSize for Q4_0 is the size of one block (e.g., 18 bytes)
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        // The scale is the first F16 (2 bytes) in the block
        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_0 scale at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}). DataOffset: $dataOffset, BlockByteOffset: $blockByteOffset.")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return io.github.kotlinmania.llama.ore.halfToFloat(scaleBits)
    }

    /**
     * Retrieves a single 4-bit quantized weight (nibble) from a specific block in a Q4_0 tensor.
     * The returned Byte contains the raw 4-bit value (0-15).
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK4_0 - 1).
     * @return The quantized 4-bit weight as a Byte (value 0-15).
     */
    fun getQ4_0NibbleWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0) { "Tensor type must be Q4_0 to get nibble weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < io.github.kotlinmania.llama.ore.QK4_0) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${io.github.kotlinmania.llama.ore.QK4_0 -1}) for Q4_0 block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val qsArrayBaseOffsetInBlock = 2uL // The F16 scale takes the first 2 bytes
        val byteContainingNibbleIndex = itemIndexInBlock / 2 // Each byte stores two 4-bit nibbles

        val finalByteToReadOffset = dataOffset + blockByteOffset + qsArrayBaseOffsetInBlock + byteContainingNibbleIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToReadOffset.toInt() < 0 || finalByteToReadOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to read Q4_0 nibble weight byte at offset ${finalByteToReadOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds (0-${buffer.size - Byte.SIZE_BYTES}).")
        }
        val packedByte = buffer[finalByteToReadOffset.toInt()]

        val nibble = if (itemIndexInBlock % 2 == 0) {
            packedByte.readBits(0, 4)
        } else {
            packedByte.readBits(4, 4)
        }
        return nibble.toByte()
    }

    // --- Q4_1 Accessors ---

    /**
     * Retrieves the F16 scale ('d') for a specific block in a Q4_1 quantized tensor.
     */
    fun getQ4_1BlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "Tensor type must be Q4_1 to get block scale 'd'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize // type.byteSize for Q4_1 is 20 bytes
        val finalScaleByteOffset = dataOffset + blockByteOffset // Scale 'd' is the first F16

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_1 scale 'd' at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - SIZE_BYTES}).")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return io.github.kotlinmania.llama.ore.halfToFloat(scaleBits)
    }

    /**
     * Retrieves the F16 min value ('m') for a specific block in a Q4_1 quantized tensor.
     */
    fun getQ4_1BlockMin(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "Tensor type must be Q4_1 to get block min 'm'." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val minOffsetWithinBlock = SIZE_BYTES.toULong() // Min 'm' is the second F16 (after the scale 'd')
        val finalMinByteOffset = dataOffset + blockByteOffset + minOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalMinByteOffset.toInt() < 0 || finalMinByteOffset.toInt() + SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read Q4_1 min 'm' at offset ${finalMinByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - SIZE_BYTES}).")
        }

        val minBits = buffer.getShortLe(finalMinByteOffset.toInt())
        return io.github.kotlinmania.llama.ore.halfToFloat(minBits)
    }

    /**
     * Retrieves a single 4-bit quantized weight (nibble) from a specific block in a Q4_1 tensor.
     * The returned Byte contains the raw 4-bit value (0-15).
     */
    fun getQ4_1NibbleWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "Tensor type must be Q4_1 to get nibble weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < io.github.kotlinmania.llama.ore.QK4_1) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${io.github.kotlinmania.llama.ore.QK4_1 -1}) for Q4_1 block in tensor '$name'."}

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val qsBaseOffsetWithinBlock = (2 * SIZE_BYTES).toULong() // Weights start after two F16s (d and m)
        val byteContainingNibbleIndex = itemIndexInBlock / 2

        val finalByteToReadOffset = dataOffset + blockByteOffset + qsBaseOffsetWithinBlock + byteContainingNibbleIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToReadOffset.toInt() < 0 || finalByteToReadOffset.toInt() + Byte.SIZE_BYTES > buffer.size) {
             throw IndexOutOfBoundsException("Attempt to read Q4_1 nibble weight byte at offset ${finalByteToReadOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds.")
        }
        val packedByte = buffer[finalByteToReadOffset.toInt()]

        val nibble = if (itemIndexInBlock % 2 == 0) {
            packedByte.readBits(0, 4)
        } else {
            packedByte.readBits(4, 4)
        }
        return nibble.toByte()
    }
    
    // --- K-Quant Accessor Methods ---
    
    /**
     * Retrieves the super-block scale (d) for a Q2_K block.
     * Q2_K structure: scales[QK_K/16], qs[QK_K/4], d (F16), dmin (F16)
     */
    fun getQ2_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q2_K) { "Tensor type must be Q2_K to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalByteOffset = dataOffset + blockByteOffset + (io.github.kotlinmania.llama.ore.QK_K /16 + io.github.kotlinmania.llama.ore.QK_K /4).toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }
    
    /**
     * Retrieves the super-block scale for mins (dmin) for a Q2_K block.
     */
    fun getQ2_KBlockScaleMin(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q2_K) { "Tensor type must be Q2_K to get block scale min." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalByteOffset = dataOffset + blockByteOffset + (io.github.kotlinmania.llama.ore.QK_K /16 + io.github.kotlinmania.llama.ore.QK_K /4).toULong() + 2uL

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }
    
    /**
     * Retrieves the packed scale/min byte for a Q2_K sub-block (lower 4 bits = scale, upper 4 bits = min).
     */
    fun getQ2_KScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, scaleIndex: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q2_K) { "Tensor type must be Q2_K to get scale/min." }
        require(scaleIndex >= 0 && scaleIndex < io.github.kotlinmania.llama.ore.QK_K /16) { "scaleIndex $scaleIndex out of bounds for Q2_K scales" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val scalesOffsetWithinBlock = scaleIndex.toULong()
        val finalByteOffset = dataOffset + blockByteOffset + scalesOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return buffer[finalByteOffset.toInt()]
    }
    
    /**
     * Retrieves a quantized weight byte from a Q2_K block (contains 4 weights, 2 bits each).
     */
    fun getQ2_KQuant(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, quantIndex: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q2_K) { "Tensor type must be Q2_K to get quant." }
        require(quantIndex >= 0 && quantIndex < io.github.kotlinmania.llama.ore.QK_K /4) { "quantIndex $quantIndex out of bounds for Q2_K quants" }
        
        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val quantOffsetWithinBlock = (io.github.kotlinmania.llama.ore.QK_K /16).toULong() + quantIndex.toULong()
        val finalByteOffset = dataOffset + blockByteOffset + quantOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return buffer[finalByteOffset.toInt()]
    }

    /**
     * Retrieves the super-block scale (d) for a Q3_K block.
     * Q3_K structure: hmask[QK_K/8], qs[QK_K/4], scales[12], d (F16)
     */
    fun getQ3_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q3_K) { "Tensor type must be Q3_K to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // d is located after hmask[QK_K/8] + qs[QK_K/4] + scales[12]
        val scaleOffsetWithinBlock = (io.github.kotlinmania.llama.ore.QK_K /8).toULong() + (io.github.kotlinmania.llama.ore.QK_K /4).toULong() + 12uL
        val finalByteOffset = dataOffset + blockByteOffset + scaleOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }

    /**
     * Retrieves the super-block scale (d) for a Q4_K block.
     * Q4_K structure: d (F16), dmin (F16), scales[io.github.kotlinmania.llama.ore.K_SCALE_SIZE], qs[QK_K/2]
     */
    fun getQ4_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // d is the first F16 in the block
        val finalByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }
    
    /**
     * Retrieves the super-block scale for mins (dmin) for a Q4_K block.
     */
    fun getQ4_KBlockScaleMin(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to get block scale min." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // dmin is the second F16 in the block (after d)
        val finalByteOffset = dataOffset + blockByteOffset + 2uL

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }

    /**
     * Retrieves a quantized scale value from the Q4_K scales array for a specific sub-block.
     * @param graphAllocator The graph allocator managing tensor memory
     * @param blockIndex The Q4_K block index (each block contains 8 sub-blocks)
     * @param subBlockIndex The sub-block index within the block (0-7)
     * @return The quantized scale value (6 bits, 0-63 range)
     */
    fun getQ4_KQuantizedScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, subBlockIndex: Int): Int {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to get quantized scale." }
        require(subBlockIndex in 0..7) { "subBlockIndex must be in range 0-7, got $subBlockIndex" }
        
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // Scales start after d (F16) + dmin (F16) = 4 bytes
        val scaleByteOffset = blockByteOffset + 4uL + subBlockIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        val scaleByte = buffer[(dataOffset + scaleByteOffset).toInt()]
        return scaleByte.readBits(0, 6)
    }

    /**
     * Retrieves a quantized min value from the Q4_K scales array for a specific sub-block.
     * @param graphAllocator The graph allocator managing tensor memory
     * @param blockIndex The Q4_K block index (each block contains 8 sub-blocks)
     * @param subBlockIndex The sub-block index within the block (0-7)
     * @return The quantized min value (6 bits, 0-63 range)
     */
    fun getQ4_KQuantizedMin(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, subBlockIndex: Int): Int {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to get quantized min." }
        require(subBlockIndex in 0..7) { "subBlockIndex must be in range 0-7, got $subBlockIndex" }
        
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        
        // Min values are packed with scales - low 2 bits in the scale byte, high 4 bits packed in bytes 8-11 of the scales array
        val scaleByteOffset = blockByteOffset + 4uL + subBlockIndex.toULong()
        val scaleByte = buffer[(dataOffset + scaleByteOffset).toInt()]
        val quantizedMinLow = scaleByte.readBits(6, 2)

        val minHighByteOffset = blockByteOffset + 4uL + 8uL + (subBlockIndex / 2).toULong()
        val minHighByte = buffer[(dataOffset + minHighByteOffset).toInt()]
        val quantizedMinHigh = if (subBlockIndex % 2 == 0) {
            minHighByte.readBits(0, 4)
        } else {
            minHighByte.readBits(4, 4)
        }

        return quantizedMinLow.insertBits(quantizedMinHigh, 2, 4)
    }

    /**
     * Retrieves a quantized weight (4-bit) from a Q4_K block.
     * @param graphAllocator The graph allocator managing tensor memory
     * @param blockIndex The Q4_K block index
     * @param elementIndex The element index within the block (0 to QK_K-1)
     * @return The quantized weight value (4 bits, 0-15 range)
     */
    fun getQ4_KWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, elementIndex: Int): Int {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to get weight." }
        require(elementIndex in 0 until io.github.kotlinmania.llama.ore.QK_K) { "elementIndex must be in range 0 until ${io.github.kotlinmania.llama.ore.QK_K}, got $elementIndex" }
        
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        
        // Weights start after d (F16) + dmin (F16) + scales[K_SCALE_SIZE] = 4 + 12 = 16 bytes
        val weightsStartOffset = blockByteOffset + 4uL + io.github.kotlinmania.llama.ore.K_SCALE_SIZE.toULong()
        
        // Each byte contains 2 weights (4 bits each)
        val byteOffset = weightsStartOffset + (elementIndex / 2).toULong()
        val weightByte = buffer[(dataOffset + byteOffset).toInt()]
        
        return if (elementIndex % 2 == 0) {
            weightByte.readBits(0, 4)
        } else {
            weightByte.readBits(4, 4)
        }
    }

    /**
     * Sets the super-block scale (d) for a Q4_K block.
     */
    fun setQ4_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, scale: Float) {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to set block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        buffer.setShortLe(finalByteOffset.toInt(),
            io.github.kotlinmania.llama.ore.floatToHalf(scale)
        )
    }

    /**
     * Sets the super-block scale for mins (dmin) for a Q4_K block.
     */
    fun setQ4_KBlockScaleMin(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, scaleMin: Float) {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to set block scale min." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalByteOffset = dataOffset + blockByteOffset + 2uL

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        buffer.setShortLe(finalByteOffset.toInt(),
            io.github.kotlinmania.llama.ore.floatToHalf(scaleMin)
        )
    }

    /**
     * Sets a quantized scale and min value in the Q4_K scales array for a specific sub-block.
     * @param graphAllocator The graph allocator managing tensor memory
     * @param blockIndex The Q4_K block index
     * @param subBlockIndex The sub-block index within the block (0-7)
     * @param quantizedScale The quantized scale value (6 bits, 0-63 range)
     * @param quantizedMin The quantized min value (6 bits, 0-63 range)
     */
    fun setQ4_KQuantizedScaleAndMin(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, subBlockIndex: Int, quantizedScale: Int, quantizedMin: Int) {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to set quantized scale and min." }
        require(subBlockIndex in 0..7) { "subBlockIndex must be in range 0-7, got $subBlockIndex" }
        require(quantizedScale in 0..63) { "quantizedScale must be in range 0-63, got $quantizedScale" }
        require(quantizedMin in 0..63) { "quantizedMin must be in range 0-63, got $quantizedMin" }
        
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        
        // Pack scale (6 bits) + min low bits (2 bits) into one byte
        val scaleByteOffset = blockByteOffset + 4uL + subBlockIndex.toULong()
        val scaleByteIndex = (dataOffset + scaleByteOffset).toInt()
        var scaleByte = buffer[scaleByteIndex]
        scaleByte = scaleByte.withBits(quantizedScale, 0, 6)
        scaleByte = scaleByte.withBits(quantizedMin, 6, 2)
        buffer[scaleByteIndex] = scaleByte
        
        // Store high 4 bits of min in correct location
        val minByteOffset = blockByteOffset + 4uL + 8uL + (subBlockIndex / 2).toULong()
        val minByteIndex = (dataOffset + minByteOffset).toInt()
        // Bounds check similar to getQ4_KQuantizedMin
        if (minByteIndex >= 0 && minByteIndex < buffer.size) {
            val minHighBits = (quantizedMin ushr 2) and ((1 shl 4) - 1)
            val bitOffset = if (subBlockIndex % 2 == 0) 0 else 4
            buffer[minByteIndex] = buffer[minByteIndex].withBits(minHighBits, bitOffset, 4)
        }
    }

    /**
     * Sets a quantized weight (4-bit) in a Q4_K block.
     * @param graphAllocator The graph allocator managing tensor memory
     * @param blockIndex The Q4_K block index
     * @param elementIndex The element index within the block (0 to QK_K-1)
     * @param weight The quantized weight value (4 bits, 0-15 range)
     */
    fun setQ4_KWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, elementIndex: Int, weight: Int) {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "Tensor type must be Q4_K to set weight." }
        require(elementIndex in 0 until io.github.kotlinmania.llama.ore.QK_K) { "elementIndex must be in range 0 until ${io.github.kotlinmania.llama.ore.QK_K}, got $elementIndex" }
        require(weight in 0..15) { "weight must be in range 0-15, got $weight" }
        
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        
        // Weights start after d (F16) + dmin (F16) + scales[K_SCALE_SIZE] = 4 + 12 = 16 bytes
        val weightsStartOffset = blockByteOffset + 4uL + io.github.kotlinmania.llama.ore.K_SCALE_SIZE.toULong()
        val byteOffset = weightsStartOffset + (elementIndex / 2).toULong()
        val byteIndex = (dataOffset + byteOffset).toInt()
        
        val bitOffset = if (elementIndex % 2 == 0) 0 else 4
        buffer[byteIndex] = buffer[byteIndex].withBits(weight, bitOffset, 4)
    }

    /**
     * Retrieves the super-block scale (d) for a Q5_K block.
     * Q5_K structure: d (F16), dmin (F16), scales[io.github.kotlinmania.llama.ore.K_SCALE_SIZE], qh[QK_K/8], qs[QK_K/2]
     */
    fun getQ5_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q5_K) { "Tensor type must be Q5_K to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }

    /**
     * Retrieves the super-block scale (d) for a Q6_K block.
     * Q6_K structure: ql[QK_K/2], qh[QK_K/4], scales[QK_K/16], d (F16)
     */
    fun getQ6_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q6_K) { "Tensor type must be Q6_K to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // d is located after ql[QK_K/2] + qh[QK_K/4] + scales[QK_K/16]
        val scaleOffsetWithinBlock = (io.github.kotlinmania.llama.ore.QK_K /2).toULong() + (io.github.kotlinmania.llama.ore.QK_K /4).toULong() + (io.github.kotlinmania.llama.ore.QK_K /16).toULong()
        val finalByteOffset = dataOffset + blockByteOffset + scaleOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return io.github.kotlinmania.llama.ore.halfToFloat(buffer.getShortLe(finalByteOffset.toInt()))
    }

    /**
     * Retrieves the super-block scale (d) for a Q8_K block.
     * Q8_K structure: d (F32), qs[io.github.kotlinmania.llama.ore.QK_K], bsums[QK_K/16]
     */
    fun getQ8_KBlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q8_K) { "Tensor type must be Q8_K to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return buffer.getFloatLe(finalByteOffset.toInt())
    }

    /**
     * Retrieves a quantized weight from a Q8_K block.
     */
    fun getQ8_KWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.Q8_K) { "Tensor type must be Q8_K to get weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < io.github.kotlinmania.llama.ore.QK_K) { "itemIndexInBlock $itemIndexInBlock out of bounds for Q8_K block" }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        // qs starts after d (F32)
        val weightOffsetWithinBlock = 4uL + itemIndexInBlock.toULong()
        val finalByteOffset = dataOffset + blockByteOffset + weightOffsetWithinBlock

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId")
        return buffer[finalByteOffset.toInt()]
    }
    
    // --- BitNet 1.58 Accessor Methods ---
    
    /**
     * Retrieves the F16 scale for a specific block in a BitNet 1.58 quantized tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @return The scale value as a Float.
     */
    fun getBitNet158BlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int): Float {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "Tensor type must be BITNET_1_58 to get block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        // The scale is the first F16 (2 bytes) in the block
        if (finalScaleByteOffset.toInt() < 0 || finalScaleByteOffset.toInt() + Short.SIZE_BYTES > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read BitNet 1.58 scale at offset ${finalScaleByteOffset.toInt()} for block $blockIndex in tensor '$name' is out of buffer bounds (0-${buffer.size - Short.SIZE_BYTES}).")
        }

        val scaleBits = buffer.getShortLe(finalScaleByteOffset.toInt())
        return io.github.kotlinmania.llama.ore.halfToFloat(scaleBits)
    }

    /**
     * Retrieves a single ternary weight (-1, 0, +1) from a specific block in a BitNet 1.58 tensor.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block (0 to QK_BITNET_1_58 - 1).
     * @return The ternary weight as a Byte (-1, 0, or +1).
     */
    fun getBitNet158TernaryWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int): Byte {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "Tensor type must be BITNET_1_58 to get ternary weight." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < io.github.kotlinmania.llama.ore.QK_BITNET_1_58) { "itemIndexInBlock $itemIndexInBlock out of bounds (0-${io.github.kotlinmania.llama.ore.QK_BITNET_1_58 - 1}) for BitNet 1.58 block in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val ternaryDataBaseOffsetInBlock = Short.SIZE_BYTES.toULong() // The F16 scale takes the first 2 bytes
        
        // We pack 5 ternary values into 1 byte using a base-3 encoding
        // 3^5 = 243 which fits in a byte (0-255)
        val groupIndex = itemIndexInBlock / 5  // Which byte contains this ternary value
        val positionInGroup = itemIndexInBlock % 5  // Position within the group of 5
        
        val finalByteToReadOffset = dataOffset + blockByteOffset + ternaryDataBaseOffsetInBlock + groupIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        if (finalByteToReadOffset.toInt() < 0 || finalByteToReadOffset.toInt() + 1 > buffer.size) {
            throw IndexOutOfBoundsException("Attempt to read BitNet 1.58 ternary weight byte at offset ${finalByteToReadOffset.toInt()} for block $blockIndex, item $itemIndexInBlock in tensor '$name' is out of buffer bounds.")
        }
        
        val packedByte = buffer[finalByteToReadOffset.toInt()].toUByte().toInt()
        
        // Decode the ternary value from base-3 encoding
        // Each byte stores 5 ternary values as: v4*3^4 + v3*3^3 + v2*3^2 + v1*3^1 + v0*3^0
        val powers = intArrayOf(1, 3, 9, 27, 81) // 3^0, 3^1, 3^2, 3^3, 3^4
        var remaining = packedByte
        val ternaryValues = IntArray(5)
        
        // Extract ternary values in reverse order (from highest power to lowest)
        for (i in 4 downTo 0) {
            ternaryValues[i] = remaining / powers[i]
            remaining %= powers[i]
        }
        
        // Convert from {0, 1, 2} to {-1, 0, +1}
        val ternaryValue = when (ternaryValues[positionInGroup]) {
            0 -> -1
            1 -> 0
            2 -> +1
            else -> throw IllegalStateException("Invalid ternary value decoded: ${ternaryValues[positionInGroup]}")
        }
        
        return ternaryValue.toByte()
    }
    
    /**
     * Sets a ternary weight (-1, 0, +1) for a specific position in a BitNet 1.58 tensor block.
     * This is used during quantization operations.
     * @param graphAllocator The graph allocator holding the buffer.
     * @param blockIndex The 0-based index of the block.
     * @param itemIndexInBlock The 0-based index of the weight within the block.
     * @param ternaryValue The ternary weight to set (-1, 0, or +1).
     */
    fun setBitNet158TernaryWeight(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, itemIndexInBlock: Int, ternaryValue: Byte) {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "Tensor type must be BITNET_1_58 to set ternary weight." }
        require(ternaryValue in -1..1) { "BitNet 1.58 ternary value must be -1, 0, or +1. Got $ternaryValue" }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }
        require(itemIndexInBlock >= 0 && itemIndexInBlock < io.github.kotlinmania.llama.ore.QK_BITNET_1_58) { "itemIndexInBlock $itemIndexInBlock out of bounds for BitNet 1.58 block in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val ternaryDataBaseOffsetInBlock = Short.SIZE_BYTES.toULong()
        
        val groupIndex = itemIndexInBlock / 5
        val positionInGroup = itemIndexInBlock % 5
        
        val finalByteToWriteOffset = dataOffset + blockByteOffset + ternaryDataBaseOffsetInBlock + groupIndex.toULong()

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        // Read current packed byte
        val currentPackedByte = buffer[finalByteToWriteOffset.toInt()].toUByte().toInt()
        
        // Convert ternary value from {-1, 0, +1} to {0, 1, 2}
        val encodedValue = when (ternaryValue.toInt()) {
            -1 -> 0
            0 -> 1
            1 -> 2
            else -> throw IllegalArgumentException("Invalid ternary value: $ternaryValue")
        }
        
        // Decode current values
        val powers = intArrayOf(1, 3, 9, 27, 81)
        var remaining = currentPackedByte
        val ternaryValues = IntArray(5)
        
        for (i in 4 downTo 0) {
            ternaryValues[i] = remaining / powers[i]
            remaining %= powers[i]
        }
        
        // Update the specific position
        ternaryValues[positionInGroup] = encodedValue
        
        // Re-encode to packed byte
        var newPackedValue = 0
        for (i in 0 until 5) {
            newPackedValue += ternaryValues[i] * powers[i]
        }
        
        buffer[finalByteToWriteOffset.toInt()] = newPackedValue.toByte()
    }
    
    /**
     * Sets the F16 scale for a specific block in a BitNet 1.58 quantized tensor.
     */
    fun setBitNet158BlockScale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, blockIndex: Int, scale: Float) {
        require(type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "Tensor type must be BITNET_1_58 to set block scale." }
        val numBlocks = getNumBlocks()
        require(blockIndex >= 0 && blockIndex < numBlocks) { "blockIndex $blockIndex out of bounds for $numBlocks blocks in tensor '$name'." }

        val blockByteOffset = blockIndex.toULong() * type.byteSize
        val finalScaleByteOffset = dataOffset + blockByteOffset

        val buffer = graphAllocator.buffers[bufferId] as? ByteArray
            ?: throw IllegalStateException("Tensor buffer not found for bufferId $bufferId for tensor '$name'.")

        val scaleBits = io.github.kotlinmania.llama.ore.floatToHalf(scale)
        buffer.setShortLe(finalScaleByteOffset.toInt(), scaleBits)
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
    var objectsBegin: io.github.kotlinmania.llama.ore.GGMLObject? = null,
    var objectsEnd: io.github.kotlinmania.llama.ore.GGMLObject? = null,
    var scratch: io.github.kotlinmania.llama.ore.GGMLScratch = io.github.kotlinmania.llama.ore.GGMLScratch(),
    var scratchSave: io.github.kotlinmania.llama.ore.GGMLScratch = io.github.kotlinmania.llama.ore.GGMLScratch(),
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
    var nodes: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = emptyArray(),
    var grads: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = emptyArray(),
    var gradAccs: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = emptyArray(),
    var leafs: Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = emptyArray(),
    /** Number of uses of each tensor, indexed by hash table slot. */
    var useCounts: IntArray? = null,
    var visitedHashSet: Any? = null,
    var order: io.github.kotlinmania.llama.ore.GGMLCGraphEvalOrder = io.github.kotlinmania.llama.ore.GGMLCGraphEvalOrder.NONE,
    /** An optional identifier to recognize same graphs; 0 means unset. */
    var uid: Long = 0L,
    var allocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator? = null
)

/**
 * Calculates the byte size of a tensor, considering its type and dimensions.
 * For block-quantized types, this calculates the size based on blocks.
 * For regular types, it's num_elements * type_byte_size.
 * For COUNT type or tensors with 0 elements, it's 0.
 */
internal fun calculateTensorByteSize(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong {
    val numElements = tensor.numElements().toULong()

    // If a tensor has zero elements (e.g. ne = [0, ...]), its byte size is 0.
    // Also, GGMLType.COUNT is defined with byteSize = 0uL, so it also results in 0.
    if (numElements == 0uL) {
        return 0uL
    }
    // For types like GGMLType.COUNT, type.byteSize is 0, which correctly yields 0.
    if (tensor.type.byteSize == 0uL && tensor.type != io.github.kotlinmania.llama.ore.GGMLType.COUNT) {
        // This case indicates an issue with a new/custom type definition if it has elements but no byteSize.
        // Standard block types have non-zero byteSize (representing block size).
        println("Warning: Tensor ${tensor.name} of type ${tensor.type} has $numElements elements but type.byteSize is 0. Effective byte size will be 0.")
        return 0uL
    }

    return when (tensor.type) {
        // Explicitly list block-quantized types. Their type.byteSize is "bytes per block".
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K -> {
            // These constants should be defined in GGMLTypes.kt or accessible.
            val elementsPerBlock = when(tensor.type) {
                io.github.kotlinmania.llama.ore.GGMLType.Q4_0 -> io.github.kotlinmania.llama.ore.QK4_0.toULong()
                io.github.kotlinmania.llama.ore.GGMLType.Q4_1 -> io.github.kotlinmania.llama.ore.QK4_1.toULong()
                io.github.kotlinmania.llama.ore.GGMLType.Q8_0 -> io.github.kotlinmania.llama.ore.QK8_0.toULong()
                io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> io.github.kotlinmania.llama.ore.QK_BITNET_1_58.toULong()
                // K-Quant types all use QK_K as block size
                io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K -> io.github.kotlinmania.llama.ore.QK_K.toULong()
                else -> {
                    // This path should ideally not be reached if the outer 'when' is exhaustive for block types
                    println("Warning: Unhandled block-quantized type ${tensor.type} in calculateTensorByteSize. Assuming elementsPerBlock = 1.")
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
