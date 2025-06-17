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
    // Q4_0 byteSize is per block: sizeof(F16 scale) + (QK4_0/2) * sizeof(I8 weights_packed)
    Q4_0("q4_0", 2uL + (QK4_0 / 2).toULong()),   // 4-bit quantized, 18 bytes per block (2 + 32/2*1)
    // Q4_1 byteSize is per block: 2 * sizeof(F16 scale/min) + (QK4_1/2) * sizeof(I8 weights_packed)
    Q4_1("q4_1", (2uL * SIZE_BYTES.toULong()) + (QK4_1 / 2).toULong()),   // 4-bit quantized: 2*F16 (scale d, min m) + QK4_1/2 bytes for packed weights = 4 + 16 = 20 bytes per block
    Q5_0("q5_0", 0uL),   // 5-bit quantized
    Q5_1("q5_1", 0uL),   // 5-bit quantized with different scaling
    // Q8_0 byteSize is per block: sizeof(Float16 for scale) + QK8_0 * sizeof(Int8 for weights)
    Q8_0("q8_0", 2uL + QK8_0.toULong()),   // 8-bit quantized, 34 bytes per block (2 + 32*1)
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
