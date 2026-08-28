// port-lint: source ggml/src/ggml-cpu/ops.cpp
package io.github.kotlinmania.llama.ore

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.getShortLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setIntLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setLongLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe
import io.github.kotlinmania.llama.ore.getBits
import io.github.kotlinmania.llama.ore.lowBits
import io.github.kotlinmania.llama.ore.simd.GGMLSimd
import io.github.kotlinmania.llama.ore.toUnsignedInt
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.roundToInt
import kotlin.math.sign
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES

/**
 * Kotlin Native port of GGML tensor computation operations.
 * 
 * This module implements the core computational functionality for tensor operations
 * in the GGML system. It provides:
 * 
 * - Low-level tensor computation kernels
 * - Quantized operation implementations  
 * - Matrix multiplication optimizations
 * - Dot product routines for various data types
 * - Memory-efficient destination-based operations
 * 
 * All operations follow the destination-based pattern where results are written
 * directly into pre-allocated tensors to minimize memory allocation overhead.
 * This aligns with GGML's memory management philosophy and enables efficient
 * graph execution.
 * 
 * The implementation supports both standard floating-point operations and
 * optimized quantized computations for memory-efficient inference.
 */

/**
 * Calculate total size of tensor elements using centralized utility.
 * This function has been moved to GGMLTensorUtils to avoid duplication.
 * 
 * @param ne Array of tensor dimensions
 * @return Total number of elements
 * @deprecated Use GGMLTensorUtils.calculateTotalSize() instead
 */
@Deprecated(
    message = "Use GGMLTensorUtils.calculateTotalSize() instead",
    replaceWith = ReplaceWith("GGMLTensorUtils.calculateTotalSize(ne)")
)
fun calculateTotalSize(ne: LongArray): Int {
    return io.github.kotlinmania.llama.ore.GGMLTensorUtils.calculateTotalSize(ne).toInt()
}

internal object GGMLDebugFlags {
    /** When true, invoke [q80F32Logger] for each Q8_0 × F32 dot product element. */
    var logQ80F32DotProducts: Boolean = false
    /** Optional callback used by tests/tools to capture debugging information. */
    var q80F32Logger: ((row: Int, col: Int, k: Int, blockIndex: Int, scale: Float, weight: Byte, f32: Float) -> Unit)? = null
}

/**
 * Computes the dot product of a row from a Q8_0 tensor and a column from an F32 tensor.
 * 
 * This function is a core building block for Q8_0 × F32 matrix multiplication operations.
 * It efficiently computes the dot product by:
 * 1. Extracting quantized weights from Q8_0 blocks
 * 2. Dequantizing using the block scale factor
 * 3. Multiplying with corresponding F32 values
 * 4. Accumulating the result
 * 
 * Memory layout assumptions:
 * - tensorQ80: M × K tensor where ne[0]=K, ne[1]=M (row-major storage)
 * - tensorF32: K × N tensor where ne[0]=N, ne[1]=K (column-major access)
 * 
 * @param graphAllocator Memory allocator for accessing tensor data
 * @param tensorQ80 Source Q8_0 quantized tensor (M × K dimensions)
 * @param tensorF32 Source F32 tensor (K × N dimensions)
 * @param rowIndexInQ80 Row index in Q8_0 tensor (0 to M-1)
 * @param colIndexInF32 Column index in F32 tensor (0 to N-1)
 * @param commonDimK Shared dimension size (must match tensorQ80.ne[0] and tensorF32.ne[1])
 * @return Computed dot product as Float
 * @throws IllegalArgumentException if tensor types don't match or dimensions are inconsistent
 */
internal fun computeDotProductQ80F32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ80: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K items per row, ne[1]=M rows)
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N items per row, ne[1]=K rows)
    rowIndexInQ80: Int,     // Row index 'i' for tensorQ80 (0 to M-1)
    colIndexInF32: Int,     // Column index 'j' for tensorF32 (0 to N-1)
    commonDimK: Int         // The shared dimension K (should be tensorQ80.ne[0] and tensorF32.ne[1])
): Float {
    require(tensorQ80.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "tensorQ80 must be Q8_0. Got ${tensorQ80.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ80.ne[0].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val flatIndexInQ80 = rowIndexInQ80 * commonDimK + k
        val blockIndexQ80 = flatIndexInQ80 / io.github.kotlinmania.llama.ore.QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % io.github.kotlinmania.llama.ore.QK8_0
        val scale = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeight = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)
        val dequantizedQ80Value = scale * qWeight.toFloat()
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
        if (io.github.kotlinmania.llama.ore.GGMLDebugFlags.logQ80F32DotProducts) {
            io.github.kotlinmania.llama.ore.GGMLDebugFlags.q80F32Logger?.invoke(rowIndexInQ80, colIndexInF32, k, blockIndexQ80, scale, qWeight, f32Value)
        }
        sumF32 += dequantizedQ80Value * f32Value
    }
    return sumF32
}

/**
 * Computes the dot product of a row from an F32 tensor and a column from a Q4_K tensor.
 * This is the symmetric version of computeDotProductQ4_KF32 for F32 x Q4_K operations.
 */
internal fun computeDotProductF32Q4_K(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,     // M x K (ne[1]=M rows, ne[0]=K elements per row)
    tensorQ4_K: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[1]=K rows, ne[0]=N elements per row)
    rowIndexInF32: Int,        // Row index for tensorF32 (0 to M-1)
    colIndexInQ4_K: Int,       // Column index for tensorQ4_K (0 to N-1)
    commonDimK: Int            // The shared dimension K
): Float {
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductF32Q4_K: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ4_K.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "computeDotProductF32Q4_K: tensorQ4_K must be Q4_K. Got ${tensorQ4_K.type}" }
    
    val M_f32 = tensorF32.ne[1].toInt()
    val K_f32 = tensorF32.ne[0].toInt()
    val K_q4k = tensorQ4_K.ne[1].toInt()
    val N_q4k = tensorQ4_K.ne[0].toInt()
    
    require(K_f32 == commonDimK) { "tensorF32's fastest dim (ne[0]) $K_f32 must match commonDimK $commonDimK" }
    require(K_q4k == commonDimK) { "tensorQ4_K's second dim (ne[1]) $K_q4k must match commonDimK $commonDimK" }
    require(rowIndexInF32 < M_f32) { "rowIndexInF32 $rowIndexInF32 out of bounds for M $M_f32" }
    require(colIndexInQ4_K < N_q4k) { "colIndexInQ4_K $colIndexInQ4_K out of bounds for N $N_q4k" }

    var sumF32 = 0.0f
    
    // Process in Q4_K blocks (QK_K elements per block)
    val numBlocks = (commonDimK + io.github.kotlinmania.llama.ore.QK_K - 1) / io.github.kotlinmania.llama.ore.QK_K
    
    for (blockIdx in 0 until numBlocks) {
        val blockStart = blockIdx * io.github.kotlinmania.llama.ore.QK_K
        val blockEnd = minOf(blockStart + io.github.kotlinmania.llama.ore.QK_K, commonDimK)
        
        // Get Q4_K block scales
        val d = tensorQ4_K.getQ4_KBlockScale(graphAllocator, blockIdx)
        val dmin = tensorQ4_K.getQ4_KBlockScaleMin(graphAllocator, blockIdx)
        
        val buffer = graphAllocator.buffers[tensorQ4_K.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        val blockByteOffset = blockIdx * tensorQ4_K.type.byteSize.toInt()
        
        // Process in 32-element sub-blocks (8 sub-blocks per Q4_K block)
        for (subBlock in 0 until 8) {
            val subBlockStart = blockStart + subBlock * 32
            val subBlockEnd = minOf(subBlockStart + 32, blockEnd)
            
            if (subBlockStart >= commonDimK) break
            
            // Get sub-block quantized scale and min
            val scaleByte = buffer[(tensorQ4_K.dataOffset + blockByteOffset.toULong() + 4uL + subBlock.toULong()).toInt()]
            val quantizedScale = scaleByte.toUnsignedInt().lowBits(6)
            val quantizedMinLow = scaleByte.getBits(6, 2)
            
            val minByteOffset = blockByteOffset + 4 + subBlock * 2 + 1
            val bufferIndex = (tensorQ4_K.dataOffset + minByteOffset.toULong()).toInt()
            val quantizedMinHigh = if (bufferIndex >= 0 && bufferIndex < buffer.size) {
                buffer[bufferIndex].getBits(0, 4)
            } else 0
            val quantizedMin = io.github.kotlinmania.llama.ore.mergeBits32(
                quantizedMinLow,
                io.github.kotlinmania.llama.ore.logicalLeft32(quantizedMinHigh, 2)
            )
            
            // Reconstruct sub-block scale and min
            val scale = (quantizedScale.toFloat() / 63.0f) * d
            val min = (quantizedMin.toFloat() / 63.0f) * d + dmin
            
            // Dot product for this sub-block
            val qsBaseOffset = blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + subBlock * 16
            
            for (i in 0 until (subBlockEnd - subBlockStart) step 2) {
                val k1 = subBlockStart + i
                val k2 = subBlockStart + i + 1
                
                if (k1 < commonDimK) {
                    val qsByte = buffer[(tensorQ4_K.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                    val q1 = qsByte.getBits(0, 4)
                    val q2 = qsByte.getBits(4, 4)
                    val f32Value1 = tensorF32.getFloat(graphAllocator, rowIndexInF32, k1)
                    val dequantizedQ4K1 = (q1.toFloat() / 15.0f) * scale + min
                    sumF32 += f32Value1 * dequantizedQ4K1
                    
                    if (k2 < commonDimK) {
                        val f32Value2 = tensorF32.getFloat(graphAllocator, rowIndexInF32, k2)
                        val dequantizedQ4K2 = (q2.toFloat() / 15.0f) * scale + min
                        sumF32 += f32Value2 * dequantizedQ4K2
                    }
                }
            }
        }
    }
    
    return sumF32
}

/**
 * Computes the dot product of a row from a Q4_K tensor and a column from another Q4_K tensor.
 * This enables direct Q4_K x Q4_K matrix multiplication without intermediate dequantization.
 */
internal fun computeDotProductQ4_KQ4_K(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ4_KA: io.github.kotlinmania.llama.ore.GGMLTensor,   // M x K (ne[1]=M rows, ne[0]=K elements per row)
    tensorQ4_KB: io.github.kotlinmania.llama.ore.GGMLTensor,   // K x N (ne[1]=K rows, ne[0]=N elements per row)
    rowIndexInA: Int,          // Row index for tensorQ4_KA (0 to M-1)
    colIndexInB: Int,          // Column index for tensorQ4_KB (0 to N-1)
    commonDimK: Int            // The shared dimension K
): Float {
    require(tensorQ4_KA.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "computeDotProductQ4_KQ4_K: tensorQ4_KA must be Q4_K. Got ${tensorQ4_KA.type}" }
    require(tensorQ4_KB.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "computeDotProductQ4_KQ4_K: tensorQ4_KB must be Q4_K. Got ${tensorQ4_KB.type}" }
    
    val M_a = tensorQ4_KA.ne[1].toInt()
    val K_a = tensorQ4_KA.ne[0].toInt()
    val K_b = tensorQ4_KB.ne[1].toInt()
    val N_b = tensorQ4_KB.ne[0].toInt()
    
    require(K_a == commonDimK) { "tensorQ4_KA's fastest dim (ne[0]) $K_a must match commonDimK $commonDimK" }
    require(K_b == commonDimK) { "tensorQ4_KB's second dim (ne[1]) $K_b must match commonDimK $commonDimK" }
    require(rowIndexInA < M_a) { "rowIndexInA $rowIndexInA out of bounds for M $M_a" }
    require(colIndexInB < N_b) { "colIndexInB $colIndexInB out of bounds for N $N_b" }

    var sumF32 = 0.0f
    
    // Process in Q4_K blocks (QK_K elements per block)  
    val numBlocks = (commonDimK + io.github.kotlinmania.llama.ore.QK_K - 1) / io.github.kotlinmania.llama.ore.QK_K
    
    for (blockIdx in 0 until numBlocks) {
        val blockStart = blockIdx * io.github.kotlinmania.llama.ore.QK_K
        val blockEnd = minOf(blockStart + io.github.kotlinmania.llama.ore.QK_K, commonDimK)
        
        // Get scales for both tensors
        val dA = tensorQ4_KA.getQ4_KBlockScale(graphAllocator, blockIdx)
        val dminA = tensorQ4_KA.getQ4_KBlockScaleMin(graphAllocator, blockIdx)
        val dB = tensorQ4_KB.getQ4_KBlockScale(graphAllocator, blockIdx)
        val dminB = tensorQ4_KB.getQ4_KBlockScaleMin(graphAllocator, blockIdx)
        
        val bufferA = graphAllocator.buffers[tensorQ4_KA.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor A buffer not found")
        val bufferB = graphAllocator.buffers[tensorQ4_KB.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor B buffer not found")
        val blockByteOffsetA = blockIdx * tensorQ4_KA.type.byteSize.toInt()
        val blockByteOffsetB = blockIdx * tensorQ4_KB.type.byteSize.toInt()
        
        // Process in 32-element sub-blocks
        for (subBlock in 0 until 8) {
            val subBlockStart = blockStart + subBlock * 32
            val subBlockEnd = minOf(subBlockStart + 32, blockEnd)
            
            if (subBlockStart >= commonDimK) break
            
            // Get sub-block scales and mins for both tensors
            // Tensor A
            val scaleByteA = bufferA[(tensorQ4_KA.dataOffset + blockByteOffsetA.toULong() + 4uL + subBlock.toULong()).toInt()]
            val scaleUnsignedA = io.github.kotlinmania.llama.ore.unsignedByte(scaleByteA)
            val quantizedScaleA = io.github.kotlinmania.llama.ore.maskLowBits32(scaleUnsignedA, 6)
            val quantizedMinLowA = io.github.kotlinmania.llama.ore.extractBits(scaleUnsignedA, 6, 2)
            val minByteOffsetA = blockByteOffsetA + 4 + subBlock * 2
            val quantizedMinHighA = if (minByteOffsetA < blockByteOffsetA + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE) {
                io.github.kotlinmania.llama.ore.lowNibble(bufferA[(tensorQ4_KA.dataOffset + minByteOffsetA.toULong() + 1uL).toInt()].toInt())
            } else 0
            val quantizedMinA = io.github.kotlinmania.llama.ore.mergeBits32(
                quantizedMinLowA,
                io.github.kotlinmania.llama.ore.logicalLeft32(quantizedMinHighA, 2)
            )
            val scaleA = (quantizedScaleA.toFloat() / 63.0f) * dA
            val minA = (quantizedMinA.toFloat() / 63.0f) * dA + dminA

            // Tensor B  
            val scaleByteB = bufferB[(tensorQ4_KB.dataOffset + blockByteOffsetB.toULong() + 4uL + subBlock.toULong()).toInt()]
            val scaleUnsignedB = io.github.kotlinmania.llama.ore.unsignedByte(scaleByteB)
            val quantizedScaleB = io.github.kotlinmania.llama.ore.maskLowBits32(scaleUnsignedB, 6)
            val quantizedMinLowB = io.github.kotlinmania.llama.ore.extractBits(scaleUnsignedB, 6, 2)
            val minByteOffsetB = blockByteOffsetB + 4 + subBlock * 2
            val quantizedMinHighB = if (minByteOffsetB < blockByteOffsetB + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE) {
                io.github.kotlinmania.llama.ore.lowNibble(bufferB[(tensorQ4_KB.dataOffset + minByteOffsetB.toULong() + 1uL).toInt()].toInt())
            } else 0
            val quantizedMinB = io.github.kotlinmania.llama.ore.mergeBits32(
                quantizedMinLowB,
                io.github.kotlinmania.llama.ore.logicalLeft32(quantizedMinHighB, 2)
            )
            val scaleB = (quantizedScaleB.toFloat() / 63.0f) * dB
            val minB = (quantizedMinB.toFloat() / 63.0f) * dB + dminB
            
            // Dot product for this sub-block
            val qsBaseOffsetA = blockByteOffsetA + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + subBlock * 16
            val qsBaseOffsetB = blockByteOffsetB + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + subBlock * 16
            
            for (i in 0 until (subBlockEnd - subBlockStart) step 2) {
                val k1 = subBlockStart + i
                val k2 = subBlockStart + i + 1
                
                if (k1 < commonDimK) {
                    // Get quantized values from both tensors
                    val qsByteA = bufferA[(tensorQ4_KA.dataOffset + qsBaseOffsetA.toULong() + (i / 2).toULong()).toInt()]
                    val qsByteB = bufferB[(tensorQ4_KB.dataOffset + qsBaseOffsetB.toULong() + (i / 2).toULong()).toInt()]

                    val qA1 = qsByteA.getBits(0, 4)
                    val qB1 = qsByteB.getBits(0, 4)
                    val dequantizedA1 = (qA1.toFloat() / 15.0f) * scaleA + minA
                    val dequantizedB1 = (qB1.toFloat() / 15.0f) * scaleB + minB
                    sumF32 += dequantizedA1 * dequantizedB1

                    if (k2 < commonDimK) {
                        val qA2 = qsByteA.getBits(4, 4)
                        val qB2 = qsByteB.getBits(4, 4)
                        val dequantizedA2 = (qA2.toFloat() / 15.0f) * scaleA + minA
                        val dequantizedB2 = (qB2.toFloat() / 15.0f) * scaleB + minB
                        sumF32 += dequantizedA2 * dequantizedB2
                    }
                }
            }
        }
    }
    
    return sumF32
}

internal fun computeDotProductQ41F32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ41: io.github.kotlinmania.llama.ore.GGMLTensor,    // Assumed layout M x K (ne[1] = M rows, ne[0] = K elements per row for access)
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,    // Assumed layout K x N (ne[1] = K rows, ne[0] = N columns for access)
    rowIndexInQ41: Int,     // Row index 'i' for tensorQ41 (0 to M-1)
    colIndexInF32: Int,     // Column index 'j' for tensorF32 (0 to N-1)
    commonDimK: Int         // The shared dimension K, should match tensorQ41.ne[0] and tensorF32.ne[1]
): Float {
    require(tensorQ41.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "computeDotProductQ41F32: tensorQ41 must be Q4_1. Got ${tensorQ41.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductQ41F32: tensorF32 must be F32. Got ${tensorF32.type}" }

    // Validate dimensions for clarity and robustness
    val M_q41 = tensorQ41.ne[1].toInt()
    val K_q41 = tensorQ41.ne[0].toInt()
    val K_f32 = tensorF32.ne[1].toInt()
    val N_f32 = tensorF32.ne[0].toInt()

    require(K_q41 == commonDimK) { "tensorQ41's fastest dim (ne[0]) $K_q41 must match commonDimK $commonDimK" }
    require(K_f32 == commonDimK) { "tensorF32's second dim (ne[1]) $K_f32 must match commonDimK $commonDimK for KxN layout" }
    require(rowIndexInQ41 < M_q41) { "rowIndexInQ41 $rowIndexInQ41 out of bounds for M $M_q41" }
    require(colIndexInF32 < N_f32) { "colIndexInF32 $colIndexInF32 out of bounds for N $N_f32" }

    var sumF32 = 0.0f

    for (k in 0 until commonDimK) {
        // Access element tensorQ41[rowIndexInQ41, k]
        // Calculate flat index for Q4_1 tensor elements.
        // tensorQ41.ne[0] is K (elements per row).
        val flatIndexInQ41 = rowIndexInQ41 * K_q41 + k
        val blockIndexQ41 = flatIndexInQ41 / io.github.kotlinmania.llama.ore.QK4_1
        val itemInBlockQ41 = flatIndexInQ41 % io.github.kotlinmania.llama.ore.QK4_1

        val scaleD = tensorQ41.getQ4_1BlockScale(graphAllocator, blockIndexQ41)
        val minM = tensorQ41.getQ4_1BlockMin(graphAllocator, blockIndexQ41)
        val qNibble = tensorQ41.getQ4_1NibbleWeight(graphAllocator, blockIndexQ41, itemInBlockQ41) // Returns raw nibble 0-15
        val dequantizedQ41Value = scaleD * qNibble.toFloat() + minM // Dequantize Q4_1

        // Access element tensorF32[k, colIndexInF32]
        // tensorF32 is K rows (ne[1]) x N columns (ne[0]).
        // GGMLTensor.getFloat expects (idx_dim0 which is column, idx_dim1 which is row, ...)
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)

        sumF32 += dequantizedQ41Value * f32Value
    }
    return sumF32
}

/**
 * Computes the dot product of a row from a Q4_0 tensor and a column from an F32 tensor.
 */
internal fun computeDotProductQ40F32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ40: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ40: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ40.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0) { "computeDotProductQ40F32: tensorQ40 must be Q4_0. Got ${tensorQ40.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductQ40F32: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ40.ne[0].toInt() == commonDimK) { "tensorQ40 K dim (${tensorQ40.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val flatIndexInQ40 = rowIndexInQ40 * commonDimK + k
        val blockIndexQ40 = flatIndexInQ40 / io.github.kotlinmania.llama.ore.QK4_0
        val itemInBlockQ40 = flatIndexInQ40 % io.github.kotlinmania.llama.ore.QK4_0
        val scale = tensorQ40.getQ4_0BlockScale(graphAllocator, blockIndexQ40)
        val qNibble = tensorQ40.getQ4_0NibbleWeight(graphAllocator, blockIndexQ40, itemInBlockQ40)
        val dequantizedQ40Value = scale * (qNibble.toFloat() - 8.0f)
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
        sumF32 += dequantizedQ40Value * f32Value
    }
    return sumF32
}

// K-Quant Optimized Dot Product Functions

/**
 * Computes the dot product of a row from a Q2_K tensor and a column from an F32 tensor.
 */
internal fun computeDotProductQ2_KF32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ2_K: io.github.kotlinmania.llama.ore.GGMLTensor,
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,
    rowIndexInQ2_K: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ2_K.type == io.github.kotlinmania.llama.ore.GGMLType.Q2_K) { "tensorQ2_K must be Q2_K. Got ${tensorQ2_K.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ2_K.ne[0].toInt() == commonDimK) { "tensorQ2_K K dim (${tensorQ2_K.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    
    // Process in blocks of QK_K
    for (blockStart in 0 until commonDimK step io.github.kotlinmania.llama.ore.QK_K) {
        val blockEnd = minOf(blockStart + io.github.kotlinmania.llama.ore.QK_K, commonDimK)
        val blockSize = blockEnd - blockStart
        
        if (blockSize == io.github.kotlinmania.llama.ore.QK_K) {
            // Full block optimization
            val flatIndexStart = rowIndexInQ2_K * commonDimK + blockStart
            val blockIndex = flatIndexStart / io.github.kotlinmania.llama.ore.QK_K
            
            val d = tensorQ2_K.getQ2_KBlockScale(graphAllocator, blockIndex)
            val dmin = tensorQ2_K.getQ2_KBlockScaleMin(graphAllocator, blockIndex)
            
            // Process sub-blocks
            for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K /16) {
                val scaleAndMin = tensorQ2_K.getQ2_KScale(graphAllocator, blockIndex, subBlock)
                val scaleAndMinUnsigned =
                    io.github.kotlinmania.llama.ore.maskLowBits32(scaleAndMin.toInt(), 8)
                val quantizedScale =
                    io.github.kotlinmania.llama.ore.maskLowBits32(scaleAndMinUnsigned, 4)
                val quantizedMin =
                    io.github.kotlinmania.llama.ore.extractBits(scaleAndMinUnsigned, 4, 4)
                
                val scale = (quantizedScale.toFloat() / 15.0f) * d
                val min = (quantizedMin.toFloat() * d) + dmin
                
                // Process 16 values in this sub-block
                for (i in 0 until 16 step 4) {
                    val quantByte = tensorQ2_K.getQ2_KQuant(graphAllocator, blockIndex, subBlock * 4 + i / 4)
                    
                    for (j in 0 until 4) {
                        val k = blockStart + subBlock * 16 + i + j
                        if (k < blockEnd) {
                            val quantizedValue = io.github.kotlinmania.llama.ore.extractBits(
                                io.github.kotlinmania.llama.ore.unsignedByte(quantByte), j * 2, 2
                            )
                            val dequantizedValue = (quantizedValue.toFloat() / 3.0f) * scale + min
                            val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
                            sumF32 += dequantizedValue * f32Value
                        }
                    }
                }
            }
        } else {
            // Handle partial blocks by dequantizing (fallback)
            for (k in blockStart until blockEnd) {
                val flatIndex = rowIndexInQ2_K * commonDimK + k
                val blockIndex = flatIndex / io.github.kotlinmania.llama.ore.QK_K
                val itemInBlock = flatIndex % io.github.kotlinmania.llama.ore.QK_K
                
                // Simplified dequantization for partial blocks
                val d = tensorQ2_K.getQ2_KBlockScale(graphAllocator, blockIndex)
                val dmin = tensorQ2_K.getQ2_KBlockScaleMin(graphAllocator, blockIndex)
                
                val subBlock = itemInBlock / 16
                val scaleAndMin = tensorQ2_K.getQ2_KScale(graphAllocator, blockIndex, subBlock)
                val scaleAndMinUnsigned =
                    io.github.kotlinmania.llama.ore.maskLowBits32(scaleAndMin.toInt(), 8)
                val quantizedScale =
                    io.github.kotlinmania.llama.ore.maskLowBits32(scaleAndMinUnsigned, 4)
                val quantizedMin =
                    io.github.kotlinmania.llama.ore.extractBits(scaleAndMinUnsigned, 4, 4)
                
                val scale = (quantizedScale.toFloat() / 15.0f) * d
                val min = (quantizedMin.toFloat() * d) + dmin
                
                val quantByteIdx = (subBlock * 4) + ((itemInBlock % 16) / 4)
                val quantByte = tensorQ2_K.getQ2_KQuant(graphAllocator, blockIndex, quantByteIdx)
                val bitPos = ((itemInBlock % 16) % 4) * 2
                val quantizedValue = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(quantByte), bitPos, 2
                )
                
                val dequantizedValue = (quantizedValue.toFloat() / 3.0f) * scale + min
                val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
                sumF32 += dequantizedValue * f32Value
            }
        }
    }
    
    return sumF32
}

/**
 * Computes the dot product of a row from a Q4_K tensor and a column from an F32 tensor.
 */
internal fun computeDotProductQ4_KF32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ4_K: io.github.kotlinmania.llama.ore.GGMLTensor,
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,
    rowIndexInQ4_K: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ4_K.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) { "tensorQ4_K must be Q4_K. Got ${tensorQ4_K.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ4_K.ne[0].toInt() == commonDimK) { "tensorQ4_K K dim (${tensorQ4_K.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    
    // Process in blocks of QK_K
    for (blockStart in 0 until commonDimK step io.github.kotlinmania.llama.ore.QK_K) {
        val blockEnd = minOf(blockStart + io.github.kotlinmania.llama.ore.QK_K, commonDimK)
        val blockSize = blockEnd - blockStart
        
        if (blockSize == io.github.kotlinmania.llama.ore.QK_K) {
            val flatIndexStart = rowIndexInQ4_K * commonDimK + blockStart
            val blockIndex = flatIndexStart / io.github.kotlinmania.llama.ore.QK_K
            
            val d = tensorQ4_K.getQ4_KBlockScale(graphAllocator, blockIndex)
            val dmin = tensorQ4_K.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
            
            val buffer = graphAllocator.buffers[tensorQ4_K.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
            val blockByteOffset = blockIndex * tensorQ4_K.type.byteSize.toInt()
            
            // Process 8 sub-blocks of 32 elements each
            for (subBlock in 0 until 8) {
                // Get quantized scale and min for this sub-block
                val scaleByteOffset = blockByteOffset + 4 + subBlock
                val scaleByte = buffer[(tensorQ4_K.dataOffset + scaleByteOffset.toULong()).toInt()]
                val quantizedScale = scaleByte.toUnsignedInt().lowBits(6)
                val quantizedMinLow =
                    io.github.kotlinmania.llama.ore.extractBits(scaleByte.toUnsignedInt(), 6, 2)
                
                val minByteOffset = blockByteOffset + 4 + subBlock * 2 + 1
                val quantizedMinHigh = if (minByteOffset < blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE) {
                    io.github.kotlinmania.llama.ore.maskLowBits32(
                        io.github.kotlinmania.llama.ore.unsignedByte(
                            buffer[(tensorQ4_K.dataOffset + minByteOffset.toULong()).toInt()]
                        ), 4
                    )
                } else 0
                val quantizedMin = io.github.kotlinmania.llama.ore.mergeBits32(
                    quantizedMinLow,
                    io.github.kotlinmania.llama.ore.logicalLeft32(quantizedMinHigh, 2)
                )
                
                val scale = (quantizedScale.toFloat() / 63.0f) * d
                val min = (quantizedMin.toFloat() / 63.0f) * d + dmin
                
                // Process 32 4-bit values in this sub-block
                val qsBaseOffset = blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + subBlock * 16
                for (i in 0 until 32 step 2) {
                    val k1 = blockStart + subBlock * 32 + i
                    val k2 = blockStart + subBlock * 32 + i + 1
                    
                    if (k1 < blockEnd) {
                        val qsByte = buffer[(tensorQ4_K.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                        
                        val q1 = qsByte.getBits(0, 4)
                        val dequantizedValue1 = (q1.toFloat() / 15.0f) * scale + min
                        val f32Value1 = tensorF32.getFloat(graphAllocator, colIndexInF32, k1)
                        sumF32 += dequantizedValue1 * f32Value1
                        
                        if (k2 < blockEnd) {
                            val q2 = qsByte.getBits(4, 4)
                            val dequantizedValue2 = (q2.toFloat() / 15.0f) * scale + min
                            val f32Value2 = tensorF32.getFloat(graphAllocator, colIndexInF32, k2)
                            sumF32 += dequantizedValue2 * f32Value2
                        }
                    }
                }
            }
        } else {
            // Fallback for partial blocks
            for (k in blockStart until blockEnd) {
                val flatIndex = rowIndexInQ4_K * commonDimK + k
                val blockIndex = flatIndex / io.github.kotlinmania.llama.ore.QK_K
                val itemInBlock = flatIndex % io.github.kotlinmania.llama.ore.QK_K
                
                val d = tensorQ4_K.getQ4_KBlockScale(graphAllocator, blockIndex)
                val dmin = tensorQ4_K.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
                
                // Simplified dequantization
                val subBlock = itemInBlock / 32
                val buffer = graphAllocator.buffers[tensorQ4_K.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
                val blockByteOffset = blockIndex * tensorQ4_K.type.byteSize.toInt()
                
                val scaleByteOffset = blockByteOffset + 4 + subBlock
                val scaleByte = buffer[(tensorQ4_K.dataOffset + scaleByteOffset.toULong()).toInt()]
                val quantizedScale = scaleByte.toUnsignedInt().lowBits(6)
                val scale = (quantizedScale.toFloat() / 63.0f) * d
                
                val qsOffset = blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + subBlock * 16 + ((itemInBlock % 32) / 2)
                val qsByte = buffer[(tensorQ4_K.dataOffset + qsOffset.toULong()).toInt()]
                val q = if ((itemInBlock % 32) % 2 == 0) qsByte.getBits(0, 4) else qsByte.getBits(4, 4)
                
                val dequantizedValue = (q.toFloat() / 15.0f) * scale + dmin
                val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
                sumF32 += dequantizedValue * f32Value
            }
        }
    }
    
    return sumF32
}
    
/**
 * Computes the symmetric dot product of a row from an F32 tensor and a column from a Q4_1 tensor.
 * This is the symmetric case: F32 x Q4_1
 */
internal fun computeDotProductF32Q41(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ41: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInF32: Int,
    colIndexInQ41: Int,
    commonDimK: Int
): Float {
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductF32Q41: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ41.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "computeDotProductF32Q41: tensorQ41 must be Q4_1. Got ${tensorQ41.type}" }
    require(tensorF32.ne[0].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ41.ne[1].toInt() == commonDimK) { "tensorQ41 K dim (${tensorQ41.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val f32Value = tensorF32.getFloat(graphAllocator, k, rowIndexInF32)
        // Access tensorQ41[k, colIndexInQ41] - row k, column colIndexInQ41
        val flatIndexInQ41 = k * tensorQ41.ne[0].toInt() + colIndexInQ41
        val blockIndexQ41 = flatIndexInQ41 / io.github.kotlinmania.llama.ore.QK4_1
        val itemInBlockQ41 = flatIndexInQ41 % io.github.kotlinmania.llama.ore.QK4_1
        
        val scaleD = tensorQ41.getQ4_1BlockScale(graphAllocator, blockIndexQ41)
        val minM = tensorQ41.getQ4_1BlockMin(graphAllocator, blockIndexQ41)
        val qNibble = tensorQ41.getQ4_1NibbleWeight(graphAllocator, blockIndexQ41, itemInBlockQ41)
    val dequantizedQ41Value = scaleD * qNibble.toFloat() + minM
    sumF32 += f32Value * dequantizedQ41Value
    }
    return sumF32
}

/**
 * Computes the dot product of a row from a Q8_K tensor and a column from an F32 tensor.
 */
internal fun computeDotProductQ8_KF32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ8_K: io.github.kotlinmania.llama.ore.GGMLTensor,
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,
    rowIndexInQ8_K: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ8_K.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_K) { "tensorQ8_K must be Q8_K. Got ${tensorQ8_K.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ8_K.ne[0].toInt() == commonDimK) { "tensorQ8_K K dim (${tensorQ8_K.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    
    // Process in blocks of QK_K (simpler for Q8_K)
    for (blockStart in 0 until commonDimK step io.github.kotlinmania.llama.ore.QK_K) {
        val blockEnd = minOf(blockStart + io.github.kotlinmania.llama.ore.QK_K, commonDimK)
        
        if (blockEnd - blockStart == io.github.kotlinmania.llama.ore.QK_K) {
            val flatIndexStart = rowIndexInQ8_K * commonDimK + blockStart
            val blockIndex = flatIndexStart / io.github.kotlinmania.llama.ore.QK_K
            
            val d = tensorQ8_K.getQ8_KBlockScale(graphAllocator, blockIndex)
            
            // Simple dot product for Q8_K block
            for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
                val k = blockStart + i
                val quantizedValue = tensorQ8_K.getQ8_KWeight(graphAllocator, blockIndex, i)
                val dequantizedValue = quantizedValue.toFloat() * d
                val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
                sumF32 += dequantizedValue * f32Value
            }
        } else {
            // Fallback for partial blocks
            for (k in blockStart until blockEnd) {
                val flatIndex = rowIndexInQ8_K * commonDimK + k
                val blockIndex = flatIndex / io.github.kotlinmania.llama.ore.QK_K
                val itemInBlock = flatIndex % io.github.kotlinmania.llama.ore.QK_K
                
                val d = tensorQ8_K.getQ8_KBlockScale(graphAllocator, blockIndex)
                val quantizedValue = tensorQ8_K.getQ8_KWeight(graphAllocator, blockIndex, itemInBlock)
                val dequantizedValue = quantizedValue.toFloat() * d
                val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
                sumF32 += dequantizedValue * f32Value
            }
        }
    }
    
    return sumF32
}
    
/**
 * Computes the symmetric dot product of a row from an F32 tensor and a column from a Q8_0 tensor.
 * This is the symmetric case: F32 x Q8_0
 */
/**
 * Computes the symmetric dot product of a row from an F32 tensor and a column from a Q8_0 tensor.
 * This is the symmetric case: F32 x Q8_0
 */
internal fun computeDotProductF32Q80(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ80: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInF32: Int,
    colIndexInQ80: Int,
    commonDimK: Int
): Float {
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductF32Q80: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ80.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "computeDotProductF32Q80: tensorQ80 must be Q8_0. Got ${tensorQ80.type}" }
    require(tensorF32.ne[0].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ80.ne[1].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val f32Value = tensorF32.getFloat(graphAllocator, k, rowIndexInF32)
        // Access tensorQ80[k, colIndexInQ80] - row k, column colIndexInQ80
        val flatIndexInQ80 = k * tensorQ80.ne[0].toInt() + colIndexInQ80
        val blockIndexQ80 = flatIndexInQ80 / io.github.kotlinmania.llama.ore.QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % io.github.kotlinmania.llama.ore.QK8_0
        val scale = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeight = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)
        val dequantizedQ80Value = scale * qWeight.toFloat()
        sumF32 += f32Value * dequantizedQ80Value
    }
    return sumF32
}

/**
 * Computes the dot product of a row from an F32 tensor and a column from another F32 tensor.
 */
internal fun computeDotProductF32F32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorF32A: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorF32B: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInF32A: Int,
    colIndexInF32B: Int,
    commonDimK: Int
): Float {
    require(tensorF32A.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductF32F32: tensorF32A must be F32. Got ${tensorF32A.type}" }
    require(tensorF32B.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "computeDotProductF32F32: tensorF32B must be F32. Got ${tensorF32B.type}" }
    require(tensorF32A.ne[0].toInt() == commonDimK) { "tensorF32A K dim (${tensorF32A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32B.ne[1].toInt() == commonDimK) { "tensorF32B K dim (${tensorF32B.ne[1]}) must match commonDimK ($commonDimK)"}

    val bufferA = graphAllocator.buffers.getOrNull(tensorF32A.bufferId) as? ByteArray
    val bufferB = graphAllocator.buffers.getOrNull(tensorF32B.bufferId) as? ByteArray
    if (bufferA != null && bufferB != null) {
        val strideA = tensorF32A.nb[0].toInt()
        val strideB = tensorF32B.nb[1].toInt()
        if (strideA > 0 && strideB > 0) {
            val baseOffsetA = tensorF32A.dataOffset.toInt() + rowIndexInF32A * tensorF32A.nb[1].toInt()
            val baseOffsetB = tensorF32B.dataOffset.toInt() + colIndexInF32B * tensorF32B.nb[0].toInt()
            return io.github.kotlinmania.llama.ore.simd.GGMLSimd.dotF32(
                bufferA,
                baseOffsetA,
                strideA,
                bufferB,
                baseOffsetB,
                strideB,
                commonDimK
            )
        }
    }

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val f32ValueA = tensorF32A.getFloat(graphAllocator, k, rowIndexInF32A)
        val f32ValueB = tensorF32B.getFloat(graphAllocator, colIndexInF32B, k)
        sumF32 += f32ValueA * f32ValueB
    }
    return sumF32
}

/**
 * Computes the direct quantized dot product Q8_0 x Q8_0 -> F32.
 * This avoids dequantization by computing the dot product directly on quantized values.
 */
internal fun computeDotProductQ80Q80(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ80A: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ80B: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ80A: Int,
    colIndexInQ80B: Int,
    commonDimK: Int
): Float {
    require(tensorQ80A.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "computeDotProductQ80Q80: tensorQ80A must be Q8_0. Got ${tensorQ80A.type}" }
    require(tensorQ80B.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "computeDotProductQ80Q80: tensorQ80B must be Q8_0. Got ${tensorQ80B.type}" }
    require(tensorQ80A.ne[0].toInt() == commonDimK) { "tensorQ80A K dim (${tensorQ80A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ80B.ne[1].toInt() == commonDimK) { "tensorQ80B K dim (${tensorQ80B.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ80A[rowIndexInQ80A, k]
        val flatIndexInQ80A = rowIndexInQ80A * commonDimK + k
        val blockIndexQ80A = flatIndexInQ80A / io.github.kotlinmania.llama.ore.QK8_0
        val itemInBlockQ80A = flatIndexInQ80A % io.github.kotlinmania.llama.ore.QK8_0
        val scaleA = tensorQ80A.getQ8_0BlockScale(graphAllocator, blockIndexQ80A)
        val qWeightA = tensorQ80A.getQ8_0Weight(graphAllocator, blockIndexQ80A, itemInBlockQ80A)

        // Access tensorQ80B[k, colIndexInQ80B]
        val flatIndexInQ80B = k * tensorQ80B.ne[0].toInt() + colIndexInQ80B
        val blockIndexQ80B = flatIndexInQ80B / io.github.kotlinmania.llama.ore.QK8_0
        val itemInBlockQ80B = flatIndexInQ80B % io.github.kotlinmania.llama.ore.QK8_0
        val scaleB = tensorQ80B.getQ8_0BlockScale(graphAllocator, blockIndexQ80B)
        val qWeightB = tensorQ80B.getQ8_0Weight(graphAllocator, blockIndexQ80B, itemInBlockQ80B)

        // Direct quantized multiplication: (scaleA * qWeightA) * (scaleB * qWeightB) = (scaleA * scaleB) * (qWeightA * qWeightB)
        sumF32 += scaleA * scaleB * (qWeightA.toFloat() * qWeightB.toFloat())
    }
    return sumF32
}

/**
 * Computes the direct quantized dot product Q4_0 x Q4_0 -> F32.
 */
internal fun computeDotProductQ40Q40(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ40A: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ40B: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ40A: Int,
    colIndexInQ40B: Int,
    commonDimK: Int
): Float {
    require(tensorQ40A.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0) { "computeDotProductQ40Q40: tensorQ40A must be Q4_0. Got ${tensorQ40A.type}" }
    require(tensorQ40B.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0) { "computeDotProductQ40Q40: tensorQ40B must be Q4_0. Got ${tensorQ40B.type}" }
    require(tensorQ40A.ne[0].toInt() == commonDimK) { "tensorQ40A K dim (${tensorQ40A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ40B.ne[1].toInt() == commonDimK) { "tensorQ40B K dim (${tensorQ40B.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ40A[rowIndexInQ40A, k]
        val flatIndexInQ40A = rowIndexInQ40A * commonDimK + k
        val blockIndexQ40A = flatIndexInQ40A / io.github.kotlinmania.llama.ore.QK4_0
        val itemInBlockQ40A = flatIndexInQ40A % io.github.kotlinmania.llama.ore.QK4_0
        val scaleA = tensorQ40A.getQ4_0BlockScale(graphAllocator, blockIndexQ40A)
        val qNibbleA = tensorQ40A.getQ4_0NibbleWeight(graphAllocator, blockIndexQ40A, itemInBlockQ40A)

        // Access tensorQ40B[k, colIndexInQ40B]
        val flatIndexInQ40B = k * tensorQ40B.ne[0].toInt() + colIndexInQ40B
        val blockIndexQ40B = flatIndexInQ40B / io.github.kotlinmania.llama.ore.QK4_0
        val itemInBlockQ40B = flatIndexInQ40B % io.github.kotlinmania.llama.ore.QK4_0
        val scaleB = tensorQ40B.getQ4_0BlockScale(graphAllocator, blockIndexQ40B)
        val qNibbleB = tensorQ40B.getQ4_0NibbleWeight(graphAllocator, blockIndexQ40B, itemInBlockQ40B)

        // Direct quantized multiplication: Q4_0 values are centered at 8, so (qNibble - 8) * scale
        // (scaleA * (qNibbleA - 8)) * (scaleB * (qNibbleB - 8))
        val dequantA = scaleA * (qNibbleA.toFloat() - 8.0f)
        val dequantB = scaleB * (qNibbleB.toFloat() - 8.0f)
        sumF32 += dequantA * dequantB
    }
    return sumF32
}
/**
 * Computes the direct quantized dot product Q4_1 x Q4_1 -> F32.
 */
internal fun computeDotProductQ41Q41(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ41A: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ41B: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ41A: Int,
    colIndexInQ41B: Int,
    commonDimK: Int
): Float {
    require(tensorQ41A.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "computeDotProductQ41Q41: tensorQ41A must be Q4_1. Got ${tensorQ41A.type}" }
    require(tensorQ41B.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1) { "computeDotProductQ41Q41: tensorQ41B must be Q4_1. Got ${tensorQ41B.type}" }
    require(tensorQ41A.ne[0].toInt() == commonDimK) { "tensorQ41A K dim (${tensorQ41A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ41B.ne[1].toInt() == commonDimK) { "tensorQ41B K dim (${tensorQ41B.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ41A[rowIndexInQ41A, k]
        val flatIndexInQ41A = rowIndexInQ41A * commonDimK + k
        val blockIndexQ41A = flatIndexInQ41A / io.github.kotlinmania.llama.ore.QK4_1
        val itemInBlockQ41A = flatIndexInQ41A % io.github.kotlinmania.llama.ore.QK4_1
        val scaleDA = tensorQ41A.getQ4_1BlockScale(graphAllocator, blockIndexQ41A)
        val minMA = tensorQ41A.getQ4_1BlockMin(graphAllocator, blockIndexQ41A)
        val qNibbleA = tensorQ41A.getQ4_1NibbleWeight(graphAllocator, blockIndexQ41A, itemInBlockQ41A)

        // Access tensorQ41B[k, colIndexInQ41B]
        val flatIndexInQ41B = k * tensorQ41B.ne[0].toInt() + colIndexInQ41B
        val blockIndexQ41B = flatIndexInQ41B / io.github.kotlinmania.llama.ore.QK4_1
        val itemInBlockQ41B = flatIndexInQ41B % io.github.kotlinmania.llama.ore.QK4_1
        val scaleDB = tensorQ41B.getQ4_1BlockScale(graphAllocator, blockIndexQ41B)
        val minMB = tensorQ41B.getQ4_1BlockMin(graphAllocator, blockIndexQ41B)
        val qNibbleB = tensorQ41B.getQ4_1NibbleWeight(graphAllocator, blockIndexQ41B, itemInBlockQ41B)

        // Direct quantized multiplication: Q4_1 values are dequantized as scale * nibble + min
        val dequantA = scaleDA * qNibbleA.toFloat() + minMA
        val dequantB = scaleDB * qNibbleB.toFloat() + minMB
        sumF32 += dequantA * dequantB
    }
    return sumF32
}

/**
 * Computes mixed quantized dot product Q8_0 x Q4_0 -> F32.
 */
internal fun computeDotProductQ80Q40(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorQ80: io.github.kotlinmania.llama.ore.GGMLTensor,     // M x K (ne[0]=K, ne[1]=M)
    tensorQ40: io.github.kotlinmania.llama.ore.GGMLTensor,     // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ80: Int,
    colIndexInQ40: Int,
    commonDimK: Int
): Float {
    require(tensorQ80.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0) { "computeDotProductQ80Q40: tensorQ80 must be Q8_0. Got ${tensorQ80.type}" }
    require(tensorQ40.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0) { "computeDotProductQ80Q40: tensorQ40 must be Q4_0. Got ${tensorQ40.type}" }
    require(tensorQ80.ne[0].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ40.ne[1].toInt() == commonDimK) { "tensorQ40 K dim (${tensorQ40.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ80[rowIndexInQ80, k]
        val flatIndexInQ80 = rowIndexInQ80 * commonDimK + k
        val blockIndexQ80 = flatIndexInQ80 / io.github.kotlinmania.llama.ore.QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % io.github.kotlinmania.llama.ore.QK8_0
        val scaleQ80 = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeightQ80 = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)

        // Access tensorQ40[k, colIndexInQ40]
        val flatIndexInQ40 = k * tensorQ40.ne[0].toInt() + colIndexInQ40
        val blockIndexQ40 = flatIndexInQ40 / io.github.kotlinmania.llama.ore.QK4_0
        val itemInBlockQ40 = flatIndexInQ40 % io.github.kotlinmania.llama.ore.QK4_0
        val scaleQ40 = tensorQ40.getQ4_0BlockScale(graphAllocator, blockIndexQ40)
        val qNibbleQ40 = tensorQ40.getQ4_0NibbleWeight(graphAllocator, blockIndexQ40, itemInBlockQ40)

        // Direct quantized multiplication
        val dequantQ80 = scaleQ80 * qWeightQ80.toFloat()
        val dequantQ40 = scaleQ40 * (qNibbleQ40.toFloat() - 8.0f)
        sumF32 += dequantQ80 * dequantQ40
    }
    return sumF32
}

/**
 * Computes the dot product of a row from a BitNet 1.58 tensor and a column from an F32 tensor.
 * Used as a core part of BitNet 1.58 x F32 matrix multiplication.
 */
internal fun computeDotProductBitNet158F32(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorBitNet: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor,       // K x N (ne[0]=N, ne[1]=K)
    rowIndexInBitNet: Int,       // Row index for tensorBitNet (0 to M-1)
    colIndexInF32: Int,          // Column index for tensorF32 (0 to N-1)
    commonDimK: Int              // The shared dimension K
): Float {
    require(tensorBitNet.type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "tensorBitNet must be BITNET_1_58. Got ${tensorBitNet.type}" }
    require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorBitNet.ne[0].toInt() == commonDimK) { "tensorBitNet K dim (${tensorBitNet.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val flatIndexInBitNet = rowIndexInBitNet * commonDimK + k
        val blockIndexBitNet = flatIndexInBitNet / io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        val itemInBlockBitNet = flatIndexInBitNet % io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        val scale = tensorBitNet.getBitNet158BlockScale(graphAllocator, blockIndexBitNet)
        val ternaryWeight = tensorBitNet.getBitNet158TernaryWeight(graphAllocator, blockIndexBitNet, itemInBlockBitNet)
        val dequantizedBitNetValue = scale * ternaryWeight.toFloat()
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
        sumF32 += dequantizedBitNetValue * f32Value
    }
    return sumF32
}

/**
 * Computes the direct quantized dot product BitNet 1.58 x BitNet 1.58 -> F32.
 */
internal fun computeDotProductBitNet158BitNet158(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    tensorBitNetA: io.github.kotlinmania.llama.ore.GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorBitNetB: io.github.kotlinmania.llama.ore.GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInBitNetA: Int,
    colIndexInBitNetB: Int,
    commonDimK: Int
): Float {
    require(tensorBitNetA.type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "tensorBitNetA must be BITNET_1_58. Got ${tensorBitNetA.type}" }
    require(tensorBitNetB.type == io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58) { "tensorBitNetB must be BITNET_1_58. Got ${tensorBitNetB.type}" }
    require(tensorBitNetA.ne[0].toInt() == commonDimK) { "tensorBitNetA K dim (${tensorBitNetA.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorBitNetB.ne[1].toInt() == commonDimK) { "tensorBitNetB K dim (${tensorBitNetB.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorBitNetA[rowIndexInBitNetA, k]
        val flatIndexInBitNetA = rowIndexInBitNetA * commonDimK + k
        val blockIndexBitNetA = flatIndexInBitNetA / io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        val itemInBlockBitNetA = flatIndexInBitNetA % io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        val scaleA = tensorBitNetA.getBitNet158BlockScale(graphAllocator, blockIndexBitNetA)
        val ternaryWeightA = tensorBitNetA.getBitNet158TernaryWeight(graphAllocator, blockIndexBitNetA, itemInBlockBitNetA)

        // Access tensorBitNetB[k, colIndexInBitNetB]
        val flatIndexInBitNetB = k * tensorBitNetB.ne[0].toInt() + colIndexInBitNetB
        val blockIndexBitNetB = flatIndexInBitNetB / io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        val itemInBlockBitNetB = flatIndexInBitNetB % io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        val scaleB = tensorBitNetB.getBitNet158BlockScale(graphAllocator, blockIndexBitNetB)
        val ternaryWeightB = tensorBitNetB.getBitNet158TernaryWeight(graphAllocator, blockIndexBitNetB, itemInBlockBitNetB)

        // Direct ternary multiplication: (scaleA * ternaryA) * (scaleB * ternaryB)
        sumF32 += scaleA * scaleB * (ternaryWeightA.toFloat() * ternaryWeightB.toFloat())
    }
    return sumF32
}

// Helper to iterate N-dimensionally
internal fun applyNDIter(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, totalSize: Int, actionPerElement: (flatIdx: Int, indices: IntArray) -> Unit) {
    val n0 = tensor.ne[0].toInt(); val n1 = tensor.ne[1].toInt()
    val n2 = tensor.ne[2].toInt(); val n3 = tensor.ne[3].toInt()
    var currentFlatIdx = 0
    if (totalSize == 0) return

    val r = tensor.rank()
    if (r == 0 && totalSize == 1) {
        actionPerElement(currentFlatIdx++, intArrayOf()); return
    }

    for (i3 in 0 until (if (r >= 4) n3 else 1)) {
        for (i2 in 0 until (if (r >= 3) n2 else 1)) {
            for (i1 in 0 until (if (r >= 2) n1 else 1)) {
                for (i0 in 0 until (if (r >= 1) n0 else 1)) {
                    if (currentFlatIdx < totalSize) {
                        val indices = when (r) {
                            0 -> intArrayOf()
                            1 -> intArrayOf(i0)
                            2 -> intArrayOf(i0, i1)
                            3 -> intArrayOf(i0, i1, i2)
                            else -> intArrayOf(i0, i1, i2, i3)
                        }
                        actionPerElement(currentFlatIdx++, indices)
                    } else return
                }
            }
        }
    }
}

/**
 * Adds two tensors element-wise.
 */
fun computeAdd(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    a: io.github.kotlinmania.llama.ore.GGMLTensor,
    b: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for addition")
    }
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()

    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(
                a,
                totalSize
            ) { _, indices -> // Iterate based on 'a' which has same shape as dst
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                dst.setFloat(graphAllocator, v0 + v1, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(
                a,
                totalSize
            ) { _, indices -> // Iterate based on 'a'
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                // Perform addition as Float for precision, then convert back to Half (Short)
                dst.setHalf(graphAllocator, v0 + v1, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                dst.setInt(graphAllocator, valA + valB, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                dst.setShort(
                    graphAllocator,
                    (valA + valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                dst.setByte(
                    graphAllocator,
                    (valA + valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                dst.setLong(graphAllocator, valA + valB, *indices)
            }
    }
    // For quantized types, dequantize, compute, and re-quantize
    io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
    io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
    io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
    io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
    io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
    io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
    io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
    io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            val aF32 = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a)
            val bF32 = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, b)
            val tempF32 =
                io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32); tempF32.ne = dst.ne.copyOf(); tempF32.nb =
            io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                tempF32.ne,
                io.github.kotlinmania.llama.ore.GGMLType.F32,
                tempF32.ne.size
            )
            computeAdd(graphAllocator, aF32, bF32, tempF32)
            val quantizedResult =
                io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tempF32, dst.type)
            // Copy quantized data to destination
            dst.data = quantizedResult.data
        }
        else -> error("fatal error")
    }
}

fun computeMul(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    a: io.github.kotlinmania.llama.ore.GGMLTensor,
    b: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for multiplication")
    }
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()

    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                dst.setFloat(graphAllocator, v0 * v1, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                dst.setHalf(graphAllocator, v0 * v1, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                dst.setInt(graphAllocator, valA * valB, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                dst.setShort(
                    graphAllocator,
                    (valA * valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                dst.setByte(
                    graphAllocator,
                    (valA * valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                dst.setLong(graphAllocator, valA * valB, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS,
        io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> {
            val aF32 = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a)
            val bF32 = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, b)
            val tempF32 =
                io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32); tempF32.ne = dst.ne.copyOf(); tempF32.nb =
                io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                    tempF32.ne,
                    io.github.kotlinmania.llama.ore.GGMLType.F32,
                    tempF32.ne.size
                )
            computeMul(graphAllocator, aF32, bF32, tempF32)
            val quantizedResult =
                io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = quantizedResult.data
        }
        else -> error("fatal error")
    }
}

fun computeRepeatBack(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src.type == dst.type) { "REPEAT_BACK requires matching tensor types" }
    for (d in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        val srcDim = src.ne[d]
        val dstDim = dst.ne[d]
        if (srcDim == dstDim) continue
        if (dstDim == 1L && srcDim >= 1L) continue
        if (srcDim == 0L && dstDim == 0L) continue
        throw IllegalArgumentException("REPEAT_BACK shape mismatch at axis $d: src=${srcDim}, dst=${dstDim}")
    }

    val dstTotal = dst.numElements().toInt()
    if (dstTotal == 0) return

    val dstRank = dst.rank().coerceAtLeast(1)

    when (dst.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(dst, dstTotal) { _, indices ->
                dst.setFloat(
                    graphAllocator,
                    0f,
                    *indices
                )
            }
            val srcTotal = src.numElements().toInt()
            if (srcTotal == 0) return
            io.github.kotlinmania.llama.ore.applyNDIter(src, srcTotal) { _, srcIndices ->
                val target = IntArray(dstRank) { dim ->
                    val dstDimSize = dst.ne[dim].toInt().coerceAtLeast(1)
                    val srcIndexVal = if (dim < srcIndices.size) srcIndices[dim] else 0
                    if (dstDimSize == 1) 0 else srcIndexVal % dstDimSize
                }
                val current = dst.getFloat(graphAllocator, *target)
                val addition = src.getFloat(graphAllocator, *srcIndices)
                dst.setFloat(graphAllocator, current + addition, *target)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(dst, dstTotal) { _, indices ->
                dst.setHalf(
                    graphAllocator,
                    0f,
                    *indices
                )
            }
            val srcTotal = src.numElements().toInt()
            if (srcTotal == 0) return
            io.github.kotlinmania.llama.ore.applyNDIter(src, srcTotal) { _, srcIndices ->
                val target = IntArray(dstRank) { dim ->
                    val dstDimSize = dst.ne[dim].toInt().coerceAtLeast(1)
                    val srcIndexVal = if (dim < srcIndices.size) srcIndices[dim] else 0
                    if (dstDimSize == 1) 0 else srcIndexVal % dstDimSize
                }
                val current = dst.getHalf(graphAllocator, *target)
                val addition = src.getHalf(graphAllocator, *srcIndices)
                dst.setHalf(graphAllocator, current + addition, *target)
            }
        }
        else -> error("fatal error")
    }
}

fun dequantizeTensor(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne = tensor.ne.copyOf()
    if (result.type.byteSize > 0uL) {
        result.nb[0] = result.type.byteSize
        for (d in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }
    } else {
        for(d in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.nb[d] = 0uL
    }
    val numElements = tensor.numElements().toInt()
    val resultDataArray = FloatArray(numElements)
    when (tensor.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            tensor,
            numElements
        ) { flatIdx, indices ->
            if (flatIdx < numElements) resultDataArray[flatIdx] = tensor.getHalf(graphAllocator, *indices)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            tensor,
            numElements
        ) { flatIdx, indices ->
            if (flatIdx < numElements) resultDataArray[flatIdx] = tensor.getFloat(graphAllocator, *indices)
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ8_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until io.github.kotlinmania.llama.ore.QK8_0) { if (fidx < numElements) resultDataArray[fidx++] = scale * tensor.getQ8_0Weight(graphAllocator, blockIdx, itemIdxInBlock).toFloat() else break }
                if (fidx >= numElements && blockIdx < numBlocks -1) { println("Warn: Q8_0 dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q8_0 dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ4_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until io.github.kotlinmania.llama.ore.QK4_0) { if (fidx < numElements) resultDataArray[fidx++] = scale * (tensor.getQ4_0NibbleWeight(graphAllocator, blockIdx, itemIdxInBlock).toFloat() - 8.0f) else break }
                if (fidx >= numElements && blockIdx < numBlocks -1) { println("Warn: Q4_0 dequant filled array early for ${tensor.name}"); break }
            }
             if (fidx != numElements && numElements > 0) println("Warn: Q4_0 dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_1 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val dScale = tensor.getQ4_1BlockScale(graphAllocator, blockIdx)
                val mMin = tensor.getQ4_1BlockMin(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until io.github.kotlinmania.llama.ore.QK4_1) {
                    if (fidx < numElements) {
                        val qNibble = tensor.getQ4_1NibbleWeight(graphAllocator, blockIdx, itemIdxInBlock)
                        resultDataArray[fidx++] = dScale * qNibble.toFloat() + mMin
                    } else { if(fidx > 0) println("Warn: Q4_1 dequant read past numElements for ${tensor.name}"); break }
                }
                if (fidx >= numElements && blockIdx < numBlocks -1) { println("Warn: Q4_1 dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q4_1 dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        // K-Quant dequantization
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                io.github.kotlinmania.llama.ore.dequantizeQ2_KBlock(
                    graphAllocator,
                    tensor,
                    blockIdx,
                    resultDataArray,
                    fidx
                )
                fidx += io.github.kotlinmania.llama.ore.QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q2_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q2_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q3_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                io.github.kotlinmania.llama.ore.dequantizeQ3_KBlock(
                    graphAllocator,
                    tensor,
                    blockIdx,
                    resultDataArray,
                    fidx
                )
                fidx += io.github.kotlinmania.llama.ore.QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q3_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q3_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                io.github.kotlinmania.llama.ore.dequantizeQ4_KBlock(
                    graphAllocator,
                    tensor,
                    blockIdx,
                    resultDataArray,
                    fidx
                )
                fidx += io.github.kotlinmania.llama.ore.QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q4_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q4_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q5_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                io.github.kotlinmania.llama.ore.dequantizeQ5_KBlock(
                    graphAllocator,
                    tensor,
                    blockIdx,
                    resultDataArray,
                    fidx
                )
                fidx += io.github.kotlinmania.llama.ore.QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q5_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q5_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q6_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                io.github.kotlinmania.llama.ore.dequantizeQ6_KBlock(
                    graphAllocator,
                    tensor,
                    blockIdx,
                    resultDataArray,
                    fidx
                )
                fidx += io.github.kotlinmania.llama.ore.QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q6_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q6_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q8_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                io.github.kotlinmania.llama.ore.dequantizeQ8_KBlock(
                    graphAllocator,
                    tensor,
                    blockIdx,
                    resultDataArray,
                    fidx
                )
                fidx += io.github.kotlinmania.llama.ore.QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q8_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q8_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        // BitNet 1.58 dequantization
        io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getBitNet158BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until io.github.kotlinmania.llama.ore.QK_BITNET_1_58) {
                    if (fidx < numElements) {
                        val ternaryWeight = tensor.getBitNet158TernaryWeight(graphAllocator, blockIdx, itemIdxInBlock)
                        resultDataArray[fidx++] = scale * ternaryWeight.toFloat()
                    } else break 
                }
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: BitNet 1.58 dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: BitNet 1.58 dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        else -> println("Warning: dequantizeTensor from ${tensor.type} to F32 not fully implemented. Result is zeroed for ${tensor.name}.")
    }
    result.data = resultDataArray; return result
}

fun quantizeTensor(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensorF32: io.github.kotlinmania.llama.ore.GGMLTensor, targetType: io.github.kotlinmania.llama.ore.GGMLType): io.github.kotlinmania.llama.ore.GGMLTensor {
    if (tensorF32.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("quantizeTensor expects F32 input, got ${tensorF32.type}")
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = targetType); result.ne = tensorF32.ne.copyOf()
    if (targetType.byteSize > 0uL) {
        result.nb[0] = targetType.byteSize
        for (d in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }
    } else {
        if (targetType != io.github.kotlinmania.llama.ore.GGMLType.COUNT && !targetType.description.startsWith("Q", ignoreCase = true)) println("Warn: Stride calc for ${targetType.name} in quantizeTensor may be incomplete.")
        for(d in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.nb[d] = 0uL
    }
    val numElements = tensorF32.numElements().toInt()
    when (targetType) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            val resArr = ShortArray(numElements); io.github.kotlinmania.llama.ore.applyNDIter(
                tensorF32,
                numElements
            ) { fid, ind ->
                if (fid < numElements) resArr[fid] = io.github.kotlinmania.llama.ore.floatToHalf(
                    tensorF32.getFloat(
                        graphAllocator,
                        *ind
                    )
                )
            }; result.data = resArr
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            val resArr = FloatArray(numElements); io.github.kotlinmania.llama.ore.applyNDIter(
                tensorF32,
                numElements
            ) { fid, ind -> if (fid < numElements) resArr[fid] = tensorF32.getFloat(graphAllocator, *ind) }; result.data = resArr
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q8_0 -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK8_0 == 0) { "Q8_0 numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK8_0}" }
            val nBlk = numElements / io.github.kotlinmania.llama.ore.QK8_0; val blkSize = targetType.byteSize.toInt(); require(blkSize == SHORT_SIZE_BYTES + io.github.kotlinmania.llama.ore.QK8_0) { "Q8_0 block size mismatch" }
            val resArr = ByteArray(nBlk * blkSize); val f32Blk = FloatArray(io.github.kotlinmania.llama.ore.QK8_0); var curIdx = 0; var boff = 0
            io.github.kotlinmania.llama.ore.applyNDIter(tensorF32, numElements) { _, ind ->
                val itemInBlk = curIdx % io.github.kotlinmania.llama.ore.QK8_0; f32Blk[itemInBlk] =
                tensorF32.getFloat(graphAllocator, *ind)
                if (itemInBlk == io.github.kotlinmania.llama.ore.QK8_0 - 1) {
                    var amax = 0.0f; for (v in f32Blk) amax = maxOf(amax, abs(v));
                    val scale = if (amax == 0.0f) 1.0f else amax / 127.0f;
                    val invS = 1.0f / scale
                    resArr.setShortLe(boff, io.github.kotlinmania.llama.ore.floatToHalf(scale));
                    val qOff = boff + SHORT_SIZE_BYTES
                    for (k in 0 until io.github.kotlinmania.llama.ore.QK8_0) resArr[qOff + k] =
                        round(f32Blk[k] * invS).toInt().coerceIn(-128, 127).toByte()
                    boff += blkSize
                }; curIdx++
            }; result.data = resArr
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0 -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK4_0 == 0) { "Q4_0 numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK4_0}" }
            val nBlk = numElements / io.github.kotlinmania.llama.ore.QK4_0; val blkSize = targetType.byteSize.toInt(); require(blkSize == SHORT_SIZE_BYTES + io.github.kotlinmania.llama.ore.QK4_0 / 2) { "Q4_0 block size mismatch" }
            val resArr = ByteArray(nBlk * blkSize); val f32Blk = FloatArray(io.github.kotlinmania.llama.ore.QK4_0); var curIdx = 0; var boff = 0
            io.github.kotlinmania.llama.ore.applyNDIter(tensorF32, numElements) { _, ind ->
                val itemInBlk = curIdx % io.github.kotlinmania.llama.ore.QK4_0; f32Blk[itemInBlk] =
                tensorF32.getFloat(graphAllocator, *ind)
                if (itemInBlk == io.github.kotlinmania.llama.ore.QK4_0 - 1) {
                    var amax = 0.0f; for (v in f32Blk) amax = maxOf(amax, abs(v));
                    val scale = if (amax == 0.0f) 1.0f else amax / 8.0f;
                    val invS = if (scale == 0.0f) 0.0f else 1.0f / scale
                    resArr.setShortLe(boff, io.github.kotlinmania.llama.ore.floatToHalf(scale));
                    val qOff = boff + SHORT_SIZE_BYTES
                    for (j in 0 until io.github.kotlinmania.llama.ore.QK4_0 / 2) {
                        val q1 = round(f32Blk[j * 2] * invS + 8.0f).toInt().coerceIn(0, 15);
                        val q2 = round(f32Blk[j * 2 + 1] * invS + 8.0f).toInt().coerceIn(0, 15)
                        resArr[qOff + j] = (io.github.kotlinmania.llama.ore.maskLowBits32(
                            q1,
                            4
                        ) or (io.github.kotlinmania.llama.ore.logicalLeft32(
                            io.github.kotlinmania.llama.ore.maskLowBits32(
                                q2,
                                4
                            ), 4
                        ))).toByte()
                    }; boff += blkSize
                }; curIdx++
            }; result.data = resArr
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q8_1 -> { result.data = ByteArray(numElements); println("Warn: Quant F32 to ${targetType.name} NI") }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_1 -> {
            require(tensorF32.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "Input tensor for Q4_1 quantization must be F32. Got ${tensorF32.type}" } // Already checked at function start, but good for clarity
            require(numElements % io.github.kotlinmania.llama.ore.QK4_1 == 0) { "For Q4_1 quantization, total elements ($numElements) must be divisible by QK4_1 (${io.github.kotlinmania.llama.ore.QK4_1})" }

            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK4_1
            val q4_1BlockByteSize = targetType.byteSize.toInt()
            val expectedBlockSize = (2 * SHORT_SIZE_BYTES) + (io.github.kotlinmania.llama.ore.QK4_1 / 2)
            require(q4_1BlockByteSize == expectedBlockSize) { "Q4_1 block byte size mismatch. Expected $expectedBlockSize, got $q4_1BlockByteSize. Type says: ${targetType.byteSize}" }

            val q4_1DataArray = ByteArray(numBlocks * q4_1BlockByteSize)
            result.data = q4_1DataArray // Assign the data array to the result tensor prepared at the start of the function

            val f32BlockValues = FloatArray(io.github.kotlinmania.llama.ore.QK4_1)
            var currentF32ElementIndex = 0
            var q4_1ByteArrayWriteOffset = 0

            // Iterate through blocks by sequentially filling f32BlockValues
            for (blockNum in 0 until numBlocks) {
                // Populate f32BlockValues for the current block
                // This assumes tensorF32.data is a FloatArray and elements are contiguous.
                // A more robust way would use tensorF32.getFloat(graphAllocator, indices) via applyNDIter,
                // but that might be slower if direct array access is safe and possible.
                // For now, let's use a simplified sequential access matching applyNDIter's typical order.
                // This part needs careful review if tensorF32 data isn't flat or directly accessible.
                // The applyNDIter in other quantization paths fetches element by element.

                // Simplified block population:
                var f32BlockReadCount = 0
                io.github.kotlinmania.llama.ore.applyNDIter(
                    tensorF32,
                    numElements
                ) { flatIdx, indices ->
                    if (flatIdx >= blockNum * io.github.kotlinmania.llama.ore.QK4_1 && flatIdx < (blockNum + 1) * io.github.kotlinmania.llama.ore.QK4_1) {
                        if (f32BlockReadCount < io.github.kotlinmania.llama.ore.QK4_1) { // Ensure we don't write out of bounds
                            f32BlockValues[f32BlockReadCount++] = tensorF32.getFloat(graphAllocator, *indices)
                        }
                    }
                }
                // Ensure the block was fully read if applyNDIter was used in this fashion
                // This simplified block population is complex. A direct iteration is better.
                // Let's refine the block population strategy.

                // Refined block population:
                // We need to ensure that we are picking elements from tensorF32 in their logical order.
                // applyNDIter provides flatIdx and multi-dim indices. We can use flatIdx.
                // The outer loop is `for (blockNum in 0 until numBlocks)`.
                // The `currentF32ElementIndex` will track our position in the flat F32 data.
                // This will be simpler than trying to make applyNDIter fill just one block at a time.

                for (i in 0 until io.github.kotlinmania.llama.ore.QK4_1) {
                    // This assumes getFloat can handle a flat index or we convert blockNum*QK4_1 + i to ND-indices.
                    // Given applyNDIter exists, it's better to use it once over all elements
                    // and then process them in blocks, as done for Q8_0 and Q4_0.
                    // The current structure with applyNDIter outside the block loop is for those.
                    // Replicating that for Q4_1:
                    // We need to collect QK4_1 elements per block.
                    // The existing Q8_0/Q4_0 loops use applyNDIter ONCE and process blocks inside its lambda.
                    // Let's adapt that pattern.
                    // This means the 'result.data = q4_1DataArray' should be inside this when case.
                    // And the loop structure will be similar to Q8_0/Q4_0.
                    // The 'result' tensor is already set up with type and ne. Strides also.
                    // So, the previous structure of Q8_0/Q4_0 is:
                    // val resArr = ByteArray(nBlk * blkSize)
                    // val f32Blk = FloatArray(QK_K)
                    // var curIdx = 0 (overall element index)
                    // var boff = 0 (byte offset in resArr)
                    // applyNDIter(tensorF32, numElements) { _, ind ->
                    //    val itemInBlk = curIdx % QK_K; f32Blk[itemInBlk] = tensorF32.getFloat(graphAllocator, *ind)
                    //    if (itemInBlk == QK_K - 1) { process block }
                    //    curIdx++
                    // result.data = resArr
                    // This is the pattern to follow.
                }
            }
            // --- Re-writing the loop structure based on Q8_0/Q4_0 pattern ---
            val f32BlockBuffer = FloatArray(io.github.kotlinmania.llama.ore.QK4_1) // Temporary buffer for one block of F32 values
            var currentElementInF32 = 0
            var byteArrayWriteOffset = 0

            io.github.kotlinmania.llama.ore.applyNDIter(tensorF32, numElements) { _, indices ->
                val itemInBlockIndex = currentElementInF32 % io.github.kotlinmania.llama.ore.QK4_1
                f32BlockBuffer[itemInBlockIndex] = tensorF32.getFloat(graphAllocator, *indices)

                if (itemInBlockIndex == io.github.kotlinmania.llama.ore.QK4_1 - 1) { // Block is full, process it
                    val f_min = f32BlockBuffer.minOrNull() ?: 0.0f
                    val f_max = f32BlockBuffer.maxOrNull() ?: 0.0f

                    var d_scaleF32 = (f_max - f_min) / 15.0f
                    if (d_scaleF32 == 0.0f) { // Handles case where f_max == f_min
                        d_scaleF32 = 1.0f
                    }
                    val m_minF32 = f_min
                    // invDScaleF32 is guaranteed to be valid as d_scaleF32 is non-zero.
                    val invDScaleF32 = 1.0f / d_scaleF32

                    val d_scaleF16Short = io.github.kotlinmania.llama.ore.floatToHalf(d_scaleF32)
                    val m_minF16Short = io.github.kotlinmania.llama.ore.floatToHalf(m_minF32)

                    q4_1DataArray.setShortLe(byteArrayWriteOffset, d_scaleF16Short)
                    q4_1DataArray.setShortLe(byteArrayWriteOffset + SHORT_SIZE_BYTES, m_minF16Short)

                    val qsDataWriteStartOffsetInBlock = byteArrayWriteOffset + (2 * SHORT_SIZE_BYTES)
                    for (j in 0 until io.github.kotlinmania.llama.ore.QK4_1 / 2) {
                        val f32Val1 = f32BlockBuffer[j * 2]
                        val f32Val2 = f32BlockBuffer[j * 2 + 1]

                        val quantVal1 = round((f32Val1 - m_minF32) * invDScaleF32).toInt().coerceIn(0, 15)
                        val quantVal2 = round((f32Val2 - m_minF32) * invDScaleF32).toInt().coerceIn(0, 15)

                        val packedByte = io.github.kotlinmania.llama.ore.maskLowBits32(
                            quantVal1,
                            4
                        ) or (io.github.kotlinmania.llama.ore.logicalLeft32(
                            io.github.kotlinmania.llama.ore.maskLowBits32(
                                quantVal2,
                                4
                            ), 4
                        ))
                        q4_1DataArray[qsDataWriteStartOffsetInBlock + j] = packedByte.toByte()
                    }
                    byteArrayWriteOffset += q4_1BlockByteSize
                }
                currentElementInF32++
            }
            result.data = q4_1DataArray
        }
        // K-Quant quantization implementations
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_K == 0) { "Q2_K numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_K}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr

            val scratch = io.github.kotlinmania.llama.ore.Q2KScratch()
            val blockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_K)
            var destOffset = 0
            var filled = 0
            io.github.kotlinmania.llama.ore.applyNDIter(tensorF32, numElements) { _, indices ->
                val idx = filled % io.github.kotlinmania.llama.ore.QK_K
                blockValues[idx] = tensorF32.getFloat(graphAllocator, *indices)
                filled++
                if (idx == io.github.kotlinmania.llama.ore.QK_K - 1) {
                    io.github.kotlinmania.llama.ore.quantizeQ2KBlock(
                        blockValues,
                        resArr,
                        destOffset,
                        scratch
                    )
                    destOffset += blockByteSize
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q3_K -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_K == 0) { "Q3_K numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_K}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            val q3Scratch = io.github.kotlinmania.llama.ore.Q3KScratch()
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_K)
                for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                    for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }

                io.github.kotlinmania.llama.ore.quantizeQ3_KBlock(
                    blockValues,
                    resArr,
                    blockNum * blockByteSize,
                    q3Scratch
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_K -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_K == 0) { "Q4_K numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_K}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            val q4Scratch = io.github.kotlinmania.llama.ore.Q4KScratch()
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_K)
                for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                    for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }

                io.github.kotlinmania.llama.ore.quantizeQ4_KBlock(
                    blockValues,
                    resArr,
                    blockNum * blockByteSize,
                    q4Scratch
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q5_K -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_K == 0) { "Q5_K numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_K}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            val q5Scratch = io.github.kotlinmania.llama.ore.Q5KScratch()
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_K)
                for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                    for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }

                io.github.kotlinmania.llama.ore.quantizeQ5_KBlock(
                    blockValues,
                    resArr,
                    blockNum * blockByteSize,
                    q5Scratch
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q6_K -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_K == 0) { "Q6_K numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_K}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            val q6Scratch = io.github.kotlinmania.llama.ore.Q6KScratch()
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_K)
                for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                    for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }

                io.github.kotlinmania.llama.ore.quantizeQ6_KBlock(
                    blockValues,
                    resArr,
                    blockNum * blockByteSize,
                    q6Scratch
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q8_K -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_K == 0) { "Q8_K numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_K}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_K)
                for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
                    for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }

                io.github.kotlinmania.llama.ore.quantizeQ8_KBlock(
                    blockValues,
                    resArr,
                    blockNum * blockByteSize
                )
            }
        }
        // BitNet 1.58 quantization
        io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> {
            require(numElements % io.github.kotlinmania.llama.ore.QK_BITNET_1_58 == 0) { "BitNet 1.58 numElements $numElements not div by ${io.github.kotlinmania.llama.ore.QK_BITNET_1_58}" }
            val numBlocks = numElements / io.github.kotlinmania.llama.ore.QK_BITNET_1_58
            val blockByteSize = targetType.byteSize.toInt()
            require(blockByteSize == Short.SIZE_BYTES + 8) { "BitNet 1.58 block size mismatch. Expected ${Short.SIZE_BYTES + 8}, got $blockByteSize" }
            
            val bitNet158DataArray = ByteArray(numBlocks * blockByteSize)
            result.data = bitNet158DataArray
            
            val f32BlockValues = FloatArray(io.github.kotlinmania.llama.ore.QK_BITNET_1_58)
            var currentF32ElementIndex = 0
            var blockByteWriteOffset = 0
            
            // Process each block
            for (blockNum in 0 until numBlocks) {
                // Fill f32BlockValues for this block
                io.github.kotlinmania.llama.ore.applyNDIter(
                    tensorF32,
                    io.github.kotlinmania.llama.ore.QK_BITNET_1_58
                ) { idx, ind ->
                    if (currentF32ElementIndex < numElements && idx < io.github.kotlinmania.llama.ore.QK_BITNET_1_58) {
                        f32BlockValues[idx] = tensorF32.getFloat(graphAllocator, *ind)
                        currentF32ElementIndex++
                    }
                }
                
                // Calculate scale as the maximum absolute value
                var maxAbs = 0.0f
                for (v in f32BlockValues) {
                    maxAbs = maxOf(maxAbs, kotlin.math.abs(v))
                }
                val scale = if (maxAbs == 0.0f) 1.0f else maxAbs
                val invScale = if (scale == 0.0f) 0.0f else 1.0f / scale
                
                // Store scale as F16
                bitNet158DataArray.setShortLe(blockByteWriteOffset,
                    io.github.kotlinmania.llama.ore.floatToHalf(scale)
                )
                
                // Quantize values to ternary (-1, 0, +1) and pack them
                val ternaryValues = IntArray(io.github.kotlinmania.llama.ore.QK_BITNET_1_58)
                for (i in 0 until io.github.kotlinmania.llama.ore.QK_BITNET_1_58) {
                    val normalizedValue = f32BlockValues[i] * invScale
                    ternaryValues[i] = when {
                        normalizedValue > 0.5f -> 2  // +1 -> encoded as 2
                        normalizedValue < -0.5f -> 0 // -1 -> encoded as 0  
                        else -> 1                    //  0 -> encoded as 1
                    }
                }
                
                // Pack ternary values: 5 values per byte using base-3 encoding
                val ternaryDataOffset = blockByteWriteOffset + Short.SIZE_BYTES
                val powers = intArrayOf(1, 3, 9, 27, 81) // 3^0, 3^1, 3^2, 3^3, 3^4
                
                for (groupIdx in 0 until 7) { // 32 values / 5 per byte = 6.4, so 7 bytes (last byte partially used)
                    var packedValue = 0
                    for (posInGroup in 0 until 5) {
                        val valueIdx = groupIdx * 5 + posInGroup
                        if (valueIdx < io.github.kotlinmania.llama.ore.QK_BITNET_1_58) {
                            packedValue += ternaryValues[valueIdx] * powers[posInGroup]
                        }
                        // If valueIdx >= QK_BITNET_1_58, the value remains 0 (encoded as -1), which is fine
                    }
                    bitNet158DataArray[ternaryDataOffset + groupIdx] = packedValue.toByte()
                }
                
                blockByteWriteOffset += blockByteSize
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1 -> { result.data = ByteArray((numElements * 5 + 7) / 8); println("Warn: Quant F32 to ${targetType.name} NI") }
        else -> { println("Error: Unsupp target quant type $targetType"); result.data = null }
    }
    return result
}

fun computeMatMul(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val M = a.ne[1].toInt()
    val K_a = a.ne[0].toInt()
    val N = b.ne[0].toInt()
    val K_b = b.ne[1].toInt()
    if (K_a != K_b) throw IllegalArgumentException("Dim mismatch K: a.ne[0]($K_a) != b.ne[1]($K_b)")
    val K = K_a

    // Validate destination tensor dimensions 
    if (dst.ne[0].toInt() != N || dst.ne[1].toInt() != M) {
        throw IllegalArgumentException("Result tensor dimensions must match expected output size: expected [${N}, ${M}], got [${dst.ne[0]}, ${dst.ne[1]}]")
    }

    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_0 && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_0 x F32 matmul")
        
        // Write results directly into destination tensor using graph allocator
        var flatIdx = 0
        for(i in 0 until M){ 
            for(j in 0 until N){ 
                val result = io.github.kotlinmania.llama.ore.computeDotProductQ40F32(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, result, j, i) // Column j, row i
                flatIdx++
            } 
        }
        return
    }
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_1 && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_1 x F32 matmul")

        // Write results directly into destination tensor
        for (i in 0 until M) { // Iterate output rows (M)
            for (j in 0 until N) { // Iterate output columns (N)
                val dotProduct = io.github.kotlinmania.llama.ore.computeDotProductQ41F32(
                    graphAllocator,
                    a,    // Q4_1 tensor (src0)
                    b,    // F32 tensor (src1)
                    i,    // Current row in src0 (0 to M-1)
                    j,    // Current column in src1 (0 to N-1)
                    K     // Common dimension
                )
                dst.setFloat(graphAllocator, dotProduct, j, i) // Column j, row i
            }
        }
        return
    }
    
    // K-Quant optimized matrix multiplication paths
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q2_K && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q2_K x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = io.github.kotlinmania.llama.ore.computeDotProductQ2_KF32(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_K x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = io.github.kotlinmania.llama.ore.computeDotProductQ4_KF32(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    // F32 x Q4_K matrix multiplication (symmetric case)
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && b.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for F32 x Q4_K matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = io.github.kotlinmania.llama.ore.computeDotProductF32Q4_K(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    // Q4_K x Q4_K matrix multiplication (direct quantized-to-quantized)
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K && b.type == io.github.kotlinmania.llama.ore.GGMLType.Q4_K) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_K x Q4_K matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = io.github.kotlinmania.llama.ore.computeDotProductQ4_KQ4_K(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_K && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q8_K x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = io.github.kotlinmania.llama.ore.computeDotProductQ8_KF32(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.Q8_0 && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q8_0 x F32 matmul")
        
        // Write results directly into destination tensor
        for(i in 0 until M){ 
            for(j in 0 until N){ 
                val result = io.github.kotlinmania.llama.ore.computeDotProductQ80F32(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, result, j, i) // Column j, row i
            } 
        }
        return
    }

    if (a.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) {
        if (dst.type != io.github.kotlinmania.llama.ore.GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for F32 x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = io.github.kotlinmania.llama.ore.computeDotProductF32F32(
                    graphAllocator,
                    a,
                    b,
                    i,
                    j,
                    K
                )
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }

    // General matrix multiplication fallback (dst-arg only)
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match first input type for general matmul")
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            val effA=a; val effB=if(b.type== io.github.kotlinmania.llama.ore.GGMLType.F16)b else io.github.kotlinmania.llama.ore.dequantizeTensor(
                graphAllocator,
                b
            )
            if(effB.type!= io.github.kotlinmania.llama.ore.GGMLType.F16) error("fatal error")
            for(i in 0 until M){
                for(j in 0 until N){
                    var sum=0.0f
                    for(l in 0 until K){
                        sum+=effA.getHalf(graphAllocator,l,i)*effB.getHalf(graphAllocator,j,l)
                    }
                    dst.setHalf(graphAllocator, sum, j, i) // Column j, row i
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS,
        io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> {
            val aF32= io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a); val bF32=
                io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, b)
            val tempF32 =
                io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32); tempF32.ne = dst.ne.copyOf(); tempF32.nb =
                io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                    tempF32.ne,
                    io.github.kotlinmania.llama.ore.GGMLType.F32,
                    tempF32.ne.size
                )
            computeMatMul(graphAllocator,aF32,bF32,tempF32)
            val qRes=
                io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tempF32, dst.type); dst.data=qRes.data
        }
        else -> error("fatal error")
    }
}

fun computeRelu(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            totalSize
        ) { _, ind ->
            dst.setFloat(
                graphAllocator,
                if (a.getFloat(graphAllocator, *ind) > 0.0f) a.getFloat(graphAllocator, *ind) else 0.0f,
                *ind
            )
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            totalSize
        ) { _, ind ->
            dst.setHalf(
                graphAllocator,
                if (a.getHalf(graphAllocator, *ind) > 0.0f) a.getHalf(graphAllocator, *ind) else 0.0f,
                *ind
            )
        }
        else -> error("fatal error")
    }
}

fun computeGelu(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()
    val gelu = {x:Float -> x*0.5f*(1.0f+kotlin.math.tanh(0.797885f*(x+0.044715f*x*x*x)))}
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            totalSize
        ) { _, ind -> dst.setFloat(graphAllocator, gelu(a.getFloat(graphAllocator, *ind)), *ind) }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            totalSize
        ) { _, ind -> dst.setHalf(graphAllocator, gelu(a.getHalf(graphAllocator, *ind)), *ind) }
        else -> error("fatal error")
    }
}

fun computeSilu(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")

    val totalSize = dst.numElements().toInt()
    val sigmoid = {x:Float -> 1.0f / (1.0f + exp(-x))}
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            totalSize
        ) { _, ind ->
            val v = a.getFloat(graphAllocator, *ind)
            dst.setFloat(graphAllocator, v * sigmoid(v), *ind)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            totalSize
        ) { _, ind ->
            val v = a.getHalf(graphAllocator, *ind)
            dst.setHalf(graphAllocator, v * sigmoid(v), *ind)
        }
        else -> error("fatal error")
    }
}

fun computeSoftMax(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")

    val nCols = a.ne[0].toInt()
    val nRows = (a.numElements() / a.ne[0]).toInt()
    val rank = a.rank()
    val baseIndices = IntArray(rank)

    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            for (row in 0 until nRows) {
                var tmp = row
                for (d in 1 until rank) {
                    val dimSize = a.ne[d].toInt()
                    baseIndices[d] = tmp % dimSize
                    tmp /= dimSize
                }
                var maxVal = Float.NEGATIVE_INFINITY
                for (col in 0 until nCols) {
                    baseIndices[0] = col
                    val v = a.getFloat(graphAllocator, *baseIndices)
                    if (v > maxVal) maxVal = v
                }
                var sum = 0.0f
                for (col in 0 until nCols) {
                    baseIndices[0] = col
                    val e = exp(a.getFloat(graphAllocator, *baseIndices) - maxVal)
                    dst.setFloat(graphAllocator, e, *baseIndices)
                    sum += e
                }
                val invSum = 1.0f / sum
                for (col in 0 until nCols) {
                    baseIndices[0] = col
                    val v = dst.getFloat(graphAllocator, *baseIndices) * invSum
                    dst.setFloat(graphAllocator, v, *baseIndices)
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            for (row in 0 until nRows) {
                var tmp = row
                for (d in 1 until rank) {
                    val dimSize = a.ne[d].toInt()
                    baseIndices[d] = tmp % dimSize
                    tmp /= dimSize
                }
                var maxVal = Float.NEGATIVE_INFINITY
                for (col in 0 until nCols) {
                    baseIndices[0] = col
                    val v = a.getHalf(graphAllocator, *baseIndices)
                    if (v > maxVal) maxVal = v
                }
                var sum = 0.0f
                for (col in 0 until nCols) {
                    baseIndices[0] = col
                    val e = exp(a.getHalf(graphAllocator, *baseIndices) - maxVal)
                    dst.setHalf(graphAllocator, e, *baseIndices)
                    sum += e
                }
                val invSum = 1.0f / sum
                for (col in 0 until nCols) {
                    baseIndices[0] = col
                    val v = dst.getHalf(graphAllocator, *baseIndices) * invSum
                    dst.setHalf(graphAllocator, v, *baseIndices)
                }
            }
        }
        else -> error("fatal error")
    }
}

fun computeRMSNorm(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")

    val ne0 = a.ne[0].toInt()
    val ne1 = a.ne.getOrElse(1) { 1L }.toInt()
    val ne2 = a.ne.getOrElse(2) { 1L }.toInt()
    val ne3 = a.ne.getOrElse(3) { 1L }.toInt()
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) {
                var sumSq = 0.0f
                for (i0 in 0 until ne0) {
                    val v = a.getFloat(graphAllocator, i0, i1, i2, i3)
                    sumSq += v * v
                }
                val scale = 1.0f / sqrt(sumSq / ne0 + eps)
                for (i0 in 0 until ne0) {
                    val v = a.getFloat(graphAllocator, i0, i1, i2, i3)
                    dst.setFloat(graphAllocator, v * scale, i0, i1, i2, i3)
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) {
                var sumSq = 0.0f
                for (i0 in 0 until ne0) {
                    val v = a.getHalf(graphAllocator, i0, i1, i2, i3)
                    sumSq += v * v
                }
                val scale = 1.0f / sqrt(sumSq / ne0 + eps)
                for (i0 in 0 until ne0) {
                    val v = a.getHalf(graphAllocator, i0, i1, i2, i3)
                    dst.setHalf(graphAllocator, v * scale, i0, i1, i2, i3)
                }
            }
        }
        else -> error("fatal error")
    }
}

fun computeSub(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for(i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS){if(a.ne[i]!=b.ne[i])throw IllegalArgumentException("Dims mismatch")}
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val ts=dst.numElements().toInt()
    when(a.type){
        io.github.kotlinmania.llama.ore.GGMLType.F32-> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, ind ->
                dst.setFloat(
                    graphAllocator,
                    a.getFloat(graphAllocator, *ind) - b.getFloat(graphAllocator, *ind),
                    *ind
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16-> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, ind ->
                dst.setHalf(
                    graphAllocator,
                    a.getHalf(graphAllocator, *ind) - b.getHalf(graphAllocator, *ind),
                    *ind
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                dst.setInt(
                    graphAllocator,
                    a.getInt(graphAllocator, *indices) - b.getInt(graphAllocator, *indices),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                dst.setShort(
                    graphAllocator,
                    (valA - valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                dst.setByte(
                    graphAllocator,
                    (valA - valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                dst.setLong(
                    graphAllocator,
                    a.getLong(graphAllocator, *indices) - b.getLong(graphAllocator, *indices),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            val af = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a)
            val bf = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, b)
            val tempF32 =
                io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
            tempF32.ne = dst.ne.copyOf()
            tempF32.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                tempF32.ne,
                io.github.kotlinmania.llama.ore.GGMLType.F32,
                tempF32.ne.size
            )
            computeSub(graphAllocator, af, bf, tempF32)
            val qr =
                io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = qr.data
        }
        else -> error("fatal error")
    }
}

fun computeNeg(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val ts=dst.numElements().toInt()
    when(a.type){
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            ts
        ) { _, ind -> dst.setFloat(graphAllocator, -a.getFloat(graphAllocator, *ind), *ind) }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            ts
        ) { _, ind -> dst.setHalf(graphAllocator, -a.getHalf(graphAllocator, *ind), *ind) }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            val af = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a)
            val tempF32 =
                io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
            tempF32.ne = dst.ne.copyOf(); tempF32.nb =
                io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                    tempF32.ne,
                    io.github.kotlinmania.llama.ore.GGMLType.F32,
                    tempF32.ne.size
                )
            computeNeg(graphAllocator, af, tempF32)
            val qr =
                io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = qr.data
        }
        else -> error("fatal error")
    }
}

fun computeDiv(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    for(i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS){if(a.ne[i]!=b.ne[i])throw IllegalArgumentException("Dims mismatch")}
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val ts=dst.numElements().toInt()
    val div={vA:Float,vB:Float->if(vB==0.0f){if(vA==0.0f)Float.NaN else if(vA>0.0f)Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY}else{vA/vB}}
    when(a.type){
        io.github.kotlinmania.llama.ore.GGMLType.F32-> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, ind ->
                dst.setFloat(
                    graphAllocator,
                    div(a.getFloat(graphAllocator, *ind), b.getFloat(graphAllocator, *ind)),
                    *ind
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16-> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, ind ->
                dst.setHalf(
                    graphAllocator,
                    div(a.getHalf(graphAllocator, *ind), b.getHalf(graphAllocator, *ind)),
                    *ind
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                if (valB == 0) throw ArithmeticException("Division by zero for I32")
                dst.setInt(graphAllocator, valA / valB, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                if (valB == 0) throw ArithmeticException("Division by zero for I16")
                dst.setShort(
                    graphAllocator,
                    (valA / valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I8 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                if (valB == 0) throw ArithmeticException("Division by zero for I8")
                dst.setByte(
                    graphAllocator,
                    (valA / valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(),
                    *indices
                )
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { _, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                if (valB == 0L) throw ArithmeticException("Division by zero for I64")
                dst.setLong(graphAllocator, valA / valB, *indices)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            val af = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a); val bf =
                io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, b)
            val tempF32 =
                io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
            tempF32.ne = dst.ne.copyOf(); tempF32.nb =
                io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                    tempF32.ne,
                    io.github.kotlinmania.llama.ore.GGMLType.F32,
                    tempF32.ne.size
                )
            computeDiv(graphAllocator, af, bf, tempF32)
            val qr =
                io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = qr.data
        }
        else -> error("fatal error")
    }
}

// K-Quant Block Quantization Functions

/**
 * Quantizes a block of QK_K float values to Q2_K format.
 * Q2_K structure: scales[QK_K/16], qs[QK_K/4], d (F16), dmin (F16)
 * Effectively 2.625 bits per weight
 */
private const val GROUP_MAX_EPS_F = 1e-15f

private class Q2KScratch(
    val quants: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K),
    val aux: ByteArray = ByteArray(16),
    val weights: FloatArray = FloatArray(16),
    val mins: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16),
    val scales: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
)

private class Q3KScratch(
    val quants: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K),
    val scales: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
)

private class Q4KScratch(
    val quants: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K),
    val aux: ByteArray = ByteArray(32),
    val weights: FloatArray = FloatArray(32),
    val mins: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val scales: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32)
)

private fun logicalLeft32(value: Int, bits: Int): Int = value shl bits

private fun logicalRight32(value: Int, bits: Int): Int = value ushr bits

private fun maskLowBits32(value: Int, bits: Int): Int =
    if (bits >= 32) value else value and ((1 shl bits) - 1)

private fun mergeBits32(base: Int, addition: Int): Int = base or addition

private fun lowNibble(value: Int): Int = io.github.kotlinmania.llama.ore.maskLowBits32(value, 4)

private fun highNibble(value: Int): Int = io.github.kotlinmania.llama.ore.maskLowBits32(
    io.github.kotlinmania.llama.ore.logicalRight32(
        value,
        4
    ), 4
)

private fun packNibbles(low: Int, high: Int): Int =
    io.github.kotlinmania.llama.ore.mergeBits32(
        io.github.kotlinmania.llama.ore.maskLowBits32(
            low,
            4
        ),
        io.github.kotlinmania.llama.ore.logicalLeft32(
            io.github.kotlinmania.llama.ore.maskLowBits32(
                high,
                4
            ), 4
        )
    )

private fun extractBits(value: Int, shift: Int, count: Int): Int =
    io.github.kotlinmania.llama.ore.maskLowBits32(
        io.github.kotlinmania.llama.ore.logicalRight32(
            value,
            shift
        ), count
    )

private fun unsignedByte(value: Byte): Int =
    io.github.kotlinmania.llama.ore.maskLowBits32(value.toInt(), 8)

private fun extractBitsFromByte(value: Byte, shift: Int, count: Int): Int =
    io.github.kotlinmania.llama.ore.extractBits(
        io.github.kotlinmania.llama.ore.unsignedByte(
            value
        ), shift, count
    )

private fun combineNibbleIntoByte(base: Byte, highNibble: Int): Byte =
    io.github.kotlinmania.llama.ore.maskLowBits32(
        io.github.kotlinmania.llama.ore.packNibbles(
            io.github.kotlinmania.llama.ore.maskLowBits32(base.toInt(), 4),
            highNibble
        ), 8
    ).toByte()

private fun nibbleFromByte(value: Byte, index: Int): Int {
    val unsigned = io.github.kotlinmania.llama.ore.unsignedByte(value)
    return if (index == 0) io.github.kotlinmania.llama.ore.lowNibble(unsigned) else io.github.kotlinmania.llama.ore.highNibble(
        unsigned
    )
}

private fun Byte.toUnsignedInt(): Int = io.github.kotlinmania.llama.ore.unsignedByte(this)

private fun Byte.getBits(offset: Int, width: Int): Int =
    io.github.kotlinmania.llama.ore.extractBits(toUnsignedInt(), offset, width)

private fun Byte.withBits(value: Int, offset: Int, width: Int): Byte =
    io.github.kotlinmania.llama.ore.mergeIntoByte(this, value, offset, width)

private fun Int.lowBits(width: Int): Int =
    io.github.kotlinmania.llama.ore.maskLowBits32(this, width)

private fun Int.getBits(offset: Int, width: Int): Int =
    io.github.kotlinmania.llama.ore.extractBits(this, offset, width)

private fun mergeIntoByte(base: Byte, value: Int, shift: Int, width: Int): Byte {
    val clearedAddition = io.github.kotlinmania.llama.ore.logicalLeft32(
        io.github.kotlinmania.llama.ore.maskLowBits32(
            value,
            width
        ), shift
    )
    val merged = io.github.kotlinmania.llama.ore.mergeBits32(
        io.github.kotlinmania.llama.ore.unsignedByte(base), clearedAddition
    )
    return io.github.kotlinmania.llama.ore.maskLowBits32(merged, 8).toByte()
}

/** Pack two 4-bit nibbles into a single byte (low in bits 0-3, high in bits 4-7). */
private fun packNibblesInt(low: Int, high: Int): Int = ((high and 0xF) shl 4) or (low and 0xF)

/** Write [width] bits of [value] at [bitIndex] into [base], returning updated byte value. */
private fun bitplaneWriteInt(base: Int, value: Int, bitIndex: Int, width: Int): Int {
    val fieldMask = (1 shl width) - 1
    val cleared = base and (fieldMask shl bitIndex).inv()
    val toWrite = (value and fieldMask) shl bitIndex
    return (cleared or toWrite) and 0xFF
}

private fun quantizeQ2KBlock(values: FloatArray, dest: ByteArray, destOffset: Int, scratch: io.github.kotlinmania.llama.ore.Q2KScratch) {
    require(values.size == io.github.kotlinmania.llama.ore.QK_K) { "Q2_K block must have ${io.github.kotlinmania.llama.ore.QK_K} values" }

    val l = scratch.quants
    val lAux = scratch.aux
    val weights = scratch.weights
    val mins = scratch.mins
    val scales = scratch.scales

    var maxScale = 0.0f
    var maxMin = 0.0f

    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
        val base = subBlock * 16
        for (i in 0 until 16) {
            weights[i] = abs(values[base + i])
        }
        val stats = io.github.kotlinmania.llama.ore.makeQKX2Quants(
            n = 16,
            nmax = 3,
            values = values,
            valuesOffset = base,
            weights = weights,
            weightsOffset = 0,
            dest = l,
            destOffset = base,
            mins = mins,
            minsIndex = subBlock,
            aux = lAux,
            auxOffset = 0,
            rmin = -0.5f,
            rdelta = 0.1f,
            nstep = 15,
            useMad = true
        )
        scales[subBlock] = stats.scale
        if (stats.scale > maxScale) maxScale = stats.scale
        if (stats.min > maxMin) maxMin = stats.min
    }

    val scalesOffset = destOffset
    val quantsOffset = destOffset + io.github.kotlinmania.llama.ore.QK_K / 16
    val dOffset = quantsOffset + io.github.kotlinmania.llama.ore.QK_K / 4
    val dminOffset = dOffset + SHORT_SIZE_BYTES

    val dHalf = if (maxScale > 0.0f) {
        val iscale = 15.0f / maxScale
        for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val quantized = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * scales[subBlock])
                .coerceIn(0, 15)
            dest[scalesOffset + subBlock] = quantized.toByte()
        }
        io.github.kotlinmania.llama.ore.floatToHalf(maxScale / 15.0f)
    } else {
        for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) dest[scalesOffset + subBlock] = 0
        io.github.kotlinmania.llama.ore.floatToHalf(0.0f)
    }
    dest.setShortLe(dOffset, dHalf)

    val dminHalf = if (maxMin > 0.0f) {
        val iscale = 15.0f / maxMin
        for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
            val existing = io.github.kotlinmania.llama.ore.maskLowBits32(
                dest[scalesOffset + subBlock].toInt(),
                4
            )
            val quantized = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * mins[subBlock])
                .coerceIn(0, 15)
            dest[scalesOffset + subBlock] = io.github.kotlinmania.llama.ore.packNibbles(
                existing,
                quantized
            ).toByte()
        }
        io.github.kotlinmania.llama.ore.floatToHalf(maxMin / 15.0f)
    } else {
        io.github.kotlinmania.llama.ore.floatToHalf(0.0f)
    }
    dest.setShortLe(dminOffset, dminHalf)

    val dBase = io.github.kotlinmania.llama.ore.halfToFloat(dHalf)
    val dmBase = io.github.kotlinmania.llama.ore.halfToFloat(dminHalf)

    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 16) {
        val scaleByte =
            io.github.kotlinmania.llama.ore.maskLowBits32(dest[scalesOffset + subBlock].toInt(), 8)
        val scaleLow = io.github.kotlinmania.llama.ore.lowNibble(scaleByte)
        val scaleHigh = io.github.kotlinmania.llama.ore.highNibble(scaleByte)
        val dScale = dBase * scaleLow
        val base = subBlock * 16
        if (dScale == 0.0f) {
            for (lane in 0 until 16) l[base + lane] = 0
            continue
        }
        val dm = dmBase * scaleHigh
        for (lane in 0 until 16) {
            val value = values[base + lane]
            val quantized = io.github.kotlinmania.llama.ore.nearestIntFloat((value + dm) / dScale)
                .coerceIn(0, 3)
            l[base + lane] = quantized.toByte()
        }

        for (group in 0 until io.github.kotlinmania.llama.ore.QK_K / 64) {
            val idx0 = base + group * 4
            val v0 = io.github.kotlinmania.llama.ore.maskLowBits32(l[idx0].toInt() and 0xFF, 2)
            val v1 = io.github.kotlinmania.llama.ore.maskLowBits32(l[idx0 + 1].toInt() and 0xFF, 2)
            val v2 = io.github.kotlinmania.llama.ore.maskLowBits32(l[idx0 + 2].toInt() and 0xFF, 2)
            val v3 = io.github.kotlinmania.llama.ore.maskLowBits32(l[idx0 + 3].toInt() and 0xFF, 2)

            val combined = io.github.kotlinmania.llama.ore.mergeBits32(
                io.github.kotlinmania.llama.ore.mergeBits32(
                    v0,
                    io.github.kotlinmania.llama.ore.logicalLeft32(v1, 2)
                ),
                io.github.kotlinmania.llama.ore.mergeBits32(
                    io.github.kotlinmania.llama.ore.logicalLeft32(
                        v2,
                        4
                    ), io.github.kotlinmania.llama.ore.logicalLeft32(v3, 6)
                )
            )
            dest[quantsOffset + subBlock * (io.github.kotlinmania.llama.ore.QK_K / 64) + group] = io.github.kotlinmania.llama.ore.maskLowBits32(
                combined,
                8
            ).toByte()
        }
    }
}

/**
 * Quantizes a block of QK_K float values to Q3_K format.
 * Q3_K structure: hmask[QK_K/8], qs[QK_K/4], scales[12], d (F16)
 * Effectively 3.4375 bits per weight
 */
private fun quantizeQ3_KBlock(
    blockValues: FloatArray,
    dest: ByteArray,
    destOffset: Int,
    scratch: io.github.kotlinmania.llama.ore.Q3KScratch
) {
    require(blockValues.size == io.github.kotlinmania.llama.ore.QK_K) { "Q3_K block must have ${io.github.kotlinmania.llama.ore.QK_K} values" }

    val quants = scratch.quants
    val scales = scratch.scales

    val subBlocks = io.github.kotlinmania.llama.ore.QK_K / 16
    var maxScale = 0.0f
    var maxAbsScale = 0.0f
    for (sub in 0 until subBlocks) {
        val base = sub * 16
        val scale = io.github.kotlinmania.llama.ore.makeQ3Quants(
            n = 16,
            nmax = 4,
            values = blockValues,
            valuesOffset = base,
            dest = quants,
            destOffset = base,
            doRmse = true
        )
        scales[sub] = scale
        val absScale = abs(scale)
        if (absScale > maxAbsScale) {
            maxAbsScale = absScale
            maxScale = scale
        }
    }

    val hmaskOffset = destOffset
    val qsOffset = hmaskOffset + io.github.kotlinmania.llama.ore.QK_K / 8
    val scalesOffset = qsOffset + io.github.kotlinmania.llama.ore.QK_K / 4
    val dOffset = scalesOffset + 12

    dest.fill(0.toByte(), hmaskOffset, hmaskOffset + io.github.kotlinmania.llama.ore.QK_K / 8)
    dest.fill(0.toByte(), qsOffset, qsOffset + io.github.kotlinmania.llama.ore.QK_K / 4)
    dest.fill(0.toByte(), scalesOffset, scalesOffset + 12)

    if (maxAbsScale < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS_F) {
        dest.setShortLe(dOffset, io.github.kotlinmania.llama.ore.floatToHalf(0f))
        return
    }

    val iscale = -32.0f / maxScale
    dest.setShortLe(dOffset, io.github.kotlinmania.llama.ore.floatToHalf(1f / iscale))
    val dBase = io.github.kotlinmania.llama.ore.halfToFloat(dest.getShortLe(dOffset))

    for (sub in 0 until subBlocks) {
        var qScale = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * scales[sub])
            .coerceIn(-32, 31)
        qScale += 32
        val lowIndex = scalesOffset + if (sub < 8) sub else sub - 8
        val lowShift = if (sub < 8) 0 else 4
        val lowBits = io.github.kotlinmania.llama.ore.maskLowBits32(qScale, 4)
        dest[lowIndex] =
            io.github.kotlinmania.llama.ore.mergeIntoByte(dest[lowIndex], lowBits, lowShift, 4)

        val highIndex = scalesOffset + 8 + (sub % 4)
        val bitOffset = (sub / 4) * 2
        val highBits = io.github.kotlinmania.llama.ore.logicalRight32(qScale, 4)
        dest[highIndex] =
            io.github.kotlinmania.llama.ore.mergeIntoByte(dest[highIndex], highBits, bitOffset, 2)
    }

    for (sub in 0 until subBlocks) {
        val scaleInt = io.github.kotlinmania.llama.ore.readQ3Scale(dest, scalesOffset, sub)
        val scaleValue = dBase * scaleInt
        if (scaleValue == 0.0f) continue
        val base = sub * 16
        for (lane in 0 until 16) {
            val q = io.github.kotlinmania.llama.ore.nearestIntFloat(blockValues[base + lane] / scaleValue)
                .coerceIn(-4, 3) + 4
            quants[base + lane] = q.toByte()
        }
    }

    var maskIndex = 0
    var bitPlane = 0
    for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
        var q = quants[i].toInt()
        if (q > 3) {
            q -= 4
            quants[i] = q.toByte()
            val maskPos = hmaskOffset + maskIndex
            dest[maskPos] =
                io.github.kotlinmania.llama.ore.mergeIntoByte(dest[maskPos], 1, bitPlane, 1)
        }
        maskIndex++
        if (maskIndex == io.github.kotlinmania.llama.ore.QK_K / 8) {
            maskIndex = 0
            bitPlane++
        }
    }

    var qsIndex = qsOffset
    for (chunk in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
        for (l in 0 until 32) {
            val q0 = io.github.kotlinmania.llama.ore.maskLowBits32(quants[chunk + l].toInt(), 2)
            val q1 =
                io.github.kotlinmania.llama.ore.maskLowBits32(quants[chunk + l + 32].toInt(), 2)
            val q2 =
                io.github.kotlinmania.llama.ore.maskLowBits32(quants[chunk + l + 64].toInt(), 2)
            val q3 =
                io.github.kotlinmania.llama.ore.maskLowBits32(quants[chunk + l + 96].toInt(), 2)
            dest[qsIndex + l] = (((q3 and 3) shl 6) or ((q2 and 3) shl 4) or ((q1 and 3) shl 2) or (q0 and 3)).toByte()
        }
        qsIndex += 32
    }
}

private fun readQ3Scale(dest: ByteArray, scalesOffset: Int, index: Int): Int {
    val lowIndex = scalesOffset + if (index < 8) index else index - 8
    val lowShift = if (index < 8) 0 else 4
    val low = io.github.kotlinmania.llama.ore.extractBits(
        io.github.kotlinmania.llama.ore.unsignedByte(dest[lowIndex]), lowShift, 4
    )
    val highIndex = scalesOffset + 8 + (index % 4)
    val highShift = (index / 4) * 2
    val high = io.github.kotlinmania.llama.ore.extractBits(
        io.github.kotlinmania.llama.ore.unsignedByte(dest[highIndex]), highShift, 2
    )
    return io.github.kotlinmania.llama.ore.mergeBits32(
        low,
        io.github.kotlinmania.llama.ore.logicalLeft32(high, 4)
    ) - 32
}

/**
 * Quantizes a block of QK_K float values to Q4_K format.
 * Q4_K structure: d (F16), dmin (F16), scales[io.github.kotlinmania.llama.ore.K_SCALE_SIZE], qs[QK_K/2]
 * Effectively 4.5 bits per weight
 */
private fun quantizeQ4_KBlock(
    blockValues: FloatArray,
    dest: ByteArray,
    destOffset: Int,
    scratch: io.github.kotlinmania.llama.ore.Q4KScratch
) {
    require(blockValues.size == io.github.kotlinmania.llama.ore.QK_K) { "Q4_K block must have ${io.github.kotlinmania.llama.ore.QK_K} values" }

    val quants = scratch.quants
    val aux = scratch.aux
    val weights = scratch.weights
    val mins = scratch.mins
    val scales = scratch.scales

    var maxScale = 0.0f
    var maxMin = 0.0f

    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
        val base = subBlock * 32
        var sumX2 = 0.0f
        for (i in 0 until 32) {
            val value = blockValues[base + i]
            sumX2 += value * value
        }
        val avX = sqrt(sumX2 / 32.0f)
        for (i in 0 until 32) {
            weights[i] = avX + abs(blockValues[base + i])
        }
        val stats = io.github.kotlinmania.llama.ore.makeQKX2Quants(
            n = 32,
            nmax = 15,
            values = blockValues,
            valuesOffset = base,
            weights = weights,
            weightsOffset = 0,
            dest = quants,
            destOffset = base,
            mins = mins,
            minsIndex = subBlock,
            aux = aux,
            auxOffset = 0,
            rmin = -1.0f,
            rdelta = 0.1f,
            nstep = 20,
            useMad = false
        )
        scales[subBlock] = stats.scale
        if (stats.scale > maxScale) maxScale = stats.scale
        if (stats.min > maxMin) maxMin = stats.min
    }

    val scalesOffset = destOffset + 4
    for (i in scalesOffset until scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE) {
        dest[i] = 0
    }

    val dHalf =
        io.github.kotlinmania.llama.ore.floatToHalf(if (maxScale > 0.0f) maxScale / 63.0f else 0.0f)
    dest.setShortLe(destOffset, dHalf)
    val d = io.github.kotlinmania.llama.ore.halfToFloat(dHalf)

    val dminHalf =
        io.github.kotlinmania.llama.ore.floatToHalf(if (maxMin > 0.0f) maxMin / 63.0f else 0.0f)
    dest.setShortLe(destOffset + 2, dminHalf)
    val dmin = io.github.kotlinmania.llama.ore.halfToFloat(dminHalf)

    val scaleQuantizer = if (maxScale > 0.0f) 63.0f / maxScale else 0.0f
    val minQuantizer = if (maxMin > 0.0f) 63.0f / maxMin else 0.0f

    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
        val ls = io.github.kotlinmania.llama.ore.nearestIntFloat(scaleQuantizer * scales[subBlock])
            .coerceIn(0, 63)
        val lm = io.github.kotlinmania.llama.ore.nearestIntFloat(minQuantizer * mins[subBlock])
            .coerceIn(0, 63)
        if (subBlock < 4) {
            dest[scalesOffset + subBlock] = ls.toByte()
            dest[scalesOffset + subBlock + 4] = lm.toByte()
        } else {
            val packedIndex = scalesOffset + subBlock + 4
            dest[packedIndex] = io.github.kotlinmania.llama.ore.packNibblesInt(
                ls and 0xF,
                lm and 0xF
            ).toByte()

            val scaleHighIdx = scalesOffset + subBlock - 4
            dest[scaleHighIdx] = io.github.kotlinmania.llama.ore.bitplaneWriteInt(
                dest[scaleHighIdx].toInt() and 0xFF,
                ls ushr 4,
                6,
                2
            ).toByte()

            val minHighIdx = scalesOffset + subBlock
            dest[minHighIdx] = io.github.kotlinmania.llama.ore.bitplaneWriteInt(
                dest[minHighIdx].toInt() and 0xFF,
                lm ushr 4,
                6,
                2
            ).toByte()
        }
    }

    for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
        quants[i] = 0
    }

    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K / 32) {
        val (scaleInt, minInt) = io.github.kotlinmania.llama.ore.getScaleMinK4(
            dest,
            scalesOffset,
            subBlock
        )
        val scaleValue = d * scaleInt
        val minValue = dmin * minInt
        val base = subBlock * 32
        if (scaleValue == 0.0f) {
            continue
        }
        for (lane in 0 until 32) {
            val value = blockValues[base + lane]
            val q = io.github.kotlinmania.llama.ore.nearestIntFloat((value + minValue) / scaleValue)
                .coerceIn(0, 15)
            quants[base + lane] = q.toByte()
        }
    }

    var writeOffset = scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE
    for (chunkStart in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
        for (l in 0 until 32) {
            val low = quants[chunkStart + l].toInt() and 0xF
            val high = quants[chunkStart + l + 32].toInt() and 0xF
            dest[writeOffset + l] = io.github.kotlinmania.llama.ore.packNibblesInt(low, high).toByte()
        }
        writeOffset += 32
    }
}

private fun getScaleMinK4(scales: ByteArray, scalesOffset: Int, index: Int): Pair<Int, Int> {
    return if (index < 4) {
        val scale = io.github.kotlinmania.llama.ore.maskLowBits32(
            io.github.kotlinmania.llama.ore.unsignedByte(scales[scalesOffset + index]), 6
        )
        val min = io.github.kotlinmania.llama.ore.maskLowBits32(
            io.github.kotlinmania.llama.ore.unsignedByte(scales[scalesOffset + index + 4]), 6
        )
        scale to min
    } else {
        val packed = io.github.kotlinmania.llama.ore.unsignedByte(scales[scalesOffset + index + 4])
        val scaleHigh = io.github.kotlinmania.llama.ore.extractBits(
            io.github.kotlinmania.llama.ore.unsignedByte(scales[scalesOffset + index - 4]), 6, 2
        )
        val minHigh = io.github.kotlinmania.llama.ore.extractBits(
            io.github.kotlinmania.llama.ore.unsignedByte(scales[scalesOffset + index]), 6, 2
        )
        val scale = io.github.kotlinmania.llama.ore.mergeBits32(
            io.github.kotlinmania.llama.ore.maskLowBits32(
                packed,
                4
            ), io.github.kotlinmania.llama.ore.logicalLeft32(scaleHigh, 4)
        )
        val min = io.github.kotlinmania.llama.ore.mergeBits32(
            io.github.kotlinmania.llama.ore.extractBits(
                packed,
                4,
                4
            ), io.github.kotlinmania.llama.ore.logicalLeft32(minHigh, 4)
        )
        scale to min
    }
}

/**
 * Quantizes a block of QK_K float values to Q5_K format.
 * Q5_K structure: d (F16), dmin (F16), scales[io.github.kotlinmania.llama.ore.K_SCALE_SIZE], qh[QK_K/8], qs[QK_K/2]
 * Effectively 5.5 bits per weight
 */

private class Q5KScratch(
    val quants: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K),
    val aux: ByteArray = ByteArray(32),
    val weights: FloatArray = FloatArray(32),
    val mins: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val scales: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val sw: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val ls: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val lm: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32)
)

private fun quantizeQ5_KBlock(
    blockValues: FloatArray,
    dest: ByteArray,
    destOffset: Int,
    scratch: io.github.kotlinmania.llama.ore.Q5KScratch
) {
    require(blockValues.size == io.github.kotlinmania.llama.ore.QK_K) { "Q5_K block must have ${io.github.kotlinmania.llama.ore.QK_K} values" }

    val quants = scratch.quants
    val aux = scratch.aux
    val weights = scratch.weights
    val mins = scratch.mins
    val scales = scratch.scales
    val sw = scratch.sw
    val ls = scratch.ls
    val lm = scratch.lm

    sw.fill(0f)
    var sumX2 = 0.0f
    for (value in blockValues) sumX2 += value * value
    val sigma2 = sumX2 / io.github.kotlinmania.llama.ore.QK_K
    val subBlocks = io.github.kotlinmania.llama.ore.QK_K / 32
    for (sub in 0 until subBlocks) {
        val base = sub * 32
        var sumW = 0.0f
        for (i in 0 until 32) {
            val v = blockValues[base + i]
            val w = sqrt(sigma2 + v * v)
            weights[i] = w
            sumW += w
        }
        sw[sub] = sumW
        val stats = io.github.kotlinmania.llama.ore.makeQKX3Quants(
            n = 32,
            nmax = 31,
            values = blockValues,
            valuesOffset = base,
            weights = weights,
            weightsOffset = 0,
            dest = quants,
            destOffset = base,
            mins = mins,
            minsIndex = sub,
            aux = aux,
            auxOffset = 0,
            rmin = -0.9f,
            rdelta = 0.05f,
            nstep = 36,
            useMad = false
        )
        scales[sub] = stats.scale
    }

    var dm = io.github.kotlinmania.llama.ore.makeQPQuants(
        n = subBlocks,
        nmax = 15,
        values = scales,
        valuesOffset = 0,
        dest = ls,
        destOffset = 0,
        quantWeights = sw,
        weightsOffset = 0
    )
    var mm = io.github.kotlinmania.llama.ore.makeQPQuants(
        n = subBlocks,
        nmax = 15,
        values = mins,
        valuesOffset = 0,
        dest = lm,
        destOffset = 0,
        quantWeights = sw,
        weightsOffset = 0
    )

    dest.setShortLe(destOffset, io.github.kotlinmania.llama.ore.floatToHalf(dm))
    dest.setShortLe(destOffset + 2, io.github.kotlinmania.llama.ore.floatToHalf(mm))
    dm = io.github.kotlinmania.llama.ore.halfToFloat(dest.getShortLe(destOffset))
    mm = io.github.kotlinmania.llama.ore.halfToFloat(dest.getShortLe(destOffset + 2))

    val scalesOffset = destOffset + 4
    for (i in 0 until io.github.kotlinmania.llama.ore.K_SCALE_SIZE) dest[scalesOffset + i] = 0

    for (sub in 0 until subBlocks) {
        val lsInt = ls[sub].toInt() and 0xFF
        val lmInt = lm[sub].toInt() and 0xFF
        if (sub < 4) {
            dest[scalesOffset + sub] = lsInt.toByte()
            dest[scalesOffset + sub + 4] = lmInt.toByte()
        } else {
            dest[scalesOffset + sub + 4] = io.github.kotlinmania.llama.ore.packNibblesInt(
                lsInt and 0xF,
                lmInt and 0xF
            ).toByte()
            val scaleHighIdx = scalesOffset + sub - 4
            dest[scaleHighIdx] = io.github.kotlinmania.llama.ore.bitplaneWriteInt(
                dest[scaleHighIdx].toInt() and 0xFF,
                lsInt ushr 4,
                6,
                2
            ).toByte()
            val minHighIdx = scalesOffset + sub
            dest[minHighIdx] = io.github.kotlinmania.llama.ore.bitplaneWriteInt(
                dest[minHighIdx].toInt() and 0xFF,
                lmInt ushr 4,
                6,
                2
            ).toByte()
        }
    }

    dest.fill(0.toByte(), scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE, scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 8)
    dest.fill(0.toByte(), scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 8, scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 2)

    val qhOffset = scalesOffset + io.github.kotlinmania.llama.ore.K_SCALE_SIZE
    val qsOffset = qhOffset + io.github.kotlinmania.llama.ore.QK_K / 8

    for (sub in 0 until subBlocks) {
        val (scaleInt, minInt) = io.github.kotlinmania.llama.ore.getScaleMinK4(
            dest,
            scalesOffset,
            sub
        )
        val scaleValue = dm * scaleInt
        if (scaleValue == 0.0f) continue
        val minValue = mm * minInt
        val base = sub * 32
        for (lane in 0 until 32) {
            val q = io.github.kotlinmania.llama.ore.nearestIntFloat((blockValues[base + lane] + minValue) / scaleValue)
                .coerceIn(0, 31)
            quants[base + lane] = q.toByte()
        }
    }

    var mask1 = 1
    var mask2 = 2
    var qsIndex = qsOffset
    for (chunk in 0 until io.github.kotlinmania.llama.ore.QK_K step 64) {
        for (j in 0 until 32) {
            var q1 = quants[chunk + j].toInt() and 0x1F
            var q2 = quants[chunk + j + 32].toInt() and 0x1F
            if (q1 > 15) {
                q1 -= 16
                dest[qhOffset + j] = (dest[qhOffset + j].toInt() or mask1).toByte()
            }
            if (q2 > 15) {
                q2 -= 16
                dest[qhOffset + j] = (dest[qhOffset + j].toInt() or mask2).toByte()
            }
            dest[qsIndex + j] = io.github.kotlinmania.llama.ore.packNibblesInt(q1, q2).toByte()
        }
        mask1 = (mask1 shl 2) and 0xFF
        mask2 = (mask2 shl 2) and 0xFF
        qsIndex += 32
    }
}

/**
 * Quantizes a block of QK_K float values to Q6_K format.
 * Q6_K structure: ql[QK_K/2], qh[QK_K/4], scales[QK_K/16], d (F16)
 * Effectively 6.5625 bits per weight
 */

private class Q6KScratch(
    val quants: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K),
    val scales: FloatArray = FloatArray(io.github.kotlinmania.llama.ore.QK_K / 16)
)

private fun quantizeQ6_KBlock(
    blockValues: FloatArray,
    dest: ByteArray,
    destOffset: Int,
    scratch: io.github.kotlinmania.llama.ore.Q6KScratch
) {
    require(blockValues.size == io.github.kotlinmania.llama.ore.QK_K) { "Q6_K block must have ${io.github.kotlinmania.llama.ore.QK_K} values" }

    val quants = scratch.quants
    val scales = scratch.scales

    val subBlocks = io.github.kotlinmania.llama.ore.QK_K / 16
    var maxScale = 0.0f
    var maxAbsScale = 0.0f
    for (sub in 0 until subBlocks) {
        val base = sub * 16
        val scale = io.github.kotlinmania.llama.ore.makeQXQuants(
            n = 16,
            nmax = 32,
            values = blockValues,
            valuesOffset = base,
            dest = quants,
            destOffset = base,
            rmseTypeInput = 1,
            quantWeights = null,
            weightsOffset = 0
        )
        scales[sub] = scale
        val absScale = abs(scale)
        if (absScale > maxAbsScale) {
            maxAbsScale = absScale
            maxScale = scale
        }
    }

    val qlOffset = destOffset
    val qhOffset = destOffset + io.github.kotlinmania.llama.ore.QK_K / 2
    val scalesOffset = qhOffset + io.github.kotlinmania.llama.ore.QK_K / 4
    val dOffset = scalesOffset + io.github.kotlinmania.llama.ore.QK_K / 16

    dest.fill(0.toByte(), qlOffset, qlOffset + io.github.kotlinmania.llama.ore.QK_K / 2)
    dest.fill(0.toByte(), qhOffset, qhOffset + io.github.kotlinmania.llama.ore.QK_K / 4)
    dest.fill(0.toByte(), scalesOffset, scalesOffset + io.github.kotlinmania.llama.ore.QK_K / 16)

    if (maxAbsScale < io.github.kotlinmania.llama.ore.GROUP_MAX_EPS_F) {
        dest.setShortLe(dOffset, io.github.kotlinmania.llama.ore.floatToHalf(0f))
        return
    }

    val iscale = -128f / maxScale
    dest.setShortLe(dOffset, io.github.kotlinmania.llama.ore.floatToHalf(1f / iscale))
    val dBase = io.github.kotlinmania.llama.ore.halfToFloat(dest.getShortLe(dOffset))

    for (sub in 0 until subBlocks) {
        val qScale = io.github.kotlinmania.llama.ore.nearestIntFloat(iscale * scales[sub])
            .coerceIn(-128, 127)
        dest[scalesOffset + sub] = qScale.toByte()
    }

    for (sub in 0 until subBlocks) {
        val scaleInt = dest[scalesOffset + sub].toInt()
        val scaleValue = dBase * scaleInt
        if (scaleValue == 0.0f) continue
        val base = sub * 16
        for (lane in 0 until 16) {
            val q = io.github.kotlinmania.llama.ore.nearestIntFloat(blockValues[base + lane] / scaleValue)
                .coerceIn(-32, 31)
            quants[base + lane] = (q + 32).toByte()
        }
    }

    var qlIndex = qlOffset
    var qhIndex = qhOffset
    for (chunk in 0 until io.github.kotlinmania.llama.ore.QK_K step 128) {
        for (l in 0 until 32) {
            val q1 = quants[chunk + l].toInt() and 0x3F
            val q2 = quants[chunk + l + 32].toInt() and 0x3F
            val q3 = quants[chunk + l + 64].toInt() and 0x3F
            val q4 = quants[chunk + l + 96].toInt() and 0x3F

            dest[qlIndex + l] = io.github.kotlinmania.llama.ore.packNibblesInt(
                q1 and 0xF,
                q3 and 0xF
            ).toByte()
            dest[qlIndex + l + 32] = io.github.kotlinmania.llama.ore.packNibblesInt(
                q2 and 0xF,
                q4 and 0xF
            ).toByte()

            val highPacked = ((q4 shr 4) shl 6) or ((q3 shr 4) shl 4) or ((q2 shr 4) shl 2) or (q1 shr 4)
            dest[qhIndex + l] = highPacked.toByte()
        }
        qlIndex += 64
        qhIndex += 32
    }
}

/**
 * Quantizes a block of QK_K float values to Q8_K format.
 * Q8_K structure: d (F32), qs[io.github.kotlinmania.llama.ore.QK_K], bsums[QK_K/16]
 * This is used for intermediate quantization and dot products
 */
private fun quantizeQ8_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == io.github.kotlinmania.llama.ore.QK_K) { "Q8_K block must have ${io.github.kotlinmania.llama.ore.QK_K} values" }
    
    // Find absolute maximum
    var amax = 0.0f
    for (value in blockValues) {
        amax = maxOf(amax, abs(value))
    }
    
    val d = if (amax > 0.0f) amax / 127.0f else 1.0f
    val invD = if (d > 0.0f) 1.0f / d else 0.0f
    
    // Write scale as F32
    dest.setFloatLe(destOffset, d)
    
    // Quantize all values to 8 bits
    for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
        val quantizedValue = round(blockValues[i] * invD).toInt().coerceIn(-128, 127)
        dest[destOffset + 4 + i] = quantizedValue.toByte()
    }
    
    // Calculate block sums for each group of 16
    for (group in 0 until io.github.kotlinmania.llama.ore.QK_K /16) {
        var sum = 0
        for (i in 0 until 16) {
            val idx = group * 16 + i
            sum += dest[destOffset + 4 + idx].toInt()
        }
        // Store sum as 16-bit integer
        dest.setShortLe(destOffset + 4 + io.github.kotlinmania.llama.ore.QK_K + group * 2, sum.toShort())
    }
}

// K-Quant Block Dequantization Functions

/**
 * Dequantizes a Q2_K block to float values.
 */
private fun dequantizeQ2_KBlock(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ2_KBlockScale(graphAllocator, blockIndex)
    val dmin = tensor.getQ2_KBlockScaleMin(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Process 16-element sub-blocks (Q2_K has 16 sub-blocks of 16 elements each)
    for (subBlock in 0 until 16) {
        // Get quantized scale for this sub-block
        val scaleAndMin = tensor.getQ2_KScale(graphAllocator, blockIndex, subBlock)
        val quantizedScale = io.github.kotlinmania.llama.ore.maskLowBits32(scaleAndMin.toInt(), 4)
        
        // Reconstruct scale (Q2_K uses simple scale mapping)
        val scale = (quantizedScale.toFloat() / 15.0f) * d
        
        // Dequantize 16 values (4 values per byte, 2 bits each)
        for (i in 0 until 16 step 4) {
            val packedByte = tensor.getQ2_KQuant(graphAllocator, blockIndex, subBlock * 4 + i / 4)
            
            for (j in 0 until 4) {
                if (elementIdx < dest.size) {
                    val quantizedValue = io.github.kotlinmania.llama.ore.extractBits(
                        io.github.kotlinmania.llama.ore.unsignedByte(packedByte), j * 2, 2
                    )
                    val dequantizedValue = (quantizedValue.toFloat() / 3.0f) * scale + dmin
                    dest[elementIdx++] = dequantizedValue
                }
            }
        }
    }
}

/**
 * Dequantizes a Q3_K block to float values.
 */
private fun dequantizeQ3_KBlock(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ3_KBlockScale(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Process 16-element sub-blocks
    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K /16) {
        // Get quantized scale for this sub-block (stored in scales array)
        val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
        val scaleOffset = blockByteOffset + io.github.kotlinmania.llama.ore.QK_K /8 + io.github.kotlinmania.llama.ore.QK_K /4 + subBlock
        val buffer = graphAllocator.buffers[tensor.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        val quantizedScale = buffer[(tensor.dataOffset + scaleOffset.toULong()).toInt()]
        
        // Reconstruct scale
        val scale = ((io.github.kotlinmania.llama.ore.maskLowBits32(quantizedScale.toInt(), 6)).toFloat() / 63.0f) * d
        
        // Dequantize 16 values
        val subBlockStart = subBlock * 16
        for (i in 0 until 16) {
            if (elementIdx < dest.size) {
                val globalIdx = subBlockStart + i
                
                // Get low 2 bits from qs
                val qsByteIdx = globalIdx / 4
                val qsBitPos = (globalIdx % 4) * 2
                val qsOffset = blockByteOffset + io.github.kotlinmania.llama.ore.QK_K /8 + qsByteIdx
                val qsValue = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(buffer[(tensor.dataOffset + qsOffset.toULong()).toInt()]),
                    qsBitPos,
                    2
                )
                
                // Get high bit from hmask
                val hmaskByteIdx = globalIdx / 8
                val hmaskBitPos = globalIdx % 8
                val hmaskOffset = blockByteOffset + hmaskByteIdx
                val hmaskValue = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(buffer[(tensor.dataOffset + hmaskOffset.toULong()).toInt()]),
                    hmaskBitPos,
                    1
                )
                
                // Combine to get 3-bit value
                val quantizedValue = qsValue or (io.github.kotlinmania.llama.ore.logicalLeft32(
                    hmaskValue,
                    2
                ))
                val signedValue = if (quantizedValue > 3) quantizedValue - 8 else quantizedValue
                
                dest[elementIdx++] = signedValue.toFloat() * scale
            }
        }
    }
}

/**
 * Dequantizes a Q4_K block to float values.
 */
private fun dequantizeQ4_KBlock(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ4_KBlockScale(graphAllocator, blockIndex)
    val dmin = tensor.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Process 32-element sub-blocks
    for (subBlock in 0 until 8) {
        // Get quantized scale and min for this sub-block from the scales array
        val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
        val buffer = graphAllocator.buffers[tensor.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
        
        // Read packed scale and min values
        val scaleByteOffset = blockByteOffset + 4 + subBlock
        val scaleByte = buffer[(tensor.dataOffset + scaleByteOffset.toULong()).toInt()]
        val quantizedScale = scaleByte.toUnsignedInt().lowBits(6)
        val quantizedMinLow =
            io.github.kotlinmania.llama.ore.extractBits(scaleByte.toUnsignedInt(), 6, 2)
        
        val minByteOffset = blockByteOffset + 4 + subBlock * 2 + 1
        val quantizedMinHigh = if (minByteOffset < blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE) {
            io.github.kotlinmania.llama.ore.maskLowBits32(
                io.github.kotlinmania.llama.ore.unsignedByte(
                    buffer[(tensor.dataOffset + minByteOffset.toULong()).toInt()]
                ), 4
            )
        } else 0
        val quantizedMin = io.github.kotlinmania.llama.ore.mergeBits32(
            quantizedMinLow,
            io.github.kotlinmania.llama.ore.logicalLeft32(quantizedMinHigh, 2)
        )
        
        // Reconstruct scale and min
        val scale = (quantizedScale.toFloat() / 63.0f) * d
        val min = (quantizedMin.toFloat() / 63.0f) * d + dmin
        
        // Dequantize 32 values (2 values per byte, 4 bits each)
        val qsBaseOffset = blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + subBlock * 16
        for (i in 0 until 32 step 2) {
            if (elementIdx < dest.size) {
                val qsByte = buffer[(tensor.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                
                val q1 = qsByte.getBits(0, 4)
                val q2 = qsByte.getBits(4, 4)
                
                dest[elementIdx++] = (q1.toFloat() / 15.0f) * scale + min
                if (elementIdx < dest.size) {
                    dest[elementIdx++] = (q2.toFloat() / 15.0f) * scale + min
                }
            }
        }
    }
}

/**
 * Dequantizes a Q5_K block to float values.
 */
private fun dequantizeQ5_KBlock(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ5_KBlockScale(graphAllocator, blockIndex)
    val buffer = graphAllocator.buffers[tensor.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
    val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
    
    var elementIdx = destOffset
    
    // Process 32-element sub-blocks  
    for (subBlock in 0 until 8) {
        // Get quantized scale and min for this sub-block
        val scaleByteOffset = blockByteOffset + 4 + subBlock
        val scaleByte = buffer[(tensor.dataOffset + scaleByteOffset.toULong()).toInt()]
        val quantizedScale = scaleByte.toUnsignedInt().lowBits(6)
        val quantizedMin =
            io.github.kotlinmania.llama.ore.extractBits(scaleByte.toUnsignedInt(), 6, 2)
        
        val scale = (quantizedScale.toFloat() / 63.0f) * d
        
        // Dequantize 32 values (5-bit: 4 bits in qs, 1 bit in qh)
        val qsBaseOffset = blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K /8 + subBlock * 16
        val qhBaseOffset = blockByteOffset + 4 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE
        
        for (i in 0 until 32 step 2) {
            if (elementIdx < dest.size) {
                // Get 4-bit values from qs
                val qsByte = buffer[(tensor.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                val qs1 = qsByte.getBits(0, 4)
                val qs2 = qsByte.getBits(4, 4)
                
                // Get high bits from qh
                val globalIdx1 = subBlock * 32 + i
                val globalIdx2 = subBlock * 32 + i + 1
                val qhByte1 = buffer[(tensor.dataOffset + qhBaseOffset.toULong() + (globalIdx1 / 8).toULong()).toInt()]
                val qhByte2 = buffer[(tensor.dataOffset + qhBaseOffset.toULong() + (globalIdx2 / 8).toULong()).toInt()]
                
                val qh1 = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(qhByte1), globalIdx1 % 8, 1
                )
                val qh2 = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(qhByte2), globalIdx2 % 8, 1
                )
                
                // Combine to get 5-bit values
                val q1 = io.github.kotlinmania.llama.ore.mergeBits32(
                    qs1,
                    io.github.kotlinmania.llama.ore.logicalLeft32(qh1, 4)
                )
                val q2 = io.github.kotlinmania.llama.ore.mergeBits32(
                    qs2,
                    io.github.kotlinmania.llama.ore.logicalLeft32(qh2, 4)
                )
                
                dest[elementIdx++] = (q1.toFloat() / 31.0f) * scale
                if (elementIdx < dest.size) {
                    dest[elementIdx++] = (q2.toFloat() / 31.0f) * scale
                }
            }
        }
    }
}

/**
 * Dequantizes a Q6_K block to float values.
 */
private fun dequantizeQ6_KBlock(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ6_KBlockScale(graphAllocator, blockIndex)
    val buffer = graphAllocator.buffers[tensor.bufferId] as? ByteArray ?: throw IllegalStateException("Tensor buffer not found")
    val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
    
    var elementIdx = destOffset
    
    // Process 16-element sub-blocks
    for (subBlock in 0 until io.github.kotlinmania.llama.ore.QK_K /16) {
        // Get 8-bit scale for this sub-block
        val scaleOffset = blockByteOffset + io.github.kotlinmania.llama.ore.QK_K /2 + io.github.kotlinmania.llama.ore.QK_K /4 + subBlock
        val quantizedScale = buffer[(tensor.dataOffset + scaleOffset.toULong()).toInt()]
        val scale = (quantizedScale.toFloat() / 127.0f) * d
        
        // Dequantize 16 values (6-bit: 4 bits in ql, 2 bits in qh)
        val qlBaseOffset = blockByteOffset + subBlock * 8
        val qhBaseOffset = blockByteOffset + io.github.kotlinmania.llama.ore.QK_K /2 + subBlock * 4
        
        for (i in 0 until 16 step 2) {
            if (elementIdx < dest.size) {
                // Get 4-bit values from ql
                val qlByte = buffer[(tensor.dataOffset + qlBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                val ql1 = io.github.kotlinmania.llama.ore.maskLowBits32(
                    io.github.kotlinmania.llama.ore.unsignedByte(qlByte), 4
                )
                val ql2 = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(qlByte), 4, 4
                )
                
                // Get 2-bit values from qh
                val qhByte = buffer[(tensor.dataOffset + qhBaseOffset.toULong() + (i / 4).toULong()).toInt()]
                val qhBitPos = (i % 4) * 2
                val qh1 = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(qhByte), qhBitPos, 2
                )
                val qh2 = io.github.kotlinmania.llama.ore.extractBits(
                    io.github.kotlinmania.llama.ore.unsignedByte(qhByte), qhBitPos + 2, 2
                )
                
                // Combine to get 6-bit values
                val q1 = io.github.kotlinmania.llama.ore.mergeBits32(
                    ql1,
                    io.github.kotlinmania.llama.ore.logicalLeft32(qh1, 4)
                )
                val q2 = io.github.kotlinmania.llama.ore.mergeBits32(
                    ql2,
                    io.github.kotlinmania.llama.ore.logicalLeft32(qh2, 4)
                )
                
                dest[elementIdx++] = ((q1.toFloat() - 32.0f) / 63.0f) * scale
                if (elementIdx < dest.size) {
                    dest[elementIdx++] = ((q2.toFloat() - 32.0f) / 63.0f) * scale
                }
            }
        }
    }
}

/**
 * Dequantizes a Q8_K block to float values.
 */
private fun dequantizeQ8_KBlock(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, tensor: io.github.kotlinmania.llama.ore.GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ8_KBlockScale(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Simple 8-bit dequantization
    for (i in 0 until io.github.kotlinmania.llama.ore.QK_K) {
        if (elementIdx < dest.size) {
            val quantizedValue = tensor.getQ8_KWeight(graphAllocator, blockIndex, i)
            dest[elementIdx++] = quantizedValue.toFloat() * d
        }
    }
}

/**
 * Compute element-wise square (SQR) into destination tensor (dst-arg only).
 */
fun computeSqr(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    dst.type = a.type
    a.ne.copyInto(dst.ne); a.nb.copyInto(dst.nb)
    val ts = a.numElements().toInt()
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            val resultData = FloatArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getFloat(graphAllocator, *ind)
                resultData[flatIdx] = value * value
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getHalf(graphAllocator, *ind)
                resultData[flatIdx] = io.github.kotlinmania.llama.ore.floatToHalf(value * value)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            val resultData = IntArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getInt(graphAllocator, *indices).toLong()
                val squared = value * value
                resultData[flatIdx] = when {
                    squared > Int.MAX_VALUE -> Int.MAX_VALUE
                    squared < Int.MIN_VALUE -> Int.MIN_VALUE
                    else -> squared.toInt()
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getShort(graphAllocator, *indices).toInt()
                val squared = value * value
                resultData[flatIdx] = squared.coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
            val resultData = LongArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getLong(graphAllocator, *indices)
                // Avoid overflow for Long
                dst.data as LongArray
                if (kotlin.math.abs(value) > 3037000499L) {
                    (dst.data as LongArray)[flatIdx] = if (value >= 0) Long.MAX_VALUE else Long.MAX_VALUE
                } else {
                    (dst.data as LongArray)[flatIdx] = value * value
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            val af = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a)
            val tmp = io.github.kotlinmania.llama.ore.GGMLTensor(af.type)
                .also { it.ne = a.ne.copyOf(); it.nb = a.nb.copyOf() }
            computeSqr(graphAllocator, af, tmp)
            val qr = io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tmp, a.type)
            dst.data = qr.data
        }
        else -> error("fatal error")
    }
}

/**
 * Compute element-wise square root (SQRT) into destination tensor (dst-arg only).
 */
fun computeSqrt(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    dst.type = a.type
    a.ne.copyInto(dst.ne); a.nb.copyInto(dst.nb)
    val ts = a.numElements().toInt()
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            val resultData = FloatArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getFloat(graphAllocator, *ind)
                resultData[flatIdx] = if (value < 0.0f) Float.NaN else kotlin.math.sqrt(value)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getHalf(graphAllocator, *ind)
                val sqrtValue = if (value < 0.0f) Float.NaN else kotlin.math.sqrt(value)
                resultData[flatIdx] = io.github.kotlinmania.llama.ore.floatToHalf(sqrtValue)
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
            val resultData = IntArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getInt(graphAllocator, *indices)
                require(value >= 0) { "Cannot compute square root of negative integer: $value" }
                resultData[flatIdx] = kotlin.math.sqrt(value.toDouble()).toInt()
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getShort(graphAllocator, *indices).toInt()
                require(value >= 0) { "Cannot compute square root of negative integer: $value" }
                resultData[flatIdx] = kotlin.math.sqrt(value.toDouble()).toInt()
                    .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
            val resultData = LongArray(ts)
            dst.data = resultData
            io.github.kotlinmania.llama.ore.applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getLong(graphAllocator, *indices)
                require(value >= 0) { "Cannot compute square root of negative long: $value" }
                resultData[flatIdx] = kotlin.math.sqrt(value.toDouble()).toLong()
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            val af = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, a)
            val tmp = io.github.kotlinmania.llama.ore.GGMLTensor(af.type)
                .also { it.ne = a.ne.copyOf(); it.nb = a.nb.copyOf() }
            computeSqrt(graphAllocator, af, tmp)
            val qr = io.github.kotlinmania.llama.ore.quantizeTensor(graphAllocator, tmp, a.type)
            dst.data = qr.data
        }
        else -> error("fatal error")
    }
}

/**
 * Transpose operation for tensor matrices.
 * Swaps the last two dimensions of a tensor (for 2D case, transposes rows and columns).
 * For higher-dimensional tensors, transposes the last two dimensions while preserving others.
 */
fun computeTranspose(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, a: io.github.kotlinmania.llama.ore.GGMLTensor, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    // Validate dimensions - dst should have swapped last two dimensions compared to src
    require(a.ne[0] == dst.ne[1] && a.ne[1] == dst.ne[0]) {
        "Transpose dimensions mismatch: src(${a.ne[0]}, ${a.ne[1]}) vs dst(${dst.ne[0]}, ${dst.ne[1]})"
    }
    
    // For dimensions beyond the first two, they should match
    for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        require(a.ne[i] == dst.ne[i]) {
            "Higher dimensions must match for transpose: src.ne[$i]=${a.ne[i]} vs dst.ne[$i]=${dst.ne[i]}"
        }
    }
    
    require(a.type == dst.type) {
        "Source and destination tensors must have the same type"
    }
    
    when (a.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            val rows = a.ne[1].toInt()
            val cols = a.ne[0].toInt()
            val batches = (2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS).fold(1L) { acc, dim -> acc * a.ne[dim] }.toInt()
            
            for (batch in 0 until batches) {
                val batchOffset = io.github.kotlinmania.llama.ore.calculateBatchOffset(a, batch)
                for (i in 0 until rows) {
                    for (j in 0 until cols) {
                        val srcIndices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { dim ->
                            when (dim) {
                                0 -> j
                                1 -> i
                                else -> (batchOffset / io.github.kotlinmania.llama.ore.calculateStrideFactor(
                                    a,
                                    dim
                                )) % a.ne[dim].toInt()
                            }
                        }
                        val dstIndices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { dim ->
                            when (dim) {
                                0 -> i  // Swapped
                                1 -> j  // Swapped
                                else -> srcIndices[dim]
                            }
                        }
                        val value = a.getFloat(graphAllocator, *srcIndices)
                        dst.setFloat(graphAllocator, value, *dstIndices)
                    }
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            val rows = a.ne[1].toInt()
            val cols = a.ne[0].toInt()
            val batches = (2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS).fold(1L) { acc, dim -> acc * a.ne[dim] }.toInt()
            
            for (batch in 0 until batches) {
                val batchOffset = io.github.kotlinmania.llama.ore.calculateBatchOffset(a, batch)
                for (i in 0 until rows) {
                    for (j in 0 until cols) {
                        val srcIndices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { dim ->
                            when (dim) {
                                0 -> j
                                1 -> i
                                else -> (batchOffset / io.github.kotlinmania.llama.ore.calculateStrideFactor(
                                    a,
                                    dim
                                )) % a.ne[dim].toInt()
                            }
                        }
                        val dstIndices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { dim ->
                            when (dim) {
                                0 -> i  // Swapped
                                1 -> j  // Swapped
                                else -> srcIndices[dim]
                            }
                        }
                        val value = a.getHalf(graphAllocator, *srcIndices)
                        dst.setHalf(graphAllocator, value, *dstIndices)
                    }
                }
            }
        }
        else -> error("fatal error")
    }
}

/**
 * Helper function to calculate batch offset for higher-dimensional transpose
 */
private fun calculateBatchOffset(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, batchIndex: Int): Int {
    var offset = batchIndex
    var stride = 1
    for (dim in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        stride *= tensor.ne[dim].toInt()
    }
    return offset * tensor.ne[0].toInt() * tensor.ne[1].toInt()
}

/**
 * Helper function to calculate stride factor for dimension calculations
 */
private fun calculateStrideFactor(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, dim: Int): Int {
    var factor = 1
    for (i in 0 until dim) {
        factor *= tensor.ne[i].toInt()
    }
    return factor
}

// ============================================================================
// Structural / advanced compute-forward operations
// Ported from ggml-cpu/ops.cpp (lines ~3000-11214)
// ============================================================================

// ---------------------------------------------------------------------------
// ggml_compute_forward_concat
// ---------------------------------------------------------------------------

/**
 * Concatenates two tensors along the dimension stored in `dst.opParams[0]`.
 *
 * For elements whose indices fall within the bounds of src0 along **all** four
 * dimensions, the value is copied from src0.  Otherwise the value comes from
 * src1 after subtracting the src0 extent along the concat dimension.
 *
 * Ported from `ggml_compute_forward_concat_f32` and the generic path.
 */
fun computeConcat(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("CONCAT requires src0")
    val src1 = dst.src[1] ?: error("CONCAT requires src1")

    val dim = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    require(dim in 0..3) { "concat dim must be 0..3, got $dim" }

    val o = LongArray(4)
    o[dim] = src0.ne[dim]

    val ne0 = dst.ne[0].toInt()
    val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt()
    val ne3 = dst.ne[3].toInt()
    val ne00 = src0.ne[0].toInt(); val ne01 = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt(); val ne03 = src0.ne[3].toInt()

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne1) {
                for (i0 in 0 until ne0) {
                    val fromSrc0 = i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03
                    val value: Float = if (fromSrc0) {
                        src0.getFloat(graphAllocator, i0, i1, i2, i3)
                    } else {
                        src1.getFloat(
                            graphAllocator,
                            (i0 - o[0].toInt()),
                            (i1 - o[1].toInt()),
                            (i2 - o[2].toInt()),
                            (i3 - o[3].toInt())
                        )
                    }
                    dst.setFloat(graphAllocator, value, i0, i1, i2, i3)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_sum_rows
// ---------------------------------------------------------------------------

/**
 * Sums each row of src0 along dimension 0.  The output has `ne0 == 1` and
 * the remaining dimensions match src0.
 *
 * Ported from `ggml_compute_forward_sum_rows_f32`.
 */
fun computeSumRows(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("SUM_ROWS requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "SUM_ROWS only supports F32, got ${src0.type}" }

    val ne00 = src0.ne[0].toInt()
    val ne01 = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt()
    val ne03 = src0.ne[3].toInt()

    for (i3 in 0 until ne03) {
        for (i2 in 0 until ne02) {
            for (i1 in 0 until ne01) {
                var rowSum = 0.0f
                for (i0 in 0 until ne00) {
                    rowSum += src0.getFloat(graphAllocator, i0, i1, i2, i3)
                }
                dst.setFloat(graphAllocator, rowSum, 0, i1, i2, i3)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_cumsum
// ---------------------------------------------------------------------------

/**
 * Computes the cumulative sum along dimension 0 for each row.
 *
 * Ported from `ggml_compute_forward_cumsum_f32`.
 */
fun computeCumsum(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("CUMSUM requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "CUMSUM only supports F32, got ${src0.type}" }

    val ne00 = src0.ne[0].toInt()
    val ne01 = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt()
    val ne03 = src0.ne[3].toInt()

    for (i3 in 0 until ne03) {
        for (i2 in 0 until ne02) {
            for (i1 in 0 until ne01) {
                var acc = 0.0f
                for (i0 in 0 until ne00) {
                    acc += src0.getFloat(graphAllocator, i0, i1, i2, i3)
                    dst.setFloat(graphAllocator, acc, i0, i1, i2, i3)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_argmax
// ---------------------------------------------------------------------------

/**
 * For each row (ne01 rows), finds the index of the maximum element along
 * dimension 0 and writes it as an I32 into dst.
 *
 * Ported from `ggml_compute_forward_argmax_f32`.
 */
fun computeArgmax(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("ARGMAX requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "ARGMAX only supports F32, got ${src0.type}" }

    val ne00 = src0.ne[0].toInt()
    val ne01 = src0.ne[1].toInt()

    for (i1 in 0 until ne01) {
        var maxIdx = 0
        var maxVal = src0.getFloat(graphAllocator, 0, i1)
        for (i0 in 1 until ne00) {
            val v = src0.getFloat(graphAllocator, i0, i1)
            if (v > maxVal) { maxVal = v; maxIdx = i0 }
        }
        dst.setInt(graphAllocator, maxIdx, i1)
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_count_equal
// ---------------------------------------------------------------------------

/**
 * Counts the number of element-wise equal I32 entries between src0 and src1.
 * Writes the result as a single I64 scalar into dst.
 *
 * Ported from `ggml_compute_forward_count_equal_i32` (single-thread path).
 */
fun computeCountEqual(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("COUNT_EQUAL requires src0")
    val src1 = dst.src[1] ?: error("COUNT_EQUAL requires src1")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.I32) { "COUNT_EQUAL only supports I32, got ${src0.type}" }
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.I32) { "COUNT_EQUAL only supports I32, got ${src1.type}" }

    val ne0 = src0.ne[0].toInt()
    val ne1 = src0.ne[1].toInt()
    val ne2 = src0.ne[2].toInt()
    val ne3 = src0.ne[3].toInt()

    var count = 0L
    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne1) {
                for (i0 in 0 until ne0) {
                    val v0 = src0.getInt(graphAllocator, i0, i1, i2, i3)
                    val v1 = src1.getInt(graphAllocator, i0, i1, i2, i3)
                    if (v0 == v1) count++
                }
            }
        }
    }
    dst.setLong(graphAllocator, count, 0)
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_get_rows
// ---------------------------------------------------------------------------

/**
 * Gathers rows from src0 using I32 indices in src1.
 * Output is always F32.  For quantized src0 types the values are dequantized
 * on the fly; for F16 they are promoted; for F32 they are copied directly.
 *
 * Ported from `ggml_compute_forward_get_rows_f32` / `_f16` (scalar paths).
 */
fun computeGetRows(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("GET_ROWS requires src0")
    val src1 = dst.src[1] ?: error("GET_ROWS requires src1 (indices)")

    val nc = src0.ne[0].toInt()    // columns per row
    val nr = src1.numElements().toInt()  // total index entries

    val ne10 = src1.ne[0].toInt()
    val ne11 = src1.ne[1].toInt()

    for (i in 0 until nr) {
        val i12 = i / (ne11 * ne10)
        val i11 = (i - i12 * ne11 * ne10) / ne10
        val i10 = i - i12 * ne11 * ne10 - i11 * ne10
        val i01 = src1.getInt(graphAllocator, i10, i11, i12)

        require(i01 >= 0 && i01 < src0.ne[1].toInt()) {
            "GET_ROWS: row index $i01 out of bounds (0..${src0.ne[1] - 1})"
        }

        when (src0.type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                for (j in 0 until nc) {
                    val v = src0.getFloat(graphAllocator, j, i01, i11, i12)
                    dst.setFloat(graphAllocator, v, j, i10, i11, i12)
                }
            }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                for (j in 0 until nc) {
                    val v = src0.getHalf(graphAllocator, j, i01, i11, i12)
                    dst.setFloat(graphAllocator, v, j, i10, i11, i12)
                }
            }
            io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
            io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
            io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
            io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
            io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
            io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
            io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
            io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
                // Dequantize source row to F32, then copy to dst
                val dequantized =
                    io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, src0)
                for (j in 0 until nc) {
                    val v = dequantized.getFloat(graphAllocator, j, i01, i11, i12)
                    dst.setFloat(graphAllocator, v, j, i10, i11, i12)
                }
            }
            else -> error("fatal error")
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_get_rows_back
// ---------------------------------------------------------------------------

/**
 * Backward pass for get_rows: scatters rows back into a zeroed tensor.
 * src0 contains the upstream gradients, src1 contains the row indices.
 * For duplicate indices the gradients are **accumulated** (summed).
 *
 * Ported from `ggml_compute_forward_get_rows_back_f32`.
 */
fun computeGetRowsBack(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("GET_ROWS_BACK requires src0")
    val src1 = dst.src[1] ?: error("GET_ROWS_BACK requires src1 (indices)")

    val nc = src0.ne[0].toInt()
    val nr = src1.numElements().toInt()

    // zero the destination
    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()
    for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) for (i0 in 0 until ne0)
        dst.setFloat(graphAllocator, 0.0f, i0, i1, i2, i3)

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            for (i in 0 until nr) {
                val r = src1.getInt(graphAllocator, i)
                for (j in 0 until nc) {
                    val cur = dst.getFloat(graphAllocator, j, r)
                    dst.setFloat(graphAllocator, cur + src0.getFloat(graphAllocator, j, i), j, r)
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            for (i in 0 until nr) {
                val r = src1.getInt(graphAllocator, i)
                for (j in 0 until nc) {
                    val cur = dst.getFloat(graphAllocator, j, r)
                    dst.setFloat(graphAllocator, cur + src0.getHalf(graphAllocator, j, i), j, r)
                }
            }
        }
        else -> error("fatal error")
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_set_rows
// ---------------------------------------------------------------------------

/**
 * Inverse of get_rows: copies each row of the F32 src0 into the destination
 * at the position given by the I32 index tensor src1.  The dst may be a
 * quantized type (the C++ version calls `from_float`); here only F32→F32 is
 * fully implemented.
 *
 * Ported from `ggml_compute_forward_set_rows_f32` (scalar, I32 indices).
 */
fun computeSetRows(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("SET_ROWS requires src0")
    val src1 = dst.src[1] ?: error("SET_ROWS requires src1 (indices)")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "SET_ROWS: src0 must be F32, got ${src0.type}" }

    val nc = src0.ne[0].toInt()
    val nr = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt()
    val ne03 = src0.ne[3].toInt()
    val ne11 = src1.ne[1].toInt()
    val ne12 = src1.ne[2].toInt()

    for (i03 in 0 until ne03) {
        for (i02 in 0 until ne02) {
            for (i in 0 until nr) {
                val i12 = i03 % ne12
                val i11 = i02 % ne11
                val i1 = src1.getInt(graphAllocator, i, i11, i12)

                require(i1 >= 0 && i1 < dst.ne[1].toInt()) {
                    "SET_ROWS: index $i1 out of bounds (0..${dst.ne[1] - 1})"
                }

                for (j in 0 until nc) {
                    val v = src0.getFloat(graphAllocator, j, i, i02, i03)
                    dst.setFloat(graphAllocator, v, j, i1, i02, i03)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_diag
// ---------------------------------------------------------------------------

/**
 * Creates a diagonal matrix from a 1-D vector.
 * src0 shape: `[ne00, 1, ne02, ne03]`  →  dst shape: `[ne00, ne00, ne02, ne03]`
 * Off-diagonal elements are set to zero.
 *
 * Ported from `ggml_compute_forward_diag_f32`.
 */
fun computeDiag(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("DIAG requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "DIAG only supports F32, got ${src0.type}" }

    val ne00 = src0.ne[0].toInt()
    val ne2 = dst.ne[2].toInt()
    val ne3 = dst.ne[3].toInt()

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne00) {
                for (i0 in 0 until ne00) {
                    val v = if (i0 == i1) src0.getFloat(graphAllocator, i1, 0, i2, i3) else 0.0f
                    dst.setFloat(graphAllocator, v, i0, i1, i2, i3)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_diag_mask_inf / diag_mask_zero
// ---------------------------------------------------------------------------

/**
 * Applies a causal (upper-triangular) mask to an F32 tensor.
 * Elements where `i0 > n_past + i1` are replaced with [maskValue]
 * (`-Infinity` for diag_mask_inf, `0` for diag_mask_zero).
 *
 * Ported from `ggml_compute_forward_diag_mask_f32`.
 *
 * @param maskValue The value written into masked positions.
 */
fun computeDiagMask(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor, maskValue: Float) {
    val src0 = dst.src[0] ?: error("DIAG_MASK requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "DIAG_MASK only supports F32, got ${src0.type}" }

    val nPast = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    require(nPast >= 0) { "n_past must be >= 0, got $nPast" }

    val nc = src0.ne[0].toInt()
    val nr = src0.ne[1].toInt()
    val nz = (src0.numElements() / (nc * nr).toLong()).toInt().coerceAtLeast(1)

    // Copy src0 → dst if not inplace
    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()
    for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) for (i0 in 0 until ne0)
        dst.setFloat(graphAllocator, src0.getFloat(graphAllocator, i0, i1, i2, i3), i0, i1, i2, i3)

    // Apply the upper-triangular mask
    for (k in 0 until nz) {
        for (j in 0 until nr) {
            for (i in nPast until nc) {
                if (i > nPast + j) {
                    dst.setFloat(graphAllocator, maskValue, i, j, k)
                }
            }
        }
    }
}

/** Convenience: applies causal mask with `-Infinity`. */
fun computeDiagMaskInf(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) =
    io.github.kotlinmania.llama.ore.computeDiagMask(
        graphAllocator,
        params,
        dst,
        Float.NEGATIVE_INFINITY
    )

/** Convenience: applies causal mask with `0`. */
fun computeDiagMaskZero(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) =
    io.github.kotlinmania.llama.ore.computeDiagMask(graphAllocator, params, dst, 0.0f)

// ---------------------------------------------------------------------------
// ggml_compute_forward_rope / rope_back  (RoPE – Rotary Position Embeddings)
// ---------------------------------------------------------------------------

/** YaRN correction dimension for a given rotation count and context. */
private fun ropeYarnCorrDim(nDims: Int, nCtxOrig: Int, nRot: Float, base: Float): Float {
    return nDims * ln(nCtxOrig / (nRot * 2.0f * kotlin.math.PI.toFloat())) / (2.0f * ln(base))
}

/** Returns the (start, end) correction-dimension pair for YaRN. */
private fun ropeYarnCorrDims(
    nDims: Int, nCtxOrig: Int, freqBase: Float, betaFast: Float, betaSlow: Float
): FloatArray {
    val start = floor(
        io.github.kotlinmania.llama.ore.ropeYarnCorrDim(
            nDims,
            nCtxOrig,
            betaFast,
            freqBase
        )
    ).coerceAtLeast(0.0f)
    val end = ceil(
        io.github.kotlinmania.llama.ore.ropeYarnCorrDim(
            nDims,
            nCtxOrig,
            betaSlow,
            freqBase
        )
    ).coerceAtMost((nDims - 1).toFloat())
    return floatArrayOf(start, end)
}

/** Ramp function for YaRN interpolation blending. */
private fun ropeYarnRamp(low: Float, high: Float, i0: Int): Float {
    val y = (i0 / 2.0f - low) / max(0.001f, high - low)
    return 1.0f - min(1.0f, max(0.0f, y))
}

/**
 * Core YaRN rotation helper: returns `(cosθ, sinθ)` for the given pair index.
 * When `ext_factor == 0` the scaling reduces to plain NTK-aware interpolation.
 */
private fun ropeYarn(
    thetaExtrap: Float, freqScale: Float, corrDims: FloatArray,
    i0: Int, extFactor: Float, mscale: Float
): Pair<Float, Float> {
    val thetaInterp = freqScale * thetaExtrap
    var theta = thetaInterp
    var ms = mscale
    if (extFactor != 0.0f) {
        val rampMix = io.github.kotlinmania.llama.ore.ropeYarnRamp(corrDims[0], corrDims[1], i0) * extFactor
        theta = thetaInterp * (1 - rampMix) + thetaExtrap * rampMix
        ms *= 1.0f + 0.1f * ln(1.0f / freqScale)
    }
    return Pair(cos(theta) * ms, sin(theta) * ms)
}

/**
 * Builds a per-element cos/sin cache for standard RoPE.
 * `cache[i] = cosθ`, `cache[i+1] = sinθ * sinSign`, stepping `θ *= thetaScale`.
 */
private fun ropeCacheInit(
    thetaBase: Float, freqScale: Float, freqFactors: FloatArray?,
    corrDims: FloatArray, ne0: Int, extFactor: Float, mscale: Float,
    sinSign: Float, thetaScale: Float
): FloatArray {
    val cache = FloatArray(ne0)
    var theta = thetaBase
    var i0 = 0
    while (i0 < ne0) {
        val ff = freqFactors?.get(i0 / 2) ?: 1.0f
        val (c, s) = io.github.kotlinmania.llama.ore.ropeYarn(
            theta / ff,
            freqScale,
            corrDims,
            i0,
            extFactor,
            mscale
        )
        cache[i0] = c
        cache[i0 + 1] = s * sinSign
        theta *= thetaScale
        i0 += 2
    }
    return cache
}

/**
 * Rotary Position Embedding (RoPE) for F32 tensors.
 * Handles NORMAL, NEOX, MROPE, VISION, and IMROPE modes.
 *
 * When [forward] is `true`, applies the standard rotation;
 * when `false`, applies the inverse (for rope_back).
 *
 * Ported from `ggml_compute_forward_rope_flt<float>` (scalar path).
 */
fun computeRope(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor, forward: Boolean = true) {
    val src0 = dst.src[0] ?: error("ROPE requires src0")
    val src1 = dst.src[1] ?: error("ROPE requires src1 (positions)")
    val src2 = dst.src.getOrNull(2)  // optional freq_factors

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16) {
        "ROPE only supports F32/F16, got ${src0.type}"
    }

    val nDims   = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)
    val mode    = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 2)
    val nCtxOrig = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 4)
    val freqBase    = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 5)
    val freqScale   = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 6)
    val extFactor   = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 7)
    val attnFactor  = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 8)
    val betaFast    = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 9)
    val betaSlow    = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 10)

    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()

    require(nDims <= ne0) { "n_dims ($nDims) > ne0 ($ne0)" }
    require(nDims % 2 == 0) { "n_dims must be even, got $nDims" }

    val thetaScale = freqBase.pow(-2.0f / nDims)
    val corrDims = io.github.kotlinmania.llama.ore.ropeYarnCorrDims(
        nDims,
        nCtxOrig,
        freqBase,
        betaFast,
        betaSlow
    )
    val sinSign = if (forward) 1.0f else -1.0f

    val isVision = mode == io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_VISION
    val mropeUsed = (mode and io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_MROPE) != 0

    val freqFactors: FloatArray? = if (src2 != null) {
        // Read freq factors from src2 (type F32)
        FloatArray(nDims / 2) { src2.getFloat(graphAllocator, it) }
    } else null

    val isF16 = src0.type == io.github.kotlinmania.llama.ore.GGMLType.F16

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            // Build cache for this sequence position
            val p = src1.getInt(graphAllocator, i2).toLong()
            val cache = io.github.kotlinmania.llama.ore.ropeCacheInit(
                p.toFloat(), freqScale, freqFactors, corrDims, ne0,
                extFactor, attnFactor, sinSign, thetaScale
            )

            for (i1 in 0 until ne1) {
                // Determine offset and n_offset based on mode
                val nOffset: Int
                val pairsN: Int
                val scale: Int
                when (mode) {
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_NORMAL -> {
                        nOffset = 1; pairsN = nDims; scale = 1
                    }
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_NEOX, io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_MROPE, io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_IMROPE -> {
                        nOffset = nDims / 2; pairsN = nDims; scale = 2
                    }
                    io.github.kotlinmania.llama.ore.GGML_ROPE_TYPE_VISION -> {
                        nOffset = nDims; pairsN = ne0; scale = 2
                    }
                    else -> error("Unsupported rope mode $mode")
                }

                // Rotate pairs
                var i0 = 0
                while (i0 < pairsN) {
                    val ic = i0 / scale
                    val cosTheta = cache[i0]
                    val sinTheta = cache[i0 + 1]

                    val x0: Float
                    val x1: Float
                    if (isF16) {
                        x0 = src0.getHalf(graphAllocator, ic, i1, i2, i3)
                        x1 = src0.getHalf(graphAllocator, ic + nOffset, i1, i2, i3)
                    } else {
                        x0 = src0.getFloat(graphAllocator, ic, i1, i2, i3)
                        x1 = src0.getFloat(graphAllocator, ic + nOffset, i1, i2, i3)
                    }

                    val r0 = x0 * cosTheta - x1 * sinTheta
                    val r1 = x0 * sinTheta + x1 * cosTheta

                    if (isF16) {
                        dst.setHalf(graphAllocator, r0, ic, i1, i2, i3)
                        dst.setHalf(graphAllocator, r1, ic + nOffset, i1, i2, i3)
                    } else {
                        dst.setFloat(graphAllocator, r0, ic, i1, i2, i3)
                        dst.setFloat(graphAllocator, r1, ic + nOffset, i1, i2, i3)
                    }
                    i0 += 2
                }

                // Copy remaining channels unchanged (for non-vision modes)
                if (!isVision) {
                    var i0r = nDims
                    while (i0r < ne0) {
                        if (isF16) {
                            dst.setHalf(graphAllocator, src0.getHalf(graphAllocator, i0r, i1, i2, i3), i0r, i1, i2, i3)
                            dst.setHalf(graphAllocator, src0.getHalf(graphAllocator, i0r + 1, i1, i2, i3), i0r + 1, i1, i2, i3)
                        } else {
                            dst.setFloat(graphAllocator, src0.getFloat(graphAllocator, i0r, i1, i2, i3), i0r, i1, i2, i3)
                            dst.setFloat(graphAllocator, src0.getFloat(graphAllocator, i0r + 1, i1, i2, i3), i0r + 1, i1, i2, i3)
                        }
                        i0r += 2
                    }
                }
            }
        }
    }
}

/** Backward pass for RoPE — identical computation with inverted sin sign. */
fun computeRopeBack(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) =
    io.github.kotlinmania.llama.ore.computeRope(graphAllocator, params, dst, forward = false)

// ---------------------------------------------------------------------------
// ggml_compute_forward_pad
// ---------------------------------------------------------------------------

/**
 * Pads an F32 tensor with zeros (or wraps circularly) along all four
 * dimensions.  Padding amounts are stored in `dst.opParams[0..7]` as
 * `(lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)`.  `opParams[8]` selects
 * circular mode when non-zero.
 *
 * Ported from `ggml_compute_forward_pad_f32`.
 */
fun computePad(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("PAD requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "PAD only supports F32, got ${src0.type}" }

    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()
    val ne00 = src0.ne[0].toInt(); val ne01 = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt(); val ne03 = src0.ne[3].toInt()

    val lp0 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    val rp0 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)
    val lp1 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 2)
    val rp1 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 3)
    val lp2 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 4)
    val rp2 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 5)
    val lp3 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 6)
    val rp3 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 7)
    val circular = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 8) != 0

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne1) {
                for (i0 in 0 until ne0) {
                    val value: Float = if (circular) {
                        val si0 = io.github.kotlinmania.llama.ore.wrapAround(i0 - lp0, ne00)
                        val si1 = io.github.kotlinmania.llama.ore.wrapAround(i1 - lp1, ne01)
                        val si2 = io.github.kotlinmania.llama.ore.wrapAround(i2 - lp2, ne02)
                        val si3 = io.github.kotlinmania.llama.ore.wrapAround(i3 - lp3, ne03)
                        src0.getFloat(graphAllocator, si0, si1, si2, si3)
                    } else {
                        if (i0 in lp0 until (ne0 - rp0) &&
                            i1 in lp1 until (ne1 - rp1) &&
                            i2 in lp2 until (ne2 - rp2) &&
                            i3 in lp3 until (ne3 - rp3)
                        ) {
                            src0.getFloat(graphAllocator, i0 - lp0, i1 - lp1, i2 - lp2, i3 - lp3)
                        } else {
                            0.0f
                        }
                    }
                    dst.setFloat(graphAllocator, value, i0, i1, i2, i3)
                }
            }
        }
    }
}

/** Circular wrap: `(coord + size) % size`. */
private fun wrapAround(coord: Int, size: Int): Int = ((coord % size) + size) % size

// ---------------------------------------------------------------------------
// ggml_compute_forward_pad_reflect_1d
// ---------------------------------------------------------------------------

/**
 * 1-D reflect-padding along dimension 0.
 * `opParams[0]` = left pad, `opParams[1]` = right pad.
 *
 * Ported from `ggml_compute_forward_pad_reflect_1d`.
 */
fun computePadReflect1D(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("PAD_REFLECT requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "PAD_REFLECT only supports F32, got ${src0.type}" }

    val p0 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    val p1 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)
    val ne00 = src0.ne[0].toInt()
    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()

    for (i3 in 0 until ne3) {
        for (i2 in 0 until ne2) {
            for (i1 in 0 until ne1) {
                // Copy original data into the padded region
                for (j in 0 until ne00) {
                    dst.setFloat(
                        graphAllocator,
                        src0.getFloat(graphAllocator, j, i1, i2, i3),
                        j + p0, i1, i2, i3
                    )
                }
                // Reflect left
                for (j in 1..p0) {
                    dst.setFloat(
                        graphAllocator,
                        dst.getFloat(graphAllocator, p0 + j, i1, i2, i3),
                        p0 - j, i1, i2, i3
                    )
                }
                // Reflect right
                val rightBase = ne0 - p1 - 1
                for (j in 1..p1) {
                    dst.setFloat(
                        graphAllocator,
                        dst.getFloat(graphAllocator, rightBase - j, i1, i2, i3),
                        rightBase + j, i1, i2, i3
                    )
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_roll
// ---------------------------------------------------------------------------

/**
 * Circular shift (roll) along all four dimensions.
 * Shift amounts are stored in `dst.opParams[0..3]`.
 *
 * Ported from `ggml_compute_forward_roll_f32`.
 */
fun computeRoll(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("ROLL requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "ROLL only supports F32, got ${src0.type}" }

    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()

    val s0 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    val s1 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)
    val s2 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 2)
    val s3 = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 3)

    for (i3 in 0 until ne3) {
        val i03 = io.github.kotlinmania.llama.ore.wrapIndex(i3 - s3, ne3)
        for (i2 in 0 until ne2) {
            val i02 = io.github.kotlinmania.llama.ore.wrapIndex(i2 - s2, ne2)
            for (i1 in 0 until ne1) {
                val i01 = io.github.kotlinmania.llama.ore.wrapIndex(i1 - s1, ne1)
                for (i0 in 0 until ne0) {
                    val i00 = io.github.kotlinmania.llama.ore.wrapIndex(i0 - s0, ne0)
                    dst.setFloat(
                        graphAllocator,
                        src0.getFloat(graphAllocator, i00, i01, i02, i03),
                        i0, i1, i2, i3
                    )
                }
            }
        }
    }
}

/** Wraps a signed index into `[0, ne)`. */
private fun wrapIndex(i: Int, ne: Int): Int {
    val r = i % ne
    return if (r < 0) r + ne else r
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_arange
// ---------------------------------------------------------------------------

/**
 * Fills a 1-D F32 tensor with `[start, start+step, start+2*step, …)`.
 * Parameters are in `dst.opParams` as floats at indices 0, 1, 2.
 *
 * Ported from `ggml_compute_forward_arange_f32`.
 */
fun computeArange(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {

    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "ARANGE only supports F32, got ${dst.type}" }

    val start = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 0)
    val stop  = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 1)
    val step  = io.github.kotlinmania.llama.ore.ggml_get_op_params_f32(dst, 2)

    val steps = ceil((stop - start) / step).toInt()

    for (i in 0 until steps) {
        dst.setFloat(graphAllocator, start + step * i, i)
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_timestep_embedding
// ---------------------------------------------------------------------------

/**
 * Sinusoidal timestep embedding used in diffusion models.
 * Each input timestep is mapped to a `dim`-length vector of
 * `[cos(freq*t), …, sin(freq*t), …]`.
 *
 * Ported from `ggml_compute_forward_timestep_embedding_f32`.
 */
fun computeTimestepEmbedding(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("TIMESTEP_EMBEDDING requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "TIMESTEP_EMBEDDING only supports F32" }

    val dim = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    val maxPeriod = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 1)
    val half = dim / 2
    val ne00 = src0.ne[0].toInt()

    for (i in 0 until ne00) {
        val timestep = src0.getFloat(graphAllocator, i)
        for (j in 0 until half) {
            val freq = exp(-ln(maxPeriod.toFloat()) * j.toFloat() / half)
            val arg = timestep * freq
            // Embed: first half = cos, second half = sin
            dst.setFloat(graphAllocator, cos(arg), j, i)
            dst.setFloat(graphAllocator, sin(arg), j + half, i)
        }
        if (dim % 2 != 0) {
            dst.setFloat(graphAllocator, 0.0f, 2 * half, i)
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_argsort
// ---------------------------------------------------------------------------

/** Sort order for argsort / top-k. */
internal const val GGML_SORT_ORDER_ASC = 0
internal const val GGML_SORT_ORDER_DESC = 1

/**
 * Returns the indices that would sort each row of src0 in the order
 * specified by `dst.opParams[0]` (0 = ascending, 1 = descending).
 *
 * Ported from `ggml_compute_forward_argsort_f32`.
 */
fun computeArgsort(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("ARGSORT requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "ARGSORT only supports F32, got ${src0.type}" }

    val ne0 = src0.ne[0].toInt()
    val nr = (src0.numElements() / ne0).toInt()
    val order = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)

    for (row in 0 until nr) {
        val i1 = row % src0.ne[1].toInt()
        val i2 = (row / src0.ne[1].toInt()) % src0.ne[2].toInt()
        val i3 = row / (src0.ne[1].toInt() * src0.ne[2].toInt())

        // Read row values
        val rowData = FloatArray(ne0) { src0.getFloat(graphAllocator, it, i1, i2, i3) }

        // Build index array and sort
        val indices = IntArray(ne0) { it }
        val sorted = indices.sortedWith(Comparator { a, b ->
            if (order == io.github.kotlinmania.llama.ore.GGML_SORT_ORDER_ASC)
                rowData[a].compareTo(rowData[b])
            else
                rowData[b].compareTo(rowData[a])
        })

        for (j in 0 until ne0) {
            dst.setInt(graphAllocator, sorted[j], j, i1, i2, i3)
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_top_k
// ---------------------------------------------------------------------------

/**
 * For each row, finds the top-k element indices (by value, descending).
 * `ne0` of the destination is the k value.  Order among the k results
 * is intentionally **not** guaranteed (matches C++ behaviour which swaps
 * the first two results).
 *
 * Ported from `ggml_compute_forward_top_k_f32`.
 */
fun computeTopK(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("TOP_K requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "TOP_K only supports F32, got ${src0.type}" }

    val ne00 = src0.ne[0].toInt()
    val topK = dst.ne[0].toInt()
    val nr = (src0.numElements() / ne00).toInt()

    for (row in 0 until nr) {
        val i1 = row % src0.ne[1].toInt()
        val i2 = (row / src0.ne[1].toInt()) % src0.ne[2].toInt()
        val i3 = row / (src0.ne[1].toInt() * src0.ne[2].toInt())

        val rowData = FloatArray(ne00) { src0.getFloat(graphAllocator, it, i1, i2, i3) }
        val indices = IntArray(ne00) { it }

        // Partial sort: find top-k largest
        val sorted = indices.sortedByDescending { rowData[it] }
        val topIndices = sorted.take(topK).toIntArray()

        // Deliberate swap of first two (matches C++ behaviour)
        if (topK > 1) {
            val tmp = topIndices[0]; topIndices[0] = topIndices[1]; topIndices[1] = tmp
        }

        for (j in 0 until topK) {
            dst.setInt(graphAllocator, topIndices[j], j, i1, i2, i3)
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_upscale
// ---------------------------------------------------------------------------

/** Scale mode flags mirroring ggml.h constants. */
internal const val GGML_SCALE_MODE_NEAREST  = 0
internal const val GGML_SCALE_MODE_BILINEAR = 1
internal const val GGML_SCALE_FLAG_ALIGN_CORNERS = 0x100

/**
 * Upscales an F32 tensor using nearest-neighbour or bilinear interpolation.
 * The scale factors are inferred from `dst.ne / src0.ne`.
 *
 * Ported from `ggml_compute_forward_upscale_f32` (nearest + bilinear paths).
 */
fun computeUpscale(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("UPSCALE requires src0")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "UPSCALE only supports F32, got ${src0.type}" }

    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()
    val ne00 = src0.ne[0].toInt(); val ne01 = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt(); val ne03 = src0.ne[3].toInt()

    var sf0 = ne0.toFloat() / ne00
    var sf1 = ne1.toFloat() / ne01
    val sf2 = ne2.toFloat() / ne02
    val sf3 = ne3.toFloat() / ne03

    val modeFlags = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(dst, 0)
    val mode = modeFlags and 0xFF
    var pixelOffset = 0.5f

    if (modeFlags and io.github.kotlinmania.llama.ore.GGML_SCALE_FLAG_ALIGN_CORNERS != 0) {
        pixelOffset = 0.0f
        if (ne0 > 1 && ne00 > 1) sf0 = (ne0 - 1).toFloat() / (ne00 - 1)
        if (ne1 > 1 && ne01 > 1) sf1 = (ne1 - 1).toFloat() / (ne01 - 1)
    }

    when (mode) {
        io.github.kotlinmania.llama.ore.GGML_SCALE_MODE_NEAREST -> {
            for (i3 in 0 until ne3) {
                val i03 = (i3 / sf3).toInt()
                for (i2 in 0 until ne2) {
                    val i02 = (i2 / sf2).toInt()
                    for (i1 in 0 until ne1) {
                        val i01 = (i1 / sf1).toInt()
                        for (i0 in 0 until ne0) {
                            val i00 = (i0 / sf0).toInt()
                            dst.setFloat(
                                graphAllocator,
                                src0.getFloat(graphAllocator, i00, i01, i02, i03),
                                i0, i1, i2, i3
                            )
                        }
                    }
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGML_SCALE_MODE_BILINEAR -> {
            for (i3 in 0 until ne3) {
                val i03 = (i3 / sf3).toInt()
                for (i2 in 0 until ne2) {
                    val i02 = (i2 / sf2).toInt()
                    for (i1 in 0 until ne1) {
                        val y = (i1.toFloat() + pixelOffset) / sf1 - pixelOffset
                        var y0 = floor(y).toInt(); var y1 = y0 + 1
                        y0 = y0.coerceIn(0, ne01 - 1)
                        y1 = y1.coerceIn(0, ne01 - 1)
                        val dy = (y - floor(y)).coerceIn(0.0f, 1.0f)
                        for (i0 in 0 until ne0) {
                            val x = (i0.toFloat() + pixelOffset) / sf0 - pixelOffset
                            var x0 = floor(x).toInt(); var x1 = x0 + 1
                            x0 = x0.coerceIn(0, ne00 - 1)
                            x1 = x1.coerceIn(0, ne00 - 1)
                            val dx = (x - floor(x)).coerceIn(0.0f, 1.0f)

                            val a = src0.getFloat(graphAllocator, x0, y0, i02, i03)
                            val b = src0.getFloat(graphAllocator, x1, y0, i02, i03)
                            val c = src0.getFloat(graphAllocator, x0, y1, i02, i03)
                            val d = src0.getFloat(graphAllocator, x1, y1, i02, i03)
                            val v = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy
                            dst.setFloat(graphAllocator, v, i0, i1, i2, i3)
                        }
                    }
                }
            }
        }
        else -> error("fatal error")
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_out_prod
// ---------------------------------------------------------------------------

/**
 * Outer product: `dst[i0, i1, i2, i3] += Σ_k src0[i0, k, i2', i3'] * src1[i1, k, i2, i3]`
 * where `i2' = i2 / (ne2/ne02)` etc.  dst is zeroed first.
 *
 * Only the F32 path is fully implemented; quantized types marked TODO.
 *
 * Ported from `ggml_compute_forward_out_prod_f32`.
 */
fun computeOutProd(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("OUT_PROD requires src0")
    val src1 = dst.src[1] ?: error("OUT_PROD requires src1")

    require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "OUT_PROD dst must be F32" }

    val ne0 = dst.ne[0].toInt(); val ne1 = dst.ne[1].toInt()
    val ne2 = dst.ne[2].toInt(); val ne3 = dst.ne[3].toInt()
    val ne01 = src0.ne[1].toInt()
    val ne02 = src0.ne[2].toInt(); val ne03 = src0.ne[3].toInt()

    val dps2 = if (ne02 > 0) ne2 / ne02 else 1
    val dps3 = if (ne03 > 0) ne3 / ne03 else 1

    // Zero dst
    for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) for (i0 in 0 until ne0)
        dst.setFloat(graphAllocator, 0.0f, i0, i1, i2, i3)

    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            for (i3 in 0 until ne3) {
                for (i2 in 0 until ne2) {
                    val i02 = i2 / dps2
                    val i03 = i3 / dps3
                    for (i1 in 0 until ne1) {
                        for (i01 in 0 until ne01) {
                            val s1v = src1.getFloat(graphAllocator, i1, i01, i2, i3)
                            for (i0 in 0 until ne0) {
                                val s0v = src0.getFloat(graphAllocator, i0, i01, i02, i03)
                                val cur = dst.getFloat(graphAllocator, i0, i1, i2, i3)
                                dst.setFloat(graphAllocator, cur + s0v * s1v, i0, i1, i2, i3)
                            }
                        }
                    }
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1, io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
        io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
        io.github.kotlinmania.llama.ore.GGMLType.Q1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ1_0, io.github.kotlinmania.llama.ore.GGMLType.TQ2_0,
        io.github.kotlinmania.llama.ore.GGMLType.MXFP4, io.github.kotlinmania.llama.ore.GGMLType.NVFP4,
        io.github.kotlinmania.llama.ore.GGMLType.IQ2_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_XS, io.github.kotlinmania.llama.ore.GGMLType.IQ2_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ3_XXS, io.github.kotlinmania.llama.ore.GGMLType.IQ3_S,
        io.github.kotlinmania.llama.ore.GGMLType.IQ1_S, io.github.kotlinmania.llama.ore.GGMLType.IQ1_M,
        io.github.kotlinmania.llama.ore.GGMLType.IQ4_NL, io.github.kotlinmania.llama.ore.GGMLType.IQ4_XS -> {
            // Dequantize src0 to F32 and compute outer product
            val src0F32 = io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, src0)
            for (i3 in 0 until ne3) {
                for (i2 in 0 until ne2) {
                    val i02 = i2 / dps2
                    val i03 = i3 / dps3
                    for (i1 in 0 until ne1) {
                        for (i01 in 0 until ne01) {
                            val s1v = src1.getFloat(graphAllocator, i1, i01, i2, i3)
                            for (i0 in 0 until ne0) {
                                val s0v = src0F32.getFloat(graphAllocator, i0, i01, i02, i03)
                                val cur = dst.getFloat(graphAllocator, i0, i1, i2, i3)
                                dst.setFloat(graphAllocator, cur + s0v * s1v, i0, i1, i2, i3)
                            }
                        }
                    }
                }
            }
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> error("fatal error") // C++: ggml_compute_forward_out_prod_f16_f32
        else -> error("fatal error")
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_cross_entropy_loss
// ---------------------------------------------------------------------------

/**
 * Cross-entropy loss: `-mean_over_rows( Σ_j  src1[j] * log_softmax(src0[j]) )`.
 * Result is a scalar F32 in dst.
 *
 * Ported from `ggml_compute_forward_cross_entropy_loss_f32` (single-thread).
 */
fun computeCrossEntropyLoss(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val src0 = dst.src[0] ?: error("CROSS_ENTROPY_LOSS requires src0")
    val src1 = dst.src[1] ?: error("CROSS_ENTROPY_LOSS requires src1")

    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "Only F32 supported" }
    require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "Only F32 supported" }

    val nc = src0.ne[0].toInt()
    val nr = (src0.numElements() / nc).toInt()

    var totalLoss = 0.0f

    for (row in 0 until nr) {
        val i1 = row % src0.ne[1].toInt()
        val i2 = (row / src0.ne[1].toInt()) % src0.ne[2].toInt()
        val i3 = row / (src0.ne[1].toInt() * src0.ne[2].toInt())

        // Find max for numerical stability
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until nc) {
            val v = src0.getFloat(graphAllocator, j, i1, i2, i3)
            if (v > maxVal) maxVal = v
        }

        // Compute log-sum-exp
        var sumExp = 0.0
        for (j in 0 until nc) {
            sumExp += exp((src0.getFloat(graphAllocator, j, i1, i2, i3) - maxVal).toDouble())
        }
        val logSumExp = ln(sumExp).toFloat()

        // Accumulate: src1 * (src0 - max - logSumExp)
        var rowLoss = 0.0f
        for (j in 0 until nc) {
            val logSoftmax = src0.getFloat(graphAllocator, j, i1, i2, i3) - maxVal - logSumExp
            rowLoss += src1.getFloat(graphAllocator, j, i1, i2, i3) * logSoftmax
        }
        totalLoss += rowLoss
    }

    dst.setFloat(graphAllocator, -totalLoss / nr, 0)
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_cross_entropy_loss_back
// ---------------------------------------------------------------------------

/**
 * Backward pass for cross-entropy loss.
 * `dst = (softmax(src0f) - src1f) * grad[0] / nr`
 *
 * Sources: `dst.src[0]` = grad (scalar), `dst.src[1]` = src0f, `dst.src[2]` = src1f.
 *
 * Ported from `ggml_compute_forward_cross_entropy_loss_back_f32`.
 */
fun computeCrossEntropyLossBack(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val grad  = dst.src[0] ?: error("CE_LOSS_BACK requires src0 (grad)")
    val src0f = dst.src[1] ?: error("CE_LOSS_BACK requires src1 (logits)")
    val src1f = dst.src[2] ?: error("CE_LOSS_BACK requires src2 (labels)")

    val nc = src0f.ne[0].toInt()
    val nr = (src0f.numElements() / nc).toInt()
    val dByNr = grad.getFloat(graphAllocator, 0) / nr

    for (row in 0 until nr) {
        val i1 = row % src0f.ne[1].toInt()
        val i2 = (row / src0f.ne[1].toInt()) % src0f.ne[2].toInt()
        val i3 = row / (src0f.ne[1].toInt() * src0f.ne[2].toInt())

        // softmax of the row
        var maxVal = Float.NEGATIVE_INFINITY
        for (j in 0 until nc) {
            val v = src0f.getFloat(graphAllocator, j, i1, i2, i3)
            if (v > maxVal) maxVal = v
        }
        var sumExp = 0.0
        for (j in 0 until nc) {
            sumExp += exp((src0f.getFloat(graphAllocator, j, i1, i2, i3) - maxVal).toDouble())
        }

        // ds0 = (softmax - label) * d_by_nr
        for (j in 0 until nc) {
            val sm = exp((src0f.getFloat(graphAllocator, j, i1, i2, i3) - maxVal).toDouble()) / sumExp
            val label = src1f.getFloat(graphAllocator, j, i1, i2, i3)
            dst.setFloat(graphAllocator, (sm.toFloat() - label) * dByNr, j, i1, i2, i3)
        }
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_opt_step_adamw
// ---------------------------------------------------------------------------

/**
 * AdamW optimizer step.  Updates weights **in-place** in src0 and running
 * moment estimates in src2 (m) and src3 (v).
 *
 * Sources:
 * - `dst.src[0]` = weights (updated in-place)
 * - `dst.src[1]` = gradients
 * - `dst.src[2]` = first moment (m, updated in-place)
 * - `dst.src[3]` = second moment (v, updated in-place)
 * - `dst.src[4]` = parameter tensor `[alpha, beta1, beta2, eps, wd, beta1h, beta2h]`
 *
 * Ported from `ggml_compute_forward_opt_step_adamw_f32`.
 */
fun computeOptStepAdamw(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val w     = dst.src[0] ?: error("ADAMW requires src0 (weights)")
    val g     = dst.src[1] ?: error("ADAMW requires src1 (grad)")
    val m     = dst.src[2] ?: error("ADAMW requires src2 (m)")
    val v     = dst.src[3] ?: error("ADAMW requires src3 (v)")
    val ap    = dst.src[4] ?: error("ADAMW requires src4 (params)")

    require(w.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "ADAMW only supports F32 weights" }

    val alpha  = ap.getFloat(graphAllocator, 0)
    val beta1  = ap.getFloat(graphAllocator, 1)
    val beta2  = ap.getFloat(graphAllocator, 2)
    val eps    = ap.getFloat(graphAllocator, 3)
    val wd     = ap.getFloat(graphAllocator, 4)
    val beta1h = ap.getFloat(graphAllocator, 5)
    val beta2h = ap.getFloat(graphAllocator, 6)
    val keep   = 1.0f - alpha * wd

    val ne0 = w.ne[0].toInt(); val ne1 = w.ne[1].toInt()
    val ne2 = w.ne[2].toInt(); val ne3 = w.ne[3].toInt()

    for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) for (i0 in 0 until ne0) {
        val gv = g.getFloat(graphAllocator, i0, i1, i2, i3)
        val mv = m.getFloat(graphAllocator, i0, i1, i2, i3) * beta1 + gv * (1.0f - beta1)
        val vv = v.getFloat(graphAllocator, i0, i1, i2, i3) * beta2 + gv * gv * (1.0f - beta2)
        m.setFloat(graphAllocator, mv, i0, i1, i2, i3)
        v.setFloat(graphAllocator, vv, i0, i1, i2, i3)

        val mh = mv * beta1h
        val vh = sqrt(vv * beta2h) + eps
        val wv = w.getFloat(graphAllocator, i0, i1, i2, i3)
        w.setFloat(graphAllocator, wv * keep - alpha * mh / vh, i0, i1, i2, i3)
    }
}

// ---------------------------------------------------------------------------
// ggml_compute_forward_opt_step_sgd
// ---------------------------------------------------------------------------

/**
 * SGD with weight decay optimizer step.
 * `w = w * (1 - alpha*wd) - alpha * grad`
 *
 * Sources:
 * - `dst.src[0]` = weights (updated in-place)
 * - `dst.src[1]` = gradients
 * - `dst.src[2]` = parameter tensor `[alpha, wd]`
 *
 * Ported from `ggml_compute_forward_opt_step_sgd_f32`.
 */
fun computeOptStepSgd(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, params: io.github.kotlinmania.llama.ore.GGMLComputeParams, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val w     = dst.src[0] ?: error("SGD requires src0 (weights)")
    val g     = dst.src[1] ?: error("SGD requires src1 (grad)")
    val sp    = dst.src[2] ?: error("SGD requires src2 (params)")

    require(w.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "SGD only supports F32 weights" }

    val alpha = sp.getFloat(graphAllocator, 0)
    val keep  = 1.0f - alpha * sp.getFloat(graphAllocator, 1)

    val ne0 = w.ne[0].toInt(); val ne1 = w.ne[1].toInt()
    val ne2 = w.ne[2].toInt(); val ne3 = w.ne[3].toInt()

    for (i3 in 0 until ne3) for (i2 in 0 until ne2) for (i1 in 0 until ne1) for (i0 in 0 until ne0) {
        val wv = w.getFloat(graphAllocator, i0, i1, i2, i3)
        val gv = g.getFloat(graphAllocator, i0, i1, i2, i3)
        w.setFloat(graphAllocator, wv * keep - alpha * gv, i0, i1, i2, i3)
    }
}

/**
 * Main compute operations object for executing graphs
 */
object GGMLComputeOps {
    /**
     * Compute a computational graph
     * 
     * @param graph The computational graph to execute
     */
    fun computeGraph(graph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
        val graphAllocator = graph.allocator ?: throw IllegalStateException("Graph must have an allocator")
        val nodes = buildList {
            for (i in 0 until graph.nNodes) {
                graph.nodes[i]?.let { add(it) }
            }
        }
        computeGraphNodes(graphAllocator, nodes)
    }

    internal fun computeGraphNodes(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, nodes: List<io.github.kotlinmania.llama.ore.GGMLTensor>) {
        nodes.forEach { computeNode(graphAllocator, it) }
    }

    /**
     * Compute a single node in the graph
     */
    internal fun computeNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        when (node.op) {
            io.github.kotlinmania.llama.ore.GGMLOp.NONE -> { /* No operation */ }
            io.github.kotlinmania.llama.ore.GGMLOp.DUP -> computeDup(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ADD -> computeAdd(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SUB -> computeSub(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.MUL -> computeMul(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.DIV -> computeDiv(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SQR -> computeSqr(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SQRT -> computeSqrt(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.NEG -> computeNeg(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.RELU -> computeRelu(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.GELU -> computeGelu(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SILU -> computeSilu(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX -> computeSoftMax(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM -> computeRmsNorm(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT -> computeMulMat(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE -> computeTranspose(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SUM -> computeSum(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.MEAN -> computeMean(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.REPEAT -> computeRepeat(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.REPEAT_BACK -> computeRepeatBack(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ABS -> computeAbsNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SGN -> computeSgnNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.STEP -> computeStepNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CEIL -> computeCeilNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.FLOOR -> computeFloorNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ROUND -> computeRoundNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.TRUNC -> computeTruncNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.EXP -> computeExpNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SIGMOID -> computeSigmoidNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.TANH -> computeTanhNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.HARDSWISH -> computeHardswishNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.HARDSIGMOID -> computeHardsigmoidNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.GELU_QUICK -> computeGeluQuickNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.GELU_ERF -> computeGeluErfNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SILU_BACK -> computeSiluBackNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SOFTPLUS -> computeSoftplusNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ELU -> computeEluNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.LEAKY_RELU -> computeLeakyReluNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.NORM -> computeNormNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM_BACK -> computeRmsNormBackNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.GROUP_NORM -> computeGroupNormNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ACC -> computeAccNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SCALE -> computeScaleNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CLAMP -> computeClampNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CONT -> computeContNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CPY -> computeCpyNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SET -> computeSetNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CONCAT -> computeConcatNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SUM_ROWS -> computeSumRowsNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CUMSUM -> computeCumsumNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ARGMAX -> computeArgmaxNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.COUNT_EQUAL -> computeCountEqualNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS -> computeGetRowsNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS_BACK -> computeGetRowsBackNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.SET_ROWS -> computeSetRowsNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.DIAG -> computeDiagNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_INF -> computeDiagMaskInfNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_ZERO -> computeDiagMaskZeroNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ROPE -> computeRopeNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ROPE_BACK -> computeRopeBackNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.PAD -> computePadNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.PAD_REFLECT_1D -> computePadReflect1DNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ROLL -> computeRollNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ARANGE -> computeArangeNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.TIMESTEP_EMBEDDING -> computeTimestepEmbeddingNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.ARGSORT -> computeArgsortNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.TOP_K -> computeTopKNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.UPSCALE -> computeUpscaleNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.OUT_PROD -> computeOutProdNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CROSS_ENTROPY_LOSS -> computeCrossEntropyLossNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.CROSS_ENTROPY_LOSS_BACK -> computeCrossEntropyLossBackNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.OPT_STEP_ADAMW -> computeOptStepAdamwNode(graphAllocator, node)
            io.github.kotlinmania.llama.ore.GGMLOp.OPT_STEP_SGD -> computeOptStepSgdNode(graphAllocator, node)
            else -> error("fatal error")
        }
    }
    
    // Helper methods for individual operations
    
    private fun computeDup(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("DUP operation requires source tensor")
        // Simple dup: copy element-wise based on type
        node.type = src.type
        node.ne = src.ne.copyOf(); node.nb =
            io.github.kotlinmania.llama.ore.calculateContiguousStrides(
                node.ne,
                node.type,
                node.ne.size
            )
        val total = src.numElements().toInt()
        when (src.type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src,
                total
            ) { _, ind -> node.setFloat(graphAllocator, src.getFloat(graphAllocator, *ind), *ind) }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src,
                total
            ) { _, ind -> node.setHalf(graphAllocator, src.getHalf(graphAllocator, *ind), *ind) }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src,
                total
            ) { _, ind -> node.setInt(graphAllocator, src.getInt(graphAllocator, *ind), *ind) }
            io.github.kotlinmania.llama.ore.GGMLType.I16 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src,
                total
            ) { _, ind -> node.setShort(graphAllocator, src.getShort(graphAllocator, *ind), *ind) }
            io.github.kotlinmania.llama.ore.GGMLType.I8  -> io.github.kotlinmania.llama.ore.applyNDIter(
                src,
                total
            ) { _, ind -> node.setByte(graphAllocator, src.getByte(graphAllocator, *ind), *ind) }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src,
                total
            ) { _, ind -> node.setLong(graphAllocator, src.getLong(graphAllocator, *ind), *ind) }
            else -> {
                // For quantized, perform raw data copy if allocated in same buffer
                node.data = src.data; node.bufferId = src.bufferId; node.dataOffset = src.dataOffset; node.nb = src.nb.copyOf()
            }
        }
    }
    
    private fun computeAdd(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("ADD operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("ADD operation requires second source tensor")
        io.github.kotlinmania.llama.ore.computeAdd(graphAllocator, src0, src1, node)
    }
    
    private fun computeSub(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("SUB operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("SUB operation requires second source tensor")
        io.github.kotlinmania.llama.ore.computeSub(graphAllocator, src0, src1, node)
    }
    
    private fun computeMul(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("MUL operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("MUL operation requires second source tensor")
        io.github.kotlinmania.llama.ore.computeMul(graphAllocator, src0, src1, node)
    }
    
    private fun computeDiv(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("DIV operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("DIV operation requires second source tensor")
        io.github.kotlinmania.llama.ore.computeDiv(graphAllocator, src0, src1, node)
    }
    
    internal fun computeSqr(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SQR operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeSqr(graphAllocator, src, node)
    }
    
    internal fun computeSqrt(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SQRT operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeSqrt(graphAllocator, src, node)
    }
    
    private fun computeNeg(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("NEG operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeNeg(graphAllocator, src, node)
    }
    
    private fun computeRelu(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("RELU operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeRelu(graphAllocator, src, node)
    }
    
    private fun computeGelu(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("GELU operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeGelu(graphAllocator, src, node)
    }

    private fun computeSilu(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SILU operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeSilu(graphAllocator, src, node)
    }

    private fun computeSoftMax(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SOFT_MAX operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeSoftMax(graphAllocator, src, node)
    }

    private fun computeRmsNorm(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("RMS_NORM operation requires source tensor")
    val eps = Float.fromBits(node.opParams[0])
        io.github.kotlinmania.llama.ore.computeRMSNorm(graphAllocator, src, eps, node)
    }
    
    private fun computeMulMat(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("MUL_MAT operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("MUL_MAT operation requires second source tensor")
        io.github.kotlinmania.llama.ore.computeMatMul(graphAllocator, src0, src1, node)
    }
    
    private fun computeSum(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SUM operation requires source tensor")
        // Simple sum over all elements into a scalar at [0,0]
        node.ne[0] = 1; node.ne[1] = 1; node.type = src.type
        var accF = 0.0f; var accI = 0L
        val total = src.numElements().toInt()
        when (src.type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                io.github.kotlinmania.llama.ore.applyNDIter(
                    src,
                    total
                ) { _, ind -> accF += src.getFloat(graphAllocator, *ind) }; node.setFloat(graphAllocator, accF, 0, 0) }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                io.github.kotlinmania.llama.ore.applyNDIter(
                    src,
                    total
                ) { _, ind -> accF += src.getHalf(graphAllocator, *ind) }; node.setHalf(graphAllocator, accF, 0, 0) }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                io.github.kotlinmania.llama.ore.applyNDIter(
                    src,
                    total
                ) { _, ind -> accI += src.getInt(graphAllocator, *ind).toLong() }; node.setInt(graphAllocator, accI.toInt(), 0, 0) }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> {
                io.github.kotlinmania.llama.ore.applyNDIter(
                    src,
                    total
                ) { _, ind -> accI += src.getLong(graphAllocator, *ind) }; node.setLong(graphAllocator, accI, 0, 0) }
            else -> error("fatal error")
        }
    }
    
    private fun computeMean(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("MEAN operation requires source tensor")
        // Mean using sum / N
        val n = src.numElements().toInt().coerceAtLeast(1)
        node.ne[0] = 1; node.ne[1] = 1; node.type = src.type
        var accF = 0.0f; var accI = 0L
        val total = src.numElements().toInt()
        when (src.type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                io.github.kotlinmania.llama.ore.applyNDIter(
                    src,
                    total
                ) { _, ind -> accF += src.getFloat(graphAllocator, *ind) }; node.setFloat(graphAllocator, accF / n, 0, 0) }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                io.github.kotlinmania.llama.ore.applyNDIter(
                    src,
                    total
                ) { _, ind -> accF += src.getHalf(graphAllocator, *ind) }; node.setHalf(graphAllocator, accF / n, 0, 0) }
            else -> error("fatal error")
        }
    }

    private fun computeRepeat(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("REPEAT operation requires source tensor")
        // Basic repeat assumes node.ne is set to desired output shape and src.ne divides node.ne
        for (d in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) require(src.ne[d] == 0L || node.ne[d] % src.ne[d] == 0L) { "REPEAT shape mismatch" }
        val total = node.numElements().toInt()
        when (src.type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
                node,
                total
            ) { _, ind ->
                val srcIdx = IntArray(ind.size) { i -> if (src.ne[i] > 0) (ind[i] % src.ne[i].toInt()) else 0 }
                node.setFloat(graphAllocator, src.getFloat(graphAllocator, *srcIdx), *ind)
            }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
                node,
                total
            ) { _, ind ->
                val srcIdx = IntArray(ind.size) { i -> if (src.ne[i] > 0) (ind[i] % src.ne[i].toInt()) else 0 }
                node.setHalf(graphAllocator, src.getHalf(graphAllocator, *srcIdx), *ind)
            }
            else -> error("fatal error")
        }
    }

    private fun computeRepeatBack(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("REPEAT_BACK operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeRepeatBack(graphAllocator, src, node)
    }

    private fun computeTranspose(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("TRANSPOSE operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeTranspose(graphAllocator, src, node)
    }

    // ---------------------------------------------------------------
    // Batch 2: new compute-node dispatchers (ops.cpp port)
    // ---------------------------------------------------------------

    // --- Element-wise unary ops ---

    private fun computeAbsNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("ABS operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "ABS"
        ) { x -> abs(x) }
    }

    private fun computeSgnNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SGN operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "SGN"
        ) { x -> sign(x) }
    }

    private fun computeStepNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("STEP operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "STEP"
        ) { x -> if (x > 0f) 1f else 0f }
    }

    private fun computeCeilNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("CEIL operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "CEIL"
        ) { x -> ceil(x) }
    }

    private fun computeFloorNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("FLOOR operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "FLOOR"
        ) { x -> floor(x) }
    }

    private fun computeRoundNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("ROUND operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "ROUND"
        ) { x -> round(x) }
    }

    private fun computeTruncNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("TRUNC operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "TRUNC"
        ) { x ->
            if (x >= 0f) floor(x) else ceil(x)
        }
    }

    private fun computeExpNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("EXP operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "EXP"
        ) { x -> exp(x) }
    }

    private fun computeSigmoidNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SIGMOID operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "SIGMOID"
        ) { x -> 1f / (1f + exp(-x)) }
    }

    private fun computeTanhNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("TANH operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "TANH"
        ) { x -> tanh(x) }
    }

    private fun computeHardswishNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("HARDSWISH operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "HARDSWISH"
        ) { x ->
            x * min(1f, max(0f, (x + 3f) / 6f))
        }
    }

    private fun computeHardsigmoidNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("HARDSIGMOID operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "HARDSIGMOID"
        ) { x ->
            min(1f, max(0f, (x + 3f) / 6f))
        }
    }

    private fun computeGeluQuickNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("GELU_QUICK operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "GELU_QUICK"
        ) { x ->
            x * (1f / (1f + exp(io.github.kotlinmania.llama.ore.GELU_QUICK_COEF * x)))
        }
    }

    private fun computeGeluErfNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("GELU_ERF operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "GELU_ERF"
        ) { x ->
            0.5f * x * (1f + io.github.kotlinmania.llama.ore.erfApprox(x * io.github.kotlinmania.llama.ore.SQRT_2_INV))
        }
    }

    private fun computeSoftplusNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SOFTPLUS operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "SOFTPLUS"
        ) { x ->
            if (x > 20f) x else ln(1f + exp(x))
        }
    }

    private fun computeEluNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("ELU operation requires source tensor")
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "ELU"
        ) { x ->
            if (x > 0f) x else (exp(x) - 1f)
        }
    }

    // --- Ops with parameters ---

    private fun computeLeakyReluNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("LEAKY_RELU operation requires source tensor")
        val negativeSlope = Float.fromBits(node.opParams[0])
        io.github.kotlinmania.llama.ore.computeUnaryElementwise(
            graphAllocator,
            src,
            node,
            "LEAKY_RELU"
        ) { x ->
            if (x > 0f) x else negativeSlope * x
        }
    }

    private fun computeSiluBackNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val grad = node.src[0] ?: throw IllegalArgumentException("SILU_BACK requires gradient tensor (src[0])")
        val src1 = node.src[1] ?: throw IllegalArgumentException("SILU_BACK requires input tensor (src[1])")
        io.github.kotlinmania.llama.ore.computeSiluBack(graphAllocator, grad, src1, node)
    }

    // --- Normalization ops ---

    private fun computeNormNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("NORM (layer norm) requires source tensor")
        val eps = Float.fromBits(node.opParams[0])
        io.github.kotlinmania.llama.ore.computeLayerNorm(graphAllocator, src, eps, node)
    }

    private fun computeRmsNormBackNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val grad = node.src[0] ?: throw IllegalArgumentException("RMS_NORM_BACK requires gradient tensor (src[0])")
        val src1 = node.src[1] ?: throw IllegalArgumentException("RMS_NORM_BACK requires input tensor (src[1])")
        val eps = Float.fromBits(node.opParams[0])
        io.github.kotlinmania.llama.ore.computeRmsNormBack(graphAllocator, grad, src1, eps, node)
    }

    private fun computeGroupNormNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("GROUP_NORM requires source tensor")
        val nGroups = node.opParams[0]
        val eps = Float.fromBits(node.opParams[1])
        io.github.kotlinmania.llama.ore.computeGroupNorm(graphAllocator, src, nGroups, eps, node)
    }

    // --- Tensor manipulation ops ---

    private fun computeAccNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("ACC requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("ACC requires second source tensor")
        io.github.kotlinmania.llama.ore.computeAcc(graphAllocator, src0, src1, node)
    }

    private fun computeScaleNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SCALE requires source tensor")
        val s = Float.fromBits(node.opParams[0])
        val b = Float.fromBits(node.opParams[1])
        io.github.kotlinmania.llama.ore.computeScale(graphAllocator, src, s, b, node)
    }

    private fun computeClampNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("CLAMP requires source tensor")
        val minVal = Float.fromBits(node.opParams[0])
        val maxVal = Float.fromBits(node.opParams[1])
        io.github.kotlinmania.llama.ore.computeClamp(graphAllocator, src, minVal, maxVal, node)
    }

    private fun computeContNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        // CONT is identical to DUP — makes a tensor contiguous
        computeDup(graphAllocator, node)
    }

    private fun computeCpyNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        // CPY is identical to DUP — copies src[0] into dst
        computeDup(graphAllocator, node)
    }

    private fun computeSetNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("SET requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("SET requires second source tensor")
        io.github.kotlinmania.llama.ore.computeSet(graphAllocator, src0, src1, node)
    }
    
    // Removed copyTensorData: compute functions now write directly into node

    // =========================================================================
    // Batch 3: Attention, SSM, Convolution, and Misc Advanced Ops
    // Port source: ggml/src/ggml-cpu/ops.cpp (lines ~2219–11214)
    // =========================================================================

    /**
     * Read a Float from [tensor] at a flat F32-element index.
     * Assumes the tensor is F32-contiguous (nb[0] == 4).
     */
    private fun readF32Flat(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        flatIndex: Int
    ): Float {
        val buf = graphAllocator.buffers[tensor.bufferId] as? ByteArray
            ?: error("Tensor buffer not found for bufferId ${tensor.bufferId}")
        val byteOff = tensor.dataOffset.toInt() + flatIndex * Float.SIZE_BYTES
        return buf.getFloatLe(byteOff)
    }

    /**
     * Write a Float into [tensor] at a flat F32-element index.
     * Assumes the tensor is F32-contiguous (nb[0] == 4).
     */
    private fun writeF32Flat(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        tensor: io.github.kotlinmania.llama.ore.GGMLTensor,
        flatIndex: Int,
        value: Float
    ) {
        val buf = graphAllocator.buffers[tensor.bufferId] as? ByteArray
            ?: error("Tensor buffer not found for bufferId ${tensor.bufferId}")
        val byteOff = tensor.dataOffset.toInt() + flatIndex * Float.SIZE_BYTES
        buf.setFloatLe(byteOff, value)
    }

    // -----------------------------------------------------------------
    // Flash Attention (forward)
    // -----------------------------------------------------------------

    /**
     * Compute flash attention with extended KV support.
     *
     * Implements the fused *online-softmax* attention algorithm from
     * [FlashAttention-2](https://arxiv.org/pdf/2205.14135):
     *
     * ```
     * O = softmax(Q·Kᵀ / √d + mask) · V
     * ```
     *
     * The C++ reference supports tiled GEMM paths, split-KV parallelism,
     * ALiBi bias, and logit softcap.  This minimal delegates to `TODO` until
     * the CPU backend formalises threading and SIMD helpers.
     *
     * Sources: `src[0]=Q`, `src[1]=K`, `src[2]=V`, `src[3]=mask` (optional),
     * `src[4]=sinks` (optional).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeFlashAttnExt(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Compute backward pass for flash attention.
     *
     * Produces gradients for Q, K, and V packed into [dst] following the
     * layout: `[grad_q | grad_k | grad_v]`.  The forward pass softmax
     * weights are recomputed on the fly.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param masked Whether causal masking is applied.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeFlashAttnBack(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        masked: Boolean,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // State Space Model (SSM) Ops
    // -----------------------------------------------------------------

    /**
     * SSM 1-D convolution (Mamba conv_x ⊛ conv1d.weight).
     *
     * For each sequence and token position, this computes a sliding-window
     * dot product between the state tensor and the convolution kernel:
     *
     * ```
     * for each (seq, token, row):
     *     dst[row] = Σ_{i=0}^{d_conv-1} src0[i + row*ncs] * src1[i + row*nc]
     * ```
     *
     * Sources: `src[0]` = conv_x `{d_conv-1+n_t, d_inner, n_seqs}`,
     *          `src[1]` = conv1d.weight `{d_conv, d_inner}`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor `{d_inner, n_t, n_seqs}` (pre-allocated).
     */
    fun computeSsmConv(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * SSM selective scan (Mamba / Mamba-2 recurrence).
     *
     * Implements the discretised state-space recurrence:
     *
     * ```
     * state = prev_state * exp(dt * A) + B * (x * dt)
     * y     = dot(state, C)
     * ```
     *
     * Supports both Mamba-1 (per-element A) and Mamba-2 (scalar A per head).
     *
     * Sources: `src[0]=s`, `src[1]=x`, `src[2]=dt`, `src[3]=A`,
     *          `src[4]=B`, `src[5]=C`, `src[6]=ids`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeSsmScan(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // RWKV WKV (v6 / v7)
    // -----------------------------------------------------------------

    /**
     * Compute RWKV-v6 WKV (Weighted Key-Value) attention.
     *
     * Fused recurrence: for each token `t`, head `h`, and state row `i`:
     *
     * ```
     * kv  = v * k
     * tmp = kv * time_first + state_prev
     * dst += tmp * r
     * state = state_prev * time_decay + kv
     * ```
     *
     * Sources: `src[0]=k`, `src[1]=v`, `src[2]=r`,
     *          `src[3]=time_faaaa`, `src[4]=time_decay`, `src[5]=state`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeRwkvWkv6(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Compute RWKV-v7 WKV attention.
     *
     * Similar to v6 but adds an extra `a` and `b` source for
     * state-attention blending:
     *
     * ```
     * sa = dot(a, state_prev)
     * state = state_prev * w + v * k + sa * b
     * dst = dot(state, r)
     * ```
     *
     * Sources: `src[0]=r`, `src[1]=w`, `src[2]=k`, `src[3]=v`,
     *          `src[4]=a`, `src[5]=b`, `src[6]=state`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeRwkvWkv7(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Gated Linear Attention (GLA)
    // -----------------------------------------------------------------

    /**
     * Compute Gated Linear Attention.
     *
     * Per-token recurrence with gated state update:
     *
     * ```
     * kv   = v * k
     * temp = prev_state * g + kv
     * dst += temp * (q * scale)
     * state = temp
     * ```
     *
     * Sources: `src[0]=k`, `src[1]=v`, `src[2]=q`, `src[3]=g`,
     *          `src[4]=state`.  `opParams[0]` = scale (float).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeGla(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Gated Delta Net
    // -----------------------------------------------------------------

    /**
     * Compute Gated Delta Net attention.
     *
     * Combines gated decay with a delta-rule state update:
     *
     * ```
     * S *= exp(g)
     * delta = (v - S·k) * beta
     * S += outer(k, delta)
     * out = S·q * scale
     * ```
     *
     * Sources: `src[0]=q`, `src[1]=k`, `src[2]=v`, `src[3]=g`,
     *          `src[4]=beta`, `src[5]=state`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeGatedDeltaNet(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Transposed 1-D Convolution
    // -----------------------------------------------------------------

    /**
     * Compute 1-D transposed convolution (deconvolution).
     *
     * Kernel `src[0]` shape `(K × Cout × Cin)` is permuted to
     * `(Cin × K × Cout)` in scratch space, then the dot-product
     * accumulation writes into the strided output.
     *
     * Sources: `src[0]` = kernel, `src[1]` = input.
     * `opParams[0]` = stride.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeConvTranspose1d(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // im2col
    // -----------------------------------------------------------------

    /**
     * Compute im2col (image-to-column) transformation for convolutions.
     *
     * Converts an image tensor into a column matrix where each column
     * contains a flattened receptive-field patch, enabling convolution
     * via matrix multiplication.
     *
     * ```
     * src0: kernel [OC, IC, KH, KW]   (shape only — data unused)
     * src1: image  [N, IC, IH, IW]
     * dst:         [N, OH, OW, IC*KH*KW]
     * ```
     *
     * `opParams`: `[s0, s1, p0, p1, d0, d1, is_2D]`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computeIm2col(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src0 = dst.src[0] ?: error("im2col requires kernel tensor (src0)")
        val src1 = dst.src[1] ?: error("im2col requires image tensor (src1)")
        require(src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "im2col: src1 must be F32" }
        require(dst.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "im2col: dst must be F32" }

        val s0   = dst.opParams[0]
        val s1   = dst.opParams[1]
        val p0   = dst.opParams[2]
        val p1   = dst.opParams[3]
        val d0   = dst.opParams[4]
        val d1   = dst.opParams[5]
        val is2D = dst.opParams[6] == 1

        val N  = if (is2D) src1.ne[3] else src1.ne[2]
        val IC = if (is2D) src1.ne[2] else src1.ne[1]
        val IH = if (is2D) src1.ne[1] else 1L
        val IW = src1.ne[0]

        val KH = if (is2D) src0.ne[1] else 1L
        val KW = src0.ne[0]

        val OH = if (is2D) dst.ne[2] else 1L
        val OW = dst.ne[1]

        val ofs0 = if (is2D) src1.nb[3] else src1.nb[2]
        val ofs1 = if (is2D) src1.nb[2] else src1.nb[1]

        for (inn in 0 until N) {
            for (ioh in 0 until OH) {
                for (iow in 0 until OW) {
                    for (iic in 0 until IC) {
                        val dstBase = ((inn * OH * OW + ioh * OW + iow) * (IC * KH * KW)).toInt()
                        for (ikh in 0 until KH) {
                            for (ikw in 0 until KW) {
                                val iiw = iow * s0 + ikw * d0 - p0
                                val iih = ioh * s1 + ikh * d1 - p1
                                val dstIdx = (dstBase + (iic * (KH * KW) + ikh * KW + ikw).toInt())
                                val value = if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                    0.0f
                                } else {
                                    val srcByteOffset = (inn.toULong() * ofs0 + iic.toULong() * ofs1).toInt()
                                    val srcLinear = srcByteOffset / Float.SIZE_BYTES + (iih * IW + iiw).toInt()
                                    readF32Flat(graphAllocator, src1, srcLinear)
                                }
                                writeF32Flat(graphAllocator, dst, dstIdx, value)
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Pooling Operations
    // -----------------------------------------------------------------

    /**
     * Compute 1-D pooling (max or average).
     *
     * Iterates over every row of the source tensor and applies a
     * sliding window of size `k` with stride `s` and padding `p`.
     *
     * `opParams`: `[op, k0, s0, p0]` where `op` is [io.github.kotlinmania.llama.ore.GGMLOpPool] ordinal.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computePool1d(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src = dst.src[0] ?: error("pool_1d requires source tensor")
        require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "pool_1d: source must be F32" }

        val opOrdinal = dst.opParams[0]
        val poolOp = io.github.kotlinmania.llama.ore.GGMLOpPool.entries[opOrdinal]
        val k0 = dst.opParams[1]
        val s0 = dst.opParams[2]
        val p0 = dst.opParams[3]

        val IW = src.ne[0].toInt()
        val OW = dst.ne[0].toInt()
        val nRows = src.numElements().toInt() / IW

        for (ir in 0 until nRows) {
            for (ow in 0 until OW) {
                var res = when (poolOp) {
                    io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> 0.0f
                    io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> -Float.MAX_VALUE
                    else -> error("Unsupported pool op: $poolOp")
                }
                var count = 0
                val base = ow * s0 - p0
                for (ki in 0 until k0) {
                    val j = base + ki
                    if (j < 0 || j >= IW) continue
                    val v = readF32Flat(graphAllocator, src, ir * IW + j)
                    when (poolOp) {
                        io.github.kotlinmania.llama.ore.GGMLOpPool.AVG -> res += v
                        io.github.kotlinmania.llama.ore.GGMLOpPool.MAX -> if (v > res) res = v
                    }
                    count++
                }
                if (poolOp == io.github.kotlinmania.llama.ore.GGMLOpPool.AVG && count > 0) {
                    res /= count
                }
                writeF32Flat(graphAllocator, dst, ir * OW + ow, res)
            }
        }
    }

    /**
     * Compute 2-D pooling (max or average).
     *
     * `opParams`: `[op, k0, k1, s0, s1, p0, p1]`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computePool2d(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Backward pass for 2-D pooling.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computePool2dBack(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Window Partition / Un-partition
    // -----------------------------------------------------------------

    /**
     * Partition a tensor into fixed-size windows (ViT / Swin Transformer).
     *
     * `opParams`: `[nep0, nep1, w]` — number of patches in each axis
     * and window size.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computeWinPart(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src0 = dst.src[0] ?: error("win_part requires source tensor")
        require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "win_part: source must be F32" }

        val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]
        val ne0  = dst.ne[0]; val ne1 = dst.ne[1]; val ne2 = dst.ne[2]; val ne3 = dst.ne[3]

        val nep0 = dst.opParams[0]
        val nep1 = dst.opParams[1]
        val w    = dst.opParams[2]

        for (py in 0 until nep1) {
            for (px in 0 until nep0) {
                val i3 = py * nep0 + px
                for (i2 in 0 until ne2.toInt()) {
                    for (i1 in 0 until ne1.toInt()) {
                        for (i0 in 0 until ne0.toInt()) {
                            val i02 = py * w + i2
                            val i01 = px * w + i1
                            val dstIdx = (i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0).toInt()
                            val value = if (i02 >= ne02 || i01 >= ne01) {
                                0.0f
                            } else {
                                val srcIdx = (i02 * ne01 * ne00 + i01 * ne00 + i0).toInt()
                                readF32Flat(graphAllocator, src0, srcIdx)
                            }
                            writeF32Flat(graphAllocator, dst, dstIdx, value)
                        }
                    }
                }
            }
        }
    }

    /**
     * Reverse window partitioning — reassemble windows into the
     * original spatial layout.
     *
     * `opParams[0]` = window size `w`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computeWinUnpart(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src0 = dst.src[0] ?: error("win_unpart requires source tensor")
        require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "win_unpart: source must be F32" }

        val ne00 = src0.ne[0]; val ne01 = src0.ne[1]; val ne02 = src0.ne[2]
        val ne0  = dst.ne[0]; val ne1 = dst.ne[1]; val ne2 = dst.ne[2]

        val w = dst.opParams[0]
        val px = (w - ne1.toInt() % w) % w
        val npx = (px + ne1.toInt()) / w

        for (i2 in 0 until ne2.toInt()) {
            for (i1 in 0 until ne1.toInt()) {
                for (i0 in 0 until ne0.toInt()) {
                    val ip2 = i2 / w
                    val ip1 = i1 / w
                    val i02 = i2 % w
                    val i01 = i1 % w
                    val srcIdx = ((ip2 * npx + ip1) * ne02.toInt() * ne01.toInt() * ne00.toInt() +
                                  i02 * ne01.toInt() * ne00.toInt() + i01 * ne00.toInt() + i0)
                    val dstIdx = (i2 * ne1.toInt() * ne0.toInt() + i1 * ne0.toInt() + i0)
                    writeF32Flat(graphAllocator, dst, dstIdx, readF32Flat(graphAllocator, src0, srcIdx))
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Relative Position Encoding
    // -----------------------------------------------------------------

    /**
     * Extract relative position embeddings (SAM-style).
     *
     * Produces a `(ne0 × ne1 × ne2)` tensor by indexing into `src0`
     * using `pos = (w - i1 - 1) + i2`.
     *
     * Reference: segment-anything `image_encoder.py` L292-L322.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeGetRelPos(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Add height and width relative position biases to an attention
     * map (SAM-style).
     *
     * ```
     * dst[jdh + j]      += src2_e
     * dst[jdw + j*ne10]  += src1_e
     * ```
     *
     * Sources: `src[0]=attn`, `src[1]=rel_h`, `src[2]=rel_w`.
     * `opParams[0]` = inplace flag.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computeAddRelPos(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Custom Map Operations
    // -----------------------------------------------------------------

    /**
     * Dispatch a user-registered custom-1 operation.
     *
     * In the C++ backend, `op_params` holds a function pointer and
     * user-data.  In Kotlin the callback must be registered separately
     * (e.g. via a lambda map keyed by tensor name).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor.
     */
    fun computeMapCustom1(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Dispatch a user-registered custom-2 (binary) operation.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor.
     */
    fun computeMapCustom2(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Dispatch a user-registered custom-3 (ternary) operation.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor.
     */
    fun computeMapCustom3(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Unary Dispatch
    // -----------------------------------------------------------------

    /**
     * Dispatch table for element-wise unary operations.
     *
     * Routes to the appropriate kernel based on the [io.github.kotlinmania.llama.ore.GGMLUnaryOp]
     * stored in `dst.opParams`.  Currently dispatches to existing
     * `computeRelu`, `computeGelu`, `computeSilu`, etc.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeUnary(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src = dst.src[0] ?: error("unary op requires source tensor")
        val unaryOp = io.github.kotlinmania.llama.ore.GGMLUnaryOp.entries[dst.opParams[0]]
        when (unaryOp) {
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.ABS    -> computeAbs(graphAllocator, dst)
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.NEG    -> io.github.kotlinmania.llama.ore.computeNeg(
                graphAllocator,
                src,
                dst
            )
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.RELU   -> io.github.kotlinmania.llama.ore.computeRelu(
                graphAllocator,
                src,
                dst
            )
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU   -> io.github.kotlinmania.llama.ore.computeGelu(
                graphAllocator,
                src,
                dst
            )
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.SILU   -> io.github.kotlinmania.llama.ore.computeSilu(
                graphAllocator,
                src,
                dst
            )
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.EXP    -> computeExp(graphAllocator, dst)
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.TANH   -> computeTanh(graphAllocator, dst)
            io.github.kotlinmania.llama.ore.GGMLUnaryOp.SIGMOID -> computeSigmoid(graphAllocator, dst)
            else -> error("unary op $unaryOp not yet ported")
        }
    }

    /** Element-wise absolute value. */
    private fun computeAbs(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = dst.src[0] ?: error("abs requires source tensor")
        require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "abs: source must be F32" }
        val n = src.numElements().toInt()
        for (i in 0 until n) {
            writeF32Flat(graphAllocator, dst, i, abs(readF32Flat(graphAllocator, src, i)))
        }
    }

    /** Element-wise exponential. */
    private fun computeExp(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = dst.src[0] ?: error("exp requires source tensor")
        require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "exp: source must be F32" }
        val n = src.numElements().toInt()
        for (i in 0 until n) {
            writeF32Flat(graphAllocator, dst, i, exp(readF32Flat(graphAllocator, src, i)))
        }
    }

    /** Element-wise tanh. */
    private fun computeTanh(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = dst.src[0] ?: error("tanh requires source tensor")
        require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tanh: source must be F32" }
        val n = src.numElements().toInt()
        for (i in 0 until n) {
            writeF32Flat(graphAllocator, dst, i, kotlin.math.tanh(readF32Flat(graphAllocator, src, i)))
        }
    }

    /** Element-wise sigmoid: 1 / (1 + exp(-x)). */
    private fun computeSigmoid(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, dst: io.github.kotlinmania.llama.ore.GGMLTensor) {
        val src = dst.src[0] ?: error("sigmoid requires source tensor")
        require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "sigmoid: source must be F32" }
        val n = src.numElements().toInt()
        for (i in 0 until n) {
            val x = readF32Flat(graphAllocator, src, i)
            writeF32Flat(graphAllocator, dst, i, 1.0f / (1.0f + exp(-x)))
        }
    }

    // -----------------------------------------------------------------
    // GLU Dispatch
    // -----------------------------------------------------------------

    /**
     * Dispatch table for Gated Linear Unit variants.
     *
     * Routes to the appropriate kernel based on the [io.github.kotlinmania.llama.ore.GGMLGluOp]
     * stored in `dst.opParams`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeGlu(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val gluOp = io.github.kotlinmania.llama.ore.GGMLGluOp.entries[dst.opParams[0]]
    }

    // -----------------------------------------------------------------
    // Fill
    // -----------------------------------------------------------------

    /**
     * Fill every element of [dst] with a constant value.
     *
     * The fill value is read from `dst.opParams[0]` interpreted as
     * a float (bit-pattern reinterpretation of the stored int32).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computeFill(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val c = Float.fromBits(dst.opParams[0])
        val n = dst.numElements().toInt()
        for (i in 0 until n) {
            writeF32Flat(graphAllocator, dst, i, c)
        }
    }

    // -----------------------------------------------------------------
    // Triangular Solve
    // -----------------------------------------------------------------

    /**
     * Solve a lower-triangular linear system `A·X = B` by forward
     * substitution (column by column).
     *
     * `src[0]` = A (n×n, lower triangular),
     * `src[1]` = B (n×k right-hand sides).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor X (pre-allocated, F32, n×k).
     */
    fun computeSolveTri(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src0 = dst.src[0] ?: error("solve_tri requires A tensor (src0)")
        val src1 = dst.src[1] ?: error("solve_tri requires B tensor (src1)")
        require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32 && src1.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "solve_tri: F32 only" }

        val n = src1.ne[1].toInt()   // rows of A (A is n×n)
        val k = src1.ne[0].toInt()   // columns of B

        // Read A, B into flat arrays for simple indexing
        val aSize = n * n
        val bSize = n * k
        val A = FloatArray(aSize) { readF32Flat(graphAllocator, src0, it) }
        val B = FloatArray(bSize) { readF32Flat(graphAllocator, src1, it) }
        val X = FloatArray(bSize)

        // Forward substitution: for each column j of X
        for (j in 0 until k) {
            for (i in 0 until n) {
                var sum = 0.0f
                for (t in 0 until i) {
                    sum += A[i * n + t] * X[t * k + j]
                }
                val diag = A[i * n + i]
                require(diag != 0.0f) { "Zero diagonal in triangular matrix at row $i" }
                X[i * k + j] = (B[i * k + j] - sum) / diag
            }
        }

        for (i in X.indices) {
            writeF32Flat(graphAllocator, dst, i, X[i])
        }
    }

    // -----------------------------------------------------------------
    // Triangular Matrix
    // -----------------------------------------------------------------

    /**
     * Extract triangular part of a matrix, zeroing elements outside
     * the triangle.
     *
     * `opParams[0]` encodes the [io.github.kotlinmania.llama.ore.GGMLTriType]:
     * - LOWER: keep where `col < row`
     * - LOWER_DIAG: keep where `col <= row`
     * - UPPER: keep where `col > row`
     * - UPPER_DIAG: keep where `col >= row`
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated, F32).
     */
    fun computeTri(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
        val src0 = dst.src[0] ?: error("tri requires source tensor")
        require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "tri: source must be F32" }

        val triType = io.github.kotlinmania.llama.ore.GGMLTriType.entries.first { it.value == dst.opParams[0] }
        val ne0 = dst.ne[0].toInt()
        val ne1 = dst.ne[1].toInt()
        val ne2 = dst.ne[2].toInt()
        val ne3 = dst.ne[3].toInt()

        val predicate: (col: Int, row: Int) -> Boolean = when (triType) {
            io.github.kotlinmania.llama.ore.GGMLTriType.LOWER      -> { col, row -> col < row }
            io.github.kotlinmania.llama.ore.GGMLTriType.LOWER_DIAG -> { col, row -> col <= row }
            io.github.kotlinmania.llama.ore.GGMLTriType.UPPER      -> { col, row -> col > row }
            io.github.kotlinmania.llama.ore.GGMLTriType.UPPER_DIAG -> { col, row -> col >= row }
        }

        var idx = 0
        for (i3 in 0 until ne3) {
            for (i2 in 0 until ne2) {
                for (i1 in 0 until ne1) {
                    for (i0 in 0 until ne0) {
                        val srcVal = readF32Flat(graphAllocator, src0, idx)
                        writeF32Flat(graphAllocator, dst, idx, if (predicate(i0, i1)) srcVal else 0.0f)
                        idx++
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Cross-Entropy Loss
    // -----------------------------------------------------------------

    /**
     * Compute cross-entropy loss: `-mean(sum(labels * log_softmax(logits)))`.
     *
     * Sources: `src[0]` = logits, `src[1]` = labels.
     * Result is a scalar written to `dst`.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination scalar tensor (pre-allocated, F32).
     */
    fun computeCrossEntropyLoss(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * Backward pass for cross-entropy loss.
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (pre-allocated).
     */
    fun computeCrossEntropyLossBack(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    // -----------------------------------------------------------------
    // Optimiser Step Operations
    // -----------------------------------------------------------------

    /**
     * AdamW optimiser step (in-place weight update).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (weights, updated in-place).
     */
    fun computeOptStepAdamw(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }

    /**
     * SGD optimiser step (in-place weight update).
     *
     * @param graphAllocator Memory allocator for tensor data access.
     * @param dst Destination tensor (weights, updated in-place).
     */
    fun computeOptStepSgd(
        graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
        dst: io.github.kotlinmania.llama.ore.GGMLTensor
    ) {
    }
    // ---- Wrappers for structural / advanced compute operations ----

    private val defaultParams = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 0, nth = 1)

    private fun computeConcatNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeConcat(graphAllocator, defaultParams, node)

    private fun computeSumRowsNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeSumRows(graphAllocator, defaultParams, node)

    private fun computeCumsumNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeCumsum(graphAllocator, defaultParams, node)

    private fun computeArgmaxNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeArgmax(graphAllocator, defaultParams, node)

    private fun computeCountEqualNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeCountEqual(graphAllocator, defaultParams, node)

    private fun computeGetRowsNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeGetRows(graphAllocator, defaultParams, node)

    private fun computeGetRowsBackNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeGetRowsBack(graphAllocator, defaultParams, node)

    private fun computeSetRowsNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeSetRows(graphAllocator, defaultParams, node)

    private fun computeDiagNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeDiag(graphAllocator, defaultParams, node)

    private fun computeDiagMaskInfNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeDiagMaskInf(graphAllocator, defaultParams, node)

    private fun computeDiagMaskZeroNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeDiagMaskZero(graphAllocator, defaultParams, node)

    private fun computeRopeNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeRope(graphAllocator, defaultParams, node)

    private fun computeRopeBackNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeRopeBack(graphAllocator, defaultParams, node)

    private fun computePadNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computePad(graphAllocator, defaultParams, node)

    private fun computePadReflect1DNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computePadReflect1D(graphAllocator, defaultParams, node)

    private fun computeRollNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeRoll(graphAllocator, defaultParams, node)

    private fun computeArangeNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeArange(graphAllocator, defaultParams, node)

    private fun computeTimestepEmbeddingNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeTimestepEmbedding(
            graphAllocator,
            defaultParams,
            node
        )

    private fun computeArgsortNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeArgsort(graphAllocator, defaultParams, node)

    private fun computeTopKNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeTopK(graphAllocator, defaultParams, node)

    private fun computeUpscaleNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeUpscale(graphAllocator, defaultParams, node)

    private fun computeOutProdNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeOutProd(graphAllocator, defaultParams, node)

    private fun computeCrossEntropyLossNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeCrossEntropyLoss(graphAllocator, defaultParams, node)

    private fun computeCrossEntropyLossBackNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeCrossEntropyLossBack(
            graphAllocator,
            defaultParams,
            node
        )

    private fun computeOptStepAdamwNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeOptStepAdamw(graphAllocator, defaultParams, node)

    private fun computeOptStepSgdNode(graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator, node: io.github.kotlinmania.llama.ore.GGMLTensor) =
        io.github.kotlinmania.llama.ore.computeOptStepSgd(graphAllocator, defaultParams, node)
}

// =====================================================================
// Constants for activation functions (ported from ggml-cpu/vec.h)
// =====================================================================

/** Coefficient used in the GELU-Quick approximation: x·σ(−1.702·x) */
private const val GELU_QUICK_COEF = -1.702f

/** 1/√2, used in the erf-based GELU: 0.5·x·(1 + erf(x/√2)) */
private const val SQRT_2_INV = 0.7071067811865475f // 1.0 / sqrt(2.0)

// =====================================================================
// Approximate erf(x) — Abramowitz & Stegun §7.1.26 (max error ≈ 1.5e-7)
// Used by GELU-ERF and GEGLU-ERF where the exact C erff() is unavailable.
// =====================================================================

private fun erfApprox(x: Float): Float {
    val a1 =  0.254829592f
    val a2 = -0.284496736f
    val a3 =  1.421413741f
    val a4 = -1.453152027f
    val a5 =  1.061405429f
    val p  =  0.3275911f
    val s = if (x < 0f) -1f else 1f
    val ax = abs(x)
    val t = 1f / (1f + p * ax)
    val y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax)
    return s * y
}

// =====================================================================
// Generic unary element-wise helper
// =====================================================================

/**
 * Applies a scalar function [op] element-wise from [src] into [dst].
 * Supports F32 and F16 types. Mirrors the scalar (non-SIMD) path of
 * every `ggml_compute_forward_<unary>` function in ops.cpp.
 */
private fun computeUnaryElementwise(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    opName: String,
    op: (Float) -> Float
) {
    require(src.type == dst.type) { "$opName: src and dst types must match" }
    val total = dst.numElements().toInt()
    when (src.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            total
        ) { _, ind ->
            dst.setFloat(graphAllocator, op(src.getFloat(graphAllocator, *ind)), *ind)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            total
        ) { _, ind ->
            dst.setHalf(graphAllocator, op(src.getHalf(graphAllocator, *ind)), *ind)
        }
        else -> error("fatal error")
    }
}

// =====================================================================
// SILU backward  — ported from ggml_compute_forward_silu_back (ops.cpp:2736)
// dx[i] = dy[i] · σ(x[i]) · (1 + x[i]·(1 − σ(x[i])))
// =====================================================================

/**
 * Computes the backward pass of the SiLU (Sigmoid Linear Unit) activation.
 *
 * Given upstream gradient [grad] (`dy`) and forward-pass input [src1] (`x`),
 * writes `dx` into [dst] where `dx[i] = dy[i] · σ(x[i]) · (1 + x[i]·(1 − σ(x[i])))`.
 */
fun computeSiluBack(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    grad: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src1.type == dst.type) { "SILU_BACK: src1 and dst types must match" }
    require(grad.type == dst.type) { "SILU_BACK: grad and dst types must match" }
    val total = dst.numElements().toInt()
    when (dst.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            total
        ) { _, ind ->
            val x = src1.getFloat(graphAllocator, *ind)
            val dy = grad.getFloat(graphAllocator, *ind)
            val s = 1f / (1f + exp(-x))
            dst.setFloat(graphAllocator, dy * s * (1f + x * (1f - s)), *ind)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            total
        ) { _, ind ->
            val x = src1.getHalf(graphAllocator, *ind)
            val dy = grad.getHalf(graphAllocator, *ind)
            val s = 1f / (1f + exp(-x))
            dst.setHalf(graphAllocator, dy * s * (1f + x * (1f - s)), *ind)
        }
        else -> error("fatal error")
    }
}

// =====================================================================
// Layer Norm  — ported from ggml_compute_forward_norm_f32 (ops.cpp:3649)
// y = (x − mean) / √(variance + eps)
// =====================================================================

/**
 * Computes Layer Normalization over the innermost dimension (ne[0]).
 *
 * For each row: `y = (x − mean) / √(variance + eps)` where mean and variance
 * are computed along the first axis.
 */
fun computeLayerNorm(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    eps: Float,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "NORM (layer norm) only supports F32, got ${src.type}" }
    require(eps >= 0f) { "NORM eps must be >= 0" }

    val ne00 = src.ne[0].toInt()
    val ne01 = src.ne[1].toInt().coerceAtLeast(1)
    val ne02 = src.ne[2].toInt().coerceAtLeast(1)
    val ne03 = src.ne[3].toInt().coerceAtLeast(1)
    val idx = IntArray(4)

    for (i03 in 0 until ne03) {
        idx[3] = i03
        for (i02 in 0 until ne02) {
            idx[2] = i02
            for (i01 in 0 until ne01) {
                idx[1] = i01
                // compute mean
                var sum = 0.0
                for (i00 in 0 until ne00) {
                    idx[0] = i00
                    sum += src.getFloat(graphAllocator, *idx).toDouble()
                }
                val mean = (sum / ne00).toFloat()

                // compute variance and write (x − mean) into dst
                var variance = 0.0
                for (i00 in 0 until ne00) {
                    idx[0] = i00
                    val v = src.getFloat(graphAllocator, *idx) - mean
                    dst.setFloat(graphAllocator, v, *idx)
                    variance += (v * v).toDouble()
                }
                variance /= ne00

                val scale = 1f / sqrt((variance + eps).toFloat())
                for (i00 in 0 until ne00) {
                    idx[0] = i00
                    val cur = dst.getFloat(graphAllocator, *idx)
                    dst.setFloat(graphAllocator, cur * scale, *idx)
                }
            }
        }
    }
}

// =====================================================================
// RMS Norm Back — ported from ggml_compute_forward_rms_norm_back_f32 (ops.cpp:3785)
// dx = (dz + x·(−sum_xdz / sum_eps)) · rrms
// =====================================================================

/**
 * Computes the backward pass of RMS Normalization.
 */
fun computeRmsNormBack(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    grad: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor,
    eps: Float,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(grad.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "RMS_NORM_BACK only supports F32" }

    val ne00 = src1.ne[0].toInt()
    val ne01 = src1.ne[1].toInt().coerceAtLeast(1)
    val ne02 = src1.ne[2].toInt().coerceAtLeast(1)
    val ne03 = src1.ne[3].toInt().coerceAtLeast(1)
    val idx = IntArray(4)

    for (i03 in 0 until ne03) {
        idx[3] = i03
        for (i02 in 0 until ne02) {
            idx[2] = i02
            for (i01 in 0 until ne01) {
                idx[1] = i01

                var sumXX = 0.0
                var sumXDZ = 0.0
                for (i00 in 0 until ne00) {
                    idx[0] = i00
                    val x = src1.getFloat(graphAllocator, *idx).toDouble()
                    val dz = grad.getFloat(graphAllocator, *idx).toDouble()
                    sumXX += x * x
                    sumXDZ += x * dz
                }
                val meanEps = (sumXX / ne00 + eps).toFloat()
                val sumEps = (sumXX + eps.toDouble() * ne00).toFloat()
                val rrms = 1f / sqrt(meanEps)

                for (i00 in 0 until ne00) {
                    idx[0] = i00
                    val x = src1.getFloat(graphAllocator, *idx)
                    val dz = grad.getFloat(graphAllocator, *idx)
                    val dx = (x * (-sumXDZ.toFloat() / sumEps) + dz) * rrms
                    dst.setFloat(graphAllocator, dx, *idx)
                }
            }
        }
    }
}

// =====================================================================
// Group Norm — ported from ggml_compute_forward_group_norm_f32 (ops.cpp:3962)
// =====================================================================

/**
 * Computes Group Normalization.
 *
 * Splits channels (ne[2]) into [nGroups] groups and normalises each group
 * over all spatial positions and channels within the group.
 */
fun computeGroupNorm(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    nGroups: Int,
    eps: Float,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "GROUP_NORM only supports F32" }

    val ne00 = src.ne[0].toInt()
    val ne01 = src.ne[1].toInt().coerceAtLeast(1)
    val nChannels = src.ne[2].toInt().coerceAtLeast(1)
    val ne03 = src.ne[3].toInt().coerceAtLeast(1)
    val nChannelsPerGroup = (nChannels + nGroups - 1) / nGroups
    val idx = IntArray(4)

    for (g in 0 until nGroups) {
        val start = g * nChannelsPerGroup
        val end = min(start + nChannelsPerGroup, nChannels)
        val step = end - start

        for (i03 in 0 until ne03) {
            idx[3] = i03
            // compute mean
            var sum = 0.0
            for (i02 in start until end) {
                idx[2] = i02
                for (i01 in 0 until ne01) {
                    idx[1] = i01
                    for (i00 in 0 until ne00) {
                        idx[0] = i00
                        sum += src.getFloat(graphAllocator, *idx).toDouble()
                    }
                }
            }
            val mean = (sum / (ne00.toLong() * ne01 * step)).toFloat()

            // compute variance and write (x − mean) into dst
            var sum2 = 0.0
            for (i02 in start until end) {
                idx[2] = i02
                for (i01 in 0 until ne01) {
                    idx[1] = i01
                    for (i00 in 0 until ne00) {
                        idx[0] = i00
                        val v = src.getFloat(graphAllocator, *idx) - mean
                        dst.setFloat(graphAllocator, v, *idx)
                        sum2 += (v * v).toDouble()
                    }
                }
            }
            val variance = (sum2 / (ne00.toLong() * ne01 * step)).toFloat()
            val scale = 1f / sqrt(variance + eps)

            for (i02 in start until end) {
                idx[2] = i02
                for (i01 in 0 until ne01) {
                    idx[1] = i01
                    for (i00 in 0 until ne00) {
                        idx[0] = i00
                        val cur = dst.getFloat(graphAllocator, *idx)
                        dst.setFloat(graphAllocator, cur * scale, *idx)
                    }
                }
            }
        }
    }
}

// =====================================================================
// ACC (accumulate) — ported from ggml_compute_forward_acc_f32 (ops.cpp:1154)
// dst = src0;  then dst[offset..] += src1
// =====================================================================

/**
 * Accumulates [src1] into [src0], writing the result into [dst].
 * The C++ version supports a byte offset via opParams; this scalar path
 * accumulates src1 element-wise at matching indices.
 */
fun computeAcc(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "ACC only supports F32, got ${src0.type}" }

    val inplace = dst.opParams[4] != 0
    val total0 = src0.numElements().toInt()

    if (!inplace) {
        io.github.kotlinmania.llama.ore.applyNDIter(src0, total0) { _, ind ->
            dst.setFloat(graphAllocator, src0.getFloat(graphAllocator, *ind), *ind)
        }
    }

    val total1 = src1.numElements().toInt()
    io.github.kotlinmania.llama.ore.applyNDIter(src1, total1) { _, ind ->
        val existing = dst.getFloat(graphAllocator, *ind)
        dst.setFloat(graphAllocator, existing + src1.getFloat(graphAllocator, *ind), *ind)
    }
}

// =====================================================================
// SCALE — ported from ggml_compute_forward_scale_f32 (ops.cpp:4382)
// dst = src * s + b
// =====================================================================

/**
 * Scales every element of [src] by factor [s] and optionally adds bias [b].
 */
fun computeScale(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    s: Float,
    b: Float,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "SCALE only supports F32, got ${src.type}" }
    val total = dst.numElements().toInt()
    if (b == 0f) {
        io.github.kotlinmania.llama.ore.applyNDIter(dst, total) { _, ind ->
            dst.setFloat(graphAllocator, src.getFloat(graphAllocator, *ind) * s, *ind)
        }
    } else {
        io.github.kotlinmania.llama.ore.applyNDIter(dst, total) { _, ind ->
            dst.setFloat(graphAllocator, src.getFloat(graphAllocator, *ind) * s + b, *ind)
        }
    }
}

// =====================================================================
// CLAMP — ported from ggml_compute_forward_clamp (ops.cpp:5472)
// dst[i] = clamp(src[i], min, max)
// =====================================================================

/**
 * Clamps every element of [src] to the range [[minVal], [maxVal]].
 */
fun computeClamp(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src: io.github.kotlinmania.llama.ore.GGMLTensor,
    minVal: Float,
    maxVal: Float,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val total = dst.numElements().toInt()
    when (src.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            total
        ) { _, ind ->
            dst.setFloat(graphAllocator, max(minVal, min(src.getFloat(graphAllocator, *ind), maxVal)), *ind)
        }
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> io.github.kotlinmania.llama.ore.applyNDIter(
            dst,
            total
        ) { _, ind ->
            dst.setHalf(graphAllocator, max(minVal, min(src.getHalf(graphAllocator, *ind), maxVal)), *ind)
        }
        else -> error("fatal error")
    }
}

// =====================================================================
// SET — ported from ggml_compute_forward_set_f32 (ops.cpp:4454)
// dst = src0 then overlay src1 at element positions of src1
// =====================================================================

/**
 * Copies [src0] into [dst], then overwrites a region with data from [src1].
 */
fun computeSet(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32 || src0.type == io.github.kotlinmania.llama.ore.GGMLType.I32) {
        "SET only supports F32 and I32, got ${src0.type}"
    }
    val inplace = dst.opParams[4] != 0

    if (!inplace) {
        val total0 = src0.numElements().toInt()
        when (src0.type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src0,
                total0
            ) { _, ind ->
                dst.setFloat(graphAllocator, src0.getFloat(graphAllocator, *ind), *ind)
            }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> io.github.kotlinmania.llama.ore.applyNDIter(
                src0,
                total0
            ) { _, ind ->
                dst.setInt(graphAllocator, src0.getInt(graphAllocator, *ind), *ind)
            }
            else -> {}
        }
    }

    val total1 = src1.numElements().toInt()
    when (src0.type) {
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            src1,
            total1
        ) { _, ind ->
            dst.setFloat(graphAllocator, src1.getFloat(graphAllocator, *ind), *ind)
        }
        io.github.kotlinmania.llama.ore.GGMLType.I32 -> io.github.kotlinmania.llama.ore.applyNDIter(
            src1,
            total1
        ) { _, ind ->
            dst.setInt(graphAllocator, src1.getInt(graphAllocator, *ind), *ind)
        }
        else -> {}
    }
}

// =====================================================================
// GLU variant helpers — each applies gate(x) · g element-wise
// =====================================================================

/**
 * ReGLU: `dst[i] = max(0, x[i]) · g[i]`
 * Ported from ggml_vec_reglu_f32 (vec.h)
 */
fun computeReglu(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.computeGluVariant(
        graphAllocator,
        src0,
        src1,
        dst,
        "REGLU"
    ) { x, g ->
        if (x > 0f) x * g else 0f
    }
}

/**
 * GEGLU: `dst[i] = gelu(x[i]) · g[i]`
 * Ported from ggml_vec_geglu_f32 (vec.h)
 */
fun computeGeglu(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    val gelu = { x: Float -> x * 0.5f * (1f + tanh(0.797885f * (x + 0.044715f * x * x * x))) }
    io.github.kotlinmania.llama.ore.computeGluVariant(
        graphAllocator,
        src0,
        src1,
        dst,
        "GEGLU"
    ) { x, g ->
        gelu(x) * g
    }
}

/**
 * SwiGLU: `dst[i] = silu(x[i]) · g[i]`  where silu(x) = x·σ(x)
 * Ported from ggml_vec_swiglu_f32 (vec.h)
 */
fun computeSwiglu(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.computeGluVariant(
        graphAllocator,
        src0,
        src1,
        dst,
        "SWIGLU"
    ) { x, g ->
        (x / (1f + exp(-x))) * g
    }
}

/**
 * SwiGLU-OAI (OpenAI variant): `dst[i] = silu_clamped(x[i]) · (g[i] + 1)`
 * Ported from ggml_compute_forward_swiglu_oai_f32 (ops.cpp:3276)
 */
fun computeSwigluOai(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    alpha: Float,
    limit: Float
) {
    require(src0.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "SWIGLU_OAI only supports F32" }
    val swapped = dst.opParams[1] != 0
    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nRows = (src0.numElements() / src0.ne[0]).toInt().coerceAtLeast(1)
    val idx = IntArray(4)

    for (row in 0 until nRows) {
        var tmp = row
        for (d in 1 until src0.ne.size) {
            val dimSize = src0.ne[d].toInt().coerceAtLeast(1)
            idx[d] = tmp % dimSize
            tmp /= dimSize
        }
        for (k in 0 until nc) {
            val gateIdx = idx.copyOf()
            val valIdx = idx.copyOf()
            if (src1 != null) {
                gateIdx[0] = k
                valIdx[0] = k
            } else {
                gateIdx[0] = if (swapped) k + nc else k
                valIdx[0] = if (swapped) k else k + nc
            }
            val x = min(
                if (src1 != null) src0.getFloat(graphAllocator, *gateIdx)
                else src0.getFloat(graphAllocator, *gateIdx),
                limit
            )
            val rawY = if (src1 != null) src1.getFloat(graphAllocator, *valIdx)
                       else src0.getFloat(graphAllocator, *valIdx)
            val y = rawY.coerceIn(-limit, limit)
            val outGlu = x / (1f + exp(alpha * (-x)))
            idx[0] = k
            dst.setFloat(graphAllocator, outGlu * (y + 1f), *idx)
        }
    }
}

/**
 * GEGLU-ERF: `dst[i] = (0.5·x·(1 + erf(x/√2))) · g`
 * Ported from ggml_vec_geglu_erf_f32 (vec.h)
 */
fun computeGegluErf(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.computeGluVariant(
        graphAllocator,
        src0,
        src1,
        dst,
        "GEGLU_ERF"
    ) { x, g ->
        0.5f * x * (1f + io.github.kotlinmania.llama.ore.erfApprox(x * io.github.kotlinmania.llama.ore.SQRT_2_INV)) * g
    }
}

/**
 * GEGLU-Quick: `dst[i] = gelu_quick(x[i]) · g[i]`
 * Ported from ggml_vec_geglu_quick_f32 (vec.h)
 */
fun computeGegluQuick(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor
) {
    io.github.kotlinmania.llama.ore.computeGluVariant(
        graphAllocator,
        src0,
        src1,
        dst,
        "GEGLU_QUICK"
    ) { x, g ->
        (x * (1f / (1f + exp(io.github.kotlinmania.llama.ore.GELU_QUICK_COEF * x)))) * g
    }
}

// =====================================================================
// Internal GLU helper — shared structure for all gated variants
// =====================================================================

/**
 * Generic GLU kernel.  If [src1] is non-null the gate comes from src0 and
 * the value from src1 (two-tensor mode, ne[0] == nc).  If src1 is null
 * the first half of src0's rows are gate and the second half are value
 * (single-tensor mode, nc = ne[0]/2).
 *
 * The `swapped` flag comes from `opParams[1]` and swaps gate/value halves.
 */
private fun computeGluVariant(
    graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator,
    src0: io.github.kotlinmania.llama.ore.GGMLTensor,
    src1: io.github.kotlinmania.llama.ore.GGMLTensor?,
    dst: io.github.kotlinmania.llama.ore.GGMLTensor,
    opName: String,
    fn: (gate: Float, value: Float) -> Float
) {
    val swapped = dst.opParams[1] != 0
    val nc = if (src1 != null) src0.ne[0].toInt() else (src0.ne[0] / 2).toInt()
    val nRows = (src0.numElements() / src0.ne[0]).toInt().coerceAtLeast(1)
    val idx = IntArray(4)

    for (row in 0 until nRows) {
        var tmp = row
        for (d in 1 until src0.ne.size) {
            val dimSize = src0.ne[d].toInt().coerceAtLeast(1)
            idx[d] = tmp % dimSize
            tmp /= dimSize
        }
        for (k in 0 until nc) {
            val gateIdx = idx.copyOf()
            val valIdx = idx.copyOf()
            if (src1 != null) {
                gateIdx[0] = k
                valIdx[0] = k
            } else {
                gateIdx[0] = if (swapped) k + nc else k
                valIdx[0] = if (swapped) k else k + nc
            }

            when (src0.type) {
                io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                    val x = src0.getFloat(graphAllocator, *gateIdx)
                    val g = if (src1 != null) src1.getFloat(graphAllocator, *valIdx)
                            else src0.getFloat(graphAllocator, *valIdx)
                    idx[0] = k
                    dst.setFloat(graphAllocator, fn(x, g), *idx)
                }
                io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                    val x = src0.getHalf(graphAllocator, *gateIdx)
                    val g = if (src1 != null) src1.getHalf(graphAllocator, *valIdx)
                            else src0.getHalf(graphAllocator, *valIdx)
                    idx[0] = k
                    dst.setHalf(graphAllocator, fn(x, g), *idx)
                }
                else -> error("fatal error")
            }
        }
    }

}
