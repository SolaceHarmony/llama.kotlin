package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.core.GGMLGraphAllocator // Required for new function signatures
import kotlin.math.abs
import kotlin.math.round
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES

/**
 * Kotlin Native port of GGML tensor computation operations.
 * This file contains the implementation of actual computation functionality for tensor operations.
 */
fun calculateTotalSize(ne: LongArray): Int {
    var totalSize = 1
    for (i in 0 until GGML_MAX_DIMS) {
        totalSize *= ne[i].toInt()
    }
    return totalSize
}

/**
 * Allocates memory for a tensor based on its type and size.
 * (Note: This function is less relevant now that compute ops use graphAllocator for results)
 */
@Suppress("unused")
fun allocateMemory(type: GGMLType, size: Int): Any {
    return when (type) {
        GGMLType.F32 -> FloatArray(size) { 0.0f }
        GGMLType.F16 -> ShortArray(size) { 0 } // Still used by quantizeTensor for F16 intermediate
        GGMLType.I8 -> ByteArray(size) { 0 }
        GGMLType.I16 -> ShortArray(size) { 0 }
        GGMLType.I32 -> IntArray(size) { 0 }
        GGMLType.I64 -> LongArray(size) { 0L }
        else -> ByteArray(size) { 0 } // Default for quantized types
    }
}

/**
 * Computes the dot product of a row from a Q8_0 tensor and a column from an F32 tensor.
 * Used as a core part of Q8_0 x F32 matrix multiplication.
 * Assumes tensorQ80 (src0) is M x K (ne = [K, M])
 * Assumes tensorF32 (src1) is K x N (ne = [N, K])
 */
internal fun computeDotProductQ80F32(
    graphAllocator: GGMLGraphAllocator,
    tensorQ80: GGMLTensor,    // M x K (ne[0]=K items per row, ne[1]=M rows)
    tensorF32: GGMLTensor,    // K x N (ne[0]=N items per row, ne[1]=K rows)
    rowIndexInQ80: Int,     // Row index 'i' for tensorQ80 (0 to M-1)
    colIndexInF32: Int,     // Column index 'j' for tensorF32 (0 to N-1)
    commonDimK: Int         // The shared dimension K (should be tensorQ80.ne[0] and tensorF32.ne[1])
): Float {
    require(tensorQ80.type == GGMLType.Q8_0) { "tensorQ80 must be Q8_0. Got ${tensorQ80.type}" }
    require(tensorF32.type == GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ80.ne[0].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val flatIndexInQ80 = rowIndexInQ80 * commonDimK + k
        val blockIndexQ80 = flatIndexInQ80 / QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % QK8_0
        val scale = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeight = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)
        val dequantizedQ80Value = scale * qWeight.toFloat()
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)
        sumF32 += dequantizedQ80Value * f32Value
    }
    return sumF32
}

internal fun computeDotProductQ41F32(
    graphAllocator: GGMLGraphAllocator,
    tensorQ41: GGMLTensor,    // Assumed layout M x K (ne[1] = M rows, ne[0] = K elements per row for access)
    tensorF32: GGMLTensor,    // Assumed layout K x N (ne[1] = K rows, ne[0] = N columns for access)
    rowIndexInQ41: Int,     // Row index 'i' for tensorQ41 (0 to M-1)
    colIndexInF32: Int,     // Column index 'j' for tensorF32 (0 to N-1)
    commonDimK: Int         // The shared dimension K, should match tensorQ41.ne[0] and tensorF32.ne[1]
): Float {
    require(tensorQ41.type == GGMLType.Q4_1) { "computeDotProductQ41F32: tensorQ41 must be Q4_1. Got ${tensorQ41.type}" }
    require(tensorF32.type == GGMLType.F32) { "computeDotProductQ41F32: tensorF32 must be F32. Got ${tensorF32.type}" }

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
        val blockIndexQ41 = flatIndexInQ41 / QK4_1
        val itemInBlockQ41 = flatIndexInQ41 % QK4_1

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
    graphAllocator: GGMLGraphAllocator,
    tensorQ40: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorF32: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ40: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ40.type == GGMLType.Q4_0) { "computeDotProductQ40F32: tensorQ40 must be Q4_0. Got ${tensorQ40.type}" }
    require(tensorF32.type == GGMLType.F32) { "computeDotProductQ40F32: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ40.ne[0].toInt() == commonDimK) { "tensorQ40 K dim (${tensorQ40.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val flatIndexInQ40 = rowIndexInQ40 * commonDimK + k
        val blockIndexQ40 = flatIndexInQ40 / QK4_0
        val itemInBlockQ40 = flatIndexInQ40 % QK4_0
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
    graphAllocator: GGMLGraphAllocator,
    tensorQ2_K: GGMLTensor,
    tensorF32: GGMLTensor,
    rowIndexInQ2_K: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ2_K.type == GGMLType.Q2_K) { "tensorQ2_K must be Q2_K. Got ${tensorQ2_K.type}" }
    require(tensorF32.type == GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ2_K.ne[0].toInt() == commonDimK) { "tensorQ2_K K dim (${tensorQ2_K.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    
    // Process in blocks of QK_K
    for (blockStart in 0 until commonDimK step QK_K) {
        val blockEnd = minOf(blockStart + QK_K, commonDimK)
        val blockSize = blockEnd - blockStart
        
        if (blockSize == QK_K) {
            // Full block optimization
            val flatIndexStart = rowIndexInQ2_K * commonDimK + blockStart
            val blockIndex = flatIndexStart / QK_K
            
            val d = tensorQ2_K.getQ2_KBlockScale(graphAllocator, blockIndex)
            val dmin = tensorQ2_K.getQ2_KBlockScaleMin(graphAllocator, blockIndex)
            
            // Process sub-blocks
            for (subBlock in 0 until QK_K/16) {
                val scaleAndMin = tensorQ2_K.getQ2_KScale(graphAllocator, blockIndex, subBlock)
                val quantizedScale = scaleAndMin.toInt() and 0x0F
                val quantizedMin = (scaleAndMin.toInt() shr 4) and 0x0F
                
                val scale = (quantizedScale.toFloat() / 15.0f) * d
                val min = (quantizedMin.toFloat() * d) + dmin
                
                // Process 16 values in this sub-block
                for (i in 0 until 16 step 4) {
                    val quantByte = tensorQ2_K.getQ2_KQuant(graphAllocator, blockIndex, subBlock * 4 + i / 4)
                    
                    for (j in 0 until 4) {
                        val k = blockStart + subBlock * 16 + i + j
                        if (k < blockEnd) {
                            val quantizedValue = (quantByte.toInt() shr (j * 2)) and 0x03
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
                val blockIndex = flatIndex / QK_K
                val itemInBlock = flatIndex % QK_K
                
                // Simplified dequantization for partial blocks
                val d = tensorQ2_K.getQ2_KBlockScale(graphAllocator, blockIndex)
                val dmin = tensorQ2_K.getQ2_KBlockScaleMin(graphAllocator, blockIndex)
                
                val subBlock = itemInBlock / 16
                val scaleAndMin = tensorQ2_K.getQ2_KScale(graphAllocator, blockIndex, subBlock)
                val quantizedScale = scaleAndMin.toInt() and 0x0F
                val quantizedMin = (scaleAndMin.toInt() shr 4) and 0x0F
                
                val scale = (quantizedScale.toFloat() / 15.0f) * d
                val min = (quantizedMin.toFloat() * d) + dmin
                
                val quantByteIdx = (subBlock * 4) + ((itemInBlock % 16) / 4)
                val quantByte = tensorQ2_K.getQ2_KQuant(graphAllocator, blockIndex, quantByteIdx)
                val bitPos = ((itemInBlock % 16) % 4) * 2
                val quantizedValue = (quantByte.toInt() shr bitPos) and 0x03
                
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
    graphAllocator: GGMLGraphAllocator,
    tensorQ4_K: GGMLTensor,
    tensorF32: GGMLTensor,
    rowIndexInQ4_K: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ4_K.type == GGMLType.Q4_K) { "tensorQ4_K must be Q4_K. Got ${tensorQ4_K.type}" }
    require(tensorF32.type == GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ4_K.ne[0].toInt() == commonDimK) { "tensorQ4_K K dim (${tensorQ4_K.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    
    // Process in blocks of QK_K
    for (blockStart in 0 until commonDimK step QK_K) {
        val blockEnd = minOf(blockStart + QK_K, commonDimK)
        val blockSize = blockEnd - blockStart
        
        if (blockSize == QK_K) {
            val flatIndexStart = rowIndexInQ4_K * commonDimK + blockStart
            val blockIndex = flatIndexStart / QK_K
            
            val d = tensorQ4_K.getQ4_KBlockScale(graphAllocator, blockIndex)
            val dmin = tensorQ4_K.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
            
            val buffer = graphAllocator.buffers[tensorQ4_K.bufferId] ?: throw IllegalStateException("Tensor buffer not found")
            val blockByteOffset = blockIndex * tensorQ4_K.type.byteSize.toInt()
            
            // Process 8 sub-blocks of 32 elements each
            for (subBlock in 0 until 8) {
                // Get quantized scale and min for this sub-block
                val scaleByteOffset = blockByteOffset + 4 + subBlock
                val scaleByte = buffer[(tensorQ4_K.dataOffset + scaleByteOffset.toULong()).toInt()]
                val quantizedScale = scaleByte.toInt() and 0x3F
                val quantizedMinLow = (scaleByte.toInt() shr 6) and 0x03
                
                val minByteOffset = blockByteOffset + 4 + subBlock * 2 + 1
                val quantizedMinHigh = if (minByteOffset < blockByteOffset + 4 + K_SCALE_SIZE) {
                    buffer[(tensorQ4_K.dataOffset + minByteOffset.toULong()).toInt()].toInt() and 0x0F
                } else 0
                val quantizedMin = quantizedMinLow or (quantizedMinHigh shl 2)
                
                val scale = (quantizedScale.toFloat() / 63.0f) * d
                val min = (quantizedMin.toFloat() / 63.0f) * d + dmin
                
                // Process 32 4-bit values in this sub-block
                val qsBaseOffset = blockByteOffset + 4 + K_SCALE_SIZE + subBlock * 16
                for (i in 0 until 32 step 2) {
                    val k1 = blockStart + subBlock * 32 + i
                    val k2 = blockStart + subBlock * 32 + i + 1
                    
                    if (k1 < blockEnd) {
                        val qsByte = buffer[(tensorQ4_K.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                        
                        val q1 = qsByte.toInt() and 0x0F
                        val dequantizedValue1 = (q1.toFloat() / 15.0f) * scale + min
                        val f32Value1 = tensorF32.getFloat(graphAllocator, colIndexInF32, k1)
                        sumF32 += dequantizedValue1 * f32Value1
                        
                        if (k2 < blockEnd) {
                            val q2 = (qsByte.toInt() shr 4) and 0x0F
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
                val blockIndex = flatIndex / QK_K
                val itemInBlock = flatIndex % QK_K
                
                val d = tensorQ4_K.getQ4_KBlockScale(graphAllocator, blockIndex)
                val dmin = tensorQ4_K.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
                
                // Simplified dequantization
                val subBlock = itemInBlock / 32
                val buffer = graphAllocator.buffers[tensorQ4_K.bufferId] ?: throw IllegalStateException("Tensor buffer not found")
                val blockByteOffset = blockIndex * tensorQ4_K.type.byteSize.toInt()
                
                val scaleByteOffset = blockByteOffset + 4 + subBlock
                val scaleByte = buffer[(tensorQ4_K.dataOffset + scaleByteOffset.toULong()).toInt()]
                val quantizedScale = scaleByte.toInt() and 0x3F
                val scale = (quantizedScale.toFloat() / 63.0f) * d
                
                val qsOffset = blockByteOffset + 4 + K_SCALE_SIZE + subBlock * 16 + ((itemInBlock % 32) / 2)
                val qsByte = buffer[(tensorQ4_K.dataOffset + qsOffset.toULong()).toInt()]
                val q = if ((itemInBlock % 32) % 2 == 0) qsByte.toInt() and 0x0F else (qsByte.toInt() shr 4) and 0x0F
                
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
    graphAllocator: GGMLGraphAllocator,
    tensorF32: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ41: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInF32: Int,
    colIndexInQ41: Int,
    commonDimK: Int
): Float {
    require(tensorF32.type == GGMLType.F32) { "computeDotProductF32Q41: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ41.type == GGMLType.Q4_1) { "computeDotProductF32Q41: tensorQ41 must be Q4_1. Got ${tensorQ41.type}" }
    require(tensorF32.ne[0].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ41.ne[1].toInt() == commonDimK) { "tensorQ41 K dim (${tensorQ41.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val f32Value = tensorF32.getFloat(graphAllocator, k, rowIndexInF32)
        // Access tensorQ41[k, colIndexInQ41] - row k, column colIndexInQ41
        val flatIndexInQ41 = k * tensorQ41.ne[0].toInt() + colIndexInQ41
        val blockIndexQ41 = flatIndexInQ41 / QK4_1
        val itemInBlockQ41 = flatIndexInQ41 % QK4_1
        
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
    graphAllocator: GGMLGraphAllocator,
    tensorQ8_K: GGMLTensor,
    tensorF32: GGMLTensor,
    rowIndexInQ8_K: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ8_K.type == GGMLType.Q8_K) { "tensorQ8_K must be Q8_K. Got ${tensorQ8_K.type}" }
    require(tensorF32.type == GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ8_K.ne[0].toInt() == commonDimK) { "tensorQ8_K K dim (${tensorQ8_K.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    
    // Process in blocks of QK_K (simpler for Q8_K)
    for (blockStart in 0 until commonDimK step QK_K) {
        val blockEnd = minOf(blockStart + QK_K, commonDimK)
        
        if (blockEnd - blockStart == QK_K) {
            val flatIndexStart = rowIndexInQ8_K * commonDimK + blockStart
            val blockIndex = flatIndexStart / QK_K
            
            val d = tensorQ8_K.getQ8_KBlockScale(graphAllocator, blockIndex)
            
            // Simple dot product for Q8_K block
            for (i in 0 until QK_K) {
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
                val blockIndex = flatIndex / QK_K
                val itemInBlock = flatIndex % QK_K
                
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
    graphAllocator: GGMLGraphAllocator,
    tensorF32: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ80: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInF32: Int,
    colIndexInQ80: Int,
    commonDimK: Int
): Float {
    require(tensorF32.type == GGMLType.F32) { "computeDotProductF32Q80: tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorQ80.type == GGMLType.Q8_0) { "computeDotProductF32Q80: tensorQ80 must be Q8_0. Got ${tensorQ80.type}" }
    require(tensorF32.ne[0].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ80.ne[1].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val f32Value = tensorF32.getFloat(graphAllocator, k, rowIndexInF32)
        // Access tensorQ80[k, colIndexInQ80] - row k, column colIndexInQ80
        val flatIndexInQ80 = k * tensorQ80.ne[0].toInt() + colIndexInQ80
        val blockIndexQ80 = flatIndexInQ80 / QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % QK8_0
        val scale = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeight = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)
        val dequantizedQ80Value = scale * qWeight.toFloat()
        sumF32 += f32Value * dequantizedQ80Value
    }
    return sumF32
}

/**
 * Computes the direct quantized dot product Q8_0 x Q8_0 -> F32.
 * This avoids dequantization by computing the dot product directly on quantized values.
 */
internal fun computeDotProductQ80Q80(
    graphAllocator: GGMLGraphAllocator,
    tensorQ80A: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ80B: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ80A: Int,
    colIndexInQ80B: Int,
    commonDimK: Int
): Float {
    require(tensorQ80A.type == GGMLType.Q8_0) { "computeDotProductQ80Q80: tensorQ80A must be Q8_0. Got ${tensorQ80A.type}" }
    require(tensorQ80B.type == GGMLType.Q8_0) { "computeDotProductQ80Q80: tensorQ80B must be Q8_0. Got ${tensorQ80B.type}" }
    require(tensorQ80A.ne[0].toInt() == commonDimK) { "tensorQ80A K dim (${tensorQ80A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ80B.ne[1].toInt() == commonDimK) { "tensorQ80B K dim (${tensorQ80B.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ80A[rowIndexInQ80A, k]
        val flatIndexInQ80A = rowIndexInQ80A * commonDimK + k
        val blockIndexQ80A = flatIndexInQ80A / QK8_0
        val itemInBlockQ80A = flatIndexInQ80A % QK8_0
        val scaleA = tensorQ80A.getQ8_0BlockScale(graphAllocator, blockIndexQ80A)
        val qWeightA = tensorQ80A.getQ8_0Weight(graphAllocator, blockIndexQ80A, itemInBlockQ80A)

        // Access tensorQ80B[k, colIndexInQ80B]
        val flatIndexInQ80B = k * tensorQ80B.ne[0].toInt() + colIndexInQ80B
        val blockIndexQ80B = flatIndexInQ80B / QK8_0
        val itemInBlockQ80B = flatIndexInQ80B % QK8_0
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
    graphAllocator: GGMLGraphAllocator,
    tensorQ40A: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ40B: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ40A: Int,
    colIndexInQ40B: Int,
    commonDimK: Int
): Float {
    require(tensorQ40A.type == GGMLType.Q4_0) { "computeDotProductQ40Q40: tensorQ40A must be Q4_0. Got ${tensorQ40A.type}" }
    require(tensorQ40B.type == GGMLType.Q4_0) { "computeDotProductQ40Q40: tensorQ40B must be Q4_0. Got ${tensorQ40B.type}" }
    require(tensorQ40A.ne[0].toInt() == commonDimK) { "tensorQ40A K dim (${tensorQ40A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ40B.ne[1].toInt() == commonDimK) { "tensorQ40B K dim (${tensorQ40B.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ40A[rowIndexInQ40A, k]
        val flatIndexInQ40A = rowIndexInQ40A * commonDimK + k
        val blockIndexQ40A = flatIndexInQ40A / QK4_0
        val itemInBlockQ40A = flatIndexInQ40A % QK4_0
        val scaleA = tensorQ40A.getQ4_0BlockScale(graphAllocator, blockIndexQ40A)
        val qNibbleA = tensorQ40A.getQ4_0NibbleWeight(graphAllocator, blockIndexQ40A, itemInBlockQ40A)

        // Access tensorQ40B[k, colIndexInQ40B]
        val flatIndexInQ40B = k * tensorQ40B.ne[0].toInt() + colIndexInQ40B
        val blockIndexQ40B = flatIndexInQ40B / QK4_0
        val itemInBlockQ40B = flatIndexInQ40B % QK4_0
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
    graphAllocator: GGMLGraphAllocator,
    tensorQ41A: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorQ41B: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ41A: Int,
    colIndexInQ41B: Int,
    commonDimK: Int
): Float {
    require(tensorQ41A.type == GGMLType.Q4_1) { "computeDotProductQ41Q41: tensorQ41A must be Q4_1. Got ${tensorQ41A.type}" }
    require(tensorQ41B.type == GGMLType.Q4_1) { "computeDotProductQ41Q41: tensorQ41B must be Q4_1. Got ${tensorQ41B.type}" }
    require(tensorQ41A.ne[0].toInt() == commonDimK) { "tensorQ41A K dim (${tensorQ41A.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ41B.ne[1].toInt() == commonDimK) { "tensorQ41B K dim (${tensorQ41B.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ41A[rowIndexInQ41A, k]
        val flatIndexInQ41A = rowIndexInQ41A * commonDimK + k
        val blockIndexQ41A = flatIndexInQ41A / QK4_1
        val itemInBlockQ41A = flatIndexInQ41A % QK4_1
        val scaleDA = tensorQ41A.getQ4_1BlockScale(graphAllocator, blockIndexQ41A)
        val minMA = tensorQ41A.getQ4_1BlockMin(graphAllocator, blockIndexQ41A)
        val qNibbleA = tensorQ41A.getQ4_1NibbleWeight(graphAllocator, blockIndexQ41A, itemInBlockQ41A)

        // Access tensorQ41B[k, colIndexInQ41B]
        val flatIndexInQ41B = k * tensorQ41B.ne[0].toInt() + colIndexInQ41B
        val blockIndexQ41B = flatIndexInQ41B / QK4_1
        val itemInBlockQ41B = flatIndexInQ41B % QK4_1
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
    graphAllocator: GGMLGraphAllocator,
    tensorQ80: GGMLTensor,     // M x K (ne[0]=K, ne[1]=M)
    tensorQ40: GGMLTensor,     // K x N (ne[0]=N, ne[1]=K)
    rowIndexInQ80: Int,
    colIndexInQ40: Int,
    commonDimK: Int
): Float {
    require(tensorQ80.type == GGMLType.Q8_0) { "computeDotProductQ80Q40: tensorQ80 must be Q8_0. Got ${tensorQ80.type}" }
    require(tensorQ40.type == GGMLType.Q4_0) { "computeDotProductQ80Q40: tensorQ40 must be Q4_0. Got ${tensorQ40.type}" }
    require(tensorQ80.ne[0].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorQ40.ne[1].toInt() == commonDimK) { "tensorQ40 K dim (${tensorQ40.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorQ80[rowIndexInQ80, k]
        val flatIndexInQ80 = rowIndexInQ80 * commonDimK + k
        val blockIndexQ80 = flatIndexInQ80 / QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % QK8_0
        val scaleQ80 = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeightQ80 = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)

        // Access tensorQ40[k, colIndexInQ40]
        val flatIndexInQ40 = k * tensorQ40.ne[0].toInt() + colIndexInQ40
        val blockIndexQ40 = flatIndexInQ40 / QK4_0
        val itemInBlockQ40 = flatIndexInQ40 % QK4_0
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
    graphAllocator: GGMLGraphAllocator,
    tensorBitNet: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorF32: GGMLTensor,       // K x N (ne[0]=N, ne[1]=K)
    rowIndexInBitNet: Int,       // Row index for tensorBitNet (0 to M-1)
    colIndexInF32: Int,          // Column index for tensorF32 (0 to N-1)
    commonDimK: Int              // The shared dimension K
): Float {
    require(tensorBitNet.type == GGMLType.BITNET_1_58) { "tensorBitNet must be BITNET_1_58. Got ${tensorBitNet.type}" }
    require(tensorF32.type == GGMLType.F32) { "tensorF32 must be F32. Got ${tensorF32.type}" }
    require(tensorBitNet.ne[0].toInt() == commonDimK) { "tensorBitNet K dim (${tensorBitNet.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        val flatIndexInBitNet = rowIndexInBitNet * commonDimK + k
        val blockIndexBitNet = flatIndexInBitNet / QK_BITNET_1_58
        val itemInBlockBitNet = flatIndexInBitNet % QK_BITNET_1_58
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
    graphAllocator: GGMLGraphAllocator,
    tensorBitNetA: GGMLTensor,    // M x K (ne[0]=K, ne[1]=M)
    tensorBitNetB: GGMLTensor,    // K x N (ne[0]=N, ne[1]=K)
    rowIndexInBitNetA: Int,
    colIndexInBitNetB: Int,
    commonDimK: Int
): Float {
    require(tensorBitNetA.type == GGMLType.BITNET_1_58) { "tensorBitNetA must be BITNET_1_58. Got ${tensorBitNetA.type}" }
    require(tensorBitNetB.type == GGMLType.BITNET_1_58) { "tensorBitNetB must be BITNET_1_58. Got ${tensorBitNetB.type}" }
    require(tensorBitNetA.ne[0].toInt() == commonDimK) { "tensorBitNetA K dim (${tensorBitNetA.ne[0]}) must match commonDimK ($commonDimK)" }
    require(tensorBitNetB.ne[1].toInt() == commonDimK) { "tensorBitNetB K dim (${tensorBitNetB.ne[1]}) must match commonDimK ($commonDimK)" }

    var sumF32 = 0.0f
    for (k in 0 until commonDimK) {
        // Access tensorBitNetA[rowIndexInBitNetA, k]
        val flatIndexInBitNetA = rowIndexInBitNetA * commonDimK + k
        val blockIndexBitNetA = flatIndexInBitNetA / QK_BITNET_1_58
        val itemInBlockBitNetA = flatIndexInBitNetA % QK_BITNET_1_58
        val scaleA = tensorBitNetA.getBitNet158BlockScale(graphAllocator, blockIndexBitNetA)
        val ternaryWeightA = tensorBitNetA.getBitNet158TernaryWeight(graphAllocator, blockIndexBitNetA, itemInBlockBitNetA)

        // Access tensorBitNetB[k, colIndexInBitNetB]
        val flatIndexInBitNetB = k * tensorBitNetB.ne[0].toInt() + colIndexInBitNetB
        val blockIndexBitNetB = flatIndexInBitNetB / QK_BITNET_1_58
        val itemInBlockBitNetB = flatIndexInBitNetB % QK_BITNET_1_58
        val scaleB = tensorBitNetB.getBitNet158BlockScale(graphAllocator, blockIndexBitNetB)
        val ternaryWeightB = tensorBitNetB.getBitNet158TernaryWeight(graphAllocator, blockIndexBitNetB, itemInBlockBitNetB)

        // Direct ternary multiplication: (scaleA * ternaryA) * (scaleB * ternaryB)
        sumF32 += scaleA * scaleB * (ternaryWeightA.toFloat() * ternaryWeightB.toFloat())
    }
    return sumF32
}

// Helper to iterate N-dimensionally
internal fun applyNDIter(tensor: GGMLTensor, totalSize: Int, actionPerElement: (flatIdx: Int, indices: IntArray) -> Unit) {
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
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor,
    dst: GGMLTensor
) {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for addition")
    }
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()

    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(a, totalSize) { _, indices -> // Iterate based on 'a' which has same shape as dst
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                dst.setFloat(graphAllocator, v0 + v1, *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(a, totalSize) { _, indices -> // Iterate based on 'a'
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                // Perform addition as Float for precision, then convert back to Half (Short)
                dst.setHalf(graphAllocator, v0 + v1, *indices)
            }
        }
        GGMLType.I32 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                dst.setInt(graphAllocator, valA + valB, *indices)
            }
        }
        GGMLType.I16 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                dst.setShort(graphAllocator, (valA + valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(), *indices)
            }
        }
        GGMLType.I8 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                dst.setByte(graphAllocator, (valA + valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(), *indices)
            }
        }
        GGMLType.I64 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                dst.setLong(graphAllocator, valA + valB, *indices)
            }
    }
    // For quantized types, dequantize, compute, and re-quantize
    GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1,
    GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K -> {
            val aF32 = dequantizeTensor(graphAllocator, a)
            val bF32 = dequantizeTensor(graphAllocator, b)
            val tempF32 = GGMLTensor(type = GGMLType.F32); tempF32.ne = dst.ne.copyOf(); tempF32.nb = calculateContiguousStrides(tempF32.ne, GGMLType.F32, tempF32.ne.size)
            computeAdd(graphAllocator, context, aF32, bF32, tempF32)
            val quantizedResult = quantizeTensor(graphAllocator, tempF32, dst.type)
            // Copy quantized data to destination
            dst.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeAdd not implemented for type ${a.type}")
    }
}

fun computeMul(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor,
    dst: GGMLTensor
) {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for multiplication")
    }
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()

    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                dst.setFloat(graphAllocator, v0 * v1, *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                dst.setHalf(graphAllocator, v0 * v1, *indices)
            }
        }
        GGMLType.I32 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                dst.setInt(graphAllocator, valA * valB, *indices)
            }
        }
        GGMLType.I16 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                dst.setShort(graphAllocator, (valA * valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(), *indices)
            }
        }
        GGMLType.I8 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                dst.setByte(graphAllocator, (valA * valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(), *indices)
            }
        }
        GGMLType.I64 -> {
            applyNDIter(a, totalSize) { _, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                dst.setLong(graphAllocator, valA * valB, *indices)
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1, GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K, GGMLType.BITNET_1_58 -> {
            val aF32 = dequantizeTensor(graphAllocator, a)
            val bF32 = dequantizeTensor(graphAllocator, b)
            val tempF32 = GGMLTensor(type = GGMLType.F32); tempF32.ne = dst.ne.copyOf(); tempF32.nb = calculateContiguousStrides(tempF32.ne, GGMLType.F32, tempF32.ne.size)
            computeMul(graphAllocator, context, aF32, bF32, tempF32)
            val quantizedResult = quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeMul not implemented for type ${a.type}")
    }
}

// Temporary return-style wrappers to ease test migration
fun computeAddRet(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val dst = GGMLTensor(type = a.type)
    dst.ne = a.ne.copyOf(); dst.nb = calculateContiguousStrides(dst.ne, dst.type, dst.rank())
    // Ensure allocation for dst in same buffer as sources if possible
    val alloc = graphAllocator
    val tempGraph = GGMLCGraph(size = 1, nodes = arrayOf(dst), grads = arrayOfNulls(1), leafs = arrayOfNulls(1), allocator = alloc)
    alloc.allocateGraph(tempGraph)
    computeAdd(graphAllocator, context, a, b, dst)
    return dst
}

fun computeMulRet(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val dst = GGMLTensor(type = a.type)
    dst.ne = a.ne.copyOf(); dst.nb = calculateContiguousStrides(dst.ne, dst.type, dst.rank())
    val alloc = graphAllocator
    val tempGraph = GGMLCGraph(size = 1, nodes = arrayOf(dst), grads = arrayOfNulls(1), leafs = arrayOfNulls(1), allocator = alloc)
    alloc.allocateGraph(tempGraph)
    computeMul(graphAllocator, context, a, b, dst)
    return dst
}

fun computeMatMulRet(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val dst = GGMLTensor(type = GGMLType.F32)
    // infer output shape: (a.ne[1] rows, b.ne[0] cols) in our column-major convention (nb[0] stride by element)
    dst.ne[0] = b.ne[0]
    dst.ne[1] = a.ne[1]
    for (i in 2 until GGML_MAX_DIMS) dst.ne[i] = 1
    dst.nb = calculateContiguousStrides(dst.ne, dst.type, dst.rank())
    val tempGraph = GGMLCGraph(size = 1, nodes = arrayOf(dst), grads = arrayOfNulls(1), leafs = arrayOfNulls(1), allocator = graphAllocator)
    graphAllocator.allocateGraph(tempGraph)
    computeMatMul(graphAllocator, context, a, b, dst)
    return dst
}

fun dequantizeTensor(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne = tensor.ne.copyOf()
    if (result.type.byteSize > 0uL) {
        result.nb[0] = result.type.byteSize
        for (d in 1 until GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }
    } else {
        for(d in 0 until GGML_MAX_DIMS) result.nb[d] = 0uL
    }
    val numElements = tensor.numElements().toInt()
    val resultDataArray = FloatArray(numElements)
    when (tensor.type) {
        GGMLType.F16 -> applyNDIter(tensor, numElements) { flatIdx, indices -> if (flatIdx < numElements) resultDataArray[flatIdx] = tensor.getHalf(graphAllocator, *indices) }
        GGMLType.F32 -> applyNDIter(tensor, numElements) { flatIdx, indices -> if (flatIdx < numElements) resultDataArray[flatIdx] = tensor.getFloat(graphAllocator, *indices) }
        GGMLType.Q8_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ8_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK8_0) { if (fidx < numElements) resultDataArray[fidx++] = scale * tensor.getQ8_0Weight(graphAllocator, blockIdx, itemIdxInBlock).toFloat() else break }
                if (fidx >= numElements && blockIdx < numBlocks -1) { println("Warn: Q8_0 dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q8_0 dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q4_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ4_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK4_0) { if (fidx < numElements) resultDataArray[fidx++] = scale * (tensor.getQ4_0NibbleWeight(graphAllocator, blockIdx, itemIdxInBlock).toFloat() - 8.0f) else break }
                if (fidx >= numElements && blockIdx < numBlocks -1) { println("Warn: Q4_0 dequant filled array early for ${tensor.name}"); break }
            }
             if (fidx != numElements && numElements > 0) println("Warn: Q4_0 dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q4_1 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val dScale = tensor.getQ4_1BlockScale(graphAllocator, blockIdx)
                val mMin = tensor.getQ4_1BlockMin(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK4_1) {
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
        GGMLType.Q2_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                dequantizeQ2_KBlock(graphAllocator, tensor, blockIdx, resultDataArray, fidx)
                fidx += QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q2_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q2_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q3_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                dequantizeQ3_KBlock(graphAllocator, tensor, blockIdx, resultDataArray, fidx)
                fidx += QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q3_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q3_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q4_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                dequantizeQ4_KBlock(graphAllocator, tensor, blockIdx, resultDataArray, fidx)
                fidx += QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q4_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q4_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q5_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                dequantizeQ5_KBlock(graphAllocator, tensor, blockIdx, resultDataArray, fidx)
                fidx += QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q5_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q5_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q6_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                dequantizeQ6_KBlock(graphAllocator, tensor, blockIdx, resultDataArray, fidx)
                fidx += QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q6_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q6_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        GGMLType.Q8_K -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                dequantizeQ8_KBlock(graphAllocator, tensor, blockIdx, resultDataArray, fidx)
                fidx += QK_K
                if (fidx >= numElements && blockIdx < numBlocks - 1) { println("Warn: Q8_K dequant filled array early for ${tensor.name}"); break }
            }
            if (fidx != numElements && numElements > 0) println("Warn: Q8_K dequant element count mismatch for ${tensor.name}: $fidx vs $numElements")
        }
        // BitNet 1.58 dequantization
        GGMLType.BITNET_1_58 -> {
            val numBlocks = tensor.getNumBlocks().toInt(); var fidx = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getBitNet158BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK_BITNET_1_58) { 
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

fun quantizeTensor(graphAllocator: GGMLGraphAllocator, tensorF32: GGMLTensor, targetType: GGMLType): GGMLTensor {
    if (tensorF32.type != GGMLType.F32) throw IllegalArgumentException("quantizeTensor expects F32 input, got ${tensorF32.type}")
    val result = GGMLTensor(type = targetType); result.ne = tensorF32.ne.copyOf()
    if (targetType.byteSize > 0uL) {
        result.nb[0] = targetType.byteSize
        for (d in 1 until GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }
    } else {
        if (targetType != GGMLType.COUNT && !targetType.description.startsWith("Q", ignoreCase = true)) println("Warn: Stride calc for ${targetType.name} in quantizeTensor may be incomplete.")
        for(d in 0 until GGML_MAX_DIMS) result.nb[d] = 0uL
    }
    val numElements = tensorF32.numElements().toInt()
    when (targetType) {
        GGMLType.F16 -> {
            val resArr = ShortArray(numElements); applyNDIter(tensorF32, numElements) { fid, ind -> if (fid < numElements) resArr[fid] = floatToHalf(tensorF32.getFloat(graphAllocator, *ind)) }; result.data = resArr
        }
        GGMLType.F32 -> {
            val resArr = FloatArray(numElements); applyNDIter(tensorF32, numElements) { fid, ind -> if (fid < numElements) resArr[fid] = tensorF32.getFloat(graphAllocator, *ind) }; result.data = resArr
        }
        GGMLType.Q8_0 -> {
            require(numElements % QK8_0 == 0) { "Q8_0 numElements $numElements not div by $QK8_0" }
            val nBlk = numElements / QK8_0; val blkSize = targetType.byteSize.toInt(); require(blkSize == SHORT_SIZE_BYTES + QK8_0) { "Q8_0 block size mismatch" }
            val resArr = ByteArray(nBlk * blkSize); val f32Blk = FloatArray(QK8_0); var curIdx = 0; var boff = 0
            applyNDIter(tensorF32, numElements) { _, ind ->
                val itemInBlk = curIdx % QK8_0; f32Blk[itemInBlk] = tensorF32.getFloat(graphAllocator, *ind)
                if (itemInBlk == QK8_0 - 1) {
                    var amax = 0.0f; for (v in f32Blk) amax = maxOf(amax, abs(v)); val scale = if (amax == 0.0f) 1.0f else amax / 127.0f; val invS = 1.0f / scale
                    resArr.setShortLe(boff, floatToHalf(scale)); val qOff = boff + SHORT_SIZE_BYTES
                    for (k in 0 until QK8_0) resArr[qOff + k] = round(f32Blk[k] * invS).toInt().coerceIn(-128, 127).toByte()
                    boff += blkSize
                }; curIdx++
            }; result.data = resArr
        }
        GGMLType.Q4_0 -> {
            require(numElements % QK4_0 == 0) { "Q4_0 numElements $numElements not div by $QK4_0" }
            val nBlk = numElements / QK4_0; val blkSize = targetType.byteSize.toInt(); require(blkSize == SHORT_SIZE_BYTES + QK4_0 / 2) { "Q4_0 block size mismatch" }
            val resArr = ByteArray(nBlk * blkSize); val f32Blk = FloatArray(QK4_0); var curIdx = 0; var boff = 0
            applyNDIter(tensorF32, numElements) { _, ind ->
                val itemInBlk = curIdx % QK4_0; f32Blk[itemInBlk] = tensorF32.getFloat(graphAllocator, *ind)
                if (itemInBlk == QK4_0 - 1) {
                    var amax = 0.0f; for (v in f32Blk) amax = maxOf(amax, abs(v)); val scale = if (amax == 0.0f) 1.0f else amax / 8.0f; val invS = if (scale == 0.0f) 0.0f else 1.0f / scale
                    resArr.setShortLe(boff, floatToHalf(scale)); val qOff = boff + SHORT_SIZE_BYTES
                    for (j in 0 until QK4_0 / 2) {
                        val q1 = round(f32Blk[j*2] * invS + 8.0f).toInt().coerceIn(0,15); val q2 = round(f32Blk[j*2+1] * invS + 8.0f).toInt().coerceIn(0,15)
                        resArr[qOff + j] = ((q1 and 0x0F) or ((q2 and 0x0F) shl 4)).toByte()
                    }; boff += blkSize
                }; curIdx++
            }; result.data = resArr
        }
        GGMLType.Q8_1 -> { result.data = ByteArray(numElements); println("Warn: Quant F32 to ${targetType.name} NI") }
        GGMLType.Q4_1 -> {
            require(tensorF32.type == GGMLType.F32) { "Input tensor for Q4_1 quantization must be F32. Got ${tensorF32.type}" } // Already checked at function start, but good for clarity
            require(numElements % QK4_1 == 0) { "For Q4_1 quantization, total elements ($numElements) must be divisible by QK4_1 ($QK4_1)" }

            val numBlocks = numElements / QK4_1
            val q4_1BlockByteSize = targetType.byteSize.toInt()
            val expectedBlockSize = (2 * SHORT_SIZE_BYTES) + (QK4_1 / 2)
            require(q4_1BlockByteSize == expectedBlockSize) { "Q4_1 block byte size mismatch. Expected $expectedBlockSize, got $q4_1BlockByteSize. Type says: ${targetType.byteSize}" }

            val q4_1DataArray = ByteArray(numBlocks * q4_1BlockByteSize)
            result.data = q4_1DataArray // Assign the data array to the result tensor prepared at the start of the function

            val f32BlockValues = FloatArray(QK4_1)
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
                applyNDIter(tensorF32, numElements) { flatIdx, indices ->
                    if (flatIdx >= blockNum * QK4_1 && flatIdx < (blockNum + 1) * QK4_1) {
                        if (f32BlockReadCount < QK4_1) { // Ensure we don't write out of bounds
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

                for (i in 0 until QK4_1) {
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
                    // }
                    // result.data = resArr
                    // This is the pattern to follow.
                }
            }
            // --- Re-writing the loop structure based on Q8_0/Q4_0 pattern ---
            val f32BlockBuffer = FloatArray(QK4_1) // Temporary buffer for one block of F32 values
            var currentElementInF32 = 0
            var byteArrayWriteOffset = 0

            applyNDIter(tensorF32, numElements) { _, indices ->
                val itemInBlockIndex = currentElementInF32 % QK4_1
                f32BlockBuffer[itemInBlockIndex] = tensorF32.getFloat(graphAllocator, *indices)

                if (itemInBlockIndex == QK4_1 - 1) { // Block is full, process it
                    val f_min = f32BlockBuffer.minOrNull() ?: 0.0f
                    val f_max = f32BlockBuffer.maxOrNull() ?: 0.0f

                    var d_scaleF32 = (f_max - f_min) / 15.0f
                    if (d_scaleF32 == 0.0f) { // Handles case where f_max == f_min
                        d_scaleF32 = 1.0f
                    }
                    val m_minF32 = f_min
                    // invDScaleF32 is guaranteed to be valid as d_scaleF32 is non-zero.
                    val invDScaleF32 = 1.0f / d_scaleF32

                    val d_scaleF16Short = floatToHalf(d_scaleF32)
                    val m_minF16Short = floatToHalf(m_minF32)

                    q4_1DataArray.setShortLe(byteArrayWriteOffset, d_scaleF16Short)
                    q4_1DataArray.setShortLe(byteArrayWriteOffset + SHORT_SIZE_BYTES, m_minF16Short)

                    val qsDataWriteStartOffsetInBlock = byteArrayWriteOffset + (2 * SHORT_SIZE_BYTES)
                    for (j in 0 until QK4_1 / 2) {
                        val f32Val1 = f32BlockBuffer[j * 2]
                        val f32Val2 = f32BlockBuffer[j * 2 + 1]

                        val quantVal1 = round((f32Val1 - m_minF32) * invDScaleF32).toInt().coerceIn(0, 15)
                        val quantVal2 = round((f32Val2 - m_minF32) * invDScaleF32).toInt().coerceIn(0, 15)

                        val packedByte = (quantVal1 and 0x0F) or ((quantVal2 and 0x0F) shl 4)
                        q4_1DataArray[qsDataWriteStartOffsetInBlock + j] = packedByte.toByte()
                    }
                    byteArrayWriteOffset += q4_1BlockByteSize
                }
                currentElementInF32++
            }
            result.data = q4_1DataArray
        }
        // K-Quant quantization implementations
        GGMLType.Q2_K -> {
            require(numElements % QK_K == 0) { "Q2_K numElements $numElements not div by $QK_K" }
            val numBlocks = numElements / QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                // Gather QK_K elements for this block
                val blockValues = FloatArray(QK_K)
                for (i in 0 until QK_K) {
                    // Convert flat index to multidimensional indices
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(GGML_MAX_DIMS)
                    for (dim in 0 until GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }
                
                quantizeQ2_KBlock(blockValues, resArr, blockNum * blockByteSize)
            }
        }
        GGMLType.Q3_K -> {
            require(numElements % QK_K == 0) { "Q3_K numElements $numElements not div by $QK_K" }
            val numBlocks = numElements / QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(QK_K)
                for (i in 0 until QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(GGML_MAX_DIMS)
                    for (dim in 0 until GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }
                
                quantizeQ3_KBlock(blockValues, resArr, blockNum * blockByteSize)
            }
        }
        GGMLType.Q4_K -> {
            require(numElements % QK_K == 0) { "Q4_K numElements $numElements not div by $QK_K" }
            val numBlocks = numElements / QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(QK_K)
                for (i in 0 until QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(GGML_MAX_DIMS)
                    for (dim in 0 until GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }
                
                quantizeQ4_KBlock(blockValues, resArr, blockNum * blockByteSize)
            }
        }
        GGMLType.Q5_K -> {
            require(numElements % QK_K == 0) { "Q5_K numElements $numElements not div by $QK_K" }
            val numBlocks = numElements / QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(QK_K)
                for (i in 0 until QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(GGML_MAX_DIMS)
                    for (dim in 0 until GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }
                
                quantizeQ5_KBlock(blockValues, resArr, blockNum * blockByteSize)
            }
        }
        GGMLType.Q6_K -> {
            require(numElements % QK_K == 0) { "Q6_K numElements $numElements not div by $QK_K" }
            val numBlocks = numElements / QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(QK_K)
                for (i in 0 until QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(GGML_MAX_DIMS)
                    for (dim in 0 until GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }
                
                quantizeQ6_KBlock(blockValues, resArr, blockNum * blockByteSize)
            }
        }
        GGMLType.Q8_K -> {
            require(numElements % QK_K == 0) { "Q8_K numElements $numElements not div by $QK_K" }
            val numBlocks = numElements / QK_K
            val blockByteSize = targetType.byteSize.toInt()
            val resArr = ByteArray(numBlocks * blockByteSize)
            result.data = resArr
            
            var currentElementIndex = 0
            for (blockNum in 0 until numBlocks) {
                val blockValues = FloatArray(QK_K)
                for (i in 0 until QK_K) {
                    val flatIdx = currentElementIndex++
                    var tempIdx = flatIdx.toLong()
                    val indices = IntArray(GGML_MAX_DIMS)
                    for (dim in 0 until GGML_MAX_DIMS) {
                        if (tensorF32.ne[dim] > 0) {
                            indices[dim] = (tempIdx % tensorF32.ne[dim]).toInt()
                            tempIdx /= tensorF32.ne[dim]
                        }
                    }
                    blockValues[i] = tensorF32.getFloat(graphAllocator, *indices)
                }
                
                quantizeQ8_KBlock(blockValues, resArr, blockNum * blockByteSize)
            }
        }
        // BitNet 1.58 quantization
        GGMLType.BITNET_1_58 -> {
            require(numElements % QK_BITNET_1_58 == 0) { "BitNet 1.58 numElements $numElements not div by $QK_BITNET_1_58" }
            val numBlocks = numElements / QK_BITNET_1_58
            val blockByteSize = targetType.byteSize.toInt()
            require(blockByteSize == Short.SIZE_BYTES + 8) { "BitNet 1.58 block size mismatch. Expected ${Short.SIZE_BYTES + 8}, got $blockByteSize" }
            
            val bitNet158DataArray = ByteArray(numBlocks * blockByteSize)
            result.data = bitNet158DataArray
            
            val f32BlockValues = FloatArray(QK_BITNET_1_58)
            var currentF32ElementIndex = 0
            var blockByteWriteOffset = 0
            
            // Process each block
            for (blockNum in 0 until numBlocks) {
                // Fill f32BlockValues for this block
                applyNDIter(tensorF32, QK_BITNET_1_58) { idx, ind ->
                    if (currentF32ElementIndex < numElements && idx < QK_BITNET_1_58) {
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
                bitNet158DataArray.setShortLe(blockByteWriteOffset, floatToHalf(scale))
                
                // Quantize values to ternary (-1, 0, +1) and pack them
                val ternaryValues = IntArray(QK_BITNET_1_58)
                for (i in 0 until QK_BITNET_1_58) {
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
                        if (valueIdx < QK_BITNET_1_58) {
                            packedValue += ternaryValues[valueIdx] * powers[posInGroup]
                        }
                        // If valueIdx >= QK_BITNET_1_58, the value remains 0 (encoded as -1), which is fine
                    }
                    bitNet158DataArray[ternaryDataOffset + groupIdx] = packedValue.toByte()
                }
                
                blockByteWriteOffset += blockByteSize
            }
        }
        GGMLType.Q5_0, GGMLType.Q5_1 -> { result.data = ByteArray((numElements * 5 + 7) / 8); println("Warn: Quant F32 to ${targetType.name} NI") }
        else -> { println("Error: Unsupp target quant type $targetType"); result.data = null }
    }
    return result
}

fun computeMatMul(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor, dst: GGMLTensor) {
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

    if (a.type == GGMLType.Q4_0 && b.type == GGMLType.F32) {
        if (dst.type != GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_0 x F32 matmul")
        
        // Write results directly into destination tensor using graph allocator
        var flatIdx = 0
        for(i in 0 until M){ 
            for(j in 0 until N){ 
                val result = computeDotProductQ40F32(graphAllocator,a,b,i,j,K)
                dst.setFloat(graphAllocator, result, j, i) // Column j, row i
                flatIdx++
            } 
        }
        return
    }
    if (a.type == GGMLType.Q4_1 && b.type == GGMLType.F32) {
        if (dst.type != GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_1 x F32 matmul")

        // Write results directly into destination tensor
        for (i in 0 until M) { // Iterate output rows (M)
            for (j in 0 until N) { // Iterate output columns (N)
                val dotProduct = computeDotProductQ41F32(
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
    if (a.type == GGMLType.Q2_K && b.type == GGMLType.F32) {
        if (dst.type != GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q2_K x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = computeDotProductQ2_KF32(graphAllocator, a, b, i, j, K)
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    if (a.type == GGMLType.Q4_K && b.type == GGMLType.F32) {
        if (dst.type != GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q4_K x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = computeDotProductQ4_KF32(graphAllocator, a, b, i, j, K)
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    if (a.type == GGMLType.Q8_K && b.type == GGMLType.F32) {
        if (dst.type != GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q8_K x F32 matmul")
        for (i in 0 until M) {
            for (j in 0 until N) {
                val dot = computeDotProductQ8_KF32(graphAllocator, a, b, i, j, K)
                dst.setFloat(graphAllocator, dot, j, i)
            }
        }
        return
    }
    
    if (a.type == GGMLType.Q8_0 && b.type == GGMLType.F32) {
        if (dst.type != GGMLType.F32) throw IllegalArgumentException("Result tensor type must be F32 for Q8_0 x F32 matmul")
        
        // Write results directly into destination tensor
        for(i in 0 until M){ 
            for(j in 0 until N){ 
                val result = computeDotProductQ80F32(graphAllocator,a,b,i,j,K)
                dst.setFloat(graphAllocator, result, j, i) // Column j, row i
            } 
        }
        return
    }

    // General matrix multiplication fallback (dst-arg only)
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match first input type for general matmul")
    when (a.type) {
        GGMLType.F32 -> {
            val effA=a; val effB=if(b.type==GGMLType.F32)b else dequantizeTensor(graphAllocator,b)
            for(i in 0 until M){
                for(j in 0 until N){
                    var sum=0.0f
                    for(l in 0 until K){
                        sum+=effA.getFloat(graphAllocator,l,i)*effB.getFloat(graphAllocator,j,l)
                    }
                    dst.setFloat(graphAllocator, sum, j, i) // Column j, row i
                }
            }
        }
        GGMLType.F16 -> {
            val effA=a; val effB=if(b.type==GGMLType.F16)b else dequantizeTensor(graphAllocator,b)
            if(effB.type!=GGMLType.F16) throw NotImplementedError("F16xnon-F16 matmul to F16 not implemented")
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
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1,GGMLType.Q2_K,GGMLType.Q3_K,GGMLType.Q4_K,GGMLType.Q5_K,GGMLType.Q6_K,GGMLType.Q8_K,GGMLType.BITNET_1_58 -> {
            val aF32=dequantizeTensor(graphAllocator,a); val bF32=dequantizeTensor(graphAllocator,b)
            val tempF32 = GGMLTensor(type = GGMLType.F32); tempF32.ne = dst.ne.copyOf(); tempF32.nb = calculateContiguousStrides(tempF32.ne, GGMLType.F32, tempF32.ne.size)
            computeMatMul(graphAllocator,context,aF32,bF32,tempF32)
            val qRes=quantizeTensor(graphAllocator,tempF32,dst.type); dst.data=qRes.data
        }
        else -> throw NotImplementedError("computeMatMul not implemented for input type ${a.type}")
    }
}

fun computeRelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, dst: GGMLTensor) {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> applyNDIter(dst, totalSize) { _, ind -> dst.setFloat(graphAllocator, if(a.getFloat(graphAllocator, *ind)>0.0f)a.getFloat(graphAllocator,*ind)else 0.0f, *ind) }
        GGMLType.F16 -> applyNDIter(dst, totalSize) { _, ind -> dst.setHalf(graphAllocator, if(a.getHalf(graphAllocator, *ind)>0.0f)a.getHalf(graphAllocator,*ind)else 0.0f, *ind) }
        else -> throw NotImplementedError("computeRelu not implemented for type ${a.type}")
    }
}

fun computeGelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, dst: GGMLTensor) {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val totalSize = dst.numElements().toInt()
    val gelu = {x:Float -> x*0.5f*(1.0f+kotlin.math.tanh(0.797885f*(x+0.044715f*x*x*x)))}
    when (a.type) {
        GGMLType.F32 -> applyNDIter(dst, totalSize) { _, ind -> dst.setFloat(graphAllocator, gelu(a.getFloat(graphAllocator, *ind)), *ind) }
        GGMLType.F16 -> applyNDIter(dst, totalSize) { _, ind -> dst.setHalf(graphAllocator, gelu(a.getHalf(graphAllocator, *ind)), *ind) }
        else -> throw NotImplementedError("computeGelu not implemented for type ${a.type}")
    }
}

fun computeSub(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor, dst: GGMLTensor) {
    for(i in 0 until GGML_MAX_DIMS){if(a.ne[i]!=b.ne[i])throw IllegalArgumentException("Dims mismatch")}
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val ts=dst.numElements().toInt()
    when(a.type){
        GGMLType.F32-> {
            applyNDIter(a,ts){_,ind->dst.setFloat(graphAllocator, a.getFloat(graphAllocator,*ind)-b.getFloat(graphAllocator,*ind), *ind)}
        }
        GGMLType.F16-> {
            applyNDIter(a,ts){_,ind->dst.setHalf(graphAllocator, a.getHalf(graphAllocator,*ind)-b.getHalf(graphAllocator,*ind), *ind)}
        }
        GGMLType.I32 -> {
            applyNDIter(a, ts) { _, indices ->
                dst.setInt(graphAllocator, a.getInt(graphAllocator, *indices) - b.getInt(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.I16 -> {
            applyNDIter(a, ts) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                dst.setShort(graphAllocator, (valA - valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(), *indices)
            }
        }
        GGMLType.I8 -> {
            applyNDIter(a, ts) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                dst.setByte(graphAllocator, (valA - valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(), *indices)
            }
        }
        GGMLType.I64 -> {
            applyNDIter(a, ts) { _, indices ->
                dst.setLong(graphAllocator, a.getLong(graphAllocator, *indices) - b.getLong(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1,
        GGMLType.Q2_K,GGMLType.Q3_K,GGMLType.Q4_K,GGMLType.Q5_K,GGMLType.Q6_K,GGMLType.Q8_K -> {
            val af = dequantizeTensor(graphAllocator, a)
            val bf = dequantizeTensor(graphAllocator, b)
            val tempF32 = GGMLTensor(type = GGMLType.F32)
            tempF32.ne = dst.ne.copyOf()
            tempF32.nb = calculateContiguousStrides(tempF32.ne, GGMLType.F32, tempF32.ne.size)
            computeSub(graphAllocator, context, af, bf, tempF32)
            val qr = quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = qr.data
        }
        else -> throw NotImplementedError("computeSub not implemented for type ${a.type}")
    }
}

fun computeNeg(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, dst: GGMLTensor) {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val ts=dst.numElements().toInt()
    when(a.type){
        GGMLType.F32 -> applyNDIter(dst,ts){_,ind->dst.setFloat(graphAllocator,-a.getFloat(graphAllocator,*ind),*ind)}
        GGMLType.F16 -> applyNDIter(dst,ts){_,ind->dst.setHalf(graphAllocator,-a.getHalf(graphAllocator,*ind),*ind)}
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1,
        GGMLType.Q2_K,GGMLType.Q3_K,GGMLType.Q4_K,GGMLType.Q5_K,GGMLType.Q6_K,GGMLType.Q8_K -> {
            val af = dequantizeTensor(graphAllocator, a)
            val tempF32 = GGMLTensor(type = GGMLType.F32)
            tempF32.ne = dst.ne.copyOf(); tempF32.nb = calculateContiguousStrides(tempF32.ne, GGMLType.F32, tempF32.ne.size)
            computeNeg(graphAllocator, context, af, tempF32)
            val qr = quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = qr.data
        }
        else -> throw NotImplementedError("computeNeg not implemented for ${a.type}")
    }
}

fun computeDiv(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor, dst: GGMLTensor) {
    for(i in 0 until GGML_MAX_DIMS){if(a.ne[i]!=b.ne[i])throw IllegalArgumentException("Dims mismatch")}
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != dst.ne[i]) throw IllegalArgumentException("Result tensor dimensions must match input dimensions")
    }
    if (dst.type != a.type) throw IllegalArgumentException("Result tensor type must match input type")
    
    val ts=dst.numElements().toInt()
    val div={vA:Float,vB:Float->if(vB==0.0f){if(vA==0.0f)Float.NaN else if(vA>0.0f)Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY}else{vA/vB}}
    when(a.type){
        GGMLType.F32-> {
            applyNDIter(a,ts){_,ind->dst.setFloat(graphAllocator, div(a.getFloat(graphAllocator,*ind),b.getFloat(graphAllocator,*ind)), *ind)}
        }
        GGMLType.F16-> {
            applyNDIter(a,ts){_,ind->dst.setHalf(graphAllocator, div(a.getHalf(graphAllocator,*ind),b.getHalf(graphAllocator,*ind)), *ind)}
        }
        GGMLType.I32 -> {
            applyNDIter(a, ts) { _, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                if (valB == 0) throw ArithmeticException("Division by zero for I32")
                dst.setInt(graphAllocator, valA / valB, *indices)
            }
        }
        GGMLType.I16 -> {
            applyNDIter(a, ts) { _, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                if (valB == 0) throw ArithmeticException("Division by zero for I16")
                dst.setShort(graphAllocator, (valA / valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort(), *indices)
            }
        }
        GGMLType.I8 -> {
            applyNDIter(a, ts) { _, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                if (valB == 0) throw ArithmeticException("Division by zero for I8")
                dst.setByte(graphAllocator, (valA / valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte(), *indices)
            }
        }
        GGMLType.I64 -> {
            applyNDIter(a, ts) { _, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                if (valB == 0L) throw ArithmeticException("Division by zero for I64")
                dst.setLong(graphAllocator, valA / valB, *indices)
            }
        }
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1,
        GGMLType.Q2_K,GGMLType.Q3_K,GGMLType.Q4_K,GGMLType.Q5_K,GGMLType.Q6_K,GGMLType.Q8_K -> {
            val af = dequantizeTensor(graphAllocator, a); val bf = dequantizeTensor(graphAllocator, b)
            val tempF32 = GGMLTensor(type = GGMLType.F32)
            tempF32.ne = dst.ne.copyOf(); tempF32.nb = calculateContiguousStrides(tempF32.ne, GGMLType.F32, tempF32.ne.size)
            computeDiv(graphAllocator, context, af, bf, tempF32)
            val qr = quantizeTensor(graphAllocator, tempF32, dst.type)
            dst.data = qr.data
        }
        else -> throw NotImplementedError("computeDiv not implemented for type ${a.type}")
    }
}

// K-Quant Block Quantization Functions

/**
 * Quantizes a block of QK_K float values to Q2_K format.
 * Q2_K structure: scales[QK_K/16], qs[QK_K/4], d (F16), dmin (F16)
 * Effectively 2.625 bits per weight
 */
private fun quantizeQ2_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == QK_K) { "Q2_K block must have $QK_K values" }
    
    // Find overall min/max for the block
    var minVal = Float.POSITIVE_INFINITY
    var maxVal = Float.NEGATIVE_INFINITY
    for (value in blockValues) {
        minVal = minOf(minVal, value)
        maxVal = maxOf(maxVal, value)
    }
    
    // Calculate super-block scales
    val range = maxVal - minVal
    val d = if (range > 0.0f) range / 3.0f else 1.0f  // scale for quantized scales 
    val dmin = minVal // scale for quantized mins
    
    // Write super-block scales (d and dmin) at the end of the block
    val dOffset = destOffset + QK_K/16 + QK_K/4
    dest.setShortLe(dOffset, floatToHalf(d))
    dest.setShortLe(dOffset + 2, floatToHalf(dmin))
    
    // Quantize in 16-element sub-blocks
    for (subBlock in 0 until QK_K/16) {
        val subBlockStart = subBlock * 16
        val subBlockEnd = subBlockStart + 16
        
        // Find min/max for this sub-block
        var subMin = Float.POSITIVE_INFINITY
        var subMax = Float.NEGATIVE_INFINITY
        for (i in subBlockStart until subBlockEnd) {
            subMin = minOf(subMin, blockValues[i])
            subMax = maxOf(subMax, blockValues[i])
        }
        
        // Calculate and store quantized scale and min for this sub-block
        val subRange = subMax - subMin
        val scale = if (subRange > 0.0f) subRange / 3.0f else 1.0f
        val quantizedScale = round((scale / d) * 15.0f).toInt().coerceIn(0, 15)
        val quantizedMin = round((subMin - dmin) / d).toInt().coerceIn(0, 15)
        
        // Pack scale and min into one byte (4 bits each)
        val scaleAndMin = (quantizedScale and 0x0F) or ((quantizedMin and 0x0F) shl 4)
        dest[destOffset + subBlock] = scaleAndMin.toByte()
        
        // Quantize the 16 values in this sub-block to 2 bits each (4 values per byte)
        for (i in 0 until 16 step 4) {
            val globalIdx = subBlockStart + i
            var packedByte = 0
            for (j in 0 until 4) {
                if (globalIdx + j < blockValues.size) {
                    val value = blockValues[globalIdx + j]
                    val quantizedValue = if (subRange > 0.0f) {
                        round(((value - subMin) / subRange) * 3.0f).toInt().coerceIn(0, 3)
                    } else 0
                    packedByte = packedByte or ((quantizedValue and 0x03) shl (j * 2))
                }
            }
            dest[destOffset + QK_K/16 + (subBlock * 4) + (i / 4)] = packedByte.toByte()
        }
    }
}

/**
 * Quantizes a block of QK_K float values to Q3_K format.
 * Q3_K structure: hmask[QK_K/8], qs[QK_K/4], scales[12], d (F16)
 * Effectively 3.4375 bits per weight
 */
private fun quantizeQ3_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == QK_K) { "Q3_K block must have $QK_K values" }
    
    // Find absolute maximum for super-block scale
    var amax = 0.0f
    for (value in blockValues) {
        amax = maxOf(amax, abs(value))
    }
    
    val d = if (amax > 0.0f) amax / 127.0f else 1.0f
    val invD = if (d > 0.0f) 1.0f / d else 0.0f
    
    // Write super-block scale d at the end
    val dOffset = destOffset + QK_K/8 + QK_K/4 + 12
    dest.setShortLe(dOffset, floatToHalf(d))
    
    // Process in 16-element sub-blocks
    for (subBlock in 0 until QK_K/16) {
        val subBlockStart = subBlock * 16
        
        // Find max absolute value in this sub-block
        var subAmax = 0.0f
        for (i in 0 until 16) {
            subAmax = maxOf(subAmax, abs(blockValues[subBlockStart + i]))
        }
        
        // Calculate and store quantized scale for this sub-block
        val scale = if (subAmax > 0.0f) subAmax / 7.0f else 1.0f
        val quantizedScale = round((scale / d) * 63.0f).toInt().coerceIn(0, 63)
        dest[destOffset + QK_K/8 + QK_K/4 + subBlock] = quantizedScale.toByte()
        
        // Quantize values in this sub-block
        val invScale = if (scale > 0.0f) 1.0f / scale else 0.0f
        for (i in 0 until 16) {
            val value = blockValues[subBlockStart + i]
            val quantizedValue = round(value * invScale).toInt().coerceIn(-7, 7)
            
            // Pack 3-bit values: 2 bits in qs, 1 bit in hmask
            val byteIdx = (subBlockStart + i) / 4
            val bitPos = ((subBlockStart + i) % 4) * 2
            val maskByteIdx = (subBlockStart + i) / 8
            val maskBitPos = (subBlockStart + i) % 8
            
            // Store low 2 bits in qs
            val lowBits = quantizedValue and 0x03
            dest[destOffset + QK_K/8 + byteIdx] = (dest[destOffset + QK_K/8 + byteIdx].toInt() or (lowBits shl bitPos)).toByte()
            
            // Store high bit in hmask
            val highBit = (quantizedValue shr 2) and 0x01
            dest[destOffset + maskByteIdx] = (dest[destOffset + maskByteIdx].toInt() or (highBit shl maskBitPos)).toByte()
        }
    }
}

/**
 * Quantizes a block of QK_K float values to Q4_K format.
 * Q4_K structure: d (F16), dmin (F16), scales[K_SCALE_SIZE], qs[QK_K/2]
 * Effectively 4.5 bits per weight
 */
private fun quantizeQ4_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == QK_K) { "Q4_K block must have $QK_K values" }
    
    // Find min and max for the entire block
    var minVal = Float.POSITIVE_INFINITY
    var maxVal = Float.NEGATIVE_INFINITY
    for (value in blockValues) {
        minVal = minOf(minVal, value)
        maxVal = maxOf(maxVal, value)
    }
    
    val range = maxVal - minVal
    val d = if (range > 0.0f) range / 255.0f else 1.0f
    val dmin = minVal
    val invD = if (d > 0.0f) 1.0f / d else 0.0f
    
    // Write super-block scales
    dest.setShortLe(destOffset, floatToHalf(d))
    dest.setShortLe(destOffset + 2, floatToHalf(dmin))
    
    // Process in 32-element sub-blocks (8 sub-blocks total)
    for (subBlock in 0 until 8) {
        val subBlockStart = subBlock * 32
        
        // Find min/max for this sub-block
        var subMin = Float.POSITIVE_INFINITY  
        var subMax = Float.NEGATIVE_INFINITY
        for (i in 0 until 32) {
            val idx = subBlockStart + i
            if (idx < blockValues.size) {
                subMin = minOf(subMin, blockValues[idx])
                subMax = maxOf(subMax, blockValues[idx])
            }
        }
        
        // Calculate sub-block scale and min
        val subRange = subMax - subMin
        val scale = if (subRange > 0.0f) subRange / 15.0f else 1.0f
        val quantizedScale = round((scale / d) * 63.0f).toInt().coerceIn(0, 63)
        val quantizedMin = round((subMin - dmin) / d).toInt().coerceIn(0, 63)
        
        // Store quantized scale and min (6 bits each, packed into 12 bits)
        if (subBlock < K_SCALE_SIZE) {
            dest[destOffset + 4 + subBlock] = ((quantizedScale and 0x3F) or ((quantizedMin and 0x03) shl 6)).toByte()
            if (subBlock * 2 + 1 < K_SCALE_SIZE) {
                dest[destOffset + 4 + subBlock * 2 + 1] = ((quantizedMin shr 2) and 0x0F).toByte()
            }
        }
        
        // Quantize and pack the 32 values (2 values per byte)
        val invScale = if (scale > 0.0f) 1.0f / scale else 0.0f
        for (i in 0 until 32 step 2) {
            val idx1 = subBlockStart + i
            val idx2 = subBlockStart + i + 1
            
            val q1 = if (idx1 < blockValues.size && subRange > 0.0f) {
                round(((blockValues[idx1] - subMin) * invScale)).toInt().coerceIn(0, 15)
            } else 0
            
            val q2 = if (idx2 < blockValues.size && subRange > 0.0f) {
                round(((blockValues[idx2] - subMin) * invScale)).toInt().coerceIn(0, 15)
            } else 0
            
            val packedNibbles = (q1 and 0x0F) or ((q2 and 0x0F) shl 4)
            dest[destOffset + 4 + K_SCALE_SIZE + subBlock * 16 + i / 2] = packedNibbles.toByte()
        }
    }
}

/**
 * Quantizes a block of QK_K float values to Q5_K format.
 * Q5_K structure: d (F16), dmin (F16), scales[K_SCALE_SIZE], qh[QK_K/8], qs[QK_K/2]  
 * Effectively 5.5 bits per weight
 */
private fun quantizeQ5_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == QK_K) { "Q5_K block must have $QK_K values" }
    
    // Similar to Q4_K but with 5-bit quantization (0-31 range)
    var minVal = Float.POSITIVE_INFINITY
    var maxVal = Float.NEGATIVE_INFINITY
    for (value in blockValues) {
        minVal = minOf(minVal, value)
        maxVal = maxOf(maxVal, value)
    }
    
    val range = maxVal - minVal
    val d = if (range > 0.0f) range / 511.0f else 1.0f
    val dmin = minVal
    
    // Write super-block scales
    dest.setShortLe(destOffset, floatToHalf(d))
    dest.setShortLe(destOffset + 2, floatToHalf(dmin))
    
    // Process in 32-element sub-blocks
    for (subBlock in 0 until 8) {
        val subBlockStart = subBlock * 32
        
        // Find min/max for this sub-block
        var subMin = Float.POSITIVE_INFINITY
        var subMax = Float.NEGATIVE_INFINITY
        for (i in 0 until 32) {
            val idx = subBlockStart + i
            if (idx < blockValues.size) {
                subMin = minOf(subMin, blockValues[idx])
                subMax = maxOf(subMax, blockValues[idx])
            }
        }
        
        val subRange = subMax - subMin
        val scale = if (subRange > 0.0f) subRange / 31.0f else 1.0f
        val quantizedScale = round((scale / d) * 63.0f).toInt().coerceIn(0, 63)
        val quantizedMin = round((subMin - dmin) / d).toInt().coerceIn(0, 63)
        
        // Store scales (similar packing as Q4_K)
        if (subBlock < K_SCALE_SIZE) {
            dest[destOffset + 4 + subBlock] = ((quantizedScale and 0x3F) or ((quantizedMin and 0x03) shl 6)).toByte()
        }
        
        // Quantize to 5 bits: 4 bits in qs, 1 bit in qh
        val invScale = if (scale > 0.0f) 1.0f / scale else 0.0f
        for (i in 0 until 32 step 2) {
            val idx1 = subBlockStart + i
            val idx2 = subBlockStart + i + 1
            
            val q1 = if (idx1 < blockValues.size && subRange > 0.0f) {
                round((blockValues[idx1] - subMin) * invScale).toInt().coerceIn(0, 31)
            } else 0
            
            val q2 = if (idx2 < blockValues.size && subRange > 0.0f) {
                round((blockValues[idx2] - subMin) * invScale).toInt().coerceIn(0, 31)
            } else 0
            
            // Store low 4 bits in qs
            val qs1 = q1 and 0x0F
            val qs2 = q2 and 0x0F
            dest[destOffset + 4 + K_SCALE_SIZE + QK_K/8 + subBlock * 16 + i / 2] = (qs1 or (qs2 shl 4)).toByte()
            
            // Store high bits in qh
            val qh1 = (q1 shr 4) and 0x01
            val qh2 = (q2 shr 4) and 0x01
            val qhByteIdx = destOffset + 4 + K_SCALE_SIZE + (idx1 / 8)
            val qhBitPos = idx1 % 8
            dest[qhByteIdx] = (dest[qhByteIdx].toInt() or (qh1 shl qhBitPos) or (qh2 shl (qhBitPos + 1))).toByte()
        }
    }
}

/**
 * Quantizes a block of QK_K float values to Q6_K format.
 * Q6_K structure: ql[QK_K/2], qh[QK_K/4], scales[QK_K/16], d (F16)
 * Effectively 6.5625 bits per weight
 */
private fun quantizeQ6_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == QK_K) { "Q6_K block must have $QK_K values" }
    
    // Find absolute maximum for the block
    var amax = 0.0f
    for (value in blockValues) {
        amax = maxOf(amax, abs(value))
    }
    
    val d = if (amax > 0.0f) amax / 127.0f else 1.0f
    val invD = if (d > 0.0f) 1.0f / d else 0.0f
    
    // Write super-block scale at the end
    val dOffset = destOffset + QK_K/2 + QK_K/4 + QK_K/16
    dest.setShortLe(dOffset, floatToHalf(d))
    
    // Process in 16-element sub-blocks
    for (subBlock in 0 until QK_K/16) {
        val subBlockStart = subBlock * 16
        
        // Find max absolute value in sub-block
        var subAmax = 0.0f
        for (i in 0 until 16) {
            val idx = subBlockStart + i
            if (idx < blockValues.size) {
                subAmax = maxOf(subAmax, abs(blockValues[idx]))
            }
        }
        
        // Calculate and store 8-bit scale for this sub-block
        val scale = if (subAmax > 0.0f) subAmax / 63.0f else 1.0f
        val quantizedScale = round((scale / d) * 127.0f).toInt().coerceIn(-128, 127)
        dest[destOffset + QK_K/2 + QK_K/4 + subBlock] = quantizedScale.toByte()
        
        // Quantize values to 6 bits: 4 bits in ql, 2 bits in qh
        val invScale = if (scale > 0.0f) 1.0f / scale else 0.0f
        for (i in 0 until 16 step 2) {
            val idx1 = subBlockStart + i
            val idx2 = subBlockStart + i + 1
            
            val q1 = if (idx1 < blockValues.size) {
                round(blockValues[idx1] * invScale + 32.0f).toInt().coerceIn(0, 63)
            } else 32
            
            val q2 = if (idx2 < blockValues.size) {
                round(blockValues[idx2] * invScale + 32.0f).toInt().coerceIn(0, 63)  
            } else 32
            
            // Store low 4 bits in ql
            val ql1 = q1 and 0x0F
            val ql2 = q2 and 0x0F
            dest[destOffset + subBlock * 8 + i / 2] = (ql1 or (ql2 shl 4)).toByte()
            
            // Store high 2 bits in qh
            val qh1 = (q1 shr 4) and 0x03
            val qh2 = (q2 shr 4) and 0x03
            val qhByteIdx = destOffset + QK_K/2 + (subBlock * 4) + (i / 4)
            val qhBitPos = (i % 4) * 2
            dest[qhByteIdx] = (dest[qhByteIdx].toInt() or (qh1 shl qhBitPos) or (qh2 shl (qhBitPos + 2))).toByte()
        }
    }
}

/**
 * Quantizes a block of QK_K float values to Q8_K format.
 * Q8_K structure: d (F32), qs[QK_K], bsums[QK_K/16]
 * This is used for intermediate quantization and dot products
 */
private fun quantizeQ8_KBlock(blockValues: FloatArray, dest: ByteArray, destOffset: Int) {
    require(blockValues.size == QK_K) { "Q8_K block must have $QK_K values" }
    
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
    for (i in 0 until QK_K) {
        val quantizedValue = round(blockValues[i] * invD).toInt().coerceIn(-128, 127)
        dest[destOffset + 4 + i] = quantizedValue.toByte()
    }
    
    // Calculate block sums for each group of 16
    for (group in 0 until QK_K/16) {
        var sum = 0
        for (i in 0 until 16) {
            val idx = group * 16 + i
            sum += dest[destOffset + 4 + idx].toInt()
        }
        // Store sum as 16-bit integer
        dest.setShortLe(destOffset + 4 + QK_K + group * 2, sum.toShort())
    }
}

// K-Quant Block Dequantization Functions

/**
 * Dequantizes a Q2_K block to float values.
 */
private fun dequantizeQ2_KBlock(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ2_KBlockScale(graphAllocator, blockIndex)
    val dmin = tensor.getQ2_KBlockScaleMin(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Process 16-element sub-blocks
    for (subBlock in 0 until QK_K/16) {
        // Get quantized scale and min for this sub-block
        val scaleAndMin = tensor.getQ2_KScale(graphAllocator, blockIndex, subBlock)
        val quantizedScale = scaleAndMin.toInt() and 0x0F
        val quantizedMin = (scaleAndMin.toInt() shr 4) and 0x0F
        
        // Reconstruct scale and min
        val scale = (quantizedScale.toFloat() / 15.0f) * d
        val min = (quantizedMin.toFloat() * d) + dmin
        
        // Dequantize 16 values (4 values per byte, 2 bits each)
        for (i in 0 until 16 step 4) {
            val packedByte = tensor.getQ2_KQuant(graphAllocator, blockIndex, subBlock * 4 + i / 4)
            
            for (j in 0 until 4) {
                if (elementIdx < dest.size) {
                    val quantizedValue = (packedByte.toInt() shr (j * 2)) and 0x03
                    val dequantizedValue = (quantizedValue.toFloat() / 3.0f) * scale + min
                    dest[elementIdx++] = dequantizedValue
                }
            }
        }
    }
}

/**
 * Dequantizes a Q3_K block to float values.
 */
private fun dequantizeQ3_KBlock(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ3_KBlockScale(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Process 16-element sub-blocks
    for (subBlock in 0 until QK_K/16) {
        // Get quantized scale for this sub-block (stored in scales array)
        val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
        val scaleOffset = blockByteOffset + QK_K/8 + QK_K/4 + subBlock
        val buffer = graphAllocator.buffers[tensor.bufferId] ?: throw IllegalStateException("Tensor buffer not found")
        val quantizedScale = buffer[(tensor.dataOffset + scaleOffset.toULong()).toInt()]
        
        // Reconstruct scale
        val scale = ((quantizedScale.toInt() and 0x3F).toFloat() / 63.0f) * d
        
        // Dequantize 16 values
        val subBlockStart = subBlock * 16
        for (i in 0 until 16) {
            if (elementIdx < dest.size) {
                val globalIdx = subBlockStart + i
                
                // Get low 2 bits from qs
                val qsByteIdx = globalIdx / 4
                val qsBitPos = (globalIdx % 4) * 2
                val qsOffset = blockByteOffset + QK_K/8 + qsByteIdx
                val qsValue = (buffer[(tensor.dataOffset + qsOffset.toULong()).toInt()].toInt() shr qsBitPos) and 0x03
                
                // Get high bit from hmask
                val hmaskByteIdx = globalIdx / 8
                val hmaskBitPos = globalIdx % 8
                val hmaskOffset = blockByteOffset + hmaskByteIdx
                val hmaskValue = (buffer[(tensor.dataOffset + hmaskOffset.toULong()).toInt()].toInt() shr hmaskBitPos) and 0x01
                
                // Combine to get 3-bit value
                val quantizedValue = qsValue or (hmaskValue shl 2)
                val signedValue = if (quantizedValue > 3) quantizedValue - 8 else quantizedValue
                
                dest[elementIdx++] = signedValue.toFloat() * scale
            }
        }
    }
}

/**
 * Dequantizes a Q4_K block to float values.
 */
private fun dequantizeQ4_KBlock(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ4_KBlockScale(graphAllocator, blockIndex)
    val dmin = tensor.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Process 32-element sub-blocks
    for (subBlock in 0 until 8) {
        // Get quantized scale and min for this sub-block from the scales array
        val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
        val buffer = graphAllocator.buffers[tensor.bufferId] ?: throw IllegalStateException("Tensor buffer not found")
        
        // Read packed scale and min values
        val scaleByteOffset = blockByteOffset + 4 + subBlock
        val scaleByte = buffer[(tensor.dataOffset + scaleByteOffset.toULong()).toInt()]
        val quantizedScale = scaleByte.toInt() and 0x3F
        val quantizedMinLow = (scaleByte.toInt() shr 6) and 0x03
        
        val minByteOffset = blockByteOffset + 4 + subBlock * 2 + 1
        val quantizedMinHigh = if (minByteOffset < blockByteOffset + 4 + K_SCALE_SIZE) {
            buffer[(tensor.dataOffset + minByteOffset.toULong()).toInt()].toInt() and 0x0F
        } else 0
        val quantizedMin = quantizedMinLow or (quantizedMinHigh shl 2)
        
        // Reconstruct scale and min
        val scale = (quantizedScale.toFloat() / 63.0f) * d
        val min = (quantizedMin.toFloat() / 63.0f) * d + dmin
        
        // Dequantize 32 values (2 values per byte, 4 bits each)
        val qsBaseOffset = blockByteOffset + 4 + K_SCALE_SIZE + subBlock * 16
        for (i in 0 until 32 step 2) {
            if (elementIdx < dest.size) {
                val qsByte = buffer[(tensor.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                
                val q1 = qsByte.toInt() and 0x0F
                val q2 = (qsByte.toInt() shr 4) and 0x0F
                
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
private fun dequantizeQ5_KBlock(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ5_KBlockScale(graphAllocator, blockIndex)
    val buffer = graphAllocator.buffers[tensor.bufferId] ?: throw IllegalStateException("Tensor buffer not found")
    val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
    
    var elementIdx = destOffset
    
    // Process 32-element sub-blocks  
    for (subBlock in 0 until 8) {
        // Get quantized scale and min for this sub-block
        val scaleByteOffset = blockByteOffset + 4 + subBlock
        val scaleByte = buffer[(tensor.dataOffset + scaleByteOffset.toULong()).toInt()]
        val quantizedScale = scaleByte.toInt() and 0x3F
        val quantizedMin = (scaleByte.toInt() shr 6) and 0x03
        
        val scale = (quantizedScale.toFloat() / 63.0f) * d
        
        // Dequantize 32 values (5-bit: 4 bits in qs, 1 bit in qh)
        val qsBaseOffset = blockByteOffset + 4 + K_SCALE_SIZE + QK_K/8 + subBlock * 16
        val qhBaseOffset = blockByteOffset + 4 + K_SCALE_SIZE
        
        for (i in 0 until 32 step 2) {
            if (elementIdx < dest.size) {
                // Get 4-bit values from qs
                val qsByte = buffer[(tensor.dataOffset + qsBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                val qs1 = qsByte.toInt() and 0x0F
                val qs2 = (qsByte.toInt() shr 4) and 0x0F
                
                // Get high bits from qh
                val globalIdx1 = subBlock * 32 + i
                val globalIdx2 = subBlock * 32 + i + 1
                val qhByte1 = buffer[(tensor.dataOffset + qhBaseOffset.toULong() + (globalIdx1 / 8).toULong()).toInt()]
                val qhByte2 = buffer[(tensor.dataOffset + qhBaseOffset.toULong() + (globalIdx2 / 8).toULong()).toInt()]
                
                val qh1 = (qhByte1.toInt() shr (globalIdx1 % 8)) and 0x01
                val qh2 = (qhByte2.toInt() shr (globalIdx2 % 8)) and 0x01
                
                // Combine to get 5-bit values
                val q1 = qs1 or (qh1 shl 4)
                val q2 = qs2 or (qh2 shl 4)
                
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
private fun dequantizeQ6_KBlock(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ6_KBlockScale(graphAllocator, blockIndex)
    val buffer = graphAllocator.buffers[tensor.bufferId] ?: throw IllegalStateException("Tensor buffer not found")
    val blockByteOffset = blockIndex * tensor.type.byteSize.toInt()
    
    var elementIdx = destOffset
    
    // Process 16-element sub-blocks
    for (subBlock in 0 until QK_K/16) {
        // Get 8-bit scale for this sub-block
        val scaleOffset = blockByteOffset + QK_K/2 + QK_K/4 + subBlock
        val quantizedScale = buffer[(tensor.dataOffset + scaleOffset.toULong()).toInt()]
        val scale = (quantizedScale.toFloat() / 127.0f) * d
        
        // Dequantize 16 values (6-bit: 4 bits in ql, 2 bits in qh)
        val qlBaseOffset = blockByteOffset + subBlock * 8
        val qhBaseOffset = blockByteOffset + QK_K/2 + subBlock * 4
        
        for (i in 0 until 16 step 2) {
            if (elementIdx < dest.size) {
                // Get 4-bit values from ql
                val qlByte = buffer[(tensor.dataOffset + qlBaseOffset.toULong() + (i / 2).toULong()).toInt()]
                val ql1 = qlByte.toInt() and 0x0F
                val ql2 = (qlByte.toInt() shr 4) and 0x0F
                
                // Get 2-bit values from qh
                val qhByte = buffer[(tensor.dataOffset + qhBaseOffset.toULong() + (i / 4).toULong()).toInt()]
                val qhBitPos = (i % 4) * 2
                val qh1 = (qhByte.toInt() shr qhBitPos) and 0x03
                val qh2 = (qhByte.toInt() shr (qhBitPos + 2)) and 0x03
                
                // Combine to get 6-bit values
                val q1 = ql1 or (qh1 shl 4)
                val q2 = ql2 or (qh2 shl 4)
                
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
private fun dequantizeQ8_KBlock(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor, blockIndex: Int, dest: FloatArray, destOffset: Int) {
    val d = tensor.getQ8_KBlockScale(graphAllocator, blockIndex)
    
    var elementIdx = destOffset
    
    // Simple 8-bit dequantization
    for (i in 0 until QK_K) {
        if (elementIdx < dest.size) {
            val quantizedValue = tensor.getQ8_KWeight(graphAllocator, blockIndex, i)
            dest[elementIdx++] = quantizedValue.toFloat() * d
        }
    }
}

/**
 * Compute element-wise square (SQR) into destination tensor (dst-arg only).
 */
private fun computeSqr(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor, dst: GGMLTensor) {
    dst.type = a.type
    a.ne.copyInto(dst.ne); a.nb.copyInto(dst.nb)
    val ts = a.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> {
            val resultData = FloatArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getFloat(graphAllocator, *ind)
                resultData[flatIdx] = value * value
            }
        }
        GGMLType.F16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getHalf(graphAllocator, *ind)
                resultData[flatIdx] = floatToHalf(value * value)
            }
        }
        GGMLType.I32 -> {
            val resultData = IntArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getInt(graphAllocator, *indices).toLong()
                val squared = value * value
                resultData[flatIdx] = when {
                    squared > Int.MAX_VALUE -> Int.MAX_VALUE
                    squared < Int.MIN_VALUE -> Int.MIN_VALUE
                    else -> squared.toInt()
                }
            }
        }
        GGMLType.I16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getShort(graphAllocator, *indices).toInt()
                val squared = value * value
                resultData[flatIdx] = squared.coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        GGMLType.I64 -> {
            val resultData = LongArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
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
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K -> {
            val af = dequantizeTensor(graphAllocator, a)
            val tmp = GGMLTensor(af.type).also { it.ne = a.ne.copyOf(); it.nb = a.nb.copyOf() }
            computeSqr(graphAllocator, context, af, tmp)
            val qr = quantizeTensor(graphAllocator, tmp, a.type)
            dst.data = qr.data
        }
        else -> throw NotImplementedError("computeSqr NI for type ${a.type}")
    }
}

/**
 * Compute element-wise square root (SQRT) into destination tensor (dst-arg only).
 */
private fun computeSqrt(graphAllocator: GGMLGraphAllocator, context: GGMLContext, a: GGMLTensor, dst: GGMLTensor) {
    dst.type = a.type
    a.ne.copyInto(dst.ne); a.nb.copyInto(dst.nb)
    val ts = a.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> {
            val resultData = FloatArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getFloat(graphAllocator, *ind)
                resultData[flatIdx] = if (value < 0.0f) Float.NaN else kotlin.math.sqrt(value)
            }
        }
        GGMLType.F16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, ind ->
                val value = a.getHalf(graphAllocator, *ind)
                val sqrtValue = if (value < 0.0f) Float.NaN else kotlin.math.sqrt(value)
                resultData[flatIdx] = floatToHalf(sqrtValue)
            }
        }
        GGMLType.I32 -> {
            val resultData = IntArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getInt(graphAllocator, *indices)
                require(value >= 0) { "Cannot compute square root of negative integer: $value" }
                resultData[flatIdx] = kotlin.math.sqrt(value.toDouble()).toInt()
            }
        }
        GGMLType.I16 -> {
            val resultData = ShortArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getShort(graphAllocator, *indices).toInt()
                require(value >= 0) { "Cannot compute square root of negative integer: $value" }
                resultData[flatIdx] = kotlin.math.sqrt(value.toDouble()).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        GGMLType.I64 -> {
            val resultData = LongArray(ts)
            dst.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val value = a.getLong(graphAllocator, *indices)
                require(value >= 0) { "Cannot compute square root of negative long: $value" }
                resultData[flatIdx] = kotlin.math.sqrt(value.toDouble()).toLong()
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1,
        GGMLType.Q2_K, GGMLType.Q3_K, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K, GGMLType.Q8_K -> {
            val af = dequantizeTensor(graphAllocator, a)
            val tmp = GGMLTensor(af.type).also { it.ne = a.ne.copyOf(); it.nb = a.nb.copyOf() }
            computeSqrt(graphAllocator, context, af, tmp)
            val qr = quantizeTensor(graphAllocator, tmp, a.type)
            dst.data = qr.data
        }
        else -> throw NotImplementedError("computeSqrt NI for type ${a.type}")
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
    fun computeGraph(graph: GGMLCGraph) {
        val graphAllocator = graph.allocator ?: throw IllegalStateException("Graph must have an allocator")
        
        // Process each node in the graph in topological order
        for (i in 0 until graph.nNodes) {
            val node = graph.nodes[i] ?: continue
            computeNode(graphAllocator, node)
        }
    }
    
    /**
     * Compute a single node in the graph
     */
    private fun computeNode(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        when (node.op) {
            GGMLOp.NONE -> { /* No operation */ }
            GGMLOp.DUP -> computeDup(graphAllocator, node)
            GGMLOp.ADD -> computeAdd(graphAllocator, node)
            GGMLOp.SUB -> computeSub(graphAllocator, node)
            GGMLOp.MUL -> computeMul(graphAllocator, node)
            GGMLOp.DIV -> computeDiv(graphAllocator, node)
            GGMLOp.SQR -> computeSqr(graphAllocator, node)
            GGMLOp.SQRT -> computeSqrt(graphAllocator, node)
            GGMLOp.NEG -> computeNeg(graphAllocator, node)
            GGMLOp.RELU -> computeRelu(graphAllocator, node)
            GGMLOp.GELU -> computeGelu(graphAllocator, node)
            GGMLOp.SILU -> computeSilu(graphAllocator, node)
            GGMLOp.RMS_NORM -> computeRmsNorm(graphAllocator, node)
            GGMLOp.MUL_MAT -> computeMulMat(graphAllocator, node)
            GGMLOp.SUM -> computeSum(graphAllocator, node)
            GGMLOp.MEAN -> computeMean(graphAllocator, node)
            GGMLOp.REPEAT -> computeRepeat(graphAllocator, node)
            // Add more operations as needed
            else -> throw NotImplementedError("Operation ${node.op} not implemented in compute graph")
        }
    }
    
    // Helper methods for individual operations
    
    private fun computeDup(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("DUP operation requires source tensor")
        val context = graphAllocator.context
        // Simple dup: copy element-wise based on type
        node.type = src.type
        node.ne = src.ne.copyOf(); node.nb = calculateContiguousStrides(node.ne, node.type, node.ne.size)
        val total = src.numElements().toInt()
        when (src.type) {
            GGMLType.F32 -> applyNDIter(src, total) { _, ind -> node.setFloat(graphAllocator, src.getFloat(graphAllocator, *ind), *ind) }
            GGMLType.F16 -> applyNDIter(src, total) { _, ind -> node.setHalf(graphAllocator, src.getHalf(graphAllocator, *ind), *ind) }
            GGMLType.I32 -> applyNDIter(src, total) { _, ind -> node.setInt(graphAllocator, src.getInt(graphAllocator, *ind), *ind) }
            GGMLType.I16 -> applyNDIter(src, total) { _, ind -> node.setShort(graphAllocator, src.getShort(graphAllocator, *ind), *ind) }
            GGMLType.I8  -> applyNDIter(src, total) { _, ind -> node.setByte(graphAllocator, src.getByte(graphAllocator, *ind), *ind) }
            GGMLType.I64 -> applyNDIter(src, total) { _, ind -> node.setLong(graphAllocator, src.getLong(graphAllocator, *ind), *ind) }
            else -> {
                // For quantized, perform raw data copy if allocated in same buffer
                node.data = src.data; node.bufferId = src.bufferId; node.dataOffset = src.dataOffset; node.nb = src.nb.copyOf()
            }
        }
    }
    
    private fun computeAdd(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("ADD operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("ADD operation requires second source tensor")
    val context = graphAllocator.context
    computeAdd(graphAllocator, context, src0, src1, node)
    }
    
    private fun computeSub(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("SUB operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("SUB operation requires second source tensor")
    val context = graphAllocator.context
    computeSub(graphAllocator, context, src0, src1, node)
    }
    
    private fun computeMul(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("MUL operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("MUL operation requires second source tensor")
    val context = graphAllocator.context
    computeMul(graphAllocator, context, src0, src1, node)
    }
    
    private fun computeDiv(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("DIV operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("DIV operation requires second source tensor")
    val context = graphAllocator.context
    computeDiv(graphAllocator, context, src0, src1, node)
    }
    
    private fun computeSqr(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SQR operation requires source tensor")
    val context = graphAllocator.context
    computeSqr(graphAllocator, context, src, node)
    }
    
    private fun computeSqrt(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SQRT operation requires source tensor")
    val context = graphAllocator.context
    computeSqrt(graphAllocator, context, src, node)
    }
    
    private fun computeNeg(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("NEG operation requires source tensor")
    val context = graphAllocator.context
    computeNeg(graphAllocator, context, src, node)
    }
    
    private fun computeRelu(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("RELU operation requires source tensor")
    val context = graphAllocator.context
    computeRelu(graphAllocator, context, src, node)
    }
    
    private fun computeGelu(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("GELU operation requires source tensor")
    val context = graphAllocator.context
    computeGelu(graphAllocator, context, src, node)
    }
    
    private fun computeSilu(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SILU operation requires source tensor")
    val context = graphAllocator.context
    // TODO: implement SiLU; quick fallback using GELU for now
    computeGelu(graphAllocator, context, src, node)
    }
    
    private fun computeRmsNorm(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("RMS_NORM operation requires source tensor")
    val context = graphAllocator.context
    // TODO: implement RMSNorm; no-op dup for now to keep graph running
    computeDup(graphAllocator, node)
    }
    
    private fun computeMulMat(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src0 = node.src[0] ?: throw IllegalArgumentException("MUL_MAT operation requires first source tensor")
        val src1 = node.src[1] ?: throw IllegalArgumentException("MUL_MAT operation requires second source tensor")
    val context = graphAllocator.context
    computeMatMul(graphAllocator, context, src0, src1, node)
    }
    
    private fun computeSum(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("SUM operation requires source tensor")
        val context = graphAllocator.context
        // Simple sum over all elements into a scalar at [0,0]
        node.ne[0] = 1; node.ne[1] = 1; node.type = src.type
        var accF = 0.0f; var accI = 0L
        val total = src.numElements().toInt()
        when (src.type) {
            GGMLType.F32 -> { applyNDIter(src, total) { _, ind -> accF += src.getFloat(graphAllocator, *ind) }; node.setFloat(graphAllocator, accF, 0, 0) }
            GGMLType.F16 -> { applyNDIter(src, total) { _, ind -> accF += src.getHalf(graphAllocator, *ind) }; node.setHalf(graphAllocator, accF, 0, 0) }
            GGMLType.I32 -> { applyNDIter(src, total) { _, ind -> accI += src.getInt(graphAllocator, *ind).toLong() }; node.setInt(graphAllocator, accI.toInt(), 0, 0) }
            GGMLType.I64 -> { applyNDIter(src, total) { _, ind -> accI += src.getLong(graphAllocator, *ind) }; node.setLong(graphAllocator, accI, 0, 0) }
            else -> throw NotImplementedError("SUM not implemented for ${src.type}")
        }
    }
    
    private fun computeMean(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("MEAN operation requires source tensor")
        val context = graphAllocator.context
        // Mean using sum / N
        val n = src.numElements().toInt().coerceAtLeast(1)
        node.ne[0] = 1; node.ne[1] = 1; node.type = src.type
        var accF = 0.0f; var accI = 0L
        val total = src.numElements().toInt()
        when (src.type) {
            GGMLType.F32 -> { applyNDIter(src, total) { _, ind -> accF += src.getFloat(graphAllocator, *ind) }; node.setFloat(graphAllocator, accF / n, 0, 0) }
            GGMLType.F16 -> { applyNDIter(src, total) { _, ind -> accF += src.getHalf(graphAllocator, *ind) }; node.setHalf(graphAllocator, accF / n, 0, 0) }
            else -> throw NotImplementedError("MEAN not implemented for ${src.type}")
        }
    }
    
    private fun computeRepeat(graphAllocator: GGMLGraphAllocator, node: GGMLTensor) {
        val src = node.src[0] ?: throw IllegalArgumentException("REPEAT operation requires source tensor")
        val context = graphAllocator.context
        // Basic repeat assumes node.ne is set to desired output shape and src.ne divides node.ne
        for (d in 0 until GGML_MAX_DIMS) require(src.ne[d] == 0L || node.ne[d] % src.ne[d] == 0L) { "REPEAT shape mismatch" }
        val total = node.numElements().toInt()
        when (src.type) {
            GGMLType.F32 -> applyNDIter(node, total) { _, ind ->
                val srcIdx = IntArray(ind.size) { i -> if (src.ne[i] > 0) (ind[i] % src.ne[i].toInt()) else 0 }
                node.setFloat(graphAllocator, src.getFloat(graphAllocator, *srcIdx), *ind)
            }
            GGMLType.F16 -> applyNDIter(node, total) { _, ind ->
                val srcIdx = IntArray(ind.size) { i -> if (src.ne[i] > 0) (ind[i] % src.ne[i].toInt()) else 0 }
                node.setHalf(graphAllocator, src.getHalf(graphAllocator, *srcIdx), *ind)
            }
            else -> throw NotImplementedError("REPEAT not implemented for ${src.type}")
        }
    }
    
    // Removed copyTensorData: compute functions now write directly into node
}
