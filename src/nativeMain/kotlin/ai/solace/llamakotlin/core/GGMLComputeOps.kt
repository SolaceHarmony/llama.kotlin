package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.core.GGMLGraphAllocator // Required for new function signatures
import kotlin.math.abs
import kotlin.math.round
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES

/**
 * Kotlin Native port of GGML tensor computation operations.
 * This file contains the implementation of actual computation functionality for tensor operations.
 */

/**
 * Calculates the total size of a tensor based on its dimensions.
 *
 * @param ne The dimensions of the tensor
 * @return The total number of elements in the tensor
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
    b: GGMLTensor
): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for addition")
    }
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()

    when (a.type) {
        GGMLType.F32 -> {
            val resultData = FloatArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices -> // Iterate based on 'a' which has same shape as result
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                resultData[flatIdx] = v0 + v1
            }
        }
        GGMLType.F16 -> {
            val resultData = ShortArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices -> // Iterate based on 'a'
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                // Perform addition as Float for precision, then convert back to Half (Short)
                resultData[flatIdx] = floatToHalf(v0 + v1)
            }
        }
        GGMLType.I32 -> {
            val resultData = IntArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                resultData[flatIdx] = valA + valB
            }
        }
        GGMLType.I16 -> {
            val resultData = ShortArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                resultData[flatIdx] = (valA + valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        GGMLType.I8 -> {
            val resultData = ByteArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                resultData[flatIdx] = (valA + valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte()
            }
        }
        GGMLType.I64 -> {
            val resultData = LongArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                resultData[flatIdx] = valA + valB
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a)
            val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeAdd(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type)
            result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeAdd not implemented for type ${a.type}")
    }
    return result
}

fun computeMul(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for multiplication")
    }
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()

    when (a.type) {
        GGMLType.F32 -> {
            val resultData = FloatArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                resultData[flatIdx] = v0 * v1
            }
        }
        GGMLType.F16 -> {
            val resultData = ShortArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                resultData[flatIdx] = floatToHalf(v0 * v1)
            }
        }
        GGMLType.I32 -> {
            val resultData = IntArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                resultData[flatIdx] = valA * valB
            }
        }
        GGMLType.I16 -> {
            val resultData = ShortArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                resultData[flatIdx] = (valA * valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        GGMLType.I8 -> {
            val resultData = ByteArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                resultData[flatIdx] = (valA * valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte()
            }
        }
        GGMLType.I64 -> {
            val resultData = LongArray(totalSize)
            result.data = resultData
            applyNDIter(a, totalSize) { flatIdx, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                resultData[flatIdx] = valA * valB
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a)
            val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeMul(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type)
            result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeMul not implemented for type ${a.type}")
    }
    return result
}

private fun dequantizeTensor(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor): GGMLTensor {
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
        else -> println("Warning: dequantizeTensor from ${tensor.type} to F32 not fully implemented. Result is zeroed for ${tensor.name}.")
    }
    result.data = resultDataArray; return result
}

private fun quantizeTensor(graphAllocator: GGMLGraphAllocator, tensorF32: GGMLTensor, targetType: GGMLType): GGMLTensor {
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
        GGMLType.Q5_0, GGMLType.Q5_1 -> { result.data = ByteArray((numElements * 5 + 7) / 8); println("Warn: Quant F32 to ${targetType.name} NI") }
        else -> { println("Error: Unsupp target quant type $targetType"); result.data = null }
    }
    return result
}

fun computeMatMul(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val M = a.ne[1].toInt()
    val K_a = a.ne[0].toInt()
    val N = b.ne[0].toInt()
    val K_b = b.ne[1].toInt()
    if (K_a != K_b) throw IllegalArgumentException("Dim mismatch K: a.ne[0]($K_a) != b.ne[1]($K_b)")
    val K = K_a

    if (a.type == GGMLType.Q4_0 && b.type == GGMLType.F32) {
        val result = GGMLTensor(GGMLType.F32); result.ne = longArrayOf(N.toLong(), M.toLong(), 1L, 1L)
        for(i_dim in 2 until GGML_MAX_DIMS) { val neA=a.ne.getOrElse(i_dim){1L}; val neB=b.ne.getOrElse(i_dim){1L}; result.ne[i_dim]=maxOf(neA,neB); if(!(neA==neB || neA==1L || neB==1L)) throw IllegalArgumentException("Matmul broadcast fail dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}") }
        result.nb[0]=result.type.byteSize; for(d in 1 until GGML_MAX_DIMS){result.nb[d]=result.ne.getOrElse(d-1){1L}.toULong()*result.nb[d-1]}
        val resArr=FloatArray(M*N); result.data=resArr; var flatIdx=0
        for(i in 0 until M){ for(j in 0 until N){ resArr[flatIdx++]=computeDotProductQ40F32(graphAllocator,a,b,i,j,K) } }
        return result
    }
    if (a.type == GGMLType.Q4_1 && b.type == GGMLType.F32) {
        // Optimized Q4_1 x F32 path, result is F32
        // a (src0) is M x K (ne[1]=M rows, ne[0]=K cols for element access)
        // b (src1) is K x N (ne[1]=K rows, ne[0]=N cols for element access)
        // Result will be M x N (ne[1]=M rows, ne[0]=N cols)

        // Note: M, K_a, N, K_b, K are already defined at the start of computeMatMul
        // val M = a.ne[1].toInt() // Already available
        // val K_a = a.ne[0].toInt() // Already available
        // val K_b = b.ne[1].toInt() // Already available
        // val N = b.ne[0].toInt()   // Already available
        // val K = K_a // Already available and validated K_a == K_b

        // Create result tensor (F32, M rows, N columns)
        // The ne array for new tensor: [N_cols, M_rows, 1, 1]
        val resultNe = longArrayOf(N.toLong(), M.toLong(), 1L, 1L)
        // Handle broadcasting for dimensions 2 and 3 if necessary, like other paths
        for(i_dim in 2 until GGML_MAX_DIMS) {
            val neA = a.ne.getOrElse(i_dim){1L}
            val neB = b.ne.getOrElse(i_dim){1L}
            resultNe[i_dim] = maxOf(neA, neB)
            if(!(neA == neB || neA == 1L || neB == 1L)) {
                throw IllegalArgumentException("Matmul broadcast fail for Q4_1xF32 on dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}")
            }
        }
        val result = GGMLTensor(GGMLType.F32, resultNe)

        // Calculate and set strides for result tensor
        result.nb[0] = result.type.byteSize
        for (d in 1 until GGML_MAX_DIMS) {
            result.nb[d] = result.ne.getOrElse(d-1) { 1L }.toULong() * result.nb[d-1]
        }

        val resultArray = FloatArray(M * N * resultNe[2].toInt() * resultNe[3].toInt()) // Adjusted for potential broadcasted dims
        result.data = resultArray

        // TODO: Handle broadcasting for dims 2 and 3 in the loops if resultNe[2] or resultNe[3] > 1.
        // For now, assuming non-broadcasted higher dims (or handled by dot product if it supports it, which it doesn't directly)
        // The current Q4_0/Q8_0 paths also don't explicitly show broadcasting in their loops but set up result.ne correctly.
        // This implies the dot product functions are called for each "slice" or the structure expects M*N iterations flatly.
        // Let's match the existing Q4_0/Q8_0 structure which assumes M*N iterations for the primary 2D matrix part.
        // If higher dims were broadcasted, M and N would need to be multiplied by those dimensions for total iterations.
        // The current structure of M*N iterations for the result array is for a single 2D plane.
        // If resultNe[2]*resultNe[3] > 1, the flat idx logic needs to be aware or the loops need to be nested.
        // The existing paths (Q4_0, Q8_0) use a simple M*N loop and flatIdx for resArr.
        // This suggests that for these specific optimized paths, broadcasting in higher dimensions might not be fully handled
        // or is implicitly handled by the calling context if these are part of larger ops.
        // For now, sticking to the M*N loop for the primary matrix plane.

        var idx = 0 // Index for flat resultArray
        for (i in 0 until M) { // Iterate output rows (M)
            for (j in 0 until N) { // Iterate output columns (N)
                // This assumes a, b are effectively 2D for the dot product part.
                // If a or b have higher dimensions that are not 1, computeDotProductQ41F32 needs to handle it,
                // or this loop structure needs to be more complex to iterate over those dimensions.
                // The current dot product functions take simple row/col indices.
                val dotProduct = computeDotProductQ41F32(
                    graphAllocator,
                    a,    // Q4_1 tensor (src0)
                    b,    // F32 tensor (src1)
                    i,    // Current row in src0 (0 to M-1)
                    j,    // Current column in src1 (0 to N-1)
                    K     // Common dimension
                )
                resultArray[idx++] = dotProduct
            }
        }
        return result
    }
    if (a.type == GGMLType.Q8_0 && b.type == GGMLType.F32) {
        val result = GGMLTensor(GGMLType.F32); result.ne = longArrayOf(N.toLong(), M.toLong(), 1L, 1L)
        for(i_dim in 2 until GGML_MAX_DIMS) { val neA=a.ne.getOrElse(i_dim){1L}; val neB=b.ne.getOrElse(i_dim){1L}; result.ne[i_dim]=maxOf(neA,neB); if(!(neA==neB || neA==1L || neB==1L)) throw IllegalArgumentException("Matmul broadcast fail dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}") }
        result.nb[0]=result.type.byteSize; for(d in 1 until GGML_MAX_DIMS){result.nb[d]=result.ne.getOrElse(d-1){1L}.toULong()*result.nb[d-1]}
        val resArr=FloatArray(M*N); result.data=resArr; var flatIdx=0
        for(i in 0 until M){ for(j in 0 until N){ resArr[flatIdx++]=computeDotProductQ80F32(graphAllocator,a,b,i,j,K) } }
        return result
    }

    val resultTensor=GGMLTensor(type=a.type); resultTensor.ne=longArrayOf(N.toLong(),M.toLong(),1,1)
    for(i_dim in 2 until GGML_MAX_DIMS){ val neA=a.ne.getOrElse(i_dim){1L}; val neB=b.ne.getOrElse(i_dim){1L}; resultTensor.ne[i_dim]=maxOf(neA,neB); if(!(neA==neB || neA==1L || neB==1L)) throw IllegalArgumentException("Matmul broadcast fail dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}") }
    if(resultTensor.type.byteSize>0uL){ resultTensor.nb[0]=resultTensor.type.byteSize; for(d in 1 until GGML_MAX_DIMS){resultTensor.nb[d]=resultTensor.ne[d-1].toULong()*resultTensor.nb[d-1]} } else {for(d in 0 until GGML_MAX_DIMS)resultTensor.nb[d]=0uL}

    when (a.type) {
        GGMLType.F32 -> {
            val effA=a; val effB=if(b.type==GGMLType.F32)b else dequantizeTensor(graphAllocator,b)
            val resArr = FloatArray(M*N); resultTensor.data=resArr; resultTensor.type = GGMLType.F32; var flatIdx=0
            for(i in 0 until M){for(j in 0 until N){var sum=0.0f; for(l in 0 until K){sum+=effA.getFloat(graphAllocator,l,i)*effB.getFloat(graphAllocator,j,l)}; resArr[flatIdx++]=sum}}
        }
        GGMLType.F16 -> {
            val effA=a; val effB=if(b.type==GGMLType.F16)b else dequantizeTensor(graphAllocator,b)
            if(effB.type!=GGMLType.F16) throw NotImplementedError("F16xnon-F16 matmul to F16 NI")
            val resArr = ShortArray(M*N); resultTensor.data=resArr; var flatIdx=0
            for(i in 0 until M){for(j in 0 until N){var sum=0.0f; for(l in 0 until K){sum+=effA.getHalf(graphAllocator,l,i)*effB.getHalf(graphAllocator,j,l)}; resArr[flatIdx++]=floatToHalf(sum)}}
        }
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1 -> {
            val aF32=dequantizeTensor(graphAllocator,a); val bF32=dequantizeTensor(graphAllocator,b)
            val resF32=computeMatMul(graphAllocator,context,aF32,bF32)
            val qRes=quantizeTensor(graphAllocator,resF32,resultTensor.type); resultTensor.data=qRes.data
        }
        else -> throw NotImplementedError("computeMatMul NI for input type ${a.type}")
    }
    return resultTensor
}

fun computeRelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> applyNDIter(result, totalSize) { _, ind -> result.setFloat(graphAllocator, if(a.getFloat(graphAllocator, *ind)>0.0f)a.getFloat(graphAllocator,*ind)else 0.0f, *ind) }
        GGMLType.F16 -> applyNDIter(result, totalSize) { _, ind -> result.setHalf(graphAllocator, if(a.getHalf(graphAllocator, *ind)>0.0f)a.getHalf(graphAllocator,*ind)else 0.0f, *ind) }
        else -> throw NotImplementedError("computeRelu NI for type ${a.type}")
    }
    return result
}

fun computeGelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    val gelu = {x:Float -> x*0.5f*(1.0f+kotlin.math.tanh(0.797885f*(x+0.044715f*x*x*x)))}
    when (a.type) {
        GGMLType.F32 -> applyNDIter(result, totalSize) { _, ind -> result.setFloat(graphAllocator, gelu(a.getFloat(graphAllocator, *ind)), *ind) }
        GGMLType.F16 -> applyNDIter(result, totalSize) { _, ind -> result.setHalf(graphAllocator, gelu(a.getHalf(graphAllocator, *ind)), *ind) }
        else -> throw NotImplementedError("computeGelu NI for type ${a.type}")
    }
    return result
}

fun computeSub(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for(i in 0 until GGML_MAX_DIMS){if(a.ne[i]!=b.ne[i])throw IllegalArgumentException("Dims mismatch")}
    val res=GGMLTensor(a.type); res.ne=a.ne.copyOf(); res.nb=a.nb.copyOf(); val ts=res.numElements().toInt()
    when(a.type){
        GGMLType.F32-> {
            val resultData = FloatArray(ts); res.data = resultData
            applyNDIter(a,ts){flatIdx,ind->resultData[flatIdx] = a.getFloat(graphAllocator,*ind)-b.getFloat(graphAllocator,*ind)}
        }
        GGMLType.F16-> {
            val resultData = ShortArray(ts); res.data = resultData
            applyNDIter(a,ts){flatIdx,ind->resultData[flatIdx] = floatToHalf(a.getHalf(graphAllocator,*ind)-b.getHalf(graphAllocator,*ind))}
        }
        GGMLType.I32 -> {
            val resultData = IntArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                resultData[flatIdx] = a.getInt(graphAllocator, *indices) - b.getInt(graphAllocator, *indices)
            }
        }
        GGMLType.I16 -> {
            val resultData = ShortArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                resultData[flatIdx] = (valA - valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        GGMLType.I8 -> {
            val resultData = ByteArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                resultData[flatIdx] = (valA - valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte()
            }
        }
        GGMLType.I64 -> {
            val resultData = LongArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                resultData[flatIdx] = a.getLong(graphAllocator, *indices) - b.getLong(graphAllocator, *indices)
            }
        }
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1->{val af=dequantizeTensor(graphAllocator,a); val bf=dequantizeTensor(graphAllocator,b); val rf=computeSub(graphAllocator,context,af,bf); val qr=quantizeTensor(graphAllocator,rf,a.type); res.data=qr.data}
        else->throw NotImplementedError("computeSub NI for type ${a.type}")
    }
    return res
}

fun computeNeg(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val res=GGMLTensor(a.type); res.ne=a.ne.copyOf(); res.nb=a.nb.copyOf(); val ts=res.numElements().toInt()
    when(a.type){
        GGMLType.F32->applyNDIter(res,ts){_,ind->res.setFloat(graphAllocator,-a.getFloat(graphAllocator,*ind),*ind)}
        GGMLType.F16->applyNDIter(res,ts){_,ind->res.setHalf(graphAllocator,-a.getHalf(graphAllocator,*ind),*ind)}
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1->{val af=dequantizeTensor(graphAllocator,a); val rf=computeNeg(graphAllocator,context,af); val qr=quantizeTensor(graphAllocator,rf,a.type); res.data=qr.data}
        else->throw NotImplementedError("computeNeg NI for ${a.type}")
    }
    return res
}

fun computeDiv(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for(i in 0 until GGML_MAX_DIMS){if(a.ne[i]!=b.ne[i])throw IllegalArgumentException("Dims mismatch")}
    val res=GGMLTensor(a.type);res.ne=a.ne.copyOf();res.nb=a.nb.copyOf();val ts=res.numElements().toInt()
    val div={vA:Float,vB:Float->if(vB==0.0f){if(vA==0.0f)Float.NaN else if(vA>0.0f)Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY}else{vA/vB}}
    when(a.type){
        GGMLType.F32-> {
            val resultData = FloatArray(ts); res.data = resultData
            applyNDIter(a,ts){flatIdx,ind->resultData[flatIdx] = div(a.getFloat(graphAllocator,*ind),b.getFloat(graphAllocator,*ind))}
        }
        GGMLType.F16-> {
            val resultData = ShortArray(ts); res.data = resultData
            applyNDIter(a,ts){flatIdx,ind->resultData[flatIdx] = floatToHalf(div(a.getHalf(graphAllocator,*ind),b.getHalf(graphAllocator,*ind)))}
        }
        GGMLType.I32 -> {
            val resultData = IntArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val valA = a.getInt(graphAllocator, *indices)
                val valB = b.getInt(graphAllocator, *indices)
                if (valB == 0) throw ArithmeticException("Division by zero for I32")
                resultData[flatIdx] = valA / valB
            }
        }
        GGMLType.I16 -> {
            val resultData = ShortArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val valA = a.getShort(graphAllocator, *indices).toInt()
                val valB = b.getShort(graphAllocator, *indices).toInt()
                if (valB == 0) throw ArithmeticException("Division by zero for I16")
                resultData[flatIdx] = (valA / valB).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
            }
        }
        GGMLType.I8 -> {
            val resultData = ByteArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val valA = a.getByte(graphAllocator, *indices).toInt()
                val valB = b.getByte(graphAllocator, *indices).toInt()
                if (valB == 0) throw ArithmeticException("Division by zero for I8")
                resultData[flatIdx] = (valA / valB).coerceIn(Byte.MIN_VALUE.toInt(), Byte.MAX_VALUE.toInt()).toByte()
            }
        }
        GGMLType.I64 -> {
            val resultData = LongArray(ts); res.data = resultData
            applyNDIter(a, ts) { flatIdx, indices ->
                val valA = a.getLong(graphAllocator, *indices)
                val valB = b.getLong(graphAllocator, *indices)
                if (valB == 0L) throw ArithmeticException("Division by zero for I64")
                resultData[flatIdx] = valA / valB
            }
        }
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1->{val af=dequantizeTensor(graphAllocator,a);val bf=dequantizeTensor(graphAllocator,b);val rf=computeDiv(graphAllocator,context,af,bf);val qr=quantizeTensor(graphAllocator,rf,a.type);res.data=qr.data}
        else->throw NotImplementedError("computeDiv NI for type ${a.type}")
    }
    return res
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

[end of src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt]

[end of src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt]
