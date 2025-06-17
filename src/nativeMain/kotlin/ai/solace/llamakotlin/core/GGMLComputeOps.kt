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
            applyNDIter(result, totalSize) { _, indices ->
                val v0 = a.getFloat(graphAllocator, *indices)
                val v1 = b.getFloat(graphAllocator, *indices)
                result.setFloat(graphAllocator, v0 + v1, *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                val v0 = a.getHalf(graphAllocator, *indices)
                val v1 = b.getHalf(graphAllocator, *indices)
                result.setHalf(graphAllocator, v0 + v1, *indices)
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
        GGMLType.F32->applyNDIter(res,ts){_,ind->res.setFloat(graphAllocator,a.getFloat(graphAllocator,*ind)-b.getFloat(graphAllocator,*ind),*ind)}
        GGMLType.F16->applyNDIter(res,ts){_,ind->res.setHalf(graphAllocator,a.getHalf(graphAllocator,*ind)-b.getHalf(graphAllocator,*ind),*ind)}
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1->{val af=dequantizeTensor(graphAllocator,a); val bf=dequantizeTensor(graphAllocator,b); val rf=computeSub(graphAllocator,context,af,bf); val qr=quantizeTensor(graphAllocator,rf,a.type); res.data=qr.data}
        else->throw NotImplementedError("computeSub NI for ${a.type}")
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
        GGMLType.F32->applyNDIter(res,ts){_,ind->res.setFloat(graphAllocator,div(a.getFloat(graphAllocator,*ind),b.getFloat(graphAllocator,*ind)),*ind)}
        GGMLType.F16->applyNDIter(res,ts){_,ind->res.setHalf(graphAllocator,div(a.getHalf(graphAllocator,*ind),b.getHalf(graphAllocator,*ind)),*ind)}
        GGMLType.Q4_0,GGMLType.Q4_1,GGMLType.Q5_0,GGMLType.Q5_1,GGMLType.Q8_0,GGMLType.Q8_1->{val af=dequantizeTensor(graphAllocator,a);val bf=dequantizeTensor(graphAllocator,b);val rf=computeDiv(graphAllocator,context,af,bf);val qr=quantizeTensor(graphAllocator,rf,a.type);res.data=qr.data}
        else->throw NotImplementedError("computeDiv NI for ${a.type}")
    }
    return res
}

[end of src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt]
