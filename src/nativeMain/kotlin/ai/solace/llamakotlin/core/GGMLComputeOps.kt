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
    // K (elements in a row of Q80) is tensorQ80.ne[0]
    // K (rows in F32) is tensorF32.ne[1]
    require(tensorQ80.ne[0].toInt() == commonDimK) { "tensorQ80 K dim (${tensorQ80.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)"}

    var sumF32 = 0.0f

    for (k in 0 until commonDimK) { // k iterates along the common dimension K
        // Access Q8_0 element: tensorQ80[rowIndexInQ80, k]
        // rowIndexInQ80 is the "row" (dim 1), k is the "item in row" (dim 0)
        val flatIndexInQ80 = rowIndexInQ80 * commonDimK + k

        val blockIndexQ80 = flatIndexInQ80 / QK8_0
        val itemInBlockQ80 = flatIndexInQ80 % QK8_0

        val scale = tensorQ80.getQ8_0BlockScale(graphAllocator, blockIndexQ80)
        val qWeight = tensorQ80.getQ8_0Weight(graphAllocator, blockIndexQ80, itemInBlockQ80)
        val dequantizedQ80Value = scale * qWeight.toFloat()

        // Access F32 element: tensorF32[k, colIndexInF32]
        // tensorF32 is K rows, N columns (ne[0]=N cols, ne[1]=K rows)
        // To get element at (row k, col colIndexInF32) -> getFloat(colIndexInF32, k)
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)

        sumF32 += dequantizedQ80Value * f32Value
    }
    return sumF32
}

/**
 * Computes the dot product of a row from a Q4_0 tensor and a column from an F32 tensor.
 *
 * @param graphAllocator The graph allocator for accessing tensor data.
 * @param tensorQ40 The Q4_0 tensor (assumed M x K, with ne[0]=K, ne[1]=M).
 * @param tensorF32 The F32 tensor (assumed K x N, with ne[0]=N, ne[1]=K).
 * @param rowIndexInQ40 The row index 'i' for tensorQ40 (0 to M-1).
 * @param colIndexInF32 The column index 'j' for tensorF32 (0 to N-1).
 * @param commonDimK The shared dimension K.
 * @return The resulting dot product as a Float.
 */
internal fun computeDotProductQ40F32(
    graphAllocator: GGMLGraphAllocator,
    tensorQ40: GGMLTensor,
    tensorF32: GGMLTensor,
    rowIndexInQ40: Int,
    colIndexInF32: Int,
    commonDimK: Int
): Float {
    require(tensorQ40.type == GGMLType.Q4_0) { "computeDotProductQ40F32: tensorQ40 must be Q4_0. Got ${tensorQ40.type}" }
    require(tensorF32.type == GGMLType.F32) { "computeDotProductQ40F32: tensorF32 must be F32. Got ${tensorF32.type}" }

    // Assuming tensorQ40.ne = [K, M, ...] (K=cols, M=rows)
    // Assuming tensorF32.ne = [N, K, ...] (N=cols, K=rows)
    require(tensorQ40.ne[0].toInt() == commonDimK) { "tensorQ40 K dim (${tensorQ40.ne[0]}) must match commonDimK ($commonDimK)"}
    require(tensorF32.ne[1].toInt() == commonDimK) { "tensorF32 K dim (${tensorF32.ne[1]}) must match commonDimK ($commonDimK)"}


    var sumF32 = 0.0f

    for (k in 0 until commonDimK) {
        // Access element tensorQ40[rowIndexInQ40, k]
        val flatIndexInQ40 = rowIndexInQ40 * commonDimK + k
        val blockIndexQ40 = flatIndexInQ40 / QK4_0
        val itemInBlockQ40 = flatIndexInQ40 % QK4_0

        val scale = tensorQ40.getQ4_0BlockScale(graphAllocator, blockIndexQ40)
        val qNibble = tensorQ40.getQ4_0NibbleWeight(graphAllocator, blockIndexQ40, itemInBlockQ40)
        val dequantizedQ40Value = scale * (qNibble.toFloat() - 8.0f)

        // Access element tensorF32[k, colIndexInF32]
        // tensorF32 is K rows (ne[1]) x N columns (ne[0]).
        // getFloat(col, row)
        val f32Value = tensorF32.getFloat(graphAllocator, colIndexInF32, k)

        sumF32 += dequantizedQ40Value * f32Value
    }
    return sumF32
}


// Helper to iterate N-dimensionally and apply action using flat index and multi-indices
internal fun applyNDIter(tensor: GGMLTensor, totalSize: Int, actionPerElement: (flatIdx: Int, indices: IntArray) -> Unit) {
    val n0 = tensor.ne[0].toInt(); val n1 = tensor.ne[1].toInt()
    val n2 = tensor.ne[2].toInt(); val n3 = tensor.ne[3].toInt()
    var currentFlatIdx = 0

    if (totalSize == 0) return

    if (n3 > 1 || (n3 == 1 && tensor.ne.size >= 4 && tensor.ne.sliceArray(0..3).any { it > 1L })) {
        for (i3 in 0 until (if (n3 == 0 && totalSize == 1) 1 else n3)) {
            for (i2 in 0 until (if (n2 == 0 && totalSize == 1 && n3 <=1) 1 else n2)) {
                for (i1 in 0 until (if (n1 == 0 && totalSize == 1 && n3 <=1 && n2 <=1) 1 else n1)) {
                    for (i0 in 0 until (if (n0 == 0 && totalSize == 1 && n3 <=1 && n2 <=1 && n1 <=1) 1 else n0)) {
                        if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0, i1, i2, i3)); else return
                    }
                }
            }
        }
    } else if (n2 > 1 || (n2 == 1 && tensor.ne.size >= 3 && tensor.ne.sliceArray(0..2).any { it > 1L })) {
        for (i2 in 0 until (if (n2 == 0 && totalSize == 1) 1 else n2)) {
            for (i1 in 0 until (if (n1 == 0 && totalSize == 1 && n2 <=1) 1 else n1)) {
                for (i0 in 0 until (if (n0 == 0 && totalSize == 1 && n2 <=1 && n1 <=1) 1 else n0)) {
                     if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0, i1, i2)); else return
                }
            }
        }
    } else if (n1 > 1 || (n1 == 1 && tensor.ne.size >= 2 && tensor.ne.sliceArray(0..1).any { it > 1L })) {
        for (i1 in 0 until (if (n1 == 0 && totalSize == 1) 1 else n1)) {
            for (i0 in 0 until (if (n0 == 0 && totalSize == 1 && n1 <=1) 1 else n0)) {
                if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0, i1)); else return
            }
        }
    } else if (n0 > 0 || totalSize == 1) {
        for (i0 in 0 until (if (n0 == 0 && totalSize == 1) 1 else n0) ) {
            if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0)); else return
        }
        else -> throw NotImplementedError("computeAdd not implemented for type ${a.type}")
    }

    if (totalSize == 1 && currentFlatIdx == 0 && tensor.ne.all { it == 1L || (it == 0L && tensor.rank() == 0) }) {
         val actualRank = tensor.rank().coerceAtLeast(1)
         val indices = IntArray(actualRank) { 0 }
         actionPerElement(currentFlatIdx++, indices)
    } else if (totalSize > 0 && currentFlatIdx == 0 && tensor.rank() == 0) {
         actionPerElement(currentFlatIdx++, intArrayOf())
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

/**
 * Dequantizes a tensor to F32 format.
 * The result tensor will have its own `FloatArray` in `data` property, not allocated from graphAllocator.
 */
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
        GGMLType.F16 -> {
            applyNDIter(tensor, numElements) { flatIdx, indices ->
                if (flatIdx < numElements) resultDataArray[flatIdx] = tensor.getHalf(graphAllocator, *indices)
            }
        }
        GGMLType.F32 -> {
            applyNDIter(tensor, numElements) { flatIdx, indices ->
                if (flatIdx < numElements) resultDataArray[flatIdx] = tensor.getFloat(graphAllocator, *indices)
            }
        }
        GGMLType.Q8_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt()
            var f32DataIndex = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ8_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK8_0) {
                    if (f32DataIndex < numElements) {
                        val qWeight = tensor.getQ8_0Weight(graphAllocator, blockIdx, itemIdxInBlock)
                        resultDataArray[f32DataIndex++] = scale * qWeight.toFloat()
                    } else {
                        if(f32DataIndex > 0)
                           println("Warning: Dequantizing Q8_0 tensor ${tensor.name}, f32DataIndex $f32DataIndex exceeded numElements $numElements at block $blockIdx, item $itemIdxInBlock.")
                        break
                    }
                }
                if (f32DataIndex >= numElements && blockIdx < numBlocks - 1) {
                     println("Warning: Filled f32DataArray for Q8_0 tensor ${tensor.name} before processing all blocks. Processed $f32DataIndex elements out of $numElements expected.")
                    break
                }
            }
            if (f32DataIndex != numElements && numElements > 0) {
                 println("Warning: Dequantization of Q8_0 tensor ${tensor.name} resulted in $f32DataIndex dequantized elements, but tensor.numElements() is $numElements.")
            }
        }
        GGMLType.Q4_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt()
            var f32DataIndex = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ4_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK4_0) {
                    if (f32DataIndex < numElements) {
                        val qNibble = tensor.getQ4_0NibbleWeight(graphAllocator, blockIdx, itemIdxInBlock)
                        // Q4_0 dequantization: scale * (nibble_value - 8.0f)
                        val dequantizedValue = scale * (qNibble.toFloat() - 8.0f)
                        resultDataArray[f32DataIndex++] = dequantizedValue
                    } else {
                        if (f32DataIndex > 0)
                            println("Warning: Dequantizing Q4_0 tensor ${tensor.name}, f32DataIndex $f32DataIndex exceeded numElements $numElements at block $blockIdx, item $itemIdxInBlock.")
                        break
                    }
                }
                if (f32DataIndex >= numElements && blockIdx < numBlocks - 1) {
                    println("Warning: Filled f32DataArray for Q4_0 tensor ${tensor.name} before processing all blocks. Processed $f32DataIndex elements out of $numElements expected.")
                    break
                }
            }
            if (f32DataIndex != numElements && numElements > 0) {
                println("Warning: Dequantization of Q4_0 tensor ${tensor.name} resulted in $f32DataIndex dequantized elements, but tensor.numElements() is $numElements.")
            }
        }
        GGMLType.Q4_0 -> {
            val numBlocks = tensor.getNumBlocks().toInt()
            var f32DataIndex = 0
            for (blockIdx in 0 until numBlocks) {
                val scale = tensor.getQ4_0BlockScale(graphAllocator, blockIdx)
                for (itemIdxInBlock in 0 until QK4_0) { // QK4_0 is accessible from GGMLTypes.kt
                    if (f32DataIndex < numElements) {
                        val qNibble = tensor.getQ4_0NibbleWeight(graphAllocator, blockIdx, itemIdxInBlock)
                        // Q4_0 dequantization: scale * (nibble_value - 8.0f)
                        val dequantizedValue = scale * (qNibble.toFloat() - 8.0f)
                        resultDataArray[f32DataIndex++] = dequantizedValue
                    } else {
                        if(f32DataIndex > 0)
                           println("Warning: Dequantizing Q4_0 tensor ${tensor.name}, f32DataIndex $f32DataIndex exceeded numElements $numElements at block $blockIdx, item $itemIdxInBlock.")
                        break
                    }
                }
                if (f32DataIndex >= numElements && blockIdx < numBlocks - 1) {
                     println("Warning: Filled f32DataArray for Q4_0 tensor ${tensor.name} before processing all blocks. Processed $f32DataIndex elements out of $numElements expected.")
                    break
                }
            }
            if (f32DataIndex != numElements && numElements > 0) {
                 println("Warning: Dequantization of Q4_0 tensor ${tensor.name} resulted in $f32DataIndex dequantized elements, but tensor.numElements() is $numElements.")
            }
        }
        else -> {
            println("Warning: dequantizeTensor from ${tensor.type} to F32 not fully implemented. Result is zeroed.")
        }
        else -> throw NotImplementedError("computeMul not implemented for type ${a.type}")
    }
    result.data = resultDataArray

    return result
}

/**
 * Quantizes an F32 tensor to a target type.
 * The result tensor will have its own typed array in `data` property, not allocated from graphAllocator.
 */
private fun quantizeTensor(graphAllocator: GGMLGraphAllocator, tensorF32: GGMLTensor, targetType: GGMLType): GGMLTensor {
    if (tensorF32.type != GGMLType.F32) {
        throw IllegalArgumentException("quantizeTensor expects an F32 input tensor, but got ${tensorF32.type}")
    }
    val result = GGMLTensor(type = targetType)
    result.ne = tensorF32.ne.copyOf()

    if (targetType.byteSize > 0uL) {
        result.nb[0] = targetType.byteSize
        for (d in 1 until GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }
    } else {
        if (targetType != GGMLType.COUNT && !targetType.description.startsWith("Q", ignoreCase = true)) {
             println("Warning: Stride calculation for target type ${targetType.name} in quantizeTensor might be incomplete.")
        }
        for(d in 0 until GGML_MAX_DIMS) result.nb[d] = 0uL
    }
    if (resultTensor.type.byteSize > 0uL) {
        resultTensor.nb[0] = resultTensor.type.byteSize
        for (d in 1 until GGML_MAX_DIMS) { resultTensor.nb[d] = resultTensor.ne[d-1].toULong() * resultTensor.nb[d-1] }
    } else { for(d in 0 until GGML_MAX_DIMS) resultTensor.nb[d] = 0uL }

    val numElements = tensorF32.numElements().toInt()

    when (targetType) {
        GGMLType.F16 -> {
            val resultDataArray = ShortArray(numElements)
            applyNDIter(tensorF32, numElements) { flatIdx, indices ->
                if (flatIdx < numElements) {
                    val f32val = tensorF32.getFloat(graphAllocator, *indices)
                    resultDataArray[flatIdx] = floatToHalf(f32val)
                }
            }
            result.data = resultDataArray
        }
        GGMLType.F32 -> {
            val resultDataArray = FloatArray(numElements)
            applyNDIter(tensorF32, numElements) { flatIdx, indices ->
                 if (flatIdx < numElements) resultDataArray[flatIdx] = tensorF32.getFloat(graphAllocator, *indices)
            }
            result.data = resultDataArray
        }
        GGMLType.Q8_0 -> {
            require(numElements % QK8_0 == 0) { "For Q8_0 quantization, total elements ($numElements) must be divisible by QK8_0 ($QK8_0)" }
            val numBlocks = numElements / QK8_0
            val q8BlockByteSize = targetType.byteSize.toInt()
            require(q8BlockByteSize == SHORT_SIZE_BYTES + QK8_0) {
                 "Mismatch in Q8_0 block byte size: expected ${SHORT_SIZE_BYTES + QK8_0}, but got $q8BlockByteSize from type.byteSize (${targetType.byteSize})"
            }
            val q8DataArray = ByteArray(numBlocks * q8BlockByteSize)

            val f32BlockValues = FloatArray(QK8_0)
            var currentF32ElementIndex = 0
            var q8ByteArrayWriteOffset = 0

            applyNDIter(tensorF32, numElements) { _, indices ->
                val itemInBlock = currentF32ElementIndex % QK8_0
                f32BlockValues[itemInBlock] = tensorF32.getFloat(graphAllocator, *indices)

                if (itemInBlock == QK8_0 - 1) {
                    var amax = 0.0f
                    for (v_idx in 0 until QK8_0) { amax = maxOf(amax, abs(f32BlockValues[v_idx])) }

                    val scaleF32 = if (amax == 0.0f) 1.0f else amax / 127.0f
                    val invScaleF32 = 1.0f / scaleF32
                    val scaleF16Short = floatToHalf(scaleF32)

                    q8DataArray.setShortLe(q8ByteArrayWriteOffset, scaleF16Short)
                    val qsDataWriteOffset = q8ByteArrayWriteOffset + SHORT_SIZE_BYTES

                    for (k in 0 until QK8_0) {
                        val fVal = f32BlockValues[k]
                        val scaledVal = fVal * invScaleF32
                        var iVal = round(scaledVal).toInt()
                        iVal = iVal.coerceIn(-128, 127)
                        q8DataArray[qsDataWriteOffset + k] = iVal.toByte()
                    }
                    q8ByteArrayWriteOffset += q8BlockByteSize
                }
                currentF32ElementIndex++
            }
            result.data = q8DataArray
        }
        GGMLType.Q4_0 -> {
            require(numElements % QK4_0 == 0) { "For Q4_0 quantization, total elements ($numElements) must be divisible by QK4_0 ($QK4_0)" }
            val numBlocks = numElements / QK4_0
            val q4BlockByteSize = targetType.byteSize.toInt()
            require(q4BlockByteSize == SHORT_SIZE_BYTES + QK4_0 / 2) {
                "Q4_0 block byte size mismatch. Expected ${SHORT_SIZE_BYTES + QK4_0 / 2}, got $q4BlockByteSize"
            }
            val q4DataArray = ByteArray(numBlocks * q4BlockByteSize)

            val f32BlockValues = FloatArray(QK4_0)
            var currentF32ElementIndex = 0
            var q4ByteArrayWriteOffset = 0

            applyNDIter(tensorF32, numElements) { _, indices ->
                val itemInBlock = currentF32ElementIndex % QK4_0
                f32BlockValues[itemInBlock] = tensorF32.getFloat(graphAllocator, *indices)

                if (itemInBlock == QK4_0 - 1) { // Block is full
                    var amax = 0.0f
                    for (v_idx in 0 until QK4_0) { amax = maxOf(amax, abs(f32BlockValues[v_idx])) }

                    val scaleF32 = if (amax == 0.0f) 1.0f else amax / 8.0f // Q4_0 range is -8 to 7
                    val invScaleF32 = if (scaleF32 == 0.0f) 0.0f else 1.0f / scaleF32 // Should be 1.0f if amax = 0
                    val scaleF16Short = floatToHalf(scaleF32)

                    q4DataArray.setShortLe(q4ByteArrayWriteOffset, scaleF16Short)
                    val qsDataWriteStartOffset = q4ByteArrayWriteOffset + SHORT_SIZE_BYTES

                    for (j in 0 until QK4_0 / 2) {
                        val f32Val1 = f32BlockValues[j * 2]
                        val f32Val2 = f32BlockValues[j * 2 + 1]

                        val quantVal1 = round(f32Val1 * invScaleF32 + 8.0f).toInt().coerceIn(0, 15)
                        val quantVal2 = round(f32Val2 * invScaleF32 + 8.0f).toInt().coerceIn(0, 15)


                        val packedByte = (quantVal1 and 0x0F) or ((quantVal2 and 0x0F) shl 4)
                        q4DataArray[qsDataWriteStartOffset + j] = packedByte.toByte()
                    }
                    q4ByteArrayWriteOffset += q4BlockByteSize
                }
                currentF32ElementIndex++
            result.data = q4DataArray
        }
        GGMLType.Q4_0 -> {
            require(numElements % QK4_0 == 0) { "For Q4_0 quantization, total elements ($numElements) must be divisible by QK4_0 ($QK4_0)" }
            val numBlocks = numElements / QK4_0
            val q4BlockByteSize = targetType.byteSize.toInt()
            require(q4BlockByteSize == SHORT_SIZE_BYTES + QK4_0 / 2) {
                "Q4_0 block byte size mismatch. Expected ${SHORT_SIZE_BYTES + QK4_0 / 2}, got $q4BlockByteSize"
            }
            val q4DataArray = ByteArray(numBlocks * q4BlockByteSize)

            val f32BlockValues = FloatArray(QK4_0)
            var currentF32ElementIndex = 0
            var q4ByteArrayWriteOffset = 0

            applyNDIter(tensorF32, numElements) { _, indices ->
                val itemInBlock = currentF32ElementIndex % QK4_0
                f32BlockValues[itemInBlock] = tensorF32.getFloat(graphAllocator, *indices)

                if (itemInBlock == QK4_0 - 1) { // Block is full
                    var amax = 0.0f
                    for (v_idx in 0 until QK4_0) { amax = maxOf(amax, abs(f32BlockValues[v_idx])) }

                    val scaleF32 = if (amax == 0.0f) 1.0f else amax / 8.0f // Q4_0 range is -8 to 7
                    val invScaleF32 = if (scaleF32 == 0.0f) 0.0f else 1.0f / scaleF32 // scaleF32 is 1.0f if amax is 0
                    val scaleF16Short = floatToHalf(scaleF32)

                    q4DataArray.setShortLe(q4ByteArrayWriteOffset, scaleF16Short)
                    val qsDataWriteStartOffset = q4ByteArrayWriteOffset + SHORT_SIZE_BYTES

                    for (j in 0 until QK4_0 / 2) {
                        val f32Val1 = f32BlockValues[j * 2]
                        val f32Val2 = f32BlockValues[j * 2 + 1]

                        val quantVal1 = round(f32Val1 * invScaleF32 + 8.0f).toInt().coerceIn(0, 15)
                        val quantVal2 = round(f32Val2 * invScaleF32 + 8.0f).toInt().coerceIn(0, 15)

                        val packedByte = (quantVal1 and 0x0F) or ((quantVal2 and 0x0F) shl 4)
                        q4DataArray[qsDataWriteStartOffset + j] = packedByte.toByte()
                    }
                    q4ByteArrayWriteOffset += q4BlockByteSize

                }
                currentF32ElementIndex++
            }
            result.data = q4DataArray
        }
        GGMLType.Q8_1 -> { result.data = ByteArray(numElements); println("Warning: Quantization F32 to ${targetType.name} not fully implemented.") }
        GGMLType.Q4_1 -> { result.data = ByteArray((numElements + 1) / 2); println("Warning: Quantization F32 to ${targetType.name} not fully implemented.") }
        GGMLType.Q5_0, GGMLType.Q5_1 -> { result.data = ByteArray((numElements * 5 + 7) / 8); println("Warning: Quantization F32 to ${targetType.name} not fully implemented.") }
        else -> {
            println("Error: Unsupported target quantization type $targetType in quantizeTensor")
            result.data = null

        }
        else -> throw NotImplementedError("computeMatMul not implemented for input type ${a.type} or this combination")
    }
    return result
}

/**
 * Multiplies two tensors element-wise.
 */
fun computeMul(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) { if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for multiplication") }

    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setFloat(graphAllocator, a.getFloat(graphAllocator, *indices) * b.getFloat(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setHalf(graphAllocator, a.getHalf(graphAllocator, *indices) * b.getHalf(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a); val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeMul(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type); result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeMul not implemented for type ${a.type}")
    }
    return result
}

/**
 * Performs matrix multiplication of two tensors.
 */
fun computeMatMul(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    // GGML's ne convention for matrices: ne[0]=cols, ne[1]=rows
    // So, a is M rows, K cols means a.ne = [K, M, ...]
    // b is K rows, N cols means b.ne = [N, K, ...]
    // result is M rows, N cols means result.ne = [N, M, ...]
    val M = a.ne[1].toInt()
    val K_a = a.ne[0].toInt()

    val N = b.ne[0].toInt()
    val K_b = b.ne[1].toInt()

    if (K_a != K_b) {
        throw IllegalArgumentException(
            "Dimension mismatch for matmul: a.ne[0] (K for a) ($K_a) != b.ne[1] (K for b) ($K_b)"
        )
    }
    val K = K_a // Common dimension

    // Optimized path for Q4_0 x F32 -> F32
    if (a.type == GGMLType.Q4_0 && b.type == GGMLType.F32) {
        // Result tensor should be F32.
        // The 'result' tensor passed to computeMatMul by GGMLOps.mulMat is typed like 'a'.
        // So, we must create a new F32 tensor here if 'a' is Q4_0.
        // Or, the caller GGMLOps.mulMat must know the output type.
        // For now, let's assume this function is responsible for returning an F32 tensor.
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne = longArrayOf(N.toLong(), M.toLong(), 1L, 1L) // Result is M rows, N cols. ne=[N (cols), M (rows)]
        for(i_dim in 2 until GGML_MAX_DIMS) {
            val neA = if (i_dim < a.ne.size) a.ne[i_dim] else 1L
            val neB = if (i_dim < b.ne.size) b.ne[i_dim] else 1L
            result.ne[i_dim] = maxOf(neA, neB)
            if (!(neA == neB || neA == 1L || neB == 1L)) {
                 throw IllegalArgumentException("Matmul broadcasting not supported for dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}")
            }
        }
        // Calculate strides for the F32 result tensor
        result.nb[0] = result.type.byteSize // GGMLType.F32.byteSize
        for (d in 1 until GGML_MAX_DIMS) { result.nb[d] = result.ne.getOrElse(d-1) { 1L }.toULong() * result.nb[d-1] }

        // Allocate data for the result tensor (as it's new and not from graphAllocator's main buffer)
        result.data = FloatArray(M * N) // This function returns a tensor with its own data

        val resultArray = result.data as FloatArray
        var flatIdx = 0
        for (i in 0 until M) { // Iterate output rows (M)
            for (j in 0 until N) { // Iterate output columns (N)
                val dotProduct = computeDotProductQ40F32(
                    graphAllocator,
                    a,    // Q4_0 tensor (src0)
                    b,    // F32 tensor (src1)
                    i,    // Current row in src0
                    j,    // Current column in src1
                    K     // Common dimension
                )
                resultArray[flatIdx++] = dotProduct
            }
        }
        return result
    }

    // Optimized path for Q8_0 x F32 -> F32
    if (a.type == GGMLType.Q8_0 && b.type == GGMLType.F32) {
        // This path also needs to ensure 'result' is F32.
        // The 'result' tensor passed by GGMLOps.mulMat will have a.type.
        // If a.type is Q8_0, but we need an F32 result, we must create a new tensor.
        val result = GGMLTensor(type = GGMLType.F32) // Ensure result is F32
        result.ne = longArrayOf(N.toLong(), M.toLong(), 1, 1) // ne[0]=cols, ne[1]=rows
        for(i_dim in 2 until GGML_MAX_DIMS) { // Handle higher dimensions via broadcasting
            val neA = if (i_dim < a.ne.size) a.ne[i_dim] else 1L
            val neB = if (i_dim < b.ne.size) b.ne[i_dim] else 1L
            resultF32.ne[i_dim] = maxOf(neA, neB)
            if (!(neA == neB || neA == 1L || neB == 1L)) {
                 throw IllegalArgumentException("Matmul broadcasting not supported for dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}")
            }
        }
        // Calculate strides for F32 result tensor
        resultF32.nb[0] = resultF32.type.byteSize // 4uL
        for (d in 1 until GGML_MAX_DIMS) { resultF32.nb[d] = resultF32.ne[d-1].toULong() * resultF32.nb[d-1] }

        // TODO: The resultF32 tensor data is not allocated from graphAllocator here.
        // This means setFloat will fail if it expects resultF32 to be in graphAllocator.buffers
        // This needs to be reconciled with how results of compute ops are handled.
        // For now, let's assume computeMatMul is responsible for allocating its result if not inplace.
        // If resultF32.data is null, setFloat will fail.
        // The caller of computeMatMul (GGMLOps.kt) creates the result tensor.
        // This function should fill the data of the `result` tensor passed to it, or a new one it returns.
        // The plan was to add an 'if' block, so 'result' is the tensor to fill.
        // So, if a.type is Q8_0 and b.type is F32, the 'result' tensor should be F32.

        // This function (computeMatMul) is expected to return a GGMLTensor.
        // The caller (e.g. GGMLOps.mulMat) creates 'result = GGMLTensor(type = a.type)'
        // If a.type is Q8_0, then result.type is Q8_0. But this path computes F32.
        // This implies that if src0 is Q8_0, the result of matmul(Q8_0, F32) must be F32,
        // and the 'result' tensor created by the caller must be F32.
        // This needs to be handled by the caller, or computeMatMul must create the correct result type.
        // Let's assume the 'result' tensor passed in (or created at start of function) is F32 type for this path.
        require(result.type == GGMLType.F32) {"Result tensor type must be F32 for Q8_0 x F32 matmul"}
        result.ne = longArrayOf(N.toLong(), M.toLong(), resultF32.ne[2], resultF32.ne[3]) // Ensure result has correct shape
        // Recalculate strides for `result` if ne changed.
        result.nb[0] = result.type.byteSize
        for (d in 1 until GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }


        for (i in 0 until M) { // Row index for 'a' and 'result' (0 to M-1)
            for (j in 0 until N) { // Column index for 'b' and 'result' (0 to N-1)
                // computeDotProductQ80F32 expects: tensorQ80 (M x K, ne=[K,M]), tensorF32 (K x N, ne=[N,K])
                // rowIndexInQ80 (for M), colIndexInF32 (for N), commonDimK
                val dotProduct = computeDotProductQ80F32(
                    graphAllocator, a, b, i, j, K
                )
                // result is M x N, ne=[N,M]. setFloat(colIdx, rowIdx, ...)
                result.setFloat(graphAllocator, dotProduct, j, i)
            }
        }
        return result // Return early as computation for this specific case is done
    }

    // Fallback for other types or if the specific Q8_0 x F32 path is not taken
    val resultTensor = GGMLTensor(type = a.type) // Default result type matches 'a'
    resultTensor.ne = longArrayOf(N.toLong(), M.toLong(), 1, 1)
    for(i_dim in 2 until GGML_MAX_DIMS) {
        val neA = if (i_dim < a.ne.size) a.ne[i_dim] else 1L
        val neB = if (i_dim < b.ne.size) b.ne[i_dim] else 1L
        resultTensor.ne[i_dim] = maxOf(neA, neB)
        if (!(neA == neB || neA == 1L || neB == 1L)) {
             throw IllegalArgumentException("Matmul broadcasting not supported for dim $i_dim: ${a.ne[i_dim]} vs ${b.ne[i_dim]}")
        }
    }
    if (resultTensor.type.byteSize > 0uL) {
        resultTensor.nb[0] = resultTensor.type.byteSize
        for (d in 1 until GGML_MAX_DIMS) { resultTensor.nb[d] = resultTensor.ne[d-1].toULong() * resultTensor.nb[d-1] }
    } else { for(d in 0 until GGML_MAX_DIMS) resultTensor.nb[d] = 0uL }


    when (if (a.type == GGMLType.Q8_0 && b.type == GGMLType.F32) GGMLType.F32 else a.type) { // This condition is complex due to above if
        GGMLType.F32 -> {
            val effA = if (a.type == GGMLType.F32) a else dequantizeTensor(graphAllocator, a)
            val effB = if (b.type == GGMLType.F32) b else dequantizeTensor(graphAllocator, b)
            // effA is M x K (ne=[K,M]), effB is K x N (ne=[N,K])
            for (i in 0 until M) {
                for (j in 0 until N) {
                    var sum = 0.0f
                    for (l_idx in 0 until K) {
                        sum += effA.getFloat(graphAllocator, l_idx, i) * effB.getFloat(graphAllocator, j, l_idx)
                    }
                    resultTensor.setFloat(graphAllocator, sum, j, i)
                }
            }
        }
        GGMLType.F16 -> {
             val effA = a
             val effB = if (b.type == GGMLType.F16) b else dequantizeTensor(graphAllocator, b) // Promote b to F32 if not F16, then F16*F32 -> F16 result? This is tricky.
                                                                                             // For F16*F16 -> F16, this is fine.
            if (effB.type != GGMLType.F16) throw NotImplementedError("F16 x non-F16 matmul resulting in F16 not standard.")

             for (i in 0 until M) {
                for (j in 0 until N) {
                    var sum = 0.0f
                    for (l_idx in 0 until K) {
                        sum += effA.getHalf(graphAllocator, l_idx, i) * effB.getHalf(graphAllocator, j, l_idx)
                    }
                    resultTensor.setHalf(graphAllocator, sum, j, i)
                }
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_1 -> { // Q8_0 already handled if b is F32
            val aF32 = dequantizeTensor(graphAllocator, a); val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeMatMul(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, resultTensor.type); resultTensor.data = quantizedResult.data
        }
         GGMLType.Q8_0 -> { // This case implies b is not F32 (e.g. Q8_0 x Q8_0)
            val aF32 = dequantizeTensor(graphAllocator, a); val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeMatMul(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, resultTensor.type); resultTensor.data = quantizedResult.data
        }

        else -> throw NotImplementedError("computeMatMul not implemented for input type ${a.type} or this combination")
    }
    return resultTensor
}

/**
 * Applies the ReLU activation function to a tensor.
 */
fun computeRelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(result, totalSize) { _, indices ->
                val v = a.getFloat(graphAllocator, *indices)
                result.setFloat(graphAllocator, if (v > 0.0f) v else 0.0f, *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                val v = a.getHalf(graphAllocator, *indices)
                result.setHalf(graphAllocator, if (v > 0.0f) v else 0.0f, *indices)
            }
        }
        else -> throw NotImplementedError("computeRelu not implemented for type ${a.type}")
    }
    return result
}

/**
 * Applies the GELU activation function to a tensor.
 */
fun computeGelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    val geluApprox = { x: Float -> x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x))) }
    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setFloat(graphAllocator, geluApprox(a.getFloat(graphAllocator, *indices)), *indices)

            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setHalf(graphAllocator, geluApprox(a.getHalf(graphAllocator, *indices)), *indices)
            }
        }
        else -> throw NotImplementedError("computeGelu not implemented for type ${a.type}")

    }
    return result
}

/**
 * Subtracts one tensor from another element-wise.
 */
fun computeSub(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) { if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for subtraction") }
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setFloat(graphAllocator, a.getFloat(graphAllocator, *indices) - b.getFloat(graphAllocator, *indices), *indices)

            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setHalf(graphAllocator, a.getHalf(graphAllocator, *indices) - b.getHalf(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a); val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeSub(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type); result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeSub not implemented for type ${a.type}")
    }
    return result
}

/**
 * Negates a tensor element-wise.
 */
fun computeNeg(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setFloat(graphAllocator, -a.getFloat(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setHalf(graphAllocator, -a.getHalf(graphAllocator, *indices), *indices)
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a)
            val resultF32 = computeNeg(graphAllocator, context, aF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type); result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeNeg not implemented for type ${a.type}")
    }
    return result
}

/**
 * Divides one tensor by another element-wise.
 */
fun computeDiv(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) { if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for division") }
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = result.numElements().toInt()
    val divideOp = { vA: Float, vB: Float ->
        if (vB == 0.0f) { if (vA == 0.0f) Float.NaN else if (vA > 0.0f) Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY }
        else { vA / vB }
    }
    when (a.type) {
        GGMLType.F32 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setFloat(graphAllocator, divideOp(a.getFloat(graphAllocator, *indices), b.getFloat(graphAllocator, *indices)), *indices)
            }
        }
        GGMLType.F16 -> {
            applyNDIter(result, totalSize) { _, indices ->
                result.setHalf(graphAllocator, divideOp(a.getHalf(graphAllocator, *indices), b.getHalf(graphAllocator, *indices)), *indices)
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a); val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeDiv(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type); result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeDiv not implemented for type ${a.type}")
    }
    return result
}
