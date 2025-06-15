package ai.solace.llamakotlin.core

import ai.solace.llamakotlin.core.GGMLGraphAllocator // Required for new function signatures

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

// Helper to iterate N-dimensionally and apply action using flat index and multi-indices
internal fun applyNDIter(tensor: GGMLTensor, totalSize: Int, actionPerElement: (flatIdx: Int, indices: IntArray) -> Unit) {
    val n0 = tensor.ne[0].toInt(); val n1 = tensor.ne[1].toInt()
    val n2 = tensor.ne[2].toInt(); val n3 = tensor.ne[3].toInt()
    var currentFlatIdx = 0

    if (totalSize == 0) return

    if (n3 > 1 || (n3 == 1 && tensor.ne.size >= 4 && tensor.ne.sliceArray(0..3).any { it > 1L })) { // 4D or higher interpreted as 4D
        for (i3 in 0 until (if (n3 == 0 && totalSize == 1) 1 else n3)) { // Handle scalar ne=[0,0,0,0]
            for (i2 in 0 until (if (n2 == 0 && totalSize == 1 && n3 <=1) 1 else n2)) {
                for (i1 in 0 until (if (n1 == 0 && totalSize == 1 && n3 <=1 && n2 <=1) 1 else n1)) {
                    for (i0 in 0 until (if (n0 == 0 && totalSize == 1 && n3 <=1 && n2 <=1 && n1 <=1) 1 else n0)) {
                        if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0, i1, i2, i3)); else return
                    }
                }
            }
        }
    } else if (n2 > 1 || (n2 == 1 && tensor.ne.size >= 3 && tensor.ne.sliceArray(0..2).any { it > 1L })) { // 3D
        for (i2 in 0 until (if (n2 == 0 && totalSize == 1) 1 else n2)) {
            for (i1 in 0 until (if (n1 == 0 && totalSize == 1 && n2 <=1) 1 else n1)) {
                for (i0 in 0 until (if (n0 == 0 && totalSize == 1 && n2 <=1 && n1 <=1) 1 else n0)) {
                     if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0, i1, i2)); else return
                }
            }
        }
    } else if (n1 > 1 || (n1 == 1 && tensor.ne.size >= 2 && tensor.ne.sliceArray(0..1).any { it > 1L })) { // 2D
        for (i1 in 0 until (if (n1 == 0 && totalSize == 1) 1 else n1)) {
            for (i0 in 0 until (if (n0 == 0 && totalSize == 1 && n1 <=1) 1 else n0)) {
                if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0, i1)); else return
            }
        }
    } else if (n0 > 0 || totalSize == 1) { // 1D or Scalar
        for (i0 in 0 until (if (n0 == 0 && totalSize == 1) 1 else n0) ) {
            if (currentFlatIdx < totalSize) actionPerElement(currentFlatIdx++, intArrayOf(i0)); else return
        }
    }
    // If after loops currentFlatIdx is still 0 but totalSize is 1 (true scalar from ne=[1,1,1,1] or similar)
    if (totalSize == 1 && currentFlatIdx == 0 && tensor.ne.all { it == 1L || it == 0L}) {
         actionPerElement(currentFlatIdx++, intArrayOf(0,0,0,0).sliceArray(0 until tensor.ne.count { it > 0L }.coerceAtLeast(1)))
    } else if (totalSize > 0 && currentFlatIdx == 0) { // Fallback for ne=[]
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
    val totalSize = calculateTotalSize(result.ne)

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
            val resultF32 = computeAdd(graphAllocator, context, aF32, bF32) // This F32 result has its own data
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type) // This creates another tensor with own data
            result.data = quantizedResult.data // Copying data array; result tensor itself is not using graphAllocator here.
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

    val totalSize = calculateTotalSize(tensor.ne)
    val resultDataArray = FloatArray(totalSize)

    when (tensor.type) {
        GGMLType.F16 -> {
            applyNDIter(tensor, totalSize) { flatIdx, indices ->
                if (flatIdx < totalSize) resultDataArray[flatIdx] = tensor.getHalf(graphAllocator, *indices)
            }
        }
        GGMLType.F32 -> {
            applyNDIter(tensor, totalSize) { flatIdx, indices ->
                if (flatIdx < totalSize) resultDataArray[flatIdx] = tensor.getFloat(graphAllocator, *indices)
            }
        }
        else -> {
            println("Warning: dequantizeTensor from ${tensor.type} to F32 not fully implemented. Result is zeroed.")
        }
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
        for(d in 0 until GGML_MAX_DIMS) result.nb[d] = 0uL // Placeholder for complex quantized types
    }

    val totalSize = calculateTotalSize(tensorF32.ne)
    when (targetType) {
        GGMLType.F16 -> {
            val resultDataArray = ShortArray(totalSize)
            applyNDIter(tensorF32, totalSize) { flatIdx, indices ->
                if (flatIdx < totalSize) {
                    val f32val = tensorF32.getFloat(graphAllocator, *indices)
                    resultDataArray[flatIdx] = floatToHalf(f32val)
                }
            }
            result.data = resultDataArray
        }
        GGMLType.F32 -> {
            val resultDataArray = FloatArray(totalSize)
            applyNDIter(tensorF32, totalSize) { flatIdx, indices ->
                 if (flatIdx < totalSize) resultDataArray[flatIdx] = tensorF32.getFloat(graphAllocator, *indices)
            }
            result.data = resultDataArray
        }
        GGMLType.Q4_0, GGMLType.Q4_1 -> { result.data = ByteArray((totalSize + 1) / 2); println("Warning: Quantization F32 to ${targetType.name} not fully implemented.") }
        GGMLType.Q5_0, GGMLType.Q5_1 -> { result.data = ByteArray((totalSize * 5 + 7) / 8); println("Warning: Quantization F32 to ${targetType.name} not fully implemented.") }
        GGMLType.Q8_0, GGMLType.Q8_1 -> { result.data = ByteArray(totalSize); println("Warning: Quantization F32 to ${targetType.name} not fully implemented.") }
        else -> {
            println("Error: Unsupported target quantization type $targetType in quantizeTensor")
            result.data = null
        }
    }
    return result
}

/**
 * Multiplies two tensors element-wise.
 */
fun computeMul(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    for (i in 0 until GGML_MAX_DIMS) { if (a.ne[i] != b.ne[i]) throw IllegalArgumentException("Incompatible dimensions for multiplication") }
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = calculateTotalSize(result.ne)
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
    val m_rows = a.ne[0].toInt(); val k_common = a.ne[1].toInt() // a is (m x k)
    val k_common_b = b.ne[0].toInt(); val n_cols = b.ne[1].toInt() // b is (k x n)
    if (k_common != k_common_b) throw IllegalArgumentException("Dimension mismatch for matmul: a.ne[1] (${a.ne[1]}) != b.ne[0] (${b.ne[0]})")

    val result = GGMLTensor(type = a.type)
    result.ne = longArrayOf(m_rows.toLong(), n_cols.toLong(), 1, 1) // Result is (m x n)
    if (result.type.byteSize > 0uL) {
        result.nb[0] = result.type.byteSize
        for (d in 1 until GGML_MAX_DIMS) { result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1] }
    } else { for(d in 0 until GGML_MAX_DIMS) result.nb[d] = 0uL }

    when (a.type) {
        GGMLType.F32 -> {
            for (i in 0 until m_rows) { // Row of result (and a)
                for (j in 0 until n_cols) { // Col of result (and b)
                    var sum = 0.0f
                    for (l_idx in 0 until k_common) { // Common dimension
                        // Access a[i, l_idx] and b[l_idx, j]
                        // Assuming ne[0] is rows, ne[1] is columns for indexing
                        sum += a.getFloat(graphAllocator, l_idx, i) * b.getFloat(graphAllocator, j, l_idx)
                    }
                    result.setFloat(graphAllocator, sum, j, i) // Set result[i, j]
                }
            }
        }
        GGMLType.F16 -> {
             for (i in 0 until m_rows) {
                for (j in 0 until n_cols) {
                    var sum = 0.0f // Accumulate in F32 for precision
                    for (l_idx in 0 until k_common) {
                        sum += a.getHalf(graphAllocator, l_idx, i) * b.getHalf(graphAllocator, j, l_idx)
                    }
                    result.setHalf(graphAllocator, sum, j, i)
                }
            }
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            val aF32 = dequantizeTensor(graphAllocator, a); val bF32 = dequantizeTensor(graphAllocator, b)
            val resultF32 = computeMatMul(graphAllocator, context, aF32, bF32)
            val quantizedResult = quantizeTensor(graphAllocator, resultF32, a.type); result.data = quantizedResult.data
        }
        else -> throw NotImplementedError("computeMatMul not implemented for type ${a.type}")
    }
    return result
}

/**
 * Applies the ReLU activation function to a tensor.
 */
fun computeRelu(graphAllocator: GGMLGraphAllocator, @Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type); result.ne = a.ne.copyOf(); result.nb = a.nb.copyOf()
    val totalSize = calculateTotalSize(result.ne)
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
    val totalSize = calculateTotalSize(result.ne)
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
    val totalSize = calculateTotalSize(result.ne)
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
    val totalSize = calculateTotalSize(result.ne)
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
    val totalSize = calculateTotalSize(result.ne)
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
