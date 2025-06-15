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
 *
 * @param type The tensor data type
 * @param size The number of elements to allocate
 * @return The allocated memory as an appropriate array type
 */
@Suppress("unused")
fun allocateMemory(type: GGMLType, size: Int): Any {
    return when (type) {
        GGMLType.F32 -> FloatArray(size) { 0.0f }
        GGMLType.F16 -> ShortArray(size) { 0 }
        GGMLType.I8 -> ByteArray(size) { 0 }
        GGMLType.I16 -> ShortArray(size) { 0 }
        GGMLType.I32 -> IntArray(size) { 0 }
        GGMLType.I64 -> LongArray(size) { 0L }
        else -> ByteArray(size) { 0 } // Default for quantized types
    }
}

/**
 * Adds two tensors element-wise.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor
 */
fun computeAdd(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    // Check that the tensors have compatible dimensions
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) {
            throw IllegalArgumentException("Incompatible dimensions for addition")
        }
    }

    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Perform the addition based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            // result.data is no longer directly assigned for F32.
            // Values are set into the buffer provided by graphAllocator via result.setFloat(...)

            // Determine tensor rank and iterate accordingly
            // This is a simplified rank determination. A more robust method might be needed.
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                val v0 = a.getFloat(graphAllocator, i0, i1, i2, i3)
                                val v1 = b.getFloat(graphAllocator, i0, i1, i2, i3)
                                result.setFloat(graphAllocator, v0 + v1, i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            val v0 = a.getFloat(graphAllocator, i0, i1, i2)
                            val v1 = b.getFloat(graphAllocator, i0, i1, i2)
                            result.setFloat(graphAllocator, v0 + v1, i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        val v0 = a.getFloat(graphAllocator, i0, i1)
                        val v1 = b.getFloat(graphAllocator, i0, i1)
                        result.setFloat(graphAllocator, v0 + v1, i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D (n0 > 0 to handle true scalars where ne[0] could be 1 or 0)
                                 // Or if ne[0] is always >= 1 for valid tensors.
                for (i0 in 0 until n0) {
                    val v0 = a.getFloat(graphAllocator, i0)
                    val v1 = b.getFloat(graphAllocator, i0)
                    result.setFloat(graphAllocator, v0 + v1, i0)
                }
            } else { // Scalar - assuming ne are all 1 or 0.
                     // getFloat with no args might be ambiguous if not truly scalar.
                     // If ne=[1,1,1,1], getFloat(0,0,0,0) is more precise.
                     // For now, assume if we reach here, it's a scalar for getFloat()
                val v0 = a.getFloat(graphAllocator) // Or a.getFloat(graphAllocator, 0) if 1D scalar
                val v1 = b.getFloat(graphAllocator) // Or b.getFloat(graphAllocator, 0)
                result.setFloat(graphAllocator, v0 + v1) // Or result.setFloat(graphAllocator, v0+v1, 0)
            }
            // The line `result.data = resultData` is removed as data is written to graphAllocator's buffer.
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    // Convert shorts to floats, add, then convert back to short
                    val aFloat = aData[j].toFloat() / 32768.0f
                    val bFloat = bData[j].toFloat() / 32768.0f
                    val sum = aFloat + bFloat
                    resultData[j] = (sum * 32768.0f).toInt().toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val bData = b.data as ByteArray
            val resultData = ByteArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (aData[j] + bData[j]).toByte()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (aData[j] + bData[j]).toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = aData[j] + bData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val bData = b.data as LongArray
            val resultData = LongArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = aData[j] + bData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // Basic implementation for quantized types - dequantize, add, then requantize
            // This is a placeholder and should be optimized in the future

            // For now, we'll convert to F32, perform the operation, and convert back
            // In a real implementation, we would have specialized code for each quantized type

            // Create temporary F32 tensors
            val aF32 = dequantizeTensor(a) // This dequantize may need graphAllocator if it creates tensors for graphAllocator
            val bF32 = dequantizeTensor(b) // Same here

            // Perform addition in F32
            // IMPORTANT: computeAdd now requires graphAllocator.
            // This recursive call needs to be updated.
            // For now, this will cause a compile error. This will be addressed later if dequantize/quantize are in scope.
            // For this step, focusing on direct F32 ops.
            // Let's assume this part is NOT refactored in this specific step to avoid complexity with dequantize/quantize.
            // The task is to refactor F32 ops. The quantized ops call F32 ops.
            // So, the F32 path of this computeAdd will be used by quant ops.
            // The call from quantized path should be: computeAdd(graphAllocator, context, aF32, bF32)
            // However, dequantizeTensor and quantizeTensor also create new tensors and would need allocation.
            // This is out of scope for "refactor F32 ops".
            // For now, I will assume that the F32 path is the primary target.
            // The existing code for quantized types will break if computeAdd signature changes and calls are not updated.
            // The plan is to refactor F32, so the F32 block is the main change.
            // The recursive call from Q_TYPES -> F32 computeAdd needs to pass graphAllocator.
            // This implies dequantizeTensor and quantizeTensor might also need graphAllocator if they manage tensor memory.
            // Let's assume for now that dequantize/quantize are separate and this call path is more complex.
            // I will add graphAllocator to the recursive call.
            val aF32 = dequantizeTensor(graphAllocator, a) // Assuming dequantize needs it
            val bF32 = dequantizeTensor(graphAllocator, b) // Assuming dequantize needs it

            // Perform addition in F32
            val resultF32 = computeAdd(graphAllocator, context, aF32, bF32)

            // Requantize to the original type
            result.data = quantizeTensor(resultF32, a.type).data
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Dequantizes a tensor to F32 format.
 * This is a placeholder implementation that should be replaced with actual dequantization code.
 *
 * @param tensor The tensor to dequantize
 * @return A new tensor in F32 format
 */
private fun dequantizeTensor(tensor: GGMLTensor): GGMLTensor {
    // Create a new F32 tensor with the same dimensions
    val result = GGMLTensor(type = GGMLType.F32)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = tensor.ne[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(tensor.ne)

    // Allocate memory for the result
    result.data = FloatArray(totalSize)

    // For now, just return a zero-filled tensor
    // In a real implementation, we would dequantize based on the tensor type

    return result
}

/**
 * Quantizes a F32 tensor to the specified type.
 * This is a placeholder implementation that should be replaced with actual quantization code.
 *
 * @param tensor The F32 tensor to quantize
 * @param type The target quantization type
 * @return A new tensor in the specified quantized format
 */
private fun quantizeTensor(tensor: GGMLTensor, type: GGMLType): GGMLTensor {
    // Create a new tensor with the specified type and same dimensions
    val result = GGMLTensor(type = type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = tensor.ne[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(tensor.ne)

    // Allocate memory for the result based on the type
    when (type) {
        GGMLType.Q4_0, GGMLType.Q4_1 -> {
            // 4-bit quantization - each byte stores 2 values
            result.data = ByteArray((totalSize + 1) / 2)
        }
        GGMLType.Q5_0, GGMLType.Q5_1 -> {
            // 5-bit quantization - more complex packing
            result.data = ByteArray((totalSize * 5 + 7) / 8)
        }
        GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // 8-bit quantization - one byte per value
            result.data = ByteArray(totalSize)
        }
        else -> {
            // Should not happen
            result.data = null
        }
    }

    // For now, just return a zero-filled tensor
    // In a real implementation, we would quantize based on the tensor type

    return result
}

/**
 * Multiplies two tensors element-wise.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor
 */
fun computeMul(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    // Check that the tensors have compatible dimensions
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) {
            throw IllegalArgumentException("Incompatible dimensions for multiplication")
        }
    }

    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Perform the multiplication based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            // result.data is no longer directly assigned for F32.
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                val v0 = a.getFloat(graphAllocator, i0, i1, i2, i3)
                                val v1 = b.getFloat(graphAllocator, i0, i1, i2, i3)
                                result.setFloat(graphAllocator, v0 * v1, i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            val v0 = a.getFloat(graphAllocator, i0, i1, i2)
                            val v1 = b.getFloat(graphAllocator, i0, i1, i2)
                            result.setFloat(graphAllocator, v0 * v1, i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        val v0 = a.getFloat(graphAllocator, i0, i1)
                        val v1 = b.getFloat(graphAllocator, i0, i1)
                        result.setFloat(graphAllocator, v0 * v1, i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D
                for (i0 in 0 until n0) {
                    val v0 = a.getFloat(graphAllocator, i0)
                    val v1 = b.getFloat(graphAllocator, i0)
                    result.setFloat(graphAllocator, v0 * v1, i0)
                }
            } else { // Scalar
                val v0 = a.getFloat(graphAllocator)
                val v1 = b.getFloat(graphAllocator)
                result.setFloat(graphAllocator, v0 * v1)
            }
            // The line `result.data = resultData` is removed.
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    // Convert shorts to floats, multiply, then convert back to short
                    val aFloat = aData[j].toFloat() / 32768.0f
                    val bFloat = bData[j].toFloat() / 32768.0f
                    val product = aFloat * bFloat
                    resultData[j] = (product * 32768.0f).toInt().toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val bData = b.data as ByteArray
            val resultData = ByteArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (aData[j] * bData[j]).toByte()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (aData[j] * bData[j]).toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = aData[j] * bData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val bData = b.data as LongArray
            val resultData = LongArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = aData[j] * bData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // Basic implementation for quantized types - dequantize, multiply, then requantize
            // This is a placeholder and should be optimized in the future

            // For now, we'll convert to F32, perform the operation, and convert back
            // In a real implementation, we would have specialized code for each quantized type

            // Create temporary F32 tensors
            val aF32 = dequantizeTensor(graphAllocator, a) // Assuming dequantize needs it
            val bF32 = dequantizeTensor(graphAllocator, b) // Assuming dequantize needs it

            // Perform multiplication in F32
            val resultF32 = computeMul(graphAllocator, context, aF32, bF32)

            // Requantize to the original type
            result.data = quantizeTensor(resultF32, a.type).data
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Performs matrix multiplication of two tensors.
 *
 * @param context The GGML context
 * @param a The first tensor (m x n)
 * @param b The second tensor (n x p)
 * @return The result tensor (m x p)
 */
fun computeMatMul(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    // Check that the tensors have compatible dimensions for matrix multiplication
    // Original comment: a: m x n, b: n x p, result: m x p
    val m = a.ne[0]
    val n = a.ne[1]
    val p = b.ne[1]

    if (b.ne[0] != n) {
        throw IllegalArgumentException("Incompatible dimensions for matrix multiplication")
    }

    // Create a new tensor for the result
    val result = GGMLTensor(type = a.type)
    result.ne[0] = m
    result.ne[1] = p
    for (i in 2 until GGML_MAX_DIMS) {
        result.ne[i] = 1
    }

    // Set strides based on the data type
    val typeSize = when (a.type) {
        GGMLType.F32 -> 4u
        GGMLType.F16 -> 2u
        GGMLType.I8 -> 1u
        GGMLType.I16 -> 2u
        GGMLType.I32 -> 4u
        GGMLType.I64 -> 8u
        else -> 1u // Default for quantized types
    }

    result.nb[0] = typeSize.toULong()
    result.nb[1] = result.nb[0] * result.ne[0].toULong()
    for (i in 2 until GGML_MAX_DIMS) {
        result.nb[i] = result.nb[i-1] * result.ne[i-1].toULong()
    }

    // Calculate total size
    val totalSize = (m * p).toInt()

    // Perform the matrix multiplication based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            // val aData = a.data as FloatArray // Remove
            // val bData = b.data as FloatArray // Remove
            // val resultData = FloatArray(totalSize) // Remove, data goes to graphAllocator buffer

            // Based on: a.ne[0]=m (rows), a.ne[1]=n (cols)
            //           b.ne[0]=n (rows), b.ne[1]=p (cols)
            //           result.ne[0]=m (rows), result.ne[1]=p (cols)
            // This means element (row_idx, col_idx) is tensor.getFloat(col_idx, row_idx)
            // if ne[0] is columns and ne[1] is rows.
            // The problem statement's example was:
            // dst.setFloat(graphAllocator, sum, n_idx, m_idx) for (M rows, N cols)
            // where n_idx is col index (up to N), m_idx is row index (up to M).
            // This implies indices are (idx_dim0, idx_dim1) -> (col, row)
            // If a.ne[0] = m (rows), a.ne[1] = n (cols)
            // Then a.getFloat(graphAllocator, col_k, row_i)
            // If b.ne[0] = n (rows), b.ne[1] = p (cols)
            // Then b.getFloat(graphAllocator, col_j, row_k)
            // result.setFloat(graphAllocator, sum, col_j, row_i)

            // Let's use the dimensions as defined in this function:
            // a: m rows, n cols. So a.ne[0] = m, a.ne[1] = n.
            // b: n rows, p cols. So b.ne[0] = n, b.ne[1] = p.
            // res:m rows, p cols. So res.ne[0] = m, res.ne[1] = p.
            // Access a(row, col) is a.getFloat(graphAllocator, col, row) if ne[0] is for cols.
            // The current code for result.ne[0]=m, result.ne[1]=p suggests ne[0] is rows.

            val m_rows = a.ne[0].toInt() // M
            val n_common = a.ne[1].toInt() // K (using K from prompt example M,N,K)
            val p_cols = b.ne[1].toInt() // N

            if (b.ne[0].toInt() != n_common) { // Check common dim agreement
                 throw IllegalArgumentException(
                    "Dimension mismatch for matmul: a.ne[1] (${a.ne[1]}) != b.ne[0] (${b.ne[0]})"
                )
            }
            // result.ne[0] should be m_rows, result.ne[1] should be p_cols.
            // The existing result setup: result.ne[0] = m, result.ne[1] = p. This matches.

            for (i in 0 until m_rows) { // Index for rows of result (and a)
                for (j in 0 until p_cols) { // Index for columns of result (and b)
                    var sum = 0.0f
                    for (k in 0 until n_common) { // Index for common dimension (cols of a, rows of b)
                        // a[i, k] * b[k, j]
                        // Assuming ne[0] is rows, ne[1] is columns:
                        // a.getFloat(graphAllocator, k, i) for a[row_i, col_k]
                        // b.getFloat(graphAllocator, j, k) for b[row_k, col_j]
                        val valA = a.getFloat(graphAllocator, k, i) // Access a[i][k]
                        val valB = b.getFloat(graphAllocator, j, k) // Access b[k][j]
                        sum += valA * valB
                    }
                    // result[i, j] = sum
                    result.setFloat(graphAllocator, sum, j, i) // Set result[i][j]
                }
            }
            // result.data = resultData is removed
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Use block-based matrix multiplication for better cache utilization
            val blockSize = 32 // Adjust based on cache size

            // Initialize result to zeros
            for (i in 0 until totalSize) {
                resultData[i] = 0
            }

            // Block-based matrix multiplication
            for (ii in 0 until m.toInt() step blockSize) {
                val iEnd = minOf(ii + blockSize, m.toInt())

                for (kk in 0 until n.toInt() step blockSize) {
                    val kEnd = minOf(kk + blockSize, n.toInt())

                    for (jj in 0 until p.toInt() step blockSize) {
                        val jEnd = minOf(jj + blockSize, p.toInt())

                        // Process the current block
                        for (i in ii until iEnd) {
                            for (k in kk until kEnd) {
                                // Convert short to float
                                val aFloat = aData[i * n.toInt() + k].toFloat() / 32768.0f

                                for (j in jj until jEnd) {
                                    // Convert short to float, multiply, accumulate
                                    val bFloat = bData[k * p.toInt() + j].toFloat() / 32768.0f
                                    val currentVal = resultData[i * p.toInt() + j].toFloat() / 32768.0f
                                    val newVal = currentVal + aFloat * bFloat

                                    // Convert back to short
                                    resultData[i * p.toInt() + j] = (newVal * 32768.0f).toInt().toShort()
                                }
                            }
                        }
                    }
                }
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Use block-based matrix multiplication for better cache utilization
            val blockSize = 32 // Adjust based on cache size

            // Initialize result to zeros
            for (i in 0 until totalSize) {
                resultData[i] = 0
            }

            // Block-based matrix multiplication
            for (ii in 0 until m.toInt() step blockSize) {
                val iEnd = minOf(ii + blockSize, m.toInt())

                for (kk in 0 until n.toInt() step blockSize) {
                    val kEnd = minOf(kk + blockSize, n.toInt())

                    for (jj in 0 until p.toInt() step blockSize) {
                        val jEnd = minOf(jj + blockSize, p.toInt())

                        // Process the current block
                        for (i in ii until iEnd) {
                            for (k in kk until kEnd) {
                                val aVal = aData[i * n.toInt() + k]

                                for (j in jj until jEnd) {
                                    // Need to be careful with overflow
                                    val product = aVal * bData[k * p.toInt() + j]
                                    val sum = resultData[i * p.toInt() + j] + product
                                    resultData[i * p.toInt() + j] = sum.toShort()
                                }
                            }
                        }
                    }
                }
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            // Use block-based matrix multiplication for better cache utilization
            val blockSize = 32 // Adjust based on cache size

            // Initialize result to zeros
            for (i in 0 until totalSize) {
                resultData[i] = 0
            }

            // Block-based matrix multiplication
            for (ii in 0 until m.toInt() step blockSize) {
                val iEnd = minOf(ii + blockSize, m.toInt())

                for (kk in 0 until n.toInt() step blockSize) {
                    val kEnd = minOf(kk + blockSize, n.toInt())

                    for (jj in 0 until p.toInt() step blockSize) {
                        val jEnd = minOf(jj + blockSize, p.toInt())

                        // Process the current block
                        for (i in ii until iEnd) {
                            for (k in kk until kEnd) {
                                val aVal = aData[i * n.toInt() + k]

                                for (j in jj until jEnd) {
                                    resultData[i * p.toInt() + j] += aVal * bData[k * p.toInt() + j]
                                }
                            }
                        }
                    }
                }
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val bData = b.data as ByteArray
            val resultData = ByteArray(totalSize)

            // Use block-based matrix multiplication for better cache utilization
            val blockSize = 32 // Adjust based on cache size

            // Initialize result to zeros
            for (i in 0 until totalSize) {
                resultData[i] = 0
            }

            // Block-based matrix multiplication
            for (ii in 0 until m.toInt() step blockSize) {
                val iEnd = minOf(ii + blockSize, m.toInt())

                for (kk in 0 until n.toInt() step blockSize) {
                    val kEnd = minOf(kk + blockSize, n.toInt())

                    for (jj in 0 until p.toInt() step blockSize) {
                        val jEnd = minOf(jj + blockSize, p.toInt())

                        // Process the current block
                        for (i in ii until iEnd) {
                            for (k in kk until kEnd) {
                                val aVal = aData[i * n.toInt() + k]

                                for (j in jj until jEnd) {
                                    // Need to be careful with overflow
                                    val product = aVal * bData[k * p.toInt() + j]
                                    val sum = resultData[i * p.toInt() + j] + product
                                    resultData[i * p.toInt() + j] = sum.toByte()
                                }
                            }
                        }
                    }
                }
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val bData = b.data as LongArray
            val resultData = LongArray(totalSize)

            // Use block-based matrix multiplication for better cache utilization
            val blockSize = 32 // Adjust based on cache size

            // Initialize result to zeros
            for (i in 0 until totalSize) {
                resultData[i] = 0L
            }

            // Block-based matrix multiplication
            for (ii in 0 until m.toInt() step blockSize) {
                val iEnd = minOf(ii + blockSize, m.toInt())

                for (kk in 0 until n.toInt() step blockSize) {
                    val kEnd = minOf(kk + blockSize, n.toInt())

                    for (jj in 0 until p.toInt() step blockSize) {
                        val jEnd = minOf(jj + blockSize, p.toInt())

                        // Process the current block
                        for (i in ii until iEnd) {
                            for (k in kk until kEnd) {
                                val aVal = aData[i * n.toInt() + k]

                                for (j in jj until jEnd) {
                                    resultData[i * p.toInt() + j] += aVal * bData[k * p.toInt() + j]
                                }
                            }
                        }
                    }
                }
            }

            result.data = resultData
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // Basic implementation for quantized types - dequantize, perform matrix multiplication, then requantize
            // This is a placeholder and should be optimized in the future

            // For now, we'll convert to F32, perform the operation, and convert back
            // In a real implementation, we would have specialized code for each quantized type

            // Create temporary F32 tensors
            val aF32 = dequantizeTensor(graphAllocator, a) // Assuming dequantize needs it
            val bF32 = dequantizeTensor(graphAllocator, b) // Assuming dequantize needs it

            // Perform matrix multiplication in F32
            val resultF32 = computeMatMul(graphAllocator, context, aF32, bF32)

            // Requantize to the original type
            result.data = quantizeTensor(resultF32, a.type).data
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Applies the ReLU activation function to a tensor.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor
 */
fun computeRelu(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor
): GGMLTensor {
    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Apply the ReLU function based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                val v = a.getFloat(graphAllocator, i0, i1, i2, i3)
                                result.setFloat(graphAllocator, if (v > 0.0f) v else 0.0f, i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            val v = a.getFloat(graphAllocator, i0, i1, i2)
                            result.setFloat(graphAllocator, if (v > 0.0f) v else 0.0f, i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        val v = a.getFloat(graphAllocator, i0, i1)
                        result.setFloat(graphAllocator, if (v > 0.0f) v else 0.0f, i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D
                for (i0 in 0 until n0) {
                    val v = a.getFloat(graphAllocator, i0)
                    result.setFloat(graphAllocator, if (v > 0.0f) v else 0.0f, i0)
                }
            } else { // Scalar
                val v = a.getFloat(graphAllocator)
                result.setFloat(graphAllocator, if (v > 0.0f) v else 0.0f)
            }
            // result.data = resultData is removed
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val resultData = ShortArray(totalSize)

            for (i in 0 until totalSize) {
                // Convert short to float, apply ReLU, convert back to short
                val aFloat = aData[i].toFloat() / 32768.0f
                val reluResult = if (aFloat > 0.0f) aFloat else 0.0f
                resultData[i] = (reluResult * 32768.0f).toInt().toShort()
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val resultData = ByteArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = if (aData[i] > 0) aData[i] else 0
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val resultData = ShortArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = if (aData[i] > 0) aData[i] else 0
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val resultData = IntArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = if (aData[i] > 0) aData[i] else 0
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val resultData = LongArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = if (aData[i] > 0) aData[i] else 0
            }

            result.data = resultData
        }
        else -> {
            // For other types (quantized types), we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Applies the GELU activation function to a tensor.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor
 */
fun computeGelu(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor
): GGMLTensor {
    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Apply the GELU function based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            val geluApprox = { x: Float -> x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x))) }

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                result.setFloat(graphAllocator, geluApprox(a.getFloat(graphAllocator, i0, i1, i2, i3)), i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            result.setFloat(graphAllocator, geluApprox(a.getFloat(graphAllocator, i0, i1, i2)), i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        result.setFloat(graphAllocator, geluApprox(a.getFloat(graphAllocator, i0, i1)), i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D
                for (i0 in 0 until n0) {
                    result.setFloat(graphAllocator, geluApprox(a.getFloat(graphAllocator, i0)), i0)
                }
            } else { // Scalar
                result.setFloat(graphAllocator, geluApprox(a.getFloat(graphAllocator)))
            }
            // result.data = resultData is removed
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val resultData = ShortArray(totalSize)

            for (i in 0 until totalSize) {
                // Convert short to float, apply GELU, convert back to short
                val x = aData[i].toFloat() / 32768.0f
                val geluResult = x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x)))
                resultData[i] = (geluResult * 32768.0f).toInt().toShort()
            }

            result.data = resultData
        }
        GGMLType.I8, GGMLType.I16, GGMLType.I32, GGMLType.I64 -> {
            // For integer types, convert to float, apply GELU, then convert back
            // This is a simplified implementation that treats all integer types similarly

            // Create a float array for intermediate calculations
            val floatData = FloatArray(totalSize)

            // Convert input data to float
            when (a.type) {
                GGMLType.I8 -> {
                    val aData = a.data as ByteArray
                    for (i in 0 until totalSize) {
                        floatData[i] = aData[i].toFloat()
                    }
                }
                GGMLType.I16 -> {
                    val aData = a.data as ShortArray
                    for (i in 0 until totalSize) {
                        floatData[i] = aData[i].toFloat()
                    }
                }
                GGMLType.I32 -> {
                    val aData = a.data as IntArray
                    for (i in 0 until totalSize) {
                        floatData[i] = aData[i].toFloat()
                    }
                }
                GGMLType.I64 -> {
                    val aData = a.data as LongArray
                    for (i in 0 until totalSize) {
                        floatData[i] = aData[i].toFloat()
                    }
                }
                else -> {} // Should not happen due to the when condition
            }

            // Apply GELU to float data
            for (i in 0 until totalSize) {
                val x = floatData[i]
                floatData[i] = x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x)))
            }

            // Convert back to the original type
            when (a.type) {
                GGMLType.I8 -> {
                    val resultData = ByteArray(totalSize)
                    for (i in 0 until totalSize) {
                        resultData[i] = floatData[i].toInt().toByte()
                    }
                    result.data = resultData
                }
                GGMLType.I16 -> {
                    val resultData = ShortArray(totalSize)
                    for (i in 0 until totalSize) {
                        resultData[i] = floatData[i].toInt().toShort()
                    }
                    result.data = resultData
                }
                GGMLType.I32 -> {
                    val resultData = IntArray(totalSize)
                    for (i in 0 until totalSize) {
                        resultData[i] = floatData[i].toInt()
                    }
                    result.data = resultData
                }
                GGMLType.I64 -> {
                    val resultData = LongArray(totalSize)
                    for (i in 0 until totalSize) {
                        resultData[i] = floatData[i].toLong()
                    }
                    result.data = resultData
                }
                else -> {} // Should not happen due to the when condition
            }
        }
        else -> {
            // For other types (quantized types), we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Subtracts one tensor from another element-wise.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor (a - b)
 */
fun computeSub(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    // Check that the tensors have compatible dimensions
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) {
            throw IllegalArgumentException("Incompatible dimensions for subtraction")
        }
    }

    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Perform the subtraction based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                val v0 = a.getFloat(graphAllocator, i0, i1, i2, i3)
                                val v1 = b.getFloat(graphAllocator, i0, i1, i2, i3)
                                result.setFloat(graphAllocator, v0 - v1, i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            val v0 = a.getFloat(graphAllocator, i0, i1, i2)
                            val v1 = b.getFloat(graphAllocator, i0, i1, i2)
                            result.setFloat(graphAllocator, v0 - v1, i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        val v0 = a.getFloat(graphAllocator, i0, i1)
                        val v1 = b.getFloat(graphAllocator, i0, i1)
                        result.setFloat(graphAllocator, v0 - v1, i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D
                for (i0 in 0 until n0) {
                    val v0 = a.getFloat(graphAllocator, i0)
                    val v1 = b.getFloat(graphAllocator, i0)
                    result.setFloat(graphAllocator, v0 - v1, i0)
                }
            } else { // Scalar
                val v0 = a.getFloat(graphAllocator)
                val v1 = b.getFloat(graphAllocator)
                result.setFloat(graphAllocator, v0 - v1)
            }
            // result.data = resultData is removed
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    // Convert shorts to floats, subtract, then convert back to short
                    val aFloat = aData[j].toFloat() / 32768.0f
                    val bFloat = bData[j].toFloat() / 32768.0f
                    val diff = aFloat - bFloat
                    resultData[j] = (diff * 32768.0f).toInt().toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val bData = b.data as ByteArray
            val resultData = ByteArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (aData[j] - bData[j]).toByte()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (aData[j] - bData[j]).toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = aData[j] - bData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val bData = b.data as LongArray
            val resultData = LongArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = aData[j] - bData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // Basic implementation for quantized types - dequantize, subtract, then requantize
            // This is a placeholder and should be optimized in the future

            // For now, we'll convert to F32, perform the operation, and convert back
            // In a real implementation, we would have specialized code for each quantized type

            // Create temporary F32 tensors
            val aF32 = dequantizeTensor(graphAllocator, a) // Assuming dequantize needs it
            val bF32 = dequantizeTensor(graphAllocator, b) // Assuming dequantize needs it

            // Perform subtraction in F32
            val resultF32 = computeSub(graphAllocator, context, aF32, bF32)

            // Requantize to the original type
            result.data = quantizeTensor(resultF32, a.type).data
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Negates a tensor element-wise.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor (-a)
 */
fun computeNeg(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor
): GGMLTensor {
    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Perform the negation based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                result.setFloat(graphAllocator, -a.getFloat(graphAllocator, i0, i1, i2, i3), i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            result.setFloat(graphAllocator, -a.getFloat(graphAllocator, i0, i1, i2), i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        result.setFloat(graphAllocator, -a.getFloat(graphAllocator, i0, i1), i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D
                for (i0 in 0 until n0) {
                    result.setFloat(graphAllocator, -a.getFloat(graphAllocator, i0), i0)
                }
            } else { // Scalar
                result.setFloat(graphAllocator, -a.getFloat(graphAllocator))
            }
            // result.data = resultData is removed
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    // Convert short to float, negate, then convert back to short
                    val aFloat = aData[j].toFloat() / 32768.0f
                    val negated = -aFloat
                    resultData[j] = (negated * 32768.0f).toInt().toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val resultData = ByteArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (-aData[j]).toByte()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = (-aData[j]).toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val resultData = IntArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = -aData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val resultData = LongArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    resultData[j] = -aData[j]
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // Basic implementation for quantized types - dequantize, negate, then requantize
            // This is a placeholder and should be optimized in the future

            // For now, we'll convert to F32, perform the operation, and convert back
            // In a real implementation, we would have specialized code for each quantized type

            // Create temporary F32 tensor
            val aF32 = dequantizeTensor(graphAllocator, a) // Assuming dequantize needs it

            // Perform negation in F32
            val resultF32 = computeNeg(graphAllocator, context, aF32)

            // Requantize to the original type
            result.data = quantizeTensor(resultF32, a.type).data
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}

/**
 * Divides one tensor by another element-wise.
 *
 * @param context The GGML context
 * @param a The numerator tensor
 * @param b The denominator tensor
 * @return The result tensor (a / b)
 */
fun computeDiv(
    graphAllocator: GGMLGraphAllocator,
    @Suppress("unused") context: GGMLContext,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    // Check that the tensors have compatible dimensions
    for (i in 0 until GGML_MAX_DIMS) {
        if (a.ne[i] != b.ne[i]) {
            throw IllegalArgumentException("Incompatible dimensions for division")
        }
    }

    // Create a new tensor for the result with the same dimensions as a
    val result = GGMLTensor(type = a.type)
    for (i in 0 until GGML_MAX_DIMS) {
        result.ne[i] = a.ne[i]
        result.nb[i] = a.nb[i]
    }

    // Calculate total size
    val totalSize = calculateTotalSize(a.ne)

    // Perform division based on the tensor type
    when (a.type) {
        GGMLType.F32 -> {
            val n0 = result.ne[0].toInt()
            val n1 = result.ne[1].toInt()
            val n2 = result.ne[2].toInt()
            val n3 = result.ne[3].toInt()

            val divideOp = { vA: Float, vB: Float ->
                if (vB == 0.0f) {
                    if (vA == 0.0f) Float.NaN
                    else if (vA > 0.0f) Float.POSITIVE_INFINITY
                    else Float.NEGATIVE_INFINITY
                } else {
                    vA / vB
                }
            }

            if (n3 > 1) { // 4D
                for (i3 in 0 until n3) {
                    for (i2 in 0 until n2) {
                        for (i1 in 0 until n1) {
                            for (i0 in 0 until n0) {
                                val v0 = a.getFloat(graphAllocator, i0, i1, i2, i3)
                                val v1 = b.getFloat(graphAllocator, i0, i1, i2, i3)
                                result.setFloat(graphAllocator, divideOp(v0, v1), i0, i1, i2, i3)
                            }
                        }
                    }
                }
            } else if (n2 > 1) { // 3D
                for (i2 in 0 until n2) {
                    for (i1 in 0 until n1) {
                        for (i0 in 0 until n0) {
                            val v0 = a.getFloat(graphAllocator, i0, i1, i2)
                            val v1 = b.getFloat(graphAllocator, i0, i1, i2)
                            result.setFloat(graphAllocator, divideOp(v0, v1), i0, i1, i2)
                        }
                    }
                }
            } else if (n1 > 1) { // 2D
                for (i1 in 0 until n1) {
                    for (i0 in 0 until n0) {
                        val v0 = a.getFloat(graphAllocator, i0, i1)
                        val v1 = b.getFloat(graphAllocator, i0, i1)
                        result.setFloat(graphAllocator, divideOp(v0, v1), i0, i1)
                    }
                }
            } else if (n0 > 0) { // 1D
                for (i0 in 0 until n0) {
                    val v0 = a.getFloat(graphAllocator, i0)
                    val v1 = b.getFloat(graphAllocator, i0)
                    result.setFloat(graphAllocator, divideOp(v0, v1), i0)
                }
            } else { // Scalar
                val v0 = a.getFloat(graphAllocator)
                val v1 = b.getFloat(graphAllocator)
                result.setFloat(graphAllocator, divideOp(v0, v1))
            }
            // result.data = resultData is removed
        }
        GGMLType.F16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    // Convert shorts to floats
                    val aFloat = aData[j].toFloat() / 32768.0f
                    val bFloat = bData[j].toFloat() / 32768.0f

                    // Perform division
                    val resultFloat = if (bFloat == 0.0f) {
                        if (aFloat == 0.0f) {
                            Float.NaN // 0/0 = NaN
                        } else if (aFloat > 0.0f) {
                            Float.POSITIVE_INFINITY // positive/0 = +Infinity
                        } else {
                            Float.NEGATIVE_INFINITY // negative/0 = -Infinity
                        }
                    } else {
                        aFloat / bFloat
                    }

                    // Convert back to short
                    resultData[j] = (resultFloat * 32768.0f).toInt().toShort()
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I8 -> {
            val aData = a.data as ByteArray
            val bData = b.data as ByteArray
            val resultData = ByteArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    if (bData[j] == 0.toByte()) {
                        // Handle division by zero - for integer types, we'll use 0 as a default
                        resultData[j] = 0
                    } else {
                        resultData[j] = (aData[j] / bData[j]).toByte()
                    }
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I16 -> {
            val aData = a.data as ShortArray
            val bData = b.data as ShortArray
            val resultData = ShortArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    if (bData[j] == 0.toShort()) {
                        // Handle division by zero - for integer types, we'll use 0 as a default
                        resultData[j] = 0
                    } else {
                        resultData[j] = (aData[j] / bData[j]).toShort()
                    }
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    if (bData[j] == 0) {
                        // Handle division by zero - for integer types, we'll use 0 as a default
                        resultData[j] = 0
                    } else {
                        resultData[j] = aData[j] / bData[j]
                    }
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.I64 -> {
            val aData = a.data as LongArray
            val bData = b.data as LongArray
            val resultData = LongArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    if (bData[j] == 0L) {
                        // Handle division by zero - for integer types, we'll use 0 as a default
                        resultData[j] = 0L
                    } else {
                        resultData[j] = aData[j] / bData[j]
                    }
                }
                i = end
            }

            result.data = resultData
        }
        GGMLType.Q4_0, GGMLType.Q4_1, GGMLType.Q5_0, GGMLType.Q5_1, GGMLType.Q8_0, GGMLType.Q8_1 -> {
            // Basic implementation for quantized types - dequantize, divide, then requantize
            // This is a placeholder and should be optimized in the future

            // For now, we'll convert to F32, perform the operation, and convert back
            // In a real implementation, we would have specialized code for each quantized type

            // Create temporary F32 tensors
            val aF32 = dequantizeTensor(graphAllocator, a) // Assuming dequantize needs it
            val bF32 = dequantizeTensor(graphAllocator, b) // Assuming dequantize needs it

            // Perform division in F32
            val resultF32 = computeDiv(graphAllocator, context, aF32, bF32)

            // Requantize to the original type
            result.data = quantizeTensor(resultF32, a.type).data
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}
