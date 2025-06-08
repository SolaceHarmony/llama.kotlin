package ai.solace.llamakotlin.core

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
fun computeAdd(@Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val resultData = FloatArray(totalSize)

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
            val aF32 = dequantizeTensor(a)
            val bF32 = dequantizeTensor(b)

            // Perform addition in F32
            val resultF32 = computeAdd(context, aF32, bF32)

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
fun computeMul(@Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val resultData = FloatArray(totalSize)

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
            val aF32 = dequantizeTensor(a)
            val bF32 = dequantizeTensor(b)

            // Perform multiplication in F32
            val resultF32 = computeMul(context, aF32, bF32)

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
fun computeMatMul(@Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    // Check that the tensors have compatible dimensions for matrix multiplication
    // a: m x n, b: n x p, result: m x p
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
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val resultData = FloatArray(totalSize)

            // Use block-based matrix multiplication for better cache utilization
            val blockSize = 32 // Adjust based on cache size

            // Initialize result to zeros
            for (i in 0 until totalSize) {
                resultData[i] = 0.0f
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
            val aF32 = dequantizeTensor(a)
            val bF32 = dequantizeTensor(b)

            // Perform matrix multiplication in F32
            val resultF32 = computeMatMul(context, aF32, bF32)

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
fun computeRelu(@Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val resultData = FloatArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = if (aData[i] > 0.0f) aData[i] else 0.0f
            }

            result.data = resultData
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
fun computeGelu(@Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val resultData = FloatArray(totalSize)

            for (i in 0 until totalSize) {
                // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                val x = aData[i]
                resultData[i] = x * 0.5f * (1.0f + kotlin.math.tanh(0.797885f * (x + 0.044715f * x * x * x)))
            }

            result.data = resultData
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
fun computeSub(@Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val resultData = FloatArray(totalSize)

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
            val aF32 = dequantizeTensor(a)
            val bF32 = dequantizeTensor(b)

            // Perform subtraction in F32
            val resultF32 = computeSub(context, aF32, bF32)

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
fun computeNeg(@Suppress("unused") context: GGMLContext, a: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val resultData = FloatArray(totalSize)

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
            val aF32 = dequantizeTensor(a)

            // Perform negation in F32
            val resultF32 = computeNeg(context, aF32)

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
fun computeDiv(@Suppress("unused") context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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
            val aData = a.data as FloatArray
            val bData = b.data as FloatArray
            val resultData = FloatArray(totalSize)

            // Process in chunks for better cache utilization
            val chunkSize = 128
            var i = 0
            while (i < totalSize) {
                val end = minOf(i + chunkSize, totalSize)
                for (j in i until end) {
                    if (bData[j] == 0.0f) {
                        // Handle division by zero
                        resultData[j] = if (aData[j] == 0.0f) {
                            Float.NaN // 0/0 = NaN
                        } else if (aData[j] > 0.0f) {
                            Float.POSITIVE_INFINITY // positive/0 = +Infinity
                        } else {
                            Float.NEGATIVE_INFINITY // negative/0 = -Infinity
                        }
                    } else {
                        resultData[j] = aData[j] / bData[j]
                    }
                }
                i = end
            }

            result.data = resultData
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
            val aF32 = dequantizeTensor(a)
            val bF32 = dequantizeTensor(b)

            // Perform division in F32
            val resultF32 = computeDiv(context, aF32, bF32)

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
