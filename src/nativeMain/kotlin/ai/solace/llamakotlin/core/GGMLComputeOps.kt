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
fun computeAdd(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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

            for (i in 0 until totalSize) {
                resultData[i] = aData[i] + bData[i]
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = aData[i] + bData[i]
            }

            result.data = resultData
        }
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

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
fun computeMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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

            for (i in 0 until totalSize) {
                resultData[i] = aData[i] * bData[i]
            }

            result.data = resultData
        }
        GGMLType.I32 -> {
            val aData = a.data as IntArray
            val bData = b.data as IntArray
            val resultData = IntArray(totalSize)

            for (i in 0 until totalSize) {
                resultData[i] = aData[i] * bData[i]
            }

            result.data = resultData
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
fun computeMatMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
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

    result.nb[0] = typeSize
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

            for (i in 0 until m.toInt()) {
                for (j in 0 until p.toInt()) {
                    var sum = 0.0f
                    for (k in 0 until n.toInt()) {
                        sum += aData[i * n.toInt() + k] * bData[k * p.toInt() + j]
                    }
                    resultData[i * p.toInt() + j] = sum
                }
            }

            result.data = resultData
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
fun computeRelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
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
        else -> {
            // For other types, we'll implement later
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
fun computeGelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
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
        else -> {
            // For other types, we'll implement later
            result.data = null
        }
    }

    return result
}
