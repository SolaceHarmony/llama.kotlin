package ai.solace.llamakotlin.core

internal fun calculateContiguousStrides(ne: LongArray, type: GGMLType, rank: Int): ULongArray {
    val nb = ULongArray(GGML_MAX_DIMS) { 0uL } // GGML_MAX_DIMS should be accessible

    if (type.byteSize == 0uL) {
        // Existing warning logic (adapted from previous subtask reports for quantizeTensor)
        if (type != GGMLType.COUNT && !type.name.startsWith("Q", ignoreCase = true) && !type.name.startsWith("q", ignoreCase = true) ) {
            println("Warning: GGMLType ${type.name} has byteSize 0. Strides will be all zeros.")
        }
        return nb // Return zeroed strides
    }

    nb[0] = type.byteSize
    if (GGML_MAX_DIMS > 1) {
        for (d in 1 until GGML_MAX_DIMS) {
            // ne is 0-indexed for dimensions. ne[0] is size of dim 0, ne[1] of dim 1, etc.
            // nb[d] is stride for dimension d.
            // nb[1] (stride for dim 1) = ne[0] * nb[0] (size of dim 0 * element size)
            // nb[d] = ne[d-1] * nb[d-1]
            // Use ne.getOrElse to handle cases where rank is less than d.
            // If rank < d, effectively ne[d-1] is 1 for stride calculation purposes beyond actual rank.
            val dimSize = ne.getOrElse(d - 1) { 1L }
            nb[d] = nb[d - 1] * (if (dimSize > 0L) dimSize.toULong() else 1uL) // Ensure positive dimSize for multiplication
        }
    }
    return nb
}

/**
 * Kotlin Native port of GGML tensor operations.
 * This file contains the implementation of basic tensor operations.
 *
 * This is a placeholder implementation that will be expanded in future versions.
 */

/**
 * Creates a new tensor with the specified dimensions and type.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @return The new tensor
 */
@Suppress("unused")
fun createTensor(context: GGMLContext, type: GGMLType): GGMLTensor {
    val tensor = GGMLTensor(type = type)

    // Set default dimensions to 1
    tensor.ne[0] = 1
    for (i in 1 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    // Set default strides based on the data type
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    // Allocate memory for the tensor if context is provided
    if (context.memBuffer != null && !context.noAlloc) {
        // In a real implementation, we would allocate memory from the context
        // For now, we'll just create an empty buffer based on the tensor type
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(1) { 0.0f }
            GGMLType.I32 -> tensor.data = IntArray(1) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(1) { 0L }
            else -> tensor.data = null // For other types, we'll implement later
        }
    }

    return tensor
}

/**
 * Creates a new 1-dimensional tensor.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @return The new tensor
 */
@Suppress("unused")
fun createTensor1D(context: GGMLContext, type: GGMLType, ne0: Int): GGMLTensor {
    // Create a new tensor with the specified type
    val tensor = GGMLTensor(type = type)

    // Set the dimensions
    tensor.ne[0] = ne0.toLong()
    for (i in 1 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    // Set strides based on the data type
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    // Allocate memory for the tensor if context is provided
    if (context.memBuffer != null && !context.noAlloc) {
        // Calculate total size
        val totalSize = ne0

        // Allocate memory based on the tensor type
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null // For quantized types, we'll implement later
        }
    }

    return tensor
}

/**
 * Creates a new 2-dimensional tensor.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @return The new tensor
 */
fun createTensor2D(context: GGMLContext, type: GGMLType, ne0: Int, ne1: Int): GGMLTensor {
    // Create a new tensor with the specified type
    val tensor = GGMLTensor(type = type)

    // Set the dimensions
    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    for (i in 2 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    // Set strides based on the data type
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    // Allocate memory for the tensor if context is provided
    if (context.memBuffer != null && !context.noAlloc) {
        // Calculate total size
        val totalSize = ne0 * ne1

        // Allocate memory based on the tensor type
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null // For quantized types, we'll implement later
        }
    }

    return tensor
}

/**
 * Adds two tensors element-wise.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor
 */
fun add(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    // Set up the operation node
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.ADD
    result.src[0] = a
    result.src[1] = b

    // If the context requests immediate computation, perform it now
    return if (context.computeImmediately) {
        computeAdd(context, a, b)
    } else {
        result
    }
}

/**
 * Multiplies two tensors element-wise.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor
 */
fun mul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.MUL
    result.src[0] = a
    result.src[1] = b

    return if (context.computeImmediately) {
        computeMul(context, a, b)
    } else {
        result
    }
}

/**
 * Performs matrix multiplication of two tensors.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor
 */
fun matMul(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.MUL_MAT
    result.src[0] = a
    result.src[1] = b

    return if (context.computeImmediately) {
        computeMatMul(context, a, b)
    } else {
        result
    }
}

/**
 * Subtracts one tensor from another element-wise.
 *
 * @param context The GGML context
 * @param a The first tensor
 * @param b The second tensor
 * @return The result tensor (a - b)
 */
fun sub(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.SUB
    result.src[0] = a
    result.src[1] = b

    return if (context.computeImmediately) {
        computeSub(context, a, b)
    } else {
        result
    }
}

/**
 * Negates a tensor element-wise.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor (-a)
 */
fun neg(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.NEG
    result.src[0] = a

    return if (context.computeImmediately) {
        computeNeg(context, a)
    } else {
        result
    }
}

/**
 * Applies the ReLU activation function to a tensor.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor
 */
fun relu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.RELU
    result.src[0] = a

    return if (context.computeImmediately) {
        computeRelu(context, a)
    } else {
        result
    }
}

/**
 * Applies the GELU activation function to a tensor.
 *
 * @param context The GGML context
 * @param a The input tensor
 * @return The result tensor
 */
fun gelu(context: GGMLContext, a: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.GELU
    result.src[0] = a

    return if (context.computeImmediately) {
        computeGelu(context, a)
    } else {
        result
    }
}

/**
 * Divides one tensor by another element-wise.
 *
 * @param context The GGML context
 * @param a The numerator tensor
 * @param b The denominator tensor
 * @return The result tensor (a / b)
 */
fun div(context: GGMLContext, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.op = GGMLOp.DIV
    result.src[0] = a
    result.src[1] = b

    return if (context.computeImmediately) {
        computeDiv(context, a, b)
    } else {
        result
    }
}

/**
 * Creates a new 3D tensor with the specified type and dimensions.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @param ne2 The number of elements in the third dimension
 * @return The new tensor
 */
fun createTensor3D(context: GGMLContext, type: GGMLType, ne0: Int, ne1: Int, ne2: Int): GGMLTensor {
    // Create a new tensor with the specified type
    val tensor = GGMLTensor(type = type)

    // Set the dimensions
    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    tensor.ne[2] = ne2.toLong()
    for (i in 3 until GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    // Set strides based on the data type
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    // Allocate memory for the tensor if context is provided
    if (context.memBuffer != null && !context.noAlloc) {
        // Calculate total size
        val totalSize = ne0 * ne1 * ne2

        // Allocate memory based on the tensor type
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null // For quantized types, we'll implement later
        }
    }

    return tensor
}

/**
 * Creates a new 4D tensor with the specified type and dimensions.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @param ne2 The number of elements in the third dimension  
 * @param ne3 The number of elements in the fourth dimension
 * @return The new tensor
 */
fun createTensor4D(context: GGMLContext, type: GGMLType, ne0: Int, ne1: Int, ne2: Int, ne3: Int): GGMLTensor {
    // Create a new tensor with the specified type
    val tensor = GGMLTensor(type = type)

    // Set the dimensions
    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    tensor.ne[2] = ne2.toLong()
    tensor.ne[3] = ne3.toLong()

    // Set strides based on the data type
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    // Allocate memory for the tensor if context is provided
    if (context.memBuffer != null && !context.noAlloc) {
        // Calculate total size
        val totalSize = ne0 * ne1 * ne2 * ne3

        // Allocate memory based on the tensor type
        when (type) {
            GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null // For quantized types, we'll implement later
        }
    }

    return tensor
}
