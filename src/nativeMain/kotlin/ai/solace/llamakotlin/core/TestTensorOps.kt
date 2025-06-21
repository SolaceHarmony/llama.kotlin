package ai.solace.llamakotlin.core

/**
 * Test file for tensor operations.
 * This file contains tests for the optimized tensor operations.
 */

fun main() {
    println("Testing optimized tensor operations")

    // Create a context
    val graphAllocator = GGMLGraphAllocator()
    val context = GGMLContext(
        memSize = (16 * 1024 * 1024).toULong(), // 16 MB
        memBuffer = null, // Not used by current allocator model
        memBufferOwned = false,
        noAlloc = false, // Should be true if graphAllocator handles all allocations
        computeImmediately = true, // This will trigger execution through the graph
        graphAllocator = graphAllocator,
        backend = ai.solace.llamakotlin.backends.cpu.CPUBackend // Set the CPU backend
    )

    // Test add operation
    testAdd(context) // context now contains graphAllocator and backend

    // Test mul operation
    testMul(context)

    // Test matMul operation
    testMatMul(context)

    // Test Q4_K x F32 matMul operation
    testMatMulQ4KF32(context)

    // Test SiLU operations
    testComputeSiluF32(context)
    testComputeSiluF16(context)

    // Test RMSNorm operations
    testComputeRMSNormF32(context)
    testComputeRMSNormF16(context)

    // Test Integer Neg operations
    testComputeNegI8(context)
    testComputeNegI16(context)
    testComputeNegI32(context)
    testComputeNegI64(context)

    // Test Integer ReLU operations
    testComputeReluI8(context)
    testComputeReluI16(context)
    testComputeReluI32(context)
    testComputeReluI64(context)

    // Test exceptions for float-only ops with Int inputs
    testComputeGeluThrowsOnInt(context)
    testComputeSiluThrowsOnInt(context)
    testComputeRMSNormThrowsOnInt(context)


    println("All tests completed successfully")
}


fun testComputeNegI8(context: GGMLContext) {
    println("\nTesting computeNeg I8:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = byteArrayOf(0, 1, -1, 10, -20, Byte.MAX_VALUE, Byte.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I8, inputData.size, inputData)
    val resultTensor = computeNeg(graphAllocator, context, tensorA)
    val expectedData = inputData.map { b -> (-b).toByte() }.toByteArray()
    val resultData = ByteArray(inputData.size) { i -> resultTensor.getByte(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeNeg I8 test passed.") else println("computeNeg I8 test FAILED.")
}

fun testComputeNegI16(context: GGMLContext) {
    println("\nTesting computeNeg I16:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = shortArrayOf(0, 1, -1, 100, -200, Short.MAX_VALUE, Short.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I16, inputData.size, inputData)
    val resultTensor = computeNeg(graphAllocator, context, tensorA)
    val expectedData = inputData.map { s -> (-s).toShort() }.toShortArray()
    val resultData = ShortArray(inputData.size) { i -> resultTensor.getShort(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeNeg I16 test passed.") else println("computeNeg I16 test FAILED.")
}

fun testComputeNegI32(context: GGMLContext) {
    println("\nTesting computeNeg I32:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = intArrayOf(0, 1, -1, 1000, -2000, Int.MAX_VALUE, Int.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I32, inputData.size, inputData)
    val resultTensor = computeNeg(graphAllocator, context, tensorA)
    val expectedData = inputData.map { i -> -i }.toIntArray()
    val resultData = IntArray(inputData.size) { i -> resultTensor.getInt(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeNeg I32 test passed.") else println("computeNeg I32 test FAILED.")
}

fun testComputeNegI64(context: GGMLContext) {
    println("\nTesting computeNeg I64:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = longArrayOf(0L, 1L, -1L, 10000L, -20000L, Long.MAX_VALUE, Long.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I64, inputData.size, inputData)
    val resultTensor = computeNeg(graphAllocator, context, tensorA)
    val expectedData = inputData.map { l -> -l }.toLongArray()
    val resultData = LongArray(inputData.size) { i -> resultTensor.getLong(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeNeg I64 test passed.") else println("computeNeg I64 test FAILED.")
}

fun testComputeReluI8(context: GGMLContext) {
    println("\nTesting computeRelu I8:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = byteArrayOf(0, 1, -1, 10, -20, Byte.MAX_VALUE, Byte.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I8, inputData.size, inputData)
    val resultTensor = computeRelu(graphAllocator, context, tensorA)
    val expectedData = inputData.map { b -> if (b < 0) 0.toByte() else b }.toByteArray()
    val resultData = ByteArray(inputData.size) { i -> resultTensor.getByte(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeRelu I8 test passed.") else println("computeRelu I8 test FAILED.")
}

fun testComputeReluI16(context: GGMLContext) {
    println("\nTesting computeRelu I16:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = shortArrayOf(0, 1, -1, 100, -200, Short.MAX_VALUE, Short.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I16, inputData.size, inputData)
    val resultTensor = computeRelu(graphAllocator, context, tensorA)
    val expectedData = inputData.map { s -> if (s < 0) 0.toShort() else s }.toShortArray()
    val resultData = ShortArray(inputData.size) { i -> resultTensor.getShort(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeRelu I16 test passed.") else println("computeRelu I16 test FAILED.")
}

fun testComputeReluI32(context: GGMLContext) {
    println("\nTesting computeRelu I32:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = intArrayOf(0, 1, -1, 1000, -2000, Int.MAX_VALUE, Int.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I32, inputData.size, inputData)
    val resultTensor = computeRelu(graphAllocator, context, tensorA)
    val expectedData = inputData.map { i -> if (i < 0) 0 else i }.toIntArray()
    val resultData = IntArray(inputData.size) { i -> resultTensor.getInt(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeRelu I32 test passed.") else println("computeRelu I32 test FAILED.")
}

fun testComputeReluI64(context: GGMLContext) {
    println("\nTesting computeRelu I64:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = longArrayOf(0L, 1L, -1L, 10000L, -20000L, Long.MAX_VALUE, Long.MIN_VALUE)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I64, inputData.size, inputData)
    val resultTensor = computeRelu(graphAllocator, context, tensorA)
    val expectedData = inputData.map { l -> if (l < 0) 0L else l }.toLongArray()
    val resultData = LongArray(inputData.size) { i -> resultTensor.getLong(graphAllocator, i) }
    var success = expectedData.contentEquals(resultData)
    if (!success) {
        println("Input:    ${inputData.joinToString()}")
        println("Expected: ${expectedData.joinToString()}")
        println("Got:      ${resultData.joinToString()}")
    }
    if (success) println("computeRelu I64 test passed.") else println("computeRelu I64 test FAILED.")
}

inline fun <reified T: Throwable> assertFailsWith(block: () -> Unit, message: String) {
    try {
        block()
        println("$message: FAILED (expected exception ${T::class.simpleName}, but none was thrown)")
    } catch (e: Throwable) {
        if (e is T) {
            println("$message: PASSED (caught ${e::class.simpleName})")
        } else {
            println("$message: FAILED (expected ${T::class.simpleName}, but caught ${e::class.simpleName})")
            throw e // Re-throw if it's an unexpected error
        }
    }
}

fun testComputeGeluThrowsOnInt(context: GGMLContext) {
    println("\nTesting computeGelu throws on Int:")
    val graphAllocator = GGMLGraphAllocator()
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I32, 1, intArrayOf(1))
    assertFailsWith<NotImplementedError>( { computeGelu(graphAllocator, context, tensorA) }, "computeGelu with I32")
}

fun testComputeSiluThrowsOnInt(context: GGMLContext) {
    println("\nTesting computeSilu throws on Int:")
    val graphAllocator = GGMLGraphAllocator()
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I32, 1, intArrayOf(1))
    assertFailsWith<NotImplementedError>( { computeSilu(graphAllocator, context, tensorA) }, "computeSilu with I32")
}

fun testComputeRMSNormThrowsOnInt(context: GGMLContext) {
    println("\nTesting computeRMSNorm throws on Int:")
    val graphAllocator = GGMLGraphAllocator()
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.I32, 1, intArrayOf(1))
    assertFailsWith<NotImplementedError>( { computeRMSNorm(graphAllocator, context, tensorA, 1e-5f) }, "computeRMSNorm with I32")
}


// Helper to manually create a 1D tensor for testing compute functions
// Note: In real graph, tensors are created via GGMLContext and graph building.
// This helper is simplified for testing isolated compute functions.
// It assumes the tensor data is directly usable by the compute functions.
fun createTestTensor1D(
    graphAllocator: GGMLGraphAllocator,
    type: GGMLType,
    size: Int,
    data: Any? = null // FloatArray for F32, ShortArray for F16
): GGMLTensor {
    val tensor = GGMLTensor(type)
    tensor.ne = longArrayOf(size.toLong(), 1L, 1L, 1L) // Represent as N x 1

    // Simplified stride calculation for 1D (or N x 1)
    tensor.nb[0] = type.byteSize
    tensor.nb[1] = size.toULong() * type.byteSize
    for (i in 2 until GGML_MAX_DIMS) {
        tensor.nb[i] = tensor.nb[i-1] // Not strictly correct for higher dims but ok for N x 1
    }

    if (data != null) {
        val bufferBytes = when (type) {
            GGMLType.F32 -> {
                val floatData = data as FloatArray
                ByteArray(floatData.size * Float.SIZE_BYTES).apply {
                    floatData.forEachIndexed { idx, value -> setFloatLe(idx * Float.SIZE_BYTES, value) }
                }
            }
            GGMLType.F16 -> {
                val shortData = data as ShortArray
                ByteArray(shortData.size * Short.SIZE_BYTES).apply {
                    shortData.forEachIndexed { idx, value -> setShortLe(idx * Short.SIZE_BYTES, value) }
                }
            }
            else -> throw IllegalArgumentException("Unsupported type for createTestTensor1D data")
        }
        tensor.bufferId = graphAllocator.allocateBuffer(bufferBytes)
        tensor.dataOffset = 0uL
        // Assigning raw data directly to tensor.data is done by compute ops for their results.
        // For input tensors to compute ops, they expect bufferId and dataOffset to be valid for graphAllocator.
        // So, we store the data in graphAllocator.
    } else {
        // Allocate zeroed buffer if no data provided
        val bufferSize = size * type.byteSize.toInt()
        tensor.bufferId = graphAllocator.allocateBuffer(ByteArray(bufferSize))
        tensor.dataOffset = 0uL
    }
    return tensor
}


fun testComputeSiluF32(context: GGMLContext) {
    println("\nTesting computeSilu F32:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f)
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.F32, inputData.size, inputData)

    val resultTensor = computeSilu(graphAllocator, context, tensorA)

    val expectedData = inputData.map { x -> x * (1.0f / (1.0f + kotlin.math.exp(-x))) }.toFloatArray()
    val resultData = FloatArray(inputData.size) { i -> resultTensor.getFloat(graphAllocator, i) } // Assuming result is flat

    var success = true
    for (i in expectedData.indices) {
        if (abs(expectedData[i] - resultData[i]) > 1e-6f) {
            println("Error at index $i: Expected ${expectedData[i]}, Got ${resultData[i]}")
            success = false
        }
    }
    if (success) println("computeSilu F32 test passed.") else println("computeSilu F32 test FAILED.")
}

fun testComputeSiluF16(context: GGMLContext) {
    println("\nTesting computeSilu F16:")
    val graphAllocator = GGMLGraphAllocator()
    val inputFloatData = floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f)
    val inputShortData = inputFloatData.map { floatToHalf(it) }.toShortArray()
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.F16, inputShortData.size, inputShortData)

    val resultTensor = computeSilu(graphAllocator, context, tensorA)

    val expectedFloatData = inputFloatData.map { x -> x * (1.0f / (1.0f + kotlin.math.exp(-x))) }
    val resultFloatData = FloatArray(inputShortData.size) { i -> resultTensor.getHalf(graphAllocator, i) }

    var success = true
    for (i in expectedFloatData.indices) {
        if (abs(expectedFloatData[i] - resultFloatData[i]) > 0.001f) { // Higher tolerance for F16
            println("Error at index $i: Expected ${expectedFloatData[i]}, Got ${resultFloatData[i]}")
            success = false
        }
    }
    if (success) println("computeSilu F16 test passed.") else println("computeSilu F16 test FAILED.")
}

fun testComputeRMSNormF32(context: GGMLContext) {
    println("\nTesting computeRMSNorm F32:")
    val graphAllocator = GGMLGraphAllocator()
    val inputData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
    val eps = 1e-5f
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.F32, inputData.size, inputData)

    val resultTensor = computeRMSNorm(graphAllocator, context, tensorA, eps)

    var sumSq = 0.0
    inputData.forEach { sumSq += it.toDouble() * it.toDouble() }
    val meanSq = sumSq / inputData.size
    val rms = kotlin.math.sqrt(meanSq + eps.toDouble()).toFloat()
    val expectedData = inputData.map { x -> x / rms }.toFloatArray()
    val resultData = FloatArray(inputData.size) { i -> resultTensor.getFloat(graphAllocator, i) }

    var success = true
    for (i in expectedData.indices) {
        if (abs(expectedData[i] - resultData[i]) > 1e-6f) {
            println("Error at index $i: Expected ${expectedData[i]}, Got ${resultData[i]}")
            success = false
        }
    }
    if (success) println("computeRMSNorm F32 test passed.") else println("computeRMSNorm F32 test FAILED.")
}

fun testComputeRMSNormF16(context: GGMLContext) {
    println("\nTesting computeRMSNorm F16:")
    val graphAllocator = GGMLGraphAllocator()
    val inputFloatData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
    val inputShortData = inputFloatData.map { floatToHalf(it) }.toShortArray()
    val eps = 1e-5f
    val tensorA = createTestTensor1D(graphAllocator, GGMLType.F16, inputShortData.size, inputShortData)

    val resultTensor = computeRMSNorm(graphAllocator, context, tensorA, eps)

    var sumSq = 0.0
    inputFloatData.forEach { sumSq += it.toDouble() * it.toDouble() } // Use original floats for ref calc precision
    val meanSq = sumSq / inputFloatData.size
    val rms = kotlin.math.sqrt(meanSq + eps.toDouble()).toFloat()
    val expectedFloatData = inputFloatData.map { x -> x / rms }
    val resultFloatData = FloatArray(inputShortData.size) { i -> resultTensor.getHalf(graphAllocator, i) }

    var success = true
    for (i in expectedFloatData.indices) {
        if (abs(expectedFloatData[i] - resultFloatData[i]) > 0.001f) { // Higher tolerance for F16
            println("Error at index $i: Expected ${expectedFloatData[i]}, Got ${resultFloatData[i]}")
            success = false
        }
    }
    if (success) println("computeRMSNorm F16 test passed.") else println("computeRMSNorm F16 test FAILED.")
}


@OptIn(ExperimentalUnsignedTypes::class) // For quantizeBlockQ4K and dequantizeBlockQ4K
fun testMatMulQ4KF32(context: GGMLContext) {
    println("\nTesting Q4_K x F32 MatMul operation:")
    val graphAllocator = GGMLGraphAllocator() // Create a graph allocator instance

    // Define matrix dimensions
    val M = 2
    val K = QK4_K * 1 // K must be a multiple of QK4_K (256). Let K = 256 for simplicity.
    val N = 3

    // 1. Create and populate Q4_K tensor 'a' (M x K)
    val tensorA_Q4K = GGMLTensor(GGMLType.Q4_K)
    tensorA_Q4K.ne = longArrayOf(K.toLong(), M.toLong(), 1L, 1L)
    // Strides for Q4_K are tricky if thinking element-wise. GGML handles data pointer + block access.
    // For this test, we only need tensorA_Q4K.data to be correctly populated with block bytes.
    // tensorA_Q4K.nb will be set up by GGMLGraphBuilder in real use.

    val numElementsA = M * K
    require(numElementsA % QK4_K == 0) { "Total elements in A must be multiple of QK4_K" }
    val numBlocksA = numElementsA / QK4_K
    val tensorADataBytes = ByteArray(numBlocksA * GGMLType.Q4_K.byteSize.toInt())

    // Create reference F32 data for tensor A then quantize it
    val tensorA_RefF32Data = FloatArray(numElementsA) { i -> (i % 100 + 1) * 0.01f } // Sample data

    for (blockIdx in 0 until numBlocksA) {
        val blockFloats = tensorA_RefF32Data.copyOfRange(blockIdx * QK4_K, (blockIdx + 1) * QK4_K)
        val qBlockBytes = quantizeBlockQ4K(blockFloats) // From GGMLQuants.kt
        qBlockBytes.copyInto(tensorADataBytes, destinationOffset = blockIdx * GGMLType.Q4_K.byteSize.toInt())
    }
    // In a real scenario, GGMLGraphBuilder.buildForward() would allocate space in a graph buffer.
    // For isolated testing of computeMatMul, we manually assign bufferId and data to graphAllocator.
    tensorA_Q4K.bufferId = 0 // Example buffer ID
    tensorA_Q4K.dataOffset = 0uL // Assuming it's at the start of its allocated buffer
    graphAllocator.buffers[tensorA_Q4K.bufferId] = tensorADataBytes


    // 2. Create and populate F32 tensor 'b' (K x N)
    val tensorB_F32 = GGMLTensor(GGMLType.F32)
    tensorB_F32.ne = longArrayOf(N.toLong(), K.toLong(), 1L, 1L) // ne[0]=N (cols), ne[1]=K (rows)
    // Strides for F32 (row-major from user perspective, but getFloat(col, row) implies specific internal layout)
    tensorB_F32.nb[0] = tensorB_F32.type.byteSize // Stride for dim 0 (N, columns)
    tensorB_F32.nb[1] = tensorB_F32.ne[0].toULong() * tensorB_F32.nb[0] // Stride for dim 1 (K, rows)
    for (d in 2 until GGML_MAX_DIMS) { tensorB_F32.nb[d] = tensorB_F32.ne[d-1].toULong() * tensorB_F32.nb[d-1] }

    val tensorBDataFloats = FloatArray(K * N) { i -> ((i + 50) % 120 - 30) * 0.01f } // Sample data
    // Convert FloatArray to ByteArray for graphAllocator
    val tensorBDataBytes = ByteArray(tensorBDataFloats.size * Float.SIZE_BYTES)
    tensorBDataFloats.forEachIndexed { index, flValue -> tensorBDataBytes.setFloatLe(index * Float.SIZE_BYTES, flValue) }

    tensorB_F32.bufferId = 1 // Example buffer ID
    tensorB_F32.dataOffset = 0uL
    graphAllocator.buffers[tensorB_F32.bufferId] = tensorBDataBytes


    println("Tensor A (Q4_K, ${tensorA_Q4K.ne.joinToString()}) first few reference F32 values: ${tensorA_RefF32Data.sliceArray(0..minOf(7, numElementsA-1)).joinToString()}")
    println("Tensor B (F32, ${tensorB_F32.ne.joinToString()}) first few values: ${tensorBDataFloats.sliceArray(0..minOf(7, K*N-1)).joinToString()}")

    // 3. Perform matrix multiplication using computeMatMul
    var resultTensor: GGMLTensor? = null
    try {
        // computeMatMul is expected to allocate its own result tensor and manage its data via graphAllocator
        resultTensor = computeMatMul(graphAllocator, context, tensorA_Q4K, tensorB_F32)
    } catch (e: Exception) {
        println("ERROR during Q4_K x F32 MatMul computation: ${e.message}")
        e.printStackTrace()
        println("MatMul Q4_K x F32 test FAILED.")
        return
    }

    // Extract result data from graphAllocator's buffer using resultTensor's info
    val resultBuffer = graphAllocator.buffers[resultTensor.bufferId]
        ?: throw IllegalStateException("Result tensor buffer not found in graphAllocator.")
    val resultDataFloats = FloatArray(M * N)
    for(i in 0 until M*N) {
        // Assuming result tensor is also F32 and flat for this simple M*N case
        resultDataFloats[i] = resultBuffer.getFloatLe(resultTensor.dataOffset.toInt() + i * Float.SIZE_BYTES)
    }

    println("Result Tensor (F32, ${resultTensor.ne.joinToString()}) first few values: ${resultDataFloats.sliceArray(0..minOf(7, M*N-1)).joinToString()}")

    // 4. Reference F32 x F32 calculation
    val tensorA_F32_ForRef_Data = FloatArray(numElementsA)
    var currentReadOffsetA = 0
    for (blockIdx in 0 until numBlocksA) {
        val qBlockBytes = ByteArray(GGMLType.Q4_K.byteSize.toInt())
        tensorADataBytes.copyInto(qBlockBytes, destinationOffset = 0, startIndex = blockIdx * GGMLType.Q4_K.byteSize.toInt(), endIndex = (blockIdx + 1) * GGMLType.Q4_K.byteSize.toInt())
        val dequantBlockFloats = dequantizeBlockQ4K(qBlockBytes)
        dequantBlockFloats.copyInto(tensorA_F32_ForRef_Data, destinationOffset = currentReadOffsetA)
        currentReadOffsetA += QK4_K
    }

    val referenceResultData = FloatArray(M * N)
    for (i in 0 until M) { // Row of A (and output)
        for (j in 0 until N) { // Col of B (and output)
            var sum = 0.0f
            for (k_idx in 0 until K) { // Common dimension
                // A_ref[i, k_idx] * B[k_idx, j]
                // tensorA_F32_ForRef_Data is M*K, effectively row major: A[row][col] = data[row*K + col]
                // tensorBDataFloats is K*N, effectively row major: B[row][col] = data[row*N + col]
                // So B[k_idx, j] means row k_idx, col j.
                sum += tensorA_F32_ForRef_Data[i * K + k_idx] * tensorBDataFloats[k_idx * N + j]
            }
            referenceResultData[i * N + j] = sum
        }
    }
    println("Reference F32 MatMul first few values: ${referenceResultData.sliceArray(0..minOf(7, M*N-1)).joinToString()}")


    // 5. Compare results
    var success = true
    var totalError = 0.0f
    var maxError = 0.0f
    for (i in 0 until M * N) {
        val error = abs(referenceResultData[i] - resultDataFloats[i])
        totalError += error
        if (error > maxError) maxError = error
        val threshold = 0.1f // Adjust threshold based on expected Q4_K precision
        if (error > threshold) {
            println("Error at index $i: Expected(F32) = ${referenceResultData[i]}, Got(Q4KMatMul) = ${resultDataFloats[i]}, Diff = $error")
            // success = false // Don't fail on first error, summarize
        }
    }
    val avgError = totalError / (M * N)
    println("MatMul Q4_K x F32: Avg Error = $avgError, Max Error = $maxError")

    if (avgError > 0.05f || maxError > 0.2f) { // Example overall thresholds
        success = false
    }

    if (success) {
        println("MatMul Q4_K x F32 test passed.")
    } else {
        println("MatMul Q4_K x F32 test FAILED (errors exceed thresholds).")
    }
}

/**
 * Tests the optimized add operation.
 */
fun testAdd(context: GGMLContext) {
 * Tests the optimized add operation.
 */
fun testAdd(context: GGMLContext) {
    println("\nTesting optimized add operation:")

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 4, 4)
    val b = createTensor2D(context, GGMLType.F32, 4, 4)

    val graphAllocator = context.graphAllocator!! // Get from context

    // Create tensors using a helper that puts data into graphAllocator
    // Assuming createTensor2D is updated or similar to createTestTensor1D
    val initialAData = FloatArray(16) { it.toFloat() }
    val initialBData = FloatArray(16) { (it + 1).toFloat() }

    val a = createTestTensor2D(graphAllocator, GGMLType.F32, 4, 4, initialAData)
    val b = createTestTensor2D(graphAllocator, GGMLType.F32, 4, 4, initialBData)


    println("Tensor a: [${initialAData.take(16).joinToString()}]")
    println("Tensor b: [${initialBData.take(16).joinToString()}]")

    // Call the graph building 'add' op. Since context.computeImmediately is true,
    // this will go through CPUBackend.executeOp.
    val c = add(context, a, b) // Use add from GGMLOps.kt

    // Extract data from tensor c, which should have its data in graphAllocator
    val cBuffer = graphAllocator.buffers[c.bufferId]
        ?: throw IllegalStateException("Result tensor c's buffer not found in graphAllocator.")
    val cData = FloatArray(16) { i ->
        cBuffer.getFloatLe(c.dataOffset.toInt() + i * Float.SIZE_BYTES)
    }
    println("a + b: [${cData.take(16).joinToString()}]")

    // Verify results
    var success = true
    for (i in 0 until 16) {
        val expected = initialAData[i] + initialBData[i]
        val actual = cData[i]
        if (kotlin.math.abs(expected - actual) > 1e-6f) { // Use tolerance for float comparison
            println("Error at index $i: expected $expected, got $actual")
            success = false
        }
    }

    if (success) {
        println("Add operation test passed.")
    } else {
        println("Add operation test failed.")
    }
}

// Placeholder for createTestTensor2D, adapt from createTestTensor1D
fun createTestTensor2D(
    graphAllocator: GGMLGraphAllocator,
    type: GGMLType,
    rows: Int,
    cols: Int,
    data: Any? // FloatArray, ShortArray, etc.
): GGMLTensor {
    val tensor = GGMLTensor(type)
    tensor.ne = longArrayOf(cols.toLong(), rows.toLong(), 1L, 1L)
    tensor.nb = calculateContiguousStrides(tensor.ne, tensor.type, tensor.rank())

    if (data != null) {
        val numElements = rows * cols
        val bufferBytes = when (type) {
            GGMLType.F32 -> {
                val floatData = data as FloatArray
                require(floatData.size == numElements)
                ByteArray(numElements * Float.SIZE_BYTES).apply {
                    floatData.forEachIndexed { idx, value -> setFloatLe(idx * Float.SIZE_BYTES, value) }
                }
            }
            GGMLType.F16 -> {
                val shortData = data as ShortArray
                require(shortData.size == numElements)
                ByteArray(numElements * Short.SIZE_BYTES).apply {
                    shortData.forEachIndexed { idx, value -> setShortLe(idx * Short.SIZE_BYTES, value) }
                }
            }
            // Add other types as needed
            else -> throw IllegalArgumentException("Unsupported type for createTestTensor2D data: $type")
        }
        tensor.bufferId = graphAllocator.allocateBuffer(bufferBytes)
        tensor.dataOffset = 0uL
    } else {
        val bufferSize = rows * cols * type.byteSize.toInt()
        tensor.bufferId = graphAllocator.allocateBuffer(ByteArray(bufferSize))
        tensor.dataOffset = 0uL
    }
    return tensor
}


/**
 * Tests the optimized mul operation.
 */
fun testMul(context: GGMLContext) {
    println("\nTesting optimized mul operation:")

    // Create tensors
    val graphAllocator = context.graphAllocator!!

    val initialAData = FloatArray(16) { it.toFloat() }
    val initialBData = FloatArray(16) { (it + 1).toFloat() }

    val a = createTestTensor2D(graphAllocator, GGMLType.F32, 4, 4, initialAData)
    val b = createTestTensor2D(graphAllocator, GGMLType.F32, 4, 4, initialBData)

    println("Tensor a: [${initialAData.take(16).joinToString()}]")
    println("Tensor b: [${initialBData.take(16).joinToString()}]")

    // Call the graph building 'mul' op.
    val c = mul(context, a, b) // from GGMLOps.kt

    val cBuffer = graphAllocator.buffers[c.bufferId]
        ?: throw IllegalStateException("Result tensor c's buffer not found in graphAllocator.")
    val cData = FloatArray(16) { i ->
        cBuffer.getFloatLe(c.dataOffset.toInt() + i * Float.SIZE_BYTES)
    }
    println("a * b: [${cData.take(16).joinToString()}]")

    // Verify results
    var success = true
    for (i in 0 until 16) {
        val expected = initialAData[i] * initialBData[i]
        val actual = cData[i]
        if (kotlin.math.abs(expected - actual) > 1e-6f) {
            println("Error at index $i: expected $expected, got $actual")
            success = false
        }
    }

    if (success) {
        println("Mul operation test passed.")
    } else {
        println("Mul operation test failed")
    }
}

/**
 * Tests the optimized matMul operation.
 */
fun testMatMul(context: GGMLContext) {
    println("\nTesting optimized matMul operation:")

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 4, 4)
    val b = createTensor2D(context, GGMLType.F32, 4, 4)

    // Initialize tensor data
    val aData = a.data as FloatArray
    val bData = b.data as FloatArray
    for (i in 0 until 16) {
        aData[i] = i.toFloat()
        bData[i] = (i + 1).toFloat()
    }

    println("Tensor a: [${aData.take(16).joinToString()}]")
    println("Tensor b: [${bData.take(16).joinToString()}]")

    // Test optimized matMul operation
    val c = computeMatMul(context, a, b)
    val cData = c.data as FloatArray
    println("a @ b: [${cData.take(16).joinToString()}]")

    // Verify results by computing the matrix multiplication manually
    var success = true
    for (i in 0 until 4) {
        for (j in 0 until 4) {
            var expected = 0.0f
            for (k in 0 until 4) {
                expected += aData[i * 4 + k] * bData[k * 4 + j]
            }
            val actual = cData[i * 4 + j]
            if (kotlin.math.abs(expected - actual) > 0.0001f) {
                println("Error at index (${i},${j}): expected $expected, got $actual")
                success = false
            }
        }
    }

    if (success) {
        println("MatMul operation test passed")
    } else {
        println("MatMul operation test failed")
    }
}
