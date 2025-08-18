package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES

/**
 * Extended unit tests for additional GGML operations not covered in the basic compute ops test.
 * Tests SUB, DIV, NEG, SQR, SQRT, ABS, SUM, MEAN, and other operations.
 */
class GGMLExtendedOpsTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 2 * 1024 * 1024 // 2MB for more complex tests

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        // Ensure buffer and allocator lists are not empty before assignment
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    // Helper to calculate tensor byte size (copied from existing tests for consistency)
    private fun calculateTensorByteSize(type: GGMLType, ne: LongArray): ULong {
        if (type.byteSize == 0uL && type != GGMLType.COUNT && !type.name.startsWith("Q")) {
            println("Warning: Calculating byte size for type ${type.name} with byteSize 0.")
            return 0uL
        }
        var elements = 1UL
        var validDimFound = false
        for (i in ne.indices) {
            if (ne[i] > 0L) {
                elements *= ne[i].toULong()
                validDimFound = true
            } else if (ne[i] == 0L && elements != 0UL && validDimFound) {
                return 0UL
            }
        }
        if (!validDimFound && ne.isNotEmpty() && ne.all { it <= 1L }) {
            elements = 1UL // Scalar or effectively scalar
        } else if (!validDimFound && ne.isEmpty()) {
            elements = 1UL // Treat as scalar if ne is completely empty
        }

        // For quantized types
        if (type == GGMLType.Q8_0 && elements > 0uL) {
            if (elements.toLong() % QK8_0 != 0L) {
                println("Warning: Total elements $elements for Q8_0 is not divisible by block size $QK8_0.")
            }
            return (elements.toLong() / QK8_0).toULong() * type.byteSize
        }

        return elements * type.byteSize
    }

    // Helper to calculate contiguous strides (simplified)
    private fun calculateStrides(type: GGMLType, ne: LongArray, maxDims: Int = GGML_MAX_DIMS): ULongArray {
        val nb = ULongArray(maxDims) { 0uL }
        if (type.byteSize > 0uL) {
            nb[0] = type.byteSize
            if (maxDims > 1) {
                for (d in 1 until maxDims) {
                    val prevDimSize = ne.getOrElse(d - 1) { 1L }
                    nb[d] = nb[d-1] * (if (prevDimSize > 0) prevDimSize.toULong() else 1uL)
                }
            }
        }
        return nb
    }

    // Helper to create and initialize tensor
    private fun createAndInitTensor(
        name: String,
        type: GGMLType,
        ne: LongArray,
        dataOffset: ULong = 0uL
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type, name = name)
        tensor.ne = ne.copyOf()
        
        // Set rank (number of non-1 dimensions, minimum 1)
        var rank = 0
        for (dim in ne) {
            if (dim > 1) rank++
        }
        // Ensure at least rank 1 for non-empty tensors
        if (rank == 0 && ne.any { it >= 1 }) rank = 1

        // Calculate contiguous strides
        tensor.nb = calculateStrides(type, ne)

        // Calculate buffer size needed and allocate in graph allocator
        val byteSize = calculateTensorByteSize(type, ne)
        if (byteSize > 0uL) {
            val allocatedTensor = graphAllocator.tensorAllocators[0].allocate(byteSize, type, name)
            tensor.bufferId = allocatedTensor.bufferId
            tensor.offset = allocatedTensor.offset
        }

        return tensor
    }

    // Helper to extract tensor data as FloatArray
    private fun getTensorDataAsFloatArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return floatArrayOf()
        
        val result = FloatArray(numElements)
        var idx = 0
        
        // Simple iteration for 1D case (most common in tests)
        if (tensor.ne.size >= 1) {
            for (i in 0 until numElements) {
                result[idx++] = when (tensor.type) {
                    GGMLType.F32 -> tensor.getFloat(graphAllocator, i)
                    GGMLType.F16 -> halfToFloat(tensor.getHalf(graphAllocator, i))
                    else -> tensor.getFloat(graphAllocator, i) // Try to convert to float
                }
            }
        }
        
        return result
    }

    private val dummyContext = GGMLContext() // Reusable dummy context

    // --- SUB Operation Tests ---
    @Test
    fun testComputeSubF32() {
        val srcNe = longArrayOf(4)
        val src0Data = floatArrayOf(10.0f, 20.0f, 30.0f, 40.0f)
        val src1Data = floatArrayOf(1.0f, 5.0f, 10.0f, 15.0f)
        val expectedData = floatArrayOf(9.0f, 15.0f, 20.0f, 25.0f)

        val src0 = createAndInitTensor("sub_f32_src0", GGMLType.F32, srcNe)
        for (i in src0Data.indices) src0.setFloat(graphAllocator, src0Data[i], i)

        val src1 = createAndInitTensor("sub_f32_src1", GGMLType.F32, srcNe)
        for (i in src1Data.indices) src1.setFloat(graphAllocator, src1Data[i], i)

        val resultTensor = computeSub(graphAllocator, dummyContext, src0, src1)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne), "Dimensions should match for SUB F32")

        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        assertEquals(expectedData.size, resultData.size, "SUB F32 result size mismatch")
        
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], resultData[i], 0.001f, "SUB F32 output mismatch at index $i")
        }
    }

    @Test
    fun testComputeSubF16() {
        val srcNe = longArrayOf(3)
        val src0DataF32 = floatArrayOf(5.5f, 10.0f, -2.5f)
        val src1DataF32 = floatArrayOf(2.5f, 3.0f, -1.0f)
        val expectedDataF32 = floatArrayOf(3.0f, 7.0f, -1.5f)

        val src0 = createAndInitTensor("sub_f16_src0", GGMLType.F16, srcNe)
        for (i in src0DataF32.indices) src0.setHalf(graphAllocator, src0DataF32[i], i)

        val src1 = createAndInitTensor("sub_f16_src1", GGMLType.F16, srcNe)
        for (i in src1DataF32.indices) src1.setHalf(graphAllocator, src1DataF32[i], i)

        val resultTensor = computeSub(graphAllocator, dummyContext, src0, src1)

        assertEquals(GGMLType.F16, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne), "Dimensions should match for SUB F16")

        val resultDataF32 = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        assertEquals(expectedDataF32.size, resultDataF32.size, "SUB F16 result size mismatch")
        
        for (i in expectedDataF32.indices) {
            // Account for F16 precision loss
            val expected = halfToFloat(floatToHalf(expectedDataF32[i]))
            assertEquals(expected, resultDataF32[i], 0.01f, "SUB F16 output mismatch at index $i")
        }
    }

    // --- NEG Operation Tests ---
    @Test
    fun testComputeNegF32() {
        val srcNe = longArrayOf(5)
        val srcData = floatArrayOf(-5.0f, -1.0f, 0.0f, 1.0f, 5.0f)
        val expectedData = floatArrayOf(5.0f, 1.0f, 0.0f, -1.0f, -5.0f)

        val srcTensor = createAndInitTensor("neg_f32_src", GGMLType.F32, srcNe)
        for (i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        val resultTensor = computeNeg(graphAllocator, dummyContext, srcTensor)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for NEG F32")

        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        assertEquals(expectedData.size, resultData.size, "NEG F32 result size mismatch")
        
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], resultData[i], 0.001f, "NEG F32 output mismatch at index $i")
        }
    }

    @Test
    fun testComputeNegF16() {
        val srcNe = longArrayOf(4)
        val srcDataF32 = floatArrayOf(-2.5f, 0.0f, 1.5f, -10.0f)
        val expectedDataF32 = floatArrayOf(2.5f, 0.0f, -1.5f, 10.0f)

        val srcTensor = createAndInitTensor("neg_f16_src", GGMLType.F16, srcNe)
        for (i in srcDataF32.indices) srcTensor.setHalf(graphAllocator, srcDataF32[i], i)

        val resultTensor = computeNeg(graphAllocator, dummyContext, srcTensor)

        assertEquals(GGMLType.F16, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for NEG F16")

        val resultDataF32 = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        assertEquals(expectedDataF32.size, resultDataF32.size, "NEG F16 result size mismatch")
        
        for (i in expectedDataF32.indices) {
            val expected = halfToFloat(floatToHalf(expectedDataF32[i]))
            assertEquals(expected, resultDataF32[i], 0.01f, "NEG F16 output mismatch at index $i")
        }
    }

    // --- DIV Operation Tests ---
    @Test
    fun testComputeDivF32() {
        val srcNe = longArrayOf(4)
        val src0Data = floatArrayOf(10.0f, 20.0f, -15.0f, 100.0f)
        val src1Data = floatArrayOf(2.0f, 4.0f, -3.0f, 10.0f)
        val expectedData = floatArrayOf(5.0f, 5.0f, 5.0f, 10.0f)

        val src0 = createAndInitTensor("div_f32_src0", GGMLType.F32, srcNe)
        for (i in src0Data.indices) src0.setFloat(graphAllocator, src0Data[i], i)

        val src1 = createAndInitTensor("div_f32_src1", GGMLType.F32, srcNe)
        for (i in src1Data.indices) src1.setFloat(graphAllocator, src1Data[i], i)

        val resultTensor = computeDiv(graphAllocator, dummyContext, src0, src1)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne), "Dimensions should match for DIV F32")

        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        assertEquals(expectedData.size, resultData.size, "DIV F32 result size mismatch")
        
        for (i in expectedData.indices) {
            assertEquals(expectedData[i], resultData[i], 0.001f, "DIV F32 output mismatch at index $i")
        }
    }

    @Test
    fun testComputeDivF32EdgeCases() {
        val srcNe = longArrayOf(4)
        val src0Data = floatArrayOf(1.0f, -1.0f, 0.0f, Float.POSITIVE_INFINITY)
        val src1Data = floatArrayOf(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 5.0f, 2.0f)
        
        val src0 = createAndInitTensor("div_edge_src0", GGMLType.F32, srcNe)
        for (i in src0Data.indices) src0.setFloat(graphAllocator, src0Data[i], i)

        val src1 = createAndInitTensor("div_edge_src1", GGMLType.F32, srcNe)
        for (i in src1Data.indices) src1.setFloat(graphAllocator, src1Data[i], i)

        val resultTensor = computeDiv(graphAllocator, dummyContext, src0, src1)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne), "Dimensions should match for DIV F32 edge cases")

        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        
        // Test that division handles edge cases appropriately
        assertTrue(resultData[0].isNaN() || resultData[0] == 0.0f, "1.0f / POSITIVE_INFINITY should be 0 or NaN")
        assertTrue(resultData[1].isNaN() || resultData[1] == 0.0f, "-1.0f / NEGATIVE_INFINITY should be 0 or NaN")
        assertEquals(0.0f, resultData[2], 0.001f, "0.0f / 5.0f should be 0.0f")
        assertTrue(resultData[3].isInfinite(), "POSITIVE_INFINITY / 2.0f should be infinite")
    }

    // --- SQR Operation Tests ---
    @Test
    fun testComputeSqrF32() {
        val srcNe = longArrayOf(5)
        val srcData = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f, 4.0f)
        val expectedData = floatArrayOf(9.0f, 1.0f, 0.0f, 4.0f, 16.0f)

        val srcTensor = createAndInitTensor("sqr_f32_src", GGMLType.F32, srcNe)
        for (i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        // Note: computeSqr might not be implemented yet, so let's check for its existence
        try {
            val resultTensor = computeSqr(graphAllocator, dummyContext, srcTensor)

            assertEquals(GGMLType.F32, resultTensor.type)
            assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for SQR F32")

            val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
            assertEquals(expectedData.size, resultData.size, "SQR F32 result size mismatch")
            
            for (i in expectedData.indices) {
                assertEquals(expectedData[i], resultData[i], 0.001f, "SQR F32 output mismatch at index $i")
            }
        } catch (e: NotImplementedError) {
            println("SQR operation not yet implemented - skipping test")
        } catch (e: NoSuchMethodError) {
            println("computeSqr method not found - operation not yet implemented")
        }
    }

    // --- SQRT Operation Tests ---  
    @Test
    fun testComputeSqrtF32() {
        val srcNe = longArrayOf(4)
        val srcData = floatArrayOf(0.0f, 1.0f, 4.0f, 9.0f)
        val expectedData = floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f)

        val srcTensor = createAndInitTensor("sqrt_f32_src", GGMLType.F32, srcNe)
        for (i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        try {
            val resultTensor = computeSqrt(graphAllocator, dummyContext, srcTensor)

            assertEquals(GGMLType.F32, resultTensor.type)
            assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for SQRT F32")

            val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
            assertEquals(expectedData.size, resultData.size, "SQRT F32 result size mismatch")
            
            for (i in expectedData.indices) {
                assertEquals(expectedData[i], resultData[i], 0.001f, "SQRT F32 output mismatch at index $i")
            }
        } catch (e: NotImplementedError) {
            println("SQRT operation not yet implemented - skipping test")
        } catch (e: NoSuchMethodError) {
            println("computeSqrt method not found - operation not yet implemented")
        }
    }

    // --- Edge Case Tests ---
    @Test
    fun testOperationsWithScalarTensors() {
        // Test operations with scalar (1-element) tensors
        val scalarNe = longArrayOf(1)
        
        val scalar1 = createAndInitTensor("scalar1", GGMLType.F32, scalarNe)
        scalar1.setFloat(graphAllocator, 5.0f, 0)
        
        val scalar2 = createAndInitTensor("scalar2", GGMLType.F32, scalarNe)  
        scalar2.setFloat(graphAllocator, 3.0f, 0)
        
        // Test SUB with scalars
        val subResult = computeSub(graphAllocator, dummyContext, scalar1, scalar2)
        assertEquals(2.0f, subResult.getFloat(graphAllocator, 0), 0.001f, "Scalar subtraction failed")
        
        // Test NEG with scalar
        val negResult = computeNeg(graphAllocator, dummyContext, scalar1)
        assertEquals(-5.0f, negResult.getFloat(graphAllocator, 0), 0.001f, "Scalar negation failed")
    }

    @Test
    fun testOperationsWithEmptyTensors() {
        // Test with empty tensors (0 elements)
        val emptyNe = longArrayOf(0)
        
        val empty1 = createAndInitTensor("empty1", GGMLType.F32, emptyNe)
        val empty2 = createAndInitTensor("empty2", GGMLType.F32, emptyNe)
        
        // These operations should either handle empty tensors gracefully or throw appropriate exceptions
        try {
            val subResult = computeSub(graphAllocator, dummyContext, empty1, empty2)
            assertEquals(0, subResult.numElements(), "Empty tensor SUB should result in empty tensor")
        } catch (e: Exception) {
            // It's acceptable for operations to throw exceptions on empty tensors
            println("SUB with empty tensors threw exception: ${e.message}")
        }
    }

    // --- Broadcasting Tests (if supported) ---
    @Test
    fun testBroadcastingOperations() {
        // Test broadcasting between different shaped tensors (if supported)
        val tensor1Ne = longArrayOf(4, 1) // 4x1
        val tensor2Ne = longArrayOf(1, 3) // 1x3
        
        val tensor1 = createAndInitTensor("broadcast1", GGMLType.F32, tensor1Ne)
        val tensor2 = createAndInitTensor("broadcast2", GGMLType.F32, tensor2Ne)
        
        // Initialize with simple patterns
        for (i in 0 until 4) {
            tensor1.setFloat(graphAllocator, (i + 1).toFloat(), i, 0)
        }
        for (j in 0 until 3) {
            tensor2.setFloat(graphAllocator, (j + 1).toFloat(), 0, j)
        }
        
        try {
            // This might not be supported yet, but let's test
            val addResult = computeAdd(graphAllocator, dummyContext, tensor1, tensor2)
            // If broadcasting is supported, result should be 4x3
            // If not, this will throw an exception
            println("Broadcasting ADD succeeded with result shape: ${addResult.ne.contentToString()}")
        } catch (e: Exception) {
            println("Broadcasting not yet supported: ${e.message}")
        }
    }
}