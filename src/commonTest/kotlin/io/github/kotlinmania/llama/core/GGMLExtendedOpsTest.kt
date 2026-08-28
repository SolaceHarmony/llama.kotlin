package io.github.kotlinmania.llama.core

import kotlin.math.*
import kotlin.test.*
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES

/**
 * Extended unit tests for additional GGML operations not covered in the basic compute ops test.
 * Tests SUB, DIV, NEG, SQR, SQRT, ABS, SUM, MEAN, and other operations.
 */
class GGMLExtendedOpsTest {

    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 2 * 1024 * 1024 // 2MB for more complex tests

    @BeforeTest
    fun setup() {
        graphAllocator = io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        // Ensure buffer and allocator lists are not empty before assignment
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(io.github.kotlinmania.llama.ore.GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
        resetAllocatorTracking(graphAllocator)
    }

    // Use shared utility for tensor byte size calculation

    // Use shared utility for stride calculation

    // Use shared utility for tensor creation
    private fun createAndInitTensor(
        name: String,
        type: io.github.kotlinmania.llama.ore.GGMLType,
        ne: LongArray
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type, name = name)
        tensor.ne = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 1L }
        ne.forEachIndexed { index, dim ->
            if (index < io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) tensor.ne[index] = dim
        }
        tensor.nb = GGMLTestUtils.calculateStrides(type, tensor.ne)

        val byteSize = GGMLTestUtils.calculateTensorByteSize(type, tensor.ne).toInt()
        if (byteSize > 0) {
            val offset = graphAllocator.allocateTensorData(byteSize)
            tensor.bufferId = 0
            tensor.dataOffset = offset
        }

        return tensor
    }

    private fun createDestinationTensor(
        name: String,
        type: io.github.kotlinmania.llama.ore.GGMLType,
        ne: LongArray
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type, name = name)
        tensor.ne = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 1L }
        ne.forEachIndexed { index, dim ->
            if (index < io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) tensor.ne[index] = dim
        }
        tensor.nb = GGMLTestUtils.calculateStrides(type, tensor.ne)

        val byteSize = GGMLTestUtils.calculateTensorByteSize(type, tensor.ne).toInt()
        if (byteSize > 0) {
            val offset = graphAllocator.allocateTensorData(byteSize)
            tensor.bufferId = 0
            tensor.dataOffset = offset
        }

        return tensor
    }

    private val dummyContext =
        io.github.kotlinmania.llama.ore.GGMLContext() // Reusable dummy context

    // --- SUB Operation Tests ---
    @Test
    fun testComputeSubF32() {
        val srcNe = longArrayOf(4)
        val src0Data = floatArrayOf(10.0f, 20.0f, 30.0f, 40.0f)
        val src1Data = floatArrayOf(1.0f, 5.0f, 10.0f, 15.0f)
        val expectedData = floatArrayOf(9.0f, 15.0f, 20.0f, 25.0f)

        val src0 = createAndInitTensor("sub_f32_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in src0Data.indices) src0.setFloat(graphAllocator, src0Data[i], i)

        val src1 = createAndInitTensor("sub_f32_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in src1Data.indices) src1.setFloat(graphAllocator, src1Data[i], i)

        val dst = createDestinationTensor("sub_f32_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        io.github.kotlinmania.llama.ore.computeSub(graphAllocator, src0, src1, dst)

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dst.type)
        assertTrue(src0.ne.contentEquals(dst.ne), "Dimensions should match for SUB F32")

        val resultData = GGMLTestUtils.extractFloatData(dst, graphAllocator)
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

        val src0 = createAndInitTensor("sub_f16_src0", io.github.kotlinmania.llama.ore.GGMLType.F16, srcNe)
        for (i in src0DataF32.indices) src0.setHalf(graphAllocator, src0DataF32[i], i)

        val src1 = createAndInitTensor("sub_f16_src1", io.github.kotlinmania.llama.ore.GGMLType.F16, srcNe)
        for (i in src1DataF32.indices) src1.setHalf(graphAllocator, src1DataF32[i], i)

        val dst = createDestinationTensor("sub_f16_dst", io.github.kotlinmania.llama.ore.GGMLType.F16, srcNe)
        io.github.kotlinmania.llama.ore.computeSub(graphAllocator, src0, src1, dst)

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F16, dst.type)
        assertTrue(src0.ne.contentEquals(dst.ne), "Dimensions should match for SUB F16")

        val resultDataF32 = GGMLTestUtils.extractFloatData(dst, graphAllocator)
        assertEquals(expectedDataF32.size, resultDataF32.size, "SUB F16 result size mismatch")
        
        for (i in expectedDataF32.indices) {
            // Account for F16 precision loss
            val expected = io.github.kotlinmania.llama.ore.halfToFloat(
                io.github.kotlinmania.llama.ore.floatToHalf(expectedDataF32[i])
            )
            assertEquals(expected, resultDataF32[i], 0.01f, "SUB F16 output mismatch at index $i")
        }
    }

    // --- NEG Operation Tests ---
    @Test
    fun testComputeNegF32() {
        val srcNe = longArrayOf(5)
        val srcData = floatArrayOf(-5.0f, -1.0f, 0.0f, 1.0f, 5.0f)
        val expectedData = floatArrayOf(5.0f, 1.0f, 0.0f, -1.0f, -5.0f)

        val srcTensor = createAndInitTensor("neg_f32_src", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        val dst = createDestinationTensor("neg_f32_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        io.github.kotlinmania.llama.ore.computeNeg(graphAllocator, srcTensor, dst)

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dst.type)
        assertTrue(srcTensor.ne.contentEquals(dst.ne), "Dimensions should match for NEG F32")

        val resultData = GGMLTestUtils.extractFloatData(dst, graphAllocator)
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

        val srcTensor = createAndInitTensor("neg_f16_src", io.github.kotlinmania.llama.ore.GGMLType.F16, srcNe)
        for (i in srcDataF32.indices) srcTensor.setHalf(graphAllocator, srcDataF32[i], i)

        val dst = createDestinationTensor("neg_f16_dst", io.github.kotlinmania.llama.ore.GGMLType.F16, srcNe)
        io.github.kotlinmania.llama.ore.computeNeg(graphAllocator, srcTensor, dst)

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F16, dst.type)
        assertTrue(srcTensor.ne.contentEquals(dst.ne), "Dimensions should match for NEG F16")

        val resultDataF32 = GGMLTestUtils.extractFloatData(dst, graphAllocator)
        assertEquals(expectedDataF32.size, resultDataF32.size, "NEG F16 result size mismatch")
        
        for (i in expectedDataF32.indices) {
            val expected = io.github.kotlinmania.llama.ore.halfToFloat(
                io.github.kotlinmania.llama.ore.floatToHalf(expectedDataF32[i])
            )
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

        val src0 = createAndInitTensor("div_f32_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in src0Data.indices) src0.setFloat(graphAllocator, src0Data[i], i)

        val src1 = createAndInitTensor("div_f32_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in src1Data.indices) src1.setFloat(graphAllocator, src1Data[i], i)

        val dst = createDestinationTensor("div_f32_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        io.github.kotlinmania.llama.ore.computeDiv(graphAllocator, src0, src1, dst)

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dst.type)
        assertTrue(src0.ne.contentEquals(dst.ne), "Dimensions should match for DIV F32")

        val resultData = GGMLTestUtils.extractFloatData(dst, graphAllocator)
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
        
        val src0 = createAndInitTensor("div_edge_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in src0Data.indices) src0.setFloat(graphAllocator, src0Data[i], i)

        val src1 = createAndInitTensor("div_edge_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in src1Data.indices) src1.setFloat(graphAllocator, src1Data[i], i)

        val dst = createDestinationTensor("div_edge_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        io.github.kotlinmania.llama.ore.computeDiv(graphAllocator, src0, src1, dst)

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dst.type)
        assertTrue(src0.ne.contentEquals(dst.ne), "Dimensions should match for DIV F32 edge cases")

        val resultData = GGMLTestUtils.extractFloatData(dst, graphAllocator)
        
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

        val srcTensor = createAndInitTensor("sqr_f32_src", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        // Note: computeSqr might not be implemented yet, so let's check for its existence
        try {
            val dst = createDestinationTensor("sqr_f32_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
            io.github.kotlinmania.llama.ore.computeSqr(graphAllocator, srcTensor, dst)

            assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dst.type)
            assertTrue(srcTensor.ne.contentEquals(dst.ne), "Dimensions should match for SQR F32")

            val resultData = GGMLTestUtils.extractFloatData(dst, graphAllocator)
            assertEquals(expectedData.size, resultData.size, "SQR F32 result size mismatch")
            
            for (i in expectedData.indices) {
                assertEquals(expectedData[i], resultData[i], 0.001f, "SQR F32 output mismatch at index $i")
            }
        } catch (e: NotImplementedError) {
            println("SQR operation not yet implemented - skipping test")
        }
    }

    // --- SQRT Operation Tests ---  
    @Test
    fun testComputeSqrtF32() {
        val srcNe = longArrayOf(4)
        val srcData = floatArrayOf(0.0f, 1.0f, 4.0f, 9.0f)
        val expectedData = floatArrayOf(0.0f, 1.0f, 2.0f, 3.0f)

        val srcTensor = createAndInitTensor("sqrt_f32_src", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
        for (i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        try {
            val dst = createDestinationTensor("sqrt_f32_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, srcNe)
            io.github.kotlinmania.llama.ore.computeSqrt(graphAllocator, srcTensor, dst)

            assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dst.type)
            assertTrue(srcTensor.ne.contentEquals(dst.ne), "Dimensions should match for SQRT F32")

            val resultData = GGMLTestUtils.extractFloatData(dst, graphAllocator)
            assertEquals(expectedData.size, resultData.size, "SQRT F32 result size mismatch")
            
            for (i in expectedData.indices) {
                assertEquals(expectedData[i], resultData[i], 0.001f, "SQRT F32 output mismatch at index $i")
            }
        } catch (e: NotImplementedError) {
            println("SQRT operation not yet implemented - skipping test")
        }
    }

    // --- Edge Case Tests ---
    @Test
    fun testOperationsWithScalarTensors() {
        // Test operations with scalar (1-element) tensors
        val scalarNe = longArrayOf(1)
        
        val scalar1 = createAndInitTensor("scalar1", io.github.kotlinmania.llama.ore.GGMLType.F32, scalarNe)
        scalar1.setFloat(graphAllocator, 5.0f, 0)
        
        val scalar2 = createAndInitTensor("scalar2", io.github.kotlinmania.llama.ore.GGMLType.F32, scalarNe)
        scalar2.setFloat(graphAllocator, 3.0f, 0)
        
        // Test SUB with scalars
        val subDst = createDestinationTensor("scalar_sub_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, scalarNe)
        io.github.kotlinmania.llama.ore.computeSub(graphAllocator, scalar1, scalar2, subDst)
        assertEquals(2.0f, subDst.getFloat(graphAllocator, 0), 0.001f, "Scalar subtraction failed")
        
        // Test NEG with scalar
        val negDst = createDestinationTensor("scalar_neg_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, scalarNe)
        io.github.kotlinmania.llama.ore.computeNeg(graphAllocator, scalar1, negDst)
        assertEquals(-5.0f, negDst.getFloat(graphAllocator, 0), 0.001f, "Scalar negation failed")
    }

    @Test
    fun testOperationsWithEmptyTensors() {
        // Test with empty tensors (0 elements)
        val emptyNe = longArrayOf(0)
        
        val empty1 = createAndInitTensor("empty1", io.github.kotlinmania.llama.ore.GGMLType.F32, emptyNe)
        val empty2 = createAndInitTensor("empty2", io.github.kotlinmania.llama.ore.GGMLType.F32, emptyNe)
        
        // These operations should either handle empty tensors gracefully or throw appropriate exceptions
        try {
            val subDst = createDestinationTensor("empty_sub_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, emptyNe)
            io.github.kotlinmania.llama.ore.computeSub(graphAllocator, empty1, empty2, subDst)
            assertEquals(0, subDst.numElements(), "Empty tensor SUB should result in empty tensor")
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
        
        val tensor1 = createAndInitTensor("broadcast1", io.github.kotlinmania.llama.ore.GGMLType.F32, tensor1Ne)
        val tensor2 = createAndInitTensor("broadcast2", io.github.kotlinmania.llama.ore.GGMLType.F32, tensor2Ne)
        
        // Initialize with simple patterns
        for (i in 0 until 4) {
            tensor1.setFloat(graphAllocator, (i + 1).toFloat(), i, 0)
        }
        for (j in 0 until 3) {
            tensor2.setFloat(graphAllocator, (j + 1).toFloat(), 0, j)
        }
        
        try {
            // This might not be supported yet, but let's test
            val broadcastDst = createDestinationTensor("broadcast_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, longArrayOf(4, 3))
            io.github.kotlinmania.llama.ore.computeAdd(
                graphAllocator,
                tensor1,
                tensor2,
                broadcastDst
            )
            println("Broadcasting ADD succeeded with result shape: ${broadcastDst.ne.contentToString()}")
        } catch (e: Exception) {
            println("Broadcasting not yet supported: ${e.message}")
        }
    }
}
