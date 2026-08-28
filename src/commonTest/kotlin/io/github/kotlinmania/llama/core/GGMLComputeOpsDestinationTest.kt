package io.github.kotlinmania.llama.core

import kotlin.math.exp
import kotlin.test.*

/**
 * Tests for the new destination tensor interface of compute operations.
 * Validates that compute ops write correctly into pre-allocated destination tensors
 * managed by the graph allocator rather than creating new tensors with fresh arrays.
 */
class GGMLComputeOpsDestinationTest {

    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private val bufferSize = 1 * 1024 * 1024 // 1MB

    // Use shared utilities for stride and byte size calculations

    // Helper to create and initialize tensor in graph allocator
    private fun createAndInitTensor(
        name: String,
        type: io.github.kotlinmania.llama.ore.GGMLType,
        dims: LongArray,
        dataOffset: ULong = 0uL,
        bufferId: Int = 0,
        fillSequence: Boolean = false,
        startValue: Number = 0.0f,
        step: Number = 1.0f
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)
        tensor.name = name

        // Pad ne to GGML_MAX_DIMS with 1s for unused dimensions
        tensor.ne = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dimSize ->
            if (index < io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) tensor.ne[index] = dimSize
        }

        tensor.nb = GGMLTestUtils.calculateStrides(type, tensor.ne)
        tensor.bufferId = bufferId
        tensor.dataOffset = dataOffset

        // Initialize with sequence or zero
        if (fillSequence) {
            var currentValue = startValue.toFloat()
            val stepFloat = step.toFloat()
            val numElements = dims.fold(1L) { acc, dim -> acc * dim }.toInt()

            when (type) {
                io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
                    for (i in 0 until numElements) {
                        tensor.setFloat(graphAllocator, currentValue, i)
                        currentValue += stepFloat
                    }
                }
                io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
                    for (i in 0 until numElements) {
                        tensor.setHalf(graphAllocator, currentValue, i)
                        currentValue += stepFloat
                    }
                }
                io.github.kotlinmania.llama.ore.GGMLType.I32 -> {
                    var intValue = startValue.toInt()
                    val stepInt = step.toInt()
                    for (i in 0 until numElements) {
                        tensor.setInt(graphAllocator, intValue, i)
                        intValue += stepInt
                    }
                }
                else -> {
                    // Default: fill with zeros for other types
                }
            }
        }

        return tensor
    }

    @BeforeTest
    fun setUp() {
        graphAllocator = io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
    }

    @Test
    fun testComputeAddWithDestination() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims = longArrayOf(4) // Simple 1D tensor
        val size = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dims)

        // Create source tensors
        val src0 = createAndInitTensor("add_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset, fillSequence = true, startValue = 1.0f, step = 1.0f)
        currentOffset += size
        val src1 = createAndInitTensor("add_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset, fillSequence = true, startValue = 10.0f, step = 2.0f)
        currentOffset += size
        
        // Create destination tensor (pre-allocated)
        val dst = createAndInitTensor("add_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)

        // Verify initial destination is zeros
        for (i in 0 until dims[0].toInt()) {
            assertEquals(0.0f, dst.getFloat(graphAllocator, i), "Initial dst should be zero at index $i")
        }

        // Perform addition into destination tensor
        io.github.kotlinmania.llama.ore.computeAdd(graphAllocator, src0, src1, dst)

        // Verify results are written to destination
        for (i in 0 until dims[0].toInt()) {
            val expected = src0.getFloat(graphAllocator, i) + src1.getFloat(graphAllocator, i)
            val actual = dst.getFloat(graphAllocator, i)
            assertEquals(expected, actual, "ADD result mismatch at index $i")
        }
    }

    @Test 
    fun testComputeMulWithDestination() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims = longArrayOf(3)
        val size = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dims)

        val src0 = createAndInitTensor("mul_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset, fillSequence = true, startValue = 2.0f, step = 1.0f)
        currentOffset += size
        val src1 = createAndInitTensor("mul_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset, fillSequence = true, startValue = 3.0f, step = 2.0f)
        currentOffset += size
        val dst = createAndInitTensor("mul_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)

        io.github.kotlinmania.llama.ore.computeMul(graphAllocator, src0, src1, dst)

        for (i in 0 until dims[0].toInt()) {
            val expected = src0.getFloat(graphAllocator, i) * src1.getFloat(graphAllocator, i)
            val actual = dst.getFloat(graphAllocator, i)
            assertEquals(expected, actual, "MUL result mismatch at index $i")
        }
    }

    @Test
    fun testComputeReluWithDestination() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims = longArrayOf(5)
        val size = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dims)

        // Create source with mix of positive and negative values
        val src = createAndInitTensor("relu_src", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)
        val testValues = floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f)
        for (i in testValues.indices) {
            src.setFloat(graphAllocator, testValues[i], i)
        }
        currentOffset += size
        
        val dst = createAndInitTensor("relu_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)

        io.github.kotlinmania.llama.ore.computeRelu(graphAllocator, src, dst)

        for (i in testValues.indices) {
            val expected = if (testValues[i] > 0.0f) testValues[i] else 0.0f
            val actual = dst.getFloat(graphAllocator, i)
            assertEquals(expected, actual, "RELU result mismatch at index $i")
        }
    }

    @Test
    fun testComputeSoftMaxWithDestination() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims = longArrayOf(4)
        val size = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dims)

        val src = createAndInitTensor("softmax_src", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)
        val testValues = floatArrayOf(1f, 2f, 3f, 4f)
        for (i in testValues.indices) {
            src.setFloat(graphAllocator, testValues[i], i)
        }
        currentOffset += size

        val dst = createAndInitTensor("softmax_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)

        io.github.kotlinmania.llama.ore.computeSoftMax(graphAllocator, src, dst)

        val max = testValues.maxOrNull()!!
        val expVals = testValues.map { exp(it - max) }
        val sum = expVals.sum()
        for (i in testValues.indices) {
            val expected = expVals[i] / sum
            val actual = dst.getFloat(graphAllocator, i)
            assertEquals(expected, actual, 1e-5f, "SoftMax result mismatch at index $i")
        }
    }

    @Test
    fun testComputeSubWithDestination() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims = longArrayOf(3)
        val size = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.I32, dims)

        val src0 = createAndInitTensor("sub_src0", io.github.kotlinmania.llama.ore.GGMLType.I32, dims, currentOffset, fillSequence = true, startValue = 10, step = 5)
        currentOffset += size
        val src1 = createAndInitTensor("sub_src1", io.github.kotlinmania.llama.ore.GGMLType.I32, dims, currentOffset, fillSequence = true, startValue = 2, step = 1)
        currentOffset += size
        val dst = createAndInitTensor("sub_dst", io.github.kotlinmania.llama.ore.GGMLType.I32, dims, currentOffset)

        io.github.kotlinmania.llama.ore.computeSub(graphAllocator, src0, src1, dst)

        for (i in 0 until dims[0].toInt()) {
            val expected = src0.getInt(graphAllocator, i) - src1.getInt(graphAllocator, i)
            val actual = dst.getInt(graphAllocator, i)
            assertEquals(expected, actual, "SUB result mismatch at index $i")
        }
    }

    @Test
    fun testComputeMatMulWithDestination() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        
        // Simple 2x3 * 3x2 = 2x2 matrix multiplication
        val M = 2
        val K = 3  
        val N = 2
        
        val aDims = longArrayOf(K.toLong(), M.toLong()) // 3x2 (K x M)
        val bDims = longArrayOf(N.toLong(), K.toLong()) // 2x3 (N x K)
        val dstDims = longArrayOf(N.toLong(), M.toLong()) // 2x2 (N x M)
        
        val aSize = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, aDims)
        val bSize = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, bDims)
        val dstSize = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dstDims)

        // Matrix A: 2x3 = [[1,2,3], [4,5,6]]
        val srcA = createAndInitTensor("matmul_a", io.github.kotlinmania.llama.ore.GGMLType.F32, aDims, currentOffset)
        val aValues = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f)
        for (i in aValues.indices) {
            srcA.setFloat(graphAllocator, aValues[i], i)
        }
        currentOffset += aSize

        // Matrix B: 3x2 = [[7,8], [9,10], [11,12]]  
        val srcB = createAndInitTensor("matmul_b", io.github.kotlinmania.llama.ore.GGMLType.F32, bDims, currentOffset)
        val bValues = floatArrayOf(7f, 8f, 9f, 10f, 11f, 12f)
        for (i in bValues.indices) {
            srcB.setFloat(graphAllocator, bValues[i], i)
        }
        currentOffset += bSize

        // Pre-allocate destination tensor
        val dst = createAndInitTensor("matmul_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, dstDims, currentOffset)

        // Perform matrix multiplication
        io.github.kotlinmania.llama.ore.computeMatMul(graphAllocator, srcA, srcB, dst)

        // Expected result: 2x2 = [[58, 64], [139, 154]]
        val expected = floatArrayOf(58f, 64f, 139f, 154f)
        
        for (i in expected.indices) {
            val actual = dst.getFloat(graphAllocator, i)
            assertEquals(expected[i], actual, 0.001f, "MatMul result mismatch at index $i")
        }
    }

    @Test
    fun testDimensionValidation() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims1 = longArrayOf(3)
        val dims2 = longArrayOf(4) // Different size
        val size = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dims1)

        val src0 = createAndInitTensor("dim_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, dims1, currentOffset)
        currentOffset += size
        val src1 = createAndInitTensor("dim_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, dims1, currentOffset)
        currentOffset += size
        val badDst = createAndInitTensor("dim_bad_dst", io.github.kotlinmania.llama.ore.GGMLType.F32, dims2, currentOffset) // Wrong dimensions

        // Should throw exception due to dimension mismatch
        assertFailsWith<IllegalArgumentException>("Should throw for dimension mismatch") {
            io.github.kotlinmania.llama.ore.computeAdd(graphAllocator, src0, src1, badDst)
        }
    }

    @Test
    fun testTypeValidation() {
        val context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL
        val dims = longArrayOf(3)
        val sizeF32 = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F32, dims)
        val sizeF16 = calculateTensorByteSize(io.github.kotlinmania.llama.ore.GGMLType.F16, dims)

        val src0 = createAndInitTensor("type_src0", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)
        currentOffset += sizeF32
        val src1 = createAndInitTensor("type_src1", io.github.kotlinmania.llama.ore.GGMLType.F32, dims, currentOffset)
        currentOffset += sizeF32
        val badDst = createAndInitTensor("type_bad_dst", io.github.kotlinmania.llama.ore.GGMLType.F16, dims, currentOffset) // Wrong type

        // Should throw exception due to type mismatch
        assertFailsWith<IllegalArgumentException>("Should throw for type mismatch") {
            io.github.kotlinmania.llama.ore.computeAdd(graphAllocator, src0, src1, badDst)
        }
    }
}
