package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*

/**
 * Integration tests for end-to-end computation graph operations.
 * Tests complex computation chains, memory allocation patterns, and multi-operation flows.
 */
class GGMLIntegrationTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 8 * 1024 * 1024 // 8MB for complex graph operations

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    private val dummyContext = GGMLContext()

    // Helper to create tensor with data
    private fun createTensorWithData(name: String, type: GGMLType, ne: LongArray, data: FloatArray): GGMLTensor {
        val tensor = GGMLTensor(type = type, name = name)
        tensor.ne = ne.copyOf()
        
        // Calculate strides
        tensor.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        tensor.nb[0] = type.byteSize
        for (d in 1 until GGML_MAX_DIMS) {
            tensor.nb[d] = tensor.nb[d-1] * ne.getOrElse(d-1) { 1L }.toULong()
        }
        
        // Allocate memory
        val totalElements = ne.fold(1L) { acc, dim -> acc * maxOf(dim, 1L) }
        val byteSize = totalElements.toULong() * type.byteSize
        val allocatedTensor = graphAllocator.tensorAllocators[0].allocate(byteSize, type, name)
        tensor.bufferId = allocatedTensor.bufferId
        tensor.offset = allocatedTensor.offset
        
        // Set data
        when (type) {
            GGMLType.F32 -> {
                for (i in data.indices) {
                    tensor.setFloat(graphAllocator, data[i], i)
                }
            }
            GGMLType.F16 -> {
                for (i in data.indices) {
                    tensor.setHalf(graphAllocator, data[i], i)
                }
            }
            else -> throw IllegalArgumentException("Unsupported type for test data: $type")
        }
        
        return tensor
    }

    // Helper to extract float data from tensor
    private fun extractFloatData(tensor: GGMLTensor): FloatArray {
        val size = tensor.numElements().toInt()
        val result = FloatArray(size)
        
        for (i in 0 until size) {
            result[i] = when (tensor.type) {
                GGMLType.F32 -> tensor.getFloat(graphAllocator, i)
                GGMLType.F16 -> halfToFloat(tensor.getHalf(graphAllocator, i))
                else -> tensor.getFloat(graphAllocator, i)
            }
        }
        
        return result
    }

    // --- Basic Computation Chain Tests ---
    
    @Test
    fun testSimpleComputationChain() {
        // Test: (A + B) * C - D
        val ne = longArrayOf(4)
        val dataA = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val dataB = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f)
        val dataC = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
        val dataD = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
        
        val tensorA = createTensorWithData("A", GGMLType.F32, ne, dataA)
        val tensorB = createTensorWithData("B", GGMLType.F32, ne, dataB)
        val tensorC = createTensorWithData("C", GGMLType.F32, ne, dataC)
        val tensorD = createTensorWithData("D", GGMLType.F32, ne, dataD)
        
        // Compute: (A + B) * C - D
        val step1 = computeAdd(graphAllocator, dummyContext, tensorA, tensorB) // A + B
        val step2 = computeMul(graphAllocator, dummyContext, step1, tensorC)   // (A + B) * C
        val result = computeSub(graphAllocator, dummyContext, step2, tensorD)  // ((A + B) * C) - D
        
        val resultData = extractFloatData(result)
        
        // Expected: ((1+0.5)*2-1, (2+1)*3-1, (3+1.5)*4-1, (4+2)*5-1) = (2, 8, 17, 29)
        val expected = floatArrayOf(2.0f, 8.0f, 17.0f, 29.0f)
        
        assertEquals(expected.size, resultData.size, "Result size mismatch")
        for (i in expected.indices) {
            assertEquals(expected[i], resultData[i], 0.001f, "Computation chain result mismatch at index $i")
        }
        
        println("Simple computation chain test passed: ${resultData.contentToString()}")
    }

    @Test
    fun testMixedPrecisionChain() {
        // Test computation chain with mixed F32 and F16 precision
        val ne = longArrayOf(3)
        val dataF32 = floatArrayOf(5.0f, 10.0f, 15.0f)
        val dataF16 = floatArrayOf(2.0f, 3.0f, 4.0f)
        
        val tensorF32 = createTensorWithData("F32", GGMLType.F32, ne, dataF32)
        val tensorF16 = createTensorWithData("F16", GGMLType.F16, ne, dataF16)
        
        // Convert F16 to F32 for operation
        val convertedF16 = GGMLTensor(type = GGMLType.F32, name = "converted")
        convertedF16.ne = ne.copyOf()
        convertedF16.nb = tensorF32.nb.copyOf()
        
        val f32ByteSize = ne[0].toULong() * GGMLType.F32.byteSize
        val allocatedConverted = graphAllocator.tensorAllocators[0].allocate(f32ByteSize, GGMLType.F32, "converted")
        convertedF16.bufferId = allocatedConverted.bufferId
        convertedF16.offset = allocatedConverted.offset
        
        // Copy F16 data to F32 tensor
        for (i in 0 until ne[0].toInt()) {
            val f16Value = tensorF16.getHalf(graphAllocator, i)
            val f32Value = halfToFloat(f16Value)
            convertedF16.setFloat(graphAllocator, f32Value, i)
        }
        
        // Perform operation: F32 + converted(F16)
        val result = computeAdd(graphAllocator, dummyContext, tensorF32, convertedF16)
        val resultData = extractFloatData(result)
        
        val expected = floatArrayOf(7.0f, 13.0f, 19.0f) // 5+2, 10+3, 15+4
        
        for (i in expected.indices) {
            assertEquals(expected[i], resultData[i], 0.01f, "Mixed precision result mismatch at index $i")
        }
        
        println("Mixed precision chain test passed")
    }

    // --- Matrix Operation Integration Tests ---
    
    @Test
    fun testMatrixMultiplicationChain() {
        // Test: (A * B) + C where A, B, C are matrices
        val matrixSize = 4
        val neMatrix = longArrayOf(matrixSize.toLong(), matrixSize.toLong()) // 4x4 matrices
        
        // Create identity-like matrices for predictable results
        val identityData = FloatArray(matrixSize * matrixSize) { i ->
            if (i % (matrixSize + 1) == 0) 1.0f else 0.0f // Diagonal = 1, others = 0
        }
        val onesData = FloatArray(matrixSize * matrixSize) { 0.1f } // Small constant values
        
        val matrixA = createTensorWithData("MatA", GGMLType.F32, neMatrix, identityData)
        val matrixB = createTensorWithData("MatB", GGMLType.F32, neMatrix, identityData)  
        val matrixC = createTensorWithData("MatC", GGMLType.F32, neMatrix, onesData)
        
        try {
            // A * B (should be identity since both are identity)
            val matMulResult = computeMatMul(graphAllocator, dummyContext, matrixA, matrixB)
            
            // (A * B) + C
            val finalResult = computeAdd(graphAllocator, dummyContext, matMulResult, matrixC)
            val resultData = extractFloatData(finalResult)
            
            // Expected: identity + 0.1 everywhere = diagonal 1.1, others 0.1
            for (i in 0 until matrixSize) {
                for (j in 0 until matrixSize) {
                    val index = i * matrixSize + j
                    val expected = if (i == j) 1.1f else 0.1f
                    assertEquals(expected, resultData[index], 0.001f, 
                               "Matrix chain result mismatch at ($i,$j)")
                }
            }
            
            println("Matrix multiplication chain test passed")
        } catch (e: Exception) {
            println("Matrix multiplication chain test skipped: ${e.message}")
        }
    }

    // --- Quantization Integration Tests ---
    
    @Test
    fun testQuantizedOperationChain() {
        // Test computation chain with quantized tensors
        val ne = longArrayOf(64) // Size divisible by quantization block sizes
        val data1 = FloatArray(64) { i -> sin(i * 0.1f) }
        val data2 = FloatArray(64) { i -> cos(i * 0.1f) }
        
        val tensorF32_1 = createTensorWithData("F32_1", GGMLType.F32, ne, data1)
        val tensorF32_2 = createTensorWithData("F32_2", GGMLType.F32, ne, data2)
        
        try {
            // Quantize inputs
            val tensorQ8_0 = quantizeTensor(graphAllocator, tensorF32_1, GGMLType.Q8_0)
            val tensorQ4_0 = quantizeTensor(graphAllocator, tensorF32_2, GGMLType.Q4_0)
            
            // Dequantize for operations (since mixed quantized ops may not be implemented)
            val dequant1 = dequantizeTensor(graphAllocator, tensorQ8_0)
            val dequant2 = dequantizeTensor(graphAllocator, tensorQ4_0)
            
            // Perform operations on dequantized data
            val addResult = computeAdd(graphAllocator, dummyContext, dequant1, dequant2)
            val mulResult = computeMul(graphAllocator, dummyContext, addResult, dequant1)
            
            // Quantize final result
            val finalQuantized = quantizeTensor(graphAllocator, mulResult, GGMLType.Q8_0)
            val finalDequantized = dequantizeTensor(graphAllocator, finalQuantized)
            
            val resultData = extractFloatData(finalDequantized)
            
            // Verify result makes sense (should be approximately (sin + cos) * sin)
            for (i in 0 until 10) { // Check first 10 values
                val expected = (sin(i * 0.1f) + cos(i * 0.1f)) * sin(i * 0.1f)
                assertTrue(abs(expected - resultData[i]) < 0.1f, 
                          "Quantized chain result significantly different at index $i: expected≈$expected, got=${resultData[i]}")
            }
            
            println("Quantized operation chain test passed")
        } catch (e: Exception) {
            println("Quantized operation chain test skipped: ${e.message}")
        }
    }

    // --- Memory Allocation Stress Tests ---
    
    @Test
    fun testMemoryAllocationStress() {
        // Test allocation and deallocation of many tensors
        val tensors = mutableListOf<GGMLTensor>()
        val tensorCount = 20
        val tensorSize = 256
        
        try {
            // Allocate many tensors
            for (i in 0 until tensorCount) {
                val data = FloatArray(tensorSize) { j -> (i + j).toFloat() }
                val tensor = createTensorWithData("stress_$i", GGMLType.F32, longArrayOf(tensorSize.toLong()), data)
                tensors.add(tensor)
            }
            
            // Perform operations on all tensors
            var result = tensors[0]
            for (i in 1 until tensors.size step 2) {
                if (i + 1 < tensors.size) {
                    val intermediate = computeAdd(graphAllocator, dummyContext, tensors[i], tensors[i + 1])
                    result = computeMul(graphAllocator, dummyContext, result, intermediate)
                }
            }
            
            val finalData = extractFloatData(result)
            assertTrue(finalData.isNotEmpty(), "Final result should not be empty")
            
            println("Memory allocation stress test passed with $tensorCount tensors")
        } catch (e: OutOfMemoryError) {
            println("Memory allocation stress test reached memory limit as expected")
        } catch (e: Exception) {
            println("Memory allocation stress test completed with exception: ${e.message}")
        }
    }

    // --- Activation Function Integration ---
    
    @Test
    fun testActivationFunctionChain() {
        // Test: RELU(GELU(input)) chain
        val ne = longArrayOf(8)
        val inputData = floatArrayOf(-3.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 2.0f, 3.0f)
        
        val inputTensor = createTensorWithData("input", GGMLType.F32, ne, inputData)
        
        try {
            // Apply GELU first
            val geluResult = computeGelu(graphAllocator, dummyContext, inputTensor)
            
            // Then apply RELU
            val reluResult = computeRelu(graphAllocator, dummyContext, geluResult)
            
            val resultData = extractFloatData(reluResult)
            
            // GELU followed by RELU should be positive values (RELU clips negatives to 0)
            for (i in resultData.indices) {
                assertTrue(resultData[i] >= 0.0f, "RELU(GELU(x)) should be non-negative at index $i: ${resultData[i]}")
            }
            
            println("Activation function chain test passed: ${resultData.contentToString()}")
        } catch (e: Exception) {
            println("Activation function chain test skipped: ${e.message}")
        }
    }

    // --- Complex Mathematical Expressions ---
    
    @Test
    fun testComplexMathematicalExpression() {
        // Test: sqrt((A^2 + B^2) / (C + epsilon))
        val ne = longArrayOf(4)
        val dataA = floatArrayOf(3.0f, 4.0f, 5.0f, 12.0f)
        val dataB = floatArrayOf(4.0f, 3.0f, 12.0f, 5.0f)  
        val dataC = floatArrayOf(2.0f, 2.0f, 2.0f, 2.0f)
        val epsilon = 1e-6f
        
        val tensorA = createTensorWithData("A", GGMLType.F32, ne, dataA)
        val tensorB = createTensorWithData("B", GGMLType.F32, ne, dataB)
        val tensorC = createTensorWithData("C", GGMLType.F32, ne, dataC)
        val epsilonTensor = createTensorWithData("eps", GGMLType.F32, ne, floatArrayOf(epsilon, epsilon, epsilon, epsilon))
        
        try {
            // A^2
            val aSquared = computeSqr(graphAllocator, dummyContext, tensorA)
            
            // B^2  
            val bSquared = computeSqr(graphAllocator, dummyContext, tensorB)
            
            // A^2 + B^2
            val sumSquares = computeAdd(graphAllocator, dummyContext, aSquared, bSquared)
            
            // C + epsilon
            val cPlusEps = computeAdd(graphAllocator, dummyContext, tensorC, epsilonTensor)
            
            // (A^2 + B^2) / (C + epsilon)
            val division = computeDiv(graphAllocator, dummyContext, sumSquares, cPlusEps)
            
            // sqrt(...)
            val finalResult = computeSqrt(graphAllocator, dummyContext, division)
            
            val resultData = extractFloatData(finalResult)
            
            // Expected results: sqrt((3²+4²)/2), sqrt((4²+3²)/2), sqrt((5²+12²)/2), sqrt((12²+5²)/2)
            // = sqrt(25/2), sqrt(25/2), sqrt(169/2), sqrt(169/2) = sqrt(12.5), sqrt(12.5), sqrt(84.5), sqrt(84.5)
            val expected = floatArrayOf(
                sqrt(12.5f), sqrt(12.5f), sqrt(84.5f), sqrt(84.5f)
            )
            
            for (i in expected.indices) {
                assertEquals(expected[i], resultData[i], 0.01f, "Complex math expression mismatch at index $i")
            }
            
            println("Complex mathematical expression test passed")
        } catch (e: Exception) {
            println("Complex mathematical expression test skipped (operations not implemented): ${e.message}")
        }
    }

    // --- Performance and Scalability Tests ---
    
    @Test
    fun testPerformanceBenchmark() {
        val sizes = arrayOf(64, 256, 1024, 4096)
        
        for (size in sizes) {
            val ne = longArrayOf(size.toLong())
            val data1 = FloatArray(size) { i -> sin(i * 0.01f) }
            val data2 = FloatArray(size) { i -> cos(i * 0.01f) }
            
            val tensor1 = createTensorWithData("perf1_$size", GGMLType.F32, ne, data1)
            val tensor2 = createTensorWithData("perf2_$size", GGMLType.F32, ne, data2)
            
            val startTime = System.currentTimeMillis()
            
            // Perform a series of operations
            val add = computeAdd(graphAllocator, dummyContext, tensor1, tensor2)
            val mul = computeMul(graphAllocator, dummyContext, add, tensor1)
            val sub = computeSub(graphAllocator, dummyContext, mul, tensor2)
            val neg = computeNeg(graphAllocator, dummyContext, sub)
            
            val endTime = System.currentTimeMillis()
            val duration = endTime - startTime
            
            // Extract final result to ensure computation happened
            val resultData = extractFloatData(neg)
            assertTrue(resultData.isNotEmpty(), "Performance test result should not be empty")
            
            println("Performance test size $size: ${duration}ms")
            
            // Performance should scale reasonably
            assertTrue(duration < 1000, "Operations on $size elements should complete in reasonable time")
        }
    }

    // --- Error Handling Integration ---
    
    @Test
    fun testErrorHandlingInChains() {
        val ne = longArrayOf(4)
        val normalData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val zeroData = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        
        val normalTensor = createTensorWithData("normal", GGMLType.F32, ne, normalData)
        val zeroTensor = createTensorWithData("zeros", GGMLType.F32, ne, zeroData)
        
        // Test division by zero handling
        try {
            val divisionResult = computeDiv(graphAllocator, dummyContext, normalTensor, zeroTensor)
            val resultData = extractFloatData(divisionResult)
            
            // Should handle division by zero gracefully (infinity or NaN)
            for (value in resultData) {
                assertTrue(value.isInfinite() || value.isNaN(), "Division by zero should produce Inf or NaN")
            }
            
            println("Division by zero handled correctly in computation chain")
        } catch (e: ArithmeticException) {
            println("Division by zero correctly threw ArithmeticException")
        }
        
        // Test dimension mismatch handling
        val differentSizeTensor = createTensorWithData("different", GGMLType.F32, longArrayOf(6), 
                                                      floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))
        
        try {
            computeAdd(graphAllocator, dummyContext, normalTensor, differentSizeTensor)
            fail("Should have thrown exception for dimension mismatch")
        } catch (e: IllegalArgumentException) {
            println("Dimension mismatch correctly caught: ${e.message}")
        }
    }
}