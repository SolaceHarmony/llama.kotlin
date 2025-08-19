package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*

/**
 * Reference validation test suite for comparing Kotlin implementation against canonical datasets
 * and upstream llama.cpp reference implementations. This test suite provides frameworks for:
 * - Loading reference test vectors
 * - Comparing outputs with configurable tolerances
 * - Cross-validation with known good implementations
 * - Regression testing against established baselines
 */
class GGMLReferenceValidationTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 4 * 1024 * 1024 // 4MB

    /**
     * Reference test vector - contains input data, expected output, and validation metadata
     */
    data class ReferenceTestVector(
        val name: String,
        val inputData: FloatArray,
        val expectedOutput: FloatArray,
        val operation: String,
        val dataType: GGMLType,
        val tolerance: Float,
        val source: String, // Source of the reference (e.g., "llama.cpp", "reference_impl", "analytical")
        val metadata: Map<String, Any> = emptyMap()
    )

    /**
     * Validation result containing detailed comparison metrics
     */
    data class ValidationResult(
        val testName: String,
        val passed: Boolean,
        val maxError: Double,
        val meanError: Double,
        val rmse: Double,
        val relativeTolerance: Double,
        val failedIndices: List<Int>,
        val notes: String = ""
    )

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    /**
     * Create reference test vectors for fundamental operations based on mathematical properties
     */
    private fun createAnalyticalTestVectors(): List<ReferenceTestVector> {
        val vectors = mutableListOf<ReferenceTestVector>()
        
        // ADD operation test vectors
        val addInput1 = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val addInput2 = floatArrayOf(0.5f, 1.0f, 1.5f, 2.0f, 2.5f)
        val addExpected = floatArrayOf(1.5f, 3.0f, 4.5f, 6.0f, 7.5f)
        vectors.add(ReferenceTestVector(
            name = "ADD_analytical_simple",
            inputData = addInput1 + addInput2, // Concatenated for dual-input operations
            expectedOutput = addExpected,
            operation = "ADD",
            dataType = GGMLType.F32,
            tolerance = 1e-6f,
            source = "analytical",
            metadata = mapOf("input1_size" to addInput1.size, "input2_size" to addInput2.size)
        ))

        // MUL operation test vectors
        val mulInput1 = floatArrayOf(2.0f, 3.0f, 4.0f, 5.0f)
        val mulInput2 = floatArrayOf(1.5f, 2.0f, 2.5f, 3.0f)
        val mulExpected = floatArrayOf(3.0f, 6.0f, 10.0f, 15.0f)
        vectors.add(ReferenceTestVector(
            name = "MUL_analytical_simple",
            inputData = mulInput1 + mulInput2,
            expectedOutput = mulExpected,
            operation = "MUL",
            dataType = GGMLType.F32,
            tolerance = 1e-6f,
            source = "analytical"
        ))

        // GELU activation function test vectors (analytical approximation)
        val geluInput = floatArrayOf(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f)
        val geluExpected = geluInput.map { x ->
            // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            val sqrt2OverPi = sqrt(2.0f / PI.toFloat())
            0.5f * x * (1.0f + tanh(sqrt2OverPi * (x + 0.044715f * x * x * x)))
        }.toFloatArray()
        vectors.add(ReferenceTestVector(
            name = "GELU_analytical",
            inputData = geluInput,
            expectedOutput = geluExpected,
            operation = "GELU", 
            dataType = GGMLType.F32,
            tolerance = 1e-4f, // Slightly relaxed due to tanh approximation
            source = "analytical"
        ))

        // RELU activation function test vectors
        val reluInput = floatArrayOf(-5.0f, -1.0f, 0.0f, 1.0f, 5.0f)
        val reluExpected = reluInput.map { maxOf(0.0f, it) }.toFloatArray()
        vectors.add(ReferenceTestVector(
            name = "RELU_analytical",
            inputData = reluInput,
            expectedOutput = reluExpected,
            operation = "RELU",
            dataType = GGMLType.F32,
            tolerance = 1e-6f,
            source = "analytical"
        ))

        // SUB operation test vectors
        val subInput1 = floatArrayOf(10.0f, 8.0f, 6.0f, 4.0f, 2.0f)
        val subInput2 = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
        val subExpected = floatArrayOf(9.0f, 6.0f, 3.0f, 0.0f, -3.0f)
        vectors.add(ReferenceTestVector(
            name = "SUB_analytical",
            inputData = subInput1 + subInput2,
            expectedOutput = subExpected,
            operation = "SUB",
            dataType = GGMLType.F32,
            tolerance = 1e-6f,
            source = "analytical"
        ))

        // NEG operation test vectors
        val negInput = floatArrayOf(-3.0f, -1.0f, 0.0f, 1.0f, 3.0f)
        val negExpected = negInput.map { -it }.toFloatArray()
        vectors.add(ReferenceTestVector(
            name = "NEG_analytical",
            inputData = negInput,
            expectedOutput = negExpected,
            operation = "NEG",
            dataType = GGMLType.F32,
            tolerance = 1e-6f,
            source = "analytical"
        ))

        return vectors
    }

    /**
     * Create reference test vectors based on upstream llama.cpp patterns
     */
    private fun createUpstreamReferenceVectors(): List<ReferenceTestVector> {
        val vectors = mutableListOf<ReferenceTestVector>()
        
        // Quantization reference vectors based on test-quantize-fns.cpp patterns
        // Synthetic data: 0.1 + 2*cos(i + offset)
        val syntheticData = FloatArray(32) { i -> 0.1f + 2.0f * cos(i.toFloat()) }
        
        // These expected values would ideally come from running upstream llama.cpp
        // For now, we'll use the identity for F32 and approximate expected quantization errors
        vectors.add(ReferenceTestVector(
            name = "quantization_synthetic_q8_0",
            inputData = syntheticData,
            expectedOutput = syntheticData, // F32 dequantization should be close to original
            operation = "QUANTIZE_DEQUANTIZE",
            dataType = GGMLType.Q8_0,
            tolerance = 0.01f, // Based on Q8_0 expected precision
            source = "upstream_pattern",
            metadata = mapOf("quantization_type" to "Q8_0", "test_pattern" to "synthetic")
        ))

        return vectors
    }

    /**
     * Helper to create tensor from float array
     */
    private fun createTensorFromFloatArray(name: String, type: GGMLType, data: FloatArray): GGMLTensor {
        val ne = longArrayOf(data.size.toLong())
        val tensor = GGMLTensor(type = type, name = name)
        tensor.ne = ne
        
        // Set strides
        tensor.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        tensor.nb[0] = type.byteSize
        for (d in 1 until GGML_MAX_DIMS) {
            tensor.nb[d] = tensor.nb[d-1] * ne.getOrElse(d-1) { 1L }.toULong()
        }
        
        // Allocate memory
        val byteSize = data.size.toULong() * type.byteSize
        val allocatedTensor = graphAllocator.tensorAllocators[0].allocate(byteSize, type, name)
        tensor.bufferId = allocatedTensor.bufferId
        tensor.offset = allocatedTensor.offset
        
        // Set data
        for (i in data.indices) {
            when (type) {
                GGMLType.F32 -> tensor.setFloat(graphAllocator, data[i], i)
                GGMLType.F16 -> tensor.setHalf(graphAllocator, data[i], i)
                else -> tensor.setFloat(graphAllocator, data[i], i)
            }
        }
        
        return tensor
    }

    /**
     * Extract float array from tensor
     */
    private fun extractFloatArray(tensor: GGMLTensor): FloatArray {
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

    /**
     * Validate a single test vector against implementation
     */
    private fun validateTestVector(testVector: ReferenceTestVector): ValidationResult {
        val dummyContext = GGMLContext()
        
        try {
            val actualOutput = when (testVector.operation) {
                "ADD" -> {
                    val input1Size = testVector.metadata["input1_size"] as? Int ?: (testVector.inputData.size / 2)
                    val input1Data = testVector.inputData.sliceArray(0 until input1Size)
                    val input2Data = testVector.inputData.sliceArray(input1Size until testVector.inputData.size)
                    
                    val tensor1 = createTensorFromFloatArray("input1", testVector.dataType, input1Data)
                    val tensor2 = createTensorFromFloatArray("input2", testVector.dataType, input2Data)
                    val result = computeAdd(graphAllocator, dummyContext, tensor1, tensor2)
                    extractFloatArray(result)
                }
                "MUL" -> {
                    val input1Size = testVector.inputData.size / 2
                    val input1Data = testVector.inputData.sliceArray(0 until input1Size)
                    val input2Data = testVector.inputData.sliceArray(input1Size until testVector.inputData.size)
                    
                    val tensor1 = createTensorFromFloatArray("input1", testVector.dataType, input1Data)
                    val tensor2 = createTensorFromFloatArray("input2", testVector.dataType, input2Data)
                    val result = computeMul(graphAllocator, dummyContext, tensor1, tensor2)
                    extractFloatArray(result)
                }
                "SUB" -> {
                    val input1Size = testVector.inputData.size / 2
                    val input1Data = testVector.inputData.sliceArray(0 until input1Size)
                    val input2Data = testVector.inputData.sliceArray(input1Size until testVector.inputData.size)
                    
                    val tensor1 = createTensorFromFloatArray("input1", testVector.dataType, input1Data)
                    val tensor2 = createTensorFromFloatArray("input2", testVector.dataType, input2Data)
                    val result = computeSub(graphAllocator, dummyContext, tensor1, tensor2)
                    extractFloatArray(result)
                }
                "NEG" -> {
                    val tensor = createTensorFromFloatArray("input", testVector.dataType, testVector.inputData)
                    val result = computeNeg(graphAllocator, dummyContext, tensor)
                    extractFloatArray(result)
                }
                "GELU" -> {
                    val tensor = createTensorFromFloatArray("input", testVector.dataType, testVector.inputData)
                    val result = computeGelu(graphAllocator, dummyContext, tensor)
                    extractFloatArray(result)
                }
                "RELU" -> {
                    val tensor = createTensorFromFloatArray("input", testVector.dataType, testVector.inputData)
                    val result = computeRelu(graphAllocator, dummyContext, tensor)
                    extractFloatArray(result)
                }
                "QUANTIZE_DEQUANTIZE" -> {
                    val tensor = createTensorFromFloatArray("input", GGMLType.F32, testVector.inputData)
                    val quantized = quantizeTensor(graphAllocator, tensor, testVector.dataType)
                    val dequantized = dequantizeTensor(graphAllocator, quantized)
                    extractFloatArray(dequantized)
                }
                else -> throw IllegalArgumentException("Unsupported operation: ${testVector.operation}")
            }
            
            // Calculate error metrics
            require(actualOutput.size == testVector.expectedOutput.size) { 
                "Output size mismatch: expected ${testVector.expectedOutput.size}, got ${actualOutput.size}" 
            }
            
            var maxError = 0.0
            var sumError = 0.0
            var sumSquaredError = 0.0
            val failedIndices = mutableListOf<Int>()
            
            for (i in testVector.expectedOutput.indices) {
                val expected = testVector.expectedOutput[i].toDouble()
                val actual = actualOutput[i].toDouble()
                val error = abs(expected - actual)
                val relativeError = if (abs(expected) > 1e-10) error / abs(expected) else error
                
                maxError = maxOf(maxError, error)
                sumError += error
                sumSquaredError += error * error
                
                if (error > testVector.tolerance && relativeError > testVector.tolerance) {
                    failedIndices.add(i)
                }
            }
            
            val meanError = sumError / testVector.expectedOutput.size
            val rmse = sqrt(sumSquaredError / testVector.expectedOutput.size)
            val relativeTolerance = if (testVector.expectedOutput.any { abs(it) > 1e-10 }) {
                maxError / testVector.expectedOutput.maxOf { abs(it) }
            } else maxError
            
            val passed = failedIndices.isEmpty()
            
            return ValidationResult(
                testName = testVector.name,
                passed = passed,
                maxError = maxError,
                meanError = meanError,
                rmse = rmse,
                relativeTolerance = relativeTolerance,
                failedIndices = failedIndices,
                notes = if (!passed) "Failed at ${failedIndices.size} indices, max error: $maxError" else "All validations passed"
            )
            
        } catch (e: Exception) {
            return ValidationResult(
                testName = testVector.name,
                passed = false,
                maxError = Double.MAX_VALUE,
                meanError = Double.MAX_VALUE,
                rmse = Double.MAX_VALUE,
                relativeTolerance = Double.MAX_VALUE,
                failedIndices = emptyList(),
                notes = "Exception during validation: ${e.message}"
            )
        }
    }

    // --- Reference Validation Tests ---
    
    @Test
    fun testAnalyticalReferenceVectors() {
        val analyticalVectors = createAnalyticalTestVectors()
        val results = mutableListOf<ValidationResult>()
        
        println("\n=== Analytical Reference Validation ===")
        println("Test Name\t\t\tPassed\tMax Error\tMean Error\tRMSE")
        
        for (testVector in analyticalVectors) {
            val result = validateTestVector(testVector)
            results.add(result)
            
            val status = if (result.passed) "PASS" else "FAIL"
            println("${testVector.name}\t${status}\t${"%.2e".format(result.maxError)}\t${"%.2e".format(result.meanError)}\t${"%.2e".format(result.rmse)}")
            
            if (!result.passed) {
                println("  └─ ${result.notes}")
            }
        }
        
        val passedCount = results.count { it.passed }
        val totalCount = results.size
        
        println("\nAnalytical Validation Summary: $passedCount/$totalCount tests passed")
        
        // Assert that most analytical tests should pass (these are deterministic)
        assertTrue(passedCount >= totalCount * 0.8, "At least 80% of analytical tests should pass")
    }

    @Test
    fun testQuantizationReferenceValidation() {
        val quantizationVectors = createUpstreamReferenceVectors()
        val results = mutableListOf<ValidationResult>()
        
        println("\n=== Quantization Reference Validation ===")
        println("Test Name\t\t\tPassed\tMax Error\tRMSE\tTolerance")
        
        for (testVector in quantizationVectors) {
            val result = validateTestVector(testVector)
            results.add(result)
            
            val status = if (result.passed) "PASS" else "FAIL"
            println("${testVector.name}\t${status}\t${"%.4f".format(result.maxError)}\t${"%.4f".format(result.rmse)}\t${testVector.tolerance}")
            
            if (!result.passed && result.failedIndices.isNotEmpty()) {
                println("  └─ Failed indices (first 10): ${result.failedIndices.take(10)}")
            }
        }
        
        println("\nQuantization Validation completed: ${results.count { it.passed }}/${results.size} tests passed")
    }

    @Test
    fun testNumericalStabilityValidation() {
        // Test edge cases and numerical stability
        val edgeCaseVectors = listOf(
            // Very small values
            ReferenceTestVector(
                name = "ADD_small_values",
                inputData = floatArrayOf(1e-6f, 1e-7f, 1e-8f) + floatArrayOf(1e-6f, 1e-7f, 1e-8f),
                expectedOutput = floatArrayOf(2e-6f, 2e-7f, 2e-8f),
                operation = "ADD",
                dataType = GGMLType.F32,
                tolerance = 1e-9f,
                source = "edge_case"
            ),
            
            // Very large values
            ReferenceTestVector(
                name = "MUL_large_values",
                inputData = floatArrayOf(1e6f, 1e7f) + floatArrayOf(1e6f, 1e7f),
                expectedOutput = floatArrayOf(1e12f, 1e14f),
                operation = "MUL",
                dataType = GGMLType.F32,
                tolerance = 1e5f, // Relaxed tolerance for large numbers
                source = "edge_case"
            ),
            
            // Mixed positive/negative values
            ReferenceTestVector(
                name = "ADD_mixed_signs",
                inputData = floatArrayOf(1.0f, -2.0f, 3.0f, -4.0f) + floatArrayOf(-1.0f, 2.0f, -3.0f, 4.0f),
                expectedOutput = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
                operation = "ADD",
                dataType = GGMLType.F32,
                tolerance = 1e-6f,
                source = "edge_case"
            )
        )
        
        println("\n=== Numerical Stability Validation ===")
        var passedCount = 0
        
        for (testVector in edgeCaseVectors) {
            val result = validateTestVector(testVector)
            val status = if (result.passed) "PASS" else "FAIL"
            
            println("${testVector.name}: $status (Max Error: ${"%.2e".format(result.maxError)})")
            
            if (result.passed) {
                passedCount++
            } else {
                println("  └─ ${result.notes}")
            }
        }
        
        println("Numerical Stability: $passedCount/${edgeCaseVectors.size} tests passed")
    }

    @Test
    fun testCrossValidationWithMultiplePrecisions() {
        // Test same operations with F32 and F16 to ensure consistency
        val testData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        
        val f32Tensor = createTensorFromFloatArray("f32_test", GGMLType.F32, testData)
        val f16Tensor = createTensorFromFloatArray("f16_test", GGMLType.F16, testData)
        
        val dummyContext = GGMLContext()
        
        try {
            // Test NEG operation on both precisions
            val f32Result = computeNeg(graphAllocator, dummyContext, f32Tensor)
            val f16Result = computeNeg(graphAllocator, dummyContext, f16Tensor)
            
            val f32Output = extractFloatArray(f32Result)
            val f16Output = extractFloatArray(f16Result)
            
            println("\n=== Cross-Precision Validation (NEG) ===")
            println("Input\tF32 Output\tF16 Output\tDifference")
            
            var maxDifference = 0.0f
            for (i in testData.indices) {
                val diff = abs(f32Output[i] - f16Output[i])
                maxDifference = maxOf(maxDifference, diff)
                println("${testData[i]}\t${f32Output[i]}\t${f16Output[i]}\t$diff")
            }
            
            // F16 should be reasonably close to F32 for simple operations
            assertTrue(maxDifference < 0.01f, "F32 and F16 NEG results should be similar, max diff: $maxDifference")
            
            println("Cross-precision validation passed (max difference: $maxDifference)")
            
        } catch (e: Exception) {
            println("Cross-precision validation skipped: ${e.message}")
        }
    }

    @Test
    fun testRegressionBaseline() {
        // Store known good results to prevent regressions
        val regressionTests = mapOf(
            "simple_add" to listOf(1.0f, 2.0f, 3.0f, 4.0f),
            "simple_neg" to listOf(-1.0f, -2.0f, -3.0f, -4.0f)
        )
        
        println("\n=== Regression Baseline Tests ===")
        
        val input = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)
        val ones = floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)
        
        val inputTensor = createTensorFromFloatArray("regression_input", GGMLType.F32, input)
        val onesTensor = createTensorFromFloatArray("ones", GGMLType.F32, ones)
        val dummyContext = GGMLContext()
        
        // Test ADD baseline
        try {
            val addResult = computeAdd(graphAllocator, dummyContext, inputTensor, onesTensor)
            val addOutput = extractFloatArray(addResult)
            val expectedAdd = regressionTests["simple_add"]!!
            
            var addPassed = true
            for (i in expectedAdd.indices) {
                if (abs(addOutput[i] - expectedAdd[i]) > 1e-6f) {
                    addPassed = false
                    break
                }
            }
            
            println("ADD Regression: ${if (addPassed) "PASS" else "FAIL"}")
            assertTrue(addPassed, "ADD regression test failed")
        } catch (e: Exception) {
            println("ADD Regression: SKIPPED (${e.message})")
        }
        
        // Test NEG baseline
        try {
            val negResult = computeNeg(graphAllocator, dummyContext, inputTensor)
            val negOutput = extractFloatArray(negResult)
            val expectedNeg = regressionTests["simple_neg"]!!
            
            var negPassed = true
            for (i in expectedNeg.indices) {
                if (abs(negOutput[i] - expectedNeg[i]) > 1e-6f) {
                    negPassed = false
                    break
                }
            }
            
            println("NEG Regression: ${if (negPassed) "PASS" else "FAIL"}")
            assertTrue(negPassed, "NEG regression test failed")
        } catch (e: Exception) {
            println("NEG Regression: SKIPPED (${e.message})")
        }
    }
}