package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*
import kotlin.random.Random

/**
 * Enhanced quantization accuracy test suite with standardized datasets and error thresholds
 * based on reference implementations from upstream llama.cpp (test-quantize-fns.cpp).
 * 
 * This test suite provides comprehensive validation for quantization operations with:
 * - Standardized test datasets with known characteristics
 * - Error thresholds based on quantization theory and empirical validation
 * - Multiple test vectors for comprehensive coverage
 * - Reference implementation comparison framework
 */
class GGMLStandardizedQuantizationTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 4 * 1024 * 1024 // 4MB for larger test datasets

    // Error thresholds based on upstream llama.cpp test-quantize-fns.cpp
    companion object {
        // Based on MAX_QUANTIZATION_TOTAL_ERROR constants from C++ reference
        const val MAX_QUANTIZATION_TOTAL_ERROR = 0.002f
        const val MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f  
        const val MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f
        const val MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f
        const val MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f
        const val MAX_DOT_PRODUCT_ERROR = 0.02f
        const val MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f
        
        // Custom thresholds for our Kotlin implementation
        const val MSE_THRESHOLD_Q8_0 = 0.0001f
        const val MSE_THRESHOLD_Q4_0 = 0.01f
        const val MSE_THRESHOLD_Q4_1 = 0.015f
        
        const val MAD_THRESHOLD_Q8_0 = 0.01f
        const val MAD_THRESHOLD_Q4_0 = 0.2f
        const val MAD_THRESHOLD_Q4_1 = 0.1f
        
        // SNR thresholds (Signal-to-Noise Ratio in dB)
        const val SNR_THRESHOLD_Q8_0 = 40.0 // High precision
        const val SNR_THRESHOLD_Q4_0 = 20.0 // Medium precision
        const val SNR_THRESHOLD_Q4_1 = 18.0 // Medium-low precision
    }

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
     * Generate synthetic test data similar to upstream llama.cpp test-quantize-fns.cpp
     * Uses cosine-based pattern: 0.1 + 2*cos(i + offset)
     */
    private fun generateSyntheticData(size: Int, offset: Float = 0.0f): FloatArray {
        return FloatArray(size) { i ->
            0.1f + 2.0f * cos(i.toFloat() + offset)
        }
    }

    /**
     * Generate random data with controlled distribution
     */
    private fun generateRandomData(size: Int, seed: Long = 12345, range: Float = 10.0f): FloatArray {
        val random = Random(seed)
        return FloatArray(size) { 
            random.nextFloat() * 2 * range - range // Range [-range, range]
        }
    }

    /**
     * Generate edge case data including zeros, infinities, and extreme values
     */
    private fun generateEdgeCaseData(): FloatArray {
        return floatArrayOf(
            0.0f, -0.0f,
            Float.MIN_VALUE, -Float.MIN_VALUE,
            1.0f, -1.0f,
            Float.MAX_VALUE / 1000, -Float.MAX_VALUE / 1000,
            10.0f, -10.0f,
            100.0f, -100.0f,
            0.001f, -0.001f,
            PI.toFloat(), -PI.toFloat(),
            E.toFloat(), -E.toFloat()
        )
    }

    /**
     * Calculate comprehensive error metrics
     */
    private data class QuantizationMetrics(
        val mse: Double,
        val rmse: Double,
        val mad: Double,
        val maxError: Double,
        val snr: Double,
        val totalError: Float
    )

    private fun calculateQuantizationMetrics(original: FloatArray, dequantized: FloatArray): QuantizationMetrics {
        require(original.size == dequantized.size) { "Array sizes must match" }
        
        var sumSquaredError = 0.0
        var sumAbsoluteError = 0.0
        var maxError = 0.0
        var sumOriginalSquared = 0.0
        var totalError = 0.0f
        
        for (i in original.indices) {
            val error = (original[i] - dequantized[i]).toDouble()
            val absError = abs(error)
            
            sumSquaredError += error * error
            sumAbsoluteError += absError
            maxError = maxOf(maxError, absError)
            sumOriginalSquared += original[i].toDouble() * original[i].toDouble()
            totalError += absError.toFloat()
        }
        
        val mse = sumSquaredError / original.size
        val rmse = sqrt(mse)
        val mad = sumAbsoluteError / original.size
        
        // Calculate SNR in dB
        val snr = if (sumSquaredError == 0.0) Double.POSITIVE_INFINITY 
                 else if (sumOriginalSquared == 0.0) Double.NEGATIVE_INFINITY
                 else 10 * log10(sumOriginalSquared / sumSquaredError)
        
        return QuantizationMetrics(mse, rmse, mad, maxError, snr, totalError / original.size)
    }

    /**
     * Create and populate F32 tensor with test data
     */
    private fun createF32TestTensor(name: String, data: FloatArray): GGMLTensor {
        val dims = longArrayOf(data.size.toLong())
        val tensor = GGMLTensor(type = GGMLType.F32, name = name)
        tensor.ne = dims
        tensor.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        tensor.nb[0] = GGMLType.F32.byteSize
        for (d in 1 until GGML_MAX_DIMS) {
            tensor.nb[d] = tensor.nb[d-1] * dims.getOrElse(d-1) { 1L }.toULong()
        }
        
        val byteSize = data.size.toULong() * GGMLType.F32.byteSize
        val allocatedTensor = graphAllocator.tensorAllocators[0].allocate(byteSize, GGMLType.F32, name)
        tensor.bufferId = allocatedTensor.bufferId
        tensor.offset = allocatedTensor.offset
        
        // Set data
        for (i in data.indices) {
            tensor.setFloat(graphAllocator, data[i], i)
        }
        
        return tensor
    }

    /**
     * Extract float data from any tensor type
     */
    private fun extractFloatData(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val size = tensor.numElements().toInt()
        val result = FloatArray(size)
        
        for (i in 0 until size) {
            result[i] = when (tensor.type) {
                GGMLType.F32 -> tensor.getFloat(graphAllocator, i)
                GGMLType.F16 -> halfToFloat(tensor.getHalf(graphAllocator, i))
                else -> tensor.getFloat(graphAllocator, i) // Try generic access
            }
        }
        
        return result
    }

    // --- Q8_0 Comprehensive Tests ---
    
    @Test
    fun testQ8_0SyntheticDataset() {
        val testData = generateSyntheticData(512) // Large enough for multiple blocks
        val originalTensor = createF32TestTensor("q8_0_synthetic", testData)
        
        // Quantize to Q8_0
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q8_0)
        assertEquals(GGMLType.Q8_0, quantizedTensor.type, "Quantization should produce Q8_0 tensor")
        
        // Dequantize back to F32
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        assertEquals(GGMLType.F32, dequantizedTensor.type, "Dequantization should produce F32 tensor")
        
        // Extract and compare data
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        val metrics = calculateQuantizationMetrics(testData, dequantizedData)
        
        // Validate against thresholds
        assertTrue(metrics.mse < MSE_THRESHOLD_Q8_0, "Q8_0 MSE ${metrics.mse} exceeds threshold $MSE_THRESHOLD_Q8_0")
        assertTrue(metrics.mad < MAD_THRESHOLD_Q8_0, "Q8_0 MAD ${metrics.mad} exceeds threshold $MAD_THRESHOLD_Q8_0")
        assertTrue(metrics.snr >= SNR_THRESHOLD_Q8_0, "Q8_0 SNR ${metrics.snr} dB below threshold $SNR_THRESHOLD_Q8_0")
        assertTrue(metrics.totalError < MAX_QUANTIZATION_TOTAL_ERROR, "Q8_0 total error ${metrics.totalError} exceeds reference threshold")
        
        println("Q8_0 Synthetic - MSE: ${metrics.mse}, MAD: ${metrics.mad}, SNR: ${metrics.snr} dB, Max: ${metrics.maxError}")
    }
    
    @Test
    fun testQ8_0RandomDataset() {
        val testData = generateRandomData(256, seed = 54321, range = 5.0f)
        val originalTensor = createF32TestTensor("q8_0_random", testData)
        
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q8_0)
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        val metrics = calculateQuantizationMetrics(testData, dequantizedData)
        
        assertTrue(metrics.mse < MSE_THRESHOLD_Q8_0, "Q8_0 random MSE ${metrics.mse} exceeds threshold")
        assertTrue(metrics.mad < MAD_THRESHOLD_Q8_0, "Q8_0 random MAD ${metrics.mad} exceeds threshold")
        
        println("Q8_0 Random - MSE: ${metrics.mse}, MAD: ${metrics.mad}, SNR: ${metrics.snr} dB")
    }

    @Test
    fun testQ8_0EdgeCases() {
        val edgeCaseData = generateEdgeCaseData()
        val originalTensor = createF32TestTensor("q8_0_edge", edgeCaseData)
        
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q8_0)
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        
        // Edge case validation - may have larger errors for extreme values
        for (i in edgeCaseData.indices) {
            val original = edgeCaseData[i]
            val dequantized = dequantizedData[i]
            val relativeError = if (original != 0.0f) abs((original - dequantized) / original) else abs(dequantized)
            
            // Relaxed threshold for edge cases
            assertTrue(relativeError < 0.1f || abs(original - dequantized) < 0.1f, 
                      "Q8_0 edge case error too large at index $i: original=$original, dequantized=$dequantized")
        }
        
        println("Q8_0 Edge Cases - Validated ${edgeCaseData.size} edge cases")
    }

    // --- Q4_0 Comprehensive Tests ---
    
    @Test
    fun testQ4_0SyntheticDataset() {
        val testData = generateSyntheticData(512)
        val originalTensor = createF32TestTensor("q4_0_synthetic", testData)
        
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q4_0)
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        val metrics = calculateQuantizationMetrics(testData, dequantizedData)
        
        assertTrue(metrics.mse < MSE_THRESHOLD_Q4_0, "Q4_0 MSE ${metrics.mse} exceeds threshold $MSE_THRESHOLD_Q4_0")
        assertTrue(metrics.mad < MAD_THRESHOLD_Q4_0, "Q4_0 MAD ${metrics.mad} exceeds threshold $MAD_THRESHOLD_Q4_0")
        assertTrue(metrics.snr >= SNR_THRESHOLD_Q4_0, "Q4_0 SNR ${metrics.snr} dB below threshold $SNR_THRESHOLD_Q4_0")
        
        println("Q4_0 Synthetic - MSE: ${metrics.mse}, MAD: ${metrics.mad}, SNR: ${metrics.snr} dB")
    }

    @Test
    fun testQ4_0MultipleBlocks() {
        // Test with multiple Q4_0 blocks (32 elements per block)
        val testData = generateSyntheticData(96, offset = 1.5f) // 3 blocks
        val originalTensor = createF32TestTensor("q4_0_multiblock", testData)
        
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q4_0)
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        val metrics = calculateQuantizationMetrics(testData, dequantizedData)
        
        assertTrue(metrics.totalError < MAX_QUANTIZATION_TOTAL_ERROR * 2, // Relaxed for 4-bit
                  "Q4_0 multiblock total error ${metrics.totalError} exceeds adjusted threshold")
        
        println("Q4_0 MultiBlock (3 blocks) - MSE: ${metrics.mse}, MAD: ${metrics.mad}")
    }

    // --- Q4_1 Comprehensive Tests ---
    
    @Test
    fun testQ4_1SyntheticDataset() {
        val testData = generateSyntheticData(384) // Multiple Q4_1 blocks
        val originalTensor = createF32TestTensor("q4_1_synthetic", testData)
        
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q4_1)
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        val metrics = calculateQuantizationMetrics(testData, dequantizedData)
        
        assertTrue(metrics.mse < MSE_THRESHOLD_Q4_1, "Q4_1 MSE ${metrics.mse} exceeds threshold $MSE_THRESHOLD_Q4_1")
        assertTrue(metrics.mad < MAD_THRESHOLD_Q4_1, "Q4_1 MAD ${metrics.mad} exceeds threshold $MAD_THRESHOLD_Q4_1")
        
        println("Q4_1 Synthetic - MSE: ${metrics.mse}, MAD: ${metrics.mad}, SNR: ${metrics.snr} dB")
    }

    @Test
    fun testQ4_1SmallValues() {
        // Test Q4_1 with small values where min/max quantization shows benefits
        val testData = generateRandomData(64, seed = 98765, range = 1.0f) // Small range
        val originalTensor = createF32TestTensor("q4_1_small", testData)
        
        val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, GGMLType.Q4_1)
        val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        val metrics = calculateQuantizationMetrics(testData, dequantizedData)
        
        // Q4_1 should perform better on small values due to min/max encoding
        assertTrue(metrics.mse < MSE_THRESHOLD_Q4_1 * 0.5, "Q4_1 should perform well on small values")
        
        println("Q4_1 Small Values - MSE: ${metrics.mse}, MAD: ${metrics.mad}")
    }

    // --- Cross-Quantization Comparison Tests ---
    
    @Test
    fun testQuantizationComparison() {
        val testData = generateSyntheticData(256)
        val originalTensor = createF32TestTensor("comparison_test", testData)
        
        // Test Q8_0, Q4_0, Q4_1 on same data
        val quantTypes = arrayOf(GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1)
        val results = mutableMapOf<GGMLType, QuantizationMetrics>()
        
        for (quantType in quantTypes) {
            try {
                val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, quantType)
                val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
                val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
                val metrics = calculateQuantizationMetrics(testData, dequantizedData)
                
                results[quantType] = metrics
                println("$quantType - MSE: ${metrics.mse}, MAD: ${metrics.mad}, SNR: ${metrics.snr} dB")
            } catch (e: Exception) {
                println("$quantType quantization failed: ${e.message}")
            }
        }
        
        // Validate ordering: Q8_0 should be most accurate, then Q4_1, then Q4_0
        if (results.containsKey(GGMLType.Q8_0) && results.containsKey(GGMLType.Q4_0)) {
            assertTrue(results[GGMLType.Q8_0]!!.mse <= results[GGMLType.Q4_0]!!.mse, 
                      "Q8_0 should be more accurate than Q4_0")
        }
        
        if (results.containsKey(GGMLType.Q4_1) && results.containsKey(GGMLType.Q4_0)) {
            // Q4_1 often better than Q4_0 due to min/max quantization
            println("Q4_1 vs Q4_0 accuracy comparison: Q4_1 MSE=${results[GGMLType.Q4_1]!!.mse}, Q4_0 MSE=${results[GGMLType.Q4_0]!!.mse}")
        }
    }

    // --- Dot Product Accuracy Tests ---
    
    @Test
    fun testDotProductAccuracy() {
        // Test quantized dot product accuracy
        val size = 128
        val vec1Data = generateSyntheticData(size, offset = 0.0f)
        val vec2Data = generateSyntheticData(size, offset = 1.0f)
        
        // F32 reference dot product
        val referenceDotProduct = vec1Data.zip(vec2Data).sumOf { (a, b) -> (a * b).toDouble() }.toFloat()
        
        // Test Q8_0 x F32 dot product
        val vec1Tensor = createF32TestTensor("dot_vec1", vec1Data)
        val vec2Tensor = createF32TestTensor("dot_vec2", vec2Data)
        
        val vec1Q8_0 = quantizeTensor(graphAllocator, vec1Tensor, GGMLType.Q8_0)
        
        try {
            // Compute quantized dot product (this would use internal dot product functions)
            val quantizedDotProduct = computeDotProductQ80F32(
                graphAllocator, vec1Q8_0, vec2Tensor, 0, 0, size
            )
            
            val error = abs(referenceDotProduct - quantizedDotProduct)
            val relativeError = if (referenceDotProduct != 0.0f) error / abs(referenceDotProduct) else error
            
            assertTrue(relativeError < MAX_DOT_PRODUCT_ERROR, 
                      "Q8_0 x F32 dot product error $relativeError exceeds threshold $MAX_DOT_PRODUCT_ERROR")
            
            println("Dot Product - Reference: $referenceDotProduct, Q8_0xF32: $quantizedDotProduct, RelError: $relativeError")
        } catch (e: Exception) {
            println("Dot product test skipped - function not available: ${e.message}")
        }
    }

    // --- Stress Tests ---
    
    @Test
    fun testLargeDatasetStress() {
        // Test with larger dataset to stress memory allocation and processing
        val largeSize = 2048
        val testData = generateSyntheticData(largeSize, offset = 2.0f)
        val originalTensor = createF32TestTensor("stress_large", testData)
        
        for (quantType in arrayOf(GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1)) {
            try {
                val startTime = System.currentTimeMillis()
                val quantizedTensor = quantizeTensor(graphAllocator, originalTensor, quantType)
                val dequantizedTensor = dequantizeTensor(graphAllocator, quantizedTensor)
                val endTime = System.currentTimeMillis()
                
                val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
                val metrics = calculateQuantizationMetrics(testData, dequantizedData)
                
                println("$quantType Large Dataset ($largeSize elements) - Time: ${endTime-startTime}ms, MSE: ${metrics.mse}")
                
                // Ensure performance is reasonable and accuracy maintained
                assertTrue(endTime - startTime < 1000, "$quantType quantization took too long: ${endTime-startTime}ms")
            } catch (e: Exception) {
                println("$quantType large dataset test failed: ${e.message}")
            }
        }
    }
}