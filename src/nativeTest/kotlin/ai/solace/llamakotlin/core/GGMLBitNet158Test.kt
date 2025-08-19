package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*
import kotlin.random.Random

/**
 * Comprehensive test suite for BitNet 1.58 quantization.
 * 
 * BitNet 1.58 uses ternary weights (-1, 0, +1) with a single scale factor per block.
 * This provides approximately 1.58 bits per weight (log2(3) ≈ 1.58).
 * 
 * Test coverage includes:
 * - Basic tensor creation and data access
 * - Quantization/dequantization accuracy
 * - Round-trip precision validation 
 * - Tensor operations and matrix multiplication
 * - Edge cases and error conditions
 */
class GGMLBitNet158Test {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 4 * 1024 * 1024 // 4MB

    // Error thresholds for BitNet 1.58
    companion object {
        // BitNet 1.58 has limited precision due to ternary quantization
        const val MSE_THRESHOLD_BITNET_158 = 0.2f // Mean Squared Error
        const val MAD_THRESHOLD_BITNET_158 = 0.5f // Mean Absolute Deviation  
        const val SNR_THRESHOLD_BITNET_158 = 8.0 // Signal-to-Noise Ratio in dB
        const val MAX_TERNARY_ERROR = 0.05f // Maximum error for ternary conversion
    }

    @BeforeTest
    fun setup() {
        testBuffer = ByteArray(bufferSize) { 0 }
        graphAllocator = GGMLGraphAllocator()
        graphAllocator.buffers[0] = testBuffer
        
        // Initialize a context 
        val context = GGMLContext()
        graphAllocator.context = context
    }

    @AfterTest
    fun cleanup() {
        // Clean up if needed
    }

    // === Utility Functions ===
    
    private fun createF32TestTensor(name: String, data: FloatArray): GGMLTensor {
        val tensor = GGMLTensor(type = GGMLType.F32, name = name)
        tensor.ne[0] = data.size.toLong()
        tensor.ne[1] = 1L
        tensor.ne[2] = 1L  
        tensor.ne[3] = 1L
        tensor.nb = calculateContiguousStrides(tensor.ne, GGMLType.F32, tensor.rank())
        
        // Calculate tensor size and allocate
        val tensorByteSize = calculateTensorByteSize(tensor).toInt()
        val offset = graphAllocator.allocateTensorData(tensorByteSize)
        tensor.bufferId = 0
        tensor.dataOffset = offset.toULong()
        
        // Copy data
        for (i in data.indices) {
            tensor.setFloat(graphAllocator, data[i], i)
        }
        
        return tensor
    }
    
    private fun generateTestData(size: Int, seed: Int = 42): FloatArray {
        val random = Random(seed)
        return FloatArray(size) { random.nextFloat() * 4.0f - 2.0f } // Range: -2.0 to +2.0
    }
    
    private fun generateTernaryData(size: Int, seed: Int = 42): FloatArray {
        val random = Random(seed)
        return FloatArray(size) { 
            when (random.nextInt(3)) {
                0 -> -1.0f
                1 -> 0.0f
                else -> 1.0f
            }
        }
    }
    
    private fun extractFloatData(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        val result = FloatArray(numElements)
        
        when (tensor.type) {
            GGMLType.F32 -> {
                for (i in 0 until numElements) {
                    result[i] = tensor.getFloat(graphAllocator, i)
                }
            }
            else -> {
                throw IllegalArgumentException("extractFloatData only supports F32 tensors currently")
            }
        }
        
        return result
    }
    
    private fun calculateQuantizationMetrics(original: FloatArray, quantized: FloatArray): QuantizationMetrics {
        require(original.size == quantized.size) { "Arrays must have same size" }
        
        var mse = 0.0f
        var mad = 0.0f
        var signalPower = 0.0f
        var noisePower = 0.0f
        
        for (i in original.indices) {
            val diff = original[i] - quantized[i]
            mse += diff * diff
            mad += abs(diff)
            signalPower += original[i] * original[i]
            noisePower += diff * diff
        }
        
        val n = original.size
        mse /= n
        mad /= n
        signalPower /= n
        noisePower /= n
        
        val snr = if (noisePower > 0) 10 * log10(signalPower / noisePower) else Double.POSITIVE_INFINITY
        
        return QuantizationMetrics(mse, mad, snr.toFloat(), sqrt(mse))
    }
    
    data class QuantizationMetrics(
        val mse: Float,      // Mean Squared Error
        val mad: Float,      // Mean Absolute Deviation
        val snr: Float,      // Signal-to-Noise Ratio in dB
        val totalError: Float // Root Mean Squared Error
    )

    // === Basic Functionality Tests ===

    @Test
    fun testBitNet158TensorCreation() {
        // Test creating a BitNet 1.58 tensor
        val originalData = generateTestData(QK_BITNET_1_58 * 4) // 4 blocks
        val f32Tensor = createF32TestTensor("test_f32", originalData)
        
        // Quantize to BitNet 1.58
        val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
        
        // Verify properties
        assertEquals(GGMLType.BITNET_1_58, bitNetTensor.type)
        assertEquals(originalData.size.toLong(), bitNetTensor.numElements())
        assertEquals(4L, bitNetTensor.getNumBlocks())
        
        println("BitNet 1.58 tensor created successfully with ${bitNetTensor.numElements()} elements in ${bitNetTensor.getNumBlocks()} blocks")
    }

    @Test
    fun testBitNet158BlockAccessors() {
        val blockSize = QK_BITNET_1_58
        val numBlocks = 2
        val originalData = generateTernaryData(blockSize * numBlocks)
        val f32Tensor = createF32TestTensor("test_block_access", originalData)
        
        val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
        
        // Test block scale accessors
        for (blockIdx in 0 until numBlocks) {
            val scale = bitNetTensor.getBitNet158BlockScale(graphAllocator, blockIdx)
            assertTrue(scale > 0, "Block scale should be positive, got $scale")
            println("Block $blockIdx scale: $scale")
        }
        
        // Test ternary weight accessors
        for (blockIdx in 0 until numBlocks) {
            for (itemIdx in 0 until blockSize) {
                val weight = bitNetTensor.getBitNet158TernaryWeight(graphAllocator, blockIdx, itemIdx)
                assertTrue(weight in -1..1, "Ternary weight should be -1, 0, or 1, got $weight")
                
                // Test weight setting (round trip)
                val originalWeight = weight
                bitNetTensor.setBitNet158TernaryWeight(graphAllocator, blockIdx, itemIdx, 1)
                assertEquals(1, bitNetTensor.getBitNet158TernaryWeight(graphAllocator, blockIdx, itemIdx))
                
                // Restore original
                bitNetTensor.setBitNet158TernaryWeight(graphAllocator, blockIdx, itemIdx, originalWeight)
                assertEquals(originalWeight, bitNetTensor.getBitNet158TernaryWeight(graphAllocator, blockIdx, itemIdx))
            }
        }
    }

    @Test
    fun testBitNet158QuantizationAccuracy() {
        val testSizes = listOf(32, 64, 128, 256)
        
        for (size in testSizes) {
            // Ensure size is a multiple of block size
            val adjustedSize = (size / QK_BITNET_1_58) * QK_BITNET_1_58
            if (adjustedSize <= 0) continue
            
            val originalData = generateTestData(adjustedSize, size)
            val f32Tensor = createF32TestTensor("test_accuracy_$size", originalData)
            
            // Quantize to BitNet 1.58
            val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
            assertEquals(GGMLType.BITNET_1_58, bitNetTensor.type)
            
            // Dequantize back to F32
            val dequantizedTensor = dequantizeTensor(graphAllocator, bitNetTensor)
            assertEquals(GGMLType.F32, dequantizedTensor.type)
            
            val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
            val metrics = calculateQuantizationMetrics(originalData, dequantizedData)
            
            // Validate metrics against thresholds
            assertTrue(metrics.mse < MSE_THRESHOLD_BITNET_158, 
                "BitNet 1.58 MSE ${metrics.mse} exceeds threshold $MSE_THRESHOLD_BITNET_158 for size $adjustedSize")
            assertTrue(metrics.mad < MAD_THRESHOLD_BITNET_158, 
                "BitNet 1.58 MAD ${metrics.mad} exceeds threshold $MAD_THRESHOLD_BITNET_158 for size $adjustedSize")
            assertTrue(metrics.snr >= SNR_THRESHOLD_BITNET_158, 
                "BitNet 1.58 SNR ${metrics.snr} dB below threshold $SNR_THRESHOLD_BITNET_158 for size $adjustedSize")
            
            println("BitNet 1.58 size $adjustedSize - MSE: ${metrics.mse}, MAD: ${metrics.mad}, SNR: ${metrics.snr} dB")
        }
    }
    
    @Test
    fun testBitNet158TernaryValues() {
        // Test specifically with data that should map cleanly to ternary values
        val perfectTernaryData = floatArrayOf(-1.5f, -0.1f, 0.0f, 0.1f, 1.5f, -2.0f, 0.0f, 2.0f) + 
                                 FloatArray(QK_BITNET_1_58 - 8) { 0.0f } // Pad to block size
        
        val f32Tensor = createF32TestTensor("ternary_test", perfectTernaryData)
        val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
        val dequantizedTensor = dequantizeTensor(graphAllocator, bitNetTensor)
        val dequantizedData = extractFloatData(dequantizedTensor, graphAllocator)
        
        // Check that values are properly quantized to ternary
        for (i in 0 until min(8, dequantizedData.size)) {
            val dequantized = dequantizedData[i]
            val scale = bitNetTensor.getBitNet158BlockScale(graphAllocator, 0)
            val normalizedValue = dequantized / scale
            
            // Should be close to -1, 0, or +1
            val closestTernary = when {
                normalizedValue > 0.5f -> 1.0f
                normalizedValue < -0.5f -> -1.0f
                else -> 0.0f
            }
            
            assertTrue(abs(normalizedValue - closestTernary) < MAX_TERNARY_ERROR, 
                "Value $dequantized (normalized: $normalizedValue) should be close to ternary value $closestTernary")
        }
        
        println("BitNet 1.58 ternary quantization test passed")
    }

    @Test
    fun testBitNet158DotProduct() {
        val commonDim = QK_BITNET_1_58 * 2 // 2 blocks
        
        // Create BitNet 1.58 tensor
        val bitNetData = generateTernaryData(commonDim)
        val bitNetTensor = quantizeTensor(graphAllocator, 
            createF32TestTensor("bitnet_dot", bitNetData), GGMLType.BITNET_1_58)
        
        // Create F32 tensor  
        val f32Data = generateTestData(commonDim)
        val f32Tensor = createF32TestTensor("f32_dot", f32Data)
        f32Tensor.ne[0] = 1L // N=1 (single column)
        f32Tensor.ne[1] = commonDim.toLong() // K=commonDim
        
        // Test dot product
        val dotResult = computeDotProductBitNet158F32(
            graphAllocator, bitNetTensor, f32Tensor, 
            rowIndexInBitNet = 0, colIndexInF32 = 0, commonDimK = commonDim
        )
        
        // Verify result is reasonable (not NaN, not infinite)
        assertTrue(dotResult.isFinite(), "Dot product result should be finite, got $dotResult")
        
        // Compare with reference implementation using dequantized tensors
        val dequantizedBitNet = dequantizeTensor(graphAllocator, bitNetTensor)
        val dequantizedData = extractFloatData(dequantizedBitNet, graphAllocator)
        
        var referenceDot = 0.0f
        for (i in 0 until commonDim) {
            referenceDot += dequantizedData[i] * f32Data[i]
        }
        
        val error = abs(dotResult - referenceDot)
        assertTrue(error < 1.0f, "Dot product error $error should be reasonable")
        
        println("BitNet 1.58 dot product: $dotResult, reference: $referenceDot, error: $error")
    }

    @Test
    fun testBitNet158MatrixMultiplication() {
        // Test small matrix multiplication: (2x64) * (64x3) = (2x3)
        val M = 2
        val K = QK_BITNET_1_58 * 2 // 2 blocks
        val N = 3
        
        // Create input matrices
        val bitNetData = generateTernaryData(M * K)
        val f32Data = generateTestData(K * N)
        
        val bitNetTensor = quantizeTensor(graphAllocator, 
            createF32TestTensor("bitnet_matmul_a", bitNetData), GGMLType.BITNET_1_58)
        bitNetTensor.ne[0] = K.toLong()
        bitNetTensor.ne[1] = M.toLong()
        
        val f32Tensor = createF32TestTensor("f32_matmul_b", f32Data)
        f32Tensor.ne[0] = N.toLong()
        f32Tensor.ne[1] = K.toLong()
        
        // Create result tensor
        val resultTensor = GGMLTensor(type = GGMLType.F32, name = "result")
        resultTensor.ne[0] = N.toLong()
        resultTensor.ne[1] = M.toLong()
        resultTensor.nb = calculateContiguousStrides(resultTensor.ne, GGMLType.F32, resultTensor.rank())
        
        val resultSize = calculateTensorByteSize(resultTensor).toInt()
        val resultOffset = graphAllocator.allocateTensorData(resultSize)
        resultTensor.bufferId = 0
        resultTensor.dataOffset = resultOffset.toULong()
        
        // Perform matrix multiplication using the existing computeMatMul infrastructure
        val context = GGMLContext()
        computeMatMul(graphAllocator, context, bitNetTensor, f32Tensor, resultTensor)
        
        // Verify result dimensions
        assertEquals(N.toLong(), resultTensor.ne[0])
        assertEquals(M.toLong(), resultTensor.ne[1])
        
        // Check that results are finite
        for (i in 0 until M) {
            for (j in 0 until N) {
                val value = resultTensor.getFloat(graphAllocator, j, i)
                assertTrue(value.isFinite(), "Result[$i,$j] should be finite, got $value")
            }
        }
        
        println("BitNet 1.58 matrix multiplication test passed (${M}x${K}) * (${K}x${N}) = (${M}x${N})")
    }

    @Test
    fun testBitNet158EdgeCases() {
        // Test with all zeros
        val zerosData = FloatArray(QK_BITNET_1_58) { 0.0f }
        val zerosTensor = quantizeTensor(graphAllocator, 
            createF32TestTensor("zeros", zerosData), GGMLType.BITNET_1_58)
        val dequantizedZeros = extractFloatData(dequantizeTensor(graphAllocator, zerosTensor), graphAllocator)
        
        for (value in dequantizedZeros) {
            assertEquals(0.0f, value, 0.001f, "Zero values should remain zero after quantization")
        }
        
        // Test with extreme values
        val extremeData = FloatArray(QK_BITNET_1_58) { if (it % 2 == 0) Float.MAX_VALUE else Float.MIN_VALUE }
        val extremeTensor = quantizeTensor(graphAllocator, 
            createF32TestTensor("extreme", extremeData), GGMLType.BITNET_1_58)
        
        // Should not crash and should produce finite results
        val dequantizedExtreme = extractFloatData(dequantizeTensor(graphAllocator, extremeTensor), graphAllocator)
        for (value in dequantizedExtreme) {
            assertTrue(value.isFinite(), "Extreme values should produce finite results after quantization")
        }
        
        println("BitNet 1.58 edge cases test passed")
    }

    @Test
    fun testBitNet158ErrorHandling() {
        val validData = generateTestData(QK_BITNET_1_58)
        val f32Tensor = createF32TestTensor("error_test", validData)
        val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
        
        // Test invalid block index
        assertFailsWith<IllegalArgumentException> {
            bitNetTensor.getBitNet158BlockScale(graphAllocator, -1)
        }
        
        assertFailsWith<IllegalArgumentException> {
            bitNetTensor.getBitNet158BlockScale(graphAllocator, 999)
        }
        
        // Test invalid item index
        assertFailsWith<IllegalArgumentException> {
            bitNetTensor.getBitNet158TernaryWeight(graphAllocator, 0, -1)
        }
        
        assertFailsWith<IllegalArgumentException> {
            bitNetTensor.getBitNet158TernaryWeight(graphAllocator, 0, QK_BITNET_1_58)
        }
        
        // Test invalid ternary value
        assertFailsWith<IllegalArgumentException> {
            bitNetTensor.setBitNet158TernaryWeight(graphAllocator, 0, 0, 2) // Valid range is -1 to 1
        }
        
        assertFailsWith<IllegalArgumentException> {
            bitNetTensor.setBitNet158TernaryWeight(graphAllocator, 0, 0, -2)
        }
        
        println("BitNet 1.58 error handling test passed")
    }

    @Test
    fun testBitNet158Performance() {
        val sizes = listOf(1024, 2048, 4096)
        
        for (size in sizes) {
            val adjustedSize = (size / QK_BITNET_1_58) * QK_BITNET_1_58
            if (adjustedSize <= 0) continue
            
            val data = generateTestData(adjustedSize)
            val f32Tensor = createF32TestTensor("perf_test_$size", data)
            
            // Measure quantization time
            val startQuant = kotlin.system.getTimeMillis()
            val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
            val quantTime = kotlin.system.getTimeMillis() - startQuant
            
            // Measure dequantization time
            val startDequant = kotlin.system.getTimeMillis()
            val dequantizedTensor = dequantizeTensor(graphAllocator, bitNetTensor)
            val dequantTime = kotlin.system.getTimeMillis() - startDequant
            
            println("BitNet 1.58 size $adjustedSize - Quantization: ${quantTime}ms, Dequantization: ${dequantTime}ms")
            
            // Verify correctness wasn't sacrificed for performance
            assertNotNull(bitNetTensor.data, "Quantization should produce valid data")
            assertEquals(GGMLType.BITNET_1_58, bitNetTensor.type)
            assertEquals(GGMLType.F32, dequantizedTensor.type)
        }
    }
}