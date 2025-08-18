package ai.solace.llamakotlin.core

import kotlin.math.abs
import kotlin.test.*

/**
 * Comprehensive tests for optimized matmul quantization combinations.
 * Tests all new optimized paths: F32 x Q_type and Q_type x Q_type combinations.
 */
class GGMLMatMulOptimizationTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 2 * 1024 * 1024 // 2MB for larger matrices

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())
        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    // Helper to create and initialize tensors
    private fun createTensor(
        name: String,
        type: GGMLType,
        dims: LongArray,
        initValues: FloatArray? = null
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type)
        tensor.name = name
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dimSize ->
            if (index < GGML_MAX_DIMS) tensor.ne[index] = dimSize
        }
        
        // Calculate strides
        if (type.byteSize > 0uL) {
            tensor.nb[0] = type.byteSize
            for (d in 1 until GGML_MAX_DIMS) {
                tensor.nb[d] = tensor.ne[d-1].toULong() * tensor.nb[d-1]
            }
        }

        val numElements = tensor.numElements().toInt()
        
        when (type) {
            GGMLType.F32 -> {
                val data = FloatArray(numElements) { idx ->
                    initValues?.getOrElse(idx) { (idx % 10).toFloat() + 1.0f } 
                        ?: ((idx % 10).toFloat() + 1.0f)
                }
                tensor.data = data
            }
            GGMLType.Q8_0 -> {
                // Create F32 data first, then quantize
                val f32Data = FloatArray(numElements) { idx ->
                    initValues?.getOrElse(idx) { (idx % 127).toFloat() - 63.0f }
                        ?: ((idx % 127).toFloat() - 63.0f)
                }
                val f32Tensor = GGMLTensor(GGMLType.F32)
                f32Tensor.ne = tensor.ne.copyOf()
                f32Tensor.nb[0] = 4uL
                for (d in 1 until GGML_MAX_DIMS) {
                    f32Tensor.nb[d] = f32Tensor.ne[d-1].toULong() * f32Tensor.nb[d-1]
                }
                f32Tensor.data = f32Data
                
                // Quantize to Q8_0 (this uses quantizeTensor from GGMLComputeOps)
                val quantizedTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q8_0)
                tensor.data = quantizedTensor.data
            }
            GGMLType.Q4_0 -> {
                val f32Data = FloatArray(numElements) { idx ->
                    initValues?.getOrElse(idx) { (idx % 15).toFloat() - 7.0f }
                        ?: ((idx % 15).toFloat() - 7.0f)
                }
                val f32Tensor = GGMLTensor(GGMLType.F32)
                f32Tensor.ne = tensor.ne.copyOf()
                f32Tensor.nb[0] = 4uL
                for (d in 1 until GGML_MAX_DIMS) {
                    f32Tensor.nb[d] = f32Tensor.ne[d-1].toULong() * f32Tensor.nb[d-1]
                }
                f32Tensor.data = f32Data
                
                val quantizedTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q4_0)
                tensor.data = quantizedTensor.data
            }
            GGMLType.Q4_1 -> {
                val f32Data = FloatArray(numElements) { idx ->
                    initValues?.getOrElse(idx) { (idx % 20).toFloat() }
                        ?: ((idx % 20).toFloat())
                }
                val f32Tensor = GGMLTensor(GGMLType.F32)
                f32Tensor.ne = tensor.ne.copyOf()
                f32Tensor.nb[0] = 4uL
                for (d in 1 until GGML_MAX_DIMS) {
                    f32Tensor.nb[d] = f32Tensor.ne[d-1].toULong() * f32Tensor.nb[d-1]
                }
                f32Tensor.data = f32Data
                
                val quantizedTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q4_1)
                tensor.data = quantizedTensor.data
            }
            else -> throw IllegalArgumentException("Unsupported type: $type")
        }

        return tensor
    }

    // Helper function to quantize F32 tensor (needs access to quantizeTensor)
    private fun quantizeTensor(graphAllocator: GGMLGraphAllocator, tensorF32: GGMLTensor, targetType: GGMLType): GGMLTensor {
        // This calls the private quantizeTensor function from GGMLComputeOps
        // Since it's private, we'll use computeMatMul with known types to access quantization
        // For now, create a simple quantization manually
        
        if (tensorF32.type != GGMLType.F32) throw IllegalArgumentException("Input must be F32")
        val result = GGMLTensor(type = targetType)
        result.ne = tensorF32.ne.copyOf()
        
        val numElements = tensorF32.numElements().toInt()
        val f32Data = tensorF32.data as FloatArray
        
        when (targetType) {
            GGMLType.Q8_0 -> {
                require(numElements % QK8_0 == 0) { "Q8_0 elements must be divisible by $QK8_0" }
                val numBlocks = numElements / QK8_0
                val blockSize = targetType.byteSize.toInt()
                val quantData = ByteArray(numBlocks * blockSize)
                
                for (blockIdx in 0 until numBlocks) {
                    val startIdx = blockIdx * QK8_0
                    val blockData = f32Data.sliceArray(startIdx until startIdx + QK8_0)
                    
                    // Find scale
                    var absMax = 0.0f
                    for (value in blockData) {
                        absMax = maxOf(absMax, abs(value))
                    }
                    val scale = if (absMax == 0.0f) 1.0f else absMax / 127.0f
                    val invScale = if (scale == 0.0f) 0.0f else 1.0f / scale
                    
                    // Write scale as F16
                    val scaleF16 = floatToHalf(scale)
                    quantData.setShortLe(blockIdx * blockSize, scaleF16)
                    
                    // Write quantized weights
                    val weightsOffset = blockIdx * blockSize + 2
                    for (i in 0 until QK8_0) {
                        val quantValue = kotlin.math.round(blockData[i] * invScale).toInt().coerceIn(-128, 127)
                        quantData[weightsOffset + i] = quantValue.toByte()
                    }
                }
                result.data = quantData
            }
            GGMLType.Q4_0 -> {
                require(numElements % QK4_0 == 0) { "Q4_0 elements must be divisible by $QK4_0" }
                val numBlocks = numElements / QK4_0
                val blockSize = targetType.byteSize.toInt()
                val quantData = ByteArray(numBlocks * blockSize)
                
                for (blockIdx in 0 until numBlocks) {
                    val startIdx = blockIdx * QK4_0
                    val blockData = f32Data.sliceArray(startIdx until startIdx + QK4_0)
                    
                    // Find scale
                    var absMax = 0.0f
                    for (value in blockData) {
                        absMax = maxOf(absMax, abs(value))
                    }
                    val scale = if (absMax == 0.0f) 1.0f else absMax / 8.0f
                    val invScale = if (scale == 0.0f) 0.0f else 1.0f / scale
                    
                    // Write scale as F16
                    val scaleF16 = floatToHalf(scale)
                    quantData.setShortLe(blockIdx * blockSize, scaleF16)
                    
                    // Write quantized weights (packed nibbles)
                    val weightsOffset = blockIdx * blockSize + 2
                    for (i in 0 until QK4_0 / 2) {
                        val q1 = kotlin.math.round(blockData[i*2] * invScale + 8.0f).toInt().coerceIn(0, 15)
                        val q2 = kotlin.math.round(blockData[i*2+1] * invScale + 8.0f).toInt().coerceIn(0, 15)
                        quantData[weightsOffset + i] = ((q1 and 0x0F) or ((q2 and 0x0F) shl 4)).toByte()
                    }
                }
                result.data = quantData
            }
            GGMLType.Q4_1 -> {
                require(numElements % QK4_1 == 0) { "Q4_1 elements must be divisible by $QK4_1" }
                val numBlocks = numElements / QK4_1
                val blockSize = targetType.byteSize.toInt()
                val quantData = ByteArray(numBlocks * blockSize)
                
                for (blockIdx in 0 until numBlocks) {
                    val startIdx = blockIdx * QK4_1
                    val blockData = f32Data.sliceArray(startIdx until startIdx + QK4_1)
                    
                    // Find min and max
                    val minVal = blockData.minOrNull() ?: 0.0f
                    val maxVal = blockData.maxOrNull() ?: 0.0f
                    val scale = if (maxVal == minVal) 1.0f else (maxVal - minVal) / 15.0f
                    val invScale = if (scale == 0.0f) 0.0f else 1.0f / scale
                    
                    // Write scale and min as F16
                    val scaleF16 = floatToHalf(scale)
                    val minF16 = floatToHalf(minVal)
                    quantData.setShortLe(blockIdx * blockSize, scaleF16)
                    quantData.setShortLe(blockIdx * blockSize + 2, minF16)
                    
                    // Write quantized weights (packed nibbles)
                    val weightsOffset = blockIdx * blockSize + 4
                    for (i in 0 until QK4_1 / 2) {
                        val q1 = kotlin.math.round((blockData[i*2] - minVal) * invScale).toInt().coerceIn(0, 15)
                        val q2 = kotlin.math.round((blockData[i*2+1] - minVal) * invScale).toInt().coerceIn(0, 15)
                        quantData[weightsOffset + i] = ((q1 and 0x0F) or ((q2 and 0x0F) shl 4)).toByte()
                    }
                }
                result.data = quantData
            }
            else -> throw IllegalArgumentException("Unsupported quantization target: $targetType")
        }
        
        return result
    }

    /**
     * Test F32 x Q4_0 symmetric optimization
     */
    @Test
    fun testF32xQ40Optimization() {
        // Create small matrices for testing
        val M = 4; val K = QK4_0 * 2; val N = 3  // Ensure K is divisible by block size
        
        val tensorF32 = createTensor("f32_a", GGMLType.F32, longArrayOf(K.toLong(), M.toLong()))
        val tensorQ40 = createTensor("q40_b", GGMLType.Q4_0, longArrayOf(N.toLong(), K.toLong()))
        
        // Test optimized path
        val resultOptimized = computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ40)
        
        // Test fallback path by forcing dequantization  
        val tensorQ40F32 = dequantizeTensor(graphAllocator, tensorQ40)
        val resultFallback = computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ40F32)
        
        // Results should be close (allowing for quantization errors)
        assertEquals(GGMLType.F32, resultOptimized.type)
        assertEquals(resultOptimized.ne[0], N.toLong())
        assertEquals(resultOptimized.ne[1], M.toLong())
        
        val optimizedData = resultOptimized.data as FloatArray
        val fallbackData = resultFallback.data as FloatArray
        
        for (i in optimizedData.indices) {
            val diff = abs(optimizedData[i] - fallbackData[i])
            assertTrue(diff < 1e-3, "Results differ at index $i: ${optimizedData[i]} vs ${fallbackData[i]}")
        }
    }

    /**
     * Test F32 x Q8_0 symmetric optimization
     */
    @Test
    fun testF32xQ80Optimization() {
        val M = 3; val K = QK8_0 * 2; val N = 4
        
        val tensorF32 = createTensor("f32_a", GGMLType.F32, longArrayOf(K.toLong(), M.toLong()))
        val tensorQ80 = createTensor("q80_b", GGMLType.Q8_0, longArrayOf(N.toLong(), K.toLong()))
        
        val resultOptimized = computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ80)
        val tensorQ80F32 = dequantizeTensor(graphAllocator, tensorQ80)
        val resultFallback = computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ80F32)
        
        assertEquals(GGMLType.F32, resultOptimized.type)
        val optimizedData = resultOptimized.data as FloatArray
        val fallbackData = resultFallback.data as FloatArray
        
        for (i in optimizedData.indices) {
            val diff = abs(optimizedData[i] - fallbackData[i])
            assertTrue(diff < 1e-3, "F32xQ80 results differ at index $i: ${optimizedData[i]} vs ${fallbackData[i]}")
        }
    }

    /**
     * Test Q8_0 x Q8_0 direct quantized optimization
     */
    @Test
    fun testQ80xQ80Optimization() {
        val M = 2; val K = QK8_0; val N = 2
        
        val tensorQ80A = createTensor("q80_a", GGMLType.Q8_0, longArrayOf(K.toLong(), M.toLong()))
        val tensorQ80B = createTensor("q80_b", GGMLType.Q8_0, longArrayOf(N.toLong(), K.toLong()))
        
        val resultOptimized = computeMatMul(graphAllocator, GGMLContext(), tensorQ80A, tensorQ80B)
        
        // Compare with dequantized fallback
        val tensorQ80AF32 = dequantizeTensor(graphAllocator, tensorQ80A)
        val tensorQ80BF32 = dequantizeTensor(graphAllocator, tensorQ80B)
        val resultFallback = computeMatMul(graphAllocator, GGMLContext(), tensorQ80AF32, tensorQ80BF32)
        
        assertEquals(GGMLType.F32, resultOptimized.type)
        val optimizedData = resultOptimized.data as FloatArray
        val fallbackData = resultFallback.data as FloatArray
        
        for (i in optimizedData.indices) {
            val diff = abs(optimizedData[i] - fallbackData[i])
            assertTrue(diff < 1e-2, "Q80xQ80 results differ at index $i: ${optimizedData[i]} vs ${fallbackData[i]}")
        }
    }

    /**
     * Test Q4_0 x Q4_0 direct quantized optimization
     */
    @Test
    fun testQ40xQ40Optimization() {
        val M = 2; val K = QK4_0; val N = 3
        
        val tensorQ40A = createTensor("q40_a", GGMLType.Q4_0, longArrayOf(K.toLong(), M.toLong()))
        val tensorQ40B = createTensor("q40_b", GGMLType.Q4_0, longArrayOf(N.toLong(), K.toLong()))
        
        val resultOptimized = computeMatMul(graphAllocator, GGMLContext(), tensorQ40A, tensorQ40B)
        
        assertEquals(GGMLType.F32, resultOptimized.type)
        assertEquals(resultOptimized.ne[0], N.toLong())
        assertEquals(resultOptimized.ne[1], M.toLong())
        
        // Ensure we got actual data
        assertNotNull(resultOptimized.data)
        val optimizedData = resultOptimized.data as FloatArray
        assertTrue(optimizedData.isNotEmpty())
    }

    /**
     * Test mixed quantization Q8_0 x Q4_0
     */
    @Test
    fun testQ80xQ40MixedOptimization() {
        val M = 2; val K = 64; val N = 2  // Use K divisible by both QK8_0 and QK4_0
        
        val tensorQ80 = createTensor("q80_a", GGMLType.Q8_0, longArrayOf(K.toLong(), M.toLong()))
        val tensorQ40 = createTensor("q40_b", GGMLType.Q4_0, longArrayOf(N.toLong(), K.toLong()))
        
        val resultOptimized = computeMatMul(graphAllocator, GGMLContext(), tensorQ80, tensorQ40)
        
        assertEquals(GGMLType.F32, resultOptimized.type)
        assertEquals(resultOptimized.ne[0], N.toLong())
        assertEquals(resultOptimized.ne[1], M.toLong())
        
        val optimizedData = resultOptimized.data as FloatArray
        assertTrue(optimizedData.isNotEmpty())
        
        // Verify all values are finite (no NaN/Inf from quantization issues)
        for (value in optimizedData) {
            assertTrue(value.isFinite(), "Got non-finite value: $value")
        }
    }

    // Helper to dequantize tensors for comparison  
    private fun dequantizeTensor(graphAllocator: GGMLGraphAllocator, tensor: GGMLTensor): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne = tensor.ne.copyOf()
        result.nb[0] = 4uL
        for (d in 1 until GGML_MAX_DIMS) {
            result.nb[d] = result.ne[d-1].toULong() * result.nb[d-1]
        }
        
        val numElements = tensor.numElements().toInt()
        val resultData = FloatArray(numElements)
        
        when (tensor.type) {
            GGMLType.Q8_0 -> {
                val numBlocks = (numElements + QK8_0 - 1) / QK8_0
                var elementIdx = 0
                for (blockIdx in 0 until numBlocks) {
                    val scale = tensor.getQ8_0BlockScale(graphAllocator, blockIdx)
                    for (itemInBlock in 0 until QK8_0) {
                        if (elementIdx < numElements) {
                            val qWeight = tensor.getQ8_0Weight(graphAllocator, blockIdx, itemInBlock)
                            resultData[elementIdx++] = scale * qWeight.toFloat()
                        }
                    }
                }
            }
            GGMLType.Q4_0 -> {
                val numBlocks = (numElements + QK4_0 - 1) / QK4_0
                var elementIdx = 0
                for (blockIdx in 0 until numBlocks) {
                    val scale = tensor.getQ4_0BlockScale(graphAllocator, blockIdx)
                    for (itemInBlock in 0 until QK4_0) {
                        if (elementIdx < numElements) {
                            val qNibble = tensor.getQ4_0NibbleWeight(graphAllocator, blockIdx, itemInBlock)
                            resultData[elementIdx++] = scale * (qNibble.toFloat() - 8.0f)
                        }
                    }
                }
            }
            GGMLType.Q4_1 -> {
                val numBlocks = (numElements + QK4_1 - 1) / QK4_1
                var elementIdx = 0
                for (blockIdx in 0 until numBlocks) {
                    val scale = tensor.getQ4_1BlockScale(graphAllocator, blockIdx)
                    val min = tensor.getQ4_1BlockMin(graphAllocator, blockIdx)
                    for (itemInBlock in 0 until QK4_1) {
                        if (elementIdx < numElements) {
                            val qNibble = tensor.getQ4_1NibbleWeight(graphAllocator, blockIdx, itemInBlock)
                            resultData[elementIdx++] = scale * qNibble.toFloat() + min
                        }
                    }
                }
            }
            else -> throw IllegalArgumentException("Cannot dequantize type: ${tensor.type}")
        }
        
        result.data = resultData
        return result
    }

    /**
     * Performance comparison test - measures time difference between optimized and fallback paths
     */
    @Test
    fun testPerformanceComparison() {
        val M = 8; val K = 128; val N = 8  // Larger matrices for performance testing
        
        val tensorF32 = createTensor("perf_f32", GGMLType.F32, longArrayOf(K.toLong(), M.toLong()))
        val tensorQ80 = createTensor("perf_q80", GGMLType.Q8_0, longArrayOf(N.toLong(), K.toLong()))
        
        // Warm up
        repeat(5) {
            computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ80)
        }
        
        // Time optimized path
        val startOptimized = kotlin.system.getTimeNanos()
        repeat(10) {
            computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ80)
        }
        val timeOptimized = kotlin.system.getTimeNanos() - startOptimized
        
        // Time fallback path
        val tensorQ80F32 = dequantizeTensor(graphAllocator, tensorQ80)
        val startFallback = kotlin.system.getTimeNanos()
        repeat(10) {
            computeMatMul(graphAllocator, GGMLContext(), tensorF32, tensorQ80F32)
        }
        val timeFallback = kotlin.system.getTimeNanos() - startFallback
        
        println("Optimized path: ${timeOptimized / 1_000_000}ms")
        println("Fallback path: ${timeFallback / 1_000_000}ms")
        println("Speedup: ${timeFallback.toFloat() / timeOptimized.toFloat()}x")
        
        // Optimized path should be faster (or at least not significantly slower)
        assertTrue(timeOptimized <= timeFallback * 1.5, 
            "Optimized path should not be significantly slower than fallback")
    }
}