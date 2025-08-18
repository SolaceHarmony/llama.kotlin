package ai.solace.llamakotlin.core

import kotlin.math.abs
import kotlin.math.pow
import kotlin.test.*

class GGMLKQuantAccuracyTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 2 * 1024 * 1024 // 2MB for larger K-Quant blocks

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())
        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    private fun createAndPopulateF32Tensor(
        name: String,
        dims: LongArray,
        values: FloatArray,
        dataOffset: ULong = 0uL,
        bufferId: Int = 0
    ): GGMLTensor {
        val tensor = GGMLTensor(GGMLType.F32)
        tensor.name = name
        
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dimSize ->
            if (index < GGML_MAX_DIMS) tensor.ne[index] = dimSize
        }
        
        // Calculate strides
        tensor.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        if (tensor.type.byteSize > 0uL) {
            tensor.nb[0] = tensor.type.byteSize
            for (d in 1 until GGML_MAX_DIMS) {
                val prevDimSize = tensor.ne.getOrElse(d - 1) { 1L }
                tensor.nb[d] = tensor.nb[d-1] * (if (prevDimSize > 0) prevDimSize.toULong() else 1uL)
            }
        }
        
        tensor.bufferId = bufferId
        tensor.dataOffset = dataOffset
        tensor.data = null

        val numElements = tensor.numElements().toInt()
        require(values.size == numElements) { "Provided FloatArray size (${values.size}) must match tensor element count ($numElements)." }

        // Set tensor values
        for (i in values.indices) {
            val indices = IntArray(GGML_MAX_DIMS)
            var tempIdx = i.toLong()
            for (dim in 0 until GGML_MAX_DIMS) {
                if (tensor.ne[dim] > 0) {
                    indices[dim] = (tempIdx % tensor.ne[dim]).toInt()
                    tempIdx /= tensor.ne[dim]
                }
            }
            tensor.setFloat(graphAllocator, values[i], *indices)
        }
        
        return tensor
    }

    private fun getTensorDataAsFloatArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        val floatArray = FloatArray(numElements)
        
        for (i in 0 until numElements) {
            val indices = IntArray(GGML_MAX_DIMS)
            var tempIdx = i.toLong()
            for (dim in 0 until GGML_MAX_DIMS) {
                if (tensor.ne[dim] > 0) {
                    indices[dim] = (tempIdx % tensor.ne[dim]).toInt()
                    tempIdx /= tensor.ne[dim]
                }
            }
            
            floatArray[i] = when (tensor.type) {
                GGMLType.F32 -> tensor.getFloat(graphAllocator, *indices)
                GGMLType.F16 -> tensor.getHalf(graphAllocator, *indices)
                else -> throw IllegalArgumentException("Unsupported tensor type ${tensor.type} for direct float array extraction.")
            }
        }
        
        return floatArray
    }

    private fun calculateMeanSquaredError(original: FloatArray, new: FloatArray): Double {
        require(original.size == new.size) { "Arrays must have the same size for MSE." }
        if (original.isEmpty()) return 0.0

        var sumOfSquaredErrors = 0.0
        for (i in original.indices) {
            sumOfSquaredErrors += (original[i] - new[i]).toDouble().pow(2)
        }
        return sumOfSquaredErrors / original.size
    }

    private fun calculateMeanAbsoluteDifference(original: FloatArray, new: FloatArray): Double {
        require(original.size == new.size) { "Arrays must have the same size for MAD." }
        if (original.isEmpty()) return 0.0
        
        var sumAbsDiff = 0.0
        for (i in original.indices) {
            sumAbsDiff += abs(original[i] - new[i]).toDouble()
        }
        return sumAbsDiff / original.size
    }

    @Test
    fun testQ2_KAccuracy() {
        val numElements = QK_K * 2 // Test with 2 blocks
        val originalF32Data = FloatArray(numElements) { i ->
            when {
                i % QK_K < QK_K/4 -> (i % QK_K).toFloat() / (QK_K/4).toFloat() * 2.0f - 1.0f // Block 1: -1.0 to 1.0
                i % QK_K < QK_K/2 -> if ((i % QK_K) % 2 == 0) 0.5f else -0.5f // Block 1: alternating
                i % QK_K < 3*QK_K/4 -> (i % QK_K - QK_K/2).toFloat() / (QK_K/4).toFloat() * 0.25f // Block 1: small range
                else -> ((i % QK_K - 3*QK_K/4).toFloat() / (QK_K/4).toFloat()) * 3.0f - 1.5f // Block 1: wider range
            }
        }

        val dims = longArrayOf(numElements.toLong())
        val f32SrcTensor = createAndPopulateF32Tensor("f32Src_Q2K_Test", dims, originalF32Data, dataOffset = 0uL)

        // Quantize to Q2_K
        val q2kTensor = quantizeTensor(graphAllocator, f32SrcTensor, GGMLType.Q2_K)
        assertEquals(GGMLType.Q2_K, q2kTensor.type)
        assertTrue(q2kTensor.ne.contentEquals(f32SrcTensor.ne))
        assertNotNull(q2kTensor.data)
        assertTrue(q2kTensor.data is ByteArray)

        // Dequantize Q2_K back to F32
        val f32DequantizedTensor = dequantizeTensor(graphAllocator, q2kTensor)
        assertEquals(GGMLType.F32, f32DequantizedTensor.type)
        assertTrue(f32DequantizedTensor.ne.contentEquals(f32SrcTensor.ne))
        assertNotNull(f32DequantizedTensor.data)
        assertTrue(f32DequantizedTensor.data is FloatArray)

        // Extract data for comparison
        val dequantizedF32Data = getTensorDataAsFloatArray(f32DequantizedTensor, graphAllocator)

        // Perform accuracy assertions
        assertEquals(originalF32Data.size, dequantizedF32Data.size)

        val mse = calculateMeanSquaredError(originalF32Data, dequantizedF32Data)
        val mseThreshold = 0.1 // Q2_K has lower precision, so higher threshold
        assertTrue(mse < mseThreshold, "Q2_K MSE $mse too high (threshold $mseThreshold)")

        val mad = calculateMeanAbsoluteDifference(originalF32Data, dequantizedF32Data)
        val madThreshold = 0.3
        assertTrue(mad < madThreshold, "Q2_K Mean Absolute Difference $mad too high (threshold $madThreshold)")
    }

    @Test
    fun testQ4_KAccuracy() {
        val numElements = QK_K * 2 // Test with 2 blocks
        val originalF32Data = FloatArray(numElements) { i ->
            when (i / QK_K) {
                0 -> (i % QK_K).toFloat() / QK_K.toFloat() * 4.0f - 2.0f // Block 1: -2.0 to 2.0
                else -> if ((i % QK_K) % 16 < 8) 1.0f else -1.0f // Block 2: alternating blocks of 1.0/-1.0
            }
        }

        val dims = longArrayOf(numElements.toLong())
        val f32SrcTensor = createAndPopulateF32Tensor("f32Src_Q4K_Test", dims, originalF32Data, dataOffset = 0uL)

        // Quantize to Q4_K
        val q4kTensor = quantizeTensor(graphAllocator, f32SrcTensor, GGMLType.Q4_K)
        assertEquals(GGMLType.Q4_K, q4kTensor.type)
        assertTrue(q4kTensor.ne.contentEquals(f32SrcTensor.ne))
        assertNotNull(q4kTensor.data)
        assertTrue(q4kTensor.data is ByteArray)

        // Dequantize back to F32
        val f32DequantizedTensor = dequantizeTensor(graphAllocator, q4kTensor)
        assertEquals(GGMLType.F32, f32DequantizedTensor.type)
        assertNotNull(f32DequantizedTensor.data)
        assertTrue(f32DequantizedTensor.data is FloatArray)

        // Compare accuracy
        val dequantizedF32Data = getTensorDataAsFloatArray(f32DequantizedTensor, graphAllocator)
        assertEquals(originalF32Data.size, dequantizedF32Data.size)

        val mse = calculateMeanSquaredError(originalF32Data, dequantizedF32Data)
        val mseThreshold = 0.05 // Q4_K has better precision than Q2_K
        assertTrue(mse < mseThreshold, "Q4_K MSE $mse too high (threshold $mseThreshold)")

        val mad = calculateMeanAbsoluteDifference(originalF32Data, dequantizedF32Data)
        val madThreshold = 0.2
        assertTrue(mad < madThreshold, "Q4_K Mean Absolute Difference $mad too high (threshold $madThreshold)")
    }

    @Test
    fun testQ8_KAccuracy() {
        val numElements = QK_K * 1 // Test with 1 block
        val originalF32Data = FloatArray(numElements) { i ->
            when {
                i < QK_K/4 -> i.toFloat() / (QK_K/4).toFloat() * 10.0f // 0 to 10
                i < QK_K/2 -> -(i - QK_K/4).toFloat() / (QK_K/4).toFloat() * 10.0f // 0 to -10
                i < 3*QK_K/4 -> if (i % 2 == 0) 50.0f else -50.0f // Alternating large values
                else -> (i - 3*QK_K/4).toFloat() * 0.1f - 1.0f // Small values around -1
            }
        }

        val dims = longArrayOf(numElements.toLong())
        val f32SrcTensor = createAndPopulateF32Tensor("f32Src_Q8K_Test", dims, originalF32Data, dataOffset = 0uL)

        // Quantize to Q8_K
        val q8kTensor = quantizeTensor(graphAllocator, f32SrcTensor, GGMLType.Q8_K)
        assertEquals(GGMLType.Q8_K, q8kTensor.type)
        assertTrue(q8kTensor.ne.contentEquals(f32SrcTensor.ne))
        assertNotNull(q8kTensor.data)
        assertTrue(q8kTensor.data is ByteArray)

        // Dequantize back to F32
        val f32DequantizedTensor = dequantizeTensor(graphAllocator, q8kTensor)
        assertEquals(GGMLType.F32, f32DequantizedTensor.type)

        // Compare accuracy
        val dequantizedF32Data = getTensorDataAsFloatArray(f32DequantizedTensor, graphAllocator)
        assertEquals(originalF32Data.size, dequantizedF32Data.size)

        val mse = calculateMeanSquaredError(originalF32Data, dequantizedF32Data)
        val mseThreshold = 0.02 // Q8_K should have good precision
        assertTrue(mse < mseThreshold, "Q8_K MSE $mse too high (threshold $mseThreshold)")

        val mad = calculateMeanAbsoluteDifference(originalF32Data, dequantizedF32Data)
        val madThreshold = 0.1
        assertTrue(mad < madThreshold, "Q8_K Mean Absolute Difference $mad too high (threshold $madThreshold)")
    }
}