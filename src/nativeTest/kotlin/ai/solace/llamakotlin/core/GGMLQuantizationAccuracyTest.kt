package ai.solace.llamakotlin.core

import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt // Not used yet, but often useful for stats
import kotlin.test.*
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES


class GGMLQuantizationAccuracyTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 1 * 1024 * 1024 // 1MB

    // --- Start Copied/Adapted Helpers ---
    // NOTE: These should ideally be in a common test utility file.
    // For now, they are included here for self-containment of this test file.

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

    // Removed local helper calculateTensorByteSize(tensor: GGMLTensor)
    // Will use global ai.solace.llamakotlin.core.calculateTensorByteSize(tensor: GGMLTensor)

    // Copied from GGMLComputeOpsTest.kt - should be in a common test util
    internal fun applyNDIter(
        tensorNe: LongArray, // Pass ne directly
        tensorRank: Int,     // Pass rank directly
        totalElementsToIterate: Int,
        actionPerElement: (flatIdx: Int, indices: IntArray) -> Unit
    ) {
        val n0 = tensorNe.getOrElse(0) { 1L }.toInt()
        val n1 = tensorNe.getOrElse(1) { 1L }.toInt()
        val n2 = tensorNe.getOrElse(2) { 1L }.toInt()
        val n3 = tensorNe.getOrElse(3) { 1L }.toInt()
        var currentFlatIdx = 0

        if (totalElementsToIterate == 0) return

        // Determine effective dimensions for iteration based on rank, defaulting to 1 for ranks < 4
        val effN3 = if (tensorRank >= 4) n3 else 1
        val effN2 = if (tensorRank >= 3) n2 else 1
        val effN1 = if (tensorRank >= 2) n1 else 1
        val effN0 = if (tensorRank >= 1) n0 else 1

        if (tensorRank == 0 && totalElementsToIterate == 1) { // Scalar
             actionPerElement(currentFlatIdx++, intArrayOf())
             return
        }


        for (i3 in 0 until effN3) {
            for (i2 in 0 until effN2) {
                for (i1 in 0 until effN1) {
                    for (i0 in 0 until effN0) {
                        if (currentFlatIdx < totalElementsToIterate) {
                            val indices = when (tensorRank) {
                                0 -> intArrayOf() // Should be handled by scalar case above
                                1 -> intArrayOf(i0)
                                2 -> intArrayOf(i0, i1)
                                3 -> intArrayOf(i0, i1, i2)
                                else -> intArrayOf(i0, i1, i2, i3) // Handles rank 4+
                            }.sliceArray(0 until tensorRank.coerceAtLeast(1)) // Ensure indices match rank, at least 1 for 1D
                             if(tensorRank == 0 && totalElementsToIterate ==1) { // True scalar from ne=[]
                                actionPerElement(currentFlatIdx++, intArrayOf())
                             } else {
                                actionPerElement(currentFlatIdx++, indices)
                             }
                        } else return
                    }
                }
            }
        }
    }


    // --- End Copied/Adapted Helpers ---


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
        dims: LongArray, // Effective dimensions
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
        tensor.nb = calculateStrides(tensor.type, tensor.ne)
        tensor.bufferId = bufferId
        tensor.dataOffset = dataOffset
        tensor.data = null

        val numElements = tensor.numElements().toInt()
        require(values.size == numElements) { "Provided FloatArray size (${values.size}) must match tensor element count ($numElements)." }

        val tensorByteSize = calculateTensorByteSize(tensor) // Use global function
        assertTrue(dataOffset + tensorByteSize <= bufferSize.toULong(),
            "Test tensor '$name' setup (offset $dataOffset + size $tensorByteSize) " +
            "dims ${dims.joinToString()} (effective ne: ${tensor.ne.joinToString()}) exceeds buffer capacity ($bufferSize).")

        var dataIdx = 0
        applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
            if (dataIdx < values.size) {
                tensor.setFloat(graphAllocator, values[dataIdx++], *indices)
            }
        }
        return tensor
    }

    internal fun getTensorDataAsFloatArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        if (tensor.type == GGMLType.F32 && tensor.data is FloatArray) {
            val fa = tensor.data as FloatArray
            if (fa.size == numElements) return fa.copyOf() // Return a copy to prevent external modification
        }

        val floatArray = FloatArray(numElements)
        var idx = 0

        applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
            if (idx < numElements) {
                floatArray[idx++] = when (tensor.type) {
                    GGMLType.F32 -> tensor.getFloat(graphAllocator, *indices)
                    GGMLType.F16 -> tensor.getHalf(graphAllocator, *indices)
                    else -> throw IllegalArgumentException("Unsupported tensor type ${tensor.type} for direct float array extraction. Dequantize to F32 first.")
                }
            }
        }
        return floatArray
    }

    internal fun calculateMeanSquaredError(original: FloatArray, new: FloatArray): Double {
        require(original.size == new.size) { "Arrays must have the same size for MSE." }
        if (original.isEmpty()) return 0.0

        var sumOfSquaredErrors = 0.0
        for (i in original.indices) {
            sumOfSquaredErrors += (original[i] - new[i]).toDouble().pow(2)
        }
        return sumOfSquaredErrors / original.size
    }

    internal fun calculateSignalToNoiseRatio(signal: FloatArray, noiseSource: FloatArray, originalSignal: FloatArray): Double {
        require(signal.size == noiseSource.size && signal.size == originalSignal.size) { "Arrays must have the same size for SNR." }
        if (signal.isEmpty()) return Double.POSITIVE_INFINITY

        var sumSignalSq = 0.0
        var sumNoiseSq = 0.0
        for (i in signal.indices) {
            sumSignalSq += originalSignal[i].toDouble().pow(2) // Power of the original, clean signal
            val noise = signal[i] - noiseSource[i] // The noise is the difference between quantized and original
            sumNoiseSq += noise.toDouble().pow(2)
        }
        if (sumNoiseSq == 0.0) return Double.POSITIVE_INFINITY
        if (sumSignalSq == 0.0 && sumNoiseSq > 0.0) return 0.0 // Or negative infinity dB if signal is zero
        return 10 * log10(sumSignalSq / sumNoiseSq)
    }

    @Test
    fun quantizationAccuracyTestSetup() {
        assertTrue(true, "Setup for GGMLQuantizationAccuracyTest complete.")
        val dims = longArrayOf(2) // Test with 1D array of 2 elements
        val testTensor = createAndPopulateF32Tensor("testF32", dims, floatArrayOf(1.0f, 2.0f))
        assertNotNull(testTensor)
        assertEquals(1.0f, testTensor.getFloat(graphAllocator, 0))
        assertEquals(2.0f, testTensor.getFloat(graphAllocator, 1))
        val dataBack = getTensorDataAsFloatArray(testTensor, graphAllocator)
        assertEquals(1.0f, dataBack[0])
        assertEquals(2.0f, dataBack[1])
    }

    @Test
    fun testQ8_0Accuracy() {
        val numElements = QK8_0 * 4 // Test with a few blocks, e.g., 4 blocks = 128 elements
        val originalF32Data = FloatArray(numElements) { idx ->
            // Create a diverse range of values
            when {
                idx % QK8_0 == 0 -> 0.0f // Start of a block with zero
                idx % QK8_0 == 1 -> 127.0f // Max positive for Q8 scaling
                idx % QK8_0 == 2 -> -128.0f // Min negative for Q8 scaling (won't be hit if scale is from abs max)
                                         // Actually, scale is based on amax / 127. So values map to [-127, 127] ideally.
                                         // Let's use values that will result in diverse q values.
                idx < QK8_0 -> (idx.toFloat() / (QK8_0 -1).toFloat()) * 10.0f // 0 to 10
                idx < QK8_0 * 2 -> ( (idx-QK8_0).toFloat() / (QK8_0 -1).toFloat() ) * -10.0f // 0 to -10
                idx < QK8_0 * 3 -> if (idx % 2 == 0) 50.5f else -50.5f // Alternating large
                else -> (idx - QK8_0 * 3).toFloat() * 0.1f - 1.0f // Small values around -1
            }
        }

        val dims = longArrayOf(numElements.toLong()) // 1D tensor
        val f32SrcTensor = createAndPopulateF32Tensor("f32Src_Q8Test", GGMLType.F32, dims, originalF32Data, dataOffset = 0uL)

        // 1. Quantize to Q8_0
        val q8Tensor = quantizeTensor(graphAllocator, f32SrcTensor, GGMLType.Q8_0)
        assertEquals(GGMLType.Q8_0, q8Tensor.type)
        assertTrue(q8Tensor.ne.contentEquals(f32SrcTensor.ne), "Dimensions should match after Q8_0 quantization")
        assertNotNull(q8Tensor.data, "Q8_0 tensor data should not be null after quantization")
        assertTrue(q8Tensor.data is ByteArray, "Q8_0 tensor data should be ByteArray")

        // 2. Dequantize Q8_0 back to F32
        val f32DequantizedTensor = dequantizeTensor(graphAllocator, q8Tensor)
        assertEquals(GGMLType.F32, f32DequantizedTensor.type)
        assertTrue(f32DequantizedTensor.ne.contentEquals(f32SrcTensor.ne), "Dimensions should match after Q8_0 dequantization")
        assertNotNull(f32DequantizedTensor.data, "Dequantized F32 tensor data should not be null")
        assertTrue(f32DequantizedTensor.data is FloatArray, "Dequantized F32 tensor data should be FloatArray")


        // 3. Extract data for comparison
        val dequantizedF32Data = getTensorDataAsFloatArray(f32DequantizedTensor, graphAllocator)

        // 4. Perform Accuracy Assertions
        assertEquals(originalF32Data.size, dequantizedF32Data.size, "Data array sizes should match")

        val mse = calculateMeanSquaredError(originalF32Data, dequantizedF32Data)
        // Threshold for Q8_0. Max error for one element for amax=127, scale=1 is 0.5. MSE can be (0.5)^2 = 0.25
        // If data range is smaller, MSE should be smaller.
        // Example: data in [-10, 10], amax=10, scale = 10/127. Max error on dequantized = scale*0.5 = (10/127)*0.5 approx 0.039
        // MSE approx (0.039)^2 = 0.0015
        val mseThreshold = 0.05 // This threshold is empirical and depends heavily on test data range.
        assertTrue(mse < mseThreshold, "Q8_0 MSE $mse too high (threshold $mseThreshold). Data range can affect this.")

        var sumAbsDiff = 0.0
        for(i in originalF32Data.indices) {
            sumAbsDiff += abs(originalF32Data[i] - dequantizedF32Data[i])
        }
        val meanAbsDiff = if (originalF32Data.isEmpty()) 0.0 else sumAbsDiff / originalF32Data.size
        val madThreshold = 0.2 // Also empirical. Max error could be around scale/2.
        assertTrue(meanAbsDiff < madThreshold, "Q8_0 Mean Absolute Difference $meanAbsDiff too high (threshold $madThreshold)")

        // println("Q8_0 Test: MSE = $mse, Mean Absolute Difference = $meanAbsDiff")
    }

    @Test
    fun testQ4_0Accuracy() {
        val numElements = QK4_0 * 4 // Test with a few blocks, e.g., 4 blocks = 128 elements
        val originalF32Data = FloatArray(numElements) { i ->
            // Create a diverse range of values, similar to Q8_0 test but scaled for Q4_0's effective range (-8 to +7)
            when {
                i % QK4_0 == 0 -> 0.0f
                i % QK4_0 == 1 -> 7.0f  // Test max positive scaled value
                i % QK4_0 == 2 -> -8.0f // Test min negative scaled value
                i < QK4_0 -> (i.toFloat() / (QK4_0 -1).toFloat()) * 1.0f // Block 1: 0 to 1
                i < QK4_0 * 2 -> ( (i-QK4_0).toFloat() / (QK4_0 -1).toFloat() ) * -1.0f // Block 2: 0 to -1
                i < QK4_0 * 3 -> if (i % 2 == 0) 0.75f else -0.75f // Block 3: Alternating
                else -> ((i - QK4_0 * 3).toFloat() / (QK4_0-1).toFloat() * 16.0f) - 8.0f // Block 4: Spread across -8 to +8
            }
        }

        val dims = longArrayOf(numElements.toLong()) // 1D tensor
        // Ensure enough space for F32 source, Q4 representation (smaller), and F32 dequantized
        // For this test, f32SrcTensor is in graphAllocator's buffer.
        // q4Tensor and f32DequantizedTensor will have their own .data arrays.
        val f32SrcTensor = createAndPopulateF32Tensor("f32Src_Q4Test", GGMLType.F32, dims, originalF32Data, dataOffset = 0uL)

        // 1. Quantize to Q4_0
        val q4Tensor = quantizeTensor(graphAllocator, f32SrcTensor, GGMLType.Q4_0)
        assertEquals(GGMLType.Q4_0, q4Tensor.type)
        assertTrue(q4Tensor.ne.contentEquals(f32SrcTensor.ne), "Dimensions should match after Q4_0 quantization")
        assertNotNull(q4Tensor.data, "Q4_0 tensor data should not be null after quantization")
        assertTrue(q4Tensor.data is ByteArray, "Q4_0 tensor data should be ByteArray")


        // 2. Dequantize Q4_0 back to F32
        val f32DequantizedTensor = dequantizeTensor(graphAllocator, q4Tensor)
        assertEquals(GGMLType.F32, f32DequantizedTensor.type)
        assertTrue(f32DequantizedTensor.ne.contentEquals(f32SrcTensor.ne), "Dimensions should match after Q4_0 dequantization")
        assertNotNull(f32DequantizedTensor.data, "Dequantized F32 tensor data should not be null")
        assertTrue(f32DequantizedTensor.data is FloatArray, "Dequantized F32 tensor data should be FloatArray")

        // 3. Extract data for comparison
        val dequantizedF32Data = getTensorDataAsFloatArray(f32DequantizedTensor, graphAllocator)

        // 4. Perform Accuracy Assertions
        assertEquals(originalF32Data.size, dequantizedF32Data.size, "Data array sizes should match")

        val mse = calculateMeanSquaredError(originalF32Data, dequantizedF32Data)
        // Threshold for Q4_0. Max error for one element is roughly (amax/8)/2.
        // If data is [-1,1], amax=1, scale=1/8. Max elem error ~ 1/16 = 0.0625. MSE ~ (0.0625)^2 ~ 0.0039.
        val mseThresholdQ4_0 = 0.02 // Adjusted, can be tuned based on actual results
        assertTrue(mse < mseThresholdQ4_0, "Q4_0 MSE $mse too high (threshold $mseThresholdQ4_0). Data range can affect this.")

        var sumAbsDiff = 0.0
        for(i in originalF32Data.indices) {
            sumAbsDiff += abs(originalF32Data[i] - dequantizedF32Data[i])
        }
        val meanAbsDiff = if (originalF32Data.isEmpty()) 0.0 else sumAbsDiff / originalF32Data.size
        // Max typical error for Q4_0 for range [-1,1] (where amax=1, scale=1/8) is ~1/8 = 0.125
        val madThresholdQ4_0 = 0.25 // Adjusted, can be tuned
        assertTrue(meanAbsDiff < madThresholdQ4_0, "Q4_0 Mean Absolute Difference $meanAbsDiff too high (threshold $madThresholdQ4_0)")

        // println("Q4_0 Test: MSE = $mse, Mean Absolute Difference = $meanAbsDiff")
    }

    // Helper for Mean Absolute Difference
    internal fun calculateMeanAbsoluteDifference(original: FloatArray, new: FloatArray): Double {
        require(original.size == new.size) { "Arrays must have the same size for MAD." }
        if (original.isEmpty()) return 0.0
        var sumAbsDiff = 0.0
        for (i in original.indices) {
            sumAbsDiff += abs(original[i] - new[i]).toDouble()
        }
        return sumAbsDiff / original.size
    }

    @Test
    fun testQ4_1Accuracy() {
        val numElements = QK4_1 * 4 // Test with a few blocks, e.g., 4 blocks = 128 elements
        val originalF32Data = FloatArray(numElements) { i ->
            // Create a diverse range of values for Q4_1 testing
            // Q4_1 uses d*nibble + m. Nibble is 0-15.
            // Test data that results in varied min/max per block.
            val blockNum = i / QK4_1
            val withinBlockIdx = i % QK4_1
            when (blockNum) {
                0 -> (withinBlockIdx.toFloat() / (QK4_1 -1).toFloat()) * 2.0f - 1.0f // Block 0: -1.0 to 1.0
                1 -> (withinBlockIdx.toFloat() / (QK4_1 -1).toFloat()) * 0.5f + 0.25f // Block 1: 0.25 to 0.75
                2 -> if (withinBlockIdx % 2 == 0) 5.0f else 4.0f // Block 2: Alternating 5.0, 4.0
                else -> (withinBlockIdx - QK4_1/2).toFloat() * 0.1f // Block 3: Centered around 0, small range
            }
        }

        val dims = longArrayOf(numElements.toLong(), 1L, 1L, 1L)
        // Ensure f32SrcTensor is placed in the graphAllocator's managed buffer for this test
        // Adapting to existing createAndPopulateF32Tensor signature
        val f32SrcTensor = createAndPopulateF32Tensor("f32Src_Q4_1Test", GGMLType.F32, dims, originalF32Data, dataOffset = 0uL)


        // 1. Quantize to Q4_1
        // quantizeTensor returns a new tensor with its own .data ByteArray
        val q4_1Tensor = quantizeTensor(graphAllocator, f32SrcTensor, GGMLType.Q4_1)
        assertEquals(GGMLType.Q4_1, q4_1Tensor.type)
        assertTrue(q4_1Tensor.ne.contentEquals(f32SrcTensor.ne), "Dimensions should match after Q4_1 quantization")
        assertNotNull(q4_1Tensor.data, "Q4_1 tensor data should not be null after quantization")
        assertTrue(q4_1Tensor.data is ByteArray, "Q4_1 tensor data should be ByteArray")


        // 2. Dequantize Q4_1 back to F32
        // dequantizeTensor also returns a new tensor with its own .data FloatArray
        val f32DequantizedTensor = dequantizeTensor(graphAllocator, q4_1Tensor)
        assertEquals(GGMLType.F32, f32DequantizedTensor.type)
        assertTrue(f32DequantizedTensor.ne.contentEquals(f32SrcTensor.ne), "Dimensions should match after Q4_1 dequantization")
        assertNotNull(f32DequantizedTensor.data, "F32 dequantized tensor data should not be null")
        assertTrue(f32DequantizedTensor.data is FloatArray, "Dequantized F32 tensor data should be FloatArray")

        // 3. Extract data for comparison
        // getTensorDataAsFloatArray can read from f32SrcTensor (in graphAllocator buffer)
        // and from f32DequantizedTensor (which has its own .data FloatArray)
        val retrievedOriginalF32Data = getTensorDataAsFloatArray(f32SrcTensor, graphAllocator) // Verifies initial data setup
        val dequantizedF32Data = getTensorDataAsFloatArray(f32DequantizedTensor, graphAllocator)

        // 4. Perform Accuracy Assertions
        assertEquals(originalF32Data.size, dequantizedF32Data.size, "Data array sizes should match")

        val mse = calculateMeanSquaredError(retrievedOriginalF32Data, dequantizedF32Data)
        // Q4_1 error: d/2 where d=(max-min)/15. If range is 2 (-1 to 1), d=2/15=0.133. Max error = 0.066. MSE ~ (0.066)^2 ~ 0.0044
        val mseThresholdQ4_1 = 0.015 // Start with a threshold slightly higher than Q4_0
        assertTrue(mse < mseThresholdQ4_1, "Q4_1 MSE $mse too high (threshold $mseThresholdQ4_1)")

        val mad = calculateMeanAbsoluteDifference(retrievedOriginalF32Data, dequantizedF32Data)
        val madThresholdQ4_1 = 0.1 // Adjusted from prompt's 0.2 as it seemed high vs max error
        assertTrue(mad < madThresholdQ4_1, "Q4_1 Mean Absolute Difference $mad too high (threshold $madThresholdQ4_1)")

        // println("Q4_1 Test: MSE = $mse, Mean Absolute Difference = $mad") // For debugging/tuning
    }
}
