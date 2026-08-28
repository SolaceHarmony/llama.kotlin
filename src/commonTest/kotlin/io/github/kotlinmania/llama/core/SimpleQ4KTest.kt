package io.github.kotlinmania.llama.core

import kotlin.test.*
import kotlin.math.*

class SimpleQ4KTest {
    
    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 2 * 1024 * 1024 // 2MB

    @BeforeTest
    fun setup() {
        graphAllocator = io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(io.github.kotlinmania.llama.ore.GGMLDynTensorAllocator())
        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    @Test
    fun testQ4KTypeDefinition() {
        // Test basic Q4_K type definition
        assertEquals("q4_k", io.github.kotlinmania.llama.ore.GGMLType.Q4_K.description)
        assertTrue(io.github.kotlinmania.llama.ore.GGMLType.Q4_K.byteSize > 0u)
        println("Q4_K byteSize: ${io.github.kotlinmania.llama.ore.GGMLType.Q4_K.byteSize}")
        
        // Test constants
        assertEquals(256, io.github.kotlinmania.llama.ore.QK_K)
        assertEquals(12, io.github.kotlinmania.llama.ore.K_SCALE_SIZE)
        
        // Expected size: 2*F16 (d,dmin) + K_SCALE_SIZE + QK_K/2 = 4 + 12 + 128 = 144 bytes
        assertEquals(144uL, io.github.kotlinmania.llama.ore.GGMLType.Q4_K.byteSize)
        println("Q4_K constants validated successfully")
    }

    private fun createAndPopulateF32Tensor(
        name: String,
        dims: LongArray,
        values: FloatArray,
        dataOffset: ULong = 0uL,
        bufferId: Int = 0
    ): io.github.kotlinmania.llama.ore.GGMLTensor {
        val tensor =
            io.github.kotlinmania.llama.ore.GGMLTensor(io.github.kotlinmania.llama.ore.GGMLType.F32)
        tensor.name = name
        
        tensor.ne = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dimSize ->
            if (index < io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) tensor.ne[index] = dimSize
        }
        
        // Calculate strides
        tensor.nb = ULongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 0uL }
        if (tensor.type.byteSize > 0uL) {
            tensor.nb[0] = tensor.type.byteSize
            for (d in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
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
            val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
            var tempIdx = i.toLong()
            for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                if (tensor.ne[dim] > 0) {
                    indices[dim] = (tempIdx % tensor.ne[dim]).toInt()
                    tempIdx /= tensor.ne[dim]
                }
            }
            tensor.setFloat(graphAllocator, values[i], *indices)
        }
        
        return tensor
    }

    private fun getTensorDataAsFloatArray(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        val floatArray = FloatArray(numElements)
        
        for (i in 0 until numElements) {
            val indices = IntArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
            var tempIdx = i.toLong()
            for (dim in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
                if (tensor.ne[dim] > 0) {
                    indices[dim] = (tempIdx % tensor.ne[dim]).toInt()
                    tempIdx /= tensor.ne[dim]
                }
            }
            
            floatArray[i] = when (tensor.type) {
                io.github.kotlinmania.llama.ore.GGMLType.F32 -> tensor.getFloat(graphAllocator, *indices)
                io.github.kotlinmania.llama.ore.GGMLType.F16 -> tensor.getHalf(graphAllocator, *indices)
                else -> throw IllegalArgumentException("Unsupported tensor type ${tensor.type} for direct float array extraction.")
            }
        }
        
        return floatArray
    }

    @Test
    fun testQ4KBasicQuantization() {
        // Create simple F32 tensor to quantize (one block)
        val numElements = io.github.kotlinmania.llama.ore.QK_K // One block
        val originalData = FloatArray(numElements) { i -> 
            when {
                i < io.github.kotlinmania.llama.ore.QK_K /4 -> (i.toFloat() / (io.github.kotlinmania.llama.ore.QK_K /4).toFloat()) * 4.0f - 2.0f  // Range -2.0 to 2.0
                i < io.github.kotlinmania.llama.ore.QK_K /2 -> if (i % 2 == 0) 1.0f else -1.0f  // Alternating
                i < 3* io.github.kotlinmania.llama.ore.QK_K /4 -> (i - io.github.kotlinmania.llama.ore.QK_K /2).toFloat() * 0.01f  // Small values
                else -> (i - 3* io.github.kotlinmania.llama.ore.QK_K /4).toFloat() / (io.github.kotlinmania.llama.ore.QK_K /4).toFloat() * 8.0f - 4.0f  // Range -4.0 to 4.0
            }
        }
        
        val dims = longArrayOf(numElements.toLong())
        val f32Tensor = createAndPopulateF32Tensor("test_f32_q4k", dims, originalData)
        
        // Try to quantize to Q4_K
        val q4kTensor = io.github.kotlinmania.llama.ore.quantizeTensor(
            graphAllocator,
            f32Tensor,
            io.github.kotlinmania.llama.ore.GGMLType.Q4_K
        )
        
        // Basic checks
        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.Q4_K, q4kTensor.type)
        assertEquals(numElements.toLong(), q4kTensor.ne[0])
        assertNotNull(q4kTensor.data)
        assertTrue(q4kTensor.data is ByteArray)
        
        // Test dequantization
        val dequantizedTensor =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, q4kTensor)
        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dequantizedTensor.type)
        assertEquals(numElements.toLong(), dequantizedTensor.ne[0])
        
        val dequantizedData = getTensorDataAsFloatArray(dequantizedTensor, graphAllocator)
        assertEquals(originalData.size, dequantizedData.size)
        
        // Check accuracy (Q4_K should have reasonable precision)
        var sumSquaredError = 0.0f
        var maxError = 0.0f
        for (i in originalData.indices) {
            val error = abs(originalData[i] - dequantizedData[i])
            sumSquaredError += (error * error)
            maxError = maxOf(maxError, error)
        }
        val mse = sumSquaredError / originalData.size
        
        println("Q4_K Quantization Results:")
        println("MSE: $mse")
        println("Max Error: $maxError")
        println("Sample original: ${originalData.take(10).joinToString(", ")}")
        println("Sample dequantized: ${dequantizedData.take(10).joinToString(", ")}")
        
        // Q4_K should have better precision than Q4_0
        assertTrue(mse < 0.1, "Q4_K MSE too high: $mse")
        assertTrue(maxError < 2.0f, "Q4_K max error too high: $maxError")
        
        println("Q4_K quantization test passed!")
    }

    @Test
    fun testQ4KAccessorFunctions() {
        // Create a Q4_K tensor with known values
        val numElements = io.github.kotlinmania.llama.ore.QK_K
        val testData = FloatArray(numElements) { i -> (i % 16).toFloat() - 8.0f } // Range -8 to 7
        
        val dims = longArrayOf(numElements.toLong())
        val f32Tensor = createAndPopulateF32Tensor("test_f32_accessor", dims, testData)
        val q4kTensor = io.github.kotlinmania.llama.ore.quantizeTensor(
            graphAllocator,
            f32Tensor,
            io.github.kotlinmania.llama.ore.GGMLType.Q4_K
        )
        
        // Test accessor functions
        val blockIndex = 0
        val d = q4kTensor.getQ4_KBlockScale(graphAllocator, blockIndex)
        val dmin = q4kTensor.getQ4_KBlockScaleMin(graphAllocator, blockIndex)
        
        assertTrue(d > 0.0f, "Q4_K block scale should be positive: $d")
        println("Q4_K block scale: $d, dmin: $dmin")
        
        // Test sub-block scale and min accessors
        for (subBlock in 0..7) {
            val quantizedScale = q4kTensor.getQ4_KQuantizedScale(graphAllocator, blockIndex, subBlock)
            val quantizedMin = q4kTensor.getQ4_KQuantizedMin(graphAllocator, blockIndex, subBlock)
            
            assertTrue(quantizedScale in 0..63, "Quantized scale should be 0-63: $quantizedScale")
            assertTrue(quantizedMin in 0..63, "Quantized min should be 0-63: $quantizedMin")
        }
        
        // Test weight accessors
        for (i in 0 until minOf(32, numElements)) {
            val weight = q4kTensor.getQ4_KWeight(graphAllocator, blockIndex, i)
            assertTrue(weight in 0..15, "Q4_K weight should be 0-15: $weight")
        }
        
        println("Q4_K accessor functions test passed!")
    }
}
