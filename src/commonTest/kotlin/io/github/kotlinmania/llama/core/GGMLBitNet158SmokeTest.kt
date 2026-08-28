package io.github.kotlinmania.llama.core

import kotlin.test.*

/**
 * Simple smoke test for BitNet 1.58 quantization to verify basic functionality.
 */
class GGMLBitNet158SmokeTest {

    @Test
    fun testBitNet158BasicFunctionality() {
        // Create a simple test setup
        val testBuffer = ByteArray(1024 * 1024) { 0 }
        val graphAllocator = io.github.kotlinmania.llama.ore.GGMLGraphAllocator()
        if (graphAllocator.buffers.isEmpty()) {
            graphAllocator.buffers.add(testBuffer)
        } else {
            graphAllocator.buffers[0] = testBuffer
        }
        if (graphAllocator.tensorAllocators.isEmpty()) {
            graphAllocator.tensorAllocators.add(
                io.github.kotlinmania.llama.ore.GGMLDynTensorAllocator(
                    bufferSize = testBuffer.size.toULong()
                )
            )
        } else {
            graphAllocator.tensorAllocators[0].reset(testBuffer.size.toULong())
        }
        resetAllocatorTracking(graphAllocator)
        graphAllocator.context = io.github.kotlinmania.llama.ore.GGMLContext()
        var currentOffset = 0uL

        // Test data that should work well with ternary quantization
        val testData = floatArrayOf(
            -1.0f, 0.0f, 1.0f, -0.5f, 0.5f, -2.0f, 2.0f, 0.0f,
            1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, -1.0f,
            0.0f, 1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f,
            -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f
        ) // 32 values (1 block)
        
        // Create F32 tensor
        val f32Tensor = io.github.kotlinmania.llama.ore.GGMLTensor(
            type = io.github.kotlinmania.llama.ore.GGMLType.F32,
            name = "test_f32"
        )
        f32Tensor.ne[0] = testData.size.toLong()
        f32Tensor.ne[1] = 1L
        f32Tensor.ne[2] = 1L  
        f32Tensor.ne[3] = 1L
        f32Tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
            f32Tensor.ne,
            io.github.kotlinmania.llama.ore.GGMLType.F32,
            f32Tensor.rank()
        )
        
        val f32Size = calculateTensorByteSize(f32Tensor).toInt()
        val alignment = 16uL
        val f32Offset = ((currentOffset + (alignment - 1uL)) / alignment) * alignment
        currentOffset = f32Offset + f32Size.toULong()
        f32Tensor.bufferId = 0
        f32Tensor.dataOffset = f32Offset
        
        // Copy test data
        for (i in testData.indices) {
            f32Tensor.setFloat(graphAllocator, testData[i], i)
        }
        
        // Test quantization to BitNet 1.58
        val bitNetTensor = io.github.kotlinmania.llama.ore.quantizeTensor(
            graphAllocator,
            f32Tensor,
            io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58
        )
        assertNotNull(bitNetTensor, "Quantization should succeed")
        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58, bitNetTensor.type, "Quantized tensor should have BitNet 1.58 type")
        assertEquals(testData.size.toLong(), bitNetTensor.numElements(), "Element count should be preserved")
        assertEquals(1L, bitNetTensor.getNumBlocks(), "Should have 1 block for 32 elements")
        
        // Test block scale accessor
        val scale = bitNetTensor.getBitNet158BlockScale(graphAllocator, 0)
        assertTrue(scale > 0, "Block scale should be positive, got $scale")
        
        // Test ternary weight accessors
        for (i in 0 until io.github.kotlinmania.llama.ore.QK_BITNET_1_58) {
            val weight = bitNetTensor.getBitNet158TernaryWeight(graphAllocator, 0, i)
            assertTrue(weight in -1..1, "Ternary weight should be -1, 0, or 1, got $weight at index $i")
        }
        
        // Test dequantization
        val dequantizedTensor =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, bitNetTensor)
        assertNotNull(dequantizedTensor, "Dequantization should succeed")
        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, dequantizedTensor.type, "Dequantized tensor should be F32")
        assertEquals(testData.size.toLong(), dequantizedTensor.numElements(), "Element count should be preserved")
        
        // Verify dequantized values are reasonable (all finite)
        for (i in testData.indices) {
            val dequantizedValue = dequantizedTensor.getFloat(graphAllocator, i)
            assertTrue(dequantizedValue.isFinite(), "Dequantized value at index $i should be finite, got $dequantizedValue")
        }
        
        println("✅ BitNet 1.58 smoke test passed!")
        println("   - Scale: $scale")
        println("   - Sample weights: ${(0 until kotlin.math.min(8,
            io.github.kotlinmania.llama.ore.QK_BITNET_1_58
        )).map { 
            bitNetTensor.getBitNet158TernaryWeight(graphAllocator, 0, it) 
        }}")
        println("   - Sample dequantized: ${(0 until kotlin.math.min(8, testData.size)).map { 
            dequantizedTensor.getFloat(graphAllocator, it) 
        }}")
    }
}
