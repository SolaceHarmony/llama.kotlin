#!/usr/bin/env kotlin

/*
 * BitNet 1.58 Quantization Demonstration
 * 
 * This script demonstrates the key features of the newly implemented
 * BitNet 1.58 quantization in llama.kotlin:
 * 
 * 1. Ternary quantization (-1, 0, +1) 
 * 2. Efficient packing (5 ternary values per byte)
 * 3. Scale-based quantization with F16 precision
 * 4. Integration with existing tensor operations
 * 
 * Usage: kotlin -cp build/classes/kotlin/linuxX64/main demo_bitnet158.kts
 */

import ai.solace.llamakotlin.core.*

fun demonstrateBitNet158() {
    println("=== BitNet 1.58 Quantization Demonstration ===")
    println()
    
    // Setup
    val graphAllocator = GGMLGraphAllocator()
    val testBuffer = ByteArray(1024 * 1024)
    graphAllocator.buffers[0] = testBuffer
    graphAllocator.context = GGMLContext()
    
    println("1. Original Data:")
    val originalData = floatArrayOf(
        -2.5f, -1.2f, -0.3f, 0.0f, 0.3f, 1.2f, 2.5f, -1.0f,
        1.0f, 0.0f, -1.5f, 1.8f, -0.1f, 0.9f, -2.0f, 0.5f,
        -0.8f, 1.3f, 0.2f, -1.7f, 0.0f, 1.0f, -1.0f, 0.0f,
        2.1f, -0.5f, 0.7f, -1.9f, 0.4f, 1.6f, -0.2f, 0.8f
    )
    println("   ${originalData.joinToString(", ") { "%.1f".format(it) }}")
    println("   Size: ${originalData.size} elements (${originalData.size * 4} bytes)")
    println()
    
    // Create F32 tensor
    val f32Tensor = GGMLTensor(type = GGMLType.F32, name = "demo_f32")
    f32Tensor.ne[0] = originalData.size.toLong()
    f32Tensor.ne[1] = 1L
    f32Tensor.nb = calculateContiguousStrides(f32Tensor.ne, GGMLType.F32, f32Tensor.rank())
    
    val f32Size = calculateTensorByteSize(f32Tensor).toInt()
    val f32Offset = graphAllocator.allocateTensorData(f32Size)
    f32Tensor.bufferId = 0
    f32Tensor.dataOffset = f32Offset.toULong()
    
    for (i in originalData.indices) {
        f32Tensor.setFloat(graphAllocator, originalData[i], i)
    }
    
    println("2. BitNet 1.58 Quantization:")
    val bitNetTensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.BITNET_1_58)
    println("   Type: ${bitNetTensor.type}")
    println("   Blocks: ${bitNetTensor.getNumBlocks()}")
    println("   Block Size: ${GGMLType.BITNET_1_58.byteSize} bytes")
    println("   Total Size: ${calculateTensorByteSize(bitNetTensor)} bytes")
    
    val compressionRatio = (originalData.size * 4.0f) / calculateTensorByteSize(bitNetTensor).toFloat()
    println("   Compression Ratio: %.2fx".format(compressionRatio))
    println()
    
    println("3. Block Details:")
    for (blockIdx in 0 until bitNetTensor.getNumBlocks().toInt()) {
        val scale = bitNetTensor.getBitNet158BlockScale(graphAllocator, blockIdx)
        println("   Block $blockIdx - Scale: %.4f".format(scale))
        
        print("   Ternary weights: ")
        val weights = mutableListOf<Byte>()
        for (i in 0 until kotlin.math.min(16, QK_BITNET_1_58)) {
            val weight = bitNetTensor.getBitNet158TernaryWeight(graphAllocator, blockIdx, i)
            weights.add(weight)
        }
        println(weights.joinToString(", ") { "%2d".format(it) } + 
                if (QK_BITNET_1_58 > 16) " ..." else "")
    }
    println()
    
    println("4. Dequantization:")
    val dequantizedTensor = dequantizeTensor(graphAllocator, bitNetTensor)
    val dequantizedData = FloatArray(originalData.size)
    for (i in originalData.indices) {
        dequantizedData[i] = dequantizedTensor.getFloat(graphAllocator, i)
    }
    println("   ${dequantizedData.joinToString(", ") { "%.1f".format(it) }}")
    println()
    
    println("5. Quantization Quality:")
    var mse = 0.0f
    var mad = 0.0f
    for (i in originalData.indices) {
        val diff = originalData[i] - dequantizedData[i]
        mse += diff * diff
        mad += kotlin.math.abs(diff)
    }
    mse /= originalData.size
    mad /= originalData.size
    
    val snr = if (mse > 0) {
        val signalPower = originalData.map { it * it }.average()
        10 * kotlin.math.log10(signalPower / mse)
    } else Double.POSITIVE_INFINITY
    
    println("   MSE (Mean Squared Error): %.6f".format(mse))
    println("   MAD (Mean Absolute Deviation): %.6f".format(mad))
    println("   SNR (Signal-to-Noise Ratio): %.2f dB".format(snr))
    println()
    
    println("6. Efficiency Analysis:")
    println("   Original: F32 (32 bits/element)")
    println("   BitNet 1.58: ~%.2f bits/element".format(kotlin.math.log2(3.0)))
    println("   Space Saving: %.1f%%".format((1.0f - 1.0f/compressionRatio) * 100))
    println()
    
    println("✅ BitNet 1.58 demonstration completed successfully!")
}

// Run demonstration
demonstrateBitNet158()