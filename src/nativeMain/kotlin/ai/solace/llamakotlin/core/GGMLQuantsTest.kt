package ai.solace.llamakotlin.core

import kotlin.math.abs
import kotlin.random.Random

@ExperimentalUnsignedTypes
fun main() {
    println("Testing GGMLQuants operations...")
    testQ4KQuantizationDequantizationCycle()
    testQ4KQuantizationAccuracy()
    // Add more test calls here as needed
    println("GGMLQuants tests completed.")
}

// Helper functions for error metrics
fun calculateMSE(original: FloatArray, quantized: FloatArray): Float {
    require(original.size == quantized.size) { "Arrays must have the same size for MSE calculation." }
    if (original.isEmpty()) return 0.0f
    var sumSqError = 0.0
    for (i in original.indices) {
        val error = original[i] - quantized[i]
        sumSqError += error * error
    }
    return (sumSqError / original.size).toFloat()
}

fun calculateMAD(original: FloatArray, quantized: FloatArray): Float {
    require(original.size == quantized.size) { "Arrays must have the same size for MAD calculation." }
    if (original.isEmpty()) return 0.0f
    var sumAbsError = 0.0
    for (i in original.indices) {
        sumAbsError += abs(original[i] - quantized[i])
    }
    return (sumAbsError / original.size).toFloat()
}

@ExperimentalUnsignedTypes
fun testQ4KQuantizationAccuracy() {
    println("\nTesting Q4_K Quantization Accuracy:")

    // Generate Test Data: Sine wave from -PI to PI, scaled e.g. to -8.0 to 8.0
    val size = QK4_K // 256 elements
    val originalFloats = FloatArray(size)
    val amplitude = 8.0f
    for (i in 0 until size) {
        originalFloats[i] = amplitude * kotlin.math.sin(2 * kotlin.math.PI * i.toDouble() / size).toFloat()
    }
    // Add some specific values: zero, positive, negative, near boundaries if possible
    if (size > 4) {
        originalFloats[0] = 0.0f
        originalFloats[1] = 1.0f
        originalFloats[2] = -1.0f
        originalFloats[3] = 7.9f
        originalFloats[4] = -7.9f
    }


    println("Original first 8 floats for accuracy test: ${originalFloats.sliceArray(0..minOf(7, size-1)).joinToString()}")

    // Quantize
    var q4kBytes: ByteArray? = null
    try {
        q4kBytes = quantizeBlockQ4K(originalFloats)
    } catch (e: Exception) {
        println("ERROR during quantization in accuracy test: ${e.message}")
        e.printStackTrace()
        println("Q4_K Quantization Accuracy test FAILED (quantization error).")
        return
    }

    // Dequantize
    var dequantizedFloats: FloatArray? = null
    try {
        dequantizedFloats = dequantizeBlockQ4K(q4kBytes)
    } catch (e: Exception) {
        println("ERROR during dequantization in accuracy test: ${e.message}")
        e.printStackTrace()
        println("Q4_K Quantization Accuracy test FAILED (dequantization error).")
        return
    }
    println("Dequantized first 8 floats for accuracy test: ${dequantizedFloats.sliceArray(0..minOf(7, size-1)).joinToString()}")

    // Calculate Error Metrics
    val mse = calculateMSE(originalFloats, dequantizedFloats)
    val mad = calculateMAD(originalFloats, dequantizedFloats)

    println("Calculated MSE: $mse")
    println("Calculated MAD: $mad")

    // Define and Assert Thresholds
    // These thresholds are initial guesses and might need tuning based on the
    // implemented quantization quality and comparison with reference C++ tests.
    // Q4_K is a very low precision type, so errors will be higher than Q8_0.
    val mseThreshold = 0.02f // Increased from typical Q8 values
    val madThreshold = 0.1f  // Increased from typical Q8 values

    var testPassed = true
    if (mse > mseThreshold) {
        println("FAILED: MSE $mse exceeds threshold $mseThreshold.")
        testPassed = false
    }
    if (mad > madThreshold) {
        println("FAILED: MAD $mad exceeds threshold $madThreshold.")
        testPassed = false
    }

    if (testPassed) {
        println("Q4_K Quantization Accuracy test passed.")
    } else {
        println("Q4_K Quantization Accuracy test FAILED.")
    }
}


@ExperimentalUnsignedTypes
fun testQ4KQuantizationDequantizationCycle() {
    println("\nTesting Q4_K Quantization <-> Dequantization Cycle:")

    // 1. Create a sample FloatArray
    val originalFloats = FloatArray(QK4_K) { Random.nextFloat() * 2.0f - 1.0f } // Random floats between -1 and 1

    println("Original first 8 floats: ${originalFloats.sliceArray(0..7).joinToString()}")

    // 2. Quantize
    var q4kBytes: ByteArray? = null
    try {
        q4kBytes = quantizeBlockQ4K(originalFloats)
        println("Quantization successful. Output ByteArray size: ${q4kBytes.size}")
        if (q4kBytes.size != GGMLType.Q4_K.byteSize.toInt()) {
            println("ERROR: Quantized byte array size is incorrect! Expected ${GGMLType.Q4_K.byteSizetoInt()}, Got ${q4kBytes.size}")
            return
        }
    } catch (e: Exception) {
        println("ERROR during quantization: ${e.message}")
        e.printStackTrace()
        return
    }

    // 3. Dequantize
    var dequantizedFloats: FloatArray? = null
    try {
        dequantizedFloats = dequantizeBlockQ4K(q4kBytes)
        println("Dequantization successful. Output FloatArray size: ${dequantizedFloats.size}")
        if (dequantizedFloats.size != QK4_K) {
            println("ERROR: Dequantized float array size is incorrect! Expected ${QK4_K}, Got ${dequantizedFloats.size}")
            return
        }
    } catch (e: Exception) {
        println("ERROR during dequantization: ${e.message}")
        e.printStackTrace()
        return
    }

    println("Dequantized first 8 floats: ${dequantizedFloats.sliceArray(0..7).joinToString()}")

    // 4. Compare original and dequantized floats
    var totalAbsoluteError = 0.0f
    var maxAbsoluteError = 0.0f
    var success = true
    for (i in originalFloats.indices) {
        val error = abs(originalFloats[i] - dequantizedFloats[i])
        totalAbsoluteError += error
        if (error > maxAbsoluteError) {
            maxAbsoluteError = error
        }
        // Define an acceptable error margin. For Q4_K, errors can be somewhat significant.
        // This threshold might need adjustment depending on the quality of the quantization.
        // For a simple test, let's use a relatively loose threshold.
        val threshold = 0.1f // Max 0.1f error for individual float
        if (error > threshold) {
            println("ERROR at index $i: Original = ${originalFloats[i]}, Dequantized = ${dequantizedFloats[i]}, Error = $error")
            // success = false // Don't fail for individual errors yet, summarize first
        }
    }

    val averageAbsoluteError = totalAbsoluteError / originalFloats.size
    println("Average Absolute Error: $averageAbsoluteError")
    println("Maximum Absolute Error: $maxAbsoluteError")

    // Define overall test success criteria
    // These are example thresholds and might need tuning.
    val avgErrorThreshold = 0.05f
    val maxErrorThreshold = 0.2f // Looser than individual check for overall pass

    if (averageAbsoluteError > avgErrorThreshold || maxAbsoluteError > maxErrorThreshold) {
        success = false
    }

    if (success) {
        println("Q4_K Quantization <-> Dequantization Cycle test passed.")
    } else {
        println("Q4_K Quantization <-> Dequantization Cycle test FAILED (errors exceed thresholds).")
    }
}
