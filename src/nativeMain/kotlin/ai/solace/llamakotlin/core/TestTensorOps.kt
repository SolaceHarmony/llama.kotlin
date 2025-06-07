package ai.solace.llamakotlin.core

/**
 * Test file for tensor operations.
 * This file contains tests for the optimized tensor operations.
 */

fun main() {
    println("Testing optimized tensor operations")

    // Create a context
    val context = GGMLContext(
        memSize = (16 * 1024 * 1024).toULong(), // 16 MB
        memBuffer = null,
        memBufferOwned = false,
        noAlloc = false,
        computeImmediately = true // Important: set to true to compute immediately
    )

    // Test add operation
    testAdd(context)

    // Test mul operation
    testMul(context)

    // Test matMul operation
    testMatMul(context)

    println("All tests completed successfully")
}

/**
 * Tests the optimized add operation.
 */
fun testAdd(context: GGMLContext) {
    println("\nTesting optimized add operation:")

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 4, 4)
    val b = createTensor2D(context, GGMLType.F32, 4, 4)

    // Initialize tensor data
    val aData = a.data as FloatArray
    val bData = b.data as FloatArray
    for (i in 0 until 16) {
        aData[i] = i.toFloat()
        bData[i] = (i + 1).toFloat()
    }

    println("Tensor a: [${aData.take(16).joinToString()}]")
    println("Tensor b: [${bData.take(16).joinToString()}]")

    // Test optimized add operation
    val c = computeAdd(context, a, b)
    val cData = c.data as FloatArray
    println("a + b: [${cData.take(16).joinToString()}]")

    // Verify results
    var success = true
    for (i in 0 until 16) {
        val expected = aData[i] + bData[i]
        val actual = cData[i]
        if (expected != actual) {
            println("Error at index $i: expected $expected, got $actual")
            success = false
        }
    }

    if (success) {
        println("Add operation test passed")
    } else {
        println("Add operation test failed")
    }
}

/**
 * Tests the optimized mul operation.
 */
fun testMul(context: GGMLContext) {
    println("\nTesting optimized mul operation:")

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 4, 4)
    val b = createTensor2D(context, GGMLType.F32, 4, 4)

    // Initialize tensor data
    val aData = a.data as FloatArray
    val bData = b.data as FloatArray
    for (i in 0 until 16) {
        aData[i] = i.toFloat()
        bData[i] = (i + 1).toFloat()
    }

    println("Tensor a: [${aData.take(16).joinToString()}]")
    println("Tensor b: [${bData.take(16).joinToString()}]")

    // Test optimized mul operation
    val c = computeMul(context, a, b)
    val cData = c.data as FloatArray
    println("a * b: [${cData.take(16).joinToString()}]")

    // Verify results
    var success = true
    for (i in 0 until 16) {
        val expected = aData[i] * bData[i]
        val actual = cData[i]
        if (expected != actual) {
            println("Error at index $i: expected $expected, got $actual")
            success = false
        }
    }

    if (success) {
        println("Mul operation test passed")
    } else {
        println("Mul operation test failed")
    }
}

/**
 * Tests the optimized matMul operation.
 */
fun testMatMul(context: GGMLContext) {
    println("\nTesting optimized matMul operation:")

    // Create tensors
    val a = createTensor2D(context, GGMLType.F32, 4, 4)
    val b = createTensor2D(context, GGMLType.F32, 4, 4)

    // Initialize tensor data
    val aData = a.data as FloatArray
    val bData = b.data as FloatArray
    for (i in 0 until 16) {
        aData[i] = i.toFloat()
        bData[i] = (i + 1).toFloat()
    }

    println("Tensor a: [${aData.take(16).joinToString()}]")
    println("Tensor b: [${bData.take(16).joinToString()}]")

    // Test optimized matMul operation
    val c = computeMatMul(context, a, b)
    val cData = c.data as FloatArray
    println("a @ b: [${cData.take(16).joinToString()}]")

    // Verify results by computing the matrix multiplication manually
    var success = true
    for (i in 0 until 4) {
        for (j in 0 until 4) {
            var expected = 0.0f
            for (k in 0 until 4) {
                expected += aData[i * 4 + k] * bData[k * 4 + j]
            }
            val actual = cData[i * 4 + j]
            if (kotlin.math.abs(expected - actual) > 0.0001f) {
                println("Error at index (${i},${j}): expected $expected, got $actual")
                success = false
            }
        }
    }

    if (success) {
        println("MatMul operation test passed")
    } else {
        println("MatMul operation test failed")
    }
}
