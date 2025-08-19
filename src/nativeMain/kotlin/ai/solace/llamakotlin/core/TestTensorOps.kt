package ai.solace.llamakotlin.core

/**
 * Test file for tensor operations.
 * This file contains tests for the optimized tensor operations.
 */

fun main() {
    println("Testing optimized tensor operations")

    val context = GGMLContext()
    val allocator = GGMLGraphAllocator()

    // Test add operation
    testAdd(context, allocator)

    // Test mul operation
    testMul(context, allocator)

    // Test matMul operation
    testMatMul(context, allocator)

    println("All tests completed successfully")
}

/**
 * Tests the optimized add operation.
 */
fun testAdd(context: GGMLContext, graphAllocator: GGMLGraphAllocator) {
    println("\nTesting optimized add operation:")

    // Create tensors
    val a = GGMLTensor(type = GGMLType.F32).also { it.ne[0]=4; it.ne[1]=4; it.nb = calculateContiguousStrides(it.ne, it.type, it.rank()) }
    val b = GGMLTensor(type = GGMLType.F32).also { it.ne[0]=4; it.ne[1]=4; it.nb = calculateContiguousStrides(it.ne, it.type, it.rank()) }
    val tmpGraph = createGraph(2).also { it.nodes[0]=a; it.nodes[1]=b; it.nNodes=2; it.allocator = graphAllocator }
    graphAllocator.allocateGraph(tmpGraph)

    // Initialize tensor data
    for (i in 0 until 4) for (j in 0 until 4) {
        a.setFloat(graphAllocator, (i*4 + j).toFloat(), j, i)
        b.setFloat(graphAllocator, (i*4 + j + 1).toFloat(), j, i)
    }

    val aData = FloatArray(16) { idx -> a.getFloat(graphAllocator, idx % 4, idx / 4) }
    val bData = FloatArray(16) { idx -> b.getFloat(graphAllocator, idx % 4, idx / 4) }
    println("Tensor a: [${aData.joinToString()}]")
    println("Tensor b: [${bData.joinToString()}]")

    // Test optimized add operation
    val c = computeAddRet(graphAllocator, context, a, b)
    val cData = FloatArray(16) { idx -> c.getFloat(graphAllocator, idx % 4, idx / 4) }
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
fun testMul(context: GGMLContext, graphAllocator: GGMLGraphAllocator) {
    println("\nTesting optimized mul operation:")

    // Create tensors
    val a = GGMLTensor(type = GGMLType.F32).also { it.ne[0]=4; it.ne[1]=4; it.nb = calculateContiguousStrides(it.ne, it.type, it.rank()) }
    val b = GGMLTensor(type = GGMLType.F32).also { it.ne[0]=4; it.ne[1]=4; it.nb = calculateContiguousStrides(it.ne, it.type, it.rank()) }
    val tmpGraph = createGraph(2).also { it.nodes[0]=a; it.nodes[1]=b; it.nNodes=2; it.allocator = graphAllocator }
    graphAllocator.allocateGraph(tmpGraph)

    // Initialize tensor data
    for (i in 0 until 4) for (j in 0 until 4) {
        a.setFloat(graphAllocator, (i*4 + j).toFloat(), j, i)
        b.setFloat(graphAllocator, (i*4 + j + 1).toFloat(), j, i)
    }

    val aData = FloatArray(16) { idx -> a.getFloat(graphAllocator, idx % 4, idx / 4) }
    val bData = FloatArray(16) { idx -> b.getFloat(graphAllocator, idx % 4, idx / 4) }
    println("Tensor a: [${aData.joinToString()}]")
    println("Tensor b: [${bData.joinToString()}]")

    // Test optimized mul operation
    val c = computeMulRet(graphAllocator, context, a, b)
    val cData = FloatArray(16) { idx -> c.getFloat(graphAllocator, idx % 4, idx / 4) }
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
fun testMatMul(context: GGMLContext, graphAllocator: GGMLGraphAllocator) {
    println("\nTesting optimized matMul operation:")

    // Create tensors
    val a = GGMLTensor(type = GGMLType.F32).also { it.ne[0]=4; it.ne[1]=4; it.nb = calculateContiguousStrides(it.ne, it.type, it.rank()) }
    val b = GGMLTensor(type = GGMLType.F32).also { it.ne[0]=4; it.ne[1]=4; it.nb = calculateContiguousStrides(it.ne, it.type, it.rank()) }
    val tmpGraph = createGraph(2).also { it.nodes[0]=a; it.nodes[1]=b; it.nNodes=2; it.allocator = graphAllocator }
    graphAllocator.allocateGraph(tmpGraph)

    // Initialize tensor data
    for (i in 0 until 4) for (j in 0 until 4) {
        a.setFloat(graphAllocator, (i*4 + j).toFloat(), j, i)
        b.setFloat(graphAllocator, (i*4 + j + 1).toFloat(), j, i)
    }

    val aData = FloatArray(16) { idx -> a.getFloat(graphAllocator, idx % 4, idx / 4) }
    val bData = FloatArray(16) { idx -> b.getFloat(graphAllocator, idx % 4, idx / 4) }
    println("Tensor a: [${aData.joinToString()}]")
    println("Tensor b: [${bData.joinToString()}]")

    // Test optimized matMul operation
    val c = computeMatMulRet(graphAllocator, context, a, b)
    val cData = FloatArray(16) { idx -> c.getFloat(graphAllocator, idx % 4, idx / 4) }
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
