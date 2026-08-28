package io.github.kotlinmania.llama.core

import kotlin.math.abs
import kotlin.test.*
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

/**
 * Comprehensive tests for optimized matmul quantization combinations.
 * Tests all new optimized paths: F32 x Q_type and Q_type x Q_type combinations.
 */
@OptIn(ExperimentalTime::class)
class GGMLMatMulOptimizationTest {

    private lateinit var graphAllocator: io.github.kotlinmania.llama.ore.GGMLGraphAllocator
    private lateinit var context: io.github.kotlinmania.llama.ore.GGMLContext
    private val bufferSize = 2 * 1024 * 1024 // 2MB for larger matrices

    @BeforeTest
    fun setup() {
        val (allocator, _) = GGMLTestUtils.createTestAllocator(bufferSize)
        graphAllocator = allocator
        context = io.github.kotlinmania.llama.ore.GGMLContext()
    }

    @Test
    fun testF32xQ40Optimization() {
        val M = 4; val K = io.github.kotlinmania.llama.ore.QK4_0 * 2; val N = 3
        val tensorF32 = GGMLTestUtils.createF32Matrix(graphAllocator, "f32_a", M, K) { (it % 10).toFloat() + 1.0f }
        val tensorQ40 = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q40_b", io.github.kotlinmania.llama.ore.GGMLType.Q4_0, K, N) { (it % 15).toFloat() - 7.0f }

        val optimized = GGMLTestUtils.allocateMatMulResult(graphAllocator, "optimized_q40", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorF32,
            tensorQ40,
            optimized
        )

        val tensorQ40F32 =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, tensorQ40)
        val fallback = GGMLTestUtils.allocateMatMulResult(graphAllocator, "fallback_q40", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorF32,
            tensorQ40F32,
            fallback
        )

        val optimizedData = GGMLTestUtils.extractFloatData(optimized, graphAllocator)
        val fallbackData = GGMLTestUtils.extractFloatData(fallback, graphAllocator)
        optimizedData.forEachIndexed { idx, value ->
            val diff = abs(value - fallbackData[idx])
            assertTrue(diff < 1e-3, "Results differ at index $idx: $value vs ${fallbackData[idx]}")
        }
    }

    /**
     * Test F32 x Q8_0 symmetric optimization
     */
    @Test
    fun testF32xQ80Optimization() {
        val M = 3; val K = io.github.kotlinmania.llama.ore.QK8_0 * 2; val N = 4
        
        val tensorF32 = GGMLTestUtils.createF32Matrix(graphAllocator, "f32_a", M, K) { (it % 13).toFloat() - 6.0f }
        val tensorQ80 = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q80_b", io.github.kotlinmania.llama.ore.GGMLType.Q8_0, K, N) { (it % 127).toFloat() - 63.0f }

        val optimized = GGMLTestUtils.allocateMatMulResult(graphAllocator, "optimized_q80", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorF32,
            tensorQ80,
            optimized
        )

        val tensorQ80F32 =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, tensorQ80)
        val fallback = GGMLTestUtils.allocateMatMulResult(graphAllocator, "fallback_q80", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorF32,
            tensorQ80F32,
            fallback
        )

        val optimizedData = GGMLTestUtils.extractFloatData(optimized, graphAllocator)
        val fallbackData = GGMLTestUtils.extractFloatData(fallback, graphAllocator)

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
        val M = 2; val K = io.github.kotlinmania.llama.ore.QK8_0; val N = 2
        
        val tensorQ80A = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q80_a", io.github.kotlinmania.llama.ore.GGMLType.Q8_0, M, K) { (it % 127).toFloat() - 63.0f }
        val tensorQ80B = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q80_b", io.github.kotlinmania.llama.ore.GGMLType.Q8_0, K, N) { ((it * 3) % 127).toFloat() - 63.0f }

        val optimized = GGMLTestUtils.allocateMatMulResult(graphAllocator, "optimized_q80xq80", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorQ80A,
            tensorQ80B,
            optimized
        )

        val tensorQ80AF32 =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, tensorQ80A)
        val tensorQ80BF32 =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, tensorQ80B)
        val fallback = GGMLTestUtils.allocateMatMulResult(graphAllocator, "fallback_q80xq80", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorQ80AF32,
            tensorQ80BF32,
            fallback
        )

        val optimizedData = GGMLTestUtils.extractFloatData(optimized, graphAllocator)
        val fallbackData = GGMLTestUtils.extractFloatData(fallback, graphAllocator)

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
        val M = 2; val K = io.github.kotlinmania.llama.ore.QK4_0; val N = 3
        
        val tensorQ40A = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q40_a", io.github.kotlinmania.llama.ore.GGMLType.Q4_0, M, K) { (it % 15).toFloat() - 7.0f }
        val tensorQ40B = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q40_b", io.github.kotlinmania.llama.ore.GGMLType.Q4_0, K, N) { ((it * 5) % 15).toFloat() - 7.0f }

        val resultOptimized = GGMLTestUtils.allocateMatMulResult(graphAllocator, "optimized_q40xq40", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorQ40A,
            tensorQ40B,
            resultOptimized
        )

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, resultOptimized.type)
        assertEquals(resultOptimized.ne[0], N.toLong())
        assertEquals(resultOptimized.ne[1], M.toLong())

        val optimizedData = GGMLTestUtils.extractFloatData(resultOptimized, graphAllocator)
        assertTrue(optimizedData.isNotEmpty())
    }

    /**
     * Test mixed quantization Q8_0 x Q4_0
     */
    @Test
    fun testQ80xQ40MixedOptimization() {
        val M = 2; val K = 64; val N = 2  // Use K divisible by both QK8_0 and QK4_0
        
        val tensorQ80 = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q80_a", io.github.kotlinmania.llama.ore.GGMLType.Q8_0, M, K) { (it % 127).toFloat() - 63.0f }
        val tensorQ40 = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "q40_b", io.github.kotlinmania.llama.ore.GGMLType.Q4_0, K, N) { ((it * 7) % 15).toFloat() - 7.0f }

        val resultOptimized = GGMLTestUtils.allocateMatMulResult(graphAllocator, "optimized_q80xq40", M, N)
        io.github.kotlinmania.llama.ore.computeMatMul(
            graphAllocator,
            tensorQ80,
            tensorQ40,
            resultOptimized
        )

        assertEquals(io.github.kotlinmania.llama.ore.GGMLType.F32, resultOptimized.type)
        assertEquals(resultOptimized.ne[0], N.toLong())
        assertEquals(resultOptimized.ne[1], M.toLong())

        val optimizedData = GGMLTestUtils.extractFloatData(resultOptimized, graphAllocator)
        assertTrue(optimizedData.isNotEmpty())

        optimizedData.forEach { value ->
            assertTrue(value.isFinite(), "Got non-finite value: $value")
        }
    }

    /**
     * Performance comparison test - measures time difference between optimized and fallback paths
     */
    @Test
    fun testPerformanceComparison() {
        val M = 8; val K = 128; val N = 8  // Larger matrices for performance testing
        
        val tensorF32 = GGMLTestUtils.createF32Matrix(graphAllocator, "perf_f32", M, K) { (it % 17).toFloat() - 8.0f }
        val tensorQ80 = GGMLTestUtils.createQuantizedMatrix(graphAllocator, "perf_q80", io.github.kotlinmania.llama.ore.GGMLType.Q8_0, K, N) { ((it * 11) % 127).toFloat() - 63.0f }

        val optimizedResult = GGMLTestUtils.allocateMatMulResult(graphAllocator, "perf_opt", M, N)
        repeat(5) {
            io.github.kotlinmania.llama.ore.computeMatMul(
                graphAllocator,
                tensorF32,
                tensorQ80,
                optimizedResult
            )
        }

        val optimizedMark = TimeSource.Monotonic.markNow()
        repeat(10) {
            io.github.kotlinmania.llama.ore.computeMatMul(
                graphAllocator,
                tensorF32,
                tensorQ80,
                optimizedResult
            )
        }
        val timeOptimized = optimizedMark.elapsedNow().inWholeNanoseconds

        val tensorQ80F32 =
            io.github.kotlinmania.llama.ore.dequantizeTensor(graphAllocator, tensorQ80)
        val fallbackResult = GGMLTestUtils.allocateMatMulResult(graphAllocator, "perf_fallback", M, N)
        repeat(5) {
            io.github.kotlinmania.llama.ore.computeMatMul(
                graphAllocator,
                tensorF32,
                tensorQ80F32,
                fallbackResult
            )
        }

        val fallbackMark = TimeSource.Monotonic.markNow()
        repeat(10) {
            io.github.kotlinmania.llama.ore.computeMatMul(
                graphAllocator,
                tensorF32,
                tensorQ80F32,
                fallbackResult
            )
        }
        val timeFallback = fallbackMark.elapsedNow().inWholeNanoseconds

        println("Optimized path: ${timeOptimized / 1_000_000}ms")
        println("Fallback path: ${timeFallback / 1_000_000}ms")
        if (timeOptimized > 0) {
            println("Speedup: ${timeFallback.toDouble() / timeOptimized.toDouble()}x")
        }

        assertTrue(timeOptimized <= timeFallback * 3 / 2,
            "Optimized path should not be significantly slower than fallback")
    }
}
