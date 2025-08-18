package ai.solace.llamakotlin.core

import kotlin.math.abs
import kotlin.test.*

/**
 * Performance microbenchmarks for quantized matmul operations.
 * Profiles all quantization combinations and measures performance improvements.
 */
class GGMLMatMulBenchmarkTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 4 * 1024 * 1024 // 4MB for larger test matrices

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())
        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    // Helper to create test matrices of various sizes
    private fun createTestMatrix(
        name: String,
        type: GGMLType,
        rows: Int,
        cols: Int,
        seed: Int = 42
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type)
        tensor.name = name
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        tensor.ne[0] = cols.toLong()  // First dim is column count
        tensor.ne[1] = rows.toLong()  // Second dim is row count

        // Calculate strides
        if (type.byteSize > 0uL) {
            tensor.nb[0] = type.byteSize
            for (d in 1 until GGML_MAX_DIMS) {
                tensor.nb[d] = tensor.ne[d-1].toULong() * tensor.nb[d-1]
            }
        }

        val numElements = (rows * cols).toLong()
        
        when (type) {
            GGMLType.F32 -> {
                // Create deterministic test data based on seed
                val data = FloatArray(numElements.toInt()) { idx ->
                    val x = (seed + idx) % 127
                    (x - 63).toFloat() / 10.0f  // Range approximately -6.3 to +6.3
                }
                tensor.data = data
            }
            GGMLType.Q8_0 -> {
                require(numElements % QK8_0 == 0) { "Q8_0 matrix elements must be divisible by $QK8_0" }
                val f32Data = FloatArray(numElements.toInt()) { idx ->
                    val x = (seed + idx) % 254
                    (x - 127).toFloat()  // Range -127 to +127 for Q8_0
                }
                tensor.data = quantizeToQ80(f32Data)
            }
            GGMLType.Q4_0 -> {
                require(numElements % QK4_0 == 0) { "Q4_0 matrix elements must be divisible by $QK4_0" }
                val f32Data = FloatArray(numElements.toInt()) { idx ->
                    val x = (seed + idx) % 16
                    (x - 8).toFloat()  // Range -8 to +7 for Q4_0
                }
                tensor.data = quantizeToQ40(f32Data)
            }
            GGMLType.Q4_1 -> {
                require(numElements % QK4_1 == 0) { "Q4_1 matrix elements must be divisible by $QK4_1" }
                val f32Data = FloatArray(numElements.toInt()) { idx ->
                    val x = (seed + idx) % 20
                    x.toFloat()  // Range 0 to +19 for Q4_1
                }
                tensor.data = quantizeToQ41(f32Data)
            }
            else -> throw IllegalArgumentException("Unsupported type for benchmark: $type")
        }

        return tensor
    }

    // Quantization helpers
    private fun quantizeToQ80(f32Data: FloatArray): ByteArray {
        val numElements = f32Data.size
        val numBlocks = numElements / QK8_0
        val blockSize = 2 + QK8_0  // F16 scale + 32 bytes
        val result = ByteArray(numBlocks * blockSize)
        
        for (blockIdx in 0 until numBlocks) {
            val startIdx = blockIdx * QK8_0
            var absMax = 0.0f
            for (i in startIdx until startIdx + QK8_0) {
                absMax = maxOf(absMax, abs(f32Data[i]))
            }
            val scale = if (absMax == 0.0f) 1.0f else absMax / 127.0f
            val invScale = 1.0f / scale
            
            // Write scale
            result.setShortLe(blockIdx * blockSize, floatToHalf(scale))
            
            // Write quantized weights
            val weightOffset = blockIdx * blockSize + 2
            for (i in 0 until QK8_0) {
                val quantValue = kotlin.math.round(f32Data[startIdx + i] * invScale).toInt().coerceIn(-128, 127)
                result[weightOffset + i] = quantValue.toByte()
            }
        }
        return result
    }

    private fun quantizeToQ40(f32Data: FloatArray): ByteArray {
        val numElements = f32Data.size
        val numBlocks = numElements / QK4_0
        val blockSize = 2 + QK4_0 / 2  // F16 scale + 16 packed bytes
        val result = ByteArray(numBlocks * blockSize)
        
        for (blockIdx in 0 until numBlocks) {
            val startIdx = blockIdx * QK4_0
            var absMax = 0.0f
            for (i in startIdx until startIdx + QK4_0) {
                absMax = maxOf(absMax, abs(f32Data[i]))
            }
            val scale = if (absMax == 0.0f) 1.0f else absMax / 8.0f
            val invScale = 1.0f / scale
            
            // Write scale
            result.setShortLe(blockIdx * blockSize, floatToHalf(scale))
            
            // Write quantized weights (packed)
            val weightOffset = blockIdx * blockSize + 2
            for (i in 0 until QK4_0 / 2) {
                val q1 = kotlin.math.round(f32Data[startIdx + i*2] * invScale + 8.0f).toInt().coerceIn(0, 15)
                val q2 = kotlin.math.round(f32Data[startIdx + i*2 + 1] * invScale + 8.0f).toInt().coerceIn(0, 15)
                result[weightOffset + i] = ((q1 and 0x0F) or ((q2 and 0x0F) shl 4)).toByte()
            }
        }
        return result
    }

    private fun quantizeToQ41(f32Data: FloatArray): ByteArray {
        val numElements = f32Data.size
        val numBlocks = numElements / QK4_1
        val blockSize = 4 + QK4_1 / 2  // 2*F16 (scale+min) + 16 packed bytes
        val result = ByteArray(numBlocks * blockSize)
        
        for (blockIdx in 0 until numBlocks) {
            val startIdx = blockIdx * QK4_1
            var minVal = f32Data[startIdx]
            var maxVal = f32Data[startIdx]
            for (i in startIdx until startIdx + QK4_1) {
                minVal = minOf(minVal, f32Data[i])
                maxVal = maxOf(maxVal, f32Data[i])
            }
            val scale = if (maxVal == minVal) 1.0f else (maxVal - minVal) / 15.0f
            val invScale = 1.0f / scale
            
            // Write scale and min
            result.setShortLe(blockIdx * blockSize, floatToHalf(scale))
            result.setShortLe(blockIdx * blockSize + 2, floatToHalf(minVal))
            
            // Write quantized weights
            val weightOffset = blockIdx * blockSize + 4
            for (i in 0 until QK4_1 / 2) {
                val q1 = kotlin.math.round((f32Data[startIdx + i*2] - minVal) * invScale).toInt().coerceIn(0, 15)
                val q2 = kotlin.math.round((f32Data[startIdx + i*2 + 1] - minVal) * invScale).toInt().coerceIn(0, 15)
                result[weightOffset + i] = ((q1 and 0x0F) or ((q2 and 0x0F) shl 4)).toByte()
            }
        }
        return result
    }

    // Benchmarking helper
    private fun benchmarkMatMul(
        name: String,
        tensorA: GGMLTensor,
        tensorB: GGMLTensor,
        warmupRuns: Int = 5,
        benchmarkRuns: Int = 10
    ): Long {
        // Warmup
        repeat(warmupRuns) {
            computeMatMul(graphAllocator, GGMLContext(), tensorA, tensorB)
        }
        
        // Benchmark
        val startTime = kotlin.system.getTimeNanos()
        repeat(benchmarkRuns) {
            computeMatMul(graphAllocator, GGMLContext(), tensorA, tensorB)
        }
        val totalTime = kotlin.system.getTimeNanos() - startTime
        val avgTime = totalTime / benchmarkRuns
        
        println("$name: ${avgTime / 1_000_000}ms avg (${benchmarkRuns} runs)")
        return avgTime
    }

    /**
     * Benchmark matrix sizes commonly used in LLM inference
     */
    @Test
    fun benchmarkCommonMatrixSizes() {
        println("\n=== Common Matrix Size Benchmarks ===")
        
        val testSizes = listOf(
            Triple(128, 512, 256),  // Small attention head
            Triple(256, 1024, 512), // Medium layer
            Triple(512, 2048, 1024) // Large layer
        )
        
        for ((M, K, N) in testSizes) {
            println("\nMatrix size: ${M}x${K} × ${K}x${N} = ${M}x${N}")
            
            // F32 baseline
            val f32A = createTestMatrix("f32_a", GGMLType.F32, M, K, 42)
            val f32B = createTestMatrix("f32_b", GGMLType.F32, K, N, 84)
            val baselineTime = benchmarkMatMul("F32×F32 (baseline)", f32A, f32B)
            
            // Q8_0 combinations
            val q80A = createTestMatrix("q80_a", GGMLType.Q8_0, M, K, 42)
            val q80B = createTestMatrix("q80_b", GGMLType.Q8_0, K, N, 84)
            
            val q80f32Time = benchmarkMatMul("Q8_0×F32 (optimized)", q80A, f32B)
            val f32q80Time = benchmarkMatMul("F32×Q8_0 (optimized)", f32A, q80B)
            val q80q80Time = benchmarkMatMul("Q8_0×Q8_0 (optimized)", q80A, q80B)
            
            // Q4_0 combinations
            val q40A = createTestMatrix("q40_a", GGMLType.Q4_0, M, K, 42)
            val q40B = createTestMatrix("q40_b", GGMLType.Q4_0, K, N, 84)
            
            val q40f32Time = benchmarkMatMul("Q4_0×F32 (optimized)", q40A, f32B)
            val f32q40Time = benchmarkMatMul("F32×Q4_0 (optimized)", f32A, q40B)
            val q40q40Time = benchmarkMatMul("Q4_0×Q4_0 (optimized)", q40A, q40B)
            
            // Print speedup analysis
            println("\nSpeedup vs F32×F32 baseline:")
            println("  Q8_0×F32: ${baselineTime.toFloat() / q80f32Time}x")
            println("  F32×Q8_0: ${baselineTime.toFloat() / f32q80Time}x")
            println("  Q8_0×Q8_0: ${baselineTime.toFloat() / q80q80Time}x")
            println("  Q4_0×F32: ${baselineTime.toFloat() / q40f32Time}x")
            println("  F32×Q4_0: ${baselineTime.toFloat() / f32q40Time}x")
            println("  Q4_0×Q4_0: ${baselineTime.toFloat() / q40q40Time}x")
        }
    }

    /**
     * Benchmark all supported quantization combinations
     */
    @Test
    fun benchmarkAllQuantCombinations() {
        println("\n=== All Quantization Combinations Benchmark ===")
        
        val M = 64; val K = 256; val N = 128  // Medium-sized matrices
        
        val types = listOf(GGMLType.F32, GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_1)
        val results = mutableMapOf<String, Long>()
        
        for (typeA in types) {
            for (typeB in types) {
                try {
                    val tensorA = createTestMatrix("tensor_a", typeA, M, K, 42)
                    val tensorB = createTestMatrix("tensor_b", typeB, K, N, 84)
                    
                    val combName = "${typeA.name}×${typeB.name}"
                    val time = benchmarkMatMul(combName, tensorA, tensorB, warmupRuns = 3, benchmarkRuns = 5)
                    results[combName] = time
                    
                } catch (e: Exception) {
                    println("${typeA.name}×${typeB.name}: FAILED (${e.message})")
                }
            }
        }
        
        // Analysis
        println("\nPerformance Analysis:")
        val baselineTime = results["F32×F32"] ?: 0L
        if (baselineTime > 0) {
            results.entries.sortedBy { it.value }.forEach { (name, time) ->
                val speedup = baselineTime.toFloat() / time
                val efficiency = if (name.contains("Q")) "OPTIMIZED" else "BASELINE"
                println("  $name: ${speedup}x ($efficiency)")
            }
        }
    }

    /**
     * Stress test with large matrices to measure memory efficiency
     */
    @Test
    fun stressTestLargeMatrices() {
        println("\n=== Large Matrix Stress Test ===")
        
        // Test progressively larger matrices to find limits
        val sizes = listOf(
            Triple(32, 128, 64),
            Triple(64, 256, 128),
            Triple(128, 512, 256)
        )
        
        for ((M, K, N) in sizes) {
            println("\nTesting ${M}×${K} × ${K}×${N}")
            
            try {
                val q80A = createTestMatrix("large_q80_a", GGMLType.Q8_0, M, K)
                val q40B = createTestMatrix("large_q40_b", GGMLType.Q4_0, K, N)
                
                val mixedTime = benchmarkMatMul("Q8_0×Q4_0 (${M}×${K}×${N})", q80A, q40B, warmupRuns = 2, benchmarkRuns = 3)
                
                // Memory usage estimate
                val memoryUsed = (M * K * 34/32 + K * N * 18/32 + M * N * 4) / 1024  // Rough estimate in KB
                println("  Estimated memory: ${memoryUsed}KB")
                
            } catch (e: OutOfMemoryError) {
                println("  OUT OF MEMORY at size ${M}×${K}×${N}")
                break
            } catch (e: Exception) {
                println("  FAILED: ${e.message}")
            }
        }
    }

    /**
     * Accuracy validation for all optimized paths
     */
    @Test
    fun validateOptimizedAccuracy() {
        println("\n=== Accuracy Validation ===")
        
        val M = 16; val K = 64; val N = 32
        val tolerance = 1e-2f  // Allow for quantization errors
        
        // Create reference F32 matrices
        val f32A = createTestMatrix("ref_a", GGMLType.F32, M, K, 123)
        val f32B = createTestMatrix("ref_b", GGMLType.F32, K, N, 456)
        val reference = computeMatMul(graphAllocator, GGMLContext(), f32A, f32B)
        val refData = reference.data as FloatArray
        
        // Test quantized combinations
        val testCases = listOf(
            Pair("Q8_0×F32", Pair(GGMLType.Q8_0, GGMLType.F32)),
            Pair("F32×Q8_0", Pair(GGMLType.F32, GGMLType.Q8_0)),
            Pair("Q8_0×Q8_0", Pair(GGMLType.Q8_0, GGMLType.Q8_0)),
            Pair("Q4_0×F32", Pair(GGMLType.Q4_0, GGMLType.F32)),
            Pair("F32×Q4_0", Pair(GGMLType.F32, GGMLType.Q4_0)),
            Pair("Q4_0×Q4_0", Pair(GGMLType.Q4_0, GGMLType.Q4_0))
        )
        
        for ((name, types) in testCases) {
            try {
                val testA = if (types.first == GGMLType.F32) f32A 
                          else createTestMatrix("test_a", types.first, M, K, 123)
                val testB = if (types.second == GGMLType.F32) f32B 
                          else createTestMatrix("test_b", types.second, K, N, 456)
                
                val result = computeMatMul(graphAllocator, GGMLContext(), testA, testB)
                val resultData = result.data as FloatArray
                
                var maxError = 0.0f
                var avgError = 0.0f
                for (i in refData.indices) {
                    val error = abs(refData[i] - resultData[i])
                    maxError = maxOf(maxError, error)
                    avgError += error
                }
                avgError /= refData.size
                
                val passed = maxError <= tolerance
                println("  $name: max_err=$maxError, avg_err=$avgError [${if (passed) "PASS" else "FAIL"}]")
                
                if (!passed) {
                    println("    First few mismatches:")
                    for (i in 0 until minOf(5, refData.size)) {
                        if (abs(refData[i] - resultData[i]) > tolerance) {
                            println("      [$i] ref=${refData[i]}, got=${resultData[i]}, diff=${abs(refData[i] - resultData[i])}")
                        }
                    }
                }
                
            } catch (e: Exception) {
                println("  $name: ERROR - ${e.message}")
            }
        }
    }

    /**
     * Profile specific dot product kernels
     */
    @Test
    fun profileDotProductKernels() {
        println("\n=== Dot Product Kernel Profiling ===")
        
        val K = 1024  // Long vectors for dot product testing
        val iterations = 1000
        
        // Create test vectors
        val f32Vec = createTestMatrix("f32_vec", GGMLType.F32, 1, K, 42)
        val q80Vec = createTestMatrix("q80_vec", GGMLType.Q8_0, 1, K, 42)
        val q40Vec = createTestMatrix("q40_vec", GGMLType.Q4_0, 1, K, 42)
        
        // Profile individual dot product operations
        println("Single dot product timing (${iterations} iterations):")
        
        // F32 × Q8_0
        var startTime = kotlin.system.getTimeNanos()
        repeat(iterations) {
            computeDotProductF32Q80(graphAllocator, f32Vec, q80Vec, 0, 0, K)
        }
        val f32q80Time = (kotlin.system.getTimeNanos() - startTime) / iterations
        println("  F32×Q8_0: ${f32q80Time / 1000}µs per dot product")
        
        // Q8_0 × Q8_0
        startTime = kotlin.system.getTimeNanos()
        repeat(iterations) {
            computeDotProductQ80Q80(graphAllocator, q80Vec, q80Vec, 0, 0, K)
        }
        val q80q80Time = (kotlin.system.getTimeNanos() - startTime) / iterations
        println("  Q8_0×Q8_0: ${q80q80Time / 1000}µs per dot product")
        
        // Q4_0 × Q4_0
        startTime = kotlin.system.getTimeNanos()
        repeat(iterations) {
            computeDotProductQ40Q40(graphAllocator, q40Vec, q40Vec, 0, 0, K)
        }
        val q40q40Time = (kotlin.system.getTimeNanos() - startTime) / iterations
        println("  Q4_0×Q4_0: ${q40q40Time / 1000}µs per dot product")
        
        // Efficiency analysis
        println("Dot product efficiency:")
        println("  Q8_0×Q8_0 vs F32×Q8_0: ${f32q80Time.toFloat() / q80q80Time}x")
        println("  Q4_0×Q4_0 vs F32×Q8_0: ${f32q80Time.toFloat() / q40q40Time}x")
    }
}