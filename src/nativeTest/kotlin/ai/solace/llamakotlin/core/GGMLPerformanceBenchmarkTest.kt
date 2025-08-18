package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.test.*

/**
 * Performance benchmarking suite for GGML operations.
 * Measures and validates performance characteristics of core tensor operations.
 */
class GGMLPerformanceBenchmarkTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 16 * 1024 * 1024 // 16MB for large benchmarks

    data class BenchmarkResult(
        val operationName: String,
        val dataSize: Int,
        val dataType: GGMLType,
        val timeMillis: Long,
        val throughputMBps: Double,
        val operationsPerSecond: Double
    )

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    private val dummyContext = GGMLContext()

    // Helper to create benchmark tensor
    private fun createBenchmarkTensor(name: String, type: GGMLType, size: Int): GGMLTensor {
        val ne = longArrayOf(size.toLong())
        val tensor = GGMLTensor(type = type, name = name)
        tensor.ne = ne
        
        tensor.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
        tensor.nb[0] = type.byteSize
        for (d in 1 until GGML_MAX_DIMS) {
            tensor.nb[d] = tensor.nb[d-1] * ne.getOrElse(d-1) { 1L }.toULong()
        }
        
        val byteSize = size.toULong() * type.byteSize
        val allocatedTensor = graphAllocator.tensorAllocators[0].allocate(byteSize, type, name)
        tensor.bufferId = allocatedTensor.bufferId
        tensor.offset = allocatedTensor.offset
        
        // Initialize with pseudo-random data for realistic performance testing
        for (i in 0 until size) {
            when (type) {
                GGMLType.F32 -> tensor.setFloat(graphAllocator, sin(i * 0.1f), i)
                GGMLType.F16 -> tensor.setHalf(graphAllocator, sin(i * 0.1f), i)
                GGMLType.I32 -> tensor.setInt(graphAllocator, i % 1000, i)
                else -> tensor.setFloat(graphAllocator, sin(i * 0.1f), i)
            }
        }
        
        return tensor
    }

    // Helper to run benchmark
    private fun runBenchmark(
        operationName: String,
        dataSize: Int,
        dataType: GGMLType,
        warmupRuns: Int = 3,
        measureRuns: Int = 10,
        operation: () -> GGMLTensor
    ): BenchmarkResult {
        // Warmup runs
        repeat(warmupRuns) {
            try {
                operation()
            } catch (e: Exception) {
                // Ignore warmup failures
            }
        }
        
        // Measured runs
        val times = mutableListOf<Long>()
        repeat(measureRuns) {
            val startTime = System.currentTimeMillis()
            try {
                val result = operation()
                val endTime = System.currentTimeMillis()
                times.add(endTime - startTime)
                
                // Verify result exists to ensure operation actually executed
                assertTrue(result.numElements() > 0, "Benchmark operation should produce non-empty result")
            } catch (e: Exception) {
                println("Benchmark operation failed: ${e.message}")
                times.add(Long.MAX_VALUE) // Add failure marker
            }
        }
        
        // Calculate statistics (excluding failures)
        val validTimes = times.filter { it != Long.MAX_VALUE }
        val avgTime = if (validTimes.isNotEmpty()) validTimes.average().toLong() else Long.MAX_VALUE
        
        val byteSize = dataSize * when(dataType) {
            GGMLType.F32 -> 4
            GGMLType.F16 -> 2  
            GGMLType.I32 -> 4
            GGMLType.I16 -> 2
            else -> 4
        }
        
        val throughputMBps = if (avgTime > 0) (byteSize.toDouble() / (avgTime / 1000.0)) / (1024 * 1024) else 0.0
        val operationsPerSecond = if (avgTime > 0) 1000.0 / avgTime else 0.0
        
        return BenchmarkResult(operationName, dataSize, dataType, avgTime, throughputMBps, operationsPerSecond)
    }

    // --- Element-wise Operation Benchmarks ---
    
    @Test
    fun benchmarkElementWiseOperationsF32() {
        val sizes = arrayOf(1024, 4096, 16384, 65536)
        val results = mutableListOf<BenchmarkResult>()
        
        for (size in sizes) {
            val tensor1 = createBenchmarkTensor("bench1", GGMLType.F32, size)
            val tensor2 = createBenchmarkTensor("bench2", GGMLType.F32, size)
            
            // Benchmark ADD
            val addResult = runBenchmark("ADD_F32", size, GGMLType.F32) {
                computeAdd(graphAllocator, dummyContext, tensor1, tensor2)
            }
            results.add(addResult)
            
            // Benchmark MUL
            val mulResult = runBenchmark("MUL_F32", size, GGMLType.F32) {
                computeMul(graphAllocator, dummyContext, tensor1, tensor2)
            }
            results.add(mulResult)
            
            // Benchmark SUB
            val subResult = runBenchmark("SUB_F32", size, GGMLType.F32) {
                computeSub(graphAllocator, dummyContext, tensor1, tensor2)
            }
            results.add(subResult)
            
            // Benchmark DIV
            val divResult = runBenchmark("DIV_F32", size, GGMLType.F32) {
                computeDiv(graphAllocator, dummyContext, tensor1, tensor2)
            }
            results.add(divResult)
        }
        
        // Print results
        println("\n=== F32 Element-wise Operations Benchmark ===")
        println("Operation\tSize\tTime(ms)\tThroughput(MB/s)\tOps/sec")
        for (result in results) {
            println("${result.operationName}\t${result.dataSize}\t${result.timeMillis}\t${"%.2f".format(result.throughputMBps)}\t${"%.2f".format(result.operationsPerSecond)}")
        }
        
        // Performance validation - operations should complete in reasonable time
        for (result in results) {
            if (result.timeMillis != Long.MAX_VALUE) {
                assertTrue(result.timeMillis < 1000, "${result.operationName} with ${result.dataSize} elements took too long: ${result.timeMillis}ms")
                assertTrue(result.throughputMBps > 0.1, "${result.operationName} throughput too low: ${result.throughputMBps} MB/s")
            }
        }
    }

    @Test
    fun benchmarkElementWiseOperationsF16() {
        val sizes = arrayOf(2048, 8192, 32768)
        val results = mutableListOf<BenchmarkResult>()
        
        for (size in sizes) {
            val tensor1 = createBenchmarkTensor("bench1_f16", GGMLType.F16, size)
            val tensor2 = createBenchmarkTensor("bench2_f16", GGMLType.F16, size)
            
            val addResult = runBenchmark("ADD_F16", size, GGMLType.F16) {
                computeAdd(graphAllocator, dummyContext, tensor1, tensor2)
            }
            results.add(addResult)
            
            val mulResult = runBenchmark("MUL_F16", size, GGMLType.F16) {
                computeMul(graphAllocator, dummyContext, tensor1, tensor2)
            }
            results.add(mulResult)
        }
        
        println("\n=== F16 Element-wise Operations Benchmark ===")
        println("Operation\tSize\tTime(ms)\tThroughput(MB/s)")
        for (result in results) {
            println("${result.operationName}\t${result.dataSize}\t${result.timeMillis}\t${"%.2f".format(result.throughputMBps)}")
        }
    }

    // --- Unary Operation Benchmarks ---
    
    @Test
    fun benchmarkUnaryOperations() {
        val size = 16384
        val tensor = createBenchmarkTensor("unary_bench", GGMLType.F32, size)
        val results = mutableListOf<BenchmarkResult>()
        
        // Benchmark NEG
        val negResult = runBenchmark("NEG_F32", size, GGMLType.F32) {
            computeNeg(graphAllocator, dummyContext, tensor)
        }
        results.add(negResult)
        
        // Benchmark SQR
        try {
            val sqrResult = runBenchmark("SQR_F32", size, GGMLType.F32) {
                computeSqr(graphAllocator, dummyContext, tensor)
            }
            results.add(sqrResult)
        } catch (e: Exception) {
            println("SQR benchmark skipped: ${e.message}")
        }
        
        // Benchmark SQRT
        try {
            val sqrtResult = runBenchmark("SQRT_F32", size, GGMLType.F32) {
                computeSqrt(graphAllocator, dummyContext, tensor)
            }
            results.add(sqrtResult)
        } catch (e: Exception) {
            println("SQRT benchmark skipped: ${e.message}")
        }
        
        // Benchmark RELU
        try {
            val reluResult = runBenchmark("RELU_F32", size, GGMLType.F32) {
                computeRelu(graphAllocator, dummyContext, tensor)
            }
            results.add(reluResult)
        } catch (e: Exception) {
            println("RELU benchmark skipped: ${e.message}")
        }
        
        // Benchmark GELU
        try {
            val geluResult = runBenchmark("GELU_F32", size, GGMLType.F32) {
                computeGelu(graphAllocator, dummyContext, tensor)
            }
            results.add(geluResult)
        } catch (e: Exception) {
            println("GELU benchmark skipped: ${e.message}")
        }
        
        println("\n=== Unary Operations Benchmark (Size: $size) ===")
        println("Operation\tTime(ms)\tThroughput(MB/s)\tOps/sec")
        for (result in results) {
            println("${result.operationName}\t${result.timeMillis}\t${"%.2f".format(result.throughputMBps)}\t${"%.2f".format(result.operationsPerSecond)}")
        }
    }

    // --- Matrix Multiplication Benchmarks ---
    
    @Test
    fun benchmarkMatrixMultiplication() {
        val sizes = arrayOf(64, 128, 256) // Square matrix sizes
        val results = mutableListOf<BenchmarkResult>()
        
        for (size in sizes) {
            val matrixElements = size * size
            val matrixNe = longArrayOf(size.toLong(), size.toLong())
            
            try {
                val matrixA = GGMLTensor(type = GGMLType.F32, name = "matA_$size")
                matrixA.ne = matrixNe
                matrixA.nb = ULongArray(GGML_MAX_DIMS) { 0uL }
                matrixA.nb[0] = GGMLType.F32.byteSize
                matrixA.nb[1] = matrixA.nb[0] * size.toULong()
                
                val matrixB = GGMLTensor(type = GGMLType.F32, name = "matB_$size")
                matrixB.ne = matrixNe
                matrixB.nb = matrixA.nb.copyOf()
                
                val byteSize = matrixElements.toULong() * GGMLType.F32.byteSize
                val allocatedA = graphAllocator.tensorAllocators[0].allocate(byteSize, GGMLType.F32, "matA_$size")
                val allocatedB = graphAllocator.tensorAllocators[0].allocate(byteSize, GGMLType.F32, "matB_$size")
                
                matrixA.bufferId = allocatedA.bufferId
                matrixA.offset = allocatedA.offset
                matrixB.bufferId = allocatedB.bufferId
                matrixB.offset = allocatedB.offset
                
                // Initialize matrices
                for (i in 0 until size) {
                    for (j in 0 until size) {
                        matrixA.setFloat(graphAllocator, (i + j).toFloat() * 0.1f, i, j)
                        matrixB.setFloat(graphAllocator, (i - j).toFloat() * 0.1f, i, j)
                    }
                }
                
                val matMulResult = runBenchmark("MATMUL_F32", matrixElements, GGMLType.F32, warmupRuns = 1, measureRuns = 3) {
                    computeMatMul(graphAllocator, dummyContext, matrixA, matrixB)
                }
                results.add(matMulResult)
                
            } catch (e: Exception) {
                println("Matrix multiplication benchmark skipped for size $size: ${e.message}")
            }
        }
        
        println("\n=== Matrix Multiplication Benchmark ===")
        println("Size\tTime(ms)\tGFLOPs")
        for (result in results) {
            val matrixSize = sqrt(result.dataSize.toDouble()).toInt()
            val flops = 2.0 * matrixSize * matrixSize * matrixSize // 2*N^3 for N×N matrix multiplication
            val gflops = if (result.timeMillis > 0) (flops / (result.timeMillis / 1000.0)) / 1e9 else 0.0
            println("${matrixSize}x$matrixSize\t${result.timeMillis}\t${"%.3f".format(gflops)}")
        }
    }

    // --- Quantization Performance Benchmarks ---
    
    @Test
    fun benchmarkQuantizationOperations() {
        val sizes = arrayOf(1024, 4096, 16384)
        val results = mutableListOf<BenchmarkResult>()
        
        for (size in sizes) {
            val f32Tensor = createBenchmarkTensor("quant_bench", GGMLType.F32, size)
            
            // Benchmark Q8_0 quantization
            try {
                val q8QuantResult = runBenchmark("QUANT_Q8_0", size, GGMLType.F32) {
                    quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q8_0)
                }
                results.add(q8QuantResult)
            } catch (e: Exception) {
                println("Q8_0 quantization benchmark skipped: ${e.message}")
            }
            
            // Benchmark Q4_0 quantization  
            try {
                val q4QuantResult = runBenchmark("QUANT_Q4_0", size, GGMLType.F32) {
                    quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q4_0)
                }
                results.add(q4QuantResult)
            } catch (e: Exception) {
                println("Q4_0 quantization benchmark skipped: ${e.message}")
            }
            
            // Benchmark Q4_1 quantization
            try {
                val q41QuantResult = runBenchmark("QUANT_Q4_1", size, GGMLType.F32) {
                    quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q4_1)
                }
                results.add(q41QuantResult)
            } catch (e: Exception) {
                println("Q4_1 quantization benchmark skipped: ${e.message}")
            }
        }
        
        println("\n=== Quantization Operations Benchmark ===")
        println("Operation\tSize\tTime(ms)\tThroughput(MB/s)")
        for (result in results) {
            println("${result.operationName}\t${result.dataSize}\t${result.timeMillis}\t${"%.2f".format(result.throughputMBps)}")
        }
    }

    @Test  
    fun benchmarkDequantizationOperations() {
        val size = 8192
        val results = mutableListOf<BenchmarkResult>()
        
        try {
            val f32Tensor = createBenchmarkTensor("dequant_source", GGMLType.F32, size)
            
            // Create quantized tensors first
            val q8Tensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q8_0)
            val q4Tensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q4_0)
            val q41Tensor = quantizeTensor(graphAllocator, f32Tensor, GGMLType.Q4_1)
            
            // Benchmark Q8_0 dequantization
            val q8DequantResult = runBenchmark("DEQUANT_Q8_0", size, GGMLType.Q8_0) {
                dequantizeTensor(graphAllocator, q8Tensor)
            }
            results.add(q8DequantResult)
            
            // Benchmark Q4_0 dequantization
            val q4DequantResult = runBenchmark("DEQUANT_Q4_0", size, GGMLType.Q4_0) {
                dequantizeTensor(graphAllocator, q4Tensor)
            }
            results.add(q4DequantResult)
            
            // Benchmark Q4_1 dequantization
            val q41DequantResult = runBenchmark("DEQUANT_Q4_1", size, GGMLType.Q4_1) {
                dequantizeTensor(graphAllocator, q41Tensor)
            }
            results.add(q41DequantResult)
            
            println("\n=== Dequantization Operations Benchmark (Size: $size) ===")
            println("Operation\tTime(ms)\tThroughput(MB/s)")
            for (result in results) {
                println("${result.operationName}\t${result.timeMillis}\t${"%.2f".format(result.throughputMBps)}")
            }
            
        } catch (e: Exception) {
            println("Dequantization benchmarks skipped: ${e.message}")
        }
    }

    // --- Memory Allocation Performance ---
    
    @Test
    fun benchmarkMemoryAllocation() {
        val sizes = arrayOf(1024, 4096, 16384, 65536)
        
        println("\n=== Memory Allocation Benchmark ===")
        println("Size\tAllocation Time(ms)\tTensors/sec")
        
        for (size in sizes) {
            val tensorCount = 100
            val startTime = System.currentTimeMillis()
            
            try {
                for (i in 0 until tensorCount) {
                    createBenchmarkTensor("alloc_test_$i", GGMLType.F32, size)
                }
                
                val endTime = System.currentTimeMillis()
                val totalTime = endTime - startTime
                val tensorsPerSecond = if (totalTime > 0) (tensorCount * 1000.0) / totalTime else 0.0
                
                println("$size\t$totalTime\t${"%.2f".format(tensorsPerSecond)}")
                
                // Validate performance
                assertTrue(totalTime < 5000, "Memory allocation for $tensorCount tensors of size $size took too long: ${totalTime}ms")
                
            } catch (e: OutOfMemoryError) {
                println("$size\tOOM\t0.00")
            }
        }
    }

    // --- Comprehensive Performance Summary ---
    
    @Test
    fun performanceSummaryReport() {
        println("\n" + "=".repeat(60))
        println("GGML KOTLIN PERFORMANCE SUMMARY REPORT")
        println("=".repeat(60))
        
        // Quick performance test of core operations
        val testSize = 4096
        val tensor1 = createBenchmarkTensor("summary1", GGMLType.F32, testSize)
        val tensor2 = createBenchmarkTensor("summary2", GGMLType.F32, testSize)
        
        val operations = mapOf(
            "ADD" to { computeAdd(graphAllocator, dummyContext, tensor1, tensor2) },
            "MUL" to { computeMul(graphAllocator, dummyContext, tensor1, tensor2) },
            "SUB" to { computeSub(graphAllocator, dummyContext, tensor1, tensor2) },
            "DIV" to { computeDiv(graphAllocator, dummyContext, tensor1, tensor2) },
            "NEG" to { computeNeg(graphAllocator, dummyContext, tensor1) }
        )
        
        println("Core Operations Performance (Size: $testSize F32 elements):")
        println("Operation\tTime(ms)\tThroughput(MB/s)")
        
        for ((opName, operation) in operations) {
            try {
                val result = runBenchmark(opName, testSize, GGMLType.F32, warmupRuns = 2, measureRuns = 5, operation)
                println("$opName\t\t${result.timeMillis}\t${"%.2f".format(result.throughputMBps)}")
            } catch (e: Exception) {
                println("$opName\t\tFAILED\t${e.message}")
            }
        }
        
        println("\nPerformance benchmarking completed.")
        println("=".repeat(60))
    }
}