package ai.solace.llamakotlin.core

import kotlin.math.*
import kotlin.random.Random

/**
 * Common utilities and helper functions for GGML testing.
 * This file contains reusable test infrastructure to reduce code duplication
 * and ensure consistent testing patterns across all test files.
 */
object GGMLTestUtils {
    
    // Default buffer size for test allocations
    const val DEFAULT_TEST_BUFFER_SIZE = 4 * 1024 * 1024 // 4MB
    
    /**
     * Initialize a standard test allocator with buffer
     */
    fun createTestAllocator(bufferSize: Int = DEFAULT_TEST_BUFFER_SIZE): Pair<GGMLGraphAllocator, ByteArray> {
        val graphAllocator = GGMLGraphAllocator()
        val testBuffer = ByteArray(bufferSize)
        
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
        
        return Pair(graphAllocator, testBuffer)
    }
    
    /**
     * Calculate tensor byte size for any tensor type
     */
    fun calculateTensorByteSize(type: GGMLType, ne: LongArray): ULong {
        if (type.byteSize == 0uL && type != GGMLType.COUNT && !type.name.startsWith("Q")) {
            return 0uL
        }
        
        var elements = 1UL
        var validDimFound = false
        
        for (i in ne.indices) {
            if (ne[i] > 0L) {
                elements *= ne[i].toULong()
                validDimFound = true
            } else if (ne[i] == 0L && elements != 0UL && validDimFound) {
                return 0UL
            }
        }
        
        if (!validDimFound && ne.isNotEmpty() && ne.all { it <= 1L }) {
            elements = 1UL // Scalar or effectively scalar
        } else if (!validDimFound && ne.isEmpty()) {
            elements = 1UL // Treat as scalar if ne is completely empty
        }

        // Handle quantized types
        return when (type) {
            GGMLType.Q8_0 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % QK8_0 != 0L) {
                        println("Warning: Total elements $elements for Q8_0 is not divisible by block size $QK8_0.")
                    }
                    (elements.toLong() / QK8_0).toULong() * type.byteSize
                } else 0uL
            }
            GGMLType.Q4_0 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % QK4_0 != 0L) {
                        println("Warning: Total elements $elements for Q4_0 is not divisible by block size $QK4_0.")
                    }
                    (elements.toLong() / QK4_0).toULong() * type.byteSize
                } else 0uL
            }
            GGMLType.Q4_1 -> {
                if (elements > 0uL) {
                    if (elements.toLong() % QK4_1 != 0L) {
                        println("Warning: Total elements $elements for Q4_1 is not divisible by block size $QK4_1.")
                    }
                    (elements.toLong() / QK4_1).toULong() * type.byteSize
                } else 0uL
            }
            else -> elements * type.byteSize
        }
    }
    
    /**
     * Calculate contiguous tensor strides
     */
    fun calculateStrides(type: GGMLType, ne: LongArray, maxDims: Int = GGML_MAX_DIMS): ULongArray {
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
    
    /**
     * Create and initialize a tensor with data
     */
    fun createTensorWithData(
        graphAllocator: GGMLGraphAllocator,
        name: String,
        type: GGMLType,
        ne: LongArray,
        data: FloatArray,
        dataOffset: ULong = 0uL
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type, name = name)
        tensor.ne = ne.copyOf()
        tensor.nb = calculateStrides(type, ne)
        
        val byteSize = calculateTensorByteSize(type, ne)
        if (byteSize > 0uL) {
            val allocatedTensor = graphAllocator.tensorAllocators[0].allocate(byteSize, type, name)
            tensor.bufferId = allocatedTensor.bufferId
            tensor.offset = allocatedTensor.offset
        }
        
        // Set data based on tensor type
        for (i in data.indices) {
            when (type) {
                GGMLType.F32 -> tensor.setFloat(graphAllocator, data[i], i)
                GGMLType.F16 -> tensor.setHalf(graphAllocator, data[i], i)
                GGMLType.I32 -> tensor.setInt(graphAllocator, data[i].toInt(), i)
                GGMLType.I16 -> tensor.setShort(graphAllocator, data[i].toInt().toShort(), i)
                else -> tensor.setFloat(graphAllocator, data[i], i)
            }
        }
        
        return tensor
    }
    
    /**
     * Extract float data from any tensor type
     */
    fun extractFloatData(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return floatArrayOf()
        
        val result = FloatArray(numElements)
        
        for (i in 0 until numElements) {
            result[i] = when (tensor.type) {
                GGMLType.F32 -> tensor.getFloat(graphAllocator, i)
                GGMLType.F16 -> halfToFloat(tensor.getHalf(graphAllocator, i))
                GGMLType.I32 -> tensor.getInt(graphAllocator, i).toFloat()
                GGMLType.I16 -> tensor.getShort(graphAllocator, i).toFloat()
                else -> tensor.getFloat(graphAllocator, i)
            }
        }
        
        return result
    }
    
    /**
     * Data generators for consistent test data across test files
     */
    object DataGenerators {
        
        /**
         * Generate synthetic data using cosine pattern (matches upstream llama.cpp)
         */
        fun syntheticCosine(size: Int, offset: Float = 0.0f): FloatArray {
            return FloatArray(size) { i ->
                0.1f + 2.0f * cos(i.toFloat() + offset)
            }
        }
        
        /**
         * Generate random data with controlled distribution
         */
        fun randomUniform(size: Int, seed: Long = 12345, range: Float = 10.0f): FloatArray {
            val random = Random(seed)
            return FloatArray(size) { 
                random.nextFloat() * 2 * range - range
            }
        }
        
        /**
         * Generate random normal distribution data
         */
        fun randomNormal(size: Int, seed: Long = 12345, mean: Float = 0.0f, stdDev: Float = 1.0f): FloatArray {
            val random = Random(seed)
            return FloatArray(size) {
                val u1 = random.nextFloat()
                val u2 = random.nextFloat()
                val z0 = sqrt(-2.0f * ln(u1)) * cos(2.0f * PI.toFloat() * u2)
                mean + z0 * stdDev
            }
        }
        
        /**
         * Generate edge case values
         */
        fun edgeCases(): FloatArray {
            return floatArrayOf(
                0.0f, -0.0f,
                Float.MIN_VALUE, -Float.MIN_VALUE,
                Float.MAX_VALUE / 1000, -Float.MAX_VALUE / 1000,
                1.0f, -1.0f,
                10.0f, -10.0f,
                100.0f, -100.0f,
                0.001f, -0.001f,
                PI.toFloat(), -PI.toFloat(),
                E.toFloat(), -E.toFloat(),
                1e-6f, -1e-6f,
                1e6f, -1e6f
            )
        }
        
        /**
         * Generate identity matrix data
         */
        fun identityMatrix(size: Int): FloatArray {
            return FloatArray(size * size) { i ->
                if (i % (size + 1) == 0) 1.0f else 0.0f
            }
        }
        
        /**
         * Generate linear sequence data
         */
        fun linearSequence(size: Int, start: Float = 0.0f, step: Float = 1.0f): FloatArray {
            return FloatArray(size) { i ->
                start + i * step
            }
        }
    }
    
    /**
     * Error analysis utilities
     */
    object ErrorAnalysis {
        
        data class ErrorMetrics(
            val mse: Double,
            val rmse: Double,
            val mad: Double,
            val maxError: Double,
            val snr: Double,
            val relativeError: Double,
            val percentileErrors: Map<Int, Double> // 50th, 90th, 95th, 99th percentiles
        )
        
        /**
         * Calculate comprehensive error metrics between two arrays
         */
        fun calculateErrorMetrics(reference: FloatArray, actual: FloatArray): ErrorMetrics {
            require(reference.size == actual.size) { "Arrays must have same size" }
            
            val errors = DoubleArray(reference.size)
            var sumSquaredError = 0.0
            var sumAbsoluteError = 0.0
            var sumReferenceSquared = 0.0
            var maxError = 0.0
            
            for (i in reference.indices) {
                val error = (reference[i] - actual[i]).toDouble()
                val absError = abs(error)
                
                errors[i] = absError
                sumSquaredError += error * error
                sumAbsoluteError += absError
                sumReferenceSquared += reference[i].toDouble() * reference[i].toDouble()
                maxError = maxOf(maxError, absError)
            }
            
            val mse = sumSquaredError / reference.size
            val rmse = sqrt(mse)
            val mad = sumAbsoluteError / reference.size
            
            val snr = if (sumSquaredError == 0.0) Double.POSITIVE_INFINITY
                     else if (sumReferenceSquared == 0.0) Double.NEGATIVE_INFINITY  
                     else 10 * log10(sumReferenceSquared / sumSquaredError)
            
            val maxRef = reference.maxOf { abs(it).toDouble() }
            val relativeError = if (maxRef > 1e-10) maxError / maxRef else maxError
            
            // Calculate percentile errors
            val sortedErrors = errors.sorted()
            val percentiles = mapOf(
                50 to sortedErrors[(sortedErrors.size * 0.5).toInt()],
                90 to sortedErrors[(sortedErrors.size * 0.9).toInt()],
                95 to sortedErrors[(sortedErrors.size * 0.95).toInt()],
                99 to sortedErrors[minOf((sortedErrors.size * 0.99).toInt(), sortedErrors.size - 1)]
            )
            
            return ErrorMetrics(
                mse = mse,
                rmse = rmse,
                mad = mad,
                maxError = maxError,
                snr = snr,
                relativeError = relativeError,
                percentileErrors = percentiles
            )
        }
        
        /**
         * Check if error metrics pass given thresholds
         */
        fun passesThresholds(
            metrics: ErrorMetrics,
            mseThreshold: Double = Double.MAX_VALUE,
            madThreshold: Double = Double.MAX_VALUE,
            maxErrorThreshold: Double = Double.MAX_VALUE,
            snrThreshold: Double = Double.MIN_VALUE,
            relativeErrorThreshold: Double = Double.MAX_VALUE
        ): Boolean {
            return metrics.mse <= mseThreshold &&
                   metrics.mad <= madThreshold &&
                   metrics.maxError <= maxErrorThreshold &&
                   metrics.snr >= snrThreshold &&
                   metrics.relativeError <= relativeErrorThreshold
        }
    }
    
    /**
     * Performance measurement utilities
     */
    object PerformanceUtils {
        
        data class BenchmarkResult(
            val operationName: String,
            val dataSize: Int,
            val iterations: Int,
            val avgTimeMillis: Long,
            val minTimeMillis: Long,
            val maxTimeMillis: Long,
            val throughputMBps: Double,
            val operationsPerSecond: Double
        )
        
        /**
         * Run a benchmarked operation with warmup
         */
        fun benchmark(
            operationName: String,
            dataSize: Int,
            warmupRuns: Int = 3,
            measureRuns: Int = 10,
            operation: () -> Unit
        ): BenchmarkResult {
            // Warmup
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
                    operation()
                    val endTime = System.currentTimeMillis()
                    times.add(endTime - startTime)
                } catch (e: Exception) {
                    times.add(Long.MAX_VALUE) // Failure marker
                }
            }
            
            val validTimes = times.filter { it != Long.MAX_VALUE }
            val avgTime = if (validTimes.isNotEmpty()) validTimes.average().toLong() else Long.MAX_VALUE
            val minTime = validTimes.minOrNull() ?: Long.MAX_VALUE
            val maxTime = validTimes.maxOrNull() ?: Long.MAX_VALUE
            
            val bytesPerElement = 4 // Assume F32 for throughput calculation
            val throughput = if (avgTime > 0) {
                (dataSize * bytesPerElement.toDouble() / (avgTime / 1000.0)) / (1024 * 1024)
            } else 0.0
            
            val opsPerSecond = if (avgTime > 0) 1000.0 / avgTime else 0.0
            
            return BenchmarkResult(
                operationName = operationName,
                dataSize = dataSize,
                iterations = validTimes.size,
                avgTimeMillis = avgTime,
                minTimeMillis = minTime,
                maxTimeMillis = maxTime,
                throughputMBps = throughput,
                operationsPerSecond = opsPerSecond
            )
        }
    }
    
    /**
     * Test validation utilities
     */
    object ValidationUtils {
        
        /**
         * Assert arrays are equal within tolerance
         */
        fun assertArraysEqual(
            expected: FloatArray,
            actual: FloatArray,
            tolerance: Float,
            message: String = "Arrays should be equal within tolerance"
        ) {
            require(expected.size == actual.size) { "Array sizes must match: expected ${expected.size}, actual ${actual.size}" }
            
            for (i in expected.indices) {
                val diff = abs(expected[i] - actual[i])
                if (diff > tolerance) {
                    throw AssertionError("$message - Index $i: expected ${expected[i]}, actual ${actual[i]}, diff $diff > tolerance $tolerance")
                }
            }
        }
        
        /**
         * Assert tensor dimensions match
         */
        fun assertDimensionsMatch(tensor1: GGMLTensor, tensor2: GGMLTensor, message: String = "Tensor dimensions should match") {
            if (!tensor1.ne.contentEquals(tensor2.ne)) {
                throw AssertionError("$message - Expected: ${tensor1.ne.contentToString()}, Actual: ${tensor2.ne.contentToString()}")
            }
        }
        
        /**
         * Assert performance is within bounds
         */
        fun assertPerformanceInBounds(
            result: PerformanceUtils.BenchmarkResult,
            maxTimeMillis: Long,
            minThroughputMBps: Double = 0.0,
            message: String = "Performance should be within bounds"
        ) {
            if (result.avgTimeMillis > maxTimeMillis) {
                throw AssertionError("$message - Time ${result.avgTimeMillis}ms exceeds limit ${maxTimeMillis}ms")
            }
            
            if (result.throughputMBps < minThroughputMBps) {
                throw AssertionError("$message - Throughput ${result.throughputMBps} MB/s below minimum ${minThroughputMBps} MB/s")
            }
        }
    }
}