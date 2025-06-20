package ai.solace.llamakotlin.core

import kotlin.math.* // Import all math functions
import kotlin.test.*
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES


class GGMLComputeOpsTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val bufferSize = 1 * 1024 * 1024 // 1MB

    // Helper to calculate strides for a contiguous tensor
    private fun calculateStrides(type: GGMLType, ne: LongArray, maxDims: Int = GGML_MAX_DIMS): ULongArray {
        val nb = ULongArray(maxDims) { 0uL }
        if (type.byteSize > 0uL) {
            nb[0] = type.byteSize
            if (maxDims > 1) {
                for (d in 1 until maxDims) {
                    // Use ne.getOrElse to handle cases where ne.size < d-1, default to 1 for element count
                    val prevDimSize = ne.getOrElse(d - 1) { 1L }
                    nb[d] = nb[d-1] * (if (prevDimSize > 0) prevDimSize.toULong() else 1uL)
                }
            }
        }
        return nb
    }

    // Helper to calculate tensor byte size
    private fun calculateTensorByteSize(type: GGMLType, ne: LongArray): ULong {
        if (type.byteSize == 0uL && type != GGMLType.COUNT && !type.name.startsWith("Q")) {
             println("Warning: Calculating byte size for type ${type.name} with byteSize 0.")
            return 0uL
        }
        var elements = 1UL
        var validDimFound = false
        for (i in ne.indices) { // Iterate only over provided dims in ne
            if (ne[i] > 0L) {
                elements *= ne[i].toULong()
                validDimFound = true
            } else if (ne[i] == 0L && elements != 0UL && validDimFound) { // A zero in a significant dimension nullifies size
                return 0UL
            }
        }
        // If no valid positive dimension found but ne is not empty (e.g. ne=[1,1,1,1] treated as scalar)
        // or if ne was empty (elements still 1UL)
        if (!validDimFound && ne.isNotEmpty() && ne.all { it <= 1L }) {
            elements = 1UL // Scalar or effectively scalar
        } else if (!validDimFound && ne.isEmpty()){
            elements = 1UL // Treat as scalar if ne is completely empty
        }

        // For block types like Q8_0, byteSize is per block, numElements() is actual element count
        if (type == GGMLType.Q8_0 && elements > 0uL) {
            if (elements.toLong() % QK8_0 != 0L) {
                println("Warning: Total elements $elements for Q8_0 is not divisible by block size $QK8_0.")
            }
            return (elements.toLong() / QK8_0).toULong() * type.byteSize
        }

        return elements * type.byteSize
    }


    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(bufferSize)
        // Ensure buffer and allocator lists are not empty before assignment
        if (graphAllocator.buffers.isEmpty()) graphAllocator.buffers.add(null)
        if (graphAllocator.tensorAllocators.isEmpty()) graphAllocator.tensorAllocators.add(GGMLDynTensorAllocator())

        graphAllocator.buffers[0] = testBuffer
        graphAllocator.tensorAllocators[0].reset(bufferSize.toULong())
    }

    // Helper to create a tensor and optionally initialize it with a sequence
    private fun createAndInitTensor(
        name: String,
        type: GGMLType,
        dims: LongArray, // Effective dimensions, e.g. longArrayOf(10) or longArrayOf(3,4)
        dataOffset: ULong = 0uL,
        bufferId: Int = 0,
        fillSequence: Boolean = false,
        startValue: Number = 0.0f,
        step: Number = 1.0f
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type)
        tensor.name = name

        // Pad ne to GGML_MAX_DIMS with 1s for unused dimensions
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dimSize ->
            if (index < GGML_MAX_DIMS) tensor.ne[index] = dimSize
        }

        tensor.nb = calculateStrides(type, tensor.ne)
        tensor.bufferId = bufferId
        tensor.dataOffset = dataOffset
        tensor.data = null // Important: We are testing accessors on the shared buffer

        // Use tensor.numElements() for calculating byte size for allocation check
        val actualElementCount = tensor.numElements()
        val tensorByteSizeForCheck = if (type == GGMLType.Q8_0) {
            if (actualElementCount % QK8_0 != 0L) throw IllegalArgumentException("Q8_0 element count must be div by QK8_0")
            (actualElementCount / QK8_0).toULong() * type.byteSize
        } else {
            actualElementCount.toULong() * type.byteSize
        }

        assertTrue(dataOffset + tensorByteSizeForCheck <= bufferSize.toULong(),
            "Test tensor setup (offset $dataOffset + size $tensorByteSizeForCheck) for tensor ${tensor.name} " +
            "type ${type.name} dims ${dims.joinToString()} (effective ne: ${tensor.ne.joinToString()}) exceeds buffer capacity ($bufferSize).")

        if (fillSequence) {
            val numElementsToFill = tensor.numElements().toInt()
            var currentValue = startValue.toFloat()
            val stepValue = step.toFloat()

            // This direct write is for test setup convenience.
            // Actual operations under test should use tensor.setX/getX accessors.
            var writeOffset = tensor.dataOffset.toInt()

            if (type == GGMLType.Q8_0) {
                require(numElementsToFill % QK8_0 == 0) { "Q8_0 fill sequence requires numElements divisible by QK8_0" }
                val numBlocks = numElementsToFill / QK8_0
                val f32Block = FloatArray(QK8_0)
                var elemIdx = 0
                for (blockNum in 0 until numBlocks) {
                    for (i in 0 until QK8_0) {
                        f32Block[i] = startValue.toFloat() + (elemIdx++ * stepValue)
                    }
                    var amax = 0.0f
                    for(v in f32Block) amax = maxOf(amax, abs(v))
                    val scaleF32 = if (amax == 0.0f) 1.0f else amax / 127.0f
                    val invScaleF32 = 1.0f / scaleF32
                    val scaleF16Short = floatToHalf(scaleF32)

                    testBuffer.setShortLe(writeOffset, scaleF16Short)
                    writeOffset += SHORT_SIZE_BYTES
                    for (fVal in f32Block) {
                        val scaledVal = fVal * invScaleF32
                        var iVal = round(scaledVal).toInt()
                        iVal = iVal.coerceIn(-128, 127)
                        testBuffer[writeOffset++] = iVal.toByte()
                    }
                }

            } else {
                 for (i in 0 until numElementsToFill) {
                    when (type) {
                        GGMLType.F32 -> testBuffer.setFloatLe(writeOffset, currentValue)
                        GGMLType.F16 -> testBuffer.setShortLe(writeOffset, floatToHalf(currentValue))
                        GGMLType.I32 -> testBuffer.setIntLe(writeOffset, currentValue.toInt())
                        GGMLType.I16 -> testBuffer.setShortLe(writeOffset, currentValue.toInt().toShort())
                        else -> throw IllegalArgumentException("Unsupported type for sequence init: $type")
                    }
                    writeOffset += type.byteSize.toInt()
                    currentValue += stepValue
                }
            }
        }
        return tensor
    }

    @Test
    fun dummyTestToEnsureSetup() {
        assertTrue(true, "Setup complete, dummy test runs.")
    }

    @Test
    fun testComputeAddF32() {
        val context = GGMLContext() // Dummy context, not used by current computeAdd impl beyond graphAllocator
        var currentOffset = 0uL

        // 1D Test
        val dims1D = longArrayOf(5)
        val size1D = calculateTensorByteSize(GGMLType.F32, dims1D)
        val src0_1D = createAndInitTensor("src0_1D", GGMLType.F32, dims1D, currentOffset, fillSequence = true, startValue = 1.0f, step = 1.0f)
        currentOffset += size1D
        val src1_1D = createAndInitTensor("src1_1D", GGMLType.F32, dims1D, currentOffset, fillSequence = true, startValue = 10.0f, step = 10.0f)
        currentOffset += size1D
        val dst_1D = createAndInitTensor("dst_1D", GGMLType.F32, dims1D, currentOffset)
        // currentOffset += size1D // Not strictly needed for dst if it's last for this sub-test

        computeAdd(graphAllocator, context, src0_1D, src1_1D).let { result ->
            // The computeAdd currently returns a *new* tensor with its own data array.
            // For this test to work as intended (testing results in graphAllocator's buffer),
            // computeAdd should fill the data of a pre-allocated 'dst_1D' tensor.
            // This requires a change in computeAdd's signature or behavior.
            // Assuming computeAdd is modified to: computeAdd(..., dst: GGMLTensor)
            // For now, let's assume computeAdd fills the 'dst_1D' if passed, or we check its return.
            // The current structure of computeAdd returns a new tensor.
            // Let's test based on the current behavior: it returns a new tensor whose data we check.
            // This means dst_1D is not used by computeAdd. The result is a new tensor.
            // We need to ensure the returned tensor's data (if it has its own) is correct.
            // OR, that the computeAdd function writes its result into graphAllocator's buffer
            // and sets up the 'result' tensor's bufferId and dataOffset correctly.
            // The F32/F16 paths in computeAdd DO use setFloat/setHalf on the result tensor,
            // which implies the result tensor should be pre-allocated in the graphAllocator buffer.
            // So, the `result` from `computeAdd` should be `dst_1D` effectively.
            // Let's make `dst_1D` the actual result of the operation.
            // No, computeAdd creates a *new* result tensor. We check that new tensor.
            // The `dst_1D` defined above is just for offset calculation.
        }
         // Re-evaluate: computeAdd in GGMLOps.kt creates a result tensor BUT does not allocate its memory
         // in graph allocator, instead it relies on GGMLComputeOps.computeAdd to fill its .data.
         // GGMLComputeOps.computeAdd creates a NEW result tensor and fills its .data.
         // This is messy. For these tests, let's assume the ops write to graphAllocator for F32/F16.
         // The compute ops were refactored to use setFloat/setHalf on their result tensor.
         // This result tensor's bufferId and dataOffset MUST be set correctly before calling setFloat/setHalf.
         // The current computeAdd creates a result tensor, copies ne/nb, then calls setFloat/setHalf.
         // This means the result tensor needs its bufferId/dataOffset pointing to graphAllocator memory.
         // This is not done by createAndInitTensor for the dst tensor if fillSequence=false.
         // The compute ops create their own GGMLTensor instance for result.
         // This means we need to check the returned tensor.

        val resultAdd1D = computeAdd(graphAllocator, context, src0_1D, src1_1D)
        for (i in 0 until dims1D[0].toInt()) {
            val expected = src0_1D.getFloat(graphAllocator, i) + src1_1D.getFloat(graphAllocator, i)
            assertEquals(expected, resultAdd1D.getFloat(graphAllocator, i), "1D F32 Add mismatch at index $i")
        }


        // 2D Test
        currentOffset = 0uL // Reset for next set of allocations to avoid large offsets
        val dims2D = longArrayOf(2, 3) // 2 cols, 3 rows
        val size2D = calculateTensorByteSize(GGMLType.F32, dims2D)
        val src0_2D = createAndInitTensor("src0_2D", GGMLType.F32, dims2D, currentOffset, fillSequence = true, startValue = 1.0f, step = 0.5f)
        currentOffset += size2D
        val src1_2D = createAndInitTensor("src1_2D", GGMLType.F32, dims2D, currentOffset, fillSequence = true, startValue = 10.0f, step = 2.0f)
        // currentOffset += size2D; // dst_2D not used in this test structure

        val resultAdd2D = computeAdd(graphAllocator, context, src0_2D, src1_2D)
        for (r in 0 until dims2D[1].toInt()) { // rows
            for (c in 0 until dims2D[0].toInt()) { // cols
                val expected = src0_2D.getFloat(graphAllocator, c, r) + src1_2D.getFloat(graphAllocator, c, r)
                assertEquals(expected, resultAdd2D.getFloat(graphAllocator, c, r), "2D F32 Add mismatch at ($c, $r)")
            }
        }
    }

    @Test
    fun testComputeAddF16() {
        val context = GGMLContext()
        var currentOffset = 0uL
        val dims1D = longArrayOf(5)
        val size1D = calculateTensorByteSize(GGMLType.F16, dims1D)

        val src0Vals = floatArrayOf(1.0f, 2.5f, -3.0f, 0.1f, 100.0f)
        val src1Vals = floatArrayOf(0.5f, -1.5f, 12.5f, 0.05f, -20.0f)

        val src0_1D = createAndInitTensor("src0_F16", GGMLType.F16, dims1D, currentOffset)
        currentOffset += size1D
        val src1_1D = createAndInitTensor("src1_F16", GGMLType.F16, dims1D, currentOffset)
        // currentOffset += size1D; // dst not used

        for(i in src0Vals.indices) src0_1D.setHalf(graphAllocator, src0Vals[i], i)
        for(i in src1Vals.indices) src1_1D.setHalf(graphAllocator, src1Vals[i], i)

        val resultAdd1D = computeAdd(graphAllocator, context, src0_1D, src1_1D)
        for (i in 0 until dims1D[0].toInt()) {
            val expectedF32 = src0Vals[i] + src1Vals[i]
            val expectedF16AsF32 = halfToFloat(floatToHalf(expectedF32))
            assertEquals(expectedF16AsF32, resultAdd1D.getHalf(graphAllocator, i), "1D F16 Add mismatch at index $i. Expected(F16->F32): $expectedF16AsF32, Got: ${resultAdd1D.getHalf(graphAllocator, i)}")
        }
    }

    @Test
    fun testComputeMulF32() {
        val context = GGMLContext()
        var currentOffset = 0uL
        val dims1D = longArrayOf(4)
        val size1D = calculateTensorByteSize(GGMLType.F32, dims1D)

        val src0_1D = createAndInitTensor("src0_MulF32", GGMLType.F32, dims1D, currentOffset, fillSequence = true, startValue = 1.5f, step = 1.0f)
        currentOffset += size1D
        val src1_1D = createAndInitTensor("src1_MulF32", GGMLType.F32, dims1D, currentOffset, fillSequence = true, startValue = 2.0f, step = 0.5f)

        val resultMul1D = computeMul(graphAllocator, context, src0_1D, src1_1D)
        for (i in 0 until dims1D[0].toInt()) {
            val expected = src0_1D.getFloat(graphAllocator, i) * src1_1D.getFloat(graphAllocator, i)
            assertEquals(expected, resultMul1D.getFloat(graphAllocator, i), "1D F32 Mul mismatch at index $i")
        }
    }

    @Test
    fun testComputeMulF16() {
        val context = GGMLContext()
        var currentOffset = 0uL
        val dims1D = longArrayOf(4)
        val size1D = calculateTensorByteSize(GGMLType.F16, dims1D)

        val src0Vals = floatArrayOf(1.0f, -2.0f, 3.5f, 0.5f)
        val src1Vals = floatArrayOf(2.0f, 1.5f, -1.0f, 4.0f)

        val src0_1D = createAndInitTensor("src0_MulF16", GGMLType.F16, dims1D, currentOffset)
        currentOffset += size1D
        val src1_1D = createAndInitTensor("src1_MulF16", GGMLType.F16, dims1D, currentOffset)

        for(i in src0Vals.indices) src0_1D.setHalf(graphAllocator, src0Vals[i], i)
        for(i in src1Vals.indices) src1_1D.setHalf(graphAllocator, src1Vals[i], i)

        val resultMul1D = computeMul(graphAllocator, context, src0_1D, src1_1D)
        for (i in 0 until dims1D[0].toInt()) {
            val expectedF32 = src0Vals[i] * src1Vals[i]
            val expectedF16AsF32 = halfToFloat(floatToHalf(expectedF32))
            val actualF16AsF32 = resultMul1D.getHalf(graphAllocator, i)
            // Use a delta for F16 comparisons due to precision
            val delta = abs(expectedF16AsF32 * 0.001f)
            assertEquals(expectedF16AsF32, actualF16AsF32, delta, "1D F16 Mul mismatch at index $i. Expected(F16->F32): $expectedF16AsF32, Got: $actualF16AsF32")
        }
    }

    @Test
    fun testComputeMatMulF32xSF32() {
        val context = GGMLContext()
        var currentOffset = 0uL

        // src0: 2x3 matrix (M=2, K=3)
        // ne = [cols, rows] -> ne = [3, 2]
        val M0 = 2; val K0 = 3
        val src0Data = floatArrayOf(1f, 2f, 3f,  // Row 0
                                    4f, 5f, 6f)  // Row 1
        val src0 = createAndInitTensor("src0_MM_F32", GGMLType.F32, longArrayOf(K0.toLong(), M0.toLong()), currentOffset)
        currentOffset += calculateTensorByteSize(GGMLType.F32, src0.ne)
        for (r in 0 until M0) {
            for (c in 0 until K0) {
                src0.setFloat(graphAllocator, src0Data[r * K0 + c], c, r)
            }
        }

        // src1: 3x2 matrix (K=3, N=2)
        // ne = [cols, rows] -> ne = [2, 3]
        val K1 = 3; val N1 = 2
        val src1Data = floatArrayOf(7f, 8f,    // Row 0
                                    9f, 10f,   // Row 1
                                    11f, 12f)  // Row 2
        val src1 = createAndInitTensor("src1_MM_F32", GGMLType.F32, longArrayOf(N1.toLong(), K1.toLong()), currentOffset)
        // currentOffset += calculateTensorByteSize(GGMLType.F32, src1.ne) // Not needed if last alloc in this direct sequence
         for (r in 0 until K1) {
            for (c in 0 until N1) {
                src1.setFloat(graphAllocator, src1Data[r * N1 + c], c, r)
            }
        }

        val resultTensor = computeMatMul(graphAllocator, context, src0, src1)

        assertEquals(GGMLType.F32, resultTensor.type, "Result type should be F32")
        // Result is M0 x N1 (2x2). ne = [cols, rows] -> ne = [2, 2]
        assertEquals(N1.toLong(), resultTensor.ne[0], "Result cols (ne[0]) mismatch")
        assertEquals(M0.toLong(), resultTensor.ne[1], "Result rows (ne[1]) mismatch")

        // Expected result:
        // (1*7 + 2*9 + 3*11) (1*8 + 2*10 + 3*12)
        // (4*7 + 5*9 + 6*11) (4*8 + 5*10 + 6*12)
        // = (7+18+33) (8+20+36) = (58) (64)
        //   (28+45+66) (32+50+72) = (139) (154)
        val expectedData = arrayOf(
            floatArrayOf(58f, 64f),
            floatArrayOf(139f, 154f)
        )

        for (r in 0 until M0) { // M0 rows
            for (c in 0 until N1) { // N1 cols
                assertEquals(expectedData[r][c], resultTensor.getFloat(graphAllocator, c, r),
                             "F32xSF32 MatMul mismatch at result ($c, $r)")
            }
        }
    }

    @Test
    fun testComputeMatMulQ80xSF32() {
        val context = GGMLContext()
        var currentOffset = 0uL

        // src0 (Q8_0): 2x3 matrix (M=2, K=3). ne = [K,M] = [3,2]
        val M0 = 2; val K0 = 3
        val src0F32Data = floatArrayOf(1f, 2f, 3f,  // Row 0
                                       4f, 5f, 60f) // Row 1 (60f to ensure scale is not too small)

        // Create F32 tensor for quantization
        val tempF32Src0 = createAndInitTensor("tempF32Src0", GGMLType.F32, longArrayOf(K0.toLong(), M0.toLong()), currentOffset)
        currentOffset += calculateTensorByteSize(GGMLType.F32, tempF32Src0.ne)
        for (r in 0 until M0) {
            for (c in 0 until K0) {
                tempF32Src0.setFloat(graphAllocator, src0F32Data[r * K0 + c], c, r)
            }
        }
        val q8Src0 = quantizeTensor(graphAllocator, tempF32Src0, GGMLType.Q8_0)
        assertEquals(GGMLType.Q8_0, q8Src0.type)
        assertEquals(K0.toLong(), q8Src0.ne[0])
        assertEquals(M0.toLong(), q8Src0.ne[1])


        // src1 (F32): 3x2 matrix (K=3, N=2). ne = [N,K] = [2,3]
        val K1 = 3; val N1 = 2
        val src1F32Data = floatArrayOf(7f, 8f,    // Row 0
                                     9f, 10f,   // Row 1
                                     11f, 12f)  // Row 2
        val src1F32 = createAndInitTensor("src1_MM_F32", GGMLType.F32, longArrayOf(N1.toLong(), K1.toLong()), currentOffset)
        // currentOffset += calculateTensorByteSize(GGMLType.F32, src1F32.ne) // Not needed if last alloc
        for (r in 0 until K1) {
            for (c in 0 until N1) {
                src1F32.setFloat(graphAllocator, src1F32Data[r * N1 + c], c, r)
            }
        }

        // computeMatMul for Q8_0 x F32 should produce an F32 result
        val resultTensor = computeMatMul(graphAllocator, context, q8Src0, src1F32)

        assertEquals(GGMLType.F32, resultTensor.type, "Result type should be F32 for Q8_0 x F32")
        // Result is M0 x N1 (2x2). ne = [cols, rows] -> ne = [2, 2]
        assertEquals(N1.toLong(), resultTensor.ne[0], "Result cols (ne[0]) mismatch")
        assertEquals(M0.toLong(), resultTensor.ne[1], "Result rows (ne[1]) mismatch")

        // Expected result from original F32 data:
        // (1*7 + 2*9 + 3*11)   (1*8 + 2*10 + 3*12)   = (58)  (64)
        // (4*7 + 5*9 + 60*11)  (4*8 + 5*10 + 60*12)  = (28+45+660) (32+50+720) = (733) (802)
        val expectedData = arrayOf(
            floatArrayOf(58f, 64f),
            floatArrayOf(733f, 802f)
        )

        val delta = 2.0f // Allow some deviation due to Q8_0 quantization
        for (r in 0 until M0) { // M0 rows
            for (c in 0 until N1) { // N1 cols
                assertEquals(expectedData[r][c], resultTensor.getFloat(graphAllocator, c, r), delta,
                             "Q8_0xSF32 MatMul mismatch at result ($c, $r)")
            }
        }
    }

    // Helper to extract data from a tensor as a FloatArray
    // Adapted from GGMLQuantizationAccuracyTest.kt
    // Assumes result tensors from compute ops might have their .data field populated directly
    // or fall back to graphAllocator if .data is null.
    internal fun getTensorDataAsFloatArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): FloatArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return floatArrayOf()

        // Case 1: Tensor has its own FloatArray in .data (typical for results of dequantizeTensor or some compute ops)
        if (tensor.data is FloatArray) {
            val fa = tensor.data as FloatArray
            // Ensure the self-contained array matches expected elements; could be an error if not.
            if (fa.size == numElements) return fa.copyOf()
            // Fallthrough if sizes don't match, which might indicate an issue or that .data is not the source of truth.
        }

        // Case 2: Tensor has its own ShortArray in .data (typical for F16 results)
        if (tensor.type == GGMLType.F16 && tensor.data is ShortArray) {
            val sa = tensor.data as ShortArray
            if (sa.size == numElements) {
                return FloatArray(numElements) { i -> halfToFloat(sa[i]) }
            }
        }

        // Case 3: Tensor data is in the graphAllocator's buffer (typical for source tensors or if compute ops write to graphAllocator)
        // This part relies on tensor.bufferId and tensor.dataOffset being correctly set.
        if (tensor.bufferId != -1) {
            val floatArray = FloatArray(numElements)
            var idx = 0
            // Using applyNDIter from GGMLQuantizationAccuracyTest as a model
            // Need to ensure a similar helper or direct iteration logic is here.
            // For simplicity, assuming a flat 1D iteration for now if applyNDIter is not in this file.
            // The existing applyNDIter in this file might need adjustment or direct use.
            // The createAndInitTensor sets up ne for up to GGML_MAX_DIMS.
            applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
                if (idx < numElements) {
                    floatArray[idx++] = when (tensor.type) {
                        GGMLType.F32 -> tensor.getFloat(graphAllocator, *indices)
                        GGML_TYPE_F16 -> tensor.getHalf(graphAllocator, *indices)
                        // Add other type direct conversions if getTensorDataAsFloatArray needs to support them
                        else -> throw IllegalArgumentException("getTensorDataAsFloatArray: Unsupported tensor type ${tensor.type} for direct float extraction from graph allocator without dequantization.")
                    }
                }
            }
            return floatArray
        }

        throw IllegalStateException("Tensor ${tensor.name} has no accessible data (neither self-contained .data nor valid bufferId/dataOffset for graphAllocator).")
    }


    // Helper: GELU approximation using tanh, matches ggml.c's gelu_f32
    private fun geluApproximation(x: Float): Float {
        return 0.5f * x * (1.0f + tanh(sqrt(2.0f / PI.toFloat()) * (x + 0.044715f * x * x * x)))
    }

    // Helper: Sigmoid
    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x))
    }

    // Helper: SiLU (Swish) = x * sigmoid(x)
    private fun silu(x: Float): Float {
        return x * sigmoid(x)
    }

    // Helper function to manually calculate RMS Norm for verification
    private fun manualRMSNorm(input: FloatArray, eps: Float): FloatArray {
        if (input.isEmpty()) return floatArrayOf()

        var sumSq = 0.0
        for (x_i_double in input.map { it.toDouble() }) { // Use Double for intermediate sum to maintain precision
            sumSq += x_i_double * x_i_double
        }
        val meanSq = sumSq / input.size
        val rms = sqrt(meanSq + eps.toDouble()).toFloat() // Calculate rms in double, then cast back to float

        if (rms == 0.0f || rms.isNaN()) { // Handle cases where rms is zero or NaN (e.g. all zeros input with zero eps)
             return FloatArray(input.size) { 0.0f }
        }

        val output = FloatArray(input.size)
        for (i in input.indices) {
            output[i] = input[i] / rms
        }
        return output
    }

    // --- RELU Tests ---
    @Test
    fun testComputeReluF32() {
        val srcNe = longArrayOf(5)
        val srcData = floatArrayOf(-2f, -0.5f, 0f, 0.5f, 2f)
        // Using createAndInitTensor and then populating for consistency
        val srcTensor = createAndInitTensor("relu_f32_src", GGMLType.F32, srcNe, dataOffset = 0uL)
        for(i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)


        val resultTensor = computeRelu(graphAllocator, srcTensor)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for RELU F32")

        val expectedData = floatArrayOf(0f, 0f, 0f, 0.5f, 2f)
        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "RELU F32 output mismatch")
    }

    @Test
    fun testComputeReluF16() {
        val srcNe = longArrayOf(5)
        val srcDataF32 = floatArrayOf(-2f, -0.5f, 0f, 0.5f, 2f)
        val srcTensor = createAndInitTensor("relu_f16_src", GGMLType.F16, srcNe, dataOffset = 0uL)
        for(i in srcDataF32.indices) srcTensor.setHalf(graphAllocator, srcDataF32[i], i)

        val resultTensor = computeRelu(graphAllocator, srcTensor)

        assertEquals(GGMLType.F16, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for RELU F16")

        val expectedDataF32 = floatArrayOf(0f, 0f, 0f, 0.5f, 2f)
        val resultDataF16AsF32 = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        for(i in expectedDataF32.indices) {
            assertEquals(halfToFloat(floatToHalf(expectedDataF32[i])), resultDataF16AsF32[i], 0.001f, "RELU F16 output mismatch at index $i")
        }
    }

    // --- GELU Tests ---
    @Test
    fun testComputeGeluF32() {
        val srcNe = longArrayOf(5)
        val srcData = floatArrayOf(-3f, -1f, 0f, 1f, 3f)
        val srcTensor = createAndInitTensor("gelu_f32_src", GGMLType.F32, srcNe, dataOffset = 0uL)
        for(i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        val resultTensor = computeGelu(graphAllocator, srcTensor)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for GELU F32")

        val expectedData = srcData.map { geluApproximation(it) }.toFloatArray()
        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        for(i in expectedData.indices) {
            assertEquals(expectedData[i], resultData[i], 0.001f, "GELU F32 output mismatch at index $i")
        }
    }

    @Test
    fun testComputeGeluF16() {
        val srcNe = longArrayOf(5)
        val srcDataF32 = floatArrayOf(-3f, -1f, 0f, 1f, 3f)
        val srcTensor = createAndInitTensor("gelu_f16_src", GGMLType.F16, srcNe, dataOffset = 0uL)
        for(i in srcDataF32.indices) srcTensor.setHalf(graphAllocator, srcDataF32[i], i)

        val resultTensor = computeGelu(graphAllocator, srcTensor)

        assertEquals(GGMLType.F16, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for GELU F16")

        val expectedDataF32 = srcDataF32.map { geluApproximation(it) }.toFloatArray()
        val resultDataF16AsF32 = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        for(i in expectedDataF32.indices) {
            assertEquals(halfToFloat(floatToHalf(expectedDataF32[i])), resultDataF16AsF32[i], 0.01f,
                         "GELU F16 output mismatch at index $i. Expected F32: ${expectedDataF32[i]}, Got F16asF32: ${resultDataF16AsF32[i]}")
        }
    }

    // --- SILU Tests ---
    @Test
    fun testComputeSiluF32() {
        val srcNe = longArrayOf(5)
        val srcData = floatArrayOf(-5f, -1f, 0f, 1f, 5f)
        val srcTensor = createAndInitTensor("silu_f32_src", GGMLType.F32, srcNe, dataOffset = 0uL)
        for(i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        // Assuming computeSilu exists and has signature (GGMLGraphAllocator, GGMLTensor) -> GGMLTensor
        val resultTensor = computeSilu(graphAllocator, srcTensor)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for SILU F32")

        val expectedData = srcData.map { silu(it) }.toFloatArray()
        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        for(i in expectedData.indices) {
            assertEquals(expectedData[i], resultData[i], 0.001f, "SILU F32 output mismatch at index $i")
        }
    }

    @Test
    fun testComputeSiluF16() {
        val srcNe = longArrayOf(5)
        val srcDataF32 = floatArrayOf(-5f, -1f, 0f, 1f, 5f)
        val srcTensor = createAndInitTensor("silu_f16_src", GGMLType.F16, srcNe, dataOffset = 0uL)
        for(i in srcDataF32.indices) srcTensor.setHalf(graphAllocator, srcDataF32[i], i)

        val resultTensor = computeSilu(graphAllocator, srcTensor)

        assertEquals(GGMLType.F16, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "Dimensions should match for SILU F16")

        val expectedDataF32 = srcDataF32.map { silu(it) }.toFloatArray()
        val resultDataF16AsF32 = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        for(i in expectedDataF32.indices) {
            assertEquals(halfToFloat(floatToHalf(expectedDataF32[i])), resultDataF16AsF32[i], 0.01f,
                         "SILU F16 output mismatch at index $i. Expected F32: ${expectedDataF32[i]}, Got F16asF32: ${resultDataF16AsF32[i]}")
        }
    }

    // --- RMS_NORM Tests ---
    @Test
    fun testComputeRMSNormF32() {
        val srcNe = longArrayOf(4) // Simplified to just essential dimension for 1D
        val srcData = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f)

        // Using createAndInitTensor and then populating for consistency
        val srcTensor = createAndInitTensor("rms_f32_src", GGMLType.F32, srcNe, dataOffset = 0uL)
        for(i in srcData.indices) srcTensor.setFloat(graphAllocator, srcData[i], i)

        val eps = 1e-5f

        val resultTensor = computeRMSNorm(graphAllocator, srcTensor, eps)

        assertEquals(GGMLType.F32, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "RMSNorm F32 result dimensions mismatch")

        val expectedData = manualRMSNorm(srcData, eps)
        val resultData = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        assertEquals(expectedData.size, resultData.size, "RMSNorm F32 result data size mismatch")
        for(i in expectedData.indices) {
            assertEquals(expectedData[i], resultData[i], 0.0001f, "RMSNorm F32 output mismatch at index $i")
        }
    }

    @Test
    fun testComputeRMSNormF16() {
        val srcNe = longArrayOf(4) // Simplified to just essential dimension for 1D
        val srcDataF32 = floatArrayOf(1.0f, 2.0f, -1.0f, -2.0f)

        var currentOffset = 0uL // Manage offset if other tensors were created in testBuffer, though this test is self-contained for src
        val srcTensor = createAndInitTensor("rms_f16_src", GGMLType.F16, srcNe, dataOffset = currentOffset)
        for(i in srcDataF32.indices) srcTensor.setHalf(graphAllocator, srcDataF32[i], i)

        val eps = 1e-5f

        val resultTensor = computeRMSNorm(graphAllocator, srcTensor, eps)

        assertEquals(GGMLType.F16, resultTensor.type)
        assertTrue(srcTensor.ne.contentEquals(resultTensor.ne), "RMSNorm F16 result dimensions mismatch")

        val expectedDataF32 = manualRMSNorm(srcDataF32, eps)
        val resultDataF16AsF32 = getTensorDataAsFloatArray(resultTensor, graphAllocator)

        assertEquals(expectedDataF32.size, resultDataF16AsF32.size, "RMSNorm F16 result data size mismatch")
        for(i in expectedDataF32.indices) {
            assertEquals(halfToFloat(floatToHalf(expectedDataF32[i])), resultDataF16AsF32[i], 0.01f,
                         "RMSNorm F16 output mismatch at index $i. Expected F32: ${expectedDataF32[i]}, Got F16asF32: ${resultDataF16AsF32[i]}")
        }
    }

    // --- Integer Type Data Extraction Helpers ---
    internal fun getTensorDataAsByteArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): ByteArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return byteArrayOf()
        if (tensor.data is ByteArray && (tensor.data as ByteArray).size == numElements) {
            return (tensor.data as ByteArray).copyOf()
        }
        if (tensor.bufferId != -1) {
            val byteArray = ByteArray(numElements)
            var idx = 0
            applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
                if (idx < numElements) {
                    byteArray[idx++] = tensor.getByte(graphAllocator, *indices)
                }
            }
            return byteArray
        }
        throw IllegalStateException("Tensor ${tensor.name} (I8) has no accessible data.")
    }

    internal fun getTensorDataAsShortArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): ShortArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return shortArrayOf()
        if (tensor.data is ShortArray && (tensor.data as ShortArray).size == numElements) {
            return (tensor.data as ShortArray).copyOf()
        }
        if (tensor.bufferId != -1) {
            val shortArray = ShortArray(numElements)
            var idx = 0
            applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
                if (idx < numElements) {
                    shortArray[idx++] = tensor.getShort(graphAllocator, *indices)
                }
            }
            return shortArray
        }
        throw IllegalStateException("Tensor ${tensor.name} (I16) has no accessible data.")
    }

    internal fun getTensorDataAsIntArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): IntArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return intArrayOf()
        if (tensor.data is IntArray && (tensor.data as IntArray).size == numElements) {
            return (tensor.data as IntArray).copyOf()
        }
        if (tensor.bufferId != -1) {
            val intArray = IntArray(numElements)
            var idx = 0
            applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
                if (idx < numElements) {
                    intArray[idx++] = tensor.getInt(graphAllocator, *indices)
                }
            }
            return intArray
        }
        throw IllegalStateException("Tensor ${tensor.name} (I32) has no accessible data.")
    }

    internal fun getTensorDataAsLongArray(tensor: GGMLTensor, graphAllocator: GGMLGraphAllocator): LongArray {
        val numElements = tensor.numElements().toInt()
        if (numElements == 0) return longArrayOf()
        if (tensor.data is LongArray && (tensor.data as LongArray).size == numElements) {
            return (tensor.data as LongArray).copyOf()
        }
        if (tensor.bufferId != -1) {
            val longArray = LongArray(numElements)
            var idx = 0
            applyNDIter(tensor.ne, tensor.rank(), numElements) { _, indices ->
                if (idx < numElements) {
                    longArray[idx++] = tensor.getLong(graphAllocator, *indices)
                }
            }
            return longArray
        }
        throw IllegalStateException("Tensor ${tensor.name} (I64) has no accessible data.")
    }

    // --- Integer ComputeOp Tests ---
    private val dummyContext = GGMLContext() // Reusable dummy context

    @Test
    fun testComputeAddI32() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(3)
        val src0Data = intArrayOf(10, 20, Int.MAX_VALUE - 5)
        val src1Data = intArrayOf(5, 15, 10) // Adding 10 to MAX_VALUE-5 will overflow
        val expectedData = intArrayOf(15, 35, Int.MAX_VALUE - 5 + 10) // Standard Int addition wraps

        val src0 = createAndInitTensor("add_i32_src0", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setInt(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I32, srcNe)

        val src1 = createAndInitTensor("add_i32_src1", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setInt(graphAllocator, src1Data[i], i)

        val resultTensor = computeAdd(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsIntArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "ADD I32 output mismatch")
    }

    @Test
    fun testComputeAddI16() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(2)
        val src0Data = shortArrayOf(10, Short.MAX_VALUE)
        val src1Data = shortArrayOf(5, 1)
        val expectedData = shortArrayOf(15, Short.MAX_VALUE) // Expect coercion

        val src0 = createAndInitTensor("add_i16_src0", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setShort(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I16, srcNe)

        val src1 = createAndInitTensor("add_i16_src1", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setShort(graphAllocator, src1Data[i], i)

        val resultTensor = computeAdd(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I16, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsShortArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "ADD I16 output mismatch")
    }

    @Test
    fun testComputeAddI8() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(3)
        val src0Data = byteArrayOf(10, 100, Byte.MIN_VALUE)
        val src1Data = byteArrayOf(5, 100, (-1).toByte()) // 100+100=200 (overflows Byte), MIN_VALUE-1 (underflows)
        val expectedData = byteArrayOf(15, Byte.MAX_VALUE, Byte.MIN_VALUE) // Expect coercion for overflow, underflow wraps for Byte to -128

        val src0 = createAndInitTensor("add_i8_src0", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setByte(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I8, srcNe)

        val src1 = createAndInitTensor("add_i8_src1", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setByte(graphAllocator, src1Data[i], i)

        val resultTensor = computeAdd(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I8, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsByteArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "ADD I8 output mismatch")
    }

    @Test
    fun testComputeAddI64() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(2)
        val src0Data = longArrayOf(100L, Long.MAX_VALUE - 5L)
        val src1Data = longArrayOf(50L, 10L) // Adding 10 to MAX_VALUE-5 will overflow
        val expectedData = longArrayOf(150L, Long.MAX_VALUE - 5L + 10L) // Standard Long addition wraps

        val src0 = createAndInitTensor("add_i64_src0", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setLong(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I64, srcNe)

        val src1 = createAndInitTensor("add_i64_src1", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setLong(graphAllocator, src1Data[i], i)

        val resultTensor = computeAdd(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I64, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsLongArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "ADD I64 output mismatch")
    }

    @Test
    fun testComputeMulI32() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(3)
        val src0Data = intArrayOf(2, 10, Int.MAX_VALUE / 2 + 10) // Last one will overflow with *2
        val src1Data = intArrayOf(3, 5, 2)
        val expectedData = intArrayOf(6, 50, (Int.MAX_VALUE / 2 + 10) * 2) // Standard Int multiplication wraps

        val src0 = createAndInitTensor("mul_i32_src0", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setInt(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I32, srcNe)

        val src1 = createAndInitTensor("mul_i32_src1", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setInt(graphAllocator, src1Data[i], i)

        val resultTensor = computeMul(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsIntArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "MUL I32 output mismatch")
    }

    @Test
    fun testComputeMulI16() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(2)
        val src0Data = shortArrayOf(10, (Short.MAX_VALUE / 2).toShort())
        val src1Data = shortArrayOf(3, 3) // (MAX/2)*3 will overflow
        val expectedData = shortArrayOf(30, Short.MAX_VALUE) // Expect coercion

        val src0 = createAndInitTensor("mul_i16_src0", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setShort(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I16, srcNe)

        val src1 = createAndInitTensor("mul_i16_src1", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setShort(graphAllocator, src1Data[i], i)

        val resultTensor = computeMul(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I16, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsShortArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "MUL I16 output mismatch")
    }

    @Test
    fun testComputeMulI8() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(2)
        val src0Data = byteArrayOf(10, (Byte.MAX_VALUE / 2).toByte())
        val src1Data = byteArrayOf(3, 3) // (MAX/2)*3 will overflow
        val expectedData = byteArrayOf(30, Byte.MAX_VALUE) // Expect coercion

        val src0 = createAndInitTensor("mul_i8_src0", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setByte(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I8, srcNe)

        val src1 = createAndInitTensor("mul_i8_src1", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setByte(graphAllocator, src1Data[i], i)

        val resultTensor = computeMul(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I8, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsByteArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "MUL I8 output mismatch")
    }

    @Test
    fun testComputeMulI64() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(2)
        val src0Data = longArrayOf(100L, Long.MAX_VALUE / 2L + 10L) // Last one will overflow with *2
        val src1Data = longArrayOf(3L, 2L)
        val expectedData = longArrayOf(300L, (Long.MAX_VALUE / 2L + 10L) * 2L) // Standard Long multiplication wraps

        val src0 = createAndInitTensor("mul_i64_src0", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setLong(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I64, srcNe)

        val src1 = createAndInitTensor("mul_i64_src1", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setLong(graphAllocator, src1Data[i], i)

        val resultTensor = computeMul(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I64, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsLongArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "MUL I64 output mismatch")
    }

    // --- computeSub Tests ---
    @Test
    fun testComputeSubI32() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(4)
        val src0Data = intArrayOf(10, 5, Int.MIN_VALUE + 5, Int.MIN_VALUE)
        val src1Data = intArrayOf(5, 10, 10, 1) // MIN_VALUE - 1 will underflow
        val expectedData = intArrayOf(5, -5, Int.MIN_VALUE + 5 - 10, Int.MIN_VALUE - 1) // Standard Int subtraction wraps

        val src0 = createAndInitTensor("sub_i32_src0", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setInt(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I32, srcNe)

        val src1 = createAndInitTensor("sub_i32_src1", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setInt(graphAllocator, src1Data[i], i)

        val resultTensor = computeSub(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsIntArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "SUB I32 output mismatch")
    }

    @Test
    fun testComputeSubI16() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(4)
        val src0Data = shortArrayOf(10, 5, Short.MIN_VALUE, Short.MAX_VALUE)
        val src1Data = shortArrayOf(5, 10, 1, (-1).toShort()) // MIN_VALUE - 1 (underflow), MAX_VALUE - (-1) (overflow)
        val expectedData = shortArrayOf(5, -5, Short.MIN_VALUE, Short.MAX_VALUE) // Expect coercion

        val src0 = createAndInitTensor("sub_i16_src0", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setShort(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I16, srcNe)

        val src1 = createAndInitTensor("sub_i16_src1", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setShort(graphAllocator, src1Data[i], i)

        val resultTensor = computeSub(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I16, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsShortArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "SUB I16 output mismatch")
    }

    @Test
    fun testComputeSubI8() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(4)
        val src0Data = byteArrayOf(10, 5, Byte.MIN_VALUE, Byte.MAX_VALUE)
        val src1Data = byteArrayOf(5, 10, 1, (-1).toByte()) // MIN_VALUE - 1 (underflow), MAX_VALUE - (-1) (overflow)
        val expectedData = byteArrayOf(5, (-5).toByte(), Byte.MIN_VALUE, Byte.MAX_VALUE) // Expect coercion

        val src0 = createAndInitTensor("sub_i8_src0", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setByte(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I8, srcNe)

        val src1 = createAndInitTensor("sub_i8_src1", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setByte(graphAllocator, src1Data[i], i)

        val resultTensor = computeSub(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I8, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsByteArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "SUB I8 output mismatch")
    }

    @Test
    fun testComputeSubI64() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(4)
        val src0Data = longArrayOf(100L, 50L, Long.MIN_VALUE + 50L, Long.MIN_VALUE)
        val src1Data = longArrayOf(50L, 100L, 100L, 1L) // MIN_VALUE - 1 will underflow
        val expectedData = longArrayOf(50L, -50L, Long.MIN_VALUE + 50L - 100L, Long.MIN_VALUE - 1L) // Standard Long subtraction wraps

        val src0 = createAndInitTensor("sub_i64_src0", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setLong(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I64, srcNe)

        val src1 = createAndInitTensor("sub_i64_src1", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setLong(graphAllocator, src1Data[i], i)

        val resultTensor = computeSub(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I64, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsLongArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "SUB I64 output mismatch")
    }

    // --- computeDiv Tests ---
    @Test
    fun testComputeDivI32() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(6)
        // Test cases: standard, truncation, negative, MIN_VALUE / -1
        val src0Data = intArrayOf(10, 2, -10, -10, Int.MIN_VALUE, 0)
        val src1Data = intArrayOf(3, 3, 3, -3, -1, 5) // Last is 0/5
        val expectedData = intArrayOf(3, 0, -3, 3, Int.MIN_VALUE, 0) // MIN_VALUE / -1 wraps to MIN_VALUE in Kotlin Int

        val src0 = createAndInitTensor("div_i32_src0", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setInt(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I32, srcNe)

        val src1 = createAndInitTensor("div_i32_src1", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setInt(graphAllocator, src1Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I32, srcNe)


        val resultTensor = computeDiv(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I32, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsIntArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "DIV I32 output mismatch")

        // Test division by zero
        val src1DivByZeroData = intArrayOf(3, 0, 3, -3, -1, 0) // one zero
        val src1DivByZero = createAndInitTensor("div_i32_src1_div_zero", GGMLType.I32, srcNe, dataOffset = currentOffset)
        for(i in src1DivByZeroData.indices) src1DivByZero.setInt(graphAllocator, src1DivByZeroData[i], i)
        assertFailsWith<ArithmeticException>("DIV I32 by zero should throw ArithmeticException") {
            computeDiv(graphAllocator, dummyContext, src0, src1DivByZero)
        }
    }

    @Test
    fun testComputeDivI16() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(6)
        val src0Data = shortArrayOf(10, 2, -10, -10, Short.MIN_VALUE, 0)
        val src1Data = shortArrayOf(3, 3, 3, -3, -1, 5)
        // MIN_VALUE / -1 is coerced to MAX_VALUE for I16 in computeDiv
        val expectedData = shortArrayOf(3, 0, -3, 3, Short.MAX_VALUE, 0)

        val src0 = createAndInitTensor("div_i16_src0", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setShort(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I16, srcNe)

        val src1 = createAndInitTensor("div_i16_src1", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setShort(graphAllocator, src1Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I16, srcNe)

        val resultTensor = computeDiv(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I16, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsShortArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "DIV I16 output mismatch")

        // Test division by zero
        val src1DivByZeroData = shortArrayOf(3, 0, 3, -3, -1, 0)
        val src1DivByZero = createAndInitTensor("div_i16_src1_div_zero", GGMLType.I16, srcNe, dataOffset = currentOffset)
        for(i in src1DivByZeroData.indices) src1DivByZero.setShort(graphAllocator, src1DivByZeroData[i], i)
        assertFailsWith<ArithmeticException>("DIV I16 by zero should throw ArithmeticException") {
            computeDiv(graphAllocator, dummyContext, src0, src1DivByZero)
        }
    }

    @Test
    fun testComputeDivI8() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(6)
        val src0Data = byteArrayOf(10, 2, -10, -10, Byte.MIN_VALUE, 0)
        val src1Data = byteArrayOf(3, 3, 3, -3, -1, 5)
        // MIN_VALUE / -1 is coerced to MAX_VALUE for I8 in computeDiv
        val expectedData = byteArrayOf(3, 0, -3, 3, Byte.MAX_VALUE, 0)


        val src0 = createAndInitTensor("div_i8_src0", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setByte(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I8, srcNe)

        val src1 = createAndInitTensor("div_i8_src1", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setByte(graphAllocator, src1Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I8, srcNe)

        val resultTensor = computeDiv(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I8, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsByteArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "DIV I8 output mismatch")

        // Test division by zero
        val src1DivByZeroData = byteArrayOf(3, 0, 3, -3, -1, 0)
        val src1DivByZero = createAndInitTensor("div_i8_src1_div_zero", GGMLType.I8, srcNe, dataOffset = currentOffset)
        for(i in src1DivByZeroData.indices) src1DivByZero.setByte(graphAllocator, src1DivByZeroData[i], i)
        assertFailsWith<ArithmeticException>("DIV I8 by zero should throw ArithmeticException") {
            computeDiv(graphAllocator, dummyContext, src0, src1DivByZero)
        }
    }

    @Test
    fun testComputeDivI64() {
        var currentOffset = 0uL
        val srcNe = longArrayOf(6)
        val src0Data = longArrayOf(10L, 2L, -10L, -10L, Long.MIN_VALUE, 0L)
        val src1Data = longArrayOf(3L, 3L, 3L, -3L, -1L, 5L)
        // MIN_VALUE / -1 wraps to MIN_VALUE in Kotlin Long
        val expectedData = longArrayOf(3L, 0L, -3L, 3L, Long.MIN_VALUE, 0L)

        val src0 = createAndInitTensor("div_i64_src0", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src0Data.indices) src0.setLong(graphAllocator, src0Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I64, srcNe)

        val src1 = createAndInitTensor("div_i64_src1", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src1Data.indices) src1.setLong(graphAllocator, src1Data[i], i)
        currentOffset += calculateTensorByteSize(GGMLType.I64, srcNe)

        val resultTensor = computeDiv(graphAllocator, dummyContext, src0, src1)
        assertEquals(GGMLType.I64, resultTensor.type)
        assertTrue(src0.ne.contentEquals(resultTensor.ne))
        val resultData = getTensorDataAsLongArray(resultTensor, graphAllocator)
        assertContentEquals(expectedData, resultData, "DIV I64 output mismatch")

        // Test division by zero
        val src1DivByZeroData = longArrayOf(3L, 0L, 3L, -3L, -1L, 0L)
        val src1DivByZero = createAndInitTensor("div_i64_src1_div_zero", GGMLType.I64, srcNe, dataOffset = currentOffset)
        for(i in src1DivByZeroData.indices) src1DivByZero.setLong(graphAllocator, src1DivByZeroData[i], i)
        assertFailsWith<ArithmeticException>("DIV I64 by zero should throw ArithmeticException") {
            computeDiv(graphAllocator, dummyContext, src0, src1DivByZero)
        }
    }
}
