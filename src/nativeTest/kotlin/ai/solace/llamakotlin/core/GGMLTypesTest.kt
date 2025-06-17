package ai.solace.llamakotlin.core

import kotlin.test.*
import kotlin.Short.Companion.SIZE_BYTES as SHORT_SIZE_BYTES // For F16

class GGMLTypesTest {

    private lateinit var graphAllocator: GGMLGraphAllocator
    private lateinit var testBuffer: ByteArray
    private val testBufferSize = 1024 * 1024 // 1MB

    @BeforeTest
    fun setup() {
        graphAllocator = GGMLGraphAllocator()
        testBuffer = ByteArray(testBufferSize)
        // Directly assign the test buffer to the graph allocator's primary buffer
        // and reset the corresponding tensor allocator with the test buffer's size.
        if (graphAllocator.buffers.isNotEmpty()) {
            graphAllocator.buffers[0] = testBuffer
        } else {
            graphAllocator.buffers.add(testBuffer) // Should not happen if constructor works
        }
        if (graphAllocator.tensorAllocators.isNotEmpty()) {
            graphAllocator.tensorAllocators[0].reset(testBuffer.size.toULong())
        } else {
            // This case implies GGMLGraphAllocator constructor didn't set up tensorAllocators[0]
            // which would be a problem. For this test setup, we assume it exists.
            val dynTensorAllocator = GGMLDynTensorAllocator(bufferSize = testBuffer.size.toULong())
            graphAllocator.tensorAllocators.add(dynTensorAllocator)
        }
    }

    private fun calculateTensorByteSize(type: GGMLType, ne: LongArray): ULong {
        if (type.byteSize == 0uL && type != GGMLType.COUNT) {
            // This case should ideally not be hit for types we are testing direct access on (F32, F16, I32, I16)
            // For robust testing of other types later, this might need adjustment or specific handling.
            println("Warning: Calculating byte size for type ${type.name} with byteSize 0.")
            return 0uL
        }
        var totalElements = 1UL
        var nonZeroDimFound = false
        for (dimSize in ne) {
            if (dimSize > 0L) {
                totalElements *= dimSize.toULong()
                nonZeroDimFound = true
            }
        }
        if (!nonZeroDimFound && ne.isNotEmpty()) return type.byteSize // Scalar from ne=[1,1,1,1]
        if (totalElements == 0UL && ne.isNotEmpty()) return 0UL


        return totalElements * type.byteSize
    }

    private fun calculateStrides(type: GGMLType, ne: LongArray, maxDims: Int = GGML_MAX_DIMS): ULongArray {
        val nb = ULongArray(maxDims) { 0uL }
        if (type.byteSize > 0uL) {
            nb[0] = type.byteSize
            var currentNelements = 1L
            for(d in 0 until maxDims) { // Calculate ne based on actual rank from ne array
                if (ne[d] == 0L && d > 0) { // typical for dims beyond actual rank if padded with 0
                     if (d > 0 && ne[d-1] > 0) nb[d] = nb[d-1] * ne[d-1].toULong() else nb[d] = nb[0]
                     continue
                }
                 if (ne[d] == 1L && d > 0 && ne.sliceArray(0 until d).all{it <=1L} ) { // scalar-like up to this dim
                    currentNelements = 1L // reset for scalar like cases
                 }


                if (d == 0) {
                    nb[d] = type.byteSize
                    currentNelements = if(ne[d] > 0) ne[d] else 1L
                } else {
                     nb[d] = nb[d-1] * (if (ne[d-1] > 0) ne[d-1].toULong() else 1uL)
                }
            }
        }
        // Correction for ggml's nb definition where nb[0] is element size, nb[1] is stride for dim1 etc.
        // The loop above calculates strides such that nb[d] is the stride for dimension d
        // For contiguous:
        // nb[0] = type_size
        // nb[1] = ne[0] * type_size
        // nb[2] = ne[1] * ne[0] * type_size
        // nb[3] = ne[2] * ne[1] * ne[0] * type_size
        // The getElementByteOffset uses: offset += indices[d] * nb[d]
        // This means nb[d] should be the stride for dimension d.
        // My loop for nb:
        // nb[0] = type.byteSize
        // nb[1] = nb[0] * ne[0]
        // nb[2] = nb[1] * ne[1]  <-- This is correct for the getElementByteOffset interpretation
        // nb[d] = nb[d-1] * ne[d-1]
        // So the loop should be:
        if (type.byteSize > 0uL) {
            nb[0] = type.byteSize
            for (i in 1 until maxDims) {
                 nb[i] = nb[i-1] * (if (ne[i-1] > 0L) ne[i-1].toULong() else 1uL)
            }
        } else {
            nb.fill(0uL)
        }
        return nb
    }


    private fun createTestTensor(
        name: String,
        type: GGMLType,
        dims: LongArray, // Effective dimensions e.g. longArrayOf(10) for 1D, longArrayOf(10,5) for 2D
        dataOffsetInTestBuffer: ULong,
        bufferIdToSet: Int = 0 // Assuming buffer 0 is our testBuffer
    ): GGMLTensor {
        val tensor = GGMLTensor(type = type)
        tensor.name = name

        // Pad ne to GGML_MAX_DIMS
        tensor.ne = LongArray(GGML_MAX_DIMS) { 1L }
        dims.copyInto(tensor.ne, 0, 0, dims.size.coerceAtMost(GGML_MAX_DIMS))

        tensor.nb = calculateStrides(type, tensor.ne)

        tensor.bufferId = bufferIdToSet
        tensor.dataOffset = dataOffsetInTestBuffer
        tensor.data = null // Ensure it uses the shared buffer via accessors

        // Mark it in the graph allocator's map to simulate it being part of a graph
        // This is important if any tensor method internally tries to look itself up
        // (Not currently the case for get/set, but good for future-proofing tests)
        // graphAllocator.tensorUsageMap[tensor] = TensorUsageInfo(ownsMemory = true, bufferId = bufferIdToSet, dataOffset = dataOffsetInTestBuffer, calculatedSize = calculateTensorByteSize(type, tensor.ne))

        return tensor
    }

    @Test
    fun testFloatAccess1D() {
        val dims = longArrayOf(10)
        val offset = 100uL // Start this tensor at byte offset 100 in the buffer
        val tensor = createTestTensor("testF32_1D", GGMLType.F32, dims, offset)

        val testData = FloatArray(dims[0].toInt()) { it.toFloat() * 1.1f }

        // Set data
        for (i in testData.indices) {
            assertDoesNotThrow("Set F32 1D index $i") {
                tensor.setFloat(graphAllocator, testData[i], i)
            }
        }

        // Get and verify data
        for (i in testData.indices) {
            val retrieved = assertDoesNotThrow("Get F32 1D index $i") {
                tensor.getFloat(graphAllocator, i)
            }
            assertEquals(testData[i], retrieved, "F32 1D data mismatch at index $i")
        }
    }

    @Test
    fun testFloatAccess2D() {
        val dims = longArrayOf(3, 4) // e.g., 3 columns, 4 rows
        val offset = 200uL
        val tensor = createTestTensor("testF32_2D", GGMLType.F32, dims, offset)

        // Populate with test data: value = row * 10 + col
        for (row in 0 until dims[1].toInt()) {
            for (col in 0 until dims[0].toInt()) {
                val value = (row * 10 + col).toFloat()
                 assertDoesNotThrow("Set F32 2D index ($col, $row)") {
                    tensor.setFloat(graphAllocator, value, col, row)
                }
            }
        }

        // Get and verify
        for (row in 0 until dims[1].toInt()) {
            for (col in 0 until dims[0].toInt()) {
                val expectedValue = (row * 10 + col).toFloat()
                val retrieved = assertDoesNotThrow("Get F32 2D index ($col, $row)") {
                    tensor.getFloat(graphAllocator, col, row)
                }
                assertEquals(expectedValue, retrieved, "F32 2D data mismatch at ($col, $row)")
            }
        }
    }

    @Test
    fun testIntAccess1D() {
        val dims = longArrayOf(8)
        val offset = 300uL
        val tensor = createTestTensor("testI32_1D", GGMLType.I32, dims, offset)
        val testData = IntArray(dims[0].toInt()) { it * 10 }

        testData.forEachIndexed { i, value ->
            assertDoesNotThrow("Set I32 1D index $i") { tensor.setInt(graphAllocator, value, i) }
        }
        testData.forEachIndexed { i, expected ->
            val retrieved = assertDoesNotThrow("Get I32 1D index $i") { tensor.getInt(graphAllocator, i) }
            assertEquals(expected, retrieved, "I32 1D data mismatch at index $i")
        }
    }

    @Test
    fun testShortAccess1D() {
        val dims = longArrayOf(12)
        val offset = 400uL
        val tensor = createTestTensor("testI16_1D", GGMLType.I16, dims, offset)
        val testData = ShortArray(dims[0].toInt()) { (it * 5 - 20).toShort() }

        testData.forEachIndexed { i, value ->
            assertDoesNotThrow("Set I16 1D index $i") { tensor.setShort(graphAllocator, value, i) }
        }
        testData.forEachIndexed { i, expected ->
            val retrieved = assertDoesNotThrow("Get I16 1D index $i") { tensor.getShort(graphAllocator, i) }
            assertEquals(expected, retrieved, "I16 1D data mismatch at index $i")
        }
    }

    @Test
    fun testHalfAccess1D() {
        val dims = longArrayOf(10)
        val offset = 500uL
        val tensor = createTestTensor("testF16_1D", GGMLType.F16, dims, offset)

        // Test values including some that might have precision differences
        val originalFloats = FloatArray(dims[0].toInt()) {
            when(it) {
                0 -> 0.0f
                1 -> 1.0f
                2 -> -1.0f
                3 -> 65504.0f // Max F16 normal
                4 -> 0.000061035156f // Smallest F16 normal
                5 -> 0.1f
                6 -> 123.456f
                else -> (it - 5).toFloat() * 1234.5f
            }
        }

        originalFloats.forEachIndexed { i, originalFloat ->
            val f16Bits = floatToHalf(originalFloat) // Convert original to F16 bits
            val floatFromF16 = halfToFloat(f16Bits)   // Convert F16 bits back to F32 for comparison

            assertDoesNotThrow("Set F16 1D index $i with value $originalFloat (becomes $floatFromF16)") {
                 // We set using the original float, setHalf will convert it.
                tensor.setHalf(graphAllocator, originalFloat, i)
            }

            val retrievedFloat = assertDoesNotThrow("Get F16 1D index $i") {
                tensor.getHalf(graphAllocator, i) // getHalf reads F16 bits and converts to F32
            }
            // Compare the F32 representation of the stored F16 value
            assertEquals(floatFromF16, retrievedFloat, "F16 1D data mismatch at index $i. Original: $originalFloat, StoredAsF16ToF32: $floatFromF16, Retrieved: $retrievedFloat")
        }
    }

    @Test
    fun testAccessOutOfBounds_Indices() {
        val tensorF32 = createTestTensor("f32", GGMLType.F32, longArrayOf(2, 2), 0uL) // 2x2 tensor

        // Too many indices
        assertFailsWith<IllegalArgumentException>("Should fail with too many indices") {
            tensorF32.getFloat(graphAllocator, 0, 0, 0)
        }
        // Index out of bounds for ne[0]
        assertFailsWith<IllegalArgumentException>("Should fail with index out of bounds for ne[0]") {
            tensorF32.getFloat(graphAllocator, 2, 0)
        }
        // Index out of bounds for ne[1]
        assertFailsWith<IllegalArgumentException>("Should fail with index out of bounds for ne[1]") {
            tensorF32.getFloat(graphAllocator, 0, 2)
        }
        // Negative index
        assertFailsWith<IllegalArgumentException>("Should fail with negative index") {
            tensorF32.getFloat(graphAllocator, -1, 0)
        }
    }

    @Test
    fun testBufferBoundaryChecks() {
        val tensorSizeElements = 4
        val type = GGMLType.F32
        val elementByteSize = type.byteSize.toInt()
        val tensorByteSize = tensorSizeElements * elementByteSize // 4 * 4 = 16 bytes

        // Place tensor at an offset such that its end is near the buffer end
        val offset = (testBuffer.size - tensorByteSize).toULong()
        val tensor = createTestTensor("boundaryTest", type, longArrayOf(tensorSizeElements.toLong()), offset)

        // Valid access
        assertDoesNotThrow("Access last element (valid)") {
            tensor.setFloat(graphAllocator, 1.0f, tensorSizeElements - 1)
            assertEquals(1.0f, tensor.getFloat(graphAllocator, tensorSizeElements - 1))
        }

        // Invalid access: trying to read/write one element beyond the allocated tensor space
        // This should be caught by the accessor's internal check against buffer.size
        val lastValidFlatByteOffset = offset.toInt() + tensorByteSize - elementByteSize
        val firstInvalidFlatByteOffset = offset.toInt() + tensorByteSize

        assertTrue(lastValidFlatByteOffset + elementByteSize <= testBuffer.size, "Last valid part should be in buffer")
        assertTrue(firstInvalidFlatByteOffset > testBuffer.size - elementByteSize || firstInvalidFlatByteOffset >= testBuffer.size, "First invalid part should be outside or at boundary making read impossible")

        assertFailsWith<IndexOutOfBoundsException>("Should fail reading past allocated tensor end / buffer boundary") {
             // This attempts to read starting at offset + tensorByteSize, which is out of bounds for a full Float.
            testBuffer.getFloatLe((offset + tensorByteSize.toULong()).toInt())
        }
         assertFailsWith<IndexOutOfBoundsException>("Should fail writing past allocated tensor end / buffer boundary") {
            testBuffer.setFloatLe((offset + tensorByteSize.toULong()).toInt(), 2.0f)
        }

        // Test accessors for the element just past the end
        assertFailsWith<IndexOutOfBoundsException>("getFloat should fail reading past tensor's allocated region in buffer") {
            tensor.getFloat(graphAllocator, tensorSizeElements) // Accessing index 'tensorSizeElements' which is out of bounds for 0-indexed array of this size
        }
         assertFailsWith<IndexOutOfBoundsException>("setFloat should fail writing past tensor's allocated region in buffer") {
            tensor.setFloat(graphAllocator, 3.0f, tensorSizeElements)
        }
    }
}
