package io.github.kotlinmania.llama.core

import kotlin.test.*
import kotlin.math.*

/**
 * Tests for the DRY principle improvements and consolidated utility functions.
 * This test file validates the newly consolidated helper functions.
 */
class GGMLDryPrincipleTest {

    @Test
    fun testConsolidatedTensorDimensionValidation() {
        val tensor = GGMLTestUtils.createStandardTestTensor(
            io.github.kotlinmania.llama.ore.GGMLType.F32,
            longArrayOf(4, 4, 1)
        )
        
        // Test dimension validation utility
        assertTrue(GGMLTestUtils.validateTensorDimensions(tensor, longArrayOf(4, 4, 1)))
        assertFalse(GGMLTestUtils.validateTensorDimensions(tensor, longArrayOf(4, 4, 2)))
        assertFalse(GGMLTestUtils.validateTensorDimensions(tensor, longArrayOf(3, 4, 1)))
    }

    @Test
    fun testConsolidatedTensorComparison() {
        val (graphAllocator, _) = GGMLTestUtils.createTestAllocator()
        
        val tensor1 = GGMLTestUtils.createStandardTestTensor(io.github.kotlinmania.llama.ore.GGMLType.F32, longArrayOf(2, 2))
        val tensor2 = GGMLTestUtils.createStandardTestTensor(io.github.kotlinmania.llama.ore.GGMLType.F32, longArrayOf(2, 2))
        val tensor3 = GGMLTestUtils.createStandardTestTensor(io.github.kotlinmania.llama.ore.GGMLType.F16, longArrayOf(2, 2))
        
        // Test structural comparison
        assertTrue(GGMLTestUtils.TensorComparison.tensorsStructurallyEqual(tensor1, tensor2))
        assertFalse(GGMLTestUtils.TensorComparison.tensorsStructurallyEqual(tensor1, tensor3))
        
        // Test find differences utility
        val diff = GGMLTestUtils.TensorComparison.findFirstDifference(tensor1, tensor3, graphAllocator)
        assertNotNull(diff)
        assertTrue(diff.contains("Types differ"))
    }

    @Test
    fun testConsolidatedQuantizationTestUtils() {
        val testData = GGMLTestUtils.QuantizationTestUtils.createQuantizationTestData(32)
        
        // Verify test data has expected patterns
        assertEquals(32, testData.size)
        assertEquals(0.0f, testData[0])  // Zero case
        assertEquals(1.0f, testData[1])  // Positive unit case
        assertEquals(-1.0f, testData[2]) // Negative unit case
        
        // Test validation utility
        val sameData = testData.copyOf()
        assertTrue(
            GGMLTestUtils.QuantizationTestUtils.validateQuantizationAccuracy(
                testData, sameData, 0.001f, "identity"
            )
        )
        
        // Test with some error
        val noisyData = FloatArray(testData.size) { i -> testData[i] + 0.5f }
        assertFalse(
            GGMLTestUtils.QuantizationTestUtils.validateQuantizationAccuracy(
                testData, noisyData, 0.1f, "noisy"
            )
        )
    }

    @Test
    fun testConsolidatedNumericConversions() {
        val floatArray = floatArrayOf(1.0f, 2.5f, -1.5f, 0.0f)
        
        // Test consolidated conversion functions
        val halfArray = io.github.kotlinmania.llama.ore.convertFloatArrayToHalf(floatArray)
        val backToFloat = io.github.kotlinmania.llama.ore.convertHalfArrayToFloat(halfArray)
        
        assertEquals(floatArray.size, halfArray.size)
        assertEquals(floatArray.size, backToFloat.size)
        
        // Test array validation utility
        assertTrue(io.github.kotlinmania.llama.ore.validateNumericArray(floatArray))
        assertFalse(
            io.github.kotlinmania.llama.ore.validateNumericArray(
                floatArrayOf(
                    Float.NaN,
                    1.0f
                )
            )
        )
        assertTrue(
            io.github.kotlinmania.llama.ore.validateNumericArray(
                floatArrayOf(
                    Float.NaN,
                    1.0f
                ), allowNaN = true
            )
        )
        
        // Test clamping utility
        val clampedArray = io.github.kotlinmania.llama.ore.clampFloatArray(floatArray, -1.0f, 2.0f)
        assertEquals(-1.0f, clampedArray[2]) // -1.5f clamped to -1.0f
        assertEquals(2.0f, clampedArray[1])  // 2.5f clamped to 2.0f
        
        // Test tolerance comparison utility
        val similarArray = floatArrayOf(1.000001f, 2.500001f, -1.500001f, 0.000001f)
        assertTrue(
            io.github.kotlinmania.llama.ore.arraysEqualWithinTolerance(
                floatArray,
                similarArray,
                1e-5f
            )
        )
        assertFalse(
            io.github.kotlinmania.llama.ore.arraysEqualWithinTolerance(
                floatArray,
                similarArray,
                1e-7f
            )
        )
    }
}
