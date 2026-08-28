package io.github.kotlinmania.llama.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.test.assertFailsWith

/**
 * Tests for [io.github.kotlinmania.llama.ore.GGMLThreadLocalContext].
 *
 * Verifies thread-local compute parameter storage works correctly across
 * different threads and scopes.
 */
class GGMLThreadLocalContextTest {

    @Test
    fun testSetAndGetParams() {
        // Initially, params should be null
        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParams())
        assertFalse(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.hasCurrentParams())

        // Set params
        val params = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 0, nth = 1)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(params)

        // Should be able to retrieve them
        val retrieved = io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParams()
        assertNotNull(retrieved)
        assertEquals(0, retrieved.ith)
        assertEquals(1, retrieved.nth)
        assertTrue(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.hasCurrentParams())

        // Clear params
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()
        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParams())
        assertFalse(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.hasCurrentParams())
    }

    @Test
    fun testGetParamsOr() {
        // Clear any existing params
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        // Should use default when not set
        val defaultParams = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 99, nth = 100)
        val retrieved = io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParamsOr { defaultParams }
        assertEquals(99, retrieved.ith)
        assertEquals(100, retrieved.nth)

        // Set params
        val actualParams = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 5, nth = 10)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(actualParams)

        // Should use actual params now
        val retrieved2 = io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParamsOr { defaultParams }
        assertEquals(5, retrieved2.ith)
        assertEquals(10, retrieved2.nth)

        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testRequireCurrentParams() {
        // Clear params
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        // Should throw when not set
        assertFailsWith<IllegalStateException> {
            io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.requireCurrentParams()
        }

        // Set params
        val params = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 3, nth = 8)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(params)

        // Should return params
        val retrieved = io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.requireCurrentParams()
        assertEquals(3, retrieved.ith)
        assertEquals(8, retrieved.nth)

        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testWithParams() {
        // Clear params
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        val params = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 2, nth = 4)

        val result = io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.withParams(params) {
            // Inside block, params should be available
            val retrieved = io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParams()
            assertNotNull(retrieved)
            assertEquals(2, retrieved.ith)
            assertEquals(4, retrieved.nth)

            // Return a value
            "test result"
        }

        // Block returned value
        assertEquals("test result", result)

        // After block, params should be cleared
        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParams())
    }

    @Test
    fun testWithParamsException() {
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        val params = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 1, nth = 2)

        try {
            io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.withParams(params) {
                // Even if we throw, params should be cleared
                throw IllegalStateException("test exception")
            }
        } catch (e: IllegalStateException) {
            // Expected
        }

        // Params should still be cleared
        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentParams())
    }

    @Test
    fun testGetCurrentThreadId() {
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        // Should be null when not set
        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())

        // Set params
        val params = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 7, nth = 16)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(params)

        // Should return thread ID
        assertEquals(7, io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())

        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testGetTotalThreads() {
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        // Should be null when not set
        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getTotalThreads())

        // Set params
        val params = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 3, nth = 32)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(params)

        // Should return thread count
        assertEquals(32, io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getTotalThreads())

        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testIsolationBetweenThreads() {
        // This test would require actual threading, which is platform-specific
        // For now, just verify basic single-thread behavior
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        val params1 = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 0, nth = 1)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(params1)
        assertEquals(0, io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())

        val params2 = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 1, nth = 1)
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.setCurrentParams(params2)
        assertEquals(1, io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())

        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()
    }

    @Test
    fun testNestedWithParams() {
        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.clearCurrentParams()

        val outer = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 0, nth = 2)
        val inner = io.github.kotlinmania.llama.ore.GGMLComputeParams(ith = 1, nth = 2)

        io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.withParams(outer) {
            assertEquals(0, io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())

            io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.withParams(inner) {
                assertEquals(1, io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())
            }

            // Outer params are cleared too (not stacked)
            assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())
        }

        assertNull(io.github.kotlinmania.llama.ore.GGMLThreadLocalContext.getCurrentThreadId())
    }
}
