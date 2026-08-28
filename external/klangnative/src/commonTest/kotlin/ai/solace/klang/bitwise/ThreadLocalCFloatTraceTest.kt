package io.github.kotlinmania.llama.lang.bitwise

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Tests for [ThreadLocalCFloatTrace].
 *
 * Verifies thread-local float operation tracing works correctly.
 */
class ThreadLocalCFloatTraceTest {

    @Test
    fun testEnableAndDisable() {
        // Initially disabled
        assertFalse(ThreadLocalCFloatTrace.isEnabled())

        // Enable
        ThreadLocalCFloatTrace.enable()
        assertTrue(ThreadLocalCFloatTrace.isEnabled())

        // Disable
        ThreadLocalCFloatTrace.disable()
        assertFalse(ThreadLocalCFloatTrace.isEnabled())
    }

    @Test
    fun testLogging() {
        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.enable()

        // Log some operations
        ThreadLocalCFloatTrace.log("+", 1.0f, 2.0f, 3.0f)
        ThreadLocalCFloatTrace.log("*", 3.0f, 4.0f, 12.0f)
        ThreadLocalCFloatTrace.log("/", 12.0f, 3.0f, 4.0f)

        // Get entries
        val entries = ThreadLocalCFloatTrace.getEntries()
        assertEquals(3, entries.size)

        assertEquals("+", entries[0].operation)
        assertEquals(1.0f, entries[0].lhs)
        assertEquals(2.0f, entries[0].rhs)
        assertEquals(3.0f, entries[0].result)

        assertEquals("*", entries[1].operation)
        assertEquals(3.0f, entries[1].lhs)
        assertEquals(4.0f, entries[1].rhs)
        assertEquals(12.0f, entries[1].result)

        assertEquals("/", entries[2].operation)
        assertEquals(12.0f, entries[2].lhs)
        assertEquals(3.0f, entries[2].rhs)
        assertEquals(4.0f, entries[2].result)

        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.disable()
    }

    @Test
    fun testReset() {
        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.enable()

        // Log operations
        ThreadLocalCFloatTrace.log("+", 1.0f, 1.0f, 2.0f)
        assertEquals(1, ThreadLocalCFloatTrace.getEntries().size)

        // Reset
        ThreadLocalCFloatTrace.reset()
        assertEquals(0, ThreadLocalCFloatTrace.getEntries().size)

        // Can log again
        ThreadLocalCFloatTrace.log("-", 5.0f, 3.0f, 2.0f)
        assertEquals(1, ThreadLocalCFloatTrace.getEntries().size)

        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.disable()
    }

    @Test
    fun testWithTracing() {
        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.disable()

        val (result, traces) = ThreadLocalCFloatTrace.withTracing {
            // Log operations
            ThreadLocalCFloatTrace.log("+", 10.0f, 20.0f, 30.0f)
            ThreadLocalCFloatTrace.log("*", 5.0f, 6.0f, 30.0f)

            // Return a result
            42
        }

        // Result should be returned
        assertEquals(42, result)

        // Traces should be captured
        assertEquals(2, traces.size)
        assertEquals("+", traces[0].operation)
        assertEquals("*", traces[1].operation)

        // Tracing should be disabled and reset after block
        assertFalse(ThreadLocalCFloatTrace.isEnabled())
        assertEquals(0, ThreadLocalCFloatTrace.getEntries().size)
    }

    @Test
    fun testWithTracingException() {
        ThreadLocalCFloatTrace.reset()

        try {
            ThreadLocalCFloatTrace.withTracing {
                ThreadLocalCFloatTrace.log("+", 1.0f, 2.0f, 3.0f)
                throw IllegalStateException("test exception")
            }
        } catch (e: IllegalStateException) {
            // Expected
        }

        // Should still be disabled and reset
        assertFalse(ThreadLocalCFloatTrace.isEnabled())
        assertEquals(0, ThreadLocalCFloatTrace.getEntries().size)
    }

    @Test
    fun testGetStats() {
        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.enable()

        // Log various operations
        ThreadLocalCFloatTrace.log("+", 1.0f, 1.0f, 2.0f)
        ThreadLocalCFloatTrace.log("+", 2.0f, 2.0f, 4.0f)
        ThreadLocalCFloatTrace.log("*", 3.0f, 3.0f, 9.0f)
        ThreadLocalCFloatTrace.log("/", 9.0f, 3.0f, 3.0f)
        ThreadLocalCFloatTrace.log("+", 3.0f, 1.0f, 4.0f)

        val stats = ThreadLocalCFloatTrace.getStats()
        assertEquals(5, stats.totalOperations)
        assertTrue(stats.enabled)

        // Check operation counts
        assertEquals(3, stats.operationCounts["+"])
        assertEquals(1, stats.operationCounts["*"])
        assertEquals(1, stats.operationCounts["/"])

        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.disable()
    }

    @Test
    fun testOnlyLogsWhenEnabled() {
        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.disable()

        // Log when disabled - should not record
        ThreadLocalCFloatTrace.log("+", 1.0f, 2.0f, 3.0f)
        assertEquals(0, ThreadLocalCFloatTrace.getEntries().size)

        // Enable and log - should record
        ThreadLocalCFloatTrace.enable()
        ThreadLocalCFloatTrace.log("+", 1.0f, 2.0f, 3.0f)
        assertEquals(1, ThreadLocalCFloatTrace.getEntries().size)

        ThreadLocalCFloatTrace.reset()
        ThreadLocalCFloatTrace.disable()
    }
}
