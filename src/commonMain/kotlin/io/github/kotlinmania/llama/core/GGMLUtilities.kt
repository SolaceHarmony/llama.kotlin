package io.github.kotlinmania.llama.ore

import kotlin.math.round
import kotlin.Short.Companion.SIZE_BYTES

/**
 * Centralized utility functions to implement DRY principle across the GGML implementation.
 * This file consolidates commonly used helper functions to avoid code duplication.
 */
object GGMLUtilities {
    
    /**
     * Formats a double value to 2 decimal places for consistent display formatting.
     * Used in performance benchmarks, test outputs, and reporting.
     * 
     * @param x The double value to format
     * @return The formatted string with 2 decimal places
     */
    fun formatDouble(x: Double): String {
        val v = round(x * 100.0) / 100.0
        return v.toString()
    }
    
    /**
     * Formats a float value to 2 decimal places for consistent display formatting.
     * 
     * @param x The float value to format
     * @return The formatted string with 2 decimal places
     */
    fun formatFloat(x: Float): String = formatDouble(x.toDouble())
    
    /**
     * Creates a formatted speedup display string commonly used in benchmarks.
     * 
     * @param speedup The speedup ratio value
     * @return Formatted string like "2.45x"
     */
    fun formatSpeedup(speedup: Double): String = "${formatDouble(speedup)}x"
    
    /**
     * Creates a section divider line for demo and output formatting.
     * 
     * @param length The length of the divider line
     * @param char The character to use for the divider
     * @return The divider string
     */
    fun createDivider(length: Int, char: String = "-"): String = char.repeat(length)
    
    /**
     * Creates a formatted section header for demos and reports.
     * 
     * @param title The section title
     * @param emoji Optional emoji prefix
     * @param dividerLength Length of the divider line under the title
     * @return Formatted header string with divider
     */
    fun createSectionHeader(title: String, emoji: String = "", dividerLength: Int = 29): String {
        return buildString {
            appendLine("\n$emoji $title")
            appendLine(createDivider(dividerLength))
        }
    }
    
    /**
     * Creates a formatted bullet point for feature lists.
     * 
     * @param text The bullet point text
     * @param bullet The bullet character/symbol to use
     * @return Formatted bullet point string
     */
    fun createBulletPoint(text: String, bullet: String = "•"): String = "  $bullet $text"
    
    /**
     * Creates a formatted status line for completion indicators.
     * 
     * @param status The status text (e.g., "COMPLETE", "IN PROGRESS")
     * @param description The status description
     * @param statusIcon The status icon (e.g., "✅", "🔄")
     * @return Formatted status line
     */
    fun createStatusLine(status: String, description: String, statusIcon: String = "✅"): String {
        return "$statusIcon $status: $description"
    }
}

/**
 * Centralized ByteArray extension functions for little-endian operations.
 * Consolidates duplicate byte array manipulation functions used throughout the codebase.
 */
object ByteArrayExtensions {
    
    /**
     * Reads a 32-bit integer from the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to read from
     * @return The integer value
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.getIntLe(offset: Int): Int {
        if (offset + 3 >= size) throw IndexOutOfBoundsException("Not enough bytes to read an Int at offset $offset")
        return (this[offset].toInt() and 0xFF) or
                ((this[offset + 1].toInt() and 0xFF) shl 8) or
                ((this[offset + 2].toInt() and 0xFF) shl 16) or
                ((this[offset + 3].toInt() and 0xFF) shl 24)
    }
    
    /**
     * Reads a 32-bit float from the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to read from
     * @return The float value
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.getFloatLe(offset: Int): Float = Float.fromBits(this.getIntLe(offset))
    
    /**
     * Writes a 32-bit integer to the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to write to
     * @param value The integer value to write
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.setIntLe(offset: Int, value: Int) {
        if (offset + 3 >= size) throw IndexOutOfBoundsException("Not enough bytes to write an Int at offset $offset")
        this[offset] = (value and 0xFF).toByte()
        this[offset + 1] = ((value shr 8) and 0xFF).toByte()
        this[offset + 2] = ((value shr 16) and 0xFF).toByte()
        this[offset + 3] = ((value shr 24) and 0xFF).toByte()
    }
    
    /**
     * Writes a 32-bit float to the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to write to
     * @param value The float value to write
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.setFloatLe(offset: Int, value: Float) = this.setIntLe(offset, value.toRawBits())
    
    /**
     * Reads a 16-bit short from the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to read from
     * @return The short value
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.getShortLe(offset: Int): Short {
        if (offset + 1 >= size) throw IndexOutOfBoundsException("Not enough bytes to read a Short at offset $offset")
        return ((this[offset].toInt() and 0xFF) or
                ((this[offset + 1].toInt() and 0xFF) shl 8)).toShort()
    }
    
    /**
     * Writes a 16-bit short to the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to write to
     * @param value The short value to write
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.setShortLe(offset: Int, value: Short) {
        if (offset + 1 >= size) throw IndexOutOfBoundsException("Not enough bytes to write a Short at offset $offset")
        this[offset] = (value.toInt() and 0xFF).toByte()
        this[offset + 1] = ((value.toInt() shr 8) and 0xFF).toByte()
    }
    
    /**
     * Reads a 64-bit long from the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to read from
     * @return The long value
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.getLongLe(offset: Int): Long {
        require(offset + Long.SIZE_BYTES <= size) { "Offset $offset + ${Long.SIZE_BYTES} > size $size" }
        var result = 0L
        for (i in 0 until Long.SIZE_BYTES) {
            result = result or ((this[offset + i].toLong() and 0xFF) shl (i * 8))
        }
        return result
    }
    
    /**
     * Writes a 64-bit long to the ByteArray in little-endian format.
     * 
     * @param offset The byte offset to write to
     * @param value The long value to write
     * @throws IndexOutOfBoundsException if not enough bytes available
     */
    fun ByteArray.setLongLe(offset: Int, value: Long) {
        require(offset + Long.SIZE_BYTES <= size) { "Offset $offset + ${Long.SIZE_BYTES} > size $size" }
        for (i in 0 until Long.SIZE_BYTES) {
            this[offset + i] = ((value shr (i * 8)) and 0xFF).toByte()
        }
    }
}

/**
 * Centralized demo text utilities for consistent formatting across examples and tests.
 */
object DemoTextUtilities {
    
    /**
     * Creates a complete feature section for demo output.
     * 
     * @param sectionTitle The main section title
     * @param emoji The emoji prefix for the section
     * @param features List of feature descriptions
     * @param statusIcon The status icon to use for the main status line
     * @return Formatted section string
     */
    fun createFeatureSection(
        sectionTitle: String, 
        emoji: String, 
        features: List<String>,
        statusIcon: String = "✅"
    ): String {
        return buildString {
            append(io.github.kotlinmania.llama.ore.GGMLUtilities.createSectionHeader(sectionTitle, emoji))
            appendLine(io.github.kotlinmania.llama.ore.GGMLUtilities.createStatusLine("", "", statusIcon))
            features.forEach { feature ->
                appendLine(io.github.kotlinmania.llama.ore.GGMLUtilities.createBulletPoint(feature))
            }
        }
    }
    
    /**
     * Common feature lists for reuse across demos and documentation.
     */
    object FeatureLists {
        val kQuantizationFeatures = listOf(
            "Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K",
            "Quantization and dequantization functions", 
            "Optimized dot product routines",
            "Comprehensive accuracy tests"
        )
        
        val tensorOperationFeatures = listOf(
            "Matrix multiplication with all quantization types",
            "Element-wise operations (ADD, MUL, SUB, DIV)",
            "Activation functions (RELU, GELU, SILU)",
            "Memory-efficient in-place operations"
        )
        
        val graphOptimizationFeatures = listOf(
            "Dead code elimination",
            "Redundant operation removal",
            "Constant folding",
            "Memory optimization"
        )
        
        val backendArchitectureFeatures = listOf(
            "CPU backend with ByteArray management",
            "Flexible backend registry ready for future accelerators",
            "Multi-threading support with Kotlin coroutines"
        )
    }
}
