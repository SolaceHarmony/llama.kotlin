package io.github.kotlinmania.llama.core

import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setFloatLe
import io.github.kotlinmania.llama.ore.ByteArrayExtensions.setShortLe
import io.github.kotlinmania.llama.ore.simd.GGMLSimd
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class GGMLSimdTest {

    @Test
    fun dotF32MatchesScalar() {
        val random = Random(1234)
        val length = 256
        val strideBytes = Float.SIZE_BYTES
        val bufferA = ByteArray(length * strideBytes)
        val bufferB = ByteArray(length * strideBytes)

        val expected = FloatArray(length) { idx ->
            val value = random.nextFloat() - 0.5f
            bufferA.setFloatLe(idx * strideBytes, value)
            value
        }
        val other = FloatArray(length) { idx ->
            val value = random.nextFloat() - 0.5f
            bufferB.setFloatLe(idx * strideBytes, value)
            value
        }

        val simdResult = io.github.kotlinmania.llama.ore.simd.GGMLSimd.dotF32(bufferA, 0, strideBytes, bufferB, 0, strideBytes, length)
        val expectedResult = expected.indices.fold(0.0f) { acc, idx -> acc + expected[idx] * other[idx] }

        assertTrue(kotlin.math.abs(simdResult - expectedResult) < 1e-4f, "SIMD F32 dot deviates from scalar reference")
    }

    @Test
    fun dotF16MatchesScalar() {
        val random = Random(42)
        val length = 256
        val strideBytes = Short.SIZE_BYTES
        val bufferA = ByteArray(length * strideBytes)
        val bufferB = ByteArray(length * strideBytes)

        val expected = FloatArray(length) { idx ->
            val value = (random.nextFloat() - 0.5f)
            bufferA.setShortLe(idx * strideBytes,
                io.github.kotlinmania.llama.ore.floatToHalf(value)
            )
            value
        }
        val other = FloatArray(length) { idx ->
            val value = (random.nextFloat() - 0.5f)
            bufferB.setShortLe(idx * strideBytes,
                io.github.kotlinmania.llama.ore.floatToHalf(value)
            )
            value
        }

        val simdResult = io.github.kotlinmania.llama.ore.simd.GGMLSimd.dotF16(bufferA, 0, strideBytes, bufferB, 0, strideBytes, length)
        val expectedResult = expected.indices.fold(0.0f) { acc, idx -> acc + expected[idx] * other[idx] }

        assertTrue(kotlin.math.abs(simdResult - expectedResult) < 5e-3f, "SIMD F16 dot deviates from scalar reference")
    }
}
