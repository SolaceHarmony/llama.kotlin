package io.github.kotlinmania.llama.lang.fp

import io.github.kotlinmania.llama.lang.bitwise.Float32Math
import io.github.kotlinmania.llama.ore.GGMLBF16
import io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32
import io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16
import io.github.kotlinmania.llama.ore.ggml_compute_bf16_to_fp32
import kotlin.math.*

// port-lint: source ggml/src/ggml-cpu/vec.h + vec.cpp

/** Deterministic floating point reduction helpers backed by the soft-float math. */
object VectorOps {
    fun dotAccumulate(length: Int, lhs: FloatArray, lhsOffset: Int = 0, rhs: FloatArray, rhsOffset: Int = 0): Float {
        require(length >= 0)
        require(lhsOffset >= 0 && lhsOffset + length <= lhs.size)
        require(rhsOffset >= 0 && rhsOffset + length <= rhs.size)
        var acc = 0.0f
        for (i in 0 until length) {
            val prod = Float32Math.mul(lhs[lhsOffset + i], rhs[rhsOffset + i])
            acc = Float32Math.add(acc, prod)
        }
        return acc
    }

    fun axpy(length: Int, a: Float, x: FloatArray, xOffset: Int, y: FloatArray, yOffset: Int, out: FloatArray, outOffset: Int) {
        require(length >= 0)
        require(xOffset >= 0 && xOffset + length <= x.size)
        require(yOffset >= 0 && yOffset + length <= y.size)
        require(outOffset >= 0 && outOffset + length <= out.size)
        for (i in 0 until length) {
            val scaled = Float32Math.mul(a, x[xOffset + i])
            out[outOffset + i] = Float32Math.add(scaled, y[yOffset + i])
        }
    }
}

// ---------------------------------------------------------------------------
// Element-wise vector operations from ggml-cpu/vec.h (scalar fallbacks)
// ---------------------------------------------------------------------------

// --- Set / Copy ---

fun ggml_vec_set_i8(n: Int, x: ByteArray, v: Byte) { for (i in 0 until n) x[i] = v }
fun ggml_vec_set_i16(n: Int, x: ShortArray, v: Short) { for (i in 0 until n) x[i] = v }
fun ggml_vec_set_i32(n: Int, x: IntArray, v: Int) { for (i in 0 until n) x[i] = v }
fun ggml_vec_cpy_i32(n: Int, y: IntArray, x: IntArray) { for (i in 0 until n) y[i] = x[i] }
fun ggml_vec_set_f16(n: Int, x: ShortArray, v: Short) { for (i in 0 until n) x[i] = v }
fun ggml_vec_set_bf16(n: Int, x: ShortArray, v: Short) { for (i in 0 until n) x[i] = v }
fun ggml_vec_set_f32(n: Int, x: FloatArray, v: Float) { for (i in 0 until n) x[i] = v }
fun ggml_vec_cpy_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = x[i] }

// --- Arithmetic (f32) ---

fun ggml_vec_add_f32(n: Int, z: FloatArray, x: FloatArray, y: FloatArray) { for (i in 0 until n) z[i] = x[i] + y[i] }
fun ggml_vec_add1_f32(n: Int, z: FloatArray, x: FloatArray, v: Float) { for (i in 0 until n) z[i] = x[i] + v }
fun ggml_vec_acc_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] += x[i] }
fun ggml_vec_acc1_f32(n: Int, y: FloatArray, v: Float) { for (i in 0 until n) y[i] += v }
fun ggml_vec_sub_f32(n: Int, z: FloatArray, x: FloatArray, y: FloatArray) { for (i in 0 until n) z[i] = x[i] - y[i] }
fun ggml_vec_mul_f32(n: Int, z: FloatArray, x: FloatArray, y: FloatArray) { for (i in 0 until n) z[i] = x[i] * y[i] }
fun ggml_vec_div_f32(n: Int, z: FloatArray, x: FloatArray, y: FloatArray) { for (i in 0 until n) z[i] = x[i] / y[i] }
fun ggml_vec_neg_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = -x[i] }
fun ggml_vec_sqr_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = x[i] * x[i] }
fun ggml_vec_sqrt_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = sqrt(x[i]) }
fun ggml_vec_log_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = ln(x[i]) }
fun ggml_vec_sin_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = sin(x[i]) }
fun ggml_vec_cos_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = cos(x[i]) }
fun ggml_vec_abs_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = abs(x[i]) }
fun ggml_vec_sgn_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = if (x[i] > 0f) 1f else if (x[i] < 0f) -1f else 0f }
fun ggml_vec_step_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = if (x[i] > 0f) 1f else 0f }
fun ggml_vec_exp_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = exp(x[i]) }
fun ggml_vec_tanh_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = tanh(x[i]) }

// --- Arithmetic (f16) — operate via f32 conversion ---

fun ggml_vec_add_f16(n: Int, z: ShortArray, x: ShortArray, y: ShortArray) {
    for (i in 0 until n) z[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) + io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            y[i]
        )
    )
}
fun ggml_vec_sub_f16(n: Int, z: ShortArray, x: ShortArray, y: ShortArray) {
    for (i in 0 until n) z[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) - io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            y[i]
        )
    )
}
fun ggml_vec_mul_f16(n: Int, z: ShortArray, x: ShortArray, y: ShortArray) {
    for (i in 0 until n) z[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            y[i]
        )
    )
}
fun ggml_vec_div_f16(n: Int, z: ShortArray, x: ShortArray, y: ShortArray) {
    for (i in 0 until n) z[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) / io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            y[i]
        )
    )
}
fun ggml_vec_neg_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        -io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
    )
}
fun ggml_vec_sqr_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(v * v)
    }
}
fun ggml_vec_sqrt_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        sqrt(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_log_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ln(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_sin_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        sin(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_cos_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        cos(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_abs_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        abs(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_sgn_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(if (v > 0f) 1f else if (v < 0f) -1f else 0f)
    }
}
fun ggml_vec_step_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        if (io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) > 0f) 1f else 0f
    )
}
fun ggml_vec_exp_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        exp(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_tanh_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        tanh(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}

// --- MAD (multiply-add), Scale ---

fun ggml_vec_mad_f32(n: Int, y: FloatArray, x: FloatArray, v: Float) { for (i in 0 until n) y[i] += x[i] * v }
fun ggml_vec_mad_f16(n: Int, y: ShortArray, x: ShortArray, v: Float) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(y[i]) + io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            x[i]
        ) * v
    )
}
fun ggml_vec_mad1_f32(n: Int, y: FloatArray, x: FloatArray, s: Float, b: Float) { for (i in 0 until n) y[i] = x[i] * s + b }
fun ggml_vec_mad_f32_unroll(n: Int, xs: Int, vs: Int, y: FloatArray, xv: Array<FloatArray>, vv: FloatArray) {
    for (i in 0 until n) {
        var sum = y[i]
        for (j in 0 until xs) sum += xv[j][i] * vv[j * vs]
        y[i] = sum
    }
}
fun ggml_vec_scale_f32(n: Int, y: FloatArray, v: Float) { for (i in 0 until n) y[i] *= v }
fun ggml_vec_scale_f16(n: Int, y: ShortArray, v: Float) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(y[i]) * v
    )
}

// --- Norm ---

fun ggml_vec_norm_f32(n: Int, s: FloatArray, x: FloatArray) {
    ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1)
    s[0] = sqrt(s[0])
}
fun ggml_vec_norm_inv_f32(n: Int, s: FloatArray, x: FloatArray) {
    ggml_vec_norm_f32(n, s, x)
    s[0] = 1.0f / s[0]
}

// --- Reductions ---

fun ggml_vec_sum_f32(n: Int, s: FloatArray, x: FloatArray) { var sum = 0.0f; for (i in 0 until n) sum += x[i]; s[0] = sum }
fun ggml_vec_sum_f32_ggf(n: Int, s: DoubleArray, x: FloatArray) { var sum = 0.0; for (i in 0 until n) sum += x[i]; s[0] = sum }
fun ggml_vec_sum_f16_ggf(n: Int, s: FloatArray, x: ShortArray) { var sum = 0.0f; for (i in 0 until n) sum += io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
    x[i]
); s[0] = sum }
fun ggml_vec_sum_bf16_ggf(n: Int, s: FloatArray, x: ShortArray) { var sum = 0.0f; for (i in 0 until n) sum += io.github.kotlinmania.llama.ore.ggml_compute_bf16_to_fp32(
    io.github.kotlinmania.llama.ore.GGMLBF16(x[i].toUShort())
); s[0] = sum }
fun ggml_vec_max_f32(n: Int, s: FloatArray, x: FloatArray) { var m = Float.NEGATIVE_INFINITY; for (i in 0 until n) if (x[i] > m) m = x[i]; s[0] = m }
fun ggml_vec_argmax_f32(n: Int, x: FloatArray): Int { var m = Float.NEGATIVE_INFINITY; var idx = 0; for (i in 0 until n) if (x[i] > m) { m = x[i]; idx = i }; return idx }
fun ggml_vec_cumsum_f32(n: Int, y: FloatArray, x: FloatArray) { var sum = 0.0f; for (i in 0 until n) { sum += x[i]; y[i] = sum } }

// --- Dot products ---

fun ggml_vec_dot_f32(n: Int, s: FloatArray, bs: Int, x: FloatArray, bx: Int, y: FloatArray, by: Int, nrc: Int) {
    require(nrc == 1)
    var sum = 0.0f
    for (i in 0 until n) sum += x[i] * y[i]
    s[0] = sum
}

fun ggml_vec_dot_f16(n: Int, s: FloatArray, bs: Int, x: ShortArray, bx: Int, y: ShortArray, by: Int, nrc: Int) {
    require(nrc == 1)
    var sum = 0.0f
    for (i in 0 until n) sum += io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
        y[i]
    )
    s[0] = sum
}

fun ggml_vec_dot_bf16(n: Int, s: FloatArray, bs: Int, x: ShortArray, bx: Int, y: ShortArray, by: Int, nrc: Int) {
    require(nrc == 1)
    var sum = 0.0f
    for (i in 0 until n) sum += io.github.kotlinmania.llama.ore.ggml_compute_bf16_to_fp32(
        io.github.kotlinmania.llama.ore.GGMLBF16(x[i].toUShort())
    ) * io.github.kotlinmania.llama.ore.ggml_compute_bf16_to_fp32(
        io.github.kotlinmania.llama.ore.GGMLBF16(
            y[i].toUShort()
        )
    )
    s[0] = sum
}

fun ggml_vec_dot_f16_unroll(n: Int, xs: Int, s: FloatArray, xv: Array<ShortArray>, y: ShortArray) {
    for (j in 0 until xs) {
        var sum = 0.0f
        for (i in 0 until n) sum += io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(xv[j][i]) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            y[i]
        )
        s[j] = sum
    }
}

// --- Activations (f32) ---

fun ggml_vec_relu_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = if (x[i] > 0f) x[i] else 0f }
fun ggml_vec_leaky_relu_f32(n: Int, y: FloatArray, x: FloatArray, ns: Float) {
    for (i in 0 until n) y[i] = if (x[i] > 0f) x[i] else x[i] * ns
}
fun ggml_vec_elu_f32(n: Int, y: FloatArray, x: FloatArray) {
    for (i in 0 until n) y[i] = if (x[i] > 0f) x[i] else (exp(x[i]) - 1f)
}
fun ggml_vec_sigmoid_f32(n: Int, y: FloatArray, x: FloatArray) {
    for (i in 0 until n) y[i] = 1f / (1f + exp(-x[i]))
}
fun ggml_vec_hardsigmoid_f32(n: Int, y: FloatArray, x: FloatArray) {
    for (i in 0 until n) y[i] = (x[i] / 6f + 0.5f).coerceIn(0f, 1f)
}
fun ggml_vec_hardswish_f32(n: Int, y: FloatArray, x: FloatArray) {
    for (i in 0 until n) y[i] = x[i] * (x[i] / 6f + 0.5f).coerceIn(0f, 1f)
}

private fun ggml_silu_f32(x: Float): Float = x / (1f + exp(-x))
private fun ggml_silu_f16(x: Short): Short = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
    ggml_silu_f32(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x)
    )
)
fun ggml_vec_silu_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = ggml_silu_f32(x[i]) }
fun ggml_vec_silu_backward_f32(n: Int, dx: FloatArray, x: FloatArray, dy: FloatArray) {
    for (i in 0 until n) {
        val s = 1f / (1f + exp(-x[i]))
        dx[i] = dy[i] * s * (1f + x[i] * (1f - s))
    }
}
fun ggml_vec_swiglu_f32(n: Int, y: FloatArray, x: FloatArray, g: FloatArray) {
    for (i in 0 until n) y[i] = ggml_silu_f32(x[i]) * g[i]
}

private fun ggml_gelu_f32(x: Float): Float {
    // Approximation: 0.5*x*(1 + tanh(sqrt(2/PI)*(x + 0.044715*x^3)))
    val c = 0.7978845608028654f // sqrt(2/PI)
    return 0.5f * x * (1f + tanh(c * (x + 0.044715f * x * x * x)))
}
fun ggml_vec_gelu_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = ggml_gelu_f32(x[i]) }

private fun ggml_gelu_erf_f32(x: Float): Float {
    // Exact GELU: 0.5*x*(1 + erf(x/sqrt(2)))
    // erf approximation via tanh: erf(z) ≈ tanh(z * (1.128379f + z*z * (-0.075f)))
    val z = x * 0.7071067811865476f // x / sqrt(2)
    val erfApprox = tanh(z * (1.128379f + z * z * -0.075f))
    return 0.5f * x * (1f + erfApprox)
}
fun ggml_vec_gelu_erf_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = ggml_gelu_erf_f32(x[i]) }

private fun ggml_gelu_quick_f32(x: Float): Float = x * (1f / (1f + exp(-1.702f * x)))
fun ggml_vec_gelu_quick_f32(n: Int, y: FloatArray, x: FloatArray) { for (i in 0 until n) y[i] = ggml_gelu_quick_f32(x[i]) }

// --- GLU variants (f32) ---

fun ggml_vec_geglu_f32(n: Int, y: FloatArray, x: FloatArray, g: FloatArray) { for (i in 0 until n) y[i] = ggml_gelu_f32(x[i]) * g[i] }
fun ggml_vec_geglu_erf_f32(n: Int, y: FloatArray, x: FloatArray, g: FloatArray) { for (i in 0 until n) y[i] = ggml_gelu_erf_f32(x[i]) * g[i] }
fun ggml_vec_geglu_quick_f32(n: Int, y: FloatArray, x: FloatArray, g: FloatArray) { for (i in 0 until n) y[i] = ggml_gelu_quick_f32(x[i]) * g[i] }
fun ggml_vec_reglu_f32(n: Int, y: FloatArray, x: FloatArray, g: FloatArray) { for (i in 0 until n) y[i] = (if (x[i] > 0f) x[i] else 0f) * g[i] }

// --- Activations (f16) — operate via f32 conversion ---

fun ggml_vec_relu_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(if (v > 0f) v else 0f)
    }
}
fun ggml_vec_leaky_relu_f16(n: Int, y: ShortArray, x: ShortArray, ns: Float) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(if (v > 0f) v else v * ns)
    }
}
fun ggml_vec_elu_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(if (v > 0f) v else (exp(v) - 1f))
    }
}
fun ggml_vec_sigmoid_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        1f / (1f + exp(
            -io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        ))
    )
}
fun ggml_vec_hardsigmoid_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        (io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]) / 6f + 0.5f).coerceIn(0f, 1f)
    )
}
fun ggml_vec_hardswish_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(v * (v / 6f + 0.5f).coerceIn(0f, 1f))
    }
}
fun ggml_vec_silu_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_silu_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_silu_backward_f16(n: Int, dx: ShortArray, x: ShortArray, dy: ShortArray) {
    for (i in 0 until n) {
        val xv = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        val dyv = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(dy[i])
        val s = 1f / (1f + exp(-xv))
        dx[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(dyv * s * (1f + xv * (1f - s)))
    }
}
fun ggml_vec_swiglu_f16(n: Int, y: ShortArray, x: ShortArray, g: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_silu_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        ) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(g[i])
    )
}
fun ggml_vec_gelu_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_gelu_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_gelu_erf_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_gelu_erf_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_gelu_quick_f16(n: Int, y: ShortArray, x: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_gelu_quick_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        )
    )
}
fun ggml_vec_geglu_f16(n: Int, y: ShortArray, x: ShortArray, g: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_gelu_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        ) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(g[i])
    )
}
fun ggml_vec_geglu_erf_f16(n: Int, y: ShortArray, x: ShortArray, g: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_gelu_erf_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        ) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(g[i])
    )
}
fun ggml_vec_geglu_quick_f16(n: Int, y: ShortArray, x: ShortArray, g: ShortArray) {
    for (i in 0 until n) y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        ggml_gelu_quick_f32(
            io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i])
        ) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(g[i])
    )
}
fun ggml_vec_reglu_f16(n: Int, y: ShortArray, x: ShortArray, g: ShortArray) {
    for (i in 0 until n) { val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x[i]); y[i] =
        io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
            (if (v > 0f) v else 0f) * io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(g[i])
        )
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers: silu backward (vec.h lines 1403-1412)
// ---------------------------------------------------------------------------

fun ggml_silu_backward_f32(x: Float, dy: Float): Float {
    val s = 1.0f / (1.0f + kotlin.math.exp(-x))
    return dy * s * (1.0f + x * (1.0f - s))
}

fun ggml_silu_backward_f16(x: Short, dy: Short): Short {
    val v = io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(x)
    val s = 1.0f / (1.0f + kotlin.math.exp(-v))
    return io.github.kotlinmania.llama.ore.GGML_FP32_TO_FP16(
        io.github.kotlinmania.llama.ore.GGML_FP16_TO_FP32(
            dy
        ) * s * (1.0f + v * (1.0f - s))
    )
}

// ---------------------------------------------------------------------------
// Centered variance (vec.cpp line 471 — scalar fallback)
// ---------------------------------------------------------------------------

fun ggmlVecCvarF32(n: Int, y: FloatArray, x: FloatArray, mean: Float): Double {
    var sum = 0.0
    for (i in 0 until n) {
        val v = x[i] - mean
        y[i] = v
        sum += (v * v).toDouble()
    }
    return sum
}

// ---------------------------------------------------------------------------
// Softmax (vec.cpp line 547 — scalar fallback)
// ---------------------------------------------------------------------------

fun ggmlVecSoftMaxF32(n: Int, y: FloatArray, x: FloatArray, max: Float): Double {
    var sum = 0.0
    for (i in 0 until n) {
        val v = kotlin.math.exp(x[i] - max)
        sum += v.toDouble()
        y[i] = v
    }
    return sum
}

// ---------------------------------------------------------------------------
// Log-softmax (vec.cpp line 618 — scalar fallback)
// ---------------------------------------------------------------------------

fun ggmlVecLogSoftMaxF32(n: Int, y: FloatArray, x: FloatArray, max: Float): Double {
    var sum = 0.0
    for (i in 0 until n) {
        val v = x[i] - max
        y[i] = v
        sum += kotlin.math.exp(v.toDouble())
    }
    return kotlin.math.ln(sum)
}
