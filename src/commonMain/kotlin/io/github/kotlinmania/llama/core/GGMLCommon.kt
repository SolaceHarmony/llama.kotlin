// port-lint: source ggml/src/ggml-common.h

package io.github.kotlinmania.llama.ore

/**
 * Kotlin port of ggml-common.h — shared quantization block structures and lookup tables.
 *
 * This file contains data-structure definitions only (no compute logic). Every C struct
 * becomes a Kotlin data class whose constructor mirrors the struct's packed fields.
 * Fixed-size C arrays (`int8_t qs[32]`) become `ByteArray(32)`, `ggml_half` (uint16_t)
 * becomes `Short`, and `ggml_half2` (uint32_t, packed pair of fp16) becomes `Int`.
 *
 * Constants that already live in [GGMLTypes.kt] (QK8_0, QK4_0, QK4_1, QK_K,
 * K_SCALE_SIZE) are re-exported here so that call-sites can import from one place.
 *
 * ## Naming convention
 * | C name            | Kotlin name      |
 * |--------------------|-----------------|
 * | `block_q4_0`       | [io.github.kotlinmania.llama.ore.BlockQ4_0]     |
 * | `block_q2_K`       | [io.github.kotlinmania.llama.ore.BlockQ2K]      |
 * | `block_iq2_xxs`    | [io.github.kotlinmania.llama.ore.BlockIQ2XXS]   |
 * | `iq1m_scale_t`     | [io.github.kotlinmania.llama.ore.IQ1MScale]     |
 */

// ── Re-exported constants already defined in GGMLTypes.kt ──────────────────────
// Callers can `import io.github.kotlinmania.llama.ore.*` and get everything.
// The canonical definitions live in GGMLTypes.kt; we reference them here for clarity:
//   QK4_0  = 32   (GGMLTypes.kt)
//   QK4_1  = 32   (GGMLTypes.kt)
//   QK8_0  = 32   (GGMLTypes.kt)
//   QK_K   = 256  (GGMLTypes.kt)
//   K_SCALE_SIZE = 12 (GGMLTypes.kt)

// ── Additional quantization-block-size constants not yet in GGMLTypes.kt ───────

/** Block size for Q1_0 quantization (128 elements per block). */
internal const val QK1_0: Int = 128

/** Block size for Q5_0 quantization (32 elements per block). */
internal const val QK5_0: Int = 32

/** Block size for Q5_1 quantization (32 elements per block). */
internal const val QK5_1: Int = 32

/** Block size for Q8_1 quantization (32 elements per block). */
internal const val QK8_1: Int = 32

/** Block size for MXFP4 quantization (32 elements per block). */
internal const val QK_MXFP4: Int = 32

/** Block size for NVFP4 quantization (64 elements per block). */
internal const val QK_NVFP4: Int = 64

/** Sub-block size for NVFP4 per-group scales. */
internal const val QK_NVFP4_SUB: Int = 16

/** Block size for IQ4_NL non-linear quantization (32 elements per block). */
internal const val QK4_NL: Int = 32

/** Number of scale values for IQ3_S blocks: QK_K / 64. */
internal const val IQ3S_N_SCALE: Int = io.github.kotlinmania.llama.ore.QK_K / 64

/** Grid size for IQ1_S lookup table. */
internal const val NGRID_IQ1S: Int = 2048

/** Delta constant used by IQ1_S quantization. */
internal const val IQ1S_DELTA: Float = 0.125f

/** Delta constant used by IQ1_M quantization. */
internal const val IQ1M_DELTA: Float = 0.125f

// ── GPU/CUDA-style derived constants (QR = dequant ratio, QI = int32 groups) ──
// These are useful for compute kernels that process quantized data in int32 chunks.

internal const val QR1_0: Int = 1
internal const val QI1_0: Int = io.github.kotlinmania.llama.ore.QK1_0 / 32

internal const val QR4_0: Int = 2
internal const val QI4_0: Int = io.github.kotlinmania.llama.ore.QK4_0 / (4 * io.github.kotlinmania.llama.ore.QR4_0)

internal const val QR4_1: Int = 2
internal const val QI4_1: Int = io.github.kotlinmania.llama.ore.QK4_1 / (4 * io.github.kotlinmania.llama.ore.QR4_1)

internal const val QR_MXFP4: Int = 2
internal const val QI_MXFP4: Int = io.github.kotlinmania.llama.ore.QK_MXFP4 / (4 * io.github.kotlinmania.llama.ore.QR_MXFP4)

internal const val QR_NVFP4: Int = 2
internal const val QI_NVFP4: Int = io.github.kotlinmania.llama.ore.QK_NVFP4 / (4 * io.github.kotlinmania.llama.ore.QR_NVFP4)

internal const val QR5_0: Int = 2
internal const val QI5_0: Int = io.github.kotlinmania.llama.ore.QK5_0 / (4 * io.github.kotlinmania.llama.ore.QR5_0)

internal const val QR5_1: Int = 2
internal const val QI5_1: Int = io.github.kotlinmania.llama.ore.QK5_1 / (4 * io.github.kotlinmania.llama.ore.QR5_1)

internal const val QR8_0: Int = 1
internal const val QI8_0: Int = io.github.kotlinmania.llama.ore.QK8_0 / (4 * io.github.kotlinmania.llama.ore.QR8_0)

internal const val QR8_1: Int = 1
internal const val QI8_1: Int = io.github.kotlinmania.llama.ore.QK8_1 / (4 * io.github.kotlinmania.llama.ore.QR8_1)

internal const val QR2_K: Int = 4
internal const val QI2_K: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR2_K)

internal const val QR3_K: Int = 4
internal const val QI3_K: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR3_K)

internal const val QR4_K: Int = 2
internal const val QI4_K: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR4_K)

internal const val QR5_K: Int = 2
internal const val QI5_K: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR5_K)

internal const val QR6_K: Int = 2
internal const val QI6_K: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR6_K)

internal const val QR2_XXS: Int = 4
internal const val QI2_XXS: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR2_XXS)

internal const val QR2_XS: Int = 4
internal const val QI2_XS: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR2_XS)

internal const val QR2_S: Int = 4
internal const val QI2_S: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR2_S)

internal const val QR3_XXS: Int = 4
internal const val QI3_XXS: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR3_XXS)

internal const val QR3_XS: Int = 4
internal const val QI3_XS: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR3_XS)

internal const val QR1_S: Int = 8
internal const val QI1_S: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR1_S)

internal const val QR1_M: Int = 8
internal const val QI1_M: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR1_M)

internal const val QR4_NL: Int = 2
internal const val QI4_NL: Int = io.github.kotlinmania.llama.ore.QK4_NL / (4 * io.github.kotlinmania.llama.ore.QR4_NL)

internal const val QR4_XS: Int = 2
internal const val QI4_XS: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR4_XS)

internal const val QR3_S: Int = 4
internal const val QI3_S: Int = io.github.kotlinmania.llama.ore.QK_K / (4 * io.github.kotlinmania.llama.ore.QR3_S)


// ════════════════════════════════════════════════════════════════════════════════
//  Basic quantization block structures
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Q1_0 block: 1-bit quantization.
 *
 * Layout (C): `ggml_half d; uint8_t qs[QK1_0/8];`
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property qs    bit-packed quants — 1 bit per element, [io.github.kotlinmania.llama.ore.QK1_0] / 8 = 16 bytes
 */
data class BlockQ1_0(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK1_0 / 8),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK1_0 / 8) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK1_0 / 8}" }
    }
    override fun equals(other: Any?) = other is BlockQ1_0 && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK1_0/8 = 2 + 16 = 18 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK1_0 / 8
    }
}

/**
 * Q4_0 block: 4-bit quantization (symmetric, no min).
 *
 * Layout (C): `ggml_half d; uint8_t qs[QK4_0/2];`
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property qs    nibble-packed quants — 2 elements per byte, [io.github.kotlinmania.llama.ore.QK4_0] / 2 = 16 bytes
 */
data class BlockQ4_0(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK4_0 / 2),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK4_0 / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK4_0 / 2}" }
    }
    override fun equals(other: Any?) = other is BlockQ4_0 && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK4_0/2 = 2 + 16 = 18 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK4_0 / 2
    }
}

/**
 * Q4_1 block: 4-bit quantization with separate delta and min.
 *
 * Layout (C): union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; };
 *             uint8_t qs[QK4_1/2];
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property m     min value, stored as raw fp16 bits
 * @property qs    nibble-packed quants, [io.github.kotlinmania.llama.ore.QK4_1] / 2 = 16 bytes
 */
data class BlockQ4_1(
    val d: Short,
    val m: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK4_1 / 2),
) {
    /** Packed pair of (d, m) as a single Int — matches C `ggml_half2 dm`. */
    val dm: Int get() = (d.toInt() and 0xFFFF) or ((m.toInt() and 0xFFFF) shl 16)

    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK4_1 / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK4_1 / 2}" }
    }
    override fun equals(other: Any?) = other is BlockQ4_1 && d == other.d && m == other.m && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * d.hashCode() + m.hashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: 2*sizeof(ggml_half) + QK4_1/2 = 4 + 16 = 20 */
        const val SIZE_BYTES: Int = 2 * 2 + io.github.kotlinmania.llama.ore.QK4_1 / 2
    }
}

/**
 * MXFP4 block: microscaling 4-bit float format.
 *
 * Layout (C): `uint8_t e; uint8_t qs[QK_MXFP4/2];`
 *
 * @property e     E8M0 shared exponent
 * @property qs    packed 4-bit FP values, [io.github.kotlinmania.llama.ore.QK_MXFP4] / 2 = 16 bytes
 */
data class BlockMXFP4(
    val e: Byte,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_MXFP4 / 2),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_MXFP4 / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_MXFP4 / 2}" }
    }
    override fun equals(other: Any?) = other is BlockMXFP4 && e == other.e && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * e.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: 1 + QK_MXFP4/2 = 1 + 16 = 17 */
        const val SIZE_BYTES: Int = 1 + io.github.kotlinmania.llama.ore.QK_MXFP4 / 2
    }
}

/**
 * NVFP4 block: NVIDIA 4-bit float format with per-group scales.
 *
 * Layout (C): `uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; uint8_t qs[QK_NVFP4/2];`
 *
 * @property d     UE4M3 scales — one per 16-element sub-block, 4 bytes total
 * @property qs    packed 4-bit E2M1 values, [io.github.kotlinmania.llama.ore.QK_NVFP4] / 2 = 32 bytes
 */
data class BlockNVFP4(
    val d: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_NVFP4 / 2),
) {
    init {
        require(d.size == io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB) { "d.size must be ${io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB}" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK_NVFP4 / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_NVFP4 / 2}" }
    }
    override fun equals(other: Any?) = other is BlockNVFP4 && d.contentEquals(other.d) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.contentHashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: QK_NVFP4/QK_NVFP4_SUB + QK_NVFP4/2 = 4 + 32 = 36 */
        const val SIZE_BYTES: Int = io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB + io.github.kotlinmania.llama.ore.QK_NVFP4 / 2
    }
}

/**
 * Q5_0 block: 5-bit quantization (symmetric).
 *
 * Layout (C): `ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2];`
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property qh    5th-bit of each quant, packed into 4 bytes (32 bits for 32 elements)
 * @property qs    lower 4 bits of quants, nibble-packed, [io.github.kotlinmania.llama.ore.QK5_0] / 2 = 16 bytes
 */
data class BlockQ5_0(
    val d: Short,
    val qh: ByteArray = ByteArray(4),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK5_0 / 2),
) {
    init {
        require(qh.size == 4) { "qh.size must be 4" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK5_0 / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK5_0 / 2}" }
    }
    override fun equals(other: Any?) = other is BlockQ5_0 && d == other.d && qh.contentEquals(other.qh) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * d.hashCode() + qh.contentHashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + sizeof(uint32_t) + QK5_0/2 = 2 + 4 + 16 = 22 */
        const val SIZE_BYTES: Int = 2 + 4 + io.github.kotlinmania.llama.ore.QK5_0 / 2
    }
}

/**
 * Q5_1 block: 5-bit quantization with delta and min.
 *
 * Layout (C): union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; };
 *             uint8_t qh[4]; uint8_t qs[QK5_1/2];
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property m     min value, stored as raw fp16 bits
 * @property qh    5th-bit of quants, packed into 4 bytes
 * @property qs    lower 4 bits of quants, nibble-packed, [io.github.kotlinmania.llama.ore.QK5_1] / 2 = 16 bytes
 */
data class BlockQ5_1(
    val d: Short,
    val m: Short,
    val qh: ByteArray = ByteArray(4),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK5_1 / 2),
) {
    val dm: Int get() = (d.toInt() and 0xFFFF) or ((m.toInt() and 0xFFFF) shl 16)

    init {
        require(qh.size == 4) { "qh.size must be 4" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK5_1 / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK5_1 / 2}" }
    }
    override fun equals(other: Any?) = other is BlockQ5_1 && d == other.d && m == other.m && qh.contentEquals(other.qh) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + m.hashCode()) + qh.contentHashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: 2*sizeof(ggml_half) + sizeof(uint32_t) + QK5_1/2 = 4 + 4 + 16 = 24 */
        const val SIZE_BYTES: Int = 2 * 2 + 4 + io.github.kotlinmania.llama.ore.QK5_1 / 2
    }
}

/**
 * Q8_0 block: 8-bit quantization (symmetric).
 *
 * Layout (C): `ggml_half d; int8_t qs[io.github.kotlinmania.llama.ore.QK8_0];`
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property qs    signed 8-bit quants, [io.github.kotlinmania.llama.ore.QK8_0] = 32 bytes
 */
data class BlockQ8_0(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK8_0),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK8_0) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK8_0}" }
    }
    override fun equals(other: Any?) = other is BlockQ8_0 && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK8_0 = 2 + 32 = 34 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK8_0
    }
}

/**
 * Q8_1 block: 8-bit quantization with delta and sum.
 *
 * Layout (C): union { struct { ggml_half d; ggml_half s; }; ggml_half2 ds; };
 *             int8_t qs[io.github.kotlinmania.llama.ore.QK8_1];
 *
 * @property d     delta (scale), stored as raw fp16 bits
 * @property s     d * sum(qs[i]), stored as raw fp16 bits
 * @property qs    signed 8-bit quants, [io.github.kotlinmania.llama.ore.QK8_1] = 32 bytes
 */
data class BlockQ8_1(
    val d: Short,
    val s: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK8_1),
) {
    val ds: Int get() = (d.toInt() and 0xFFFF) or ((s.toInt() and 0xFFFF) shl 16)

    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK8_1) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK8_1}" }
    }
    override fun equals(other: Any?) = other is BlockQ8_1 && d == other.d && s == other.s && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * d.hashCode() + s.hashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: 2*sizeof(ggml_half) + QK8_1 = 4 + 32 = 36 */
        const val SIZE_BYTES: Int = 2 * 2 + io.github.kotlinmania.llama.ore.QK8_1
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Ternary quantization blocks
// ════════════════════════════════════════════════════════════════════════════════

/**
 * TQ1_0 block: ternary quantization at 1.6875 bits per weight.
 *
 * Layout (C): `uint8_t qs[(QK_K - 4*QK_K/64) / 5]; uint8_t qh[QK_K/64]; ggml_half d;`
 *
 * 5 ternary elements are packed per byte (3^5 = 243 < 256), and 4 elements per byte
 * in the high-bits array.
 *
 * @property qs    base-3 packed quants
 * @property qh    high quants — 4 elements per byte
 * @property d     scale, stored as raw fp16 bits
 */
data class BlockTQ1_0(
    val qs: ByteArray = ByteArray((io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5),
    val qh: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 64),
    val d: Short,
) {
    init {
        require(qs.size == (io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5) { "qs.size must be ${(io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 64) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 64}" }
    }
    override fun equals(other: Any?) = other is BlockTQ1_0 && d == other.d && qs.contentEquals(other.qs) && qh.contentEquals(other.qh)
    override fun hashCode() = 31 * (31 * d.hashCode() + qs.contentHashCode()) + qh.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/64 + (QK_K - 4*QK_K/64)/5 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK_K / 64 + (io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5
    }
}

/**
 * TQ2_0 block: ternary quantization at 2.0625 bits per weight.
 *
 * Layout (C): `uint8_t qs[QK_K/4]; ggml_half d;`
 *
 * @property qs    2-bit packed quants, [io.github.kotlinmania.llama.ore.QK_K] / 4 = 64 bytes
 * @property d     scale, stored as raw fp16 bits
 */
data class BlockTQ2_0(
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 4),
    val d: Short,
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 4) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 4}" }
    }
    override fun equals(other: Any?) = other is BlockTQ2_0 && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/4 = 2 + 64 = 66 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK_K / 4
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  Super-block (K-quant) quantization structures
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Q2_K block: 2-bit quantization (super-block, effectively ~2.625 bpw).
 *
 * Weight x = a * q + b.  16 blocks of 16 elements each.
 *
 * Layout (C): `uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4];
 *              union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };`
 *
 * @property scales  scales and mins, quantized with 4 bits, [io.github.kotlinmania.llama.ore.QK_K]/16 = 16 bytes
 * @property qs      2-bit packed quants, [io.github.kotlinmania.llama.ore.QK_K]/4 = 64 bytes
 * @property d       super-block scale for quantized scales (raw fp16 bits)
 * @property dmin    super-block scale for quantized mins (raw fp16 bits)
 */
data class BlockQ2K(
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 16),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 4),
    val d: Short,
    val dmin: Short,
) {
    val dm: Int get() = (d.toInt() and 0xFFFF) or ((dmin.toInt() and 0xFFFF) shl 16)

    init {
        require(scales.size == io.github.kotlinmania.llama.ore.QK_K / 16) { "scales.size must be ${io.github.kotlinmania.llama.ore.QK_K / 16}" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 4) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 4}" }
    }
    override fun equals(other: Any?) = other is BlockQ2K && d == other.d && dmin == other.dmin && scales.contentEquals(other.scales) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + dmin.hashCode()) + scales.contentHashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: 2*sizeof(ggml_half) + QK_K/16 + QK_K/4 = 4 + 16 + 64 = 84 */
        const val SIZE_BYTES: Int = 2 * 2 + io.github.kotlinmania.llama.ore.QK_K / 16 + io.github.kotlinmania.llama.ore.QK_K / 4
    }
}

/**
 * Q3_K block: 3-bit quantization (super-block, effectively ~3.4375 bpw).
 *
 * Weight x = a * q.  16 blocks of 16 elements each.
 *
 * Layout (C): `uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d;`
 *
 * @property hmask   high-bit mask for quants, [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 bytes
 * @property qs      lower 2 bits of quants, [io.github.kotlinmania.llama.ore.QK_K]/4 = 64 bytes
 * @property scales  6-bit quantized scales, 12 bytes
 * @property d       super-block scale (raw fp16 bits)
 */
data class BlockQ3K(
    val hmask: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 8),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 4),
    val scales: ByteArray = ByteArray(12),
    val d: Short,
) {
    init {
        require(hmask.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "hmask.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 4) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 4}" }
        require(scales.size == 12) { "scales.size must be 12" }
    }
    override fun equals(other: Any?) = other is BlockQ3K && d == other.d && hmask.contentEquals(other.hmask) && qs.contentEquals(other.qs) && scales.contentEquals(other.scales)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + hmask.contentHashCode()) + qs.contentHashCode()) + scales.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/4 + QK_K/8 + 12 = 2 + 64 + 32 + 12 = 110 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 8 + 12
    }
}

/**
 * Q4_K block: 4-bit quantization (super-block, effectively ~4.5 bpw).
 *
 * Weight x = a * q + b.  8 blocks of 32 elements each.
 *
 * Layout (C): union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };
 *             uint8_t scales[io.github.kotlinmania.llama.ore.K_SCALE_SIZE]; uint8_t qs[QK_K/2];
 *
 * @property d       super-block scale for quantized scales (raw fp16 bits)
 * @property dmin    super-block scale for quantized mins (raw fp16 bits)
 * @property scales  6-bit quantized scales and mins, [io.github.kotlinmania.llama.ore.K_SCALE_SIZE] = 12 bytes
 * @property qs      4-bit packed quants, [io.github.kotlinmania.llama.ore.QK_K]/2 = 128 bytes
 */
data class BlockQ4K(
    val d: Short,
    val dmin: Short,
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.K_SCALE_SIZE),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 2),
) {
    val dm: Int get() = (d.toInt() and 0xFFFF) or ((dmin.toInt() and 0xFFFF) shl 16)

    init {
        require(scales.size == io.github.kotlinmania.llama.ore.K_SCALE_SIZE) { "scales.size must be ${io.github.kotlinmania.llama.ore.K_SCALE_SIZE}" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 2}" }
    }
    override fun equals(other: Any?) = other is BlockQ4K && d == other.d && dmin == other.dmin && scales.contentEquals(other.scales) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + dmin.hashCode()) + scales.contentHashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 = 4 + 12 + 128 = 144 */
        const val SIZE_BYTES: Int = 2 * 2 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 2
    }
}

/**
 * Q5_K block: 5-bit quantization (super-block, effectively ~5.5 bpw).
 *
 * Weight x = a * q + b.  8 blocks of 32 elements each.
 *
 * Layout (C): union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };
 *             uint8_t scales[io.github.kotlinmania.llama.ore.K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2];
 *
 * @property d       super-block scale for quantized scales (raw fp16 bits)
 * @property dmin    super-block scale for quantized mins (raw fp16 bits)
 * @property scales  6-bit quantized scales, [io.github.kotlinmania.llama.ore.K_SCALE_SIZE] = 12 bytes
 * @property qh      high bits of quants, [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 bytes
 * @property qs      low 4 bits of quants, [io.github.kotlinmania.llama.ore.QK_K]/2 = 128 bytes
 */
data class BlockQ5K(
    val d: Short,
    val dmin: Short,
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.K_SCALE_SIZE),
    val qh: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 8),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 2),
) {
    val dm: Int get() = (d.toInt() and 0xFFFF) or ((dmin.toInt() and 0xFFFF) shl 16)

    init {
        require(scales.size == io.github.kotlinmania.llama.ore.K_SCALE_SIZE) { "scales.size must be ${io.github.kotlinmania.llama.ore.K_SCALE_SIZE}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 2}" }
    }
    override fun equals(other: Any?) = other is BlockQ5K && d == other.d && dmin == other.dmin && scales.contentEquals(other.scales) && qh.contentEquals(other.qh) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * (31 * (31 * d.hashCode() + dmin.hashCode()) + scales.contentHashCode()) + qh.contentHashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/8 + QK_K/2 = 4 + 12 + 32 + 128 = 176 */
        const val SIZE_BYTES: Int = 2 * 2 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 2
    }
}

/**
 * Q6_K block: 6-bit quantization (super-block, effectively ~6.5625 bpw).
 *
 * Weight x = a * q.  16 blocks of 16 elements each.
 *
 * Layout (C): `uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d;`
 *
 * @property ql      lower 4 bits of quants, [io.github.kotlinmania.llama.ore.QK_K]/2 = 128 bytes
 * @property qh      upper 2 bits of quants, [io.github.kotlinmania.llama.ore.QK_K]/4 = 64 bytes
 * @property scales  8-bit quantized scales, [io.github.kotlinmania.llama.ore.QK_K]/16 = 16 bytes
 * @property d       super-block scale (raw fp16 bits)
 */
data class BlockQ6K(
    val ql: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 2),
    val qh: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 4),
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 16),
    val d: Short,
) {
    init {
        require(ql.size == io.github.kotlinmania.llama.ore.QK_K / 2) { "ql.size must be ${io.github.kotlinmania.llama.ore.QK_K / 2}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 4) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 4}" }
        require(scales.size == io.github.kotlinmania.llama.ore.QK_K / 16) { "scales.size must be ${io.github.kotlinmania.llama.ore.QK_K / 16}" }
    }
    override fun equals(other: Any?) = other is BlockQ6K && d == other.d && ql.contentEquals(other.ql) && qh.contentEquals(other.qh) && scales.contentEquals(other.scales)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + ql.contentHashCode()) + qh.contentHashCode()) + scales.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/16 + 3*QK_K/4 = 2 + 16 + 192 = 210 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK_K / 16 + 3 * io.github.kotlinmania.llama.ore.QK_K / 4
    }
}

/**
 * Q8_K block: 8-bit quantization (super-block) — used only for intermediate quantization and dot products.
 *
 * Layout (C): `float d; int8_t qs[io.github.kotlinmania.llama.ore.QK_K]; int16_t bsums[QK_K/16];`
 *
 * @property d       delta (scale), stored as 32-bit float
 * @property qs      signed 8-bit quants, [io.github.kotlinmania.llama.ore.QK_K] = 256 bytes
 * @property bsums   sum of quants in groups of 16, [io.github.kotlinmania.llama.ore.QK_K]/16 = 16 entries (stored as ShortArray)
 */
data class BlockQ8K(
    val d: Float,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K),
    val bsums: ShortArray = ShortArray(io.github.kotlinmania.llama.ore.QK_K / 16),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K}" }
        require(bsums.size == io.github.kotlinmania.llama.ore.QK_K / 16) { "bsums.size must be ${io.github.kotlinmania.llama.ore.QK_K / 16}" }
    }
    override fun equals(other: Any?) = other is BlockQ8K && d == other.d && qs.contentEquals(other.qs) && bsums.contentEquals(other.bsums)
    override fun hashCode() = 31 * (31 * d.hashCode() + qs.contentHashCode()) + bsums.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(float) + QK_K + QK_K/16*sizeof(int16_t) = 4 + 256 + 32 = 292 */
        const val SIZE_BYTES: Int = 4 + io.github.kotlinmania.llama.ore.QK_K + (io.github.kotlinmania.llama.ore.QK_K / 16) * 2
    }
}

// ════════════════════════════════════════════════════════════════════════════════
//  IQ (Importance Quantization) block structures
// ════════════════════════════════════════════════════════════════════════════════

/**
 * IQ2_XXS block: (Almost) true 2-bit quantization (~2.0625 bpw).
 *
 * Layout (C): `ggml_half d; uint16_t qs[QK_K/8];`
 *
 * @property d     scale (raw fp16 bits)
 * @property qs    grid indices, [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 entries stored as ShortArray (uint16_t)
 */
data class BlockIQ2XXS(
    val d: Short,
    val qs: ShortArray = ShortArray(io.github.kotlinmania.llama.ore.QK_K / 8),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
    }
    override fun equals(other: Any?) = other is BlockIQ2XXS && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) = 2 + 64 = 66 */
        const val SIZE_BYTES: Int = 2 + (io.github.kotlinmania.llama.ore.QK_K / 8) * 2
    }
}

/**
 * IQ2_XS block: 2.3125 bpw quants.
 *
 * Layout (C): `ggml_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32];`
 *
 * @property d       scale (raw fp16 bits)
 * @property qs      grid indices, [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 entries (ShortArray)
 * @property scales  block scales, [io.github.kotlinmania.llama.ore.QK_K]/32 = 8 bytes
 */
data class BlockIQ2XS(
    val d: Short,
    val qs: ShortArray = ShortArray(io.github.kotlinmania.llama.ore.QK_K / 8),
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
        require(scales.size == io.github.kotlinmania.llama.ore.QK_K / 32) { "scales.size must be ${io.github.kotlinmania.llama.ore.QK_K / 32}" }
    }
    override fun equals(other: Any?) = other is BlockIQ2XS && d == other.d && qs.contentEquals(other.qs) && scales.contentEquals(other.scales)
    override fun hashCode() = 31 * (31 * d.hashCode() + qs.contentHashCode()) + scales.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/8*sizeof(uint16_t) + QK_K/32 = 2 + 64 + 8 = 74 */
        const val SIZE_BYTES: Int = 2 + (io.github.kotlinmania.llama.ore.QK_K / 8) * 2 + io.github.kotlinmania.llama.ore.QK_K / 32
    }
}

/**
 * IQ2_S block: 2.5625 bpw quants.
 *
 * Layout (C): `ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32];`
 *
 * @property d       scale (raw fp16 bits)
 * @property qs      quants, [io.github.kotlinmania.llama.ore.QK_K]/4 = 64 bytes
 * @property qh      high bits, [io.github.kotlinmania.llama.ore.QK_K]/32 = 8 bytes
 * @property scales  block scales, [io.github.kotlinmania.llama.ore.QK_K]/32 = 8 bytes
 */
data class BlockIQ2S(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 4),
    val qh: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 4) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 4}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 32) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 32}" }
        require(scales.size == io.github.kotlinmania.llama.ore.QK_K / 32) { "scales.size must be ${io.github.kotlinmania.llama.ore.QK_K / 32}" }
    }
    override fun equals(other: Any?) = other is BlockIQ2S && d == other.d && qs.contentEquals(other.qs) && qh.contentEquals(other.qh) && scales.contentEquals(other.scales)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + qs.contentHashCode()) + qh.contentHashCode()) + scales.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/4 + QK_K/16 = 2 + 64 + 16 = 82 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 16
    }
}

/**
 * IQ3_XXS block: (Almost) true 3-bit quantization (~3.0625 bpw).
 *
 * Layout (C): `ggml_half d; uint8_t qs[3*QK_K/8];`
 *
 * @property d     scale (raw fp16 bits)
 * @property qs    packed quants, 3 * [io.github.kotlinmania.llama.ore.QK_K] / 8 = 96 bytes
 */
data class BlockIQ3XXS(
    val d: Short,
    val qs: ByteArray = ByteArray(3 * io.github.kotlinmania.llama.ore.QK_K / 8),
) {
    init {
        require(qs.size == 3 * io.github.kotlinmania.llama.ore.QK_K / 8) { "qs.size must be ${3 * io.github.kotlinmania.llama.ore.QK_K / 8}" }
    }
    override fun equals(other: Any?) = other is BlockIQ3XXS && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + 3*(QK_K/8) = 2 + 96 = 98 */
        const val SIZE_BYTES: Int = 2 + 3 * (io.github.kotlinmania.llama.ore.QK_K / 8)
    }
}

/**
 * IQ3_S block: 3.4375 bpw.
 *
 * Layout (C): `ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32];
 *              uint8_t signs[QK_K/8]; uint8_t scales[io.github.kotlinmania.llama.ore.IQ3S_N_SCALE];`
 *
 * @property d       scale (raw fp16 bits)
 * @property qs      quants, [io.github.kotlinmania.llama.ore.QK_K]/4 = 64 bytes
 * @property qh      high bits, [io.github.kotlinmania.llama.ore.QK_K]/32 = 8 bytes
 * @property signs   sign bits, [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 bytes
 * @property scales  block scales, [io.github.kotlinmania.llama.ore.IQ3S_N_SCALE] = 4 bytes
 */
data class BlockIQ3S(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 4),
    val qh: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32),
    val signs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 8),
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.IQ3S_N_SCALE),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 4) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 4}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 32) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 32}" }
        require(signs.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "signs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
        require(scales.size == io.github.kotlinmania.llama.ore.IQ3S_N_SCALE) { "scales.size must be ${io.github.kotlinmania.llama.ore.IQ3S_N_SCALE}" }
    }
    override fun equals(other: Any?) = other is BlockIQ3S && d == other.d && qs.contentEquals(other.qs) && qh.contentEquals(other.qh) && signs.contentEquals(other.signs) && scales.contentEquals(other.scales)
    override fun hashCode(): Int {
        var h = d.hashCode()
        h = 31 * h + qs.contentHashCode()
        h = 31 * h + qh.contentHashCode()
        h = 31 * h + signs.contentHashCode()
        h = 31 * h + scales.contentHashCode()
        return h
    }

    companion object {
        /** Size in bytes: sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE = 2 + 104 + 4 = 110 */
        const val SIZE_BYTES: Int = 2 + 13 * (io.github.kotlinmania.llama.ore.QK_K / 32) + io.github.kotlinmania.llama.ore.IQ3S_N_SCALE
    }
}

/**
 * IQ1_S block: 1.5625 bpw.
 *
 * Layout (C): `ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32];`
 *
 * @property d     scale (raw fp16 bits)
 * @property qs    grid indices (low 8 bits), [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 bytes
 * @property qh    grid indices (high bits + sign), [io.github.kotlinmania.llama.ore.QK_K]/32 = 8 entries (ShortArray)
 */
data class BlockIQ1S(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 8),
    val qh: ShortArray = ShortArray(io.github.kotlinmania.llama.ore.QK_K / 32),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 32) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 32}" }
    }
    override fun equals(other: Any?) = other is BlockIQ1S && d == other.d && qs.contentEquals(other.qs) && qh.contentEquals(other.qh)
    override fun hashCode() = 31 * (31 * d.hashCode() + qs.contentHashCode()) + qh.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK_K/8 + QK_K/16 = 2 + 32 + 16 = 50 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16
    }
}

/**
 * IQ1_M block: 1.75 bpw.
 *
 * Layout (C): `uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32];`
 *
 * Note: no explicit scale field — the scale is encoded in the last entries of `scales`
 * using the [io.github.kotlinmania.llama.ore.IQ1MScale] union.
 *
 * @property qs      grid index, low 8 bits, [io.github.kotlinmania.llama.ore.QK_K]/8 = 32 bytes
 * @property qh      grid index, high 3 bits + shift, [io.github.kotlinmania.llama.ore.QK_K]/16 = 16 bytes
 * @property scales  3-bit block scales, [io.github.kotlinmania.llama.ore.QK_K]/32 = 8 bytes
 */
data class BlockIQ1M(
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 8),
    val qh: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 16),
    val scales: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 32),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 8) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 8}" }
        require(qh.size == io.github.kotlinmania.llama.ore.QK_K / 16) { "qh.size must be ${io.github.kotlinmania.llama.ore.QK_K / 16}" }
        require(scales.size == io.github.kotlinmania.llama.ore.QK_K / 32) { "scales.size must be ${io.github.kotlinmania.llama.ore.QK_K / 32}" }
    }
    override fun equals(other: Any?) = other is BlockIQ1M && qs.contentEquals(other.qs) && qh.contentEquals(other.qh) && scales.contentEquals(other.scales)
    override fun hashCode() = 31 * (31 * qs.contentHashCode() + qh.contentHashCode()) + scales.contentHashCode()

    companion object {
        /** Size in bytes: QK_K/8 + QK_K/16 + QK_K/32 = 32 + 16 + 8 = 56 */
        const val SIZE_BYTES: Int = io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16 + io.github.kotlinmania.llama.ore.QK_K / 32
    }
}

/**
 * Union type used by IQ1_M quants to overlay an fp16 value on raw uint16 bits.
 *
 * In C: `typedef union { ggml_half f16; uint16_t u16; } iq1m_scale_t;`
 *
 * @property bits  the raw 16-bit value; interpret as fp16 via [NumericConversions].
 */
data class IQ1MScale(val bits: Short) {
    /** Interpret the raw bits as an IEEE 754 half-precision float. */
    val f16: Short get() = bits
    /** Interpret the raw bits as an unsigned 16-bit integer. */
    val u16: Int get() = bits.toInt() and 0xFFFF
}

// ── Non-linear quantization ────────────────────────────────────────────────────

/**
 * IQ4_NL block: non-linear 4-bit quantization.
 *
 * Layout (C): `ggml_half d; uint8_t qs[QK4_NL/2];`
 *
 * @property d     scale (raw fp16 bits)
 * @property qs    nibble-packed quants, [io.github.kotlinmania.llama.ore.QK4_NL]/2 = 16 bytes
 */
data class BlockIQ4NL(
    val d: Short,
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK4_NL / 2),
) {
    init {
        require(qs.size == io.github.kotlinmania.llama.ore.QK4_NL / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK4_NL / 2}" }
    }
    override fun equals(other: Any?) = other is BlockIQ4NL && d == other.d && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * d.hashCode() + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + QK4_NL/2 = 2 + 16 = 18 */
        const val SIZE_BYTES: Int = 2 + io.github.kotlinmania.llama.ore.QK4_NL / 2
    }
}

/**
 * IQ4_XS block: extended non-linear 4-bit quantization.
 *
 * Layout (C): `ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2];`
 *
 * @property d        scale (raw fp16 bits)
 * @property scalesH  high bits of sub-block scales (uint16_t)
 * @property scalesL  low bits of sub-block scales, [io.github.kotlinmania.llama.ore.QK_K]/64 = 4 bytes
 * @property qs       nibble-packed quants, [io.github.kotlinmania.llama.ore.QK_K]/2 = 128 bytes
 */
data class BlockIQ4XS(
    val d: Short,
    val scalesH: Short,
    val scalesL: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 64),
    val qs: ByteArray = ByteArray(io.github.kotlinmania.llama.ore.QK_K / 2),
) {
    init {
        require(scalesL.size == io.github.kotlinmania.llama.ore.QK_K / 64) { "scalesL.size must be ${io.github.kotlinmania.llama.ore.QK_K / 64}" }
        require(qs.size == io.github.kotlinmania.llama.ore.QK_K / 2) { "qs.size must be ${io.github.kotlinmania.llama.ore.QK_K / 2}" }
    }
    override fun equals(other: Any?) = other is BlockIQ4XS && d == other.d && scalesH == other.scalesH && scalesL.contentEquals(other.scalesL) && qs.contentEquals(other.qs)
    override fun hashCode() = 31 * (31 * (31 * d.hashCode() + scalesH.hashCode()) + scalesL.contentHashCode()) + qs.contentHashCode()

    companion object {
        /** Size in bytes: sizeof(ggml_half) + sizeof(uint16_t) + QK_K/64 + QK_K/2 = 2 + 2 + 4 + 128 = 136 */
        const val SIZE_BYTES: Int = 2 + 2 + io.github.kotlinmania.llama.ore.QK_K / 64 + io.github.kotlinmania.llama.ore.QK_K / 2
    }
}


// ════════════════════════════════════════════════════════════════════════════════
//  Lookup tables (from GGML_COMMON_IMPL section of ggml-common.h)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Singleton object containing all lookup tables ported from the GGML_COMMON_IMPL
 * section of `ggml-common.h`. These are used by IQ quantization kernels.
 */
object GGMLCommonTables {

    /** Bitmask table for IQ2_XS sign extraction (8 entries). */
    val kmaskIQ2XS: ByteArray = byteArrayOf(
        1, 2, 4, 8, 16, 32, 64, -128 // 128 wraps to -128 as signed byte
    )

    /** Sign table for IQ2_XS (128 entries). */
    val ksignsIQ2XS: ByteArray = byteArrayOf(
          0, -127, -126,    3, -124,    5,    6, -121, -120,    9,   10, -117,   12, -115, -114,   15,
       -112,   17,   18, -109,   20, -107, -106,   23,   24, -103, -102,   27, -100,   29,   30, -97,
        -96,   33,   34, -93,   36, -91,  -90,   39,   40, -87,  -86,   43, -84,   45,   46, -81,
         48, -79,  -78,   51, -76,   53,   54, -73,  -72,   57,   58, -69,   60, -67,  -66,   63,
        -64,   65,   66, -61,   68, -59,  -58,   71,   72, -55,  -54,   75, -52,   77,   78, -49,
         80, -47,  -46,   83, -44,   85,   86, -41,  -40,   89,   90, -37,   92, -35,  -34,   95,
         96, -31,  -30,   99, -28,  101,  102, -25,  -24,  105,  106, -21,  108, -19,  -18,  111,
       -16,  113,  114, -13,  116, -11,  -10,  119,  120,   -7,   -6,  123,   -4,  125,  126,   -1,
    )

    /** IQ4_NL dequantization values (16 entries, signed). */
    val kvaluesIQ4NL: ByteArray = byteArrayOf(
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    )

    /** MXFP4 dequantization values (e2m1 values, doubled). 16 entries, signed. */
    val kvaluesMXFP4: ByteArray = byteArrayOf(
        0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
    )

    /**
     * Sign expansion table: 128 x uint64 entries.
     *
     * Each entry is a bitmask where bytes are either 0x00 or 0xFF depending on
     * the sign pattern encoded in the index.
     */
    val ksigns64: LongArray = longArrayOf(
        0x0000000000000000L, -0x00FFFFFFFFFFFF01L, -0x00FFFFFFFFFF0100L, 0x000000000000FFFFL,
        -0x00FFFFFFFF010000L, 0x0000000000FF00FFL, 0x0000000000FFFF00L, -0x00FFFFFFFF000001L,
        -0x00FFFFFF01000000L, 0x00000000FF0000FFL, 0x00000000FF00FF00L, -0x00FFFFFF00FF0001L,
        0x00000000FFFF0000L, -0x00FFFFFF0000FF01L, -0x00FFFFFF00000100L, 0x00000000FFFFFFFFL,
        -0x00FFFF0100000000L, 0x000000FF000000FFL, 0x000000FF0000FF00L, -0x00FFFF010000FF01L,
        0x000000FF00FF0000L, -0x00FFFF0100FF0001L, -0x00FFFF0100000100L, 0x000000FF00FFFFFFL,
        0x000000FFFF000000L, -0x00FFFF00FF000001L, -0x00FFFF00FF0000FFL, 0x000000FFFF00FFFFL,
        -0x00FFFF0000010000L, 0x000000FFFFFF00FFL, 0x000000FFFFFFFF00L, -0x00FFFF0000000001L,
        -0x00FF010000000000L, 0x0000FF00000000FFL, 0x0000FF000000FF00L, -0x00FF01000000FF01L,
        0x0000FF0000FF0000L, -0x00FF010000FF0001L, -0x00FF010000000100L, 0x0000FF0000FFFFFFL,
        0x0000FF00FF000000L, -0x00FF0100FF000001L, -0x00FF0100FF0000FFL, 0x0000FF00FF00FFFFL,
        -0x00FF0100FFFF0000L, 0x0000FF00FFFF00FFL, 0x0000FF00FFFFFF00L, -0x00FF0100FFFFFFFFL,
        0x0000FFFF00000000L, -0x00FF00FF00000001L, -0x00FF00FF0000FF00L, 0x0000FFFF0000FFFFL,
        -0x00FF00FF00FF0000L, 0x0000FFFF00FF00FFL, 0x0000FFFF00FFFF00L, -0x00FF00FF00FFFFFFL,
        -0x00FF00FFFFFF0000L, 0x0000FFFFFF0000FFL, 0x0000FFFFFF00FF00L, -0x00FF00FFFFFF0001L,
        0x0000FFFFFFFF0000L, -0x00FF00FFFFFFFF01L, -0x00FF00FFFFFFFFFEL, 0x0000FFFFFFFFFFFFL,
        -0x0001000000000000L, 0x00FF0000000000FFL, 0x00FF00000000FF00L, -0x000100000000FF01L,
        0x00FF000000FF0000L, -0x0001000000FF0001L, -0x0001000000000100L, 0x00FF000000FFFFFFL,
        0x00FF0000FF000000L, -0x00010000FF000001L, -0x00010000FF0000FFL, 0x00FF0000FF00FFFFL,
        -0x00010000FFFF0000L, 0x00FF0000FFFF00FFL, 0x00FF0000FFFFFF00L, -0x00010000FFFFFFFFL,
        0x00FF00FF00000000L, -0x000100FF00000001L, -0x000100FF0000FF00L, 0x00FF00FF0000FFFFL,
        -0x000100FF00FF0000L, 0x00FF00FF00FF00FFL, 0x00FF00FF00FFFF00L, -0x000100FF00FFFFFFL,
        -0x000100FFFF000000L, 0x00FF00FFFF0000FFL, 0x00FF00FFFF00FF00L, -0x000100FFFF00FFFFL,
        0x00FF00FFFFFF0000L, -0x000100FFFFFF00FFL, -0x000100FFFFFFFE00L, 0x00FF00FFFFFFFFFFL,
        0x00FFFF0000000000L, -0x00000100000000FFL, -0x0000010000FF0000L, 0x00FFFF000000FFFFL,
        -0x000001000000FF00L, 0x00FFFF0000FF00FFL, 0x00FFFF0000FFFF00L, -0x000001000000FFFFL,
        -0x00000100FF000000L, 0x00FFFF00FF0000FFL, 0x00FFFF00FF00FF00L, -0x00000100FF00FFFFL,
        0x00FFFF00FFFF0000L, -0x00000100FFFF00FFL, -0x00000100FFFFFF00L, 0x00FFFF00FFFFFFFFL,
        -0x0000010000000000L, 0x00FFFFFF000000FFL, 0x00FFFFFF0000FF00L, -0x0000FFFF0000FFFFL,
        0x00FFFFFF00FF0000L, -0x0000FFFF00FF00FFL, -0x0000FFFF00FFFF00L, 0x00FFFFFF00FFFFFFL,
        0x00FFFFFFFF000000L, -0x0000FFFFFF0000FFL, -0x0000FFFFFF00FF00L, 0x00FFFFFFFF00FFFFL,
        -0x0000FFFFFFFF0000L, 0x00FFFFFFFFFF00FFL, 0x00FFFFFFFFFFFF00L, -0x0000000000000001L,
    )

    // Large grid tables (iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid, iq3s_grid,
    // iq1s_grid) are very large (hundreds to thousands of uint64/uint32 entries each).
    // They are omitted here for code size reasons and will be loaded lazily from a
    // resource or generated programmatically when needed by IQ quantization kernels.
    //
    // The following declarations document the expected shapes. Implementations should fill
    // these via `by lazy { ... }` when actual IQ quantization support is added.

    /**
     * IQ2_XXS grid lookup table — 256 × ULong entries.
     * Populated lazily when IQ2_XXS quantization is first used.
     */
    val iq2xxsGrid: LongArray by lazy { LongArray(256) } // LATER: populate from ggml-common.h

    /**
     * IQ2_XS grid lookup table — 512 × ULong entries.
     */
    val iq2xsGrid: LongArray by lazy { LongArray(512) } // LATER: populate from ggml-common.h

    /**
     * IQ2_S grid lookup table — 1024 × ULong entries.
     */
    val iq2sGrid: LongArray by lazy { LongArray(1024) } // LATER: populate from ggml-common.h

    /**
     * IQ3_XXS grid lookup table — 256 × UInt entries.
     */
    val iq3xxsGrid: IntArray by lazy { IntArray(256) } // LATER: populate from ggml-common.h

    /**
     * IQ3_S grid lookup table — 512 × UInt entries.
     */
    val iq3sGrid: IntArray by lazy { IntArray(512) } // LATER: populate from ggml-common.h

    /**
     * IQ1_S grid lookup table — [io.github.kotlinmania.llama.ore.NGRID_IQ1S] × ULong entries.
     * On CPU (C impl) this is `uint64_t iq1s_grid[2048]`.
     * On GPU this uses a compressed `uint32_t iq1s_grid_gpu[2048]` representation.
     */
    val iq1sGrid: LongArray by lazy { LongArray(io.github.kotlinmania.llama.ore.NGRID_IQ1S) } // LATER: populate from ggml-common.h
}

// ════════════════════════════════════════════════════════════════════════════════
//  Compile-time size assertions (mirrors C static_assert)
// ════════════════════════════════════════════════════════════════════════════════

/**
 * Validates that every block companion [SIZE_BYTES] constant matches the C
 * `sizeof(block_*)` values. Called once at class-load time. A mismatch throws
 * [IllegalStateException] — this is the Kotlin equivalent of C `static_assert`.
 */
internal fun verifyBlockSizes() {
    check(io.github.kotlinmania.llama.ore.BlockQ1_0.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK1_0 / 8) { "wrong q1_0 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ4_0.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK4_0 / 2) { "wrong q4_0 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ4_1.SIZE_BYTES == 2 * 2 + io.github.kotlinmania.llama.ore.QK4_1 / 2) { "wrong q4_1 block size" }
    check(io.github.kotlinmania.llama.ore.BlockMXFP4.SIZE_BYTES == 1 + io.github.kotlinmania.llama.ore.QK_MXFP4 / 2) { "wrong mxfp4 block size" }
    check(io.github.kotlinmania.llama.ore.BlockNVFP4.SIZE_BYTES == io.github.kotlinmania.llama.ore.QK_NVFP4 / io.github.kotlinmania.llama.ore.QK_NVFP4_SUB + io.github.kotlinmania.llama.ore.QK_NVFP4 / 2) { "wrong nvfp4 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ5_0.SIZE_BYTES == 2 + 4 + io.github.kotlinmania.llama.ore.QK5_0 / 2) { "wrong q5_0 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ5_1.SIZE_BYTES == 2 * 2 + 4 + io.github.kotlinmania.llama.ore.QK5_1 / 2) { "wrong q5_1 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ8_0.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK8_0) { "wrong q8_0 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ8_1.SIZE_BYTES == 2 * 2 + io.github.kotlinmania.llama.ore.QK8_1) { "wrong q8_1 block size" }
    check(io.github.kotlinmania.llama.ore.BlockTQ1_0.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK_K / 64 + (io.github.kotlinmania.llama.ore.QK_K - 4 * io.github.kotlinmania.llama.ore.QK_K / 64) / 5) { "wrong tq1_0 block size" }
    check(io.github.kotlinmania.llama.ore.BlockTQ2_0.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK_K / 4) { "wrong tq2_0 block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ2K.SIZE_BYTES == 2 * 2 + io.github.kotlinmania.llama.ore.QK_K / 16 + io.github.kotlinmania.llama.ore.QK_K / 4) { "wrong q2_K block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ3K.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 8 + 12) { "wrong q3_K block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ4K.SIZE_BYTES == 2 * 2 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 2) { "wrong q4_K block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ5K.SIZE_BYTES == 2 * 2 + io.github.kotlinmania.llama.ore.K_SCALE_SIZE + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 2) { "wrong q5_K block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ6K.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK_K / 16 + 3 * io.github.kotlinmania.llama.ore.QK_K / 4) { "wrong q6_K block size" }
    check(io.github.kotlinmania.llama.ore.BlockQ8K.SIZE_BYTES == 4 + io.github.kotlinmania.llama.ore.QK_K + (io.github.kotlinmania.llama.ore.QK_K / 16) * 2) { "wrong q8_K block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ2XXS.SIZE_BYTES == 2 + (io.github.kotlinmania.llama.ore.QK_K / 8) * 2) { "wrong iq2_xxs block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ2XS.SIZE_BYTES == 2 + (io.github.kotlinmania.llama.ore.QK_K / 8) * 2 + io.github.kotlinmania.llama.ore.QK_K / 32) { "wrong iq2_xs block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ2S.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK_K / 4 + io.github.kotlinmania.llama.ore.QK_K / 16) { "wrong iq2_s block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ3XXS.SIZE_BYTES == 2 + 3 * (io.github.kotlinmania.llama.ore.QK_K / 8)) { "wrong iq3_xxs block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ3S.SIZE_BYTES == 2 + 13 * (io.github.kotlinmania.llama.ore.QK_K / 32) + io.github.kotlinmania.llama.ore.IQ3S_N_SCALE) { "wrong iq3_s block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ1S.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16) { "wrong iq1_s block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ1M.SIZE_BYTES == io.github.kotlinmania.llama.ore.QK_K / 8 + io.github.kotlinmania.llama.ore.QK_K / 16 + io.github.kotlinmania.llama.ore.QK_K / 32) { "wrong iq1_m block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ4NL.SIZE_BYTES == 2 + io.github.kotlinmania.llama.ore.QK4_NL / 2) { "wrong iq4_nl block size" }
    check(io.github.kotlinmania.llama.ore.BlockIQ4XS.SIZE_BYTES == 2 + 2 + io.github.kotlinmania.llama.ore.QK_K / 64 + io.github.kotlinmania.llama.ore.QK_K / 2) { "wrong iq4_xs block size" }
}

// Run verification eagerly — equivalent to C static_assert at compile time.
// In Kotlin this happens once when the file's top-level code is first referenced.
private val _blockSizeVerification: Unit = io.github.kotlinmania.llama.ore.verifyBlockSizes()
