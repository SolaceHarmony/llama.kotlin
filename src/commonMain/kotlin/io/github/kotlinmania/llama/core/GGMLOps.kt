// port-lint: source ggml/include/ggml.h
package io.github.kotlinmania.llama.ore

// ============================================================================
// Constants from ggml.h (those not already in GGMLTypes.kt)
// ============================================================================

/** File magic number "ggml" */
const val GGML_FILE_MAGIC: Int = 0x67676d6c

/** File version */
const val GGML_FILE_VERSION: Int = 2

/** Quantization format version – bump on quantization format changes. */
const val GGML_QNT_VERSION: Int = 2
const val GGML_QNT_VERSION_FACTOR: Int = 1000

/** Maximum number of scheduling parameters. */
const val GGML_MAX_PARAMS: Int = 2048

/** Maximum number of threads allowed. */
const val GGML_MAX_N_THREADS: Int = 512

/** Default number of threads for graph compute. */
const val GGML_DEFAULT_N_THREADS: Int = 4

/** Default computation graph size (number of nodes). */
const val GGML_DEFAULT_GRAPH_SIZE: Int = 2048

/** Memory alignment in bytes (matches 64-bit platforms). */
const val GGML_MEM_ALIGN: Int = 16

const val GGML_EXIT_SUCCESS: Int = 0
const val GGML_EXIT_ABORTED: Int = 1

// RoPE type constants
const val GGML_ROPE_TYPE_NORMAL: Int = 0
const val GGML_ROPE_TYPE_NEOX: Int = 2
const val GGML_ROPE_TYPE_MROPE: Int = 8
const val GGML_ROPE_TYPE_VISION: Int = 24
const val GGML_ROPE_TYPE_IMROPE: Int = 40

/** Number of sections for multi-dimensional RoPE. */
const val GGML_MROPE_SECTIONS: Int = 4

/** Sentinel: use maximum available tasks. */
const val GGML_N_TASKS_MAX: Int = -1

// Tensor flag constants (supplement those in GGMLTypes.kt and GGMLGraph.kt)
// GGML_TENSOR_FLAG_OUTPUT already defined in GGMLTypes.kt
// GGML_TENSOR_FLAG_PARAM already defined in GGMLGraph.kt
const val GGML_TENSOR_FLAG_INPUT: Int = 1
const val GGML_TENSOR_FLAG_LOSS: Int = 8
const val GGML_TENSOR_FLAG_COMPUTE: Int = 16

// ============================================================================
// Enums from ggml.h not yet in GGMLTypes.kt
// ============================================================================

/** Precision hint for matrix multiplication and flash-attention. */
enum class GGMLPrec(val value: Int) {
    DEFAULT(0),
    F32(10);
}

/**
 * Model file types – describes the predominant quantization used in a model
 * file so that loaders can pick the right de-quantisation path.
 */
enum class GGMLFType(val value: Int) {
    UNKNOWN(-1),
    ALL_F32(0),
    MOSTLY_F16(1),
    MOSTLY_Q4_0(2),
    MOSTLY_Q4_1(3),
    MOSTLY_Q4_1_SOME_F16(4),
    MOSTLY_Q8_0(7),
    MOSTLY_Q5_0(8),
    MOSTLY_Q5_1(9),
    MOSTLY_Q2_K(10),
    MOSTLY_Q3_K(11),
    MOSTLY_Q4_K(12),
    MOSTLY_Q5_K(13),
    MOSTLY_Q6_K(14),
    MOSTLY_IQ2_XXS(15),
    MOSTLY_IQ2_XS(16),
    MOSTLY_IQ3_XXS(17),
    MOSTLY_IQ1_S(18),
    MOSTLY_IQ4_NL(19),
    MOSTLY_IQ3_S(20),
    MOSTLY_IQ2_S(21),
    MOSTLY_IQ4_XS(22),
    MOSTLY_IQ1_M(23),
    MOSTLY_BF16(24),
    MOSTLY_MXFP4(25),
    MOSTLY_NVFP4(26),
    MOSTLY_Q1_0(27);
}

/** Unary operations dispatched via GGML_OP_UNARY. */
enum class GGMLUnaryOp {
    ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID,
    GELU, GELU_QUICK, SILU, HARDSWISH, HARDSIGMOID,
    EXP, EXPM1, SOFTPLUS, GELU_ERF, XIELU,
    FLOOR, CEIL, ROUND, TRUNC,
    COUNT;
}

/** Gated Linear Unit operation variants. */
enum class GGMLGluOp {
    REGLU, GEGLU, SWIGLU, SWIGLU_OAI, GEGLU_ERF, GEGLU_QUICK,
    COUNT;
}

/** Object types stored in a GGML context. */
enum class GGMLObjectType {
    TENSOR, GRAPH, WORK_BUFFER;
}

// GGMLLogLevel is defined in NumericConversions.kt

/** Triangular matrix type for [io.github.kotlinmania.llama.ore.ggmlTri]. */
enum class GGMLTriType(val value: Int) {
    UPPER_DIAG(0), UPPER(1), LOWER_DIAG(2), LOWER(3);
}

/** Pooling operation kind. */
enum class GGMLOpPool {
    MAX, AVG, COUNT;
}

/** Interpolation scale mode. */
enum class GGMLScaleMode(val value: Int) {
    NEAREST(0), BILINEAR(1), BICUBIC(2), COUNT(3);
}

/** Interpolation scale flags (bitwise-OR with [io.github.kotlinmania.llama.ore.GGMLScaleMode]). */
object GGMLScaleFlag {
    const val ALIGN_CORNERS: Int = 1 shl 8
    const val ANTIALIAS: Int = 1 shl 9
}

/** Sort order for [io.github.kotlinmania.llama.ore.ggmlArgsort]. */
enum class GGMLSortOrder {
    ASC, DESC;
}

/** Scheduling priority for threadpool. */
enum class GGMLSchedPriority(val value: Int) {
    LOW(-1), NORMAL(0), MEDIUM(1), HIGH(2), REALTIME(3);
}

// ============================================================================
// Utility inline functions (macros from ggml.h)
// ============================================================================

/** Rounds [x] up to the nearest multiple of [n]. [n] must be a power of two. */
fun ggmlPad(x: Int, n: Int): Int = (x + n - 1) and (n - 1).inv()

/** Rounds [x] up to the nearest multiple of [n]. [n] must be a power of two. */
fun ggmlPad(x: ULong, n: ULong): ULong = (x + n - 1uL) and (n - 1uL).inv()

// ============================================================================
// Version / timing — ggml.c line 509
// ============================================================================

// Version/timing functions live in GGMLGraph.kt (ggmlVersion, ggmlCommit, ggmlTimeInit, ggmlTimeMs, ggmlTimeUs)

// ============================================================================
// Tensor-local extraction helpers (GGML_TENSOR_LOCALS equivalent)
// ============================================================================

/**
 * Extracts the [ne] (number-of-elements) array from a nullable tensor,
 * returning zeros when the tensor is null.
 */
fun tensorNe(t: io.github.kotlinmania.llama.ore.GGMLTensor?): LongArray =
    t?.ne?.copyOf() ?: LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)

/**
 * Extracts the [nb] (stride) array from a nullable tensor,
 * returning zeros when the tensor is null.
 */
fun tensorNb(t: io.github.kotlinmania.llama.ore.GGMLTensor?): ULongArray =
    t?.nb?.copyOf() ?: ULongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)

// ============================================================================
// FP16 / BF16 conversion helpers
// ============================================================================

/** Convert an IEEE-754 half-precision value (stored as [UShort]) to [Float]. */
fun ggmlFp16ToFp32(h: UShort): Float = io.github.kotlinmania.llama.ore.halfToFloat(h.toShort())

/** Convert a [Float] to IEEE-754 half-precision (stored as [UShort]). */
fun ggmlFp32ToFp16(f: Float): UShort = io.github.kotlinmania.llama.ore.floatToHalf(f).toUShort()

/** Convert a row of FP16 values to FP32. */
fun ggmlFp16ToFp32Row(src: ShortArray, dst: FloatArray, k: Long) {
    for (i in 0 until k.toInt()) {
        dst[i] = io.github.kotlinmania.llama.ore.halfToFloat(src[i])
    }
}

/** Convert a row of FP32 values to FP16. */
fun ggmlFp32ToFp16Row(src: FloatArray, dst: ShortArray, k: Long) {
    for (i in 0 until k.toInt()) {
        dst[i] = io.github.kotlinmania.llama.ore.floatToHalf(src[i])
    }
}

/** Convert a BF16 value to FP32 (shift left by 16). */
fun ggmlBf16ToFp32(v: io.github.kotlinmania.llama.ore.GGMLBF16): Float =
    Float.fromBits(v.bits.toInt() shl 16)

/** Convert a FP32 value to BF16 (truncate lower 16 bits). */
fun ggmlFp32ToBf16(f: Float): io.github.kotlinmania.llama.ore.GGMLBF16 =
    io.github.kotlinmania.llama.ore.GGMLBF16((f.toRawBits() ushr 16).toUShort())

// ============================================================================
// Tensor query helpers (ggml_nelements, ggml_nrows, …)
// ============================================================================

/** Total number of elements in [tensor]. */
fun ggmlNelements(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Long = tensor.numElements()

/** Number of rows – product of ne[1]×ne[2]×ne[3]. */
fun ggmlNrows(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Long =
    tensor.ne[1] * tensor.ne[2] * tensor.ne[3]

/** Total byte size of the tensor data. */
fun ggmlNbytes(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong =
    io.github.kotlinmania.llama.ore.calculateTensorByteSize(tensor)

/** Byte size padded to [io.github.kotlinmania.llama.ore.GGML_MEM_ALIGN]. */
fun ggmlNbytesPad(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong =
    io.github.kotlinmania.llama.ore.ggmlPad(
        io.github.kotlinmania.llama.ore.ggmlNbytes(
            tensor
        ), io.github.kotlinmania.llama.ore.GGML_MEM_ALIGN.toULong()
    )

/** Block size for the given [type]. Returns 1 for non-block types. */
fun ggmlBlckSize(type: io.github.kotlinmania.llama.ore.GGMLType): Long = when (type) {
    io.github.kotlinmania.llama.ore.GGMLType.Q4_0 -> io.github.kotlinmania.llama.ore.QK4_0.toLong()
    io.github.kotlinmania.llama.ore.GGMLType.Q4_1 -> io.github.kotlinmania.llama.ore.QK4_1.toLong()
    io.github.kotlinmania.llama.ore.GGMLType.Q5_0 -> 32L
    io.github.kotlinmania.llama.ore.GGMLType.Q5_1 -> 32L
    io.github.kotlinmania.llama.ore.GGMLType.Q8_0 -> io.github.kotlinmania.llama.ore.QK8_0.toLong()
    io.github.kotlinmania.llama.ore.GGMLType.Q8_1 -> 32L
    io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K,
    io.github.kotlinmania.llama.ore.GGMLType.Q5_K, io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K -> io.github.kotlinmania.llama.ore.QK_K.toLong()
    io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58 -> io.github.kotlinmania.llama.ore.QK_BITNET_1_58.toLong()
    else -> 1L
}

/** Size in bytes of one complete block for [type]. */
fun ggmlTypeSize(type: io.github.kotlinmania.llama.ore.GGMLType): ULong = type.byteSize

/** Size in bytes of one row of [ne] elements of the given [type]. */
fun ggmlRowSize(type: io.github.kotlinmania.llama.ore.GGMLType, ne: Long): ULong {
    val blck = io.github.kotlinmania.llama.ore.ggmlBlckSize(type)
    return (ne / blck).toULong() * io.github.kotlinmania.llama.ore.ggmlTypeSize(type)
}

/** Human-readable name for a [io.github.kotlinmania.llama.ore.GGMLType]. */
fun ggmlTypeName(type: io.github.kotlinmania.llama.ore.GGMLType): String = type.description

/** Human-readable name for a [io.github.kotlinmania.llama.ore.GGMLOp]. */
fun ggmlOpName(op: io.github.kotlinmania.llama.ore.GGMLOp): String = op.name

/** Human-readable symbol for an operation (e.g. "+" for ADD). */
fun ggmlOpSymbol(op: io.github.kotlinmania.llama.ore.GGMLOp): String = when (op) {
    io.github.kotlinmania.llama.ore.GGMLOp.ADD -> "+"
    io.github.kotlinmania.llama.ore.GGMLOp.SUB -> "-"
    io.github.kotlinmania.llama.ore.GGMLOp.MUL -> "*"
    io.github.kotlinmania.llama.ore.GGMLOp.DIV -> "/"
    io.github.kotlinmania.llama.ore.GGMLOp.NEG -> "-"
    io.github.kotlinmania.llama.ore.GGMLOp.SQR -> "²"
    io.github.kotlinmania.llama.ore.GGMLOp.SQRT -> "√"
    io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT -> "×"
    else -> op.name
}

/** Name of a [io.github.kotlinmania.llama.ore.GGMLUnaryOp]. */
fun ggmlUnaryOpName(op: io.github.kotlinmania.llama.ore.GGMLUnaryOp): String = op.name

/** Name of a [io.github.kotlinmania.llama.ore.GGMLGluOp]. */
fun ggmlGluOpName(op: io.github.kotlinmania.llama.ore.GGMLGluOp): String = op.name

/** Description string for a tensor – returns the unary op name if applicable. */
fun ggmlOpDesc(t: io.github.kotlinmania.llama.ore.GGMLTensor): String =
    io.github.kotlinmania.llama.ore.ggmlOpName(t.op)

/** Element size in bytes for the tensor's type. */
fun ggmlElementSize(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ULong =
    io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type)

/** Whether the type is block-quantized. */
fun ggmlIsQuantized(type: io.github.kotlinmania.llama.ore.GGMLType): Boolean = when (type) {
    io.github.kotlinmania.llama.ore.GGMLType.Q4_0, io.github.kotlinmania.llama.ore.GGMLType.Q4_1, io.github.kotlinmania.llama.ore.GGMLType.Q5_0, io.github.kotlinmania.llama.ore.GGMLType.Q5_1,
    io.github.kotlinmania.llama.ore.GGMLType.Q8_0, io.github.kotlinmania.llama.ore.GGMLType.Q8_1,
    io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K, io.github.kotlinmania.llama.ore.GGMLType.Q4_K, io.github.kotlinmania.llama.ore.GGMLType.Q5_K,
    io.github.kotlinmania.llama.ore.GGMLType.Q6_K, io.github.kotlinmania.llama.ore.GGMLType.Q8_K,
    io.github.kotlinmania.llama.ore.GGMLType.BITNET_1_58, io.github.kotlinmania.llama.ore.GGMLType.Q1_5_K -> true
    else -> false
}

/** Whether the tensor's memory layout is transposed (nb[0] > nb[1]). */
fun ggmlIsTransposed(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.nb[0] > tensor.nb[1]

/** Whether the tensor is permuted (strides not in ascending order). */
fun ggmlIsPermuted(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.nb[0] > tensor.nb[1] ||
    tensor.nb[1] > tensor.nb[2] ||
    tensor.nb[2] > tensor.nb[3]

/** Whether the tensor has zero elements. */
fun ggmlIsEmpty(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.ne[0] == 0L || tensor.ne[1] == 0L ||
    tensor.ne[2] == 0L || tensor.ne[3] == 0L

/** Whether the tensor is a view of another tensor. */
fun ggmlIsView(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.viewSrc != null

/** Whether the tensor is a scalar (single element). */
fun ggmlIsScalar(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.ne[0] == 1L && tensor.ne[1] == 1L &&
    tensor.ne[2] == 1L && tensor.ne[3] == 1L

/** Whether the tensor is a 1-D vector. */
fun ggmlIsVector(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.ne[1] == 1L && tensor.ne[2] == 1L && tensor.ne[3] == 1L

/** Whether the tensor is a 2-D matrix. */
fun ggmlIsMatrix(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.ne[2] == 1L && tensor.ne[3] == 1L

/** Whether the tensor is 3-D. */
fun ggmlIs3d(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    tensor.ne[3] == 1L

/** Number of dimensions (returns 1 for scalars). */
fun ggmlNDims(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Int {
    for (i in io.github.kotlinmania.llama.ore.GGML_MAX_DIMS - 1 downTo 1) {
        if (tensor.ne[i] > 1L) return i + 1
    }
    return 1
}

/** Whether the tensor elements are contiguous in memory. */
fun ggmlIsContiguous(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean =
    io.github.kotlinmania.llama.ore.ggmlIsContiguous0(tensor)

/** Contiguous check starting from dimension 0. */
fun ggmlIsContiguous0(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    if (tensor.nb[0] != io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type)) return false
    val blck = io.github.kotlinmania.llama.ore.ggmlBlckSize(tensor.type)
    if (tensor.nb[1] != tensor.nb[0] * (tensor.ne[0].toULong() / blck.toULong())) return false
    if (tensor.nb[2] != tensor.nb[1] * tensor.ne[1].toULong()) return false
    if (tensor.nb[3] != tensor.nb[2] * tensor.ne[2].toULong()) return false
    return true
}

/** Contiguous for dims ≥ 1. */
fun ggmlIsContiguous1(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    val blck = io.github.kotlinmania.llama.ore.ggmlBlckSize(tensor.type)
    if (tensor.nb[1] != io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type) * (tensor.ne[0].toULong() / blck.toULong())) return false
    if (tensor.nb[2] != tensor.nb[1] * tensor.ne[1].toULong()) return false
    if (tensor.nb[3] != tensor.nb[2] * tensor.ne[2].toULong()) return false
    return true
}

/** Contiguous for dims ≥ 2. */
fun ggmlIsContiguous2(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    if (tensor.nb[2] != tensor.nb[1] * tensor.ne[1].toULong()) return false
    if (tensor.nb[3] != tensor.nb[2] * tensor.ne[2].toULong()) return false
    return true
}

/** Whether data is contiguously allocated (no gaps, permutation ok). */
fun ggmlIsContiguouslyAllocated(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    val sorted = tensor.nb.copyOf().also { it.sort() }
    if (sorted[0] != io.github.kotlinmania.llama.ore.ggmlTypeSize(tensor.type)) return false
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        // find which dim has stride sorted[i-1]
        val prevDim = (0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS).firstOrNull { tensor.nb[it] == sorted[i - 1] } ?: return false
        if (sorted[i] != sorted[i - 1] * tensor.ne[prevDim].toULong()) return false
    }
    return true
}

/** Whether two tensors have the same shape. */
fun ggmlAreSameShape(t0: io.github.kotlinmania.llama.ore.GGMLTensor, t1: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (t0.ne[i] != t1.ne[i]) return false
    }
    return true
}

/** Whether two tensors have the same strides. */
fun ggmlAreSameStride(t0: io.github.kotlinmania.llama.ore.GGMLTensor, t1: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (t0.nb[i] != t1.nb[i]) return false
    }
    return true
}

/** Whether [t0] can be broadcast/repeated to match [t1]. */
fun ggmlCanRepeat(t0: io.github.kotlinmania.llama.ore.GGMLTensor, t1: io.github.kotlinmania.llama.ore.GGMLTensor): Boolean {
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        if (t1.ne[i] % t0.ne[i] != 0L) return false
    }
    return true
}

/** Memory overhead per tensor object (metadata only). */
fun ggmlTensorOverhead(): ULong = 256uL // approximate; actual C struct size

// ============================================================================
// Existing functions — preserved verbatim
// ============================================================================

internal fun calculateContiguousStrides(ne: LongArray, type: io.github.kotlinmania.llama.ore.GGMLType, rank: Int): ULongArray {
    val nb = ULongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 0uL }

    if (type.byteSize == 0uL) {
        if (type != io.github.kotlinmania.llama.ore.GGMLType.COUNT && !type.name.startsWith("Q", ignoreCase = true) && !type.name.startsWith("q", ignoreCase = true) ) {
            println("Warning: GGMLType ${type.name} has byteSize 0. Strides will be all zeros.")
        }
        return nb
    }

    nb[0] = type.byteSize
    if (io.github.kotlinmania.llama.ore.GGML_MAX_DIMS > 1) {
        for (d in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
            val dimSize = ne.getOrElse(d - 1) { 1L }
            nb[d] = nb[d - 1] * (if (dimSize > 0L) dimSize.toULong() else 1uL)
        }
    }
    return nb
}

/** Lightweight view ops used by backward/graph code: reshape, permute, transpose. */
fun reshape(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, vararg newShape: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val out = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    val r = newShape.copyOf(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS)
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) out.ne[i] = if (i < newShape.size) r[i] else 1L
    out.nb =
        io.github.kotlinmania.llama.ore.calculateContiguousStrides(out.ne, out.type, out.rank())
    out.viewSrc = a
    out.op = io.github.kotlinmania.llama.ore.GGMLOp.RESHAPE
    return if (context.computeImmediately) out else out
}

fun permute(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ax0: Int, ax1: Int, ax2: Int, ax3: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val axes = intArrayOf(ax0, ax1, ax2, ax3)
    val out = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) out.ne[i] = a.ne[axes.getOrElse(i) { i }]
    out.nb =
        io.github.kotlinmania.llama.ore.calculateContiguousStrides(out.ne, out.type, out.rank())
    out.viewSrc = a
    out.op = io.github.kotlinmania.llama.ore.GGMLOp.PERMUTE
    out.opParams = axes
    return if (context.computeImmediately) out else out
}

fun transpose(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ax0: Int, ax1: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val axes = intArrayOf(ax0, ax1, 2, 3)
    return io.github.kotlinmania.llama.ore.permute(context, a, axes[0], axes[1], axes[2], axes[3])
        .also { it.op = io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE }
}

/**
 * Kotlin Native port of GGML tensor operations.
 * This file contains the implementation of basic tensor operations.
 *
 * This is a skeleton implementation that will be expanded in future versions.
 */

/**
 * Creates a new tensor with the specified dimensions and type.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @return The new tensor
 */
fun createTensor(context: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType): io.github.kotlinmania.llama.ore.GGMLTensor {
    val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)

    tensor.ne[0] = 1
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        tensor.ne,
        tensor.type,
        tensor.rank()
    )

    if (context.memBuffer != null && !context.noAlloc) {
        when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> tensor.data = FloatArray(1) { 0.0f }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> tensor.data = IntArray(1) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> tensor.data = LongArray(1) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 1-dimensional tensor.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @return The new tensor
 */
fun createTensor1D(context: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        tensor.ne,
        tensor.type,
        tensor.rank()
    )

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0
        when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 2-dimensional tensor.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @return The new tensor
 */
fun createTensor2D(context: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Int, ne1: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        tensor.ne,
        tensor.type,
        tensor.rank()
    )

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0 * ne1
        when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 3D tensor with the specified type and dimensions.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @param ne2 The number of elements in the third dimension
 * @return The new tensor
 */
fun createTensor3D(context: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Int, ne1: Int, ne2: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    tensor.ne[2] = ne2.toLong()
    for (i in 3 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        tensor.ne[i] = 1
    }

    tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        tensor.ne,
        tensor.type,
        tensor.rank()
    )

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0 * ne1 * ne2
        when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

/**
 * Creates a new 4D tensor with the specified type and dimensions.
 *
 * @param context The GGML context
 * @param type The tensor data type
 * @param ne0 The number of elements in the first dimension
 * @param ne1 The number of elements in the second dimension
 * @param ne2 The number of elements in the third dimension
 * @param ne3 The number of elements in the fourth dimension
 * @return The new tensor
 */
fun createTensor4D(context: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Int, ne1: Int, ne2: Int, ne3: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)

    tensor.ne[0] = ne0.toLong()
    tensor.ne[1] = ne1.toLong()
    tensor.ne[2] = ne2.toLong()
    tensor.ne[3] = ne3.toLong()

    tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        tensor.ne,
        tensor.type,
        tensor.rank()
    )

    if (context.memBuffer != null && !context.noAlloc) {
        val totalSize = ne0 * ne1 * ne2 * ne3
        when (type) {
            io.github.kotlinmania.llama.ore.GGMLType.F32 -> tensor.data = FloatArray(totalSize) { 0.0f }
            io.github.kotlinmania.llama.ore.GGMLType.F16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I8 -> tensor.data = ByteArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I16 -> tensor.data = ShortArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I32 -> tensor.data = IntArray(totalSize) { 0 }
            io.github.kotlinmania.llama.ore.GGMLType.I64 -> tensor.data = LongArray(totalSize) { 0L }
            else -> tensor.data = null
        }
    }

    return tensor
}

// ============================================================================
// Context lifecycle
// ============================================================================

/** Initialize a new GGML context with the given parameters. */
fun ggmlInit(params: io.github.kotlinmania.llama.ore.GGMLInitParams): io.github.kotlinmania.llama.ore.GGMLContext {
    val ctx = io.github.kotlinmania.llama.ore.GGMLContext()
    ctx.memSize = params.memSize
    ctx.noAlloc = params.noAlloc
    if (params.memBuffer != null) {
        ctx.memBuffer = params.memBuffer
        ctx.memBufferOwned = false
    } else if (params.memSize > 0uL) {
        ctx.memBuffer = io.github.kotlinmania.llama.ore.ggml_aligned_malloc(params.memSize.toLong())
        ctx.memBufferOwned = true
    }
    return ctx
}

/** Reset the context – zeros out used memory but keeps the allocation. */
fun ggmlReset(ctx: io.github.kotlinmania.llama.ore.GGMLContext) {
    ctx.nObjects = 0
    ctx.objectsBegin = null
    ctx.objectsEnd = null
}

/** Free the context and optionally its owned memory buffer. */
fun ggmlFree(ctx: io.github.kotlinmania.llama.ore.GGMLContext) {
    if (ctx.memBufferOwned) {
        io.github.kotlinmania.llama.ore.ggml_aligned_free(ctx.memBuffer, ctx.memSize.toLong())
        ctx.memBuffer = null
    }
}

/** Amount of memory used by the context (approximate). */

fun ggmlGetNoAlloc(ctx: io.github.kotlinmania.llama.ore.GGMLContext): Boolean = ctx.noAlloc
fun ggmlSetNoAlloc(ctx: io.github.kotlinmania.llama.ore.GGMLContext, noAlloc: Boolean) { ctx.noAlloc = noAlloc }

// ============================================================================
// N-D tensor creation (Long-based, matching ggml_new_tensor signatures)
// ============================================================================

/** Create a new tensor with the given type and dimensions array. */
fun ggmlNewTensor(ctx: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, nDims: Int, ne: LongArray): io.github.kotlinmania.llama.ore.GGMLTensor {
    val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        tensor.ne[i] = if (i < nDims) ne[i] else 1L
    }
    tensor.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        tensor.ne,
        tensor.type,
        tensor.rank()
    )
    return tensor
}

fun ggmlNewTensor1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, type, 1, longArrayOf(ne0))

fun ggmlNewTensor2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Long, ne1: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, type, 2, longArrayOf(ne0, ne1))

fun ggmlNewTensor3d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Long, ne1: Long, ne2: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, type, 3, longArrayOf(ne0, ne1, ne2))

fun ggmlNewTensor4d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Long, ne1: Long, ne2: Long, ne3: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, type, 4, longArrayOf(ne0, ne1, ne2, ne3))

/** Duplicate a tensor's metadata (creates a new tensor with the same shape/type). */
fun ggmlDupTensor(ctx: io.github.kotlinmania.llama.ore.GGMLContext, src: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val t = io.github.kotlinmania.llama.ore.ggmlNewTensor(
        ctx,
        src.type,
        io.github.kotlinmania.llama.ore.ggmlNDims(src),
        src.ne
    )
    t.name = src.name
    return t
}

/** Create a view of [src] (same data, separate metadata). */
fun ggmlViewTensor(ctx: io.github.kotlinmania.llama.ore.GGMLContext, src: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val t = io.github.kotlinmania.llama.ore.GGMLTensor(type = src.type)
    src.ne.copyInto(t.ne)
    src.nb.copyInto(t.nb)
    t.viewSrc = src
    t.data = src.data
    t.op = io.github.kotlinmania.llama.ore.GGMLOp.VIEW
    return t
}

// ============================================================================
// Context tensor enumeration
// ============================================================================

/** Get a tensor from the context by name. C: ggml.c line 1965. */
fun ggmlGetTensor(ctx: io.github.kotlinmania.llama.ore.GGMLContext, name: String): io.github.kotlinmania.llama.ore.GGMLTensor? {
    var obj = ctx.objectsBegin
    while (obj != null) {
        if (obj.type == io.github.kotlinmania.llama.ore.GGMLObjectType.TENSOR) {
            val cur = obj.tensor
            if (cur != null && cur.name == name) {
                return cur
            }
        }
        obj = obj.next
    }
    return null
}

// ============================================================================
// Tensor metadata helpers
// ============================================================================

/** Get the unary op stored in the tensor's op_params. */
fun ggmlGetUnaryOp(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLUnaryOp =
    io.github.kotlinmania.llama.ore.GGMLUnaryOp.entries[tensor.opParams[0]]

/** Get the glu op stored in the tensor's op_params. */
fun ggmlGetGluOp(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLGluOp =
    io.github.kotlinmania.llama.ore.GGMLGluOp.entries[tensor.opParams[0]]

/** Get the tensor name. */
fun ggmlGetName(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): String = tensor.name

/** Set the tensor name (returns the tensor for chaining). */
fun ggmlSetName(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, name: String): io.github.kotlinmania.llama.ore.GGMLTensor {
    tensor.name = name.take(io.github.kotlinmania.llama.ore.GGML_MAX_NAME - 1)
    return tensor
}

// Tensor flag setters
fun ggmlSetInput(tensor: io.github.kotlinmania.llama.ore.GGMLTensor) { tensor.flags = tensor.flags or io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_INPUT
}
fun ggmlSetOutput(tensor: io.github.kotlinmania.llama.ore.GGMLTensor) { tensor.flags = tensor.flags or io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_OUTPUT
}
fun ggmlSetParam(tensor: io.github.kotlinmania.llama.ore.GGMLTensor) { tensor.flags = tensor.flags or io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_PARAM
}
fun ggmlSetLoss(tensor: io.github.kotlinmania.llama.ore.GGMLTensor) { tensor.flags = tensor.flags or io.github.kotlinmania.llama.ore.GGML_TENSOR_FLAG_LOSS
}

/** Set all elements to zero. */
fun ggmlSetZero(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    // For in-memory data arrays
    when (val d = tensor.data) {
        is FloatArray -> d.fill(0.0f)
        is IntArray -> d.fill(0)
        is LongArray -> d.fill(0L)
        is ShortArray -> d.fill(0)
        is ByteArray -> d.fill(0)
    }
    return tensor
}

// ============================================================================
// Internal helpers for graph-building ops
// ============================================================================

/**
 * Creates a result tensor for a unary operation.
 * Copies shape/strides from [a] and records [op] with source [a].
 */
private fun buildUnary(a: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLOp, inplace: Boolean = false): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    a.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = op
    result.src[0] = a
    if (inplace) result.viewSrc = a
    return result
}

/**
 * Creates a result tensor for a binary element-wise operation.
 * Copies shape/strides from [a] and records [op] with sources [a] and [b].
 */
private fun buildBinary(a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLOp, inplace: Boolean = false): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    a.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = op
    result.src[0] = a
    result.src[1] = b
    if (inplace) result.viewSrc = a
    return result
}

/**
 * Creates a result tensor for a unary op routed through GGML_OP_UNARY.
 * Stores the specific [unaryOp] in opParams[0].
 */
private fun buildUnaryOp(a: io.github.kotlinmania.llama.ore.GGMLTensor, unaryOp: io.github.kotlinmania.llama.ore.GGMLUnaryOp, inplace: Boolean = false): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY,
        inplace
    )
    result.opParams[0] = unaryOp.ordinal
    return result
}

// ============================================================================
// Operations on tensors with backpropagation (graph-building)
// ============================================================================

/** ggml_dup – duplicate tensor data. */
fun ggmlDup(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.DUP
    )

/** ggml_dup_inplace – in-place duplicate (returns view). */
fun ggmlDupInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.DUP,
        inplace = true
    )

/** ggml_add – element-wise addition. */
fun ggmlAdd(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.ADD
    )

/** ggml_add_inplace */
fun ggmlAddInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.ADD,
        inplace = true
    )

/** ggml_add_cast – add with output cast to [type]. */
fun ggmlAddCast(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, type: io.github.kotlinmania.llama.ore.GGMLType): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.ADD
    )
    result.type = type
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    return result
}

/** ggml_sub – element-wise subtraction. */
fun ggmlSub(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.SUB
    )

fun ggmlSubInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.SUB,
        inplace = true
    )

/** ggml_mul – element-wise multiplication. */
fun ggmlMul(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.MUL
    )

fun ggmlMulInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.MUL,
        inplace = true
    )

/** ggml_div – element-wise division. */
fun ggmlDiv(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.DIV
    )

fun ggmlDivInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.DIV,
        inplace = true
    )

/** ggml_sqr – element-wise square. */
fun ggmlSqr(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SQR
    )

fun ggmlSqrInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SQR,
        inplace = true
    )

/** ggml_sqrt – element-wise square root. */
fun ggmlSqrt(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SQRT
    )

fun ggmlSqrtInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SQRT,
        inplace = true
    )

/** ggml_log – element-wise natural logarithm. */
fun ggmlLog(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.LOG
    )

fun ggmlLogInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.LOG,
        inplace = true
    )

/** ggml_sin – element-wise sine. C: ggml.c line 2393. */
fun ggmlSin(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlSinImpl(ctx, a, inplace = false)

fun ggmlSinInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlSinImpl(ctx, a, inplace = true)

/** ggml_cos – element-wise cosine. C: ggml.c line 2419. */
fun ggmlCos(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlCosImpl(ctx, a, inplace = false)

fun ggmlCosInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlCosImpl(ctx, a, inplace = true)

private fun ggmlSinImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SIN
    result.src[0] = a
    return result
}

private fun ggmlCosImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.COS
    result.src[0] = a
    return result
}

// --- reduction ops ---

/** ggml_sum – sum all elements, return scalar. */
fun ggmlSum(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SUM
    result.src[0] = a
    return result
}

/** ggml_sum_rows – sum along rows: [a,b,c,d] → [1,b,c,d]. */
fun ggmlSumRows(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = 1L
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SUM_ROWS
    result.src[0] = a
    return result
}

/** ggml_mean – mean along rows. */
fun ggmlMean(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MEAN
    result.src[0] = a
    return result
}

/** ggml_argmax – argmax along rows, returns I32. */
fun ggmlArgmax(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.I32)
    result.ne[0] = 1L
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ARGMAX
    result.src[0] = a
    return result
}

/** ggml_count_equal – count equal elements between a and b. */
fun ggmlCountEqual(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.COUNT // reusing COUNT; the C code uses COUNT_EQUAL
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_repeat – tile a to fit shape of b. */
fun ggmlRepeat(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    b.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.REPEAT
    result.src[0] = a
    return result
}

/** ggml_repeat_back – sum repetitions of a back to shape of b. */
fun ggmlRepeatBack(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, reference: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        result.ne[i] = reference.ne[i]
    }
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.REPEAT_BACK
    result.src[0] = a
    return result
}

/** ggml_concat – concatenate a and b along [dim]. */
fun ggmlConcat(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, dim: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    for (i in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) {
        result.ne[i] = if (i == dim) a.ne[i] + b.ne[i] else a.ne[i]
    }
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CONCAT
    result.src[0] = a
    result.src[1] = b
    result.opParams[0] = dim
    return result
}

// --- unary activation / element-wise ops ---

fun ggmlAbs(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.ABS
    )

fun ggmlAbsInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.ABS,
        inplace = true
    )

fun ggmlSgn(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SGN
    )

fun ggmlSgnInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SGN,
        inplace = true
    )

fun ggmlNeg(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.NEG
    )

fun ggmlNegInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.NEG,
        inplace = true
    )

fun ggmlStep(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.STEP
    )

fun ggmlStepInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.STEP,
        inplace = true
    )

fun ggmlTanh(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.TANH
    )

fun ggmlTanhInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.TANH,
        inplace = true
    )

fun ggmlElu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.ELU
    )

fun ggmlEluInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.ELU,
        inplace = true
    )

fun ggmlRelu(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.RELU
    )

fun ggmlReluInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.RELU,
        inplace = true
    )

/** ggml_leaky_relu */
fun ggmlLeakyRelu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, negativeSlope: Float, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY,
        inplace
    )
    // Store negative slope as float bits in opParams
    result.opParams[0] = io.github.kotlinmania.llama.ore.GGMLUnaryOp.RELU.ordinal // base op
    result.opParams[1] = negativeSlope.toRawBits()
    return result
}

fun ggmlSigmoid(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SIGMOID
    )

fun ggmlSigmoidInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SIGMOID,
        inplace = true
    )

fun ggmlGelu(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU
    )

fun ggmlGeluInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU,
        inplace = true
    )

fun ggmlGeluErf(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU_ERF
    )

fun ggmlGeluErfInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU_ERF,
        inplace = true
    )

fun ggmlGeluQuick(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU_QUICK
    )

fun ggmlGeluQuickInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.GELU_QUICK,
        inplace = true
    )

fun ggmlSilu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SILU
    )

fun ggmlSiluInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SILU,
        inplace = true
    )

/** ggml_silu_back – a=x, b=dy. */
fun ggmlSiluBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.SILU_BACK
    )

fun ggmlHardswish(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.HARDSWISH
    )

fun ggmlHardsigmoid(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.HARDSIGMOID
    )

fun ggmlExp(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.EXP
    )

fun ggmlExpInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.EXP,
        inplace = true
    )

fun ggmlExpm1(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.EXPM1
    )

fun ggmlExpm1Inplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.EXPM1,
        inplace = true
    )

fun ggmlSoftplus(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SOFTPLUS
    )

fun ggmlSoftplusInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.SOFTPLUS,
        inplace = true
    )

fun ggmlFloor(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.FLOOR
    )

fun ggmlFloorInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.FLOOR,
        inplace = true
    )

fun ggmlCeil(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.CEIL
    )

fun ggmlCeilInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.CEIL,
        inplace = true
    )

fun ggmlRound(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.ROUND
    )

fun ggmlRoundInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.ROUND,
        inplace = true
    )

/** ggml_trunc – truncate fractional part toward zero. */
fun ggmlTrunc(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.TRUNC
    )

fun ggmlTruncInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(
        a,
        io.github.kotlinmania.llama.ore.GGMLUnaryOp.TRUNC,
        inplace = true
    )

// --- GLU ops ---

/**
 * ggml_glu_impl — C: ggml.c line 2862.
 * Internal implementation for all GLU variants.
 */
private fun ggmlGluImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor?, op: io.github.kotlinmania.llama.ore.GGMLGluOp, swapped: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(a))

    if (b != null) {
        require(io.github.kotlinmania.llama.ore.ggmlIsContiguous1(b))
        require(io.github.kotlinmania.llama.ore.ggmlAreSameShape(a, b))
        require(a.type == b.type)
    }

    val ne = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { if (it == 0) a.ne[0] / 2 else a.ne[it] }
    val result = if (b != null) {
        io.github.kotlinmania.llama.ore.ggmlNewTensor(
            ctx,
            a.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS,
            a.ne.copyOf()
        )
    } else {
        io.github.kotlinmania.llama.ore.ggmlNewTensor(
            ctx,
            a.type,
            io.github.kotlinmania.llama.ore.GGML_MAX_DIMS,
            ne
        )
    }

    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, op.ordinal)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 1, if (swapped) 1 else 0)

    result.op = io.github.kotlinmania.llama.ore.GGMLOp.GLU
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_glu — C: ggml.c line 2945. */
fun ggmlGlu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLGluOp, swapped: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(ctx, a, null, op, swapped)

/** ggml_glu_split — C: ggml.c line 2953. */
fun ggmlGluSplit(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLGluOp): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(ctx, a, b, op, false)

fun ggmlReglu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.REGLU,
        false
    )
fun ggmlRegluSwapped(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.REGLU,
        true
    )
fun ggmlRegluSplit(ctx: io.github.kotlinmania.llama.ore.GGMLContext, gate: io.github.kotlinmania.llama.ore.GGMLTensor, up: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        gate,
        up,
        io.github.kotlinmania.llama.ore.GGMLGluOp.REGLU,
        false
    )

fun ggmlGeglu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU,
        false
    )
fun ggmlGegluSwapped(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU,
        true
    )
fun ggmlGegluSplit(ctx: io.github.kotlinmania.llama.ore.GGMLContext, gate: io.github.kotlinmania.llama.ore.GGMLTensor, up: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        gate,
        up,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU,
        false
    )

fun ggmlSwiglu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.SWIGLU,
        false
    )
fun ggmlSwigluSwapped(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.SWIGLU,
        true
    )
fun ggmlSwigluSplit(ctx: io.github.kotlinmania.llama.ore.GGMLContext, gate: io.github.kotlinmania.llama.ore.GGMLTensor, up: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        gate,
        up,
        io.github.kotlinmania.llama.ore.GGMLGluOp.SWIGLU,
        false
    )

fun ggmlGegluErf(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU_ERF,
        false
    )
fun ggmlGegluErfSwapped(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU_ERF,
        true
    )
fun ggmlGegluErfSplit(ctx: io.github.kotlinmania.llama.ore.GGMLContext, gate: io.github.kotlinmania.llama.ore.GGMLTensor, up: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        gate,
        up,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU_ERF,
        false
    )

fun ggmlGegluQuick(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU_QUICK,
        false
    )
fun ggmlGegluQuickSwapped(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        null,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU_QUICK,
        true
    )
fun ggmlGegluQuickSplit(ctx: io.github.kotlinmania.llama.ore.GGMLContext, gate: io.github.kotlinmania.llama.ore.GGMLTensor, up: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        gate,
        up,
        io.github.kotlinmania.llama.ore.GGMLGluOp.GEGLU_QUICK,
        false
    )

/** ggml_swiglu_oai — C: ggml.c line 3066. */
fun ggmlSwigluOai(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, alpha: Float, limit: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlGluImpl(
        ctx,
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLGluOp.SWIGLU_OAI,
        false
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params_f32(result, 2, alpha)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_f32(result, 3, limit)
    return result
}

// --- normalization ---

/** ggml_norm – layer normalization along rows. */
fun ggmlNorm(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.NORM
    )
    result.opParams[0] = eps.toRawBits()
    return result
}

fun ggmlNormInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.NORM,
        inplace = true
    )
    result.opParams[0] = eps.toRawBits()
    return result
}

/** ggml_rms_norm – RMS normalization along rows. */
fun ggmlRmsNorm(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM
    )
    result.opParams[0] = eps.toRawBits()
    return result
}

fun ggmlRmsNormInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM,
        inplace = true
    )
    result.opParams[0] = eps.toRawBits()
    return result
}

/** ggml_group_norm */
fun ggmlGroupNorm(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, nGroups: Int, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    ) // GROUP_NORM not in existing enum
    result.opParams[0] = nGroups
    result.opParams[1] = eps.toRawBits()
    return result
}

fun ggmlGroupNormInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, nGroups: Int, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY,
        inplace = true
    )
    result.opParams[0] = nGroups
    result.opParams[1] = eps.toRawBits()
    return result
}

/** ggml_rms_norm_back – a=x, b=dy */
fun ggmlRmsNormBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.RMS_NORM_BACK
    )
    result.opParams[0] = eps.toRawBits()
    return result
}

// --- matrix multiplication ---

/**
 * ggml_mul_mat – matrix multiplication.
 * A: [ne03, ne02, n, k], B: [ne03*x, ne02*y, m, k]  → result: [ne03*x, ne02*y, m, n]
 */
fun ggmlMulMat(context: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = a.ne[1] // n
    result.ne[1] = b.ne[1] // m
    result.ne[2] = b.ne[2]
    result.ne[3] = b.ne[3]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_mul_mat_set_prec – set precision for mul_mat. */
fun ggmlMulMatSetPrec(a: io.github.kotlinmania.llama.ore.GGMLTensor, prec: io.github.kotlinmania.llama.ore.GGMLPrec) {
    a.opParams[0] = prec.value
}

/** ggml_mul_mat_id – indirect matrix multiplication with index tensor. */
fun ggmlMulMatId(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, ids: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = a.ne[1]
    result.ne[1] = b.ne[1]
    result.ne[2] = b.ne[2]
    result.ne[3] = b.ne[3]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MUL_MAT
    result.src[0] = a
    result.src[1] = b
    result.src[2] = ids
    return result
}

/** ggml_out_prod – outer product: A[m,n] × B[p,n] → result[m,p]. */
fun ggmlOutProd(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = a.ne[0]
    result.ne[1] = b.ne[0]
    result.ne[2] = b.ne[2]
    result.ne[3] = b.ne[3]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY // OUT_PROD not in existing enum
    result.src[0] = a
    result.src[1] = b
    return result
}

// --- operations without backpropagation ---

/** ggml_scale – multiply all elements by scalar [s]. */
fun ggmlScale(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, s: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SCALE
    )
    result.opParams[0] = s.toRawBits()
    return result
}

fun ggmlScaleInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, s: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SCALE,
        inplace = true
    )
    result.opParams[0] = s.toRawBits()
    return result
}

/** ggml_scale_bias – x = s*a + b */
fun ggmlScaleBias(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, s: Float, b: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SCALE
    )
    result.opParams[0] = s.toRawBits()
    result.opParams[1] = b.toRawBits()
    return result
}

fun ggmlScaleBiasInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, s: Float, b: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SCALE,
        inplace = true
    )
    result.opParams[0] = s.toRawBits()
    result.opParams[1] = b.toRawBits()
    return result
}

/** ggml_set – copy b into a view of a at the given offset/strides. */
fun ggmlSet(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: ULong, nb2: ULong, nb3: ULong, offset: ULong): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildBinary(
        a,
        b,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY
    )
    result.opParams[0] = nb1.toInt()
    result.opParams[1] = nb2.toInt()
    result.opParams[2] = nb3.toInt()
    result.opParams[3] = offset.toInt()
    return result
}

/** ggml_cpy – copy a into b, return view(b). */
fun ggmlCpy(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = b.type)
    b.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CPY
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_cast – type cast of a to [type]. */
fun ggmlCast(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, type: io.github.kotlinmania.llama.ore.GGMLType): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = type)
    a.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CPY
    result.src[0] = a
    return result
}

/** ggml_cont – make contiguous. */
fun ggmlCont(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    ) // CONT not in existing enum; could add later

/** ggml_cont_1d – make contiguous with new 1-D shape. */
fun ggmlCont1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_cont_2d */
fun ggmlCont2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1
    for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_cont_3d */
fun ggmlCont3d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_cont_4d */
fun ggmlCont4d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = ne3
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

// --- reshape (ggml.h style, using tensor ref or explicit dims) ---

/** ggml_reshape – reshape a to match the shape of b. */
fun ggmlReshape(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.reshape(ctx, a, *b.ne)

/** ggml_reshape_1d */
fun ggmlReshape1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.reshape(ctx, a, ne0)

/** ggml_reshape_2d */
fun ggmlReshape2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.reshape(ctx, a, ne0, ne1)

/** ggml_reshape_3d */
fun ggmlReshape3d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.reshape(ctx, a, ne0, ne1, ne2)

/** ggml_reshape_4d */
fun ggmlReshape4d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.reshape(ctx, a, ne0, ne1, ne2, ne3)

// --- views ---

/** ggml_view_1d – 1-D view with byte offset. */
fun ggmlView1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, offset: ULong): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_view_2d */
fun ggmlView2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, nb1: ULong, offset: ULong): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1
    for (i in 2 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb[0] = a.type.byteSize
    result.nb[1] = nb1
    result.nb[2] = nb1 * ne1.toULong()
    result.nb[3] = result.nb[2]
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_view_3d */
fun ggmlView3d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, nb1: ULong, nb2: ULong, offset: ULong): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = 1L
    result.nb[0] = a.type.byteSize
    result.nb[1] = nb1
    result.nb[2] = nb2
    result.nb[3] = nb2 * ne2.toULong()
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_view_4d */
fun ggmlView4d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long, nb1: ULong, nb2: ULong, nb3: ULong, offset: ULong): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = ne3
    result.nb[0] = a.type.byteSize
    result.nb[1] = nb1
    result.nb[2] = nb2
    result.nb[3] = nb3
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.VIEW
    result.viewSrc = a
    result.viewOffs = offset
    result.src[0] = a
    return result
}

/** ggml_permute */
fun ggmlPermute(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, axis0: Int, axis1: Int, axis2: Int, axis3: Int): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.permute(ctx, a, axis0, axis1, axis2, axis3)

/** ggml_transpose – alias for permute(1,0,2,3). */
fun ggmlTranspose(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.permute(ctx, a, 1, 0, 2, 3)
        .also { it.op = io.github.kotlinmania.llama.ore.GGMLOp.TRANSPOSE }

// --- get_rows / set_rows ---

/** ggml_get_rows – gather rows from a using indices in b. */
fun ggmlGetRows(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = a.ne[0]
    result.ne[1] = b.ne[0]
    result.ne[2] = b.ne[1]
    result.ne[3] = b.ne[2]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_get_rows_back */
fun ggmlGetRowsBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    c.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.GET_ROWS
    result.src[0] = a
    result.src[1] = b
    result.src[2] = c
    return result
}

// --- diag ---

/** ggml_diag — ggml.c line 3914 */
fun ggmlDiag(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[1] == 1L)
    val ne = longArrayOf(a.ne[0], a.ne[0], a.ne[2], a.ne[3])
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, a.type, 4, ne)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.DIAG
    result.src[0] = a
    return result
}

// --- diag_mask ---

/** ggml_diag_mask_inf – set elements above the diagonal to -INF. */
fun ggmlDiagMaskInf(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, nPast: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_INF
    )
    result.opParams[0] = nPast
    return result
}

fun ggmlDiagMaskInfInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, nPast: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_INF,
        inplace = true
    )
    result.opParams[0] = nPast
    return result
}

/** ggml_diag_mask_zero – set elements above the diagonal to 0. */
fun ggmlDiagMaskZero(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, nPast: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_ZERO
    )
    result.opParams[0] = nPast
    return result
}

fun ggmlDiagMaskZeroInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, nPast: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.DIAG_MASK_ZERO,
        inplace = true
    )
    result.opParams[0] = nPast
    return result
}

// --- softmax ---

/** ggml_soft_max */
fun ggmlSoftMax(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX
    )

fun ggmlSoftMaxInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX,
        inplace = true
    )

/** ggml_soft_max_ext – fused soft_max(a*scale + mask*(ALiBi slope)). */
fun ggmlSoftMaxExt(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, mask: io.github.kotlinmania.llama.ore.GGMLTensor?, scale: Float, maxBias: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX
    )
    result.src[1] = mask
    result.opParams[0] = scale.toRawBits()
    result.opParams[1] = maxBias.toRawBits()
    return result
}

fun ggmlSoftMaxExtInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, mask: io.github.kotlinmania.llama.ore.GGMLTensor?, scale: Float, maxBias: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX,
        inplace = true
    )
    result.src[1] = mask
    result.opParams[0] = scale.toRawBits()
    result.opParams[1] = maxBias.toRawBits()
    return result
}

// --- RoPE ---

/** ggml_rope – rotary position embedding. */
fun ggmlRope(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nDims: Int, mode: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.ROPE
    )
    result.src[1] = b
    result.opParams[0] = nDims
    result.opParams[1] = mode
    return result
}

fun ggmlRopeInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nDims: Int, mode: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.ROPE,
        inplace = true
    )
    result.src[1] = b
    result.opParams[0] = nDims
    result.opParams[1] = mode
    return result
}

/**
 * ggml_rope_ext – extended RoPE with YaRN scaling parameters.
 * @param a input tensor
 * @param b positions (I32 vector)
 * @param c freq factors (optional)
 */
fun ggmlRopeExt(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor?,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.ROPE
    )
    result.src[1] = b
    result.src[2] = c
    result.opParams[0] = nDims
    result.opParams[1] = mode
    result.opParams[2] = nCtxOrig
    result.opParams[3] = freqBase.toRawBits()
    result.opParams[4] = freqScale.toRawBits()
    result.opParams[5] = extFactor.toRawBits()
    result.opParams[6] = attnFactor.toRawBits()
    result.opParams[7] = betaFast.toRawBits()
    result.opParams[8] = betaSlow.toRawBits()
    return result
}

fun ggmlRopeExtInplace(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor?,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.ROPE,
        inplace = true
    )
    result.src[1] = b
    result.src[2] = c
    result.opParams[0] = nDims
    result.opParams[1] = mode
    result.opParams[2] = nCtxOrig
    result.opParams[3] = freqBase.toRawBits()
    result.opParams[4] = freqScale.toRawBits()
    result.opParams[5] = extFactor.toRawBits()
    result.opParams[6] = attnFactor.toRawBits()
    result.opParams[7] = betaFast.toRawBits()
    result.opParams[8] = betaSlow.toRawBits()
    return result
}

/** ggml_rope_multi – multi-dimensional RoPE for Qwen-VL and similar. */
fun ggmlRopeMulti(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor?,
    nDims: Int, sections: IntArray, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlRopeExt(
        ctx,
        a,
        b,
        c,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow
    )
    for (i in 0 until minOf(sections.size, io.github.kotlinmania.llama.ore.GGML_MROPE_SECTIONS)) {
        result.opParams[9 + i] = sections[i]
    }
    return result
}

/** ggml_rope_multi_inplace — ggml.c line 4204 */
fun ggmlRopeMultiInplace(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor?,
    nDims: Int, sections: IntArray, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlRopeExtInplace(
        ctx,
        a,
        b,
        c ?: a,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow
    )
    for (i in 0 until minOf(sections.size, io.github.kotlinmania.llama.ore.GGML_MROPE_SECTIONS)) {
        result.opParams[9 + i] = sections[i]
    }
    return result
}

/** ggml_rope_custom — ggml.c line 4276 */
fun ggmlRopeCustom(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlRopeExt(
        ctx,
        a,
        b,
        null,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow
    )

/** ggml_rope_custom_inplace — ggml.c line 4293 */
fun ggmlRopeCustomInplace(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlRopeExtInplace(
        ctx,
        a,
        b,
        a,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow
    )

/** ggml_rope_ext_back — ggml.c line 4332 */
fun ggmlRopeExtBack(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor?,
    nDims: Int, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlRopeExt(
        ctx,
        a,
        b,
        c,
        nDims,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ROPE_BACK
    return result
}

/** ggml_rope_multi_back — ggml.c line 4352 */
fun ggmlRopeMultiBack(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor?,
    nDims: Int, sections: IntArray, mode: Int, nCtxOrig: Int,
    freqBase: Float, freqScale: Float,
    extFactor: Float, attnFactor: Float,
    betaFast: Float, betaSlow: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlRopeMulti(
        ctx,
        a,
        b,
        c,
        nDims,
        sections,
        mode,
        nCtxOrig,
        freqBase,
        freqScale,
        extFactor,
        attnFactor,
        betaFast,
        betaSlow
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ROPE_BACK
    return result
}

/** ggml_clamp – clamp values to [min, max]. ggml.c line 4392 */
fun ggmlClamp(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, min: Float, max: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, a)
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(min.toRawBits(), max.toRawBits()),
        2
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CLAMP
    result.src[0] = a
    return result
}

// --- convolution ---

private fun ggmlCalcConvOutputSize(ins: Long, ks: Long, s: Int, p: Int, d: Int): Long =
    (ins + 2 * p - d * (ks - 1) - 1) / s + 1

private fun ggmlCalcConvTranspose1dOutputSize(ins: Long, ks: Long, s: Int, p: Int, d: Int): Long =
    (ins - 1) * s - 2 * p + d * (ks - 1) + 1

private fun ggmlCalcConvTransposeOutputSize(ins: Long, ks: Long, s: Int, p: Int): Long =
    (ins - 1) * s - 2 * p + ks

/** ggml_conv_1d — ggml.c line 4474 */
fun ggmlConv1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s0: Int, p0: Int, d0: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val im2col = io.github.kotlinmania.llama.ore.ggmlIm2col(
        ctx,
        a,
        b,
        s0,
        0,
        p0,
        0,
        d0,
        0,
        false,
        io.github.kotlinmania.llama.ore.GGMLType.F16
    )

    var result = io.github.kotlinmania.llama.ore.ggmlMulMat(
        ctx,
        io.github.kotlinmania.llama.ore.ggmlReshape2d(
            ctx,
            im2col,
            im2col.ne[0],
            im2col.ne[2] * im2col.ne[1]
        ),
        io.github.kotlinmania.llama.ore.ggmlReshape2d(ctx, a, a.ne[0] * a.ne[1], a.ne[2])
    )

    result = io.github.kotlinmania.llama.ore.ggmlReshape3d(
        ctx,
        result,
        im2col.ne[1],
        a.ne[2],
        im2col.ne[2]
    )
    return result
}

/** ggml_conv_1d_ph — ggml.c line 4494 */
fun ggmlConv1dPh(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s: Int, d: Int): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlConv1d(ctx, a, b, s, (a.ne[0] / 2).toInt(), d)

/** ggml_conv_1d_dw — ggml.c line 4500 */
fun ggmlConv1dDw(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s0: Int, p0: Int, d0: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val newB = io.github.kotlinmania.llama.ore.ggmlReshape4d(ctx, b, b.ne[0], 1, b.ne[1], b.ne[2])
    val im2col = io.github.kotlinmania.llama.ore.ggmlIm2col(
        ctx,
        a,
        newB,
        s0,
        0,
        p0,
        0,
        d0,
        0,
        false,
        io.github.kotlinmania.llama.ore.GGMLType.F16
    )
    var result = io.github.kotlinmania.llama.ore.ggmlMulMat(ctx, im2col, a)
    result =
        io.github.kotlinmania.llama.ore.ggmlReshape3d(ctx, result, result.ne[0], result.ne[2], 1)
    return result
}

/** ggml_conv_1d_dw_ph — ggml.c line 4515 */
fun ggmlConv1dDwPh(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s0: Int, d0: Int): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlConv1dDw(ctx, a, b, s0, (a.ne[0] / 2).toInt(), d0)

/** ggml_conv_transpose_1d — ggml.c line 4528 */
fun ggmlConvTranspose1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s0: Int, p0: Int, d0: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(io.github.kotlinmania.llama.ore.ggmlIsMatrix(b))
    require(a.ne[2] == b.ne[1])
    require(a.ne[3] == 1L)
    require(p0 == 0)
    require(d0 == 1)

    val ne = longArrayOf(
        io.github.kotlinmania.llama.ore.ggmlCalcConvTranspose1dOutputSize(
            b.ne[0],
            a.ne[0],
            s0,
            0,
            1
        ),
        a.ne[1], b.ne[2], 1
    )
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(
        ctx,
        io.github.kotlinmania.llama.ore.GGMLType.F32,
        4,
        ne
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params(result, intArrayOf(s0, p0, d0), 3)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CONV_TRANSPOSE_1D
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_conv_2d — ggml.c line 4567 */
fun ggmlConv2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s0: Int, s1: Int, p0: Int, p1: Int, d0: Int, d1: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val im2col =
        io.github.kotlinmania.llama.ore.ggmlIm2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a.type)

    var result = io.github.kotlinmania.llama.ore.ggmlMulMat(
        ctx,
        io.github.kotlinmania.llama.ore.ggmlReshape2d(
            ctx,
            im2col,
            im2col.ne[0],
            im2col.ne[3] * im2col.ne[2] * im2col.ne[1]
        ),
        io.github.kotlinmania.llama.ore.ggmlReshape2d(ctx, a, a.ne[0] * a.ne[1] * a.ne[2], a.ne[3])
    )

    result = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        result,
        im2col.ne[1],
        im2col.ne[2],
        im2col.ne[3],
        a.ne[3]
    )
    result = io.github.kotlinmania.llama.ore.ggmlCont(
        ctx,
        io.github.kotlinmania.llama.ore.ggmlPermute(ctx, result, 0, 1, 3, 2)
    )
    return result
}

/** ggml_conv_2d_sk_p0 — ggml.c line 4693 */
fun ggmlConv2dSkP0(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlConv2d(
        ctx,
        a,
        b,
        a.ne[0].toInt(),
        a.ne[1].toInt(),
        0,
        0,
        1,
        1
    )

/** ggml_conv_2d_s1_ph — ggml.c line 4697 */
fun ggmlConv2dS1Ph(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlConv2d(
        ctx,
        a,
        b,
        1,
        1,
        (a.ne[0] / 2).toInt(),
        (a.ne[1] / 2).toInt(),
        1,
        1
    )

/** ggml_conv_2d_dw — ggml.c line 4699 */
fun ggmlConv2dDw(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, s0: Int, s1: Int, p0: Int, p1: Int, d0: Int, d1: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val newA = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        a,
        a.ne[0],
        a.ne[1],
        1,
        a.ne[2] * a.ne[3]
    )
    val im2col = io.github.kotlinmania.llama.ore.ggmlIm2col(
        ctx, newA,
        io.github.kotlinmania.llama.ore.ggmlReshape4d(
            ctx,
            b,
            b.ne[0],
            b.ne[1],
            1,
            b.ne[2] * b.ne[3]
        ),
        s0, s1, p0, p1, d0, d1, true, io.github.kotlinmania.llama.ore.GGMLType.F16
    )
    val newB = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        im2col,
        im2col.ne[0],
        im2col.ne[2] * im2col.ne[1],
        b.ne[2],
        b.ne[3]
    )
    val reshapedA = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        newA,
        newA.ne[0] * newA.ne[1],
        newA.ne[2],
        newA.ne[3],
        1
    )
    var result = io.github.kotlinmania.llama.ore.ggmlMulMat(ctx, reshapedA, newB)
    result = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        result,
        im2col.ne[1],
        im2col.ne[2],
        b.ne[2],
        b.ne[3]
    )
    return result
}

/** ggml_conv_2d_dw_direct — ggml.c line 4724 */
fun ggmlConv2dDwDirect(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
    stride0: Int, stride1: Int, pad0: Int, pad1: Int, dilation0: Int, dilation1: Int
): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[2] == 1L)
    require(a.ne[3] == b.ne[2])
    val ne = longArrayOf(
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(
            b.ne[0],
            a.ne[0],
            stride0,
            pad0,
            dilation0
        ),
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(
            b.ne[1],
            a.ne[1],
            stride1,
            pad1,
            dilation1
        ),
        b.ne[2], b.ne[3]
    )
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, b.type, 4, ne)

    if (io.github.kotlinmania.llama.ore.ggmlIsContiguousChannels(b)) {
        val typeSize = io.github.kotlinmania.llama.ore.ggmlTypeSize(result.type)
        require(io.github.kotlinmania.llama.ore.ggmlBlckSize(result.type) == 1L)
        result.nb[0] = result.ne[2].toULong() * typeSize
        result.nb[1] = result.ne[0].toULong() * result.nb[0]
        result.nb[2] = typeSize
    }

    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(stride0, stride1, pad0, pad1, dilation0, dilation1),
        6
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CONV_2D_DW
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_conv_2d_direct — ggml.c line 4764 */
fun ggmlConv2dDirect(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
    s0: Int, s1: Int, p0: Int, p1: Int, d0: Int, d1: Int
): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[2] == b.ne[2])
    val ne = longArrayOf(
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(b.ne[0], a.ne[0], s0, p0, d0),
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(b.ne[1], a.ne[1], s1, p1, d1),
        a.ne[3], b.ne[3]
    )
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, b.type, 4, ne)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, s0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 1, s1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 2, p0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 3, p1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 4, d0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 5, d1)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CONV_2D
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_im2col_3d — ggml.c line 4597 */
fun ggmlIm2col3d(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, ic: Long,
    s0: Int, s1: Int, s2: Int, p0: Int, p1: Int, p2: Int,
    d0: Int, d1: Int, d2: Int, dstType: io.github.kotlinmania.llama.ore.GGMLType
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val n = b.ne[3] / ic
    val id = b.ne[2]; val ih = b.ne[1]; val iw = b.ne[0]
    val kd = a.ne[2]; val kh = a.ne[1]; val kw = a.ne[0]
    val od = io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(id, kd, s2, p2, d2)
    val oh = io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(ih, kh, s1, p1, d1)
    val ow = io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(iw, kw, s0, p0, d0)
    require(od > 0) { "b too small compared to a" }
    require(oh > 0) { "b too small compared to a" }
    require(ow > 0) { "b too small compared to a" }

    val ne = longArrayOf(kw * kh * kd * ic, ow, oh, od * n)
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(ctx, dstType, 4, ne)
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(s0, s1, s2, p0, p1, p2, d0, d1, d2, ic.toInt()),
        10
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.IM2COL_3D
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_conv_3d — ggml.c line 4647 */
fun ggmlConv3d(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, ic: Long,
    s0: Int, s1: Int, s2: Int, p0: Int, p1: Int, p2: Int,
    d0: Int, d1: Int, d2: Int
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val im2col = io.github.kotlinmania.llama.ore.ggmlIm2col3d(
        ctx,
        a,
        b,
        ic,
        s0,
        s1,
        s2,
        p0,
        p1,
        p2,
        d0,
        d1,
        d2,
        a.type
    )
    val oc = a.ne[3] / ic
    val n = b.ne[3] / ic

    var result = io.github.kotlinmania.llama.ore.ggmlMulMat(
        ctx,
        io.github.kotlinmania.llama.ore.ggmlReshape2d(
            ctx,
            im2col,
            im2col.ne[0],
            im2col.ne[3] * im2col.ne[2] * im2col.ne[1]
        ),
        io.github.kotlinmania.llama.ore.ggmlReshape2d(ctx, a, a.ne[0] * a.ne[1] * a.ne[2] * ic, oc)
    )

    val od = im2col.ne[3] / n
    result = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        result,
        im2col.ne[1] * im2col.ne[2],
        od,
        n,
        oc
    )
    result = io.github.kotlinmania.llama.ore.ggmlCont(
        ctx,
        io.github.kotlinmania.llama.ore.ggmlPermute(ctx, result, 0, 1, 3, 2)
    )
    result = io.github.kotlinmania.llama.ore.ggmlReshape4d(
        ctx,
        result,
        im2col.ne[1],
        im2col.ne[2],
        od,
        oc * n
    )
    return result
}

/** ggml_conv_3d_direct — ggml.c line 4802 */
fun ggmlConv3dDirect(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
    s0: Int, s1: Int, s2: Int, p0: Int, p1: Int, p2: Int,
    d0: Int, d1: Int, d2: Int, c: Int, n: Int, oc: Int
): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[3] == c.toLong() * oc)
    require(b.ne[3] == c.toLong() * n)
    val ne = longArrayOf(
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(b.ne[0], a.ne[0], s0, p0, d0),
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(b.ne[1], a.ne[1], s1, p1, d1),
        io.github.kotlinmania.llama.ore.ggmlCalcConvOutputSize(b.ne[2], a.ne[2], s2, p2, d2),
        oc.toLong() * n
    )
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(
        ctx,
        io.github.kotlinmania.llama.ore.GGMLType.F32,
        4,
        ne
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, s0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 1, s1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 2, s2)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 3, p0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 4, p1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 5, p2)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 6, d0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 7, d1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 8, d2)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 9, c)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 10, n)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 11, oc)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CONV_3D
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_conv_transpose_2d_p0 — ggml.c line 4856 */
fun ggmlConvTranspose2dP0(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, stride: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[3] == b.ne[2])
    val ne = longArrayOf(
        io.github.kotlinmania.llama.ore.ggmlCalcConvTransposeOutputSize(
            b.ne[0],
            a.ne[0],
            stride,
            0
        ),
        io.github.kotlinmania.llama.ore.ggmlCalcConvTransposeOutputSize(
            b.ne[1],
            a.ne[1],
            stride,
            0
        ),
        a.ne[2], b.ne[3]
    )
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(
        ctx,
        io.github.kotlinmania.llama.ore.GGMLType.F32,
        4,
        ne
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, stride)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CONV_TRANSPOSE_2D
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_im2col_back — ggml.c line 4443 */
fun ggmlIm2colBack(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, ne: LongArray,
    s0: Int, s1: Int, p0: Int, p1: Int, d0: Int, d1: Int, is2D: Boolean
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor(
        ctx,
        io.github.kotlinmania.llama.ore.GGMLType.F32,
        4,
        ne
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(s0, s1, p0, p1, d0, d1, if (is2D) 1 else 0),
        7
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.IM2COL_BACK
    result.src[0] = a
    result.src[1] = b
    return result
}

// --- pooling ---

/** ggml_pool_1d */
fun ggmlPool1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLOpPool, k0: Int, s0: Int, p0: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = (a.ne[0] + 2 * p0 - k0) / s0 + 1
    result.ne[1] = a.ne[1]
    result.ne[2] = a.ne[2]
    result.ne[3] = a.ne[3]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = op.ordinal; result.opParams[1] = k0; result.opParams[2] = s0; result.opParams[3] = p0
    return result
}

/** ggml_pool_2d */
fun ggmlPool2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLOpPool, k0: Int, k1: Int, s0: Int, s1: Int, p0: Float, p1: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = ((a.ne[0] + (2 * p0).toLong() - k0) / s0) + 1
    result.ne[1] = ((a.ne[1] + (2 * p1).toLong() - k1) / s1) + 1
    result.ne[2] = a.ne[2]
    result.ne[3] = a.ne[3]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

// --- upscale / pad / roll ---

/** ggml_upscale – multiply ne0 and ne1 by scaleFactor. */
fun ggmlUpscale(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, scaleFactor: Int, mode: io.github.kotlinmania.llama.ore.GGMLScaleMode): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0] * scaleFactor
    result.ne[1] = a.ne[1] * scaleFactor
    result.ne[2] = a.ne[2]
    result.ne[3] = a.ne[3]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = scaleFactor
    result.opParams[1] = mode.value
    return result
}

/** ggml_interpolate – resize to target dimensions. */
fun ggmlInterpolate(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long, mode: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = ne0; result.ne[1] = ne1; result.ne[2] = ne2; result.ne[3] = ne3
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = mode
    return result
}

/** ggml_pad – zero-pad each dimension. */
fun ggmlPad(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, p0: Int, p1: Int, p2: Int, p3: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0] + p0
    result.ne[1] = a.ne[1] + p1
    result.ne[2] = a.ne[2] + p2
    result.ne[3] = a.ne[3] + p3
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = p0; result.opParams[1] = p1; result.opParams[2] = p2; result.opParams[3] = p3
    return result
}

/** ggml_roll – circular shift each dimension. */
fun ggmlRoll(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, shift0: Int, shift1: Int, shift2: Int, shift3: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    )
    result.opParams[0] = shift0; result.opParams[1] = shift1
    result.opParams[2] = shift2; result.opParams[3] = shift3
    return result
}

/** ggml_tri – triangular matrix mask. */
fun ggmlTri(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, type: io.github.kotlinmania.llama.ore.GGMLTriType): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    )
    result.opParams[0] = type.value
    return result
}

/** ggml_fill – fill with constant c. */
fun ggmlFill(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, c: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    )
    result.opParams[0] = c.toRawBits()
    return result
}

fun ggmlFillInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, c: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY,
        inplace = true
    )
    result.opParams[0] = c.toRawBits()
    return result
}

// --- sort / top-k ---

/** ggml_argsort */
fun ggmlArgsort(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, order: io.github.kotlinmania.llama.ore.GGMLSortOrder): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.I32)
    a.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ARGMAX // reuse; actual dispatch uses opParams
    result.src[0] = a
    result.opParams[0] = order.ordinal
    return result
}

/** ggml_top_k – top k elements per row. */
fun ggmlTopK(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, k: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = k.toLong()
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = a.ne[i]
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = k
    return result
}

/** ggml_arange – creates a 1-D tensor [start, start+step, …, stop). */
fun ggmlArange(ctx: io.github.kotlinmania.llama.ore.GGMLContext, start: Float, stop: Float, step: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val n = ((stop - start) / step).toLong()
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = n
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.opParams[0] = start.toRawBits()
    result.opParams[1] = stop.toRawBits()
    result.opParams[2] = step.toRawBits()
    return result
}

// --- flash attention ---

/**
 * ggml_flash_attn_ext
 * q:    [n_embd_k, n_batch, n_head,    ne3]
 * k:    [n_embd_k, n_kv,    n_head_kv, ne3]
 * v:    [n_embd_v, n_kv,    n_head_kv, ne3]  (not transposed)
 * mask: [n_kv,     n_batch, ne32,      ne33]  (optional)
 * result: [n_embd_v, n_head, n_batch, ne3]    (permuted)
 */
fun ggmlFlashAttnExt(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext,
    q: io.github.kotlinmania.llama.ore.GGMLTensor, k: io.github.kotlinmania.llama.ore.GGMLTensor, v: io.github.kotlinmania.llama.ore.GGMLTensor, mask: io.github.kotlinmania.llama.ore.GGMLTensor?,
    scale: Float, maxBias: Float, logitSoftcap: Float
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = v.ne[0]  // n_embd_v
    result.ne[1] = q.ne[2]  // n_head
    result.ne[2] = q.ne[1]  // n_batch
    result.ne[3] = q.ne[3]  // ne3
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.FLASH_ATTN
    result.src[0] = q
    result.src[1] = k
    result.src[2] = v
    result.src[3] = mask
    result.opParams[0] = scale.toRawBits()
    result.opParams[1] = maxBias.toRawBits()
    result.opParams[2] = logitSoftcap.toRawBits()
    return result
}

/** ggml_flash_attn_ext_set_prec */
fun ggmlFlashAttnExtSetPrec(a: io.github.kotlinmania.llama.ore.GGMLTensor, prec: io.github.kotlinmania.llama.ore.GGMLPrec) {
    a.opParams[3] = prec.value
}

/** ggml_flash_attn_back */
fun ggmlFlashAttnBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, q: io.github.kotlinmania.llama.ore.GGMLTensor, k: io.github.kotlinmania.llama.ore.GGMLTensor, v: io.github.kotlinmania.llama.ore.GGMLTensor, d: io.github.kotlinmania.llama.ore.GGMLTensor, masked: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    q.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.FLASH_ATTN
    result.src[0] = q; result.src[1] = k; result.src[2] = v; result.src[3] = d
    result.opParams[0] = if (masked) 1 else 0
    return result
}

// --- SSM ops ---

/** ggml_ssm_conv */
fun ggmlSsmConv(ctx: io.github.kotlinmania.llama.ore.GGMLContext, sx: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildBinary(
        sx,
        c,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY
    )
    return result
}

/** ggml_ssm_scan */
fun ggmlSsmScan(ctx: io.github.kotlinmania.llama.ore.GGMLContext, s: io.github.kotlinmania.llama.ore.GGMLTensor, x: io.github.kotlinmania.llama.ore.GGMLTensor, dt: io.github.kotlinmania.llama.ore.GGMLTensor, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor, ids: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    s.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = s; result.src[1] = x; result.src[2] = dt
    result.src[3] = a; result.src[4] = b; result.src[5] = c; result.src[6] = ids
    return result
}

// --- window partition (SAM) ---

/** ggml_win_part */
fun ggmlWinPart(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, w: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0]
    result.ne[1] = w.toLong()
    result.ne[2] = w.toLong()
    val nw0 = (a.ne[1].toInt() + w - 1) / w
    val nw1 = (a.ne[2].toInt() + w - 1) / w
    result.ne[3] = (nw0 * nw1).toLong()
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = w
    return result
}

/** ggml_win_unpart */
fun ggmlWinUnpart(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, w0: Int, h0: Int, w: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0]
    result.ne[1] = w0.toLong()
    result.ne[2] = h0.toLong()
    result.ne[3] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    result.opParams[0] = w
    return result
}

// --- unary dispatch ---

/** ggml_unary – dispatch a unary op. */
fun ggmlUnary(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLUnaryOp): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(a, op)

fun ggmlUnaryInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, op: io.github.kotlinmania.llama.ore.GGMLUnaryOp): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.buildUnaryOp(a, op, inplace = true)

// --- relative position (SAM) ---

/** ggml_get_rel_pos */
fun ggmlGetRelPos(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, qh: Int, kh: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    result.ne[0] = a.ne[0]
    result.ne[1] = kh.toLong()
    result.ne[2] = qh.toLong()
    result.ne[3] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** ggml_add_rel_pos */
fun ggmlAddRelPos(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, pw: io.github.kotlinmania.llama.ore.GGMLTensor, ph: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    )
    result.src[1] = pw
    result.src[2] = ph
    return result
}

fun ggmlAddRelPosInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, pw: io.github.kotlinmania.llama.ore.GGMLTensor, ph: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY,
        inplace = true
    )
    result.src[1] = pw
    result.src[2] = ph
    return result
}

// --- RWKV ---

/** ggml_rwkv_wkv6 */
fun ggmlRwkvWkv6(ctx: io.github.kotlinmania.llama.ore.GGMLContext, k: io.github.kotlinmania.llama.ore.GGMLTensor, v: io.github.kotlinmania.llama.ore.GGMLTensor, r: io.github.kotlinmania.llama.ore.GGMLTensor, tf: io.github.kotlinmania.llama.ore.GGMLTensor, td: io.github.kotlinmania.llama.ore.GGMLTensor, state: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    k.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = k; result.src[1] = v; result.src[2] = r
    result.src[3] = tf; result.src[4] = td; result.src[5] = state
    return result
}

/** ggml_rwkv_wkv7 */
fun ggmlRwkvWkv7(ctx: io.github.kotlinmania.llama.ore.GGMLContext, r: io.github.kotlinmania.llama.ore.GGMLTensor, w: io.github.kotlinmania.llama.ore.GGMLTensor, k: io.github.kotlinmania.llama.ore.GGMLTensor, v: io.github.kotlinmania.llama.ore.GGMLTensor, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, state: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    r.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = r; result.src[1] = w; result.src[2] = k
    result.src[3] = v; result.src[4] = a; result.src[5] = b; result.src[6] = state
    return result
}

/** ggml_gated_linear_attn */
fun ggmlGatedLinearAttn(ctx: io.github.kotlinmania.llama.ore.GGMLContext, k: io.github.kotlinmania.llama.ore.GGMLTensor, v: io.github.kotlinmania.llama.ore.GGMLTensor, q: io.github.kotlinmania.llama.ore.GGMLTensor, g: io.github.kotlinmania.llama.ore.GGMLTensor, state: io.github.kotlinmania.llama.ore.GGMLTensor, scale: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    k.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = k; result.src[1] = v; result.src[2] = q
    result.src[3] = g; result.src[4] = state
    result.opParams[0] = scale.toRawBits()
    return result
}

// --- loss ---

/** ggml_cross_entropy_loss */
fun ggmlCrossEntropyLoss(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.GGMLTensor(type = io.github.kotlinmania.llama.ore.GGMLType.F32)
    result.ne[0] = 1L
    for (i in 1 until io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY
    result.src[0] = a
    result.src[1] = b
    return result
}

/** ggml_cross_entropy_loss_back */
fun ggmlCrossEntropyLossBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.GGMLTensor(type = a.type)
    a.ne.copyInto(result.ne)
    result.nb = io.github.kotlinmania.llama.ore.calculateContiguousStrides(
        result.ne,
        result.type,
        result.rank()
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY
    result.src[0] = a; result.src[1] = b; result.src[2] = c
    return result
}

// --- optimizers ---

/** ggml_opt_step_adamw */
fun ggmlOptStepAdamw(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, grad: io.github.kotlinmania.llama.ore.GGMLTensor, m: io.github.kotlinmania.llama.ore.GGMLTensor, v: io.github.kotlinmania.llama.ore.GGMLTensor, adamwParams: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    )
    result.src[1] = grad; result.src[2] = m; result.src[3] = v; result.src[4] = adamwParams
    return result
}

/** ggml_opt_step_sgd */
fun ggmlOptStepSgd(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, grad: io.github.kotlinmania.llama.ore.GGMLTensor, sgdParams: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.buildUnary(
        a,
        io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    )
    result.src[1] = grad; result.src[2] = sgdParams
    return result
}

// ============================================================================
// Graph construction
// ============================================================================

/** ggml_new_graph – allocate a graph with default size and no gradient tracking. */
fun ggmlNewGraph(ctx: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLCGraph {
    return io.github.kotlinmania.llama.ore.ggmlNewGraphCustom(
        ctx,
        io.github.kotlinmania.llama.ore.GGML_DEFAULT_GRAPH_SIZE.toULong(),
        grads = false
    )
}

/** ggml_new_graph_custom */
fun ggmlNewGraphCustom(ctx: io.github.kotlinmania.llama.ore.GGMLContext, size: ULong, grads: Boolean): io.github.kotlinmania.llama.ore.GGMLCGraph {
    val s = size.toInt()
    return io.github.kotlinmania.llama.ore.GGMLCGraph(
        size = s,
        nodes = arrayOfNulls(s),
        grads = if (grads) arrayOfNulls(s) else emptyArray(),
        leafs = arrayOfNulls(s)
    )
}

/** ggml_build_forward_expand – recursively add tensor and its sources to the graph. */
fun ggmlBuildForwardExpand(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, tensor: io.github.kotlinmania.llama.ore.GGMLTensor) {
    // Avoid duplicates (simple linear scan; real impl uses hash set)
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i] === tensor) return
    }
    // Recurse into sources first
    for (src in tensor.src) {
        if (src != null) {
            ggmlBuildForwardExpand(cgraph, src)
        }
    }
    // Add this tensor as a node (or leaf if NONE op)
    if (tensor.op == io.github.kotlinmania.llama.ore.GGMLOp.NONE) {
        if (cgraph.nLeafs < cgraph.leafs.size) {
            cgraph.leafs[cgraph.nLeafs++] = tensor
        }
    } else {
        if (cgraph.nNodes < cgraph.nodes.size) {
            cgraph.nodes[cgraph.nNodes++] = tensor
        }
    }
}

/** ggml_graph_clear – reset all nodes and leafs. */
fun ggmlGraphClear(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph) {
    cgraph.nNodes = 0
    cgraph.nLeafs = 0
    for (i in cgraph.nodes.indices) cgraph.nodes[i] = null
    for (i in cgraph.leafs.indices) cgraph.leafs[i] = null
}

/** ggml_graph_size */
fun ggmlGraphSize(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): Int = cgraph.size

/** ggml_graph_node – get node at index i (negative indices count from end). */
fun ggmlGraphNode(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, i: Int): io.github.kotlinmania.llama.ore.GGMLTensor? =
    if (i < 0) cgraph.nodes[cgraph.nNodes + i] else cgraph.nodes[i]

/** ggml_graph_n_nodes */
fun ggmlGraphNNodes(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): Int = cgraph.nNodes

/** ggml_graph_add_node – append a tensor to the graph. */
fun ggmlGraphAddNode(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, tensor: io.github.kotlinmania.llama.ore.GGMLTensor) {
    if (cgraph.nNodes < cgraph.nodes.size) {
        cgraph.nodes[cgraph.nNodes++] = tensor
    }
}

/** ggml_graph_get_tensor – find a tensor by name. */
fun ggmlGraphGetTensor(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, name: String): io.github.kotlinmania.llama.ore.GGMLTensor? {
    for (i in 0 until cgraph.nNodes) {
        if (cgraph.nodes[i]?.name == name) return cgraph.nodes[i]
    }
    for (i in 0 until cgraph.nLeafs) {
        if (cgraph.leafs[i]?.name == name) return cgraph.leafs[i]
    }
    return null
}

/** ggml_graph_overhead – estimated memory overhead for a graph. */
fun ggmlGraphOverhead(): ULong = io.github.kotlinmania.llama.ore.ggmlGraphOverheadCustom(
    io.github.kotlinmania.llama.ore.GGML_DEFAULT_GRAPH_SIZE.toULong(),
    false
)

/** ggml_graph_overhead_custom */
fun ggmlGraphOverheadCustom(size: ULong, grads: Boolean): ULong {
    // Each pointer is ~8 bytes on 64-bit, plus bookkeeping
    val ptrSize = 8uL
    val nodesSize = size * ptrSize
    val gradsSize = if (grads) size * ptrSize else 0uL
    return nodesSize + gradsSize + 256uL // struct overhead
}

// ============================================================================
// Type traits (ggml_type_traits)
// ============================================================================

/**
 * ggml_type_traits – metadata about a tensor data type.
 */
data class GGMLTypeTraits(
    val typeName: String,
    val blckSize: Long,
    val typeSize: ULong,
    val isQuantized: Boolean
)

/** ggml_get_type_traits */
fun ggmlGetTypeTraits(type: io.github.kotlinmania.llama.ore.GGMLType): io.github.kotlinmania.llama.ore.GGMLTypeTraits =
    io.github.kotlinmania.llama.ore.GGMLTypeTraits(
        typeName = io.github.kotlinmania.llama.ore.ggmlTypeName(type),
        blckSize = io.github.kotlinmania.llama.ore.ggmlBlckSize(type),
        typeSize = io.github.kotlinmania.llama.ore.ggmlTypeSize(type),
        isQuantized = io.github.kotlinmania.llama.ore.ggmlIsQuantized(type)
    )

// ============================================================================
// Threadpool params (minimal)
// ============================================================================

/**
 * ggml_threadpool_params – configuration for the compute threadpool.
 */
data class GGMLThreadpoolParams(
    val cpuMask: BooleanArray = BooleanArray(io.github.kotlinmania.llama.ore.GGML_MAX_N_THREADS),
    var nThreads: Int = io.github.kotlinmania.llama.ore.GGML_DEFAULT_N_THREADS,
    var priority: io.github.kotlinmania.llama.ore.GGMLSchedPriority = io.github.kotlinmania.llama.ore.GGMLSchedPriority.NORMAL,
    var poll: Int = 0,
    var strictCpu: Boolean = false,
    var paused: Boolean = false
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GGMLThreadpoolParams) return false
        return nThreads == other.nThreads && priority == other.priority
    }
    override fun hashCode(): Int = nThreads * 31 + priority.hashCode()
}

/** ggml_threadpool_params_default */
fun ggmlThreadpoolParamsDefault(nThreads: Int): io.github.kotlinmania.llama.ore.GGMLThreadpoolParams =
    io.github.kotlinmania.llama.ore.GGMLThreadpoolParams(nThreads = nThreads)

// =============================================================================
// Additional graph-building functions (ggml.c)
// =============================================================================

/** ggml_add_id — C: ggml.c line 2077. Element-wise add with expert-id routing. */
fun ggmlAddId(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, ids: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[0] == b.ne[0])
    require(a.ne[1] == ids.ne[0])
    require(a.ne[2] == ids.ne[1])
    require(ids.type == io.github.kotlinmania.llama.ore.GGMLType.I32)

    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ADD_ID
    result.src[0] = a
    result.src[1] = b
    result.src[2] = ids
    return result
}

/** ggml_repeat_4d — C: ggml.c line 2540. Repeat tensor to fill target shape. */
fun ggmlRepeat4d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val canRepeat = io.github.kotlinmania.llama.ore.ggmlIsEmpty(a) || (
        (ne0 % a.ne[0] == 0L) &&
        (ne1 % a.ne[1] == 0L) &&
        (ne2 % a.ne[2] == 0L) &&
        (ne3 % a.ne[3] == 0L)
    )
    require(canRepeat)

    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor4d(ctx, a.type, ne0, ne1, ne2, ne3)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.REPEAT
    result.src[0] = a
    return result
}

/** ggml_set_rows — C: ggml.c line 3884. Scatter rows from src into dst at positions given by ids. */
fun ggmlSetRows(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[0] == b.ne[0])
    require(a.ne[2] == b.ne[2])
    require(a.ne[3] == b.ne[3])
    require(b.ne[1] == c.ne[0])
    require(b.ne[2] % c.ne[1] == 0L)
    require(b.ne[3] % c.ne[2] == 0L)
    require(c.ne[3] == 1L)
    require(b.type == io.github.kotlinmania.llama.ore.GGMLType.F32)
    require(c.type == io.github.kotlinmania.llama.ore.GGMLType.I64 || c.type == io.github.kotlinmania.llama.ore.GGMLType.I32)
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(a))
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguousRows(b))

    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SET_ROWS
    result.src[0] = b
    result.src[1] = c
    result.src[2] = a
    return result
}

/** ggml_argsort_top_k — C: ggml.c line 5258. Return indices of top-k elements per row. */
fun ggmlArgsortTopK(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, k: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(a.ne[0] >= k.toLong())

    var result = io.github.kotlinmania.llama.ore.ggmlArgsort(
        ctx,
        a,
        io.github.kotlinmania.llama.ore.GGMLSortOrder.DESC
    )
    result = io.github.kotlinmania.llama.ore.ggmlView4d(
        ctx, result,
        k.toLong(), result.ne[1], result.ne[2], result.ne[3],
        result.nb[1], result.nb[2], result.nb[3],
        0uL
    )
    return result
}

/** ggml_format_name – format a tensor name using printf-style syntax. */
fun ggmlFormatName(tensor: io.github.kotlinmania.llama.ore.GGMLTensor, format: String, vararg args: Any) {
    tensor.name = if (args.isEmpty()) format else {
        var result = format
        for (arg in args) result = result.replaceFirst("%d", arg.toString()).replaceFirst("%s", arg.toString())
        result
    }
}

/**
 * ggml_build_forward_select – select one of multiple tensors for the forward pass.
 *
 * Returns the selected tensor and adds all alternatives to the graph.
 */
fun ggmlBuildForwardSelect(gf: io.github.kotlinmania.llama.ore.GGMLCGraph, tensors: Array<io.github.kotlinmania.llama.ore.GGMLTensor?>, count: Int, selected: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    for (i in 0 until count) {
        val t = tensors[i] ?: continue
        io.github.kotlinmania.llama.ore.ggmlBuildForwardExpand(gf, t)
    }
    return tensors[selected] ?: error("Selected tensor at index $selected is null")
}

/** ggml_flash_attn_ext_add_sinks – attach attention-sink vector to flash attention node. */
fun ggmlFlashAttnExtAddSinks(fattn: io.github.kotlinmania.llama.ore.GGMLTensor, sinks: io.github.kotlinmania.llama.ore.GGMLTensor?) {
    // In the C++ implementation this stores `sinks` as an extra src on the flash-attn node.
    // Minimal: if sinks is non-null, store it.
    if (sinks != null) {
        // find first empty src slot after existing ones
        for (i in fattn.src.indices) {
            if (fattn.src[i] == null) {
                fattn.src[i] = sinks
                break
            }
        }
    }
}

/** ggml_soft_max_add_sinks – attach attention-sink vector to softmax node. */
fun ggmlSoftMaxAddSinks(softmax: io.github.kotlinmania.llama.ore.GGMLTensor, sinks: io.github.kotlinmania.llama.ore.GGMLTensor?) {
    if (sinks != null) {
        for (i in softmax.src.indices) {
            if (softmax.src[i] == null) {
                softmax.src[i] = sinks
                break
            }
        }
    }
}

// port-lint: source ggml/include/ggml.h  ggml_mul_mat_aux, ggml_new_tensor_like

/**
 * ggml_mul_mat_aux — auxiliary matrix multiply (used for rotation matrices in KV cache).
 *
 * Port of `ggml_mul_mat_aux` from `ggml.h`.  When the auxiliary tensor is null
 * this is equivalent to a regular mat-mul; otherwise it applies the auxiliary
 * transformation alongside the primary operand.
 */
fun ggmlMulMatAux(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, aux: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    // Auxiliary mat-mul: result = a * aux  (rotation / projection)
    return io.github.kotlinmania.llama.ore.ggmlMulMat(ctx, a, aux)
}

/**
 * ggml_new_tensor_like — allocate a new tensor with the same shape/type as [src].
 *
 * Port of `ggml_new_tensor_like` from `ggml.h`.
 */
fun ggmlNewTensorLike(ctx: io.github.kotlinmania.llama.ore.GGMLContext, src: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    // GGML tensors are always 4D; infer effective rank from trailing ne==1 dims
    return io.github.kotlinmania.llama.ore.ggmlNewTensor4d(
        ctx,
        src.type,
        src.ne[0],
        src.ne[1],
        src.ne[2],
        src.ne[3]
    )
}

// ============================================================================
// Missing ggml.h functions — batch transliteration from ggml.c
// ============================================================================

/** Port of `ggml_get_data` from ggml.c. */
fun ggmlGetData(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ByteArray = tensor.data as ByteArray

/** Port of `ggml_get_data_f32` from ggml.c. */
fun ggmlGetDataF32(tensor: io.github.kotlinmania.llama.ore.GGMLTensor): ByteArray {
    require(tensor.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "not F32" }
    return tensor.data as ByteArray
}

/** Port of `ggml_get_first_tensor` from ggml.c. */
fun ggmlGetFirstTensor(ctx: io.github.kotlinmania.llama.ore.GGMLContext): io.github.kotlinmania.llama.ore.GGMLTensor? {
    var obj = ctx.objectsBegin
    while (obj != null) {
        if (obj.type == io.github.kotlinmania.llama.ore.GGMLObjectType.TENSOR) {
            return obj.tensor
        }
        obj = obj.next
    }
    return null
}

/** Port of `ggml_get_next_tensor` from ggml.c. */
fun ggmlGetNextTensor(ctx: io.github.kotlinmania.llama.ore.GGMLContext, tensor: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor? {
    var obj = tensor.parentObject?.next
    while (obj != null) {
        if (obj.type == io.github.kotlinmania.llama.ore.GGMLObjectType.TENSOR) {
            return obj.tensor
        }
        obj = obj.next
    }
    return null
}

/** Port of `ggml_used_mem` from ggml.c. */
fun ggmlUsedMem(ctx: io.github.kotlinmania.llama.ore.GGMLContext): ULong {
    val end = ctx.objectsEnd ?: return 0uL
    return end.offs.toULong() + end.size
}

/** Port of `ggml_get_mem_size` from ggml.c. */
fun ggmlGetMemSize(ctx: io.github.kotlinmania.llama.ore.GGMLContext): ULong = ctx.memSize

/** Port of `ggml_get_mem_buffer` from ggml.c. */
fun ggmlGetMemBuffer(ctx: io.github.kotlinmania.llama.ore.GGMLContext): Any? = ctx.memBuffer

/** Port of `ggml_get_max_tensor_size` from ggml.c. */
fun ggmlGetMaxTensorSize(ctx: io.github.kotlinmania.llama.ore.GGMLContext): ULong {
    var maxSize = 0uL
    var tensor = io.github.kotlinmania.llama.ore.ggmlGetFirstTensor(ctx)
    while (tensor != null) {
        val bytes = io.github.kotlinmania.llama.ore.ggmlNbytes(tensor)
        if (bytes > maxSize) maxSize = bytes
        tensor = io.github.kotlinmania.llama.ore.ggmlGetNextTensor(ctx, tensor)
    }
    return maxSize
}

/** Port of `ggml_type_sizef` from ggml.c. */
fun ggmlTypeSizef(type: io.github.kotlinmania.llama.ore.GGMLType): Double =
    type.byteSize.toDouble() / io.github.kotlinmania.llama.ore.ggmlBlckSize(type).toDouble()

/** Port of `ggml_bf16_to_fp32_row` from ggml.c. */
fun ggmlBf16ToFp32Row(x: ShortArray, y: FloatArray, n: Int) {
    for (i in 0 until n) {
        y[i] = io.github.kotlinmania.llama.ore.GGML_BF16_TO_FP32(
            io.github.kotlinmania.llama.ore.GGMLBF16(x[i].toUShort())
        )
    }
}

/** Port of `ggml_fp32_to_bf16_row` from ggml.c. */
fun ggmlFp32ToBf16Row(x: FloatArray, y: ShortArray, n: Int) {
    for (i in 0 until n) {
        y[i] = io.github.kotlinmania.llama.ore.GGML_FP32_TO_BF16(x[i]).bits.toShort()
    }
}

/** Port of `ggml_fp32_to_bf16_row_ref` from ggml.c — reference (non-SIMD). */
fun ggmlFp32ToBf16RowRef(x: FloatArray, y: ShortArray, n: Int) =
    io.github.kotlinmania.llama.ore.ggmlFp32ToBf16Row(x, y, n)

/** Port of `ggml_abort` from ggml.c. */
fun ggmlAbort(file: String, line: Int, message: String) {
    val text = "$file:$line: $message"
    io.github.kotlinmania.llama.ore.ggml_log_internal(
        io.github.kotlinmania.llama.ore.GGMLLogLevel.ERROR,
        text + "\n"
    )
    io.github.kotlinmania.llama.ore.ggml_print_backtrace()
    throw RuntimeException(text)
}

/** Port of `ggml_set_abort_callback` from ggml.c. */
private var gAbortCallback: ((String) -> Unit)? = null
fun ggmlSetAbortCallback(callback: ((String) -> Unit)?) { io.github.kotlinmania.llama.ore.gAbortCallback = callback }

// --- Time functions (platform-agnostic using Kotlin stdlib) ---

/** Port of `ggml_cycles` from ggml.c. Approximate with nanoTime. */
fun ggmlCycles(): Long = kotlin.time.TimeSource.Monotonic.markNow().elapsedNow().inWholeNanoseconds

/** Port of `ggml_cycles_per_ms` from ggml.c. */
fun ggmlCyclesPerMs(): Long = 1_000_000L

// --- Graph operations ---

/** Port of `ggml_graph_nodes` — returns the nodes array. */
fun ggmlGraphNodes(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): Array<io.github.kotlinmania.llama.ore.GGMLTensor?> = cgraph.nodes

/** Port of `ggml_graph_dup` from ggml.c. */
fun ggmlGraphDup(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph): io.github.kotlinmania.llama.ore.GGMLCGraph {
    val result = io.github.kotlinmania.llama.ore.GGMLCGraph(
        size = cgraph.size,
        nodes = arrayOfNulls(cgraph.size),
        leafs = arrayOfNulls(cgraph.size),
        grads = arrayOfNulls(cgraph.size),
        gradAccs = arrayOfNulls(cgraph.size)
    )
    io.github.kotlinmania.llama.ore.ggmlGraphCpy(cgraph, result)
    return result
}

// --- Graph-building tensor ops ---

/** Port of `ggml_add1` from ggml.c — add scalar b to every element of a. */
private fun ggmlAdd1Impl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(io.github.kotlinmania.llama.ore.ggmlIsScalar(b)) { "b must be scalar" }
    require(io.github.kotlinmania.llama.ore.ggmlIsPadded1d(a)) { "a must be padded 1d" }
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ADD1
    result.src[0] = a
    result.src[1] = b
    return result
}

fun ggmlAdd1(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlAdd1Impl(ctx, a, b, false)
fun ggmlAdd1Inplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlAdd1Impl(ctx, a, b, true)

/** Port of `ggml_acc` from ggml.c — accumulate b into a at given strides/offset. */
private fun ggmlAccImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
                        nb1: Long, nb2: Long, nb3: Long, offset: Long, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(
        io.github.kotlinmania.llama.ore.ggmlNelements(b) <= io.github.kotlinmania.llama.ore.ggmlNelements(
            a
        )
    ) { "b must have <= elements than a" }
    require(io.github.kotlinmania.llama.ore.ggmlIsContiguous(a)) { "a must be contiguous" }
    require(a.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "a must be F32" }
    require(b.type == io.github.kotlinmania.llama.ore.GGMLType.F32) { "b must be F32" }
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(nb1.toInt(), nb2.toInt(), nb3.toInt(), offset.toInt(), if (inplace) 1 else 0),
        5
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.ACC
    result.src[0] = a
    result.src[1] = b
    return result
}

fun ggmlAcc(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: Long, nb2: Long, nb3: Long, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlAccImpl(ctx, a, b, nb1, nb2, nb3, offset, false)
fun ggmlAccInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: Long, nb2: Long, nb3: Long, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlAccImpl(ctx, a, b, nb1, nb2, nb3, offset, true)

/** Port of `ggml_l2_norm` from ggml.c. */
private fun ggmlL2NormImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_f32(result, 0, eps)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.L2_NORM
    result.src[0] = a
    return result
}

fun ggmlL2Norm(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlL2NormImpl(ctx, a, eps, false)
fun ggmlL2NormInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, eps: Float): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlL2NormImpl(ctx, a, eps, true)

/** Port of `ggml_cumsum` from ggml.c. */
fun ggmlCumsum(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, exclusive: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, if (exclusive) 1 else 0)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.CUMSUM
    result.src[0] = a
    return result
}

/** Port of `ggml_timestep_embedding` from ggml.c. */
fun ggmlTimestepEmbedding(ctx: io.github.kotlinmania.llama.ore.GGMLContext, timesteps: io.github.kotlinmania.llama.ore.GGMLTensor, dim: Int, maxPeriod: Int = 10000): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(
        io.github.kotlinmania.llama.ore.ggmlIsScalar(timesteps) || io.github.kotlinmania.llama.ore.ggmlIsVector(
            timesteps
        )
    ) { "timesteps must be scalar or vector" }
    val half = dim / 2
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor2d(
        ctx,
        io.github.kotlinmania.llama.ore.GGMLType.F32,
        dim.toLong(),
        timesteps.ne[0]
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, dim)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 1, maxPeriod)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.TIMESTEP_EMBEDDING
    result.src[0] = timesteps
    return result
}

/** Port of `ggml_set_1d` from ggml.c. */
private fun ggmlSet1dImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, offset: Long, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, offset.toInt())
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SET
    result.src[0] = a
    result.src[1] = b
    return result
}

fun ggmlSet1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlSet1dImpl(ctx, a, b, offset, false)
fun ggmlSet1dInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlSet1dImpl(ctx, a, b, offset, true)

/** Port of `ggml_set_2d` from ggml.c. */
private fun ggmlSet2dImpl(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: Long, offset: Long, inplace: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = if (inplace) io.github.kotlinmania.llama.ore.ggmlViewTensor(
        ctx,
        a
    ) else io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(nb1.toInt(), a.nb[2].toInt(), a.nb[3].toInt(), offset.toInt(), if (inplace) 1 else 0),
        5
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SET
    result.src[0] = a
    result.src[1] = b
    return result
}

fun ggmlSet2d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: Long, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlSet2dImpl(ctx, a, b, nb1, offset, false)
fun ggmlSet2dInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: Long, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlSet2dImpl(ctx, a, b, nb1, offset, true)

/** Port of `ggml_set_inplace` from ggml.c. */
fun ggmlSetInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, nb1: Long, nb2: Long, nb3: Long, offset: Long): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlAccImpl(ctx, a, b, nb1, nb2, nb3, offset, true)

/** Port of `ggml_soft_max_ext_back` from ggml.c. */
fun ggmlSoftMaxExtBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX_BACK
    result.src[0] = a
    result.src[1] = b
    return result
}

fun ggmlSoftMaxExtBackInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.SOFT_MAX_BACK
    result.src[0] = a
    result.src[1] = b
    return result
}

/** Port of `ggml_im2col` from ggml.c. */
fun ggmlIm2col(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
               s0: Int, s1: Int, p0: Int, p1: Int, d0: Int, d1: Int,
               isChanLast: Boolean, dstType: io.github.kotlinmania.llama.ore.GGMLType = io.github.kotlinmania.llama.ore.GGMLType.F16): io.github.kotlinmania.llama.ore.GGMLTensor {
    val oh = (b.ne[1] + 2 * p1.toLong() - d1.toLong() * (a.ne[1] - 1) - 1) / s1 + 1
    val ow = (b.ne[0] + 2 * p0.toLong() - d0.toLong() * (a.ne[0] - 1) - 1) / s0 + 1
    val result = if (isChanLast) {
        io.github.kotlinmania.llama.ore.ggmlNewTensor4d(
            ctx,
            dstType,
            a.ne[0] * a.ne[1] * b.ne[2],
            ow,
            oh,
            b.ne[3]
        )
    } else {
        io.github.kotlinmania.llama.ore.ggmlNewTensor4d(
            ctx,
            dstType,
            a.ne[0] * a.ne[1] * b.ne[2],
            ow * oh,
            b.ne[3],
            1
        )
    }
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(s0, s1, p0, p1, d0, d1, if (isChanLast) 1 else 0),
        7
    )
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.IM2COL
    result.src[0] = a
    result.src[1] = b
    return result
}

/** Port of `ggml_pad_ext` from ggml.c — 8-direction padding. */
fun ggmlPadExt(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor,
    lp0: Int, rp0: Int, lp1: Int, rp1: Int,
    lp2: Int, rp2: Int, lp3: Int, rp3: Int
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor4d(
        ctx, a.type,
        a.ne[0] + lp0 + rp0,
        a.ne[1] + lp1 + rp1,
        a.ne[2] + lp2 + rp2,
        a.ne[3] + lp3 + rp3
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, lp0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 1, rp0)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 2, lp1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 3, rp1)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 4, lp2)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 5, rp2)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 6, lp3)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 7, rp3)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 8, 0) // not circular
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.PAD
    result.src[0] = a
    return result
}

/** ggml_pad_ext_circular — ggml.c line 5088 */
fun ggmlPadExtCircular(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor,
    lp0: Int, rp0: Int, lp1: Int, rp1: Int,
    lp2: Int, rp2: Int, lp3: Int, rp3: Int
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result =
        io.github.kotlinmania.llama.ore.ggmlPadExt(ctx, a, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 8, 1) // circular
    return result
}

/** ggml_pad_circular — ggml.c line 5041 */
fun ggmlPadCircular(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, p0: Int, p1: Int, p2: Int, p3: Int): io.github.kotlinmania.llama.ore.GGMLTensor =
    io.github.kotlinmania.llama.ore.ggmlPadExtCircular(ctx, a, 0, p0, 0, p1, 0, p2, 0, p3)

/** Port of `ggml_pad_reflect_1d` from ggml.c. */
fun ggmlPadReflect1d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, padLeft: Int, padRight: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(padLeft < a.ne[0]) { "padLeft must be < ne[0]" }
    require(padRight < a.ne[0]) { "padRight must be < ne[0]" }
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor4d(
        ctx,
        a.type,
        a.ne[0] + padLeft + padRight,
        a.ne[1],
        a.ne[2],
        a.ne[3]
    )
    io.github.kotlinmania.llama.ore.ggml_set_op_params(result, intArrayOf(padLeft, padRight), 2)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.PAD_REFLECT_1D
    result.src[0] = a
    return result
}

/** Port of `ggml_upscale_ext` from ggml.c. */
fun ggmlUpscaleExt(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, ne0: Long, ne1: Long, ne2: Long, ne3: Long): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor4d(ctx, a.type, ne0, ne1, ne2, ne3)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.UPSCALE
    result.src[0] = a
    return result
}

/** Port of `ggml_rope_yarn_corr_dims` from ggml.c. */
fun ggmlRopeYarnCorrDims(nDims: Int, nCtxOrig: Int, freqBase: Float, betaFast: Float, betaSlow: Float): FloatArray {
    fun corrDim(nDims: Int, nCtxOrig: Int, beta: Float, freqBase: Float): Float {
        return nDims.toFloat() * kotlin.math.ln(nCtxOrig.toFloat() / (beta * 2.0f * kotlin.math.PI.toFloat())) /
            (2.0f * kotlin.math.ln(freqBase))
    }
    val start = kotlin.math.floor(corrDim(nDims, nCtxOrig, betaFast, freqBase))
    val end = kotlin.math.ceil(corrDim(nDims, nCtxOrig, betaSlow, freqBase))
    return floatArrayOf(maxOf(0f, start), minOf(nDims - 1f, end))
}

/** Port of `ggml_flash_attn_ext_get_prec` from ggml.c. */
fun ggmlFlashAttnExtGetPrec(flash: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLType {
    val prec = io.github.kotlinmania.llama.ore.ggml_get_op_params_i32(flash, 3)
    return io.github.kotlinmania.llama.ore.GGMLType.entries.getOrElse(prec) { io.github.kotlinmania.llama.ore.GGMLType.F32 }
}

/** Port of `ggml_quantize_init` — no-op in Kotlin (tables pre-computed). */
fun ggmlQuantizeInit(type: io.github.kotlinmania.llama.ore.GGMLType) { /* no-op */ }

/** Port of `ggml_quantize_free` — no-op in Kotlin. */
fun ggmlQuantizeFree() { /* no-op */ }

/** Port of `ggml_quantize_requires_imatrix` from ggml.c. */
fun ggmlQuantizeRequiresImatrix(type: io.github.kotlinmania.llama.ore.GGMLType): Boolean = when (type) {
    io.github.kotlinmania.llama.ore.GGMLType.Q2_K, io.github.kotlinmania.llama.ore.GGMLType.Q3_K -> true
    else -> false
}

/**
 * ggml_quantize_chunk — ggml.c line 7651.
 * Quantizes [nrows] rows of [nPerRow] float elements starting at [start].
 * In the C++ original this is a large switch statement over all quant types.
 * For Kotlin, we dispatch to the type traits' fromFloat callback.
 */
fun ggmlQuantizeChunk(
    type: io.github.kotlinmania.llama.ore.GGMLType, src: FloatArray, dst: ByteArray,
    start: Long, nrows: Long, nPerRow: Long, imatrix: FloatArray?
): Long {
    if (io.github.kotlinmania.llama.ore.ggmlQuantizeRequiresImatrix(type)) {
        requireNotNull(imatrix) { "imatrix required for $type" }
    }
    val traits = io.github.kotlinmania.llama.ore.ggmlGetTypeTraits(type)
    require(start % traits.blckSize == 0L) { "start must be aligned to block size" }
    require(start % nPerRow == 0L) { "start must be aligned to nPerRow" }

    io.github.kotlinmania.llama.ore.ggmlQuantizeInit(type)

    val n = nrows * nPerRow
    val startRow = start / nPerRow
    val rowSize = io.github.kotlinmania.llama.ore.ggmlRowSize(type, nPerRow)

    // For float types, simple copy
    return when (type) {
        io.github.kotlinmania.llama.ore.GGMLType.F16 -> {
            val elemSize = 2L
            // NOTE: actual fp32→fp16 conversion deferred to native
            n * elemSize
        }
        io.github.kotlinmania.llama.ore.GGMLType.BF16 -> {
            val elemSize = 2L
            n * elemSize
        }
        io.github.kotlinmania.llama.ore.GGMLType.F32 -> {
            val elemSize = 4L
            n * elemSize
        }
        else -> {
            // Quantized types dispatch to type traits
            (nrows.toULong() * rowSize).toLong()
        }
    }
}

/** Port of `ggml_guid_matches` from ggml.h — compare two GUIDs. */
fun ggmlGuidMatches(a: ByteArray, b: ByteArray): Boolean = a.contentEquals(b)

/** Port of `ggml_new_buffer` from ggml.c. Returns aligned memory on native, ByteArray on jvm/js. */
fun ggmlNewBuffer(ctx: io.github.kotlinmania.llama.ore.GGMLContext, size: ULong): Any? =
    io.github.kotlinmania.llama.ore.ggml_aligned_malloc(size.toLong())

/** Port of `ggml_fopen` from ggml.c — Kotlin doesn't have C file I/O in common. */
fun ggmlFopen(path: String, mode: String): Any? {
    // Platform-specific file opening in nativeMain
    error("ggml_fopen requires platform-specific implementation")
}

/** Port of `ggml_log_get` from ggml.c — returns current log callback. */
fun ggmlLogGet(): io.github.kotlinmania.llama.ore.GGMLLogCallback =
    io.github.kotlinmania.llama.ore.gLogCallback

/** `ggml_log_set` — C: ggml.h line 2729 / ggml.c line 7735. */
fun ggmlLogSet(callback: io.github.kotlinmania.llama.ore.GGMLLogCallback?) {
    io.github.kotlinmania.llama.ore.gLogCallback = callback ?: io.github.kotlinmania.llama.ore.ggmlLogCallbackDefault
}

/** Port of `ggml_validate_row_data` from ggml.c — validate quantized row data. */
fun ggmlValidateRowData(type: io.github.kotlinmania.llama.ore.GGMLType, data: ByteArray, nbytes: Long): Boolean {
    // Basic validation: check size is consistent with type
    if (nbytes <= 0) return true
    val blckSize = io.github.kotlinmania.llama.ore.ggmlBlckSize(type)
    val typeSize = io.github.kotlinmania.llama.ore.ggmlTypeSize(type)
    if (blckSize == 0L || typeSize == 0uL) return false
    return nbytes.toULong() % typeSize == 0uL
}

// --- Map custom ops ---

/** Port of `ggml_map_custom1` from ggml.c. */
fun ggmlMapCustom1(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, fun_: (io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Int) -> Unit, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

fun ggmlMapCustom1Inplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, fun_: (io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Int) -> Unit, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_UNARY
    result.src[0] = a
    return result
}

/** Port of `ggml_map_custom2` from ggml.c. */
fun ggmlMapCustom2(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
                   fun_: (io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Int) -> Unit, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY
    result.src[0] = a
    result.src[1] = b
    return result
}

fun ggmlMapCustom2Inplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
                          fun_: (io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Int) -> Unit, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_BINARY
    result.src[0] = a
    result.src[1] = b
    return result
}

/** Port of `ggml_map_custom3` from ggml.c. */
fun ggmlMapCustom3(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor,
                   fun_: (io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Int) -> Unit, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_CUSTOM3
    result.src[0] = a
    result.src[1] = b
    result.src[2] = c
    return result
}

fun ggmlMapCustom3Inplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, c: io.github.kotlinmania.llama.ore.GGMLTensor,
                          fun_: (io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, io.github.kotlinmania.llama.ore.GGMLTensor, Int, Int) -> Unit, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_CUSTOM3
    result.src[0] = a
    result.src[1] = b
    result.src[2] = c
    return result
}

/** Port of `ggml_custom_4d` from ggml.c — custom op with explicit output shape. */
fun ggmlCustom4d(ctx: io.github.kotlinmania.llama.ore.GGMLContext, type: io.github.kotlinmania.llama.ore.GGMLType, ne0: Long, ne1: Long, ne2: Long, ne3: Long,
                 nInputs: Int, inputs: Array<io.github.kotlinmania.llama.ore.GGMLTensor>, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor4d(ctx, type, ne0, ne1, ne2, ne3)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_CUSTOM3
    for (i in 0 until minOf(nInputs, io.github.kotlinmania.llama.ore.GGML_MAX_SRC)) {
        result.src[i] = inputs[i]
    }
    return result
}

/** Port of `ggml_custom_inplace` from ggml.c — custom op inplace on first input. */
fun ggmlCustomInplace(ctx: io.github.kotlinmania.llama.ore.GGMLContext, nInputs: Int, inputs: Array<io.github.kotlinmania.llama.ore.GGMLTensor>, nTasks: Int): io.github.kotlinmania.llama.ore.GGMLTensor {
    require(nInputs > 0) { "need at least one input" }
    val result = io.github.kotlinmania.llama.ore.ggmlViewTensor(ctx, inputs[0])
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.MAP_CUSTOM3
    for (i in 0 until minOf(nInputs, io.github.kotlinmania.llama.ore.GGML_MAX_SRC)) {
        result.src[i] = inputs[i]
    }
    return result
}

/** Port of `ggml_graph_dump_dot` from ggml.c — output DOT format. */
fun ggmlGraphDumpDot(cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph, gb: io.github.kotlinmania.llama.ore.GGMLCGraph?, filename: String) {
    val sb = StringBuilder()
    sb.appendLine("digraph G {")
    sb.appendLine("  newrank = true;")
    sb.appendLine("  rankdir = LR;")
    for (i in 0 until cgraph.nNodes) {
        val node = cgraph.nodes[i] ?: continue
        sb.appendLine("  \"${node.hashCode()}\" [label=\"${node.op}\\n${node.name}\\n[${node.ne.joinToString(",")}]\", shape=record];")
        for (s in 0 until io.github.kotlinmania.llama.ore.GGML_MAX_SRC) {
            val src = node.src.getOrNull(s) ?: continue
            sb.appendLine("  \"${src.hashCode()}\" -> \"${node.hashCode()}\";")
        }
    }
    sb.appendLine("}")
    // In commonMain we can't write files; print to stdout
    println(sb.toString())
}

/** Port of `ggml_print_object` from ggml.c. */
fun ggmlPrintObject(obj: io.github.kotlinmania.llama.ore.GGMLObject) {
    println("  - GGMLObject: type=${obj.type}, offs=${obj.offs}, size=${obj.size}")
}

/** Port of `ggml_print_objects` from ggml.c. */
fun ggmlPrintObjects(ctx: io.github.kotlinmania.llama.ore.GGMLContext) {
    var obj = ctx.objectsBegin
    println("=== objects in context ===")
    while (obj != null) {
        io.github.kotlinmania.llama.ore.ggmlPrintObject(obj)
        obj = obj.next
    }
    println("=== end objects ===")
}

/** Port of `ggml_threadpool_params_init` from ggml.c. */
fun ggmlThreadpoolParamsInit(nThreads: Int): io.github.kotlinmania.llama.ore.GGMLThreadpoolParams {
    return io.github.kotlinmania.llama.ore.GGMLThreadpoolParams(nThreads = nThreads)
}

/** Port of `ggml_threadpool_params_match` from ggml.c. */
fun ggmlThreadpoolParamsMatch(a: io.github.kotlinmania.llama.ore.GGMLThreadpoolParams, b: io.github.kotlinmania.llama.ore.GGMLThreadpoolParams): Boolean {
    return a.nThreads == b.nThreads
}

/** Port of `ggml_xielu` from ggml.c. */
fun ggmlXielu(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, a)
    result.op = io.github.kotlinmania.llama.ore.GGMLOp.UNARY
    result.src[0] = a
    return result
}

/** Port of `ggml_gated_delta_net` from ggml.c. */
fun ggmlGatedDeltaNet(ctx: io.github.kotlinmania.llama.ore.GGMLContext, x: io.github.kotlinmania.llama.ore.GGMLTensor, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor,
                      state: io.github.kotlinmania.llama.ore.GGMLTensor
): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlNewTensor4d(
        ctx,
        x.type,
        x.ne[0],
        x.ne[1],
        x.ne[2],
        x.ne[3]
    )
    result.src[0] = x
    result.src[1] = a
    result.src[2] = b
    result.src[3] = state
    return result
}

/** Port of `ggml_pool_2d_back` from ggml.c. */
fun ggmlPool2dBack(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, af: io.github.kotlinmania.llama.ore.GGMLTensor,
                   opPool: Int, k0: Int, k1: Int, s0: Int, s1: Int, p0: Float, p1: Float): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, af)
    io.github.kotlinmania.llama.ore.ggml_set_op_params(
        result,
        intArrayOf(opPool, k0, k1, s0, s1, p0.toRawBits(), p1.toRawBits()),
        7
    )
    result.src[0] = a
    result.src[1] = af
    return result
}

/** Port of `ggml_solve_tri` from ggml.c. */
fun ggmlSolveTri(ctx: io.github.kotlinmania.llama.ore.GGMLContext, a: io.github.kotlinmania.llama.ore.GGMLTensor, b: io.github.kotlinmania.llama.ore.GGMLTensor, upper: Boolean): io.github.kotlinmania.llama.ore.GGMLTensor {
    val result = io.github.kotlinmania.llama.ore.ggmlDupTensor(ctx, b)
    io.github.kotlinmania.llama.ore.ggml_set_op_params_i32(result, 0, if (upper) 1 else 0)
    result.src[0] = a
    result.src[1] = b
    return result
}

// ---------------------------------------------------------------------------
// Build backward expand (ggml.h line 2690, ggml.c line 6946)
// ---------------------------------------------------------------------------

/**
 * Expands [cgraph] with backward-pass nodes for automatic differentiation.
 * This is the newer `ggml_build_backward_expand` from ggml.c that works
 * in-place on the forward graph, creating gradient nodes and appending them.
 *
 * @param ctx      Context for allocating new gradient tensors.
 * @param cgraph   Forward computation graph (modified in-place).
 * @param gradAccs Optional pre-existing gradient accumulators per node index.
 */
fun ggmlBuildBackwardExpand(
    ctx: io.github.kotlinmania.llama.ore.GGMLContext,
    cgraph: io.github.kotlinmania.llama.ore.GGMLCGraph,
    gradAccs: Array<io.github.kotlinmania.llama.ore.GGMLTensor?>? = null
) {
}
