package ai.solace.llamakotlin.core

import kotlin.native.concurrent.SharedImmutable

/**
 * Kotlin Native port of GGML tensor library core data types.
 * This file contains the core data structures used in the GGML library.
 */

/**
 * Maximum number of dimensions in a tensor
 */
const val GGML_MAX_DIMS = 4

/**
 * Maximum number of source tensors for an operation
 */
const val GGML_MAX_SRC = 10

/**
 * Maximum number of operation parameters
 */
const val GGML_MAX_OP_PARAMS = 32

/**
 * Maximum name length for a tensor
 */
const val GGML_MAX_NAME = 64

/**
 * Tensor data types
 */
enum class GGMLType {
    F32,    // 32-bit float
    F16,    // 16-bit float
    Q4_0,   // 4-bit quantized
    Q4_1,   // 4-bit quantized with different scaling
    Q5_0,   // 5-bit quantized
    Q5_1,   // 5-bit quantized with different scaling
    Q8_0,   // 8-bit quantized
    Q8_1,   // 8-bit quantized with different scaling
    Q2_K,   // 2-bit quantized for K-quants
    Q3_K,   // 3-bit quantized for K-quants
    Q4_K,   // 4-bit quantized for K-quants
    Q5_K,   // 5-bit quantized for K-quants
    Q6_K,   // 6-bit quantized for K-quants
    Q8_K,   // 8-bit quantized for K-quants
    I8,     // 8-bit integer
    I16,    // 16-bit integer
    I32,    // 32-bit integer
    I64,    // 64-bit integer
    COUNT   // Number of types
}

/**
 * Tensor operations
 */
enum class GGMLOp {
    NONE,
    DUP,
    ADD,
    SUB,
    MUL,
    DIV,
    SQR,
    SQRT,
    SUM,
    MEAN,
    REPEAT,
    ABS,
    SGN,
    NEG,
    STEP,
    RELU,
    GELU,
    SILU,
    NORM,
    RMS_NORM,
    MUL_MAT,
    SCALE,
    CPY,
    RESHAPE,
    VIEW,
    PERMUTE,
    TRANSPOSE,
    GET_ROWS,
    DIAG_MASK_INF,
    SOFT_MAX,
    ROPE,
    CONV_1D_1S,
    CONV_1D_2S,
    FLASH_ATTN,
    FLASH_FF,
    MAP_UNARY,
    MAP_BINARY,
    COUNT
}

/**
 * Computation graph evaluation order
 */
enum class GGMLCGraphEvalOrder {
    NONE,
    FORWARD,
    BACKWARD
}

/**
 * 16-bit brain floating point type
 */
data class GGMLBF16(val bits: UShort)

/**
 * Base object structure
 */
class GGMLObject(
    var offset: ULong = 0u,
    var size: ULong = 0u,
    var next: GGMLObject? = null
)

/**
 * Tensor data structure
 */
class GGMLTensor(
    var type: GGMLType = GGMLType.F32,
    var buffer: Any? = null,
    var ne: LongArray = LongArray(GGML_MAX_DIMS) { 0L },
    var nb: ULongArray = ULongArray(GGML_MAX_DIMS) { 0u },
    var op: GGMLOp = GGMLOp.NONE,
    var opParams: IntArray = IntArray(GGML_MAX_OP_PARAMS / Int.SIZE_BYTES) { 0 },
    var flags: Int = 0,
    var grad: GGMLTensor? = null,
    var src: Array<GGMLTensor?> = Array(GGML_MAX_SRC) { null },
    var viewSrc: GGMLTensor? = null,
    var viewOffs: ULong = 0u,
    var data: Any? = null,
    var name: String = ""
)

/**
 * Scratch buffer for temporary storage
 */
class GGMLScratch(
    var offs: ULong = 0u,
    var size: ULong = 0u,
    var data: Any? = null
)

/**
 * Context for GGML operations
 */
class GGMLContext(
    var memSize: ULong = 0u,
    var memBuffer: Any? = null,
    var memBufferOwned: Boolean = false,
    var noAlloc: Boolean = false,
    var noAllocSave: Boolean = false,
    var nObjects: Int = 0,
    var objectsBegin: GGMLObject? = null,
    var objectsEnd: GGMLObject? = null,
    var scratch: GGMLScratch = GGMLScratch(),
    var scratchSave: GGMLScratch = GGMLScratch(),
    var computeImmediately: Boolean = true
)

/**
 * Parameters for initializing the GGML context
 */
class GGMLInitParams(
    var memSize: ULong = 0u,
    var memBuffer: Any? = null,
    var noAlloc: Boolean = false
)

/**
 * Computation graph
 */
class GGMLCGraph(
    var size: Int = 0,
    var nNodes: Int = 0,
    var nLeafs: Int = 0,
    var nodes: Array<GGMLTensor?> = emptyArray(),
    var grads: Array<GGMLTensor?> = emptyArray(),
    var leafs: Array<GGMLTensor?> = emptyArray(),
    var visitedHashSet: Any? = null,
    var order: GGMLCGraphEvalOrder = GGMLCGraphEvalOrder.NONE
)
