// port-lint: source llama.cpp/src/llama-impl.h llama.cpp/src/llama-impl.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import kotlin.time.TimeSource

// =============================================================================
// Logging infrastructure
// Ported from: llama-impl.h / llama-impl.cpp — logging macros, callback,
//              internal log dispatcher, and the global logger singleton.
// =============================================================================

/**
 * Log-level constants that mirror the GGML / llama.cpp convention.
 *
 * These extend the core [GGMLLogLevel] enum with the additional llama-specific
 * levels (`NONE` and `CONT`) that llama.cpp adds on top of GGML.
 */
enum class LlamaLogLevel {
    NONE,
    INFO,
    WARN,
    ERROR,
    DEBUG,
    /** Continuation — appends to the previous log line without a prefix. */
    CONT;
}

/**
 * Signature for a user-supplied log callback.
 *
 * Maps to `ggml_log_callback` (`void (*)(ggml_log_level, const char*, void*)`).
 * In Kotlin we drop the `void* user_data` pointer — callers can capture state
 * via a closure instead.
 */
typealias LlamaLogCallback = (level: LlamaLogLevel, text: String) -> Unit

/**
 * Global logger singleton.
 *
 * Port of the file-scoped `g_logger_state` in `llama-impl.cpp`.
 * Default behaviour writes to stderr (via [defaultLogCallback]).
 */
object LlamaLogger {
    var callback: LlamaLogCallback = ::defaultLogCallback
        private set

    /**
     * Install a custom log callback, or pass `null` to restore the default.
     *
     * Maps to `llama_log_set()` in the C++ API.
     */
    fun setCallback(cb: LlamaLogCallback?) {
        callback = cb ?: ::defaultLogCallback
    }

    /**
     * Retrieve the currently installed log callback.
     *
     * Maps to `llama_log_get()` in the C++ API.
     */
    fun getCallback(): LlamaLogCallback = callback
}

/**
 * Default log callback — prints to stderr, matching `llama_log_callback_default`.
 */
fun defaultLogCallback(level: LlamaLogLevel, text: String) {
    // C++ version writes to stderr via fputs.  In Kotlin/Native we use the
    // platform print functions.
    // Note: GGML_LOG_LEVEL_NONE is printed too (consistent with C++).
    val unused = level
    print(text) // stderr redirection can be done at platform level
}

// ---------------------------------------------------------------------------
// Convenience log helpers (replace C++ LLAMA_LOG_* macros)
// ---------------------------------------------------------------------------

/** General (no-level) log. Equivalent to `LLAMA_LOG(...)`. */
fun llamaLog(msg: String) = LlamaLogger.callback(LlamaLogLevel.NONE, msg)

/** Info-level log. Equivalent to `LLAMA_LOG_INFO(...)`. */
fun llamaLogInfo(msg: String) = LlamaLogger.callback(LlamaLogLevel.INFO, msg)

/** Warning-level log. Equivalent to `LLAMA_LOG_WARN(...)`. */
fun llamaLogWarn(msg: String) = LlamaLogger.callback(LlamaLogLevel.WARN, msg)

/** Error-level log. Equivalent to `LLAMA_LOG_ERROR(...)`. */
fun llamaLogError(msg: String) = LlamaLogger.callback(LlamaLogLevel.ERROR, msg)

/** Debug-level log. Equivalent to `LLAMA_LOG_DEBUG(...)`. */
fun llamaLogDebug(msg: String) = LlamaLogger.callback(LlamaLogLevel.DEBUG, msg)

/** Continuation-level log. Equivalent to `LLAMA_LOG_CONT(...)`. */
fun llamaLogCont(msg: String) = LlamaLogger.callback(LlamaLogLevel.CONT, msg)

/**
 * Internal log dispatcher with printf-style formatting.
 *
 * Port of `llama_log_internal()` / `llama_log_internal_v()` in `llama-impl.cpp`.
 * In Kotlin we accept a preformatted string — callers use string templates
 * instead of varargs.
 */
fun llamaLogInternal(level: LlamaLogLevel, text: String) {
    LlamaLogger.callback(level, text)
}

// =============================================================================
// Time helper
// Ported from: llama-impl.cpp — ggml_time_us()
// =============================================================================

/**
 * Return the current wall-clock time in microseconds.
 *
 * Port of `ggml_time_us()` / `llama_time_us()`.
 * Uses a monotonic clock baseline established at class-load time.
 */
fun llamaTimeUs(): Long {
    return LlamaTimeBaseline.mark.elapsedNow().inWholeMicroseconds
}

/** Baseline mark for [llamaTimeUs]. Initialized once at first access. */
private object LlamaTimeBaseline {
    val mark = TimeSource.Monotonic.markNow()
}

// =============================================================================
// TimeMeas – scoped time accumulator
// Ported from: llama-impl.h / llama-impl.cpp  struct time_meas
// =============================================================================

/**
 * Scoped wall-clock timer that accumulates elapsed microseconds into an
 * external counter.
 *
 * Usage mirrors the RAII `time_meas` in C++. Because Kotlin has no destructors
 * on regular classes, callers must invoke [stop] explicitly (or use [use]).
 *
 * ```kotlin
 * val tm = TimeMeas(myAccumulator)
 * // … do work …
 * tm.stop()
 * ```
 *
 * @param accumulator A lambda that receives the elapsed µs to add.
 * @param disabled    When `true`, the timer is a identity (matches `disable` param in C++).
 */
class TimeMeas(
    private val accumulator: (Long) -> Unit,
    disabled: Boolean = false,
) {
    private val mark = TimeSource.Monotonic.markNow()
    private val active = !disabled
    private var stopped = false

    /** Stop the timer and add elapsed µs to the accumulator. Idempotent. */
    fun stop() {
        if (active && !stopped) {
            stopped = true
            val elapsedUs = mark.elapsedNow().inWholeMicroseconds
            accumulator(elapsedUs)
        }
    }

    /** Convenience: execute [block], then [stop]. */
    inline fun <R> use(block: () -> R): R {
        try {
            return block()
        } finally {
            stop()
        }
    }
}

// =============================================================================
// BufferView – lightweight typed view over a memory region
// Ported from: llama-impl.h  template <typename T> struct buffer_view
// =============================================================================

/**
 * Non-owning view over a contiguous region of data.
 *
 * The C++ original is a raw-pointer + size pair.  In Kotlin we store a
 * [ByteArray] slice (offset + length) — the most common representation
 * used elsewhere in the codebase.
 *
 * @property data   Backing byte array (may be shared).
 * @property offset Start position within [data].
 * @property length Number of bytes in the view.
 */
data class BufferView(
    val data: ByteArray?,
    val offset: Int = 0,
    val length: Int = 0,
) {
    /** `true` when the view references valid data. */
    fun hasData(): Boolean = data != null && length > 0

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is BufferView) return false
        return offset == other.offset && length == other.length &&
            data.contentEquals(other.data)
    }

    override fun hashCode(): Int {
        var h = offset
        h = 31 * h + length
        h = 31 * h + (data?.contentHashCode() ?: 0)
        return h
    }
}

// =============================================================================
// String helpers
// Ported from: llama-impl.cpp  replace_all, format, llama_format_tensor_shape
// =============================================================================

/**
 * Replace every occurrence of [search] with [replace] in [s].
 *
 * Port of `replace_all()` in `llama-impl.cpp`.
 */
fun replaceAll(s: String, search: String, replace: String): String {
    if (search.isEmpty()) return s
    return s.replace(search, replace)
}

/**
 * Printf-style string formatting.
 *
 * Port of `format(const char * fmt, ...)` in `llama-impl.cpp`.
 * Kotlin/Native doesn't have `String.format`, so callers should use Kotlin
 * string templates directly. This helper is provided for compatibility —
 * it performs basic `%s`, `%d`, `%f` substitutions.
 */
fun llamaFormat(fmt: String, vararg args: Any?): String {
    if (args.isEmpty()) return fmt
    val sb = StringBuilder()
    var argIdx = 0
    var i = 0
    while (i < fmt.length) {
        if (fmt[i] == '%' && i + 1 < fmt.length && argIdx < args.size) {
            val next = fmt[i + 1]
            when (next) {
                's', 'd', 'f', 'g', 'e', 'i' -> {
                    sb.append(args[argIdx].toString())
                    argIdx++
                    i += 2
                    continue
                }
                '%' -> {
                    sb.append('%')
                    i += 2
                    continue
                }
                else -> {
                    // Skip numeric format width specifiers (e.g. %6d, %.2f)
                    var j = i + 1
                    while (j < fmt.length && (fmt[j].isDigit() || fmt[j] == '.' || fmt[j] == '-' || fmt[j] == '+')) j++
                    if (j < fmt.length && "sdfgei".contains(fmt[j])) {
                        sb.append(args[argIdx].toString())
                        argIdx++
                        i = j + 1
                        continue
                    }
                }
            }
        }
        sb.append(fmt[i])
        i++
    }
    return sb.toString()
}

/**
 * Format a tensor shape as a comma-separated dimension list, e.g. `"  4096,   128"`.
 *
 * Port of `llama_format_tensor_shape(const std::vector<int64_t>&)`.
 *
 * @param ne Dimension extents (e.g. `[4096, 128, 1, 1]`).
 */
fun llamaFormatTensorShape(ne: LongArray): String {
    return ne.joinToString(", ") { it.toString().padStart(6) }
}

/**
 * Overload that reads dimensions from a [GGMLTensor].
 *
 * Port of `llama_format_tensor_shape(const ggml_tensor*)`.
 */
fun llamaFormatTensorShape(t: GGMLTensor): String {
    return t.ne.joinToString(", ") { it.toString().padStart(6) }
}

// =============================================================================
// GGUF key-value helpers
// Ported from: llama-impl.cpp  gguf_data_to_str, gguf_kv_to_str
// =============================================================================

/**
 * GGUF data types used for key-value metadata.
 *
 * Port of `enum gguf_type` — only the types needed for [ggufDataToStr].
 */
enum class GGUFType(val value: Int) {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    UINT64(10),
    INT64(11),
    FLOAT32(6),
    FLOAT64(12),
    BOOL(7),
    STRING(8),
    ARRAY(9);
}

/**
 * Represents a single typed value from a GGUF metadata entry.
 *
 * In C++ this is raw pointer arithmetic; in Kotlin we box the value.
 */
sealed class GGUFValue {
    data class UInt8Val(val value: UByte) : GGUFValue()
    data class Int8Val(val value: Byte) : GGUFValue()
    data class UInt16Val(val value: UShort) : GGUFValue()
    data class Int16Val(val value: Short) : GGUFValue()
    data class UInt32Val(val value: UInt) : GGUFValue()
    data class Int32Val(val value: Int) : GGUFValue()
    data class UInt64Val(val value: ULong) : GGUFValue()
    data class Int64Val(val value: Long) : GGUFValue()
    data class Float32Val(val value: Float) : GGUFValue()
    data class Float64Val(val value: Double) : GGUFValue()
    data class BoolVal(val value: Boolean) : GGUFValue()
    data class StringVal(val value: String) : GGUFValue()
    data class ArrayVal(val type: GGUFType, val values: List<GGUFValue>) : GGUFValue()
}

// port-lint: source llama.cpp/src/llama-impl.cpp gguf_data_to_str
/**
 * Convert a single [GGUFValue] to its string representation.
 *
 * Port of `gguf_data_to_str()` in `llama-impl.cpp`.
 */
fun ggufDataToStr(value: GGUFValue): String = when (value) {
    is GGUFValue.UInt8Val   -> value.value.toString()
    is GGUFValue.Int8Val    -> value.value.toString()
    is GGUFValue.UInt16Val  -> value.value.toString()
    is GGUFValue.Int16Val   -> value.value.toString()
    is GGUFValue.UInt32Val  -> value.value.toString()
    is GGUFValue.Int32Val   -> value.value.toString()
    is GGUFValue.UInt64Val  -> value.value.toString()
    is GGUFValue.Int64Val   -> value.value.toString()
    is GGUFValue.Float32Val -> value.value.toString()
    is GGUFValue.Float64Val -> value.value.toString()
    is GGUFValue.BoolVal    -> if (value.value) "true" else "false"
    is GGUFValue.StringVal  -> value.value
    is GGUFValue.ArrayVal   -> value.values.joinToString(", ", "[", "]") { v ->
        if (v is GGUFValue.StringVal) {
            val escaped = v.value.replace("\\", "\\\\").replace("\"", "\\\"")
            "\"$escaped\""
        } else {
            ggufDataToStr(v)
        }
    }
}

/**
 * Convert a GGUF key-value entry to a human-readable string.
 *
 * Port of `gguf_kv_to_str()` in `llama-impl.cpp`.
 * The caller is responsible for extracting the [GGUFValue] from their GGUF
 * context implementation.
 */
fun ggufKvToStr(value: GGUFValue): String = when (value) {
    is GGUFValue.StringVal -> value.value
    is GGUFValue.ArrayVal  -> ggufDataToStr(value)
    else                   -> ggufDataToStr(value)
}

// =============================================================================
// System info
// Ported from: llama-impl.cpp — llama_print_system_info()
// =============================================================================

// NOTE: llamaPrintSystemInfo() is defined in LlamaApi.kt to avoid duplication.

// =============================================================================
// Well-known internal tensor names
// Ported from: llama-impl.h  LLAMA_TENSOR_NAME_* macros
// =============================================================================

/** Sentinel tensor name used for flash-attention intermediates. */
const val LLAMA_TENSOR_NAME_FATTN = "__fattn__"

/** Sentinel tensor name for fused GDN autoregressive intermediates. */
const val LLAMA_TENSOR_NAME_FGDN_AR = "__fgdn_ar__"

/** Sentinel tensor name for fused GDN channel intermediates. */
const val LLAMA_TENSOR_NAME_FGDN_CH = "__fgdn_ch__"
