package io.github.kotlinmania.llama.ore

import kotlin.js.ExperimentalJsExport
import kotlin.js.JsExport

@OptIn(ExperimentalJsExport::class)
@JsExport
class EmberTensor internal constructor(private val tensor: io.github.kotlinmania.llama.ore.GGMLTensor) {
    val name: String get() = tensor.name
    val type: String get() = tensor.type.name
    val shape: Array<Long> get() = tensor.ne.copyOf().toTypedArray()
}

@OptIn(ExperimentalJsExport::class)
@JsExport
object Ember {
    fun tensorF32(name: String, vararg dims: Int, value: Float = 0f): EmberTensor {
        require(dims.isNotEmpty()) { "Tensor shape must have at least one dimension" }
        val shape = LongArray(io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { 1L }
        dims.forEachIndexed { index, dim ->
            require(index < io.github.kotlinmania.llama.ore.GGML_MAX_DIMS) { "Rank greater than ${io.github.kotlinmania.llama.ore.GGML_MAX_DIMS} is not supported" }
            shape[index] = dim.toLong()
        }
        val tensor = io.github.kotlinmania.llama.ore.GGMLTensor(
            type = io.github.kotlinmania.llama.ore.GGMLType.F32,
            name = name,
            ne = shape
        )
        tensor.data = FloatArray(tensor.numElements().toInt()) { value }
        return EmberTensor(tensor)
    }
}
