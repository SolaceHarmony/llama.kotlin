package io.github.kotlinmania.llama.model

/**
 * Minimal pipeline builder used by the InferenceExample programs.
 *
 * The full inference pipeline has not yet been ported; the wiring exists so
 * the example file compiles and the API surface mirrors what the eventual
 * port will expose.
 */
class LlamaInferencePipelineBuilder {
    private var model: LlamaModel? = null
    private var tokenizer: Tokenizer? = null
    private var samplingConfig: SamplingConfig = SamplingConfig()

    private var grammar: String? = null

    fun model(model: LlamaModel): LlamaInferencePipelineBuilder = apply { this.model = model }
    fun tokenizer(tokenizer: Tokenizer): LlamaInferencePipelineBuilder = apply { this.tokenizer = tokenizer }
    fun samplingConfig(cfg: SamplingConfig): LlamaInferencePipelineBuilder = apply { this.samplingConfig = cfg }
    fun grammar(grammar: String): LlamaInferencePipelineBuilder = apply { this.grammar = grammar }

    fun build(): LlamaInferencePipeline =
        LlamaInferencePipeline(
            model ?: error("LlamaInferencePipelineBuilder.build: model is required"),
            tokenizer ?: error("LlamaInferencePipelineBuilder.build: tokenizer is required"),
            samplingConfig
        )
}

/** Result of a single (non-streaming) generation call. */
data class LlamaGenerationResult(
    val prompt: String,
    val generatedText: String,
    val fullText: String,
    val finishReason: LlamaFinishReason,
    val inputTokens: List<Int>,
    val generatedTokens: List<Int>
)

/** Result of one streaming token callback. */
data class LlamaStreamResult(
    val tokenText: String,
    val isFinished: Boolean,
    val finishReason: LlamaFinishReason?
)

/** Why generation stopped. */
enum class LlamaFinishReason { LENGTH, STOP, EOS, ERROR }

/** Information about a configured model. */
data class LlamaModelInfo(val config: LlamaConfig) {
    val parameterCount: Long = 0L
    val maxSequenceLength: Int = config.maxPositionEmbeddings
    val vocabSize: Int = config.vocabSize
}

/**
 * Minimal inference pipeline. Full port pending — generate/generateStream
 * return the prompt and an empty completion until the pipeline is
 * implemented.
 */
class LlamaInferencePipeline internal constructor(
    private val model: LlamaModel,
    @Suppress("unused") private val tokenizer: Tokenizer,
    @Suppress("unused") private val samplingConfig: SamplingConfig
) {
    fun generate(
        prompt: String,
        maxNewTokens: Int = 0,
        stopTokens: List<String> = emptyList()
    ): LlamaGenerationResult = LlamaGenerationResult(
        prompt = prompt,
        generatedText = "",
        fullText = prompt,
        finishReason = LlamaFinishReason.STOP,
        inputTokens = emptyList(),
        generatedTokens = emptyList()
    )

    fun getModelInfo(): LlamaModelInfo = LlamaModelInfo(model.config)

    fun generateStream(
        prompt: String,
        maxNewTokens: Int = 0,
        callback: (LlamaStreamResult) -> Boolean
    ) {
        callback(LlamaStreamResult(tokenText = "", isFinished = true, finishReason = LlamaFinishReason.STOP))
    }
}
