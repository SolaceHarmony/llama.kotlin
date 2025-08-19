package ai.solace.llamakotlin.model

import ai.solace.llamakotlin.core.*

/**
 * End-to-end LLaMA inference pipeline.
 * Integrates model, sampling, KV cache, and grammar constraints.
 */
class LlamaInferencePipeline(
    val model: LlamaModel,
    val tokenizer: Tokenizer,
    val samplingConfig: SamplingConfig = SamplingConfig(),
    val grammarText: String? = null
) {
    private val context = GGMLContext()
    private val graphAllocator = GGMLGraphAllocator()
    private val samplingContext = SamplingContext(samplingConfig, model.config.vocabSize)
    
    // KV cache for each layer
    private val kvCaches = Array(model.config.numHiddenLayers) { 
        KVCache(
            maxSequenceLength = model.config.maxPositionEmbeddings,
            numHeads = model.config.numAttentionHeads,
            headDim = model.config.headDim
        )
    }
    
    // Grammar constraint context (if grammar is provided)
    private val grammarContext = grammarText?.let { text ->
        val grammar = GBNFGrammar.parse(text)
        GrammarSamplingContext(grammar, tokenizer)
    }
    
    init {
        // Initialize model parameters
        graphAllocator.reserve(1024 * 1024 * 1024) // 1GB buffer
        model.initializeParameters(graphAllocator)
        
        // Initialize KV caches
        for (cache in kvCaches) {
            cache.initialize(graphAllocator)
        }
    }
    
    /**
     * Generate text given an input prompt.
     */
    fun generate(
        prompt: String,
        maxNewTokens: Int = 100,
        stopTokens: List<String> = emptyList(),
        doSample: Boolean = true
    ): GenerationResult {
        // Encode prompt
        val inputIds = tokenizer.encode(prompt).toIntArray()
        val generatedTokens = mutableListOf<Int>()
        val allTokens = inputIds.toMutableList()
        
        // Reset contexts
        samplingContext.reset()
        grammarContext?.reset()
        for (cache in kvCaches) {
            cache.clear()
        }
        
        var isFinished = false
        var finishReason = FinishReason.MAX_TOKENS
        
        for (step in 0 until maxNewTokens) {
            if (isFinished) break
            
            // Determine input for this step
            val currentInput = if (step == 0) {
                // First step: use full prompt
                inputIds
            } else {
                // Subsequent steps: use only the last generated token
                intArrayOf(generatedTokens.last())
            }
            
            // Forward pass through model
            val logits = model.forward(
                context = context,
                graphAllocator = graphAllocator,
                inputIds = currentInput,
                attentionMask = null, // Simplified - could add attention masking
                kvCaches = kvCaches
            )
            
            // Get logits for the last position (for generation)
            val lastPositionLogits = extractLastPositionLogits(logits)
            
            // Sample next token
            val nextTokenId = sampleNextToken(lastPositionLogits, allTokens)
            
            // Check for stop conditions
            val nextTokenText = tokenizer.decode(nextTokenId)
            if (stopTokens.contains(nextTokenText)) {
                isFinished = true
                finishReason = FinishReason.STOP_TOKEN
                break
            }
            
            // Add generated token
            generatedTokens.add(nextTokenId)
            allTokens.add(nextTokenId)
            
            // Update grammar context if using grammar constraints
            grammarContext?.acceptToken(nextTokenId)
            if (grammarContext?.isComplete() == true) {
                isFinished = true
                finishReason = FinishReason.GRAMMAR_COMPLETE
                break
            }
        }
        
        // Decode generated tokens
        val generatedText = tokenizer.decode(generatedTokens)
        
        return GenerationResult(
            prompt = prompt,
            generatedText = generatedText,
            fullText = prompt + generatedText,
            inputTokens = inputIds.toList(),
            generatedTokens = generatedTokens,
            allTokens = allTokens,
            finishReason = finishReason
        )
    }
    
    /**
     * Extract logits for the last position in the sequence.
     */
    private fun extractLastPositionLogits(logits: GGMLTensor): GGMLTensor {
        val seqLen = logits.ne[1].toInt()
        val vocabSize = logits.ne[0].toInt()
        
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = vocabSize.toLong()
        for (i in 1 until GGML_MAX_DIMS) result.ne[i] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(result)
        
        // Copy logits from the last sequence position
        val lastPos = seqLen - 1
        for (vocabIdx in 0 until vocabSize) {
            val value = logits.getFloat(graphAllocator, vocabIdx, lastPos, 0)
            result.setFloat(graphAllocator, value, vocabIdx)
        }
        
        return result
    }
    
    /**
     * Sample the next token using configured sampling strategy.
     */
    private fun sampleNextToken(logits: GGMLTensor, context: List<Int>): Int {
        // Apply repetition penalties based on context
        val penaltyTokens = context.takeLast(64) // Use last 64 tokens for penalty
        
        // Sample token
        var nextTokenId = samplingContext.sample(
            graphAllocator = graphAllocator,
            logits = logits,
            penaltyTokens = penaltyTokens
        )
        
        // Apply grammar constraints if enabled
        grammarContext?.let { grammarCtx ->
            // Convert logits to candidates
            val vocabSize = logits.ne[0].toInt()
            val candidates = Array(vocabSize) { i ->
                TokenData(
                    id = i,
                    logit = logits.getFloat(graphAllocator, i),
                    prob = 0.0f
                )
            }
            
            // Filter candidates based on grammar
            val filteredCandidates = grammarCtx.filterCandidates(candidates)
            
            // If current token is not allowed by grammar, resample from filtered candidates
            if (filteredCandidates.none { it.id == nextTokenId }) {
                if (filteredCandidates.isNotEmpty()) {
                    // Create a new logits tensor with only valid tokens
                    val filteredLogits = GGMLTensor(type = GGMLType.F32)
                    filteredLogits.ne[0] = vocabSize.toLong()
                    for (i in 1 until GGML_MAX_DIMS) filteredLogits.ne[i] = 1L
                    filteredLogits.nb = calculateContiguousStrides(filteredLogits.ne, filteredLogits.type, GGML_MAX_DIMS)
                    
                    graphAllocator.allocateTensor(filteredLogits)
                    
                    // Set logits to negative infinity for invalid tokens
                    val validTokenIds = filteredCandidates.map { it.id }.toSet()
                    for (i in 0 until vocabSize) {
                        if (i in validTokenIds) {
                            filteredLogits.setFloat(graphAllocator, logits.getFloat(graphAllocator, i), i)
                        } else {
                            filteredLogits.setFloat(graphAllocator, Float.NEGATIVE_INFINITY, i)
                        }
                    }
                    
                    // Resample with filtered logits
                    nextTokenId = samplingContext.sample(
                        graphAllocator = graphAllocator,
                        logits = filteredLogits,
                        penaltyTokens = penaltyTokens
                    )
                }
            }
        }
        
        return nextTokenId
    }
    
    /**
     * Stream generation token by token.
     */
    fun generateStream(
        prompt: String,
        maxNewTokens: Int = 100,
        stopTokens: List<String> = emptyList(),
        callback: (StreamingResult) -> Boolean // Return true to continue, false to stop
    ) {
        val inputIds = tokenizer.encode(prompt).toIntArray()
        val generatedTokens = mutableListOf<Int>()
        val allTokens = inputIds.toMutableList()
        
        // Reset contexts
        samplingContext.reset()
        grammarContext?.reset()
        for (cache in kvCaches) {
            cache.clear()
        }
        
        var isFinished = false
        var finishReason = FinishReason.MAX_TOKENS
        
        for (step in 0 until maxNewTokens) {
            if (isFinished) break
            
            val currentInput = if (step == 0) {
                inputIds
            } else {
                intArrayOf(generatedTokens.last())
            }
            
            // Forward pass
            val logits = model.forward(
                context = context,
                graphAllocator = graphAllocator,
                inputIds = currentInput,
                attentionMask = null,
                kvCaches = kvCaches
            )
            
            val lastPositionLogits = extractLastPositionLogits(logits)
            val nextTokenId = sampleNextToken(lastPositionLogits, allTokens)
            val nextTokenText = tokenizer.decode(nextTokenId)
            
            // Check stop conditions
            if (stopTokens.contains(nextTokenText)) {
                isFinished = true
                finishReason = FinishReason.STOP_TOKEN
            }
            
            generatedTokens.add(nextTokenId)
            allTokens.add(nextTokenId)
            
            // Update grammar context
            grammarContext?.acceptToken(nextTokenId)
            if (grammarContext?.isComplete() == true) {
                isFinished = true
                finishReason = FinishReason.GRAMMAR_COMPLETE
            }
            
            // Call callback with current state
            val streamingResult = StreamingResult(
                tokenId = nextTokenId,
                tokenText = nextTokenText,
                generatedText = tokenizer.decode(generatedTokens),
                isFinished = isFinished,
                finishReason = if (isFinished) finishReason else null
            )
            
            val shouldContinue = callback(streamingResult)
            if (!shouldContinue) {
                isFinished = true
                finishReason = FinishReason.USER_STOP
                break
            }
            
            if (grammarContext?.isComplete() == true) {
                break
            }
        }
    }
    
    /**
     * Get model information.
     */
    fun getModelInfo(): ModelInfo {
        return ModelInfo(
            config = model.config,
            vocabSize = tokenizer.getVocabSize(),
            maxSequenceLength = model.config.maxPositionEmbeddings,
            parameterCount = estimateParameterCount()
        )
    }
    
    private fun estimateParameterCount(): Long {
        val config = model.config
        var params = 0L
        
        // Embeddings
        params += config.vocabSize * config.hiddenSize
        
        // Each layer
        params += config.numHiddenLayers * (
            // Attention weights
            4 * config.hiddenSize * config.hiddenSize +
            // MLP weights  
            config.hiddenSize * config.intermediateSize * 3 +
            // Layer norms
            config.hiddenSize * 2
        )
        
        // Final layer norm and output head
        params += config.hiddenSize + config.vocabSize * config.hiddenSize
        
        return params
    }
}

/**
 * Generation result containing full information about the generation.
 */
data class GenerationResult(
    val prompt: String,
    val generatedText: String,
    val fullText: String,
    val inputTokens: List<Int>,
    val generatedTokens: List<Int>,
    val allTokens: List<Int>,
    val finishReason: FinishReason
)

/**
 * Streaming generation result for token-by-token generation.
 */
data class StreamingResult(
    val tokenId: Int,
    val tokenText: String,
    val generatedText: String,
    val isFinished: Boolean,
    val finishReason: FinishReason?
)

/**
 * Model information.
 */
data class ModelInfo(
    val config: LlamaConfig,
    val vocabSize: Int,
    val maxSequenceLength: Int,
    val parameterCount: Long
)

/**
 * Reasons why generation finished.
 */
enum class FinishReason {
    MAX_TOKENS,
    STOP_TOKEN,
    GRAMMAR_COMPLETE,
    USER_STOP,
    ERROR
}

/**
 * Builder for creating inference pipelines with different configurations.
 */
class LlamaInferencePipelineBuilder {
    private var model: LlamaModel? = null
    private var tokenizer: Tokenizer? = null
    private var samplingConfig = SamplingConfig()
    private var grammarText: String? = null
    
    fun model(model: LlamaModel) = apply { this.model = model }
    fun tokenizer(tokenizer: Tokenizer) = apply { this.tokenizer = tokenizer }
    fun samplingConfig(config: SamplingConfig) = apply { this.samplingConfig = config }
    fun grammar(grammarText: String) = apply { this.grammarText = grammarText }
    
    fun build(): LlamaInferencePipeline {
        val model = this.model ?: throw IllegalStateException("Model is required")
        val tokenizer = this.tokenizer ?: throw IllegalStateException("Tokenizer is required")
        
        return LlamaInferencePipeline(
            model = model,
            tokenizer = tokenizer,
            samplingConfig = samplingConfig,
            grammarText = grammarText
        )
    }
}