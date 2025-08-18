package ai.solace.llamakotlin.examples

import ai.solace.llamakotlin.core.*
import ai.solace.llamakotlin.model.*

/**
 * Example demonstrating the end-to-end LLaMA model pipeline usage.
 * This shows how to create and use the complete inference system.
 */
object LlamaInferenceExample {
    
    /**
     * Basic text generation example.
     */
    fun basicGenerationExample() {
        println("=== Basic Text Generation Example ===")
        
        // Create a small model configuration for demonstration
        val config = LlamaConfig(
            vocabSize = 1000,
            hiddenSize = 256,
            intermediateSize = 512,
            numHiddenLayers = 6,
            numAttentionHeads = 8,
            maxPositionEmbeddings = 1024
        )
        
        // Create model and tokenizer
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        // Configure sampling parameters
        val samplingConfig = SamplingConfig(
            temperature = 0.8f,
            topK = 50,
            topP = 0.9f,
            penaltyRepeat = 1.1f,
            seed = 42
        )
        
        // Build the inference pipeline
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .samplingConfig(samplingConfig)
            .build()
        
        // Generate text
        val prompt = "The weather today is"
        val result = pipeline.generate(
            prompt = prompt,
            maxNewTokens = 20,
            stopTokens = listOf(".", "!")
        )
        
        // Display results
        println("Prompt: ${result.prompt}")
        println("Generated: ${result.generatedText}")
        println("Full text: ${result.fullText}")
        println("Finish reason: ${result.finishReason}")
        println("Input tokens: ${result.inputTokens.size}")
        println("Generated tokens: ${result.generatedTokens.size}")
        println()
    }
    
    /**
     * Streaming generation example.
     */
    fun streamingGenerationExample() {
        println("=== Streaming Generation Example ===")
        
        val config = LlamaConfig(
            vocabSize = 500,
            hiddenSize = 128,
            intermediateSize = 256,
            numHiddenLayers = 4,
            numAttentionHeads = 4
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .samplingConfig(SamplingConfig(temperature = 0.7f, seed = 123))
            .build()
        
        print("Generating: ")
        
        pipeline.generateStream(
            prompt = "Once upon a time",
            maxNewTokens = 15
        ) { result ->
            print(result.tokenText)
            
            if (result.isFinished) {
                println()
                println("Finished: ${result.finishReason}")
            }
            
            true // Continue generation
        }
        
        println()
    }
    
    /**
     * Grammar-constrained generation example.
     */
    fun grammarConstrainedExample() {
        println("=== Grammar-Constrained Generation Example ===")
        
        val config = LlamaConfig(
            vocabSize = 200,
            hiddenSize = 64,
            intermediateSize = 128,
            numHiddenLayers = 2,
            numAttentionHeads = 4
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        // Define a simple grammar for JSON-like output
        val grammar = """
            root ::= object
            object ::= "{" pair ("," pair)* "}"
            pair ::= string ":" value
            string ::= "\"" [a-zA-Z]* "\""
            value ::= string | number | "true" | "false"
            number ::= [0-9]+
        """.trimIndent()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .grammar(grammar)
            .samplingConfig(SamplingConfig(temperature = 0.5f))
            .build()
        
        val result = pipeline.generate(
            prompt = "Generate a JSON object:",
            maxNewTokens = 30
        )
        
        println("Prompt: ${result.prompt}")
        println("Generated (grammar-constrained): ${result.generatedText}")
        println("Finish reason: ${result.finishReason}")
        println()
    }
    
    /**
     * Advanced sampling configuration example.
     */
    fun advancedSamplingExample() {
        println("=== Advanced Sampling Configuration Example ===")
        
        val config = LlamaConfig(
            vocabSize = 300,
            hiddenSize = 96,
            intermediateSize = 192,
            numHiddenLayers = 3,
            numAttentionHeads = 6
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        // Test different sampling strategies
        val samplingConfigs = listOf(
            SamplingConfig(temperature = 0.1f, seed = 1) to "Low temperature (focused)",
            SamplingConfig(temperature = 1.5f, seed = 1) to "High temperature (creative)",
            SamplingConfig(topK = 10, seed = 1) to "Top-K sampling",
            SamplingConfig(topP = 0.7f, seed = 1) to "Top-P sampling",
            SamplingConfig(mirostat = 2, mirostatTau = 5.0f, seed = 1) to "Mirostat v2"
        )
        
        val prompt = "The most important thing about AI is"
        
        for ((config, description) in samplingConfigs) {
            val pipeline = LlamaInferencePipelineBuilder()
                .model(model)
                .tokenizer(tokenizer)
                .samplingConfig(config)
                .build()
            
            val result = pipeline.generate(prompt, maxNewTokens = 10)
            
            println("$description:")
            println("  Generated: ${result.generatedText}")
            println()
        }
    }
    
    /**
     * Model information and statistics example.
     */
    fun modelInfoExample() {
        println("=== Model Information Example ===")
        
        val config = LlamaConfig(
            vocabSize = 32000,
            hiddenSize = 4096,
            intermediateSize = 11008,
            numHiddenLayers = 32,
            numAttentionHeads = 32,
            maxPositionEmbeddings = 2048
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        val modelInfo = pipeline.getModelInfo()
        
        println("Model Configuration:")
        println("  Vocabulary Size: ${modelInfo.config.vocabSize:,}")
        println("  Hidden Size: ${modelInfo.config.hiddenSize:,}")
        println("  Intermediate Size: ${modelInfo.config.intermediateSize:,}")
        println("  Number of Layers: ${modelInfo.config.numHiddenLayers}")
        println("  Number of Attention Heads: ${modelInfo.config.numAttentionHeads}")
        println("  Head Dimension: ${modelInfo.config.headDim}")
        println("  Max Position Embeddings: ${modelInfo.config.maxPositionEmbeddings:,}")
        println("  RoPE Theta: ${modelInfo.config.ropeTheta}")
        println()
        
        println("Model Statistics:")
        println("  Estimated Parameters: ${modelInfo.parameterCount:,}")
        println("  Max Sequence Length: ${modelInfo.maxSequenceLength:,}")
        println("  Tokenizer Vocab Size: ${modelInfo.vocabSize:,}")
        println()
    }
    
    /**
     * KV Cache usage example.
     */
    fun kvCacheExample() {
        println("=== KV Cache Usage Example ===")
        
        val config = LlamaConfig(
            vocabSize = 100,
            hiddenSize = 32,
            intermediateSize = 64,
            numHiddenLayers = 2,
            numAttentionHeads = 4
        )
        
        val model = LlamaModel(config)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        println("Generating with KV cache enabled (default)...")
        
        // First generation - cache will be populated
        val result1 = pipeline.generate("Hello", maxNewTokens = 5)
        println("First generation: ${result1.generatedText}")
        
        // Subsequent tokens benefit from cached keys/values
        val result2 = pipeline.generate("Hello world", maxNewTokens = 3)
        println("Second generation: ${result2.generatedText}")
        
        println("KV cache allows efficient generation of continuation tokens.")
        println()
    }
    
    /**
     * Error handling example.
     */
    fun errorHandlingExample() {
        println("=== Error Handling Example ===")
        
        // Show how to handle various error conditions
        
        // 1. Invalid model configuration
        try {
            LlamaConfig(hiddenSize = 13, numAttentionHeads = 4) // Not divisible
        } catch (e: IllegalArgumentException) {
            println("Caught expected error: ${e.message}")
        }
        
        // 2. Missing required components
        try {
            LlamaInferencePipelineBuilder().build()
        } catch (e: IllegalStateException) {
            println("Caught expected error: ${e.message}")
        }
        
        // 3. Invalid sampling parameters (this would be caught at runtime)
        val validConfig = LlamaConfig(
            vocabSize = 50,
            hiddenSize = 16,
            numAttentionHeads = 2,
            numHiddenLayers = 1,
            intermediateSize = 32
        )
        
        val model = LlamaModel(validConfig)
        val tokenizer = SimpleTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(tokenizer)
            .build()
        
        // This will work but might not generate optimal results
        val result = pipeline.generate(
            prompt = "test",
            maxNewTokens = 2
        )
        
        println("Generation completed with finish reason: ${result.finishReason}")
        println()
    }
    
    /**
     * Custom tokenizer example.
     */
    fun customTokenizerExample() {
        println("=== Custom Tokenizer Example ===")
        
        // Create a custom tokenizer with specific vocabulary
        class CustomTokenizer : Tokenizer {
            private val vocab = mapOf(
                "hello" to 1,
                "world" to 2,
                "the" to 3,
                "is" to 4,
                "great" to 5,
                " " to 6,
                "!" to 7,
                "." to 8,
                "<UNK>" to 0
            )
            
            private val reverseVocab = vocab.entries.associate { it.value to it.key }
            
            override fun encode(text: String): List<Int> {
                return text.split(" ").map { token ->
                    vocab[token] ?: vocab["<UNK>"]!!
                }
            }
            
            override fun decode(tokenId: Int): String {
                return reverseVocab[tokenId] ?: "<UNK>"
            }
            
            override fun decode(tokenIds: List<Int>): String {
                return tokenIds.joinToString(" ") { decode(it) }
            }
            
            override fun getVocabSize(): Int = vocab.size
        }
        
        val config = LlamaConfig(
            vocabSize = 9, // Match custom tokenizer size
            hiddenSize = 16,
            numAttentionHeads = 2,
            numHiddenLayers = 1,
            intermediateSize = 32
        )
        
        val model = LlamaModel(config)
        val customTokenizer = CustomTokenizer()
        
        val pipeline = LlamaInferencePipelineBuilder()
            .model(model)
            .tokenizer(customTokenizer)
            .build()
        
        val result = pipeline.generate("hello world", maxNewTokens = 3)
        
        println("Using custom tokenizer:")
        println("  Prompt: ${result.prompt}")
        println("  Generated: ${result.generatedText}")
        println("  Input token IDs: ${result.inputTokens}")
        println("  Generated token IDs: ${result.generatedTokens}")
        println()
    }
}

/**
 * Main function to run all examples.
 * Note: In a real application, model parameters would be loaded from files
 * rather than being randomly initialized.
 */
fun runInferenceExamples() {
    println("LLaMA.kotlin Inference Pipeline Examples")
    println("========================================")
    println()
    
    // Note: These examples use randomly initialized models,
    // so the generated text will be mostly nonsensical.
    // In a real application, you would load trained model weights.
    
    println("Note: Models are randomly initialized for demonstration.")
    println("In practice, you would load trained weights from model files.")
    println()
    
    LlamaInferenceExample.modelInfoExample()
    LlamaInferenceExample.basicGenerationExample()
    LlamaInferenceExample.streamingGenerationExample()
    LlamaInferenceExample.grammarConstrainedExample()
    LlamaInferenceExample.advancedSamplingExample()
    LlamaInferenceExample.kvCacheExample()
    LlamaInferenceExample.customTokenizerExample()
    LlamaInferenceExample.errorHandlingExample()
    
    println("All examples completed successfully!")
}