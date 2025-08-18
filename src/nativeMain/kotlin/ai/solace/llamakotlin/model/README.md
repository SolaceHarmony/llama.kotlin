# LLaMA Model Implementation

This directory contains a complete end-to-end implementation of the LLaMA transformer model for Kotlin/Native, providing all the components needed for text generation with advanced features like KV caching, sophisticated sampling strategies, and grammar-constrained generation.

## Architecture Overview

The implementation consists of several key components:

### Core Model Components

- **`LlamaAttention.kt`** - Multi-head attention mechanism with Rotary Position Encoding (RoPE)
- **`LlamaModel.kt`** - Complete LLaMA transformer architecture with decoder layers
- **`KVCache.kt`** - Key-Value cache system for efficient inference
- **`Sampling.kt`** - Advanced sampling strategies including temperature, top-k, top-p, Mirostat
- **`Grammar.kt`** - GBNF grammar parser and constrained generation system
- **`InferencePipeline.kt`** - End-to-end inference pipeline integrating all components

### Key Features

1. **Multi-Head Attention**
   - Scaled dot-product attention
   - Rotary Position Encoding (RoPE) for better positional understanding
   - Efficient computation with batch support

2. **KV Cache System**
   - Memory-efficient caching of keys and values
   - Supports multiple sequences/conversations
   - Automatic memory management

3. **Advanced Sampling**
   - Temperature scaling for creativity control
   - Top-k sampling for focused generation
   - Top-p (nucleus) sampling for balanced results
   - Mirostat v1 and v2 for perplexity control
   - Tail-free sampling for coherence
   - Locally typical sampling for naturalness
   - Comprehensive repetition penalties

4. **Grammar-Constrained Generation**
   - GBNF (Grammar-Based Next-token Filtering) support
   - Structured output generation (JSON, XML, etc.)
   - Custom grammar definition and parsing

5. **Production Features**
   - Streaming generation with token-by-token callbacks
   - Configurable stopping criteria
   - Comprehensive error handling
   - Memory usage optimization

## Usage Examples

### Basic Text Generation

```kotlin
val config = LlamaConfig(
    vocabSize = 32000,
    hiddenSize = 4096,
    numHiddenLayers = 32,
    numAttentionHeads = 32
)

val model = LlamaModel(config)
val tokenizer = SimpleTokenizer()

val pipeline = LlamaInferencePipelineBuilder()
    .model(model)
    .tokenizer(tokenizer)
    .samplingConfig(SamplingConfig(temperature = 0.8f))
    .build()

val result = pipeline.generate("Hello world", maxNewTokens = 50)
println(result.generatedText)
```

### Streaming Generation

```kotlin
pipeline.generateStream("Once upon a time", maxNewTokens = 100) { result ->
    print(result.tokenText)
    !result.isFinished // Continue until finished
}
```

### Grammar-Constrained Generation

```kotlin
val grammar = """
    root ::= object
    object ::= "{" pair ("," pair)* "}"
    pair ::= string ":" value
    string ::= "\"" [a-zA-Z]* "\""
    value ::= string | number
    number ::= [0-9]+
""".trimIndent()

val pipeline = LlamaInferencePipelineBuilder()
    .model(model)
    .tokenizer(tokenizer)
    .grammar(grammar)
    .build()

val result = pipeline.generate("Generate JSON:", maxNewTokens = 50)
// Output will be valid JSON according to the grammar
```

### Advanced Sampling Configuration

```kotlin
val samplingConfig = SamplingConfig(
    temperature = 0.7f,
    topK = 50,
    topP = 0.9f,
    penaltyRepeat = 1.1f,
    penaltyFreq = 0.1f,
    mirostat = 2,
    mirostatTau = 5.0f
)

val pipeline = LlamaInferencePipelineBuilder()
    .model(model)
    .tokenizer(tokenizer)
    .samplingConfig(samplingConfig)
    .build()
```

## Configuration

### LlamaConfig Parameters

- `vocabSize` - Size of the vocabulary (typically 32000 for LLaMA)
- `hiddenSize` - Hidden dimension size (4096 for LLaMA-7B)
- `numHiddenLayers` - Number of transformer layers (32 for LLaMA-7B)
- `numAttentionHeads` - Number of attention heads (32 for LLaMA-7B)
- `intermediateSize` - MLP intermediate size (11008 for LLaMA-7B)
- `maxPositionEmbeddings` - Maximum sequence length (2048 default)
- `ropeTheta` - RoPE theta parameter (10000.0 default)

### SamplingConfig Parameters

- `temperature` - Randomness control (0.0 = deterministic, 1.0 = balanced)
- `topK` - Keep only top K tokens (-1 = disabled)
- `topP` - Nucleus sampling threshold (1.0 = disabled)
- `penaltyRepeat` - Repetition penalty (1.0 = no penalty)
- `penaltyFreq` - Frequency penalty (0.0 = no penalty)
- `penaltyPresent` - Presence penalty (0.0 = no penalty)
- `mirostat` - Mirostat mode (0 = disabled, 1 = v1, 2 = v2)

## Integration with GGML

The model implementation integrates seamlessly with the existing GGML tensor operations:

- Uses existing memory allocation system (`GGMLGraphAllocator`)
- Leverages core tensor operations (`GGMLOp`)
- Compatible with quantization system
- Follows existing patterns for compute operations

See `GGMLIntegration.kt` for utilities that bridge model components with core GGML operations.

## Testing

Comprehensive test suite covers:

- Unit tests for each component
- Integration tests for end-to-end pipeline
- Performance and stress tests
- Grammar parsing and constraint tests
- Sampling strategy validation

Run tests with: `./gradlew allTests`

## Performance Considerations

1. **Memory Usage**
   - KV cache pre-allocates memory for efficiency
   - Tensor operations reuse allocated memory where possible
   - Graph-based computation minimizes temporary allocations

2. **Computation Optimization**
   - Matrix operations use optimized GGML kernels
   - Attention computation leverages existing infrastructure
   - Quantization support for memory/compute efficiency

3. **Scalability**
   - Supports various model sizes through configuration
   - Batch processing capabilities
   - Streaming generation for real-time applications

## Future Enhancements

The architecture is designed to support future improvements:

- Additional sampling strategies
- More sophisticated attention mechanisms
- Model parallelism and distributed inference
- Custom tokenizer implementations
- Extended grammar features

## Dependencies

The model implementation depends on:

- Core GGML tensor operations (`ai.solace.llamakotlin.core`)
- Kotlin coroutines for streaming
- Kotlin/Native standard library

No external dependencies are required beyond the existing project structure.