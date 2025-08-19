# LLaMA Kotlin Examples

This directory contains comprehensive examples and demonstrations of the Kotlin llama.cpp port capabilities.

## ComprehensiveIntegrationDemo.kt

A complete showcase of the Kotlin port's advanced features including:

### 🔧 Technical Demonstrations:

1. **K-Quantization Support**
   - All K-Quant formats (Q2_K through Q8_K)
   - Quantization accuracy testing
   - Compression ratio analysis

2. **Advanced Tensor Operations**
   - Destination-based computation architecture
   - Matrix multiplication with mixed precision
   - Element-wise operations and activations

3. **Graph Optimization**
   - Dead code elimination
   - Redundant operation removal
   - Constant folding
   - Memory optimization

4. **GGUF Integration**
   - Model file parsing and loading
   - Tensor information extraction
   - Integration with core tensor system

5. **Mini Inference Pipeline**
   - End-to-end computation simulation
   - Quantized weight usage
   - Activation function application

### 🚀 Usage

The comprehensive demo is automatically run when you execute the main application:

```bash
./gradlew run
```

Or call directly:

```kotlin
import ai.solace.llamakotlin.examples.runComprehensiveDemo

fun main() {
    val result = runComprehensiveDemo()
    println(result)
}
```

### 📊 Expected Output

The demo provides detailed information about:
- Quantization accuracy metrics
- Tensor operation results
- Graph optimization statistics
- Model loading capabilities
- Inference pipeline execution

This demonstrates that the Kotlin port has achieved comprehensive functionality across all major areas of the llama.cpp implementation.