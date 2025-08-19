# llama.kotlin

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A Kotlin/Native port of [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference of Meta's [LLaMA](https://arxiv.org/abs/2302.13971) model (and others) in pure Kotlin**

*Maintained by Sydney ([The Solace Project](https://github.com/SolaceHarmony))*

## Acknowledgments

This project is a Kotlin/Native port of the original [llama.cpp](https://github.com/ggerganov/llama.cpp) created by [Georgi Gerganov](https://github.com/ggerganov) and the amazing open source community. We extend our deepest gratitude to all the original contributors who made the foundational work possible. Please see the [AUTHORS](AUTHORS) file for the complete list of contributors to the original project.

**Original llama.cpp contributors deserve special recognition for their groundbreaking work in making LLM inference efficient and accessible.**

## About This Port

This is a **Kotlin/Native implementation** of llama.cpp, designed to bring the power of large language model inference to the Kotlin ecosystem with a focus on:

- **CPU and Apple Metal backends** for optimal performance on supported hardware
- **Idiomatic Kotlin API** while maintaining compatibility with original concepts
- **Memory-efficient tensor operations** adapted for Kotlin/Native's memory model
- **Comprehensive quantization support** (Q8_0, Q4_0, Q4_1, with K-Quant types in progress)
- **Automatic differentiation** for training and fine-tuning capabilities

## Current Status - Phase 2 (Core Library Translation)

The project is actively under development with significant progress in core areas:

### ✅ Completed Features
- **Memory Management**: Efficient tensor allocation with `GGMLGraphAllocator` and `GGMLDynTensorAllocator`
  - Primary ByteArray buffer with dynamic allocation within reserved space
  - Inplace tensor allocation and memory reuse logic for optimization
  - Tensor usage tracking and automatic memory freeing
- **Tensor Data Access**: Comprehensive accessor methods for all supported data types
  - F32, F16, I32, I16 data accessors with stride information
  - Efficient ByteArray-based data storage and retrieval
- **Quantization Support**: Multiple quantization formats implemented
  - Q8_0: F16 scale + 32xI8 weights (34 bytes per block)
  - Q4_0: F16 scale + 32x4-bit packed weights (18 bytes per block)  
  - Q4_1: 2x F16 scale/min + 32x4-bit packed weights (20 bytes per block)
  - Optimized dot product routines for quantized operations
- **Core Tensor Operations**: Element-wise and matrix operations with multi-type support
  - ADD, MUL, SUB, DIV, NEG, MatMul for F32/F16 and quantized types
  - Activation functions: RELU, GELU, SILU, RMSNorm
- **Automatic Differentiation**: Backward pass implementation for core operations
- **Compute Operations Architecture**: Refactored to destination-based operations
  - All compute functions write directly into allocator-managed buffers
  - Eliminated redundant memory allocations and improved efficiency
  - Aligned with GGML architecture for memory reuse and graph optimization

### 🔄 In Progress  
- **Computation Graph Optimization**: Graph optimization passes for redundant operation removal
- **Additional Quantization**: K-Quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- **CPU Backend**: Formal CPU backend structure and multi-threading support

### 📋 Testing Infrastructure
- Comprehensive unit tests for core operations under `src/nativeTest/kotlin`
- Quantization accuracy tests with MSE and MAD validation
- Memory allocator tests for graph-level memory planning
- **Destination-based compute operations test suite** (`GGMLComputeOpsDestinationTest.kt`)
  - Validates new in-place computation interface with pre-allocated destination tensors
  - Tests dimension and type mismatch error handling
  - Verifies direct integration with graph allocator memory management

## Project Documentation

For detailed development information, see:
- [**KOTLIN_PORT_CHECKLIST.md**](KOTLIN_PORT_CHECKLIST.md) - Detailed development roadmap with current progress
- [**KOTLIN_PORT_STATUS.md**](KOTLIN_PORT_STATUS.md) - Overall project status and completion overview  
- [**COMPUTE_OPERATIONS_REFACTOR_SUMMARY.md**](COMPUTE_OPERATIONS_REFACTOR_SUMMARY.md) - Major architectural refactor details
- [**GGML_COMPUTE_OPS_DESIGN.md**](GGML_COMPUTE_OPS_DESIGN.md) - Technical design for computation operations
- [**TENSOR_OPERATIONS_DESIGN.md**](TENSOR_OPERATIONS_DESIGN.md) - Design patterns for tensor operations
- [**AGENTS.md**](AGENTS.md) - Project instructions for contributors and agents

## Build and Development

This project uses **Kotlin Multiplatform** with **Gradle** as the build system.

### Prerequisites
- **Kotlin/Native** compiler and tools
- **Gradle 8.13** or later
- **macOS** (current target platform - x64 and arm64)

### Building the Project
```bash
./gradlew build
```

### Running Tests
```bash
./gradlew allTests
```

*Note: The original C/C++ sources remain in the repository during the porting process but are not required for the Kotlin build.*

## Architecture Overview

- **Source Directory**: `src/nativeMain/kotlin/ai/solace/llamakotlin`
- **Package Structure**: `ai.solace.llamakotlin.*`
- **Core Modules**:
  - `core/GGMLTypes.kt` - Core tensor data structures
  - `core/GGMLAlloc.kt` - Memory management
  - `core/GGMLOps.kt` - High-level tensor operations
  - `core/GGMLComputeOps.kt` - Low-level computation kernels
  - `core/GGMLGraph.kt` - Computation graph execution
  - `core/GGMLQuants.kt` - Quantization implementations

## Supported Platforms

- **macOS** (x64 and arm64) - Primary target with Metal backend support
- **Other Kotlin/Native targets** - Planned for future releases


## Future Model Support (Planned)

As the Kotlin port develops, we plan to support the same range of models as the original llama.cpp:

**Large Language Models:**
- LLaMA / LLaMA 2 / LLaMA 3 🦙
- Mistral 7B / Mixtral MoE
- GPT-2, Phi models, Gemma
- And many more from the original llama.cpp ecosystem

**Multimodal Models (Future):**
- LLaVA models
- Other vision-language models

*Model support will be added progressively as the core Kotlin implementation matures.*

## Contributing to llama.kotlin

We welcome contributions to the Kotlin port! Here's how you can help:

### Development Focus Areas
- **Core tensor operations** and optimization
- **Quantization methods** implementation
- **CPU backend** development and optimization  
- **Metal backend** for Apple Silicon
- **Testing and validation** against original implementations
- **Documentation** and examples

### Development Guidelines
- **Kotlin Style**: Use idiomatic Kotlin with descriptive names and comprehensive KDoc comments
- **Modular Design**: Separate tensor creation logic from compute kernels
- **Memory Efficiency**: Use ByteArray-based storage with accessor methods
- **Type Safety**: Leverage Kotlin's type system for compile-time safety
- **Testing**: Include comprehensive tests for new functionality

### Getting Started
1. Check the current progress in [KOTLIN_PORT_CHECKLIST.md](KOTLIN_PORT_CHECKLIST.md)
2. Review the [AGENTS.md](AGENTS.md) file for detailed development guidance
3. Look at existing implementations in `src/nativeMain/kotlin/ai/solace/llamakotlin/`
4. Add tests in `src/nativeTest/kotlin/`

## Relationship to Original llama.cpp

This project maintains **conceptual compatibility** with the original [llama.cpp](https://github.com/ggerganov/llama.cpp) while providing a **Kotlin/Native implementation**. 

- **Same core concepts**: Tensor operations, quantization methods, and model architectures
- **Compatible file formats**: Plans to support GGUF and other formats from the original
- **Independent development**: Optimized for Kotlin/Native's strengths and constraints
- **Complementary goals**: Different language ecosystem, same inference capabilities

## Advanced Research (Future Work)

Beyond the core llama.cpp port, this project explores cutting-edge hybrid architectures:
- **Liquid Neural Networks Integration**: Actor-based computation with adaptive time constants
- **Quantum-Classical Hybrid Framework**: Meta-Cognitive Temporal Architecture (MCTA) research  
- **Memory Cube Architecture**: Advanced caching and inference optimization strategies
- **Actor-Coroutine Integration**: Leveraging Kotlin's concurrency model for neural computation

*These are research directions for future exploration after completing the core llama.cpp port.*

## Development Status Examples

The following examples show the current capabilities of the Kotlin port:

### Tensor Operations
```kotlin
// Basic tensor creation and operations
val tensorA = ggmlNewTensor2D(ctx, GGMLType.GGML_TYPE_F32, 4, 4)
val tensorB = ggmlNewTensor2D(ctx, GGMLType.GGML_TYPE_F32, 4, 4)
val result = ggmlAdd(ctx, tensorA, tensorB)
```

### Memory Management
```kotlin
// Use GGMLGraphAllocator for efficient memory planning
val allocator = GGMLGraphAllocator()
allocator.reserve(graph, bufferSize)
val tensor = allocator.allocateTensor(type, dimensions) // Automatically uses inplace when possible
```

### Compute Operations (New Destination-based Architecture)
```kotlin
// New efficient destination-based compute operations
val src0 = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 4))
val src1 = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 4))
val dst = allocator.allocateTensor(GGMLType.F32, longArrayOf(4, 4))

// Operations write directly into pre-allocated destination
computeAdd(allocator, context, src0, src1, dst) // No return value - writes to dst
```

### Quantization
```kotlin
// Q4_0 quantization example
val quantizedTensor = quantizeToQ4_0(originalTensor)
val result = dotProductQ4_0(quantizedTensor, weights)
```

*More comprehensive examples will be available as the implementation progresses.*

## Current Usage (Development)

**Note: This is a work-in-progress Kotlin port. Full model inference capabilities are not yet available.**

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SolaceHarmony/llama.kotlin.git
   cd llama.kotlin
   ```

2. **Build the project**:
   ```bash
   ./gradlew build
   ```

3. **Run tests**:
   ```bash
   ./gradlew allTests
   ```

*Note: Network access may be required to download Gradle and Kotlin dependencies during the first build.*

## Roadmap and Development Phases

The project follows a structured development approach across multiple phases:

1. **✅ Phase 1**: Project Setup and Analysis - *Complete*
2. **🔄 Phase 2**: Core Library Translation (ggml) - *In Progress*
3. **📋 Phase 3**: CPU Backend Implementation
4. **📋 Phase 4**: Metal Backend Implementation  
5. **📋 Phase 5**: LLaMA Model Implementation
6. **📋 Phase 6**: Model Loading and File Format Support
7. **📋 Phase 7**: API and Applications
8. **📋 Phase 8**: Testing and Validation
9. **📋 Phase 9**: Documentation and Distribution
10. **📋 Phase 10**: Performance Optimization

For detailed progress tracking, see [KOTLIN_PORT_CHECKLIST.md](KOTLIN_PORT_CHECKLIST.md).

## License

This project is licensed under the **MIT License** - same as the original llama.cpp.

## Links and Resources

- **Original llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Project Documentation**: [AGENTS.md](AGENTS.md), [KOTLIN_PORT_CHECKLIST.md](KOTLIN_PORT_CHECKLIST.md)
- **Design Documents**: [GGML_COMPUTE_OPS_DESIGN.md](GGML_COMPUTE_OPS_DESIGN.md), [TENSOR_OPERATIONS_DESIGN.md](TENSOR_OPERATIONS_DESIGN.md)
- **Issues and Discussions**: [GitHub Issues](https://github.com/SolaceHarmony/llama.kotlin/issues)

---

*This project is maintained by Sydney at The Solace Project. We are grateful to the original llama.cpp community for their foundational work that makes this Kotlin port possible.*

