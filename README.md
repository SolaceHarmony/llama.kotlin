# llama.kotlin

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A Kotlin/Native port of [llama.cpp](https://github.com/ggerganov/llama.cpp) - Inference of Meta's [LLaMA](https://arxiv.org/abs/2302.13971) model (and others) in pure Kotlin**

*Maintained by **Sydney Renee** of [The Solace Project](mailto:sydney@solace.ofharmony.ai) for [KotlinMania](https://github.com/KotlinMania)*

## Acknowledgments

This project is a Kotlin/Native port of the original [llama.cpp](https://github.com/ggerganov/llama.cpp) created by [Georgi Gerganov](https://github.com/ggerganov) and the amazing open source community. We extend our deepest gratitude to all the original contributors who made the foundational work possible. Please see the [AUTHORS](AUTHORS) file for the complete list of contributors to the original project.

**Original llama.cpp contributors deserve special recognition for their groundbreaking work in making LLM inference efficient and accessible.**

## About This Port

This is a **Kotlin/Native implementation** of llama.cpp, designed to bring the power of large language model inference to the Kotlin ecosystem with a focus on:

- **Optimized CPU backend** built on deterministic klang primitives
- **Idiomatic Kotlin API** while maintaining compatibility with original concepts
- **Memory-efficient tensor operations** adapted for Kotlin/Native's memory model
- **Comprehensive quantization support** (Q8_0, Q4_0, Q4_1, BitNet 1.58, and K-Quant types Q2_K, Q3_K, Q4_K, Q5_K, Q8_K; Q6_K in progress)
- **Automatic differentiation** for training and fine-tuning capabilities

## Status Snapshot (October 7, 2025)

- The legacy `archive/` and `examples/` trees from the original llama.cpp have been removed.
- macOS-specific benchmarks were moved to `src/macosArm64Test`. Other target test suites execute normally (JS/JVM/Native metadata).

## Current Status - Phase 3-4 (Advanced Core Implementation & Backend Development)

The project is actively under development with substantial progress across multiple phases:

### ✅ Completed Features
- **Memory Management**: Advanced tensor allocation with `GGMLGraphAllocator` and `GGMLDynTensorAllocator`
  - Primary ByteArray buffer with dynamic allocation within reserved space
  - Inplace tensor allocation and memory reuse logic for optimization
  - Tensor usage tracking and automatic memory freeing
  - Graph-level memory planning with comprehensive allocation strategies
- **Tensor Data Access**: Comprehensive accessor methods for all supported data types
  - F32, F16, I32, I16 data accessors with stride information
  - Efficient ByteArray-based data storage and retrieval
  - Multi-dimensional tensor indexing with proper stride calculations
- **Advanced Quantization Support**: Comprehensive quantization ecosystem implemented
  - **Q8_0**: F16 scale + 32xI8 weights (34 bytes per block)
  - **Q4_0**: F16 scale + 32x4-bit packed weights (18 bytes per block)
  - **Q4_1**: 2x F16 scale/min + 32x4-bit packed weights (20 bytes per block)
  - **BitNet 1.58**: Ternary quantization with F16 scale + base-3 packing (10 bytes per block)
  - Optimized dot product routines for all quantized operations
  - Direct quantized-to-quantized operations (Q×Q) avoiding expensive dequantization
- **Matrix Multiplication Optimizations**: Complete optimization coverage for all quantization combinations
  - Symmetric F32×Q_type optimizations (2-3x speedup)
  - Direct Q_type×Q_type operations (3-5x speedup)
  - Mixed quantized operations (Q8_0×Q4_0, etc.)
  - Performance validation with comprehensive benchmarking suite
- **Core Tensor Operations**: Element-wise and matrix operations with multi-type support
  - ADD, MUL, SUB, DIV, NEG, SQR, SQRT, MatMul for F32/F16 and quantized types
  - Activation functions: RELU, GELU, SILU, RMSNorm
  - Broadcasting and dimension validation with proper error handling
- **GGUF File Format Support**: Complete model loading pipeline
  - Binary GGUF file parsing with metadata extraction
  - Tensor loading and integration with existing tensor system
  - Model validation through forward pass testing
  - Support for all standard GGUF data types and structures
- **Automatic Differentiation**: Backward pass implementation for core operations
  - ADD, SUB, MUL, NEG, DIV, SQR, SQRT operations
  - RELU, GELU activation functions
  - MUL_MAT (matrix multiplication)
  - SUM, MEAN, REPEAT operations
- **Compute Operations Architecture**: Major refactor to destination-based operations
  - All compute functions write directly into pre-allocated destination tensors
  - Function signatures changed from `computeAdd(...): GGMLTensor` to `computeAdd(..., dst: GGMLTensor)`
  - Eliminated memory allocation within compute operations for improved efficiency
  - Aligned with GGML architecture patterns for memory reuse and graph optimization

### 🔄 In Progress
- **Additional Quantization**: K-Quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- **CPU Backend Formalization**: Structured CPU backend with multi-threading support
- **Model Architecture Implementation**: LLaMA model structures and inference pipeline

### 📋 Comprehensive Testing Infrastructure
- **Unit Tests**: Complete coverage for core operations under `src/commonTest/kotlin`
  - Element-wise operations (ADD, MUL, SUB, DIV, NEG, SQR, SQRT)
  - Matrix operations with all quantization combinations
  - Activation functions (RELU, GELU, SILU, RMSNorm)
  - Error handling and edge cases (scalar tensors, dimension mismatches)
- **Quantization Accuracy Tests**: Rigorous validation with standardized metrics
  - MSE, RMSE, MAD, SNR analysis for all quantization types
  - Reference-aligned error thresholds based on upstream llama.cpp
  - Synthetic, random, and edge case test vectors
  - Cross-quantization performance validation
- **Performance Benchmarking**: Comprehensive performance validation suite
  - Throughput (MB/s) and operations-per-second metrics
  - Matrix multiplication optimization validation (2-5x speedups achieved)
  - Memory allocation efficiency testing
  - Scalability analysis across data sizes
- **Integration Tests**: End-to-end workflow validation
  - Complex computation chains and mathematical expressions
  - Mixed precision workflows (F32 ↔ F16)
  - Memory stress testing with large-scale operations
  - GGUF model loading and forward pass validation
- **Reference Validation Framework**: Mathematical accuracy assurance
  - Analytical reference validation against known results
  - Numerical stability and boundary condition testing
  - Cross-precision consistency validation
  - Regression baseline prevention system
- **Destination-based Operations Test Suite**: Comprehensive validation of architecture refactor
  - In-place computation interface validation (`GGMLComputeOpsDestinationTest.kt`)
  - Dimension and type mismatch error handling
  - Direct integration with graph allocator memory management

## Project Documentation

For detailed development information, see:
- [**KOTLIN_PORT_CHECKLIST.md**](KOTLIN_PORT_CHECKLIST.md) - Detailed development roadmap with current progress
- [**port_status_report.md**](port_status_report.md) - Live port status generated by `tools/ast_distance --deep`
- [**COMPUTE_OPERATIONS_REFACTOR_SUMMARY.md**](COMPUTE_OPERATIONS_REFACTOR_SUMMARY.md) - Major architectural refactor details
- [**MATMUL_OPTIMIZATION_SUMMARY.md**](MATMUL_OPTIMIZATION_SUMMARY.md) - Comprehensive matrix multiplication optimizations
- [**GGUF_IMPLEMENTATION.md**](GGUF_IMPLEMENTATION.md) - GGUF file format support implementation
- [**GGML_TESTING_SUMMARY.md**](GGML_TESTING_SUMMARY.md) - Comprehensive testing infrastructure documentation
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
./gradlew allTests  # Run all tests for available targets
# Or run tests for specific target:
./gradlew linuxX64Test  # Linux tests
```

*Note: Legacy llama.cpp C/C++ and Python examples were removed in October 2025; only files required by the Kotlin/Native port remain. Expect ggml headers to live under `external/`, `spm-headers/`, and `src/nativeInterop/`.*

## Architecture Overview

- **Source Directory**: `src/nativeMain/kotlin/io.github.kotlinmania.llama.
- **Package Structure**: `io.github.kotlinmania.llama.`
- **Core Modules**:
  - `core/GGMLTypes.kt` - Core tensor data structures
  - `core/GGMLAlloc.kt` - Memory management
  - `core/GGMLOps.kt` - High-level tensor operations
  - `core/GGMLComputeOps.kt` - Low-level computation kernels
  - `core/GGMLGraph.kt` - Computation graph execution
  - `core/GGMLQuants.kt` - Quantization implementations

## Supported Platforms

- **macOS** (x64 and arm64) - Primary target for Kotlin/Native builds
- **Linux** (x64) - Supported for CPU backend development and testing
- **Windows** (mingw-x64) - Basic support for development
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
- **Future accelerator hooks** exposed via backend registry
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
3. Look at existing implementations in `src/nativeMain/kotlin/io.github.kotlinmania.llama.
4. Add tests in `src/commonTest/kotlin/`

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

### Quantization and Optimization
```kotlin
// Q4_0 quantization example with optimized operations
val originalTensor = allocator.allocateTensor(GGMLType.F32, longArrayOf(32, 128))
val quantizedTensor = allocator.allocateTensor(GGMLType.Q4_0, longArrayOf(32, 128))
quantizeTensor(originalTensor, quantizedTensor) // Direct quantization

// Optimized matrix multiplication with quantized tensors (2-5x speedup)
val weights = allocator.allocateTensor(GGMLType.Q4_0, longArrayOf(128, 512))
val input = allocator.allocateTensor(GGMLType.F32, longArrayOf(32, 128))
val output = allocator.allocateTensor(GGMLType.F32, longArrayOf(32, 512))
computeMatMul(allocator, context, input, weights, output) // Uses optimized Q4_0×F32 kernel
```

### GGUF Model Loading
```kotlin
// Load model from GGUF file
val loader = ModelLoader()
val model = loader.loadFromFile("model.gguf")

// Access loaded tensors
val embeddings = model.getTensor("token_embeddings.weight")
val attention = model.getTensor("layers.0.attention.wq.weight")

// Validate model with forward pass
val success = model.validateForwardPass()
println("Model validation: ${if (success) "PASSED" else "FAILED"}")
```

*More comprehensive examples will be available as the implementation progresses.*

## Current Usage (Development)

**Note: This is an active Kotlin port with substantial core infrastructure complete. GGUF model loading, comprehensive quantization support, advanced memory management, and performance optimizations are functional.**

### Development Setup

1. **Open a prepared checkout**:
   ```bash
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
2. **✅ Phase 2**: Core Library Translation (ggml) - *Complete*
3. **🔄 Phase 3**: CPU Backend Implementation - *In Progress*
4. **🔄 Phase 4**: Backend Extensibility & Runtime Experiments - *Planned*
5. **📋 Phase 5**: LLaMA Model Implementation - *Starting*
6. **✅ Phase 6**: Model Loading and File Format Support - *Largely Complete (GGUF)*
7. **📋 Phase 7**: API and Applications
8. **✅ Phase 8**: Testing and Validation - *Comprehensive Infrastructure Complete*
9. **📋 Phase 9**: Documentation and Distribution
10. **🔄 Phase 10**: Performance Optimization - *Major Optimizations Complete*

For detailed progress tracking, see [KOTLIN_PORT_CHECKLIST.md](KOTLIN_PORT_CHECKLIST.md).

## License

This project is licensed under the **MIT License** - same as the original llama.cpp.

## Links and Resources

- **Original llama.cpp**: https://github.com/ggerganov/llama.cpp
- **Project Documentation**: [AGENTS.md](AGENTS.md), [KOTLIN_PORT_CHECKLIST.md](KOTLIN_PORT_CHECKLIST.md)
- **Design Documents**: [GGML_COMPUTE_OPS_DESIGN.md](GGML_COMPUTE_OPS_DESIGN.md), [TENSOR_OPERATIONS_DESIGN.md](TENSOR_OPERATIONS_DESIGN.md)
- **Issues and Discussions**: [GitHub Issues](https://github.com/KotlinMania/llama.kotlin/issues)

---

*This project is maintained by **Sydney Renee** of [The Solace Project](mailto:sydney@solace.ofharmony.ai) for [KotlinMania](https://github.com/KotlinMania). We are grateful to the original llama.cpp community for their foundational work that makes this Kotlin port possible.*
