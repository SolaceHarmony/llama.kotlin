# llama.kotlin Project Instructions

This file provides guidance for future agents working on the Kotlin port of `llama.cpp`.
It summarizes the current project state and lists recommended next steps. Before starting new work, read `KOTLIN_PORT_CHECKLIST.md` in the repository root for a detailed roadmap.

## Project Overview
- **Goal**: Create a Kotlin/Native implementation of llama.cpp, focusing on CPU and Apple Metal backends
- **Current Status**: Phase 2 (Core Library Translation) with significant progress in tensor operations, memory management, and quantization
- The repository is a work‐in‐progress port of `llama.cpp` to Kotlin/Native
- Kotlin sources live under `src/nativeMain/kotlin/ai/solace/llamakotlin`
- The original C/C++ sources remain under `src` while porting progresses
- Design notes and porting progress are documented in:
  - `KOTLIN_PORT_CHECKLIST.md` - Detailed development roadmap with current progress
  - `KOTLIN_PORT_STATUS.md` - Overall project status and completion overview  
  - `GGML_COMPUTE_OPS_DESIGN.md` - Technical design for computation operations
  - `TENSOR_OPERATIONS_DESIGN.md` - Design patterns for tensor operations
  - `CPP_CORE_ANALYSIS.md` - Analysis of original C++ codebase

## Current Implementation Status

### ✅ Completed Core Features
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
  - ADD, MUL, MatMul for F32/F16 and quantized types
  - Activation functions: RELU, GELU, SILU, RMSNorm
- **Automatic Differentiation**: Backward pass implementation for core operations
  - ADD, SUB, MUL, NEG, DIV, SQR, SQRT operations
  - RELU, GELU activation functions
  - MUL_MAT (matrix multiplication)
  - SUM, MEAN, REPEAT operations

### 🔄 In Progress  
- **Computation Graph Optimization**: Graph optimization passes for redundant operation removal
- **Additional Quantization**: K-Quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
- **CPU Backend**: Formal CPU backend structure and multi-threading support

### 📋 Testing Infrastructure
- Comprehensive unit tests for core operations under `src/nativeTest/kotlin`
- Quantization accuracy tests with MSE and MAD validation
- Memory allocator tests for graph-level memory planning
- Tensor data accessor tests for all supported types

## Coding Guidelines
- **Kotlin Style**: Use idiomatic Kotlin with descriptive names and comprehensive KDoc comments
- **Modular Design**: Separate tensor creation logic from compute kernels (see `GGMLOps.kt` vs `GGMLComputeOps.kt`)
- **Immutability**: Prefer immutable data structures where practical
- **Documentation**: Document placeholders or incomplete implementations with `TODO` comments
- **Memory Efficiency**: Use ByteArray-based storage with accessor methods rather than individual arrays
- **Type Safety**: Leverage Kotlin's type system for compile-time safety in tensor operations
- **Performance**: Consider SIMD and multi-threading opportunities in CPU-intensive operations

## Key Architecture Patterns
- **Memory Management**: Use `GGMLGraphAllocator` for graph-level memory planning with inplace optimization
- **Data Access**: Implement typed accessors (`getF32`, `setF32`, etc.) for ByteArray-backed tensors
- **Quantization**: Follow block-based quantization patterns with optimized dot product routines
- **Computation Separation**: Keep operation setup (graph building) separate from computation (execution)

## Build and Test
- **Build System**: The project uses Gradle with Kotlin Multiplatform. Build with `./gradlew build`
- **Network Dependencies**: Network access may be required to download dependencies; configure as needed
- **SSL Issues**: If Gradle fails with a `PKIX path building failed` SSL error, install Java and Gradle via SDKMAN:
  ```bash
  curl -s "https://get.sdkman.io" | bash
  source "$HOME/.sdkman/bin/sdkman-init.sh"
  sdk install java 17.0.9-tem
  sdk install gradle 8.13
  ```
  Alternatively set `GRADLE_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts"` to use the system certificate store
- **Target Platforms**: Currently targets macOS (x64 and arm64) - configured in `build.gradle.kts`
- **Test Structure**: 
  - Unit tests: `src/nativeTest/kotlin/ai/solace/llamakotlin/core/`
  - Test categories: Core operations, memory allocation, quantization accuracy, data accessors
  - Run tests: `./gradlew allTests` (when build system is accessible)
- **C++ Legacy**: C++ tests under `tests/` are not required for the Kotlin port

### Current Test Coverage
- **GGMLComputeOpsTest.kt**: Core tensor operations (ADD, MUL, MatMul, activations)
- **GGMLQuantizationAccuracyTest.kt**: Q8_0, Q4_0, Q4_1 quantization accuracy validation
- **GGMLAllocTest.kt**: Memory allocation and graph planning functionality  
- **GGMLTypesTest.kt**: Tensor data accessors and type handling

## Implementation Priorities (Based on Current Checklist Status)

### 🎯 Immediate Next Steps
1. **Advanced Quantization Support** 
   - Implement K-Quant types (Q4_K, Q2_K) with block structures and optimized operations
   - Add symmetric dot product routines for F32 x Q_type operations  
   - Extend quantization accuracy testing for all new types

2. **Computation Graph Optimization**
   - Implement graph optimization passes for redundant operation removal
   - Add automatic differentiation for remaining operations (see `KOTLIN_PORT_CHECKLIST.md` for specific list)
   - Enhance graph execution efficiency

3. **CPU Backend Formalization**
   - Create formal CPU backend structure integrating current `GGMLComputeOps.kt` logic
   - Implement multi-threading for graph computation using Kotlin coroutines
   - Investigate SIMD optimizations within Kotlin/Native constraints

4. **GGUF Format Support** 
   - Begin GGUF file parsing implementation (critical for model loading)
   - Implement model metadata reading and tensor information extraction
   - Support for quantization type detection from GGUF files

### 🔄 Mid-term Goals
1. **Metal Backend Foundation** (Phase 4)
   - Basic Metal context setup and shader compilation infrastructure
   - Simple Metal compute shader for a single ggml operation as proof-of-concept

2. **Enhanced Testing and Validation**
   - Integration tests for end-to-end tensor operations
   - Performance benchmarking against reference implementations
   - Expanded quantization accuracy testing with standardized datasets

3. **LLaMA Model Architecture** (Phase 5)
   - Begin core model structure implementation
   - Attention mechanism and feed-forward network foundations

### 📚 Implementation Resources
- **Memory Management Patterns**: See `GGMLAlloc.kt` for tensor and graph allocation strategies  
- **Quantization Implementation**: Reference `GGMLComputeOps.kt` for Q8_0/Q4_0/Q4_1 patterns
- **Data Access Patterns**: See `GGMLTensor` accessor methods for ByteArray-based storage
- **Testing Patterns**: Follow existing test structure in `src/nativeTest/kotlin/` 

## Technical Implementation Guidance

### Memory Management Best Practices
```kotlin
// Use GGMLGraphAllocator for efficient memory planning
val allocator = GGMLGraphAllocator()
allocator.reserve(graph, bufferSize)
val tensor = allocator.allocateTensor(type, dimensions) // Automatically uses inplace when possible
```

### Quantization Implementation Pattern  
```kotlin
// Follow block-based quantization pattern
// 1. Define block structure (scale + packed weights)
// 2. Implement block data accessors (getScale, getWeight)
// 3. Add quantize/dequantize functions
// 4. Create optimized dot product routine
// 5. Integrate with MatMul operation
```

### Adding New Tensor Operations
```kotlin
// 1. Add operation enum to GGMLOp
// 2. Implement computation in GGMLComputeOps.kt
// 3. Add high-level interface in GGMLOps.kt  
// 4. Create unit tests in GGMLComputeOpsTest.kt
// 5. Add backward pass for automatic differentiation
```

## Project Scope and Backend Support
- **Supported Backends**: CPU (primary focus) and Apple Metal for macOS/iOS
- **Archived Backends**: CUDA, hipBLAS, Vulkan, SYCL, MUSA, and CANN backends moved to `archive/` 
- **Platform Targets**: macOS (x64 and arm64) with future support for other Kotlin/Native targets
- **Memory Model**: Adapted for Kotlin/Native's memory management (different from C++ original)

## Advanced Research Goals (Future Work)
The project includes research into hybrid architectures beyond the core llama.cpp port:
- **Liquid Neural Networks Integration**: Actor-based computation with adaptive time constants (see `hybrid-llama-lnn-design.md`)
- **Quantum-Classical Hybrid Framework**: Meta-Cognitive Temporal Architecture (MCTA) research (see `quantum-classical-hybrid-mcta-design.md`)
- **Memory Cube Architecture**: Advanced caching and inference optimization strategies
- **Actor-Coroutine Integration**: Leveraging Kotlin's concurrency model for neural computation

*Note: These are research directions for future exploration after completing the core llama.cpp port.*

## Key Files and Modules
- **Core Types**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLTypes.kt`
- **Memory Management**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLAlloc.kt`
- **Tensor Operations**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLOps.kt`
- **Computation Logic**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLComputeOps.kt`
- **Graph Execution**: `src/nativeMain/kotlin/ai/solace/llamakotlin/core/GGMLGraph.kt`
- **Test Suite**: `src/nativeTest/kotlin/ai/solace/llamakotlin/core/`

## Challenges and Considerations
- **Memory Management**: Kotlin/Native memory model requires different patterns than C++
- **Performance**: Maintaining comparable performance to C++ original while leveraging Kotlin strengths
- **Interoperability**: Potential C interop for performance-critical sections (evaluate on case-by-case basis)
- **SIMD Limitations**: Kotlin/Native has limited SIMD support compared to C++ (explore alternatives)
- **Metal Integration**: Implementing Metal backend for Apple Silicon optimization
- **Quantization Precision**: Ensuring accuracy across all quantization formats

Follow this guide when extending the Kotlin port. Keep commits focused and include relevant tests whenever possible. For detailed implementation status, always reference `KOTLIN_PORT_CHECKLIST.md` for the most current progress information.
