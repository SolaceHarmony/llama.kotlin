# LLama.cpp Kotlin Native Port - Detailed Checklist

This checklist is based on the current state of the Kotlin Native port of llama.cpp and the requirements specified in the issue description. It provides a detailed roadmap for continuing the development of the port.

## ⚠️ BUILD STATUS (Updated: April 2026)

**Strategy Shift: Direct C++ → Kotlin Transliteration**

The project has moved away from the klang soft-float approach to direct line-by-line transliteration
of llama.cpp C++ to Kotlin, using Kotlin/Native cinterop for C API validation.

- **Kotlin Version**: 2.3.20
- **cinterop**: Configured for ggml C headers (ggml.h, gguf.h, ggml-backend.h, ggml-cpu.h)
- **Tracking**: `ast_distance --deep` for porting progress measurement
- **port-lint Headers**: Added to all ported files for explicit source tracking
- **klang**: Retired — source files remain for reference but are not compiled

## Phase 1: Project Setup and Initial Analysis (Partially Complete)

- [x] Setup Kotlin Native Development Environment
  - [x] Install Kotlin Native compiler and tools
  - [x] Configure build system (Gradle with Kotlin DSL)
  - [x] Setup project structure following Kotlin conventions

- [~] Analyze C/C++ Codebase
  - [~] Create a detailed map of all C/C++ files and their dependencies (key core components captured in CPP_CORE_ANALYSIS.md)
  - [~] Identify platform-specific code (Metal, AVX, etc.) for future reference (not currently targeted)
  - [x] Document all external dependencies (core ggml and CPU paths are largely self-contained, as noted in CPP_CORE_ANALYSIS.md)
  - [x] Separate code related to current CPU support from archived GPU backends (legacy GPU backends removed Oct 2025)

- [x] Design Kotlin Native Architecture
  - [x] Design package structure (io.github.kotlinmania.llama.)
  - [x] Plan memory management approach (Kotlin Native has different memory model than C++)
  - [x] Design API that maintains compatibility with original while being idiomatic Kotlin
  - [x] Create detailed design documents for remaining components

## Phase 2: Core Library Translation (ggml) (Complete)

- [~] Translate ggml Core Data Structures
  - [x] Define tensor data types (GGMLType enum)
  - [x] Define tensor operations (GGMLOp enum)
  - [x] Implement tensor structure (GGMLTensor class)
  - [x] Implement context structure (GGMLContext class)
  - [x] Implement computation graph structure (GGMLCGraph class)
  - [x] Implement basic memory allocation structures (GGMLTensorAllocator, GGMLGraphAllocator)
  - [~] Complete memory allocation implementation with actual functionality
    - [x] Refactored GGMLGraphAllocator to use a primary ByteArray buffer.
    - [x] GGMLTensor now stores bufferId and dataOffset, with GGMLGraphAllocator setting these.
    - [x] reserveGraph now sizes the primary ByteArray appropriately and informs the dynamic allocator.
    - [x] Implement efficient tensor data access methods/views into the backing ByteArray(s).
      - [x] Added `nb` (strides) to `GGMLTensor` and populated it in new tensor creation functions.
      - [x] Implemented `get/set` accessors on `GGMLTensor` for F32, I32, I16 using `ByteArray` helpers and stride information.
      - [x] Refactored F32 compute operations in `GGMLComputeOps.kt` to use the new data accessors.
      - [x] Implement F16 typed accessors and update relevant compute operations.
      - [ ] Further optimize data access if performance bottlenecks are identified (e.g., exploring direct memory access if feasible).
    - [x] Implement inplace tensor allocation and memory reuse logic in GGMLGraphAllocator.
      - [x] Implemented tensor usage tracking (children, views, output status, memory ownership) via `TensorUsageInfo` and `tensorUsageMap`.
      - [x] Added `canBeInplace` property to `GGMLOp` to identify suitable operations.
      - [x] `GGMLGraphAllocator.allocateTensor` now attempts inplace allocation by reusing eligible parent tensor memory.
      - [x] Implemented memory freeing logic within `GGMLGraphAllocator.allocateGraph` to deallocate memory of tensors (and view sources) once they are no longer referenced.
      - [x] Consolidated graph allocation and reservation through shared `allocateIfNeeded` / `reserveIfNeeded` helpers to preserve a single allocator API path (Sep 19, 2025).

- [x] Implement Basic Tensor Operations
  - [x] Implement tensor creation functions (createTensor, createTensor1D, createTensor2D)
  - [x] Define element-wise operations interfaces (add, mul)
  - [x] Define matrix multiplication interface (matMul)
- [x] **Major Compute Operations Refactor** (Issue #42)
    - [x] Transformed all compute operations from memory-allocating functions to destination-based operations
    - [x] Updated function signatures: `computeAdd(...): GGMLTensor` → `computeAdd(..., dst: GGMLTensor)`
    - [x] Eliminated memory allocation within compute operations for improved efficiency
    - [x] Operations now write directly into allocator-managed buffers using `dst.setFloat()`
    - [x] All core operations refactored: ADD, MUL, SUB, DIV, NEG, RELU, GELU, MatMul
    - [x] Maintained support for all tensor data types (F32, F16, I8, I16, I32, I64, quantized)
    - [x] Created comprehensive test suite `GGMLComputeOpsDestinationTest.kt`
    - [x] Added dimension/type mismatch validation and error handling
  - [x] Implement activation functions (computeRelu, computeGelu with destination-based interface)
  - [x] Implement support for all tensor data types in compute ops with new destination approach
  - [x] Implement optimized versions of tensor operations (quantized dot products integrated with destination interface)

- [~] Implement Computation Graph
  - [x] Implement forward pass computation
  - [x] Implement automatic differentiation (partial implementation)
    - [x] Implement backward pass for ADD, SUB, MUL, NEG operations
    - [x] Implement backward pass for RELU, GELU activation functions
    - [x] Implement backward pass for MUL_MAT (matrix multiplication)
    - [x] Implement backward pass for DIV, SQR, SQRT operations
    - [x] Implement backward pass for SUM, MEAN operations
    - [x] Implement backward pass for REPEAT operation
    - [x] Implement backward pass for ABS, SGN, STEP operations
    - [ ] Implement backward pass for remaining operations
  - [ ] Implement graph optimization

- [~] Implement Quantization Support
  - [~] Implement 1.5-bit integer quantization (BitNet 1.58)
    - [x] Defined BitNet 1.58 block structure (F16 scale + packed ternary values, type.byteSize = 10).
    - [x] Implemented data accessors for BitNet 1.58 blocks (`getBitNet158BlockScale`, `getBitNet158TernaryWeight`, `setBitNet158TernaryWeight`).
    - [x] Implemented ternary value quantization/dequantization with scale-based thresholds.
    - [x] Implemented efficient base-3 packing (5 ternary values per byte).
    - [x] Added dot product operations (`computeDotProductBitNet158F32`, `computeDotProductBitNet158BitNet158`).
    - [x] Added HPC128 utilities to support LUT/bias transforms in Kotlin (Oct 7, 2025).
    - [ ] Mirror merge-dev BitNet LUT + bias transform (`ggml_bitnet_transform_tensor`) to drive ternary packing and matmul parity.
    - [ ] Regenerate BitNet fixtures/tests once LUT path is ported.
  - [ ] Implement 2-bit integer quantization
  - [ ] Implement 3-bit integer quantization
  - [x] Implement 4-bit integer quantization (Q4_0 focused)
    - [x] Defined Q4_0 block structure (F16 scale + 32x4-bit packed weights, type.byteSize = 18).
    - [x] Implemented data accessors for Q4_0 blocks (`getQ4_0BlockScale`, `getQ4_0NibbleWeight`).
    - [x] Implemented Q4_0 to F32 dequantization in `dequantizeTensor`.
    - [x] Implement Q4_0 quantization (F32 to Q4_0) in `quantizeTensor`.
    - [x] Implement optimized Q4_0 dot product routines (e.g., for MatMul with F32).
      - [x] Implemented `computeDotProductQ40F32` for efficient Q4_0 x F32 operations.
      - [x] Refactored `computeMatMul` to use the optimized dot product for (Q4_0 x F32 -> F32) cases.
      - [x] Implemented optimized dot product for the symmetric F32 x Q4_0 case (`computeDotProductF32Q40`).
      - [x] Implemented direct quantized Q4_0 x Q4_0 operations (`computeDotProductQ40Q40`).
  - [x] Implement 4-bit integer quantization (Q4_1 focused)
    - [x] Defined Q4_1 block structure (2x F16 scale/min + 32x4-bit packed weights, type.byteSize = 20).
    - [x] Implemented data accessors for Q4_1 blocks (`getQ4_1BlockScale`, `getQ4_1BlockMin`, `getQ4_1NibbleWeight`).
    - [x] Implemented Q4_1 to F32 dequantization in `dequantizeTensor`.
    - [x] Implemented F32 to Q4_1 quantization in `quantizeTensor`.
    - [x] Implement optimized Q4_1 dot product routines (e.g., for MatMul with F32).
      - [x] Implemented `computeDotProductQ41F32` for efficient Q4_1 x F32 operations.
      - [x] Refactored `computeMatMul` to use the optimized dot product for (Q4_1 x F32 -> F32) cases.
      - [x] Implemented optimized dot product for the symmetric F32 x Q4_1 case (`computeDotProductF32Q41`).
      - [x] Implemented direct quantized Q4_1 x Q4_1 operations (`computeDotProductQ41Q41`).
  - [~] Implement 5-bit integer quantization
    - [x] Initial Kotlin Q5_K block quantize/dequant paths with klang bit helpers.
    - [ ] Align Q5_K scale/min packing with merge-dev `get_scale_min_k4` layout (add high-bit bitplane + mins).
    - [ ] Regenerate tests/fixtures for Q5_K once packing is corrected.
  - [~] Implement 6-bit integer quantization
    - [x] Initial Kotlin Q6_K block quantize/dequant paths with klang bit helpers.
    - [ ] Port merge-dev `make_qx_quants` + signed scale handling; ensure dequant multiplies by signed sub-block scales.
    - [ ] Regenerate tests/fixtures for Q6_K after parity pass.
  - [x] Implement 8-bit integer quantization (Q8_0 focused)
    - [x] Defined Q8_0 block structure (F16 scale + 32xI8 weights, type.byteSize = 34).
    - [x] Implemented data accessors for Q8_0 blocks (`getQ8_0BlockScale`, `getQ8_0Weight`).
    - [x] Implemented Q8_0 to F32 dequantization in `dequantizeTensor`.
    - [x] Implement Q8_0 quantization (F32 to Q8_0) in `quantizeTensor`.
    - [x] Implement optimized Q8_0 dot product routines (e.g., for MatMul with F32).
      - [x] Implemented `computeDotProductQ80F32` for efficient Q8_0 x F32 operations.
      - [x] Refactored `computeMatMul` to use the optimized dot product for (Q8_0 x F32 -> F32) cases.
      - [x] Implemented optimized dot product for the symmetric F32 x Q8_0 case (`computeDotProductF32Q80`).
      - [x] Implemented direct quantized Q8_0 x Q8_0 operations (`computeDotProductQ80Q80`).
      - [x] Implemented mixed quantized Q8_0 x Q4_0 operations (`computeDotProductQ80Q40`).
  - [~] Implement K-Quant family (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K)
    - [x] Routed all K-Quant pack/unpack paths through klang BitShift/ArithmeticBitwiseOps helpers (Oct 7, 2025).
    - [x] Added shared bitfield/packing primitives (`BitPrimitives`, `PackOps`) and memory/vector helpers to support parity work (Oct 7, 2025).
    - [x] Introduced `QuantizationHelper` with Kotlin ports of merge-dev `make_qkx2`, `make_qkx3`, `make_qp`, `make_qx`, and `make_q3_quants` (Oct 7-8, 2025).
    - [ ] Port merge-dev `make_qkx{2,3}_quants` flows for scale/min selection (Q2_K/Q3_K/Q4_K).
    - [ ] Update `GGMLTypes` K-Quant accessors to use shared helper layer and match C packing.
    - [ ] Cross-check Kotlin vs merge-dev outputs on deterministic tensors; store fixtures for regression tests.
  - [x] **Comprehensive Matrix Multiplication Optimization** (Issue #48)
    - [x] Implemented symmetric F32 x Q_type optimizations replacing expensive dequantization fallbacks
      - [x] `computeDotProductF32Q40` for F32 x Q4_0 operations
      - [x] `computeDotProductF32Q41` for F32 x Q4_1 operations
      - [x] `computeDotProductF32Q80` for F32 x Q8_0 operations
    - [x] Implemented direct quantized-to-quantized Q_type x Q_type operations
      - [x] `computeDotProductQ80Q80` for Q8_0 x Q8_0 operations
      - [x] `computeDotProductQ40Q40` for Q4_0 x Q4_0 operations
      - [x] `computeDotProductQ41Q41` for Q4_1 x Q4_1 operations
    - [x] `computeDotProductQ80Q40` for mixed Q8_0 x Q4_0 operations
    - [x] Integrated all optimizations into `computeMatMul` with proper type dispatch
    - [x] **Performance Testing & Validation**
      - [x] `GGMLMatMulOptimizationTest.kt` - Comprehensive accuracy validation comparing optimized vs fallback paths
      - [x] `GGMLMatMulBenchmarkTest.kt` - Performance microbenchmarking for common LLM matrix sizes
      - [x] Performance profiling for individual dot product kernels
      - [x] Memory usage analysis and stress testing with large matrices
      - [x] Expected speedups: 2-5x for Q×Q operations, 1.5-3x for F32×Q operations
    - [x] **Documentation**: Created `MATMUL_OPTIMIZATION_SUMMARY.md` with comprehensive implementation details
    - [ ] **Destination-API Regression Sweep (Sep 19, 2025)**
      - [ ] Instrument `computeDotProductQ80F32` with a debug hook (guarded by a flag) to log block scales/weights for failing fixtures such as `GGMLComputeOpsTest.testComputeMatMulQ80xSF32`.
      - [ ] Refactor the test allocator helpers into a resettable bump arena so every destination tensor receives a dedicated offset and the allocator mirrors production graph behaviour.
      - [ ] Introduce a shared F32 baseline matmul reference in `GGMLTestUtils` and migrate quantized matmul tests to compare against it instead of ad-hoc oracles.
      - [ ] Re-run the macOS arm64 suite with instrumentation enabled to confirm the Q8_0 × F32 path matches the reference implementation within tolerance once the above fixes land.

## Phase 3: CPU Backend Implementation

- [ ] Translate CPU-Specific Code
  - [ ] Implement basic CPU tensor operations
  - [ ] Implement BLAS integration for CPU
  - [ ] Implement ARM NEON optimizations for CPU
  - [ ] Implement x86 optimizations where possible

- [ ] Optimize CPU Performance
  - [ ] Implement multi-threading support
  - [ ] Optimize memory access patterns
  - [ ] Implement SIMD optimizations where possible in Kotlin Native

- [~] Kotlin/Native SIMD Migration
  - [x] Publish SIMD port strategy (`docs/kdocs/kotlin-native-simd-plan.md`)
  - [ ] Port float dot-product kernels (F32/F16/BF16) using `Vector128`
  - [ ] Port quantized dot-product kernels (Q4/Q5/Q8/K families)
  - [ ] Benchmark Kotlin SIMD vs. scalar fallbacks across supported targets

## Phase 2.5: KLang Soft‑Float + Wide Integer Backbone (Integration Issues)

⚠️ **CRITICAL**: KLang is now a separate repository at https://github.com/Kotlinmania/klang

### Current Integration Status
- [ ] **BLOCKED**: Resolve klang integration strategy
  - Current: Vendored sources in `external/klangnative/` with package naming conflicts
  - Options: 
    1. Publish klang as Maven/Gradle library
    2. Use submodule vendoring
    3. Fix vendored source package names
- [ ] **BLOCKED**: Fix package naming inconsistencies
  - llama.kotlin imports `io.github.kotlinmania.llama.lang.*`
  - Vendored klang has mixed `io.github.kotlinmania.llama.lang.*` and `io.github.kotlinmania.llama.lang.*`
- [ ] **BLOCKED**: Remove duplicate klang sources
  - Removed from `src/commonMain/kotlin/io.github.kotlinmania.llama.lang/` (December 2025)
  - Need to verify external sources are authoritative

### KLang Features (When Integration Fixed)

We introduced a portable, pure‑Kotlin numeric core to remove cross‑platform float drift and make SIMD/GPU backends optional rather than required for correctness.

- [x] KLang namespace and modules
  - [x] `io.github.kotlinmania.llama.lang.fp`: `CFloat32` (inline) with operators +, −, ×, ÷ delegating to bit‑exact soft‑float
  - [x] `io.github.kotlinmania.llama.lang.bitwise.Float32Math`: transliterations of compiler‑rt float32 builtins
    - [x] Division: faithful `fp_div_impl.inc` (normalize, quotient bounds, remainder→sticky, nearest‑even)
    - [ ] Multiplication: tighten to `fp_mul_impl.inc` (currently functional, not yet bit‑diff clean on all ties)
    - [ ] Square root: implement `fp_sqrt_impl.inc`
    - [ ] Conversions: int/uint ↔ float32; float32 ↔ float64
  - [x] Float-on-LHS operators (`Float +/-/*// CFloat32`)
- [x] 16‑bit limb engine for exact 128‑bit intermediates
  - [x] `io.github.kotlinmania.llama.lang.int.hpc.HPC16x4` (64‑bit, 4×16‑bit limbs): add/sub/compare/shifts
  - [x] `io.github.kotlinmania.llama.lang.int.hpc.HPC16x8` (128‑bit, 8×16‑bit limbs): add/sub/compare/shifts, initial 64×64→128 mul
  - [ ] 128/64 division (Knuth D) and full carry propagation for wide mul
- [ ] Float64 roadmap (compiler‑rt exactness)
  - [ ] Port add/sub/mul/div/sqrt using HPC16x4/x8 where 128‑bit ints are required
  - [ ] Conversions: {i32,u32,i64,u64} ↔ float64; float32 ↔ float64
- [ ] Float16/BFloat16 roadmap
  - [ ] `CFloat16` (binary16) add/sub/mul/div/sqrt + conversions
  - [ ] `CBF16` bfloat16 exact conversions (ties‑to‑even) + arithmetic

Testing status (Sep 20–21, 2025)
- [x] Targeted green: `Float32DivTest`, `FloatKlangExtensionsTest` (Float on LHS), `HPC16xTests`
- [ ] Pending: full `mulBits`/`sqrtBits` suites and Float64/Float16/BF16 coverage

Why this matters now
- Eliminates subdeterministic float drift (esp. quantization math) before SIMD/GPU paths
- Provides a deterministic reference path for future backends (JVM Vector API, NEON, Metal)

## Phase 4: Backend Extensibility Planning (Deferred)

- [ ] Define accelerator-agnostic backend interfaces and registration contracts
  - [ ] Document requirements for GPU/SIMD backends drawing from merge-dev sources
  - [ ] Keep Kotlin implementations CPU-first until accelerator scope is approved

- [ ] Prototype CPU streaming primitives that accelerators can reuse
  - [ ] Evaluate coroutine-based buffer arenas (see CHANNEL_ARENA_SPEC)
  - [ ] Validate zero-copy pathways with existing quantization workflows

- [ ] Revisit GPU/SIMD integration after CPU parity and tests are locked

## Phase 5: LLaMA Model Implementation

- [ ] Translate Model Structures
  - [ ] Implement LLaMA model architecture
  - [ ] Implement context and state management
  - [ ] Implement token handling and vocabulary

- [ ] Implement Inference Logic
  - [ ] Implement attention mechanism
  - [ ] Implement feed-forward networks
  - [ ] Implement model loading and initialization

- [ ] Implement Sampling Methods
  - [ ] Implement various sampling strategies (top-k, top-p, etc.)
  - [ ] Implement temperature scaling
  - [ ] Implement repetition penalties

- [ ] Implement Grammar-Constrained Generation
  - [ ] Implement GBNF grammar parsing
  - [ ] Implement grammar-constrained sampling

## Phase 6: Model Loading and File Format Support

- [x] Implement GGUF Format Support
  - [x] Implement GGUF file parsing (GGUFParser.kt with full binary format support)
  - [x] Implement model loading from GGUF files (ModelLoader.kt with tensor integration)
  - [x] Implement basic model conversion utilities (TestGGUFGenerator.kt for testing)
  - [x] Add comprehensive GGUF parsing tests with metadata, tensor info, and data validation
  - [x] Add integration with existing tensor system (GGMLTensor, GGMLContext)
  - [x] Add forward pass validation testing (matrix multiplication with loaded tensors)
  - [ ] Add support for loading real LLaMA model files (extend beyond test files)
  - [ ] Add support for additional quantization formats in GGUF loading
  - [ ] Add model metadata validation and error handling for corrupted files

- [ ] Implement State Saving/Loading
  - [ ] Implement session state serialization
  - [ ] Implement KV cache management
  - [ ] Implement context state management

## Phase 7: API and Applications

- [ ] Design and Implement Public API
  - [ ] Create idiomatic Kotlin API
  - [ ] Implement C interoperability layer for existing applications
  - [ ] Document API thoroughly

- [ ] Implement Command Line Applications
  - [ ] Implement llama-cli equivalent
  - [ ] Implement server application
  - [ ] Implement chat applications

- [ ] Implement Example Applications
  - [ ] Port existing example applications to Kotlin
  - [ ] Create new Kotlin-specific examples
  - [ ] Implement multimodal support (LLaVA, etc.)

## Phase 8: Testing and Validation

- [x] Implement Unit Tests
  - [x] Test core tensor operations (computation logic in GGMLComputeOps)
    - [x] Test element-wise ADD for F32 (1D, 2D) and F16 (1D).
    - [x] Test element-wise MUL for F32 (1D) and F16 (1D).
    - [x] Test element-wise SUB for F32 and F16 types.
    - [x] Test element-wise DIV for F32 and F16 types (including edge cases).
    - [x] Test element-wise NEG for F32 and F16 types.
    - [x] Test element-wise SQR for F32 and F16 types.
    - [x] Test element-wise SQRT for F32 and F16 types.
    - [x] Test `computeMatMul` for F32 x F32 operations.
    - [x] Test `computeMatMul` for Q8_0 x F32 operations (optimized path, comparing against F32 reference).
    - [x] Test activation functions (RELU, GELU, SILU) for F32 and F16 types.
    - [x] Test RMSNorm function for F32 and F16 types.
    - [x] Test operations with other data type combinations (I32, I16, mixed types).
    - [x] Test scalar tensor operations and edge cases (empty tensors, broadcasting).
    - [x] Test error handling and dimension mismatch scenarios.
  - [ ] Test model inference
  - [x] Test quantization accuracy
    - [x] Implemented Q8_0 quantize-dequantize accuracy test (verifying with MSE and MAD).
    - [x] Implemented Q4_0 quantize-dequantize accuracy test (verifying with MSE and MAD).
    - [x] Implemented Q4_1 quantize-dequantize accuracy test (verifying with MSE and MAD).
    - [x] Implemented comprehensive quantization test suite with standardized datasets.
    - [x] Implemented error thresholds based on upstream llama.cpp reference constants.
    - [x] Added synthetic data generation matching upstream test-quantize-fns.cpp patterns.
    - [x] Added random data, edge case, and cross-quantization validation.
    - [x] Added dot product accuracy testing for quantized operations.
    - [~] Test accuracy for other future quantization types as they are implemented.
      - [x] Q2_K
      - [x] Q3_K
      - [x] Q4_K
      - [x] Q5_K
      - [ ] Q6_K
    - [x] Test `GGMLDynTensorAllocator` (dynamic memory allocation within a buffer).
    - [x] Test `GGMLGraphAllocator` (graph-level memory planning: reserve, inplace allocation, freeing).
    - [x] Test `GGMLTensor` data accessors (low-level read/write for F32, I32, I16, F16).

- [x] Implement Integration Tests
  - [x] Test end-to-end computation graph operations and complex operation chains.
  - [x] Test mixed precision computation workflows (F32 ↔ F16).
  - [x] Test quantized operation integration and workflow chains.
  - [x] Test memory allocation stress scenarios.
  - [x] Test activation function chaining and complex mathematical expressions.
  - [x] Test performance benchmarks with throughput and latency metrics.
  - [x] Test error handling integration in computation chains.
  - [ ] Test end-to-end model loading and inference
  - [x] Compare computational accuracy with analytical reference implementations.

- [x] Implement Performance Validation
  - [x] Comprehensive benchmarking suite for all core operations.
  - [x] Element-wise operation performance testing (ADD, MUL, SUB, DIV, NEG).
  - [x] Unary operation benchmarks (SQR, SQRT, RELU, GELU, SILU, RMSNorm).
  - [x] Matrix multiplication performance validation.
  - [x] Quantization/dequantization performance benchmarks.
  - [x] Memory allocation and deallocation performance testing.
  - [x] Throughput (MB/s) and operations-per-second metrics.

- [x] Implement Reference Validation Framework
  - [x] Analytical reference test vectors for mathematical validation.
  - [x] Cross-implementation validation against known good results.
  - [x] Numerical stability and edge case validation.
  - [x] Cross-precision validation (F32 vs F16 consistency).
  - [x] Regression baseline testing framework.
  - [x] Configurable error tolerance validation.
  - [ ] Compare output with original C++ implementation (requires C++ reference data).

- [ ] Validate Model Compatibility
  - [ ] Test with various LLaMA models
  - [ ] Test with other supported models (Mistral, Mixtral, etc.)
  - [ ] Ensure output matches original implementation

- [x] Test Documentation and Infrastructure
  - [x] Comprehensive test methodology documentation.
  - [x] Error threshold rationale and reference implementation alignment.
  - [x] Test utility library for common testing patterns.
  - [x] Performance benchmarking guidelines.
  - [x] Reference validation framework documentation.
  - [x] **Destination-based compute operations test suite** (`GGMLComputeOpsDestinationTest.kt`)
    - [x] Comprehensive validation of new in-place computation interface
    - [x] Dimension and type mismatch error handling tests
    - [x] Integration tests with graph allocator memory management

## Phase 9: Documentation and Distribution

- [~] Create Documentation
  - [x] Create design documents for tensor operations (TENSOR_OPERATIONS_DESIGN.md)
  - [x] Create design documents for compute operations (GGML_COMPUTE_OPS_DESIGN.md)
  - [x] Document current status (now tracked live via `tools/ast_distance --deep`, see `port_status_report.md`)
  - [x] **DRY Principle Implementation and Code Quality**
    - [x] Audit codebase for duplication and implement DRY improvements
    - [x] Consolidate utility functions in `GGMLUtilities.kt` and `GGMLTensorUtils.kt`
    - [x] Standardize KDoc documentation following Kotlin standards
    - [x] Eliminate duplicate formatting functions and ByteArray extensions
    - [x] Create reusable demo components and display utilities
    - [x] Update implementation patterns documentation
  - [ ] Write comprehensive API documentation
  - [ ] Create usage guides
  - [ ] Document performance characteristics

- [ ] Setup Distribution
  - [ ] Configure Maven/Gradle publishing
  - [ ] Create release process
  - [ ] Setup continuous integration

- [ ] Create Migration Guide
  - [ ] Document differences from C++ implementation
  - [ ] Provide migration examples for existing users
  - [ ] Document performance trade-offs

## Phase 10: Performance Optimization

- [ ] Benchmark and Profile
  - [ ] Identify performance bottlenecks
  - [ ] Compare with C++ implementation
  - [ ] Document performance characteristics

- [ ] Optimize Critical Paths
  - [ ] Optimize tensor operations
  - [ ] Optimize memory usage
  - [ ] Optimize threading model

- [ ] Implement Advanced Optimizations
  - [ ] Implement speculative decoding
  - [ ] Optimize KV cache management
  - [ ] Implement model-specific optimizations

## Next Steps

With the shared klang primitives in place, the remaining work focuses on merge-dev parity, refreshed validation, and staging the backends. Immediate priorities:

1.  **Merge-Dev Quantization Parity**
    *   Migrate Q5_K/Q6_K (and any remaining K-Quant paths) to `QuantizationHelper`, matching `get_scale_min_k4`, signed scales, and high-bit packing.
    *   Update `GGMLTypes` accessors to use `BitPrimitives`/`PackOps` so read/write paths match the C layout.
    *   Mirror `ggml_bitnet_transform_tensor` (LUT + bias) so BitNet blocks are byte-identical before matmul.

2.  **Fixture & Test Regeneration**
    *   Generate golden K-Quant/BitNet blocks from the merge-dev C reference and add byte-for-byte regression tests.
    *   Re-enable `macosArm64Test` (and other native suites) using the refreshed fixtures.

3.  **Documentation & Checklist**
    *   Update quantization/bitwise design docs and regenerate `port_status_report.md` via `tools/ast_distance --deep` to reflect parity progress and new helper modules.
    *   Keep this checklist in sync after each milestone and add any additional helper documentation as needed.

4.  **Backend Enablement**
    *   Resume CPU backend formalization (threading/SIMD) once quantizer parity is complete.
    *   Stage the GGUF parsing refresh so model loading consumes the corrected tensor layouts.

## Build Environment

The project is set up as a Kotlin Multiplatform project with the following structure:

- **Root Directory**: /Volumes/stuff/Projects/llama.kotlin
- **Source Directory**: src/nativeMain/kotlin/io.github.kotlinmania.llama.lamakotlin
- **Build Configuration**: build.gradle.kts, settings.gradle.kts
- **Design Documents**: TENSOR_OPERATIONS_DESIGN.md, GGML_COMPUTE_OPS_DESIGN.md, GGUF_IMPLEMENTATION.md (in progress)
- **Status Report**: port_status_report.md (generated by `tools/ast_distance --deep`)

The project targets macOS platforms (both x64 and arm64) and uses Gradle for building. Key commands: `./gradlew build -x macosArm64Test` (current green) and `./gradlew macosArm64Test` (failing until fixtures are regenerated).

Note: All legacy llama.cpp archives, GPU backends, and examples were removed in October 2025. The repo now retains only the Kotlin/Native sources, runtimes under `external/`, klang in `src/commonMain/kotlin/io.github.kotlinmania.llama.lang`, and header snapshots in `spm-headers/` for archival reference (e.g., potential Swift/Metal integration later).

## Challenges and Considerations

Some key challenges and considerations for the Kotlin Native port:

1. **Memory Management**: Kotlin Native has a different memory model than C++, which will require careful design for efficient tensor operations
2. **Performance**: Ensuring that the Kotlin implementation maintains comparable performance to the C++ original
3. **Interoperability**: Potentially allowing interoperability with the original C++ code for components that are difficult to port
4. **Future Accelerator Integration**: Evaluate GPU/SIMD backends once the CPU path is fully validated
5. **Quantization**: Implementing efficient quantization support in Kotlin Native
