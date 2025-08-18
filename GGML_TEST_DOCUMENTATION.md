# GGML Test Suite Documentation

This document describes the comprehensive testing infrastructure for the GGML Kotlin port, including test methodologies, error thresholds, and validation approaches.

## Test Organization

### Test Files Overview

| Test File | Purpose | Coverage |
|-----------|---------|----------|
| `GGMLComputeOpsTest.kt` | Basic tensor operations | ADD, MUL, MatMul, activations (RELU, GELU, SILU), RMSNorm |
| `GGMLExtendedOpsTest.kt` | Extended operations | SUB, NEG, DIV, SQR, SQRT, edge cases, broadcasting |
| `GGMLQuantizationAccuracyTest.kt` | Basic quantization accuracy | Q8_0, Q4_0, Q4_1 MSE/MAD validation |
| `GGMLStandardizedQuantizationTest.kt` | Enhanced quantization testing | Reference thresholds, multiple test vectors, cross-validation |
| `GGMLIntegrationTest.kt` | End-to-end workflows | Computation chains, mixed precision, error handling |
| `GGMLPerformanceBenchmarkTest.kt` | Performance validation | Throughput, latency, memory allocation benchmarks |
| `GGMLReferenceValidationTest.kt` | Reference implementation comparison | Analytical validation, upstream pattern matching |
| `GGMLAllocTest.kt` | Memory management | Graph allocation, dynamic allocation |
| `GGMLTypesTest.kt` | Data type handling | Tensor data accessors |

## Error Thresholds and Validation Criteria

### Quantization Error Thresholds

Based on upstream llama.cpp `test-quantize-fns.cpp` constants:

```kotlin
// Reference error thresholds from upstream
const val MAX_QUANTIZATION_TOTAL_ERROR = 0.002f          // General quantization error
const val MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f   // 2-bit quantization
const val MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f   // 3-bit quantization
const val MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f     // Reference implementation comparison
const val MAX_DOT_PRODUCT_ERROR = 0.02f                  // Matrix dot product operations
const val MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f           // Low-bit quantized dot products

// Kotlin implementation specific thresholds
const val MSE_THRESHOLD_Q8_0 = 0.0001f    // Mean Squared Error for Q8_0
const val MSE_THRESHOLD_Q4_0 = 0.01f      // Mean Squared Error for Q4_0  
const val MSE_THRESHOLD_Q4_1 = 0.015f     // Mean Squared Error for Q4_1

const val MAD_THRESHOLD_Q8_0 = 0.01f      // Mean Absolute Difference for Q8_0
const val MAD_THRESHOLD_Q4_0 = 0.2f       // Mean Absolute Difference for Q4_0
const val MAD_THRESHOLD_Q4_1 = 0.1f       // Mean Absolute Difference for Q4_1

const val SNR_THRESHOLD_Q8_0 = 40.0       // Signal-to-Noise Ratio (dB) for Q8_0
const val SNR_THRESHOLD_Q4_0 = 20.0       // Signal-to-Noise Ratio (dB) for Q4_0
const val SNR_THRESHOLD_Q4_1 = 18.0       // Signal-to-Noise Ratio (dB) for Q4_1
```

### Rationale for Thresholds

**Q8_0 (8-bit quantization):**
- High precision quantization with 7-bit mantissa + sign
- Expected to maintain high accuracy with minimal error
- MSE threshold: 0.0001 (very low error tolerance)
- SNR threshold: 40dB (high signal quality)

**Q4_0 (4-bit symmetric quantization):**
- Medium precision quantization around zero
- Higher error expected due to 4-bit limitation
- MSE threshold: 0.01 (moderate error tolerance)
- SNR threshold: 20dB (medium signal quality)

**Q4_1 (4-bit asymmetric quantization):**
- 4-bit quantization with min/max scaling
- Often better than Q4_0 for non-zero-centered data
- MSE threshold: 0.015 (between Q8_0 and Q4_0)
- SNR threshold: 18dB (medium-low signal quality)

## Test Data Generation

### Synthetic Data Patterns

**Cosine Pattern (upstream compatible):**
```kotlin
fun generateSyntheticData(size: Int, offset: Float = 0.0f): FloatArray {
    return FloatArray(size) { i ->
        0.1f + 2.0f * cos(i.toFloat() + offset)
    }
}
```

This pattern matches the upstream llama.cpp `test-quantize-fns.cpp` synthetic data generation and provides:
- Predictable, reproducible test data
- Full dynamic range coverage (-1.9 to +2.1)
- Smooth transitions for testing quantization accuracy

**Random Data Generation:**
```kotlin
fun generateRandomData(size: Int, seed: Long = 12345, range: Float = 10.0f): FloatArray {
    val random = Random(seed)
    return FloatArray(size) { 
        random.nextFloat() * 2 * range - range // Range [-range, range]
    }
}
```

**Edge Case Data:**
- Zero values (positive and negative zero)
- Minimum and maximum representable values
- Special values (π, e, common mathematical constants)
- Large and small magnitude values

## Performance Validation

### Throughput Benchmarking

Performance metrics include:
- **Throughput (MB/s):** Data processed per second
- **Operations per second:** Number of operations completed
- **Latency (ms):** Time to complete single operation

**Performance Validation Criteria:**
- Operations should complete within reasonable time bounds
- Throughput should scale with data size
- Memory allocation should be efficient
- No memory leaks or excessive allocation

### Benchmark Categories

1. **Element-wise Operations:** ADD, MUL, SUB, DIV, NEG
2. **Unary Functions:** SQR, SQRT, RELU, GELU, SILU
3. **Matrix Operations:** MatMul with various sizes
4. **Quantization Operations:** Quantize/dequantize performance
5. **Memory Operations:** Allocation and deallocation patterns

## Integration Testing

### Computation Chain Validation

Tests complex operation sequences:
```kotlin
// Example: (A + B) * C - D
val step1 = computeAdd(graphAllocator, dummyContext, tensorA, tensorB)
val step2 = computeMul(graphAllocator, dummyContext, step1, tensorC)
val result = computeSub(graphAllocator, dummyContext, step2, tensorD)
```

### Mixed Precision Testing

Validates operations across different data types:
- F32 ↔ F16 conversion accuracy
- Quantized ↔ Float operations
- Precision loss analysis

### Error Handling Validation

Tests robustness against:
- Division by zero
- Dimension mismatches
- Invalid inputs
- Memory exhaustion

## Reference Validation

### Analytical Validation

Tests against mathematically known results:
- Identity operations
- Linear combinations  
- Activation function properties
- Matrix multiplication properties

### Cross-Implementation Validation

Compares against reference implementations:
- Upstream llama.cpp patterns
- Known good results from literature
- Cross-platform consistency

### Regression Testing

Maintains baseline results to prevent performance/accuracy regressions:
- Store known good outputs
- Validate against historical results
- Detect unintended changes

## Test Execution Guidelines

### Running Specific Test Categories

```bash
# Run all tests
./gradlew allTests

# Run specific test classes
./gradlew nativeTest --tests "*GGMLExtendedOpsTest*"
./gradlew nativeTest --tests "*GGMLPerformanceBenchmarkTest*"
./gradlew nativeTest --tests "*GGMLStandardizedQuantizationTest*"
```

### Test Configuration

Tests are designed to be:
- **Deterministic:** Consistent results across runs
- **Independent:** No dependencies between test methods
- **Comprehensive:** Cover normal, edge, and error cases
- **Fast:** Complete within reasonable time bounds

### Memory Management

Tests manage memory allocation carefully:
- Reset allocators between tests
- Use consistent buffer sizes
- Monitor for memory leaks
- Handle allocation failures gracefully

## Expected Test Results

### Success Criteria

**Unit Tests:**
- All basic operations pass analytical validation
- Quantization accuracy within defined thresholds
- Performance benchmarks complete successfully
- No memory allocation failures

**Integration Tests:**
- Complex computation chains produce expected results
- Mixed precision operations maintain accuracy
- Error handling works correctly
- Memory stress tests complete without issues

**Validation Tests:**
- Reference comparison passes within tolerances
- Cross-platform consistency maintained
- Regression baselines unchanged
- Numerical stability validated

### Failure Analysis

When tests fail, check:

1. **Accuracy Issues:**
   - Compare error metrics against thresholds
   - Check for precision loss in data type conversions
   - Validate input data ranges and characteristics

2. **Performance Issues:**
   - Profile memory allocation patterns
   - Check for algorithmic inefficiencies
   - Validate threading and parallelization

3. **Integration Issues:**
   - Verify operation chaining logic
   - Check dimension compatibility
   - Validate memory allocation between operations

## Contributing New Tests

When adding new tests:

1. **Choose Appropriate Category:**
   - Unit tests for basic operations
   - Integration tests for complex workflows
   - Performance tests for benchmark validation
   - Reference tests for accuracy validation

2. **Follow Naming Conventions:**
   - Use descriptive test names
   - Include data types and operation names
   - Specify test conditions (e.g., edge cases)

3. **Use Appropriate Thresholds:**
   - Consult existing threshold documentation
   - Consider operation complexity and precision
   - Document rationale for threshold choices

4. **Include Multiple Test Vectors:**
   - Normal operation cases
   - Edge cases and boundary conditions
   - Error conditions and invalid inputs
   - Performance stress cases

5. **Document Test Purpose:**
   - Clear description of what is being tested
   - Expected behavior and success criteria
   - Known limitations or assumptions

This testing infrastructure ensures the GGML Kotlin port maintains high accuracy, performance, and compatibility with the upstream llama.cpp implementation.