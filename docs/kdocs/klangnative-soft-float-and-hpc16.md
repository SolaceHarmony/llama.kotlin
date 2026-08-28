# KLang Soft‑Float and 16‑bit Limb Backbone

Date: 2025‑09‑21

This document summarizes the KLang numeric core added to the project: a pure‑Kotlin soft‑float stack (faithful to LLVM compiler‑rt) and a 16‑bit limb engine used for exact wide integer intermediates. It is the portability and determinism layer that our quantization paths and future SIMD/GPU backends build upon.

## Goals
- Bit‑exact IEEE‑754 behavior for Float32 (then Float16/Float64/BF16), independent of platform quirks (FTZ/DAZ, JIT, libm variants)
- No native code requirement for correctness; SIMD/GPU are optional accelerators
- Provide precise 128‑bit integer intermediates required by Float64 ops

## Modules & Types
- `io.github.kotlinmania.llama.lang.fp`
  - `CFloat32`: inline value class with operators +, −, ×, ÷
  - Future: `CFloat16`, `CBF16`, `CFloat64`
- `io.github.kotlinmania.llama.lang.bitwise`
  - `Float32Math`: compiler‑rt transliterations (add/sub/mul/div/sqrt; rounding nearest‑even by default)
  - `CFloatTrace` (diagnostics), `DoubleDouble` (widened accumulators for analysis only)
- `io.github.kotlinmania.llama.lang.int.hpc`
  - `HPC16x4` (64‑bit, 4×16‑bit limbs): add/sub/compare/shifts
  - `HPC16x8` (128‑bit, 8×16‑bit limbs): add/sub/compare/shifts, 64×64→128 mul
  - TODO: 128/64 division (Knuth D), full carry propagation passthrough for larger inputs

## Implemented (Float32)
- Division: exact port of compiler‑rt `fp_div_impl.inc`
  - Normalization of subnormals, quotient formation with R/G/S, remainder→sticky, nearest‑even rounding
  - Verified by directed tests: ratios, signed zeros, Inf/NaN ladder
- Operators: symmetric Float‑on‑LHS for ergonomics (`Float +/-/*// CFloat32`)

## Roadmap
1) Float32
   - Tighten `mulBits` to `fp_mul_impl.inc`; add boundary/tie/denorm tests
   - Implement `sqrtBits` (`fp_sqrt_impl.inc`) + tests
   - Conversions: int/uint ↔ float32; float32 ↔ float64
2) HPC16
   - Implement 128/64 division; extend 64×64→128 mul with full carry chain
3) Float64
   - Port add/sub/mul/div/sqrt using HPC16x4/x8 (compiler‑rt semantics)
   - Conversions: {i32,u32,i64,u64} ↔ f64; f32 ↔ f64
4) Float16/BFloat16
   - `CFloat16` arithmetic + conversions
   - `CBF16` exact conversions (ties‑to‑even) + arithmetic

## Rounding Modes & Exceptions
- Currently fixed to nearest, ties‑to‑even (matching common compiler‑rt build)
- Internals leave hook points to add other IEEE rounding modes later
- FP exceptions (inexact/underflow/overflow) are not surfaced yet; hooks remain

## Testing
- Bit‑exact unit tests (directed + fuzz): raw‑bit equality or both NaN
- Cross‑checks via C harnesses for edge cases
- Limb tests: JVM BigInteger oracle (tests only) will be added for large vectors

## Integration Notes
- Keep KLang as the reference path; add platform SIMD/GPU backends behind the same API once reference is green
- Prefer destination‑based tensor ops throughout; avoid legacy allocation paths

