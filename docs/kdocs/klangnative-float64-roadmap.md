# KLang Float64 Roadmap (compiler‑rt transliteration)

Updated: 2025‑09‑21

This note outlines the plan to implement IEEE‑754 binary64 (double) in pure Kotlin, faithfully ported from LLVM compiler‑rt builtins, using our 16‑bit limb backbone.

## Goals
- Bit‑exact Float64 arithmetic (+, −, ×, ÷, √) matching compiler‑rt semantics (nearest, ties‑to‑even by default).
- No native interop required for correctness; SIMD/GPU remain optional accelerators.
- Deterministic across platforms (no FTZ/DAZ surprises).

## Dependencies
- HPC limb engine
  - Done: `HPC16x4` (4×16‑bit = 64‑bit), `HPC16x8` (8×16‑bit = 128‑bit).
  - Pending: full carry chain for 64×64→128 multiply; 128/64 division (Knuth D), wide shifts/compare utilities.

## Implementation Plan
1) Integer backbone (io.github.kotlinmania.llama.lang.int.hpc)
   - Complete 64×64→128 multiply with full carry propagation (HPC16x8).
   - Implement 128/64 division (normalize → trial quotient → correction) in base 2¹⁶.
   - Add helpers: left/right shift by k bits, by limbs; compare; isZero.

2) Float64 builtins (io.github.kotlinmania.llama.lang.bitwise)
   - Port `fp_add_impl.inc` / `fp_sub_impl.inc` variants for 52‑bit fraction + hidden bit.
   - Port `fp_mul_impl.inc`: exact 106‑bit product via HPC16x8; normalize and round.
   - Port `fp_div_impl.inc`: scaled quotient + remainder→sticky, normalization and rounding.
   - Port `fp_sqrt_impl.inc`.
   - Rounding: nearest‑even initially; keep hooks for other IEEE modes and for inexact/underflow/overflow flags (no‑ops for now).

3) Public API (io.github.kotlinmania.llama.lang.fp)
   - `CFloat64` inline type with operators +, −, ×, ÷; `sqrt()` function.
   - Classification (NaN, sNaN/qNaN, ±inf, ±0, subnormal), `copysign`, `fmin/fmax` with signed‑zero rules.
   - Conversions: {i32,u32,i64,u64} ↔ f64; f32 ↔ f64 (ties‑to‑even).
   - Symmetric Float‑on‑LHS operator extensions.

4) Tests & Oracles
   - Directed bit‑exact vectors (boundaries: carry/renorm, ties, subnormals, signed zeros, NaN payloads).
   - Fuzz (JVM): compare against `BigInteger`‑backed integer oracles for limb ops; cross‑check `Double` only when not relying on host rounding quirks.
   - C harness: small binaries compiling compiler‑rt paths to validate tricky rounding cases.
   - Rule: equal raw bits or both NaN.

## Milestones
- M1: HPC16 wide mul carry + 128/64 division.
- M2: `CFloat64` add/sub + conversions; tests green.
- M3: mul/div; tests green on directed + fuzz.
- M4: sqrt; full suite green; integrate into higher layers where needed.

## Integration Notes
- Keep Float64 behind the same scalar API as Float32; SIMD/Metal backends can be added later behind expect/actual or strategy injection.
- Use Float64 only where model math requires it; default paths stay Float32 for performance unless precision mandates double.

## Risks & Mitigations
- 128‑bit intermediates: ensure limb ops are allocation‑free and constant‑time per limb to avoid perf cliffs.
- Rounding corners: maintain R/G/S lanes and sticky from remainder; add micro‑tests for half‑way cases.
- Cross‑platform determinism: rely solely on limb math and bit‑ops, not host FP, inside the core algorithms.

## Related
- docs/kdocs/klangnative-soft-float-and-hpc16.md
- docs/kdocs/kotlin-native-simd-plan.md
- docs/kdocs/test-triage-2025-09-21.md
