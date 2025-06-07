# llama.kotlin Project Instructions

This file provides guidance for future Codex agents working on the Kotlin port of `llama.cpp`.
It summarizes the current project state and lists recommended next steps. Before starting new work, read `KOTLIN_PORT_CHECKLIST.md` in the repository root for a detailed roadmap.

## Project Overview
- The repository is a work‐in‐progress port of `llama.cpp` to Kotlin/Native.
- Kotlin sources live under `src/nativeMain/kotlin/ai/solace/llamakotlin`.
- The original C/C++ sources remain under `src` while porting progresses.
- Design notes and porting progress are documented in:
  - `KOTLIN_PORT_CHECKLIST.md`
  - `KOTLIN_PORT_STATUS.md`
  - `GGML_COMPUTE_OPS_DESIGN.md`
  - `TENSOR_OPERATIONS_DESIGN.md`

## Coding Guidelines
- Use idiomatic Kotlin style with descriptive names and KDoc comments.
- Keep code modular—separate tensor creation logic from compute kernels.
- Prefer immutable data structures where practical.
- Document placeholders or incomplete implementations with `TODO` comments.

## Build and Test
- The project uses Gradle. Build and run tests with `./gradlew build`.
- Network access may be required to download dependencies; configure as needed.
- If Gradle fails with a `PKIX path building failed` SSL error, install Java and Gradle via SDKMAN:
  ```bash
  curl -s "https://get.sdkman.io" | bash
  source "$HOME/.sdkman/bin/sdkman-init.sh"
  sdk install java 17.0.9-tem
  sdk install gradle 8.13
  ```
  Alternatively set `GRADLE_OPTS="-Djavax.net.ssl.trustStore=/etc/ssl/certs/java/cacerts"` to use the system certificate store.
- Unit tests should be placed under `src/nativeTest/kotlin`.
- C++ tests under `tests` are not required for the Kotlin port.

## Recommended Next Steps
1. **Finish Tensor Operations**
   - Implement computation functions in `GGMLComputeOps.kt` as described in `GGML_COMPUTE_OPS_DESIGN.md`.
   - Expand support for additional tensor types (F16, I8, I16, I64, quantized types).
   - Optimize operations using SIMD and multithreading where possible.

2. **Computation Graph Enhancements**
   - Extend `GGMLGraph.kt` with automatic differentiation support.
   - Add graph optimization passes for redundant operation removal.

3. **CPU Backend Implementation**
   - Create CPU-specific execution paths for tensor operations.
   - Investigate Kotlin/Native interop with C for performance‑critical sections.
   - Begin integrating multi-threading support.

4. **Quantization Support**
   - Implement 1.5‑bit through 8‑bit quantization formats.
   - Add quantized operation implementations and conversion utilities.

5. **Testing Infrastructure**
   - Add unit tests covering tensor creation, basic math ops and graph execution.
   - Provide sample models or fixtures for integration tests.

6. **Documentation Updates**
   - Keep `KOTLIN_PORT_STATUS.md` and `KOTLIN_PORT_CHECKLIST.md` up to date.
   - Document new modules and APIs in the `/docs` folder.

Follow this guide when extending the Kotlin port. Keep commits focused and include
relevant tests whenever possible.
