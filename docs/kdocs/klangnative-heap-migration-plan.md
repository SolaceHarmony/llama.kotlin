# KLang Heap Migration Plan

## Current Architecture

**Tensor Memory Model:**
- `GGMLTensor.data: Any?` - Legacy field, mostly unused
- `GGMLTensor.bufferId: Int` - Which buffer this tensor belongs to
- `GGMLTensor.dataOffset: ULong` - Offset into the ByteArray buffer

**Allocator:**
- `GGMLGraphAllocator` - Manages a primary ByteArray buffer
- Offset-based allocation (similar to pointer model)
- ByteArrayExtensions for typed access (getFloatLe, setFloatLe, etc.)

## KLang Capabilities (Vendored Version)

**Memory Management (mem/ package):**
- `GlobalHeap` - Single ByteArray-backed C-style heap
- `KMalloc` - malloc/calloc/realloc/free with 16-byte alignment
- `CPointer<T>` - Type-safe pointer abstraction (Int byte offsets)
- Deterministic little-endian I/O across all platforms

**C Types (fp/ and int/ packages):**
- `CFloat32` - Bit-exact IEEE-754 single precision
- `CDouble` - Bit-exact IEEE-754 double precision
- `CFloat128` - Double-double precision (~106-bit mantissa)
- `C_UInt128`, `C_Int128` - Full 128-bit integer support

**Benefits:**
- Exact C semantics (solves Q2_K parity drift issues)
- Zero-copy operations (heap-based storage)
- Cross-platform determinism (no float rounding differences)
- Eliminates ByteArray bloat (single global heap)

## Migration Strategy

### Phase 1: Update Build Configuration
- [ ] Add external/klangnative to source sets in build.gradle.kts
- [ ] Update imports from `io.github.kotlinmania.llama.lang.bitwise` to vendored packages
- [ ] Test that existing code compiles with vendored klangnative

### Phase 2: Introduce GlobalHeap Layer
- [ ] Create `GGMLHeap` wrapper around KLang's GlobalHeap
- [ ] Add heap initialization in GGMLContext
- [ ] Keep ByteArray path as fallback during migration

### Phase 3: Migrate GGMLGraphAllocator
**Replace:**
```kotlin
// OLD: ByteArray-based
var primaryBuffer: ByteArray? = null
tensor.dataOffset: ULong
```

**With:**
```kotlin
// NEW: GlobalHeap-based
val heapPtr: Int  // Offset into GlobalHeap
```

**Changes:**
- Replace `ByteArray` allocations with `GlobalHeap.mallocBytes()`
- Replace `dataOffset` access with `GlobalHeap.lf(ptr)` / `GlobalHeap.sf(ptr, value)`
- Maintain alignment requirements (16-byte for tensors)

### Phase 4: Migrate ByteArrayExtensions
**Replace:**
```kotlin
// OLD
buffer.getFloatLe(offset)
buffer.setFloatLe(offset, value)
```

**With:**
```kotlin
// NEW
GlobalHeap.lf(ptr + offset)  // load float
GlobalHeap.sf(ptr + offset, value)  // store float
```

### Phase 5: Use CFloat32 for Compute Operations
**Quantization Code:**
- Use `CFloat32` for scale calculations (exact C semantics)
- Use `Float32Math` operations instead of Kotlin operators
- Eliminates float drift in Q2_K refinement loops

**Example:**
```kotlin
// OLD: Kotlin Float (platform-dependent rounding)
val scale = maxAbs / 127.0f

// NEW: CFloat32 (bit-exact C semantics)
val scale = CFloat32.fromFloat(maxAbs) / CFloat32.fromFloat(127.0f)
```

### Phase 6: Remove Embedded KLang
- [ ] Delete `src/commonMain/kotlin/io.github.kotlinmania.llama.lang/`
- [ ] Verify all imports point to `external/klangnative`
- [ ] Update tests to use vendored klangnative

## Migration Order

1. **Non-breaking**: Update imports, add vendored klangnative to build
2. **Low-risk**: Introduce GGMLHeap wrapper (side-by-side)
3. **Medium-risk**: Migrate allocator to use GlobalHeap internally
4. **High-value**: Use CFloat32 in quantization code (fixes parity)
5. **Cleanup**: Remove embedded klang, ByteArrayExtensions

## Expected Benefits

### Numeric Parity
- **Q2_K refinement drift** → Fixed by CFloat32 exact arithmetic
- **Q5_K/Q6_K packing** → Use KLang BitShiftEngine for exact C behavior
- **Float16 ops** → Use CFloat16 instead of Short bit-twiddling

### Memory Efficiency
- Single global heap instead of per-allocator ByteArrays
- Better memory reuse with KMalloc coalescing
- Zero-copy between tensors (pointer arithmetic)

### Code Quality
- Remove custom ByteArrayExtensions (use KLang mem ops)
- C-compatible type system (easier to port C code)
- Better documentation (KLang has comprehensive docs)

## Risks & Mitigation

**Risk**: GlobalHeap size limits
- **Mitigation**: Make heap size configurable, default to 1GB

**Risk**: Performance regression
- **Mitigation**: Benchmark before/after, KLang is optimized for this

**Risk**: Breaking existing tests
- **Mitigation**: Migrate incrementally, keep fallback paths initially

## Success Criteria

- [ ] All existing tests pass with vendored klangnative
- [ ] Q2_K snapshot test achieves parity (currently red)
- [ ] macosArm64Test suite runs green
- [ ] No performance regression vs. ByteArray implementation
- [ ] Embedded klang removed from src/

## Next Steps

1. Update build.gradle.kts to include external/klangnative
2. Create GGMLHeap wrapper class
3. Write migration tests
4. Begin Phase 1 migration
