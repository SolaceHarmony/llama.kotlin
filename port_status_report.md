# Code Port - Progress Report

**Generated:** 2026-05-07
**Source:** ggml/src
**Target:** src/commonMain/kotlin/io.github.kotlinmania.llama.ore

## Executive Summary

| Metric | Count | Percentage |
|--------|-------|------------|
| Function parity | 547/11418 matched (target 1933) | 4.8% |
| Class/type parity | 17/789 matched (target 170) | 2.2% |
| Combined symbol parity | 564/12207 matched (target 2103) | 4.6% |
| Average function body similarity | 0.00 | required score |
| Missing source functions | 10703 | 0% parity until ported |
| Missing source classes/types | 694 | 0% parity until ported |
| Missing source symbol files | 179 | 11397 symbols |
| Cheat/scoring failures | 20 | forced to 0% |
| Total source files | 256 | 100% |
| Target units (paired) | 47 | - |
| Target files (total) | 47 | - |
| Porting progress | 21 | 8.2% (matched) |
| Missing files | 235 | 91.8% |

## Port Quality Analysis

**Average Function Similarity:** 0.00

Similarity in this report is the required function-by-function body/parameter score. Class/type parity and symbol deficits are reported beside it; whole-file shape is diagnostic only.

**Work Distribution:**
- Critical (<0.60): 21 files (100.0% of matched)
- Needs review (0.60-0.84): 0 files (0.0% of matched)

## Worst Function Scores First

Every matched file is listed from lowest function body/parameter similarity upward. Missing symbol names are not capped.

| Rank | Source | Target | Function similarity | Functions | Missing functions | Types | Missing types | Tests | Symbol deficit | Priority |
|------|--------|--------|---------------------|-----------|-------------------|-------|---------------|-------|----------------|----------|
| 1 | `ggml-impl` | `core.NumericConversions [ZERO]` | 0.00 | 47/51 matched (target 77) | `ggml_aligned_malloc`, `ggml_aligned_free`, `gguf_type_size`, `gguf_write_to_buf` | 6/12 matched (target 8) | `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order`, `ggml_cgraph`, `gguf_type`, `gguf_context` | - | 10 | 58106312.0 |
| 2 | `ggml` | `core.GGMLOps [ZERO]` | 0.00 | 0/1 matched (target 431) | `ggml_uncaught_exception` | 0/0 matched (target 23) | _none_ | - | 1 | 50010112.0 |
| 3 | `ggml-backend-impl` | `core.GGMLBackendImpl [ZERO]` | 0.00 | 7/8 matched (target 39) | `ggml_backend_multi_buffer_set_usage` | 2/18 matched (target 17) | `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer`, `ggml_backend_buffer_usage`, `ggml_context`, `ggml_backend_i`, `ggml_cgraph`, `ggml_backend`, `ggml_backend_event`, `ggml_backend_device`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg` | - | 17 | 47172608.0 |
| 4 | `ggml-cpu.common` | `core.GGMLCpuCommon [ZERO]` | 0.00 | 8/8 matched (target 20) | _none_ | 2/4 matched (target 6) | `ggml_compute_params`, `ggml_tensor` | - | 2 | 47021208.0 |
| 5 | `ggml-backend` | `core.GGMLBackendUtils [STUB]` | 0.00 | 128/131 matched (target 178) | `ggml_backend_buffer_init`, `ggml_backend_buffer_copy_tensor`, `ggml_backend_buffer_is_multi_buffer` | 3/16 matched (target 7) | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer_usage`, `ggml_cgraph`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_context`, `ggml_op`, `ggml_backend_sched`, `ggml_hash_set`, `ggml_init_params`, `ggml_backend_buffer_type` | - | 16 | 20174710.0 |
| 6 | `ggml-common` | `core.GGMLCommon [ZERO]` | 0.00 | 0/2 matched (target 53) | `union`, `GGML_TABLE_BEGIN` | 0/0 matched (target 28) | _none_ | - | 2 | 16020210.0 |
| 7 | `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend [ZERO]` | 0.00 | 30/33 matched (target 159) | `ggml_backend_cpu_device_get_extra_buffers_type`, `ggml_backend_cpu_device_context`, `ggml_backend_cpu_reg` | 1/13 matched (target 14) | `ggml_backend_cpu_context`, `ggml_backend_plan_cpu`, `ggml_cgraph`, `ggml_status`, `ggml_backend_i`, `ggml_backend_cpu_device_context`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_tensor`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg` | - | 15 | 12154610.0 |
| 8 | `ggml-cpu.ggml-cpu-impl` | `core.GGMLCpuImpl [ZERO]` | 0.00 | 31/32 matched (target 39) | `ggml_vec_dot` | 2/7 matched (target 2) | `ggml_int16x8x2_t`, `ggml_uint8x16x2_t`, `ggml_uint8x16x4_t`, `ggml_int8x16x2_t`, `ggml_int8x16x4_t` | - | 6 | 11063910.0 |
| 9 | `ggml-cpu.simd-mappings` | `simd.GGMLSimd [ZERO]` | 0.00 | 20/20 matched (target 34) | _none_ | 0/0 matched (target 1) | _none_ | - | 0 | 10002010.0 |
| 10 | `ggml-cpu.traits` | `core.GGMLCpuTraits [ZERO]` | 0.00 | 6/7 matched (target 6) | `ggml_backend_cpu_get_extra_buffer_types` | 0/4 matched (target 2) | `ggml_compute_params`, `ggml_tensor`, `tensor_traits`, `extra_buffer_type` | - | 5 | 9051110.0 |
| 11 | `ggml-quants` | `core.GGMLQuantsRef [ZERO]` | 0.00 | 75/75 matched (target 122) | _none_ | 0/1 matched (target 3) | `ggml_type` | - | 1 | 6017610.0 |
| 12 | `ggml-cpu.quants` | `core.GGMLQuants [ZERO]` | 0.00 | 43/70 matched (target 75) | `quantize_row_q8_0`, `quantize_row_q8_1`, `quantize_row_q8_K`, `ggml_vec_dot_q1_0_q8_0`, `ggml_vec_dot_q4_0_q8_0`, `ggml_vec_dot_q4_1_q8_1`, `ggml_vec_dot_q5_0_q8_0`, `ggml_vec_dot_q5_1_q8_1`, `ggml_vec_dot_q8_0_q8_0`, `ggml_vec_dot_mxfp4_q8_0`, `ggml_vec_dot_nvfp4_q8_0`, `ggml_vec_dot_q2_K_q8_K`, `ggml_vec_dot_q3_K_q8_K`, `ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q5_K_q8_K`, `ggml_vec_dot_q6_K_q8_K`, `ggml_vec_dot_tq1_0_q8_K`, `ggml_vec_dot_tq2_0_q8_K`, `ggml_vec_dot_iq2_xxs_q8_K`, `ggml_vec_dot_iq2_xs_q8_K`, `ggml_vec_dot_iq2_s_q8_K`, `ggml_vec_dot_iq3_xxs_q8_K`, `ggml_vec_dot_iq1_s_q8_K`, `ggml_vec_dot_iq1_m_q8_K`, `ggml_vec_dot_iq4_nl_q8_0`, `ggml_vec_dot_iq4_xs_q8_K`, `ggml_vec_dot_iq3_s_q8_K` | 0/0 matched (target 1) | _none_ | - | 27 | 4277010.0 |
| 13 | `ggml-sycl.type` | `core.GGMLTypesTest [ZERO]` | 0.00 | 0/3 matched (target 9) | `float_to_e4m3`, `e4m3_to_float`, `__nv_fp8_e4m3` | 0/1 matched | `__nv_fp8_e4m3` | - | 4 | 2040410.0 |
| 14 | `ggml-sycl.backend` | `core.GGMLBackendTest [ZERO]` | 0.00 | 0/0 matched (target 7) | _none_ | 0/0 matched (target 1) | _none_ | - | 0 | 2000010.0 |
| 15 | `ggml-threading` | `core.GGMLScheduler [ZERO]` | 0.00 | 2/2 matched (target 18) | _none_ | 0/0 matched (target 7) | _none_ | - | 0 | 1000210.0 |
| 16 | `ggml-cpu.ops` | `core.GGMLComputeOps [STUB]` | 0.00 | 150/220 matched (target 482) | `ggml_compute_forward_dup_flt`, `ggml_compute_forward_dup_to_q`, `GGML_TENSOR_LOCALS`, `ggml_compute_forward_set_rows_f32`, `rotate_pairs`, `ggml_compute_forward_rope_flt`, `ggml_call_mul_mat`, `ggml_compute_forward_conv_transpose_2d_impl`, `ggml_compute_forward_upscale_f32`, `ggml_compute_forward_upscale`, `ggml_compute_forward_pad_f32`, `ggml_compute_forward_pad`, `ggml_compute_forward_pad_reflect_1d`, `for`, `ggml_wrap_index`, `ggml_compute_forward_roll_f32`, `ggml_compute_forward_roll`, `ggml_compute_forward_arange_f32`, `ggml_compute_forward_arange`, `ggml_compute_forward_timestep_embedding_f32`, `ggml_compute_forward_timestep_embedding`, `ggml_compute_forward_argsort_f32`, `ggml_compute_forward_argsort`, `ggml_compute_forward_top_k_f32`, `ggml_compute_forward_top_k`, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, `ggml_compute_forward_flash_attn_ext_tiled`, `ggml_flash_attn_ext_reduce_partials`, `ggml_compute_forward_flash_attn_ext_f16`, `ggml_compute_forward_flash_attn_ext`, `ggml_compute_forward_flash_attn_back_f32`, `ggml_compute_forward_flash_attn_back`, `ggml_compute_forward_ssm_conv_f32`, `ggml_compute_forward_ssm_conv`, `ggml_compute_forward_ssm_scan_f32`, `ggml_compute_forward_ssm_scan`, `ggml_compute_forward_win_part_f32`, `ggml_compute_forward_win_part`, `ggml_compute_forward_win_unpart_f32`, `ggml_compute_forward_win_unpart`, `ggml_compute_forward_unary`, `ggml_compute_forward_glu`, `ggml_compute_forward_get_rel_pos_f16`, `ggml_compute_forward_get_rel_pos`, `ggml_compute_forward_add_rel_pos_f32`, `ggml_compute_forward_add_rel_pos`, `ggml_compute_forward_rwkv_wkv6_f32`, `ggml_compute_forward_rwkv_wkv6`, `ggml_compute_forward_gla_f32`, `ggml_compute_forward_gla`, `ggml_compute_forward_solve_tri_f32`, `ggml_compute_forward_solve_tri`, `ggml_compute_forward_gated_delta_net_one_chunk`, `ggml_compute_forward_gated_delta_net_f32`, `ggml_compute_forward_gated_delta_net`, `ggml_compute_forward_rwkv_wkv7_f32`, `ggml_compute_forward_rwkv_wkv7`, `ggml_compute_forward_map_custom1`, `ggml_compute_forward_map_custom2`, `ggml_compute_forward_map_custom3`, `ggml_compute_forward_custom`, `ggml_compute_forward_cross_entropy_loss_f32`, `ggml_compute_forward_cross_entropy_loss`, `ggml_compute_forward_cross_entropy_loss_back_f32`, `ggml_compute_forward_cross_entropy_loss_back`, `ggml_compute_forward_opt_step_adamw_f32`, `ggml_compute_forward_opt_step_adamw`, `ggml_compute_forward_opt_step_sgd_f32`, `ggml_compute_forward_opt_step_sgd`, `ggml_compute_forward_mul_mat` | 1/10 matched (target 15) | `ggml_tensor`, `ggml_sort_order`, `cmp_argsort`, `cmp_top_k`, `ggml_compute_params`, `ggml_map_custom1_op_params`, `ggml_map_custom2_op_params`, `ggml_map_custom3_op_params`, `ggml_custom_op_params` | - | 79 | 813010.0 |
| 17 | `ggml-openvino.utils` | `core.GGMLTensorUtils [ZERO]` | 0.00 | 0/22 matched (target 6) | `graph_key`, `ov_runtime_context`, `ov_graph_compute`, `ov_graph_compute_dynamic`, `ov_graph_compute_static`, `checksum`, `print_input_tensor_info`, `print_output_tensor_info`, `pad_input`, `padded`, `set_zero_diagonal`, `get_inp_pos_tensor`, `get_is_prefill`, `get_ov_input_tensor`, `get_ov_input_tensor_static_decode`, `get_ov_input_tensor_static_prefill`, `create_ov_output_tensor`, `is_naive`, `naive_compute`, `key`, `new_state_tensor`, `convert_ggml_input_to_ov` | 0/5 matched (target 1) | `graph_key`, `graph_key_hash`, `ov_runtime_context`, `ggml_status`, `ggml_cgraph` | - | 27 | 272710.0 |
| 18 | `openvino.utils` | `core.GGMLTestUtils [ZERO]` | 0.00 | 0/17 matched (target 46) | `getCurrentTime`, `dump_ov_model`, `num_inputs_check`, `non_cont_dim`, `argsort_descend`, `sorted_descend`, `is_permuted`, `permute`, `get_dimensions`, `rename_outputs_with_suffix`, `make_sin_cos`, `process_view_input`, `translate_1to1_match_2_inputs`, `rope_yarn_ramp_mix`, `dim_ids_vec`, `ggml_rope_yarn_corr_dim`, `ggml_rope_yarn_corr_dims` | 0/0 matched (target 13) | _none_ | - | 17 | 171710.0 |
| 19 | `ggml-virtgpu.ggml-backend` | `core.GGMLBackend [ZERO]` | 0.00 | 0/6 matched (target 84) | `ggml_backend_remoting_get_name`, `ggml_backend_remoting_free`, `ggml_backend_remoting_graph_compute`, `ggml_backend_remoting_graph_optimize`, `ggml_backend_remoting_guid`, `ggml_backend_remoting_device_init` | 0/0 matched (target 19) | _none_ | - | 6 | 60610.0 |
| 20 | `ggml-sycl.quants` | `core.GGMLCpuQuants [ZERO]` | 0.00 | 0/3 matched (target 45) | `get_block_offset`, `get_d_offset`, `block_to_q8_1_ratio` | 0/2 matched (target 0) | `block_q_t`, `traits` | - | 5 | 50510.0 |
| 21 | `backend.backend` | `core.GGMLBackendIntegrationTest` | 0.00 | 0/4 matched (target 3) | `log_to_file_callback`, `apir_backend_deinit`, `apir_backend_initialize`, `apir_backend_dispatcher` | 0/2 matched (target 1) | `ggml_log_level`, `virgl_apir_callbacks` | - | 6 | 60610.0 |

## Cheat Detection / Scoring Failures

- `ggml-impl` -> `core.NumericConversions [ZERO]`: function-by-function score forced to 0. NumericConversions.kt: snake_case identifier `ggml_up32` in Kotlin code; NumericConversions.kt: snake_case identifier `fp32_from_bits` in Kotlin comments; NumericConversions.kt: Rust lifetime explanation in Kotlin comments
- `ggml` -> `core.GGMLOps [ZERO]`: function-by-function score forced to 0. GGMLOps.kt: snake_case identifier `ggml_aligned_malloc` in Kotlin code; GGMLOps.kt: snake_case identifier `ggml_nelements` in Kotlin comments; GGMLOps.kt: Rust lifetime explanation in Kotlin comments; GGMLTypes.kt: snake_case identifier `getQ8_0BlockScale` in Kotlin code; GGMLTypes.kt: snake_case identifier `type_size` in Kotlin comments; GGMLTypes.kt: Rust `match` expression in Kotlin comments
- `ggml-backend-impl` -> `core.GGMLBackendImpl [ZERO]`: function-by-function score forced to 0. GGMLBackendImpl.kt: snake_case identifier `size_t` in Kotlin comments; GGMLBackendImpl.kt: score-padding suppression annotation `@Suppress` in Kotlin code; GGMLBackendImpl.kt: Rust lifetime explanation in Kotlin comments
- `ggml-cpu.common` -> `core.GGMLCpuCommon [ZERO]`: function-by-function score forced to 0. GGMLCpuCommon.kt: snake_case identifier `ggml_fa_tile_config` in Kotlin comments
- `ggml-backend` -> `core.GGMLBackendUtils [STUB]`: function-by-function score forced to 0. target contains TODO/stub/placeholder markers in function bodies; GGMLBackendUtils.kt: snake_case identifier `ggml_hash_size` in Kotlin code; GGMLBackendUtils.kt: snake_case identifier `ggml_backend_buft_name` in Kotlin comments; GGMLBackendUtils.kt: Rust lifetime explanation in Kotlin comments
- `ggml-common` -> `core.GGMLCommon [ZERO]`: function-by-function score forced to 0. GGMLCommon.kt: snake_case identifier `BlockQ1_0` in Kotlin code; GGMLCommon.kt: snake_case identifier `int8_t` in Kotlin comments; GGMLCommon.kt: Rust lifetime explanation in Kotlin comments; GGMLIQGrids.kt: snake_case identifier `iq2xxs_grid` in Kotlin code
- `ggml-cpu.ggml-cpu` -> `core.GGMLCpuBackend [ZERO]`: function-by-function score forced to 0. GGMLCpuBackend.kt: snake_case identifier `ggml_aligned_malloc` in Kotlin code; GGMLCpuBackend.kt: snake_case identifier `get_name` in Kotlin comments; GGMLCpuBackend.kt: Rust lifetime explanation in Kotlin comments; GGMLCpuExecutor.kt: snake_case identifier `g_numa_n_nodes` in Kotlin code; GGMLCpuExecutor.kt: snake_case identifier `ggml_cplan` in Kotlin comments; GGMLCpuExecutor.kt: Rust lifetime explanation in Kotlin comments
- `ggml-cpu.ggml-cpu-impl` -> `core.GGMLCpuImpl [ZERO]`: function-by-function score forced to 0. GGMLCpuImpl.kt: snake_case identifier `ggml_threadpool` in Kotlin comments
- `ggml-cpu.simd-mappings` -> `simd.GGMLSimd [ZERO]`: function-by-function score forced to 0. GGMLSimd.kt: snake_case identifier `neon_compute_fp16_to_fp32` in Kotlin comments
- `ggml-cpu.traits` -> `core.GGMLCpuTraits [ZERO]`: function-by-function score forced to 0. GGMLCpuTraits.kt: snake_case identifier `tensor_traits` in Kotlin comments
- `ggml-quants` -> `core.GGMLQuantsRef [ZERO]`: function-by-function score forced to 0. GGMLQuantsRef.kt: snake_case identifier `best_index_int8` in Kotlin code; GGMLQuantsRef.kt: snake_case identifier `iq2xxs_grid` in Kotlin comments
- `ggml-cpu.quants` -> `core.GGMLQuants [ZERO]`: function-by-function score forced to 0. GGMLQuants.kt: snake_case identifier `kmask_iq2xs` in Kotlin code; GGMLQuants.kt: snake_case identifier `ggml_vec_dot_q1_0_q8_0_generic` in Kotlin comments
- `ggml-sycl.type` -> `core.GGMLTypesTest [ZERO]`: function-by-function score forced to 0. GGMLTypesTest.kt: snake_case identifier `testAccessOutOfBounds_Indices` in Kotlin code; GGMLTypesTest.kt: Rust lifetime explanation in Kotlin comments
- `ggml-sycl.backend` -> `core.GGMLBackendTest [ZERO]`: function-by-function score forced to 0. no source functions found; report scoring is function-by-function only
- `ggml-threading` -> `core.GGMLScheduler [ZERO]`: function-by-function score forced to 0. GGMLScheduler.kt: snake_case identifier `ggml_critical_section_start` in Kotlin code
- `ggml-cpu.ops` -> `core.GGMLComputeOps [STUB]`: function-by-function score forced to 0. target contains TODO/stub/placeholder markers in function bodies; GGMLComputeOps.kt: snake_case identifier `getQ8_0BlockScale` in Kotlin code; GGMLComputeOps.kt: snake_case identifier `computeDotProductQ4_KF32` in Kotlin comments; GGMLComputeOps.kt: Rust `match` expression in Kotlin comments; GGMLComputeOps.kt: Rust lifetime explanation in Kotlin comments; GGMLComputeForward.kt: snake_case identifier `ggml_vec_gelu_f32` in Kotlin code; GGMLComputeForward.kt: snake_case identifier `snake_case` in Kotlin comments
- `ggml-openvino.utils` -> `core.GGMLTensorUtils [ZERO]`: function-by-function score forced to 0. GGMLTensorUtils.kt: Rust lifetime explanation in Kotlin comments
- `openvino.utils` -> `core.GGMLTestUtils [ZERO]`: function-by-function score forced to 0. GGMLTestUtils.kt: Rust lifetime explanation in Kotlin comments
- `ggml-virtgpu.ggml-backend` -> `core.GGMLBackend [ZERO]`: function-by-function score forced to 0. GGMLBackend.kt: snake_case identifier `ggml_hash_size` in Kotlin code; GGMLBackend.kt: snake_case identifier `ggml_backend_buffer_type_t` in Kotlin comments
- `ggml-sycl.quants` -> `core.GGMLCpuQuants [ZERO]`: function-by-function score forced to 0. GGMLCpuQuants.kt: snake_case identifier `quantize_row_q1_0_ref` in Kotlin code; GGMLCpuQuants.kt: snake_case identifier `int8_t` in Kotlin comments

### Critical Ports (Similarity < 0.60, Worst First)

These files need significant work:

- `ggml-impl` -> `core.NumericConversions [ZERO]` (0.00, 58 deps)
- `ggml` -> `core.GGMLOps [ZERO]` (0.00, 50 deps)
- `ggml-backend-impl` -> `core.GGMLBackendImpl [ZERO]` (0.00, 47 deps)
- `ggml-cpu.common` -> `core.GGMLCpuCommon [ZERO]` (0.00, 47 deps)
- `ggml-backend` -> `core.GGMLBackendUtils [STUB]` (0.00, 20 deps)
- `ggml-common` -> `core.GGMLCommon [ZERO]` (0.00, 16 deps)
- `ggml-cpu.ggml-cpu` -> `core.GGMLCpuBackend [ZERO]` (0.00, 12 deps)
- `ggml-cpu.ggml-cpu-impl` -> `core.GGMLCpuImpl [ZERO]` (0.00, 11 deps)
- `ggml-cpu.simd-mappings` -> `simd.GGMLSimd [ZERO]` (0.00, 10 deps)
- `ggml-cpu.traits` -> `core.GGMLCpuTraits [ZERO]` (0.00, 9 deps)
- `ggml-quants` -> `core.GGMLQuantsRef [ZERO]` (0.00, 6 deps)
- `ggml-cpu.quants` -> `core.GGMLQuants [ZERO]` (0.00, 4 deps)
- `ggml-sycl.type` -> `core.GGMLTypesTest [ZERO]` (0.00, 2 deps)
- `ggml-sycl.backend` -> `core.GGMLBackendTest [ZERO]` (0.00, 2 deps)
- `ggml-threading` -> `core.GGMLScheduler [ZERO]` (0.00, 1 deps)
- `ggml-cpu.ops` -> `core.GGMLComputeOps [STUB]` (0.00)
- `ggml-openvino.utils` -> `core.GGMLTensorUtils [ZERO]` (0.00)
- `openvino.utils` -> `core.GGMLTestUtils [ZERO]` (0.00)
- `ggml-virtgpu.ggml-backend` -> `core.GGMLBackend [ZERO]` (0.00)
- `ggml-sycl.quants` -> `core.GGMLCpuQuants [ZERO]` (0.00)
- `backend.backend` -> `core.GGMLBackendIntegrationTest` (0.00)

## Incorrect Ports (Missing Types)

These files are matched (often via `// port-lint`) but appear to be missing one or more type declarations
present in the Rust source file.

| Source | Target | Missing types | Examples |
|--------|--------|---------------|----------|
| `ggml-impl` | `core.NumericConversions [ZERO]` | 6/12 | `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order`, `ggml_cgraph`, `gguf_type`, `gguf_context` |
| `ggml-backend-impl` | `core.GGMLBackendImpl [ZERO]` | 16/18 | `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer`, `ggml_backend_buffer_usage`, `ggml_context`, `ggml_backend_i`, `ggml_cgraph`, `ggml_backend`, `ggml_backend_event`, `ggml_backend_device`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg` |
| `ggml-cpu.common` | `core.GGMLCpuCommon [ZERO]` | 2/4 | `ggml_compute_params`, `ggml_tensor` |
| `ggml-backend` | `core.GGMLBackendUtils [STUB]` | 13/16 | `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer_usage`, `ggml_cgraph`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_context`, `ggml_op`, `ggml_backend_sched`, `ggml_hash_set`, `ggml_init_params`, `ggml_backend_buffer_type` |
| `ggml-cpu.ggml-cpu` | `core.GGMLCpuBackend [ZERO]` | 12/13 | `ggml_backend_cpu_context`, `ggml_backend_plan_cpu`, `ggml_cgraph`, `ggml_status`, `ggml_backend_i`, `ggml_backend_cpu_device_context`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_tensor`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg` |
| `ggml-cpu.ggml-cpu-impl` | `core.GGMLCpuImpl [ZERO]` | 5/7 | `ggml_int16x8x2_t`, `ggml_uint8x16x2_t`, `ggml_uint8x16x4_t`, `ggml_int8x16x2_t`, `ggml_int8x16x4_t` |
| `ggml-cpu.traits` | `core.GGMLCpuTraits [ZERO]` | 4/4 | `ggml_compute_params`, `ggml_tensor`, `tensor_traits`, `extra_buffer_type` |
| `ggml-quants` | `core.GGMLQuantsRef [ZERO]` | 1/1 | `ggml_type` |
| `ggml-sycl.type` | `core.GGMLTypesTest [ZERO]` | 1/1 | `__nv_fp8_e4m3` |
| `ggml-cpu.ops` | `core.GGMLComputeOps [STUB]` | 9/10 | `ggml_tensor`, `ggml_sort_order`, `cmp_argsort`, `cmp_top_k`, `ggml_compute_params`, `ggml_map_custom1_op_params`, `ggml_map_custom2_op_params`, `ggml_map_custom3_op_params`, `ggml_custom_op_params` |
| `ggml-openvino.utils` | `core.GGMLTensorUtils [ZERO]` | 5/5 | `graph_key`, `graph_key_hash`, `ov_runtime_context`, `ggml_status`, `ggml_cgraph` |
| `backend.backend` | `core.GGMLBackendIntegrationTest` | 2/2 | `ggml_log_level`, `virgl_apir_callbacks` |
| `ggml-sycl.quants` | `core.GGMLCpuQuants [ZERO]` | 2/2 | `block_q_t`, `traits` |

## High Priority Missing Files

| Rank | Source file | Expected target | Deps | Functions | Classes/types | Symbols | Source path | Expected path |
|------|-------------|-----------------|------|-----------|---------------|---------|-------------|---------------|
| 1 | `ggml-vulkan.ggml-vulkan` | `ggmlvulkan.Ggml-vulkan` | 1 | 700 | 88 | 788 | `ggml-vulkan/ggml-vulkan.cpp` | `ggmlvulkan/Ggml-vulkan.kt` |
| 2 | `dpct.helper` | `ggmlsycl.dpct.Helper` | 9 | 619 | 29 | 648 | `ggml-sycl/dpct/helper.hpp` | `ggmlsycl/dpct/Helper.kt` |
| 3 | `ggml-opencl.ggml-opencl` | `ggmlopencl.Ggml-opencl` | 1 | 481 | 31 | 512 | `ggml-opencl/ggml-opencl.cpp` | `ggmlopencl/Ggml-opencl.kt` |
| 4 | `ggml-metal.ggml-metal-ops` | `ggmlmetal.Ggml-metal-ops` | 1 | 465 | 4 | 469 | `ggml-metal/ggml-metal-ops.h` | `ggmlmetal/Ggml-metal-ops.kt` |
| 5 | `ggml-cpu.repack` | `ggmlcpu.Repack` | 4 | 443 | 21 | 464 | `ggml-cpu/repack.h` | `ggmlcpu/Repack.kt` |
| 6 | `llamafile.sgemm` | `ggmlcpu.llamafile.Sgemm` | 0 | 398 | 9 | 407 | `ggml-cpu/llamafile/sgemm.h` | `ggmlcpu/llamafile/Sgemm.kt` |
| 7 | `ggml-sycl.ggml-sycl` | `ggmlsycl.Ggml-sycl` | 3 | 383 | 21 | 404 | `ggml-sycl/ggml-sycl.cpp` | `ggmlsycl/Ggml-sycl.kt` |
| 8 | `ggml-sycl.element_wise` | `ggmlsycl.ElementWise` | 2 | 325 | 1 | 326 | `ggml-sycl/element_wise.hpp` | `ggmlsycl/ElementWise.kt` |
| 9 | `ggml-hexagon.ggml-hexagon` | `ggmlhexagon.Ggml-hexagon` | 1 | 299 | 24 | 323 | `ggml-hexagon/ggml-hexagon.cpp` | `ggmlhexagon/Ggml-hexagon.kt` |
| 10 | `ggml-rpc.ggml-rpc` | `ggmlrpc.Ggml-rpc` | 1 | 252 | 64 | 316 | `ggml-rpc/ggml-rpc.cpp` | `ggmlrpc/Ggml-rpc.kt` |
| 11 | `ggml-metal.ggml-metal-device` | `ggmlmetal.Ggml-metal-device` | 3 | 252 | 17 | 269 | `ggml-metal/ggml-metal-device.h` | `ggmlmetal/Ggml-metal-device.kt` |
| 12 | `ggml-cpu.vec` | `ggmlcpu.Vec` | 1 | 244 | 0 | 244 | `ggml-cpu/vec.h` | `ggmlcpu/Vec.kt` |
| 13 | `ggml-webgpu.ggml-webgpu-shader-lib` | `ggmlwebgpu.Ggml-webgpu-shader-lib` | 1 | 183 | 61 | 244 | `ggml-webgpu/ggml-webgpu-shader-lib.hpp` | `ggmlwebgpu/Ggml-webgpu-shader-lib.kt` |
| 14 | `ggml-cann.aclnn_ops` | `ggmlcann.AclnnOps` | 1 | 230 | 3 | 233 | `ggml-cann/aclnn_ops.h` | `ggmlcann/AclnnOps.kt` |
| 15 | `gguf` | `Gguf` | 1 | 213 | 14 | 227 | `gguf.cpp` | `Gguf.kt` |
| 16 | `ggml-webgpu.ggml-webgpu` | `ggmlwebgpu.Ggml-webgpu` | 1 | 202 | 17 | 219 | `ggml-webgpu/ggml-webgpu.cpp` | `ggmlwebgpu/Ggml-webgpu.kt` |
| 17 | `ggml-cann.ggml-cann` | `ggmlcann.Ggml-cann` | 2 | 201 | 17 | 218 | `ggml-cann/ggml-cann.cpp` | `ggmlcann/Ggml-cann.kt` |
| 18 | `ggml-openvino.ggml-decoder` | `ggmlopenvino.Ggml-decoder` | 1 | 205 | 4 | 209 | `ggml-openvino/ggml-decoder.h` | `ggmlopenvino/Ggml-decoder.kt` |
| 19 | `ggml-sycl.common` | `ggmlsycl.Common` | 0 | 191 | 12 | 203 | `ggml-sycl/common.hpp` | `ggmlsycl/Common.kt` |
| 20 | `ggml-metal.ggml-metal` | `ggmlmetal.Ggml-metal` | 1 | 171 | 7 | 178 | `ggml-metal/ggml-metal.cpp` | `ggmlmetal/Ggml-metal.kt` |
| 21 | `amx.mmq` | `ggmlcpu.amx.Mmq` | 0 | 150 | 12 | 162 | `ggml-cpu/amx/mmq.h` | `ggmlcpu/amx/Mmq.kt` |
| 22 | `ggml-backend-meta` | `Ggml-backend-meta` | 0 | 125 | 23 | 148 | `ggml-backend-meta.cpp` | `Ggml-backend-meta.kt` |
| 23 | `x86.cpu-feats` | `ggmlcpu.arch.x86.Cpu-feats` | 0 | 142 | 1 | 143 | `ggml-cpu/arch/x86/cpu-feats.cpp` | `ggmlcpu/arch/x86/Cpu-feats.kt` |
| 24 | `ggml-sycl.mmq` | `ggmlsycl.Mmq` | 3 | 138 | 0 | 138 | `ggml-sycl/mmq.hpp` | `ggmlsycl/Mmq.kt` |
| 25 | `ggml-cpu.unary-ops` | `ggmlcpu.Unary-ops` | 1 | 133 | 2 | 135 | `ggml-cpu/unary-ops.h` | `ggmlcpu/Unary-ops.kt` |
| 26 | `ggml-openvino.ggml-openvino` | `ggmlopenvino.Ggml-openvino` | 2 | 103 | 8 | 111 | `ggml-openvino/ggml-openvino.cpp` | `ggmlopenvino/Ggml-openvino.kt` |
| 27 | `shared.apir_cs` | `ggmlvirtgpu.backend.shared.ApirCs` | 4 | 104 | 2 | 106 | `ggml-virtgpu/backend/shared/apir_cs.h` | `ggmlvirtgpu/backend/shared/ApirCs.kt` |
| 28 | `kleidiai.kernels` | `ggmlcpu.kleidiai.Kernels` | 1 | 99 | 7 | 106 | `ggml-cpu/kleidiai/kernels.h` | `ggmlcpu/kleidiai/Kernels.kt` |
| 29 | `ggml-sycl.vecdotq` | `ggmlsycl.Vecdotq` | 3 | 101 | 1 | 102 | `ggml-sycl/vecdotq.hpp` | `ggmlsycl/Vecdotq.kt` |
| 30 | `ggml-opt` | `Ggml-opt` | 0 | 84 | 13 | 97 | `ggml-opt.cpp` | `Ggml-opt.kt` |
| 31 | `ggml-sycl.cpy` | `ggmlsycl.Cpy` | 2 | 91 | 0 | 91 | `ggml-sycl/cpy.hpp` | `ggmlsycl/Cpy.kt` |
| 32 | `ggml-webgpu.pre_wgsl` | `ggmlwebgpu.PreWgsl` | 1 | 81 | 8 | 89 | `ggml-webgpu/pre_wgsl.hpp` | `ggmlwebgpu/PreWgsl.kt` |
| 33 | `kleidiai.kleidiai` | `ggmlcpu.kleidiai.Kleidiai` | 1 | 77 | 12 | 89 | `ggml-cpu/kleidiai/kleidiai.h` | `ggmlcpu/kleidiai/Kleidiai.kt` |
| 34 | `ggml-cann.common` | `ggmlcann.Common` | 0 | 75 | 10 | 85 | `ggml-cann/common.h` | `ggmlcann/Common.kt` |
| 35 | `ggml-zdnn.ggml-zdnn` | `ggmlzdnn.Ggml-zdnn` | 1 | 82 | 2 | 84 | `ggml-zdnn/ggml-zdnn.cpp` | `ggmlzdnn/Ggml-zdnn.kt` |
| 36 | `spacemit.ime` | `ggmlcpu.spacemit.Ime` | 1 | 71 | 12 | 83 | `ggml-cpu/spacemit/ime.h` | `ggmlcpu/spacemit/Ime.kt` |
| 37 | `ggml-openvino.ggml-openvino-extra` | `ggmlopenvino.Ggml-openvino-extra` | 4 | 68 | 9 | 77 | `ggml-openvino/ggml-openvino-extra.h` | `ggmlopenvino/Ggml-openvino-extra.kt` |
| 38 | `ggml-sycl.mmvq` | `ggmlsycl.Mmvq` | 1 | 75 | 0 | 75 | `ggml-sycl/mmvq.hpp` | `ggmlsycl/Mmvq.kt` |
| 39 | `ggml-sycl.convert` | `ggmlsycl.Convert` | 17 | 70 | 0 | 70 | `ggml-sycl/convert.hpp` | `ggmlsycl/Convert.kt` |
| 40 | `ggml-hexagon.htp-drv` | `ggmlhexagon.Htp-drv` | 1 | 64 | 3 | 67 | `ggml-hexagon/htp-drv.h` | `ggmlhexagon/Htp-drv.kt` |
| 41 | `ggml-zendnn.ggml-zendnn` | `ggmlzendnn.Ggml-zendnn` | 1 | 56 | 9 | 65 | `ggml-zendnn/ggml-zendnn.cpp` | `ggmlzendnn/Ggml-zendnn.kt` |
| 42 | `ggml-sycl.dequantize` | `ggmlsycl.Dequantize` | 5 | 64 | 0 | 64 | `ggml-sycl/dequantize.hpp` | `ggmlsycl/Dequantize.kt` |
| 43 | `ggml-backend-reg` | `Ggml-backend-reg` | 0 | 60 | 3 | 63 | `ggml-backend-reg.cpp` | `Ggml-backend-reg.kt` |
| 44 | `ggml-blas.ggml-blas` | `ggmlblas.Ggml-blas` | 1 | 49 | 10 | 59 | `ggml-blas/ggml-blas.cpp` | `ggmlblas/Ggml-blas.kt` |
| 45 | `htp.hvx-base` | `ggmlhexagon.htp.Hvx-base` | 11 | 58 | 0 | 58 | `ggml-hexagon/htp/hvx-base.h` | `ggmlhexagon/htp/Hvx-base.kt` |
| 46 | `arm.repack` | `ggmlcpu.arch.arm.Repack` | 0 | 58 | 0 | 58 | `ggml-cpu/arch/arm/repack.cpp` | `ggmlcpu/arch/arm/Repack.kt` |
| 47 | `openvino.decoder` | `ggmlopenvino.openvino.Decoder` | 3 | 56 | 1 | 57 | `ggml-openvino/openvino/decoder.h` | `ggmlopenvino/openvino/Decoder.kt` |
| 48 | `ggml-openvino.ggml-quants` | `ggmlopenvino.Ggml-quants` | 0 | 56 | 1 | 57 | `ggml-openvino/ggml-quants.h` | `ggmlopenvino/Ggml-quants.kt` |
| 49 | `ggml-metal.ggml-metal-common` | `ggmlmetal.Ggml-metal-common` | 1 | 48 | 7 | 55 | `ggml-metal/ggml-metal-common.h` | `ggmlmetal/Ggml-metal-common.kt` |
| 50 | `ggml-sycl.dmmv` | `ggmlsycl.Dmmv` | 1 | 55 | 0 | 55 | `ggml-sycl/dmmv.hpp` | `ggmlsycl/Dmmv.kt` |
| 51 | `vulkan-shaders.vulkan-shaders-gen` | `ggmlvulkan.vulkanshaders.Vulkan-shaders-gen` | 0 | 53 | 2 | 55 | `ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | `ggmlvulkan/vulkanshaders/Vulkan-shaders-gen.kt` |
| 52 | `htp.hvx-copy` | `ggmlhexagon.htp.Hvx-copy` | 1 | 52 | 0 | 52 | `ggml-hexagon/htp/hvx-copy.h` | `ggmlhexagon/htp/Hvx-copy.kt` |
| 53 | `x86.repack` | `ggmlcpu.arch.x86.Repack` | 0 | 52 | 0 | 52 | `ggml-cpu/arch/x86/repack.cpp` | `ggmlcpu/arch/x86/Repack.kt` |
| 54 | `ggml-sycl.fattn-common` | `ggmlsycl.Fattn-common` | 3 | 50 | 0 | 50 | `ggml-sycl/fattn-common.hpp` | `ggmlsycl/Fattn-common.kt` |
| 55 | `htp.hvx-reduce` | `ggmlhexagon.htp.Hvx-reduce` | 1 | 48 | 0 | 48 | `ggml-hexagon/htp/hvx-reduce.h` | `ggmlhexagon/htp/Hvx-reduce.kt` |
| 56 | `ggml-sycl.fattn-tile` | `ggmlsycl.Fattn-tile` | 11 | 47 | 0 | 47 | `ggml-sycl/fattn-tile.hpp` | `ggmlsycl/Fattn-tile.kt` |
| 57 | `ggml-sycl.binbcast` | `ggmlsycl.Binbcast` | 1 | 46 | 1 | 47 | `ggml-sycl/binbcast.hpp` | `ggmlsycl/Binbcast.kt` |
| 58 | `ggml-virtgpu.virtgpu` | `ggmlvirtgpu.Virtgpu` | 3 | 43 | 2 | 45 | `ggml-virtgpu/virtgpu.h` | `ggmlvirtgpu/Virtgpu.kt` |
| 59 | `shared.apir_cs_ggml` | `ggmlvirtgpu.backend.shared.ApirCsGgml` | 2 | 44 | 0 | 44 | `ggml-virtgpu/backend/shared/apir_cs_ggml.h` | `ggmlvirtgpu/backend/shared/ApirCsGgml.kt` |
| 60 | `openvino.node_context` | `ggmlopenvino.openvino.NodeContext` | 19 | 40 | 2 | 42 | `ggml-openvino/openvino/node_context.h` | `ggmlopenvino/openvino/NodeContext.kt` |
| 61 | `amx.amx` | `ggmlcpu.amx.Amx` | 2 | 35 | 6 | 41 | `ggml-cpu/amx/amx.h` | `ggmlcpu/amx/Amx.kt` |
| 62 | `htp.hvx-arith` | `ggmlhexagon.htp.Hvx-arith` | 2 | 41 | 0 | 41 | `ggml-hexagon/htp/hvx-arith.h` | `ggmlhexagon/htp/Hvx-arith.kt` |
| 63 | `htp.hex-dma` | `ggmlhexagon.htp.Hex-dma` | 1 | 39 | 2 | 41 | `ggml-hexagon/htp/hex-dma.h` | `ggmlhexagon/htp/Hex-dma.kt` |
| 64 | `ggml-virtgpu.virtgpu-utils` | `ggmlvirtgpu.Virtgpu-utils` | 3 | 32 | 3 | 35 | `ggml-virtgpu/virtgpu-utils.h` | `ggmlvirtgpu/Virtgpu-utils.kt` |
| 65 | `ggml-sycl.norm` | `ggmlsycl.Norm` | 2 | 35 | 0 | 35 | `ggml-sycl/norm.hpp` | `ggmlsycl/Norm.kt` |
| 66 | `ggml-cpu.binary-ops` | `ggmlcpu.Binary-ops` | 1 | 32 | 2 | 34 | `ggml-cpu/binary-ops.h` | `ggmlcpu/Binary-ops.kt` |
| 67 | `ggml-sycl.rope` | `ggmlsycl.Rope` | 1 | 31 | 2 | 33 | `ggml-sycl/rope.hpp` | `ggmlsycl/Rope.kt` |
| 68 | `ggml-cann.acl_tensor` | `ggmlcann.AclTensor` | 1 | 28 | 1 | 29 | `ggml-cann/acl_tensor.h` | `ggmlcann/AclTensor.kt` |
| 69 | `htp.hex-utils` | `ggmlhexagon.htp.Hex-utils` | 8 | 28 | 0 | 28 | `ggml-hexagon/htp/hex-utils.h` | `ggmlhexagon/htp/Hex-utils.kt` |
| 70 | `riscv.repack` | `ggmlcpu.arch.riscv.Repack` | 0 | 26 | 0 | 26 | `ggml-cpu/arch/riscv/repack.cpp` | `ggmlcpu/arch/riscv/Repack.kt` |
| 71 | `ggml-sycl.upscale` | `ggmlsycl.Upscale` | 1 | 25 | 0 | 25 | `ggml-sycl/upscale.hpp` | `ggmlsycl/Upscale.kt` |
| 72 | `ggml-virtgpu.ggml-backend-device` | `ggmlvirtgpu.Ggml-backend-device` | 0 | 24 | 1 | 25 | `ggml-virtgpu/ggml-backend-device.cpp` | `ggmlvirtgpu/Ggml-backend-device.kt` |
| 73 | `ggml-sycl.softmax` | `ggmlsycl.Softmax` | 3 | 23 | 1 | 24 | `ggml-sycl/softmax.hpp` | `ggmlsycl/Softmax.kt` |
| 74 | `ggml-hexagon.op-desc` | `ggmlhexagon.Op-desc` | 1 | 22 | 2 | 24 | `ggml-hexagon/op-desc.h` | `ggmlhexagon/Op-desc.kt` |
| 75 | `htp.hmx-queue` | `ggmlhexagon.htp.Hmx-queue` | 1 | 21 | 3 | 24 | `ggml-hexagon/htp/hmx-queue.h` | `ggmlhexagon/htp/Hmx-queue.kt` |
| 76 | `htp.htp-ctx` | `ggmlhexagon.htp.Htp-ctx` | 0 | 17 | 7 | 24 | `ggml-hexagon/htp/htp-ctx.h` | `ggmlhexagon/htp/Htp-ctx.kt` |
| 77 | `backend.backend-dispatched.gen` | `ggmlvirtgpu.backend.Backend-dispatched.gen` | 1 | 23 | 0 | 23 | `ggml-virtgpu/backend/backend-dispatched.gen.h` | `ggmlvirtgpu/backend/Backend-dispatched.gen.kt` |
| 78 | `ggml-virtgpu.virtgpu-forward.gen` | `ggmlvirtgpu.Virtgpu-forward.gen` | 1 | 22 | 1 | 23 | `ggml-virtgpu/virtgpu-forward.gen.h` | `ggmlvirtgpu/Virtgpu-forward.gen.kt` |
| 79 | `openvino.translate_session` | `ggmlopenvino.openvino.TranslateSession` | 1 | 22 | 1 | 23 | `ggml-openvino/openvino/translate_session.h` | `ggmlopenvino/openvino/TranslateSession.kt` |
| 80 | `htp.hvx-dump` | `ggmlhexagon.htp.Hvx-dump` | 0 | 22 | 0 | 22 | `ggml-hexagon/htp/hvx-dump.h` | `ggmlhexagon/htp/Hvx-dump.kt` |
| 81 | `ggml-virtgpu.ggml-backend-reg` | `ggmlvirtgpu.Ggml-backend-reg` | 0 | 21 | 0 | 21 | `ggml-virtgpu/ggml-backend-reg.cpp` | `ggmlvirtgpu/Ggml-backend-reg.kt` |
| 82 | `ggml-backend-dl` | `Ggml-backend-dl` | 1 | 19 | 1 | 20 | `ggml-backend-dl.h` | `Ggml-backend-dl.kt` |
| 83 | `ggml-metal.ggml-metal-context` | `ggmlmetal.Ggml-metal-context` | 1 | 16 | 4 | 20 | `ggml-metal/ggml-metal-context.h` | `ggmlmetal/Ggml-metal-context.kt` |
| 84 | `htp.hvx-scale` | `ggmlhexagon.htp.Hvx-scale` | 1 | 20 | 0 | 20 | `ggml-hexagon/htp/hvx-scale.h` | `ggmlhexagon/htp/Hvx-scale.kt` |
| 85 | `backend.backend-dispatched-device` | `ggmlvirtgpu.backend.Backend-dispatched-device` | 0 | 20 | 0 | 20 | `ggml-virtgpu/backend/backend-dispatched-device.cpp` | `ggmlvirtgpu/backend/Backend-dispatched-device.kt` |
| 86 | `ggml-sycl.concat` | `ggmlsycl.Concat` | 9 | 19 | 0 | 19 | `ggml-sycl/concat.hpp` | `ggmlsycl/Concat.kt` |
| 87 | `ggml-virtgpu.virtgpu-forward-device` | `ggmlvirtgpu.Virtgpu-forward-device` | 0 | 18 | 0 | 18 | `ggml-virtgpu/virtgpu-forward-device.cpp` | `ggmlvirtgpu/Virtgpu-forward-device.kt` |
| 88 | `spacemit.ime1_kernels` | `ggmlcpu.spacemit.Ime1Kernels` | 0 | 18 | 0 | 18 | `ggml-cpu/spacemit/ime1_kernels.cpp` | `ggmlcpu/spacemit/Ime1Kernels.kt` |
| 89 | `ggml-hexagon.libdl` | `ggmlhexagon.Libdl` | 1 | 16 | 1 | 17 | `ggml-hexagon/libdl.h` | `ggmlhexagon/Libdl.kt` |
| 90 | `ggml-sycl.gated_delta_net` | `ggmlsycl.GatedDeltaNet` | 1 | 16 | 0 | 16 | `ggml-sycl/gated_delta_net.hpp` | `ggmlsycl/GatedDeltaNet.kt` |
| 91 | `htp.hex-dump` | `ggmlhexagon.htp.Hex-dump` | 1 | 16 | 0 | 16 | `ggml-hexagon/htp/hex-dump.h` | `ggmlhexagon/htp/Hex-dump.kt` |
| 92 | `htp.hvx-div` | `ggmlhexagon.htp.Hvx-div` | 1 | 16 | 0 | 16 | `ggml-hexagon/htp/hvx-div.h` | `ggmlhexagon/htp/Hvx-div.kt` |
| 93 | `htp.hvx-sigmoid` | `ggmlhexagon.htp.Hvx-sigmoid` | 1 | 16 | 0 | 16 | `ggml-hexagon/htp/hvx-sigmoid.h` | `ggmlhexagon/htp/Hvx-sigmoid.kt` |
| 94 | `ggml-virtgpu.ggml-backend-buffer` | `ggmlvirtgpu.Ggml-backend-buffer` | 0 | 16 | 0 | 16 | `ggml-virtgpu/ggml-backend-buffer.cpp` | `ggmlvirtgpu/Ggml-backend-buffer.kt` |
| 95 | `ggml-sycl.set_rows` | `ggmlsycl.SetRows` | 2 | 15 | 0 | 15 | `ggml-sycl/set_rows.hpp` | `ggmlsycl/SetRows.kt` |
| 96 | `ggml-virtgpu.virtgpu-shm` | `ggmlvirtgpu.Virtgpu-shm` | 3 | 12 | 2 | 14 | `ggml-virtgpu/virtgpu-shm.h` | `ggmlvirtgpu/Virtgpu-shm.kt` |
| 97 | `ggml-sycl.wkv` | `ggmlsycl.Wkv` | 1 | 14 | 0 | 14 | `ggml-sycl/wkv.hpp` | `ggmlsycl/Wkv.kt` |
| 98 | `backend.backend-dispatched-buffer` | `ggmlvirtgpu.backend.Backend-dispatched-buffer` | 0 | 14 | 0 | 14 | `ggml-virtgpu/backend/backend-dispatched-buffer.cpp` | `ggmlvirtgpu/backend/Backend-dispatched-buffer.kt` |
| 99 | `ggml-sycl.quantize` | `ggmlsycl.Quantize` | 2 | 10 | 3 | 13 | `ggml-sycl/quantize.hpp` | `ggmlsycl/Quantize.kt` |
| 100 | `ggml-sycl.getrows` | `ggmlsycl.Getrows` | 0 | 13 | 0 | 13 | `ggml-sycl/getrows.hpp` | `ggmlsycl/Getrows.kt` |
| 101 | `ggml-zdnn.utils` | `ggmlzdnn.Utils` | 23 | 12 | 0 | 12 | `ggml-zdnn/utils.hpp` | `ggmlzdnn/Utils.kt` |
| 102 | `ggml-cpu.simd-gemm` | `ggmlcpu.Simd-gemm` | 1 | 12 | 0 | 12 | `ggml-cpu/simd-gemm.h` | `ggmlcpu/Simd-gemm.kt` |
| 103 | `ggml-sycl.im2col` | `ggmlsycl.Im2col` | 1 | 12 | 0 | 12 | `ggml-sycl/im2col.hpp` | `ggmlsycl/Im2col.kt` |
| 104 | `htp.hvx-sqrt` | `ggmlhexagon.htp.Hvx-sqrt` | 1 | 12 | 0 | 12 | `ggml-hexagon/htp/hvx-sqrt.h` | `ggmlhexagon/htp/Hvx-sqrt.kt` |
| 105 | `backend.apir_cs_ggml-rpc-back` | `ggmlvirtgpu.backend.ApirCsGgml-rpc-back` | 0 | 12 | 0 | 12 | `ggml-virtgpu/backend/apir_cs_ggml-rpc-back.cpp` | `ggmlvirtgpu/backend/ApirCsGgml-rpc-back.kt` |
| 106 | `backend.backend-dispatched-buffer-type` | `ggmlvirtgpu.backend.Backend-dispatched-buffer-type` | 0 | 12 | 0 | 12 | `ggml-virtgpu/backend/backend-dispatched-buffer-type.cpp` | `ggmlvirtgpu/backend/Backend-dispatched-buffer-type.kt` |
| 107 | `ggml-virtgpu.virtgpu-forward-buffer` | `ggmlvirtgpu.Virtgpu-forward-buffer` | 0 | 12 | 0 | 12 | `ggml-virtgpu/virtgpu-forward-buffer.cpp` | `ggmlvirtgpu/Virtgpu-forward-buffer.kt` |
| 108 | `htp.htp-ops` | `ggmlhexagon.htp.Htp-ops` | 3 | 0 | 11 | 11 | `ggml-hexagon/htp/htp-ops.h` | `ggmlhexagon/htp/Htp-ops.kt` |
| 109 | `ggml-sycl.fattn` | `ggmlsycl.Fattn` | 1 | 10 | 1 | 11 | `ggml-sycl/fattn.hpp` | `ggmlsycl/Fattn.kt` |
| 110 | `ggml-sycl.pad` | `ggmlsycl.Pad` | 1 | 11 | 0 | 11 | `ggml-sycl/pad.hpp` | `ggmlsycl/Pad.kt` |
| 111 | `amx.common` | `ggmlcpu.amx.Common` | 0 | 10 | 1 | 11 | `ggml-cpu/amx/common.h` | `ggmlcpu/amx/Common.kt` |
| 112 | `ggml-sycl.fattn-vec` | `ggmlsycl.Fattn-vec` | 37 | 10 | 0 | 10 | `ggml-sycl/fattn-vec.hpp` | `ggmlsycl/Fattn-vec.kt` |
| 113 | `htp.hvx-inverse` | `ggmlhexagon.htp.Hvx-inverse` | 3 | 10 | 0 | 10 | `ggml-hexagon/htp/hvx-inverse.h` | `ggmlhexagon/htp/Hvx-inverse.kt` |
| 114 | `shared.apir_cs_rpc` | `ggmlvirtgpu.backend.shared.ApirCsRpc` | 3 | 9 | 1 | 10 | `ggml-virtgpu/backend/shared/apir_cs_rpc.h` | `ggmlvirtgpu/backend/shared/ApirCsRpc.kt` |
| 115 | `ggml-cpu.hbm` | `ggmlcpu.Hbm` | 1 | 9 | 1 | 10 | `ggml-cpu/hbm.h` | `ggmlcpu/Hbm.kt` |
| 116 | `ggml-virtgpu.ggml-backend-buffer-type` | `ggmlvirtgpu.Ggml-backend-buffer-type` | 0 | 10 | 0 | 10 | `ggml-virtgpu/ggml-backend-buffer-type.cpp` | `ggmlvirtgpu/Ggml-backend-buffer-type.kt` |
| 117 | `ggml-virtgpu.virtgpu-forward-buffer-type` | `ggmlvirtgpu.Virtgpu-forward-buffer-type` | 0 | 10 | 0 | 10 | `ggml-virtgpu/virtgpu-forward-buffer-type.cpp` | `ggmlvirtgpu/Virtgpu-forward-buffer-type.kt` |
| 118 | `ggml-virtgpu.ggml-remoting` | `ggmlvirtgpu.Ggml-remoting` | 8 | 7 | 2 | 9 | `ggml-virtgpu/ggml-remoting.h` | `ggmlvirtgpu/Ggml-remoting.kt` |
| 119 | `openvino.input_model` | `ggmlopenvino.openvino.InputModel` | 3 | 6 | 3 | 9 | `ggml-openvino/openvino/input_model.h` | `ggmlopenvino/openvino/InputModel.kt` |
| 120 | `shared.api_remoting` | `ggmlvirtgpu.backend.shared.ApiRemoting` | 3 | 6 | 3 | 9 | `ggml-virtgpu/backend/shared/api_remoting.h` | `ggmlvirtgpu/backend/shared/ApiRemoting.kt` |
| 121 | `htp.worker-pool` | `ggmlhexagon.htp.Worker-pool` | 1 | 9 | 0 | 9 | `ggml-hexagon/htp/worker-pool.h` | `ggmlhexagon/htp/Worker-pool.kt` |
| 122 | `ggml-sycl.ssm_conv` | `ggmlsycl.SsmConv` | 2 | 8 | 0 | 8 | `ggml-sycl/ssm_conv.hpp` | `ggmlsycl/SsmConv.kt` |
| 123 | `openvino.frontend` | `ggmlopenvino.openvino.Frontend` | 1 | 7 | 1 | 8 | `ggml-openvino/openvino/frontend.h` | `ggmlopenvino/openvino/Frontend.kt` |
| 124 | `vendors.hip` | `ggmlcuda.vendors.Hip` | 0 | 8 | 0 | 8 | `ggml-cuda/vendors/hip.h` | `ggmlcuda/vendors/Hip.kt` |
| 125 | `ggml-sycl.conv` | `ggmlsycl.Conv` | 1 | 7 | 0 | 7 | `ggml-sycl/conv.hpp` | `ggmlsycl/Conv.kt` |
| 126 | `ggml-sycl.gemm` | `ggmlsycl.Gemm` | 1 | 6 | 1 | 7 | `ggml-sycl/gemm.hpp` | `ggmlsycl/Gemm.kt` |
| 127 | `ggml-sycl.roll` | `ggmlsycl.Roll` | 1 | 7 | 0 | 7 | `ggml-sycl/roll.hpp` | `ggmlsycl/Roll.kt` |
| 128 | `ggml-sycl.tsembd` | `ggmlsycl.Tsembd` | 1 | 7 | 0 | 7 | `ggml-sycl/tsembd.hpp` | `ggmlsycl/Tsembd.kt` |
| 129 | `htp.hex-fastdiv` | `ggmlhexagon.htp.Hex-fastdiv` | 1 | 6 | 1 | 7 | `ggml-hexagon/htp/hex-fastdiv.h` | `ggmlhexagon/htp/Hex-fastdiv.kt` |
| 130 | `backend.backend-virgl-apir` | `ggmlvirtgpu.backend.Backend-virgl-apir` | 6 | 5 | 1 | 6 | `ggml-virtgpu/backend/backend-virgl-apir.h` | `ggmlvirtgpu/backend/Backend-virgl-apir.kt` |
| 131 | `backend.backend-dispatched` | `ggmlvirtgpu.backend.Backend-dispatched` | 5 | 5 | 1 | 6 | `ggml-virtgpu/backend/backend-dispatched.h` | `ggmlvirtgpu/backend/Backend-dispatched.kt` |
| 132 | `htp.hvx-exp` | `ggmlhexagon.htp.Hvx-exp` | 1 | 6 | 0 | 6 | `ggml-hexagon/htp/hvx-exp.h` | `ggmlhexagon/htp/Hvx-exp.kt` |
| 133 | `ggml-virtgpu.apir_cs_ggml-rpc-front` | `ggmlvirtgpu.ApirCsGgml-rpc-front` | 0 | 6 | 0 | 6 | `ggml-virtgpu/apir_cs_ggml-rpc-front.cpp` | `ggmlvirtgpu/ApirCsGgml-rpc-front.kt` |
| 134 | `riscv.cpu-feats` | `ggmlcpu.arch.riscv.Cpu-feats` | 0 | 4 | 2 | 6 | `ggml-cpu/arch/riscv/cpu-feats.cpp` | `ggmlcpu/arch/riscv/Cpu-feats.kt` |
| 135 | `ggml-sycl.set` | `ggmlsycl.Set` | 6 | 5 | 0 | 5 | `ggml-sycl/set.hpp` | `ggmlsycl/Set.kt` |
| 136 | `ggml-sycl.add-id` | `ggmlsycl.Add-id` | 1 | 5 | 0 | 5 | `ggml-sycl/add-id.hpp` | `ggmlsycl/Add-id.kt` |
| 137 | `ggml-sycl.count-equal` | `ggmlsycl.Count-equal` | 1 | 5 | 0 | 5 | `ggml-sycl/count-equal.hpp` | `ggmlsycl/Count-equal.kt` |
| 138 | `ggml-sycl.gla` | `ggmlsycl.Gla` | 1 | 5 | 0 | 5 | `ggml-sycl/gla.hpp` | `ggmlsycl/Gla.kt` |
| 139 | `ggml-sycl.pad_reflect_1d` | `ggmlsycl.PadReflect1d` | 1 | 5 | 0 | 5 | `ggml-sycl/pad_reflect_1d.hpp` | `ggmlsycl/PadReflect1d.kt` |
| 140 | `arm.cpu-feats` | `ggmlcpu.arch.arm.Cpu-feats` | 0 | 4 | 1 | 5 | `ggml-cpu/arch/arm/cpu-feats.cpp` | `ggmlcpu/arch/arm/Cpu-feats.kt` |
| 141 | `powerpc.cpu-feats` | `ggmlcpu.arch.powerpc.Cpu-feats` | 0 | 4 | 1 | 5 | `ggml-cpu/arch/powerpc/cpu-feats.cpp` | `ggmlcpu/arch/powerpc/Cpu-feats.kt` |
| 142 | `s390.cpu-feats` | `ggmlcpu.arch.s390.Cpu-feats` | 0 | 4 | 1 | 5 | `ggml-cpu/arch/s390/cpu-feats.cpp` | `ggmlcpu/arch/s390/Cpu-feats.kt` |
| 143 | `backend.backend-convert` | `ggmlvirtgpu.backend.Backend-convert` | 1 | 4 | 0 | 4 | `ggml-virtgpu/backend/backend-convert.h` | `ggmlvirtgpu/backend/Backend-convert.kt` |
| 144 | `ggml-zdnn.mmf` | `ggmlzdnn.Mmf` | 1 | 3 | 1 | 4 | `ggml-zdnn/mmf.hpp` | `ggmlzdnn/Mmf.kt` |
| 145 | `htp.hvx-floor` | `ggmlhexagon.htp.Hvx-floor` | 1 | 4 | 0 | 4 | `ggml-hexagon/htp/hvx-floor.h` | `ggmlhexagon/htp/Hvx-floor.kt` |
| 146 | `pass.eliminate_zp` | `ggmlopenvino.openvino.pass.EliminateZp` | 1 | 4 | 0 | 4 | `ggml-openvino/openvino/pass/eliminate_zp.h` | `ggmlopenvino/openvino/pass/EliminateZp.kt` |
| 147 | `backend.backend-dispatched-backend` | `ggmlvirtgpu.backend.Backend-dispatched-backend` | 0 | 4 | 0 | 4 | `ggml-virtgpu/backend/backend-dispatched-backend.cpp` | `ggmlvirtgpu/backend/Backend-dispatched-backend.kt` |
| 148 | `ggml-virtgpu.virtgpu-forward-backend` | `ggmlvirtgpu.Virtgpu-forward-backend` | 0 | 4 | 0 | 4 | `ggml-virtgpu/virtgpu-forward-backend.cpp` | `ggmlvirtgpu/Virtgpu-forward-backend.kt` |
| 149 | `ggml-zdnn.common` | `ggmlzdnn.Common` | 0 | 0 | 4 | 4 | `ggml-zdnn/common.hpp` | `ggmlzdnn/Common.kt` |
| 150 | `htp.hmx-ops` | `ggmlhexagon.htp.Hmx-ops` | 0 | 3 | 1 | 4 | `ggml-hexagon/htp/hmx-ops.h` | `ggmlhexagon/htp/Hmx-ops.kt` |
| 151 | `htp.hmx-utils` | `ggmlhexagon.htp.Hmx-utils` | 0 | 4 | 0 | 4 | `ggml-hexagon/htp/hmx-utils.h` | `ggmlhexagon/htp/Hmx-utils.kt` |
| 152 | `openvino.op_table` | `ggmlopenvino.openvino.OpTable` | 18 | 3 | 0 | 3 | `ggml-openvino/openvino/op_table.h` | `ggmlopenvino/openvino/OpTable.kt` |
| 153 | `spacemit.ime_kernels` | `ggmlcpu.spacemit.ImeKernels` | 2 | 3 | 0 | 3 | `ggml-cpu/spacemit/ime_kernels.h` | `ggmlcpu/spacemit/ImeKernels.kt` |
| 154 | `ggml-sycl.outprod` | `ggmlsycl.Outprod` | 1 | 3 | 0 | 3 | `ggml-sycl/outprod.hpp` | `ggmlsycl/Outprod.kt` |
| 155 | `ggml-sycl.repeat_back` | `ggmlsycl.RepeatBack` | 1 | 3 | 0 | 3 | `ggml-sycl/repeat_back.hpp` | `ggmlsycl/RepeatBack.kt` |
| 156 | `shared.apir_backend.gen` | `ggmlvirtgpu.backend.shared.ApirBackend.gen` | 1 | 2 | 1 | 3 | `ggml-virtgpu/backend/shared/apir_backend.gen.h` | `ggmlvirtgpu/backend/shared/ApirBackend.gen.kt` |
| 157 | `op.reshape` | `ggmlopenvino.openvino.op.Reshape` | 9 | 2 | 0 | 2 | `ggml-openvino/openvino/op/reshape.cpp` | `ggmlopenvino/openvino/op/Reshape.kt` |
| 158 | `op.transpose` | `ggmlopenvino.openvino.op.Transpose` | 7 | 2 | 0 | 2 | `ggml-openvino/openvino/op/transpose.cpp` | `ggmlopenvino/openvino/op/Transpose.kt` |
| 159 | `shared.apir_backend` | `ggmlvirtgpu.backend.shared.ApirBackend` | 6 | 2 | 0 | 2 | `ggml-virtgpu/backend/shared/apir_backend.h` | `ggmlvirtgpu/backend/shared/ApirBackend.kt` |
| 160 | `op.get_rows` | `ggmlopenvino.openvino.op.GetRows` | 2 | 2 | 0 | 2 | `ggml-openvino/openvino/op/get_rows.cpp` | `ggmlopenvino/openvino/op/GetRows.kt` |
| 161 | `ggml-virtgpu.virtgpu-apir` | `ggmlvirtgpu.Virtgpu-apir` | 1 | 0 | 2 | 2 | `ggml-virtgpu/virtgpu-apir.h` | `ggmlvirtgpu/Virtgpu-apir.kt` |
| 162 | `pass.mark_decompression_convert_constant_folding` | `ggmlopenvino.openvino.pass.MarkDecompressionConvertConstantFolding` | 1 | 1 | 1 | 2 | `ggml-openvino/openvino/pass/mark_decompression_convert_constant_folding.h` | `ggmlopenvino/openvino/pass/MarkDecompressionConvertConstantFolding.kt` |
| 163 | `pass.squeeze_matmul` | `ggmlopenvino.openvino.pass.SqueezeMatmul` | 1 | 2 | 0 | 2 | `ggml-openvino/openvino/pass/squeeze_matmul.h` | `ggmlopenvino/openvino/pass/SqueezeMatmul.kt` |
| 164 | `op.cont` | `ggmlopenvino.openvino.op.Cont` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/cont.cpp` | `ggmlopenvino/openvino/op/Cont.kt` |
| 165 | `op.cpy` | `ggmlopenvino.openvino.op.Cpy` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/cpy.cpp` | `ggmlopenvino/openvino/op/Cpy.kt` |
| 166 | `op.flash_attn_ext` | `ggmlopenvino.openvino.op.FlashAttnExt` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/flash_attn_ext.cpp` | `ggmlopenvino/openvino/op/FlashAttnExt.kt` |
| 167 | `op.glu_geglu` | `ggmlopenvino.openvino.op.GluGeglu` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/glu_geglu.cpp` | `ggmlopenvino/openvino/op/GluGeglu.kt` |
| 168 | `op.glu_swiglu` | `ggmlopenvino.openvino.op.GluSwiglu` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/glu_swiglu.cpp` | `ggmlopenvino/openvino/op/GluSwiglu.kt` |
| 169 | `op.mulmat` | `ggmlopenvino.openvino.op.Mulmat` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/mulmat.cpp` | `ggmlopenvino/openvino/op/Mulmat.kt` |
| 170 | `op.permute` | `ggmlopenvino.openvino.op.Permute` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/permute.cpp` | `ggmlopenvino/openvino/op/Permute.kt` |
| 171 | `op.rms_norm` | `ggmlopenvino.openvino.op.RmsNorm` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/rms_norm.cpp` | `ggmlopenvino/openvino/op/RmsNorm.kt` |
| 172 | `op.rope` | `ggmlopenvino.openvino.op.Rope` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/rope.cpp` | `ggmlopenvino/openvino/op/Rope.kt` |
| 173 | `op.scale` | `ggmlopenvino.openvino.op.Scale` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/scale.cpp` | `ggmlopenvino/openvino/op/Scale.kt` |
| 174 | `op.set_rows` | `ggmlopenvino.openvino.op.SetRows` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/set_rows.cpp` | `ggmlopenvino/openvino/op/SetRows.kt` |
| 175 | `op.softmax` | `ggmlopenvino.openvino.op.Softmax` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/softmax.cpp` | `ggmlopenvino/openvino/op/Softmax.kt` |
| 176 | `op.unary_silu` | `ggmlopenvino.openvino.op.UnarySilu` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/unary_silu.cpp` | `ggmlopenvino/openvino/op/UnarySilu.kt` |
| 177 | `op.view` | `ggmlopenvino.openvino.op.View` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/op/view.cpp` | `ggmlopenvino/openvino/op/View.kt` |
| 178 | `pass.fuse_to_sdpa` | `ggmlopenvino.openvino.pass.FuseToSdpa` | 0 | 2 | 0 | 2 | `ggml-openvino/openvino/pass/fuse_to_sdpa.h` | `ggmlopenvino/openvino/pass/FuseToSdpa.kt` |
| 179 | `include.apir_hw` | `ggmlvirtgpu.include.ApirHw` | 1 | 0 | 1 | 1 | `ggml-virtgpu/include/apir_hw.h` | `ggmlvirtgpu/include/ApirHw.kt` |
| 180 | `ggml-sycl.presets` | `ggmlsycl.Presets` | 8 | 0 | 0 | 0 | `ggml-sycl/presets.hpp` | `ggmlsycl/Presets.kt` |
| 181 | `ggml-virtgpu.virtgpu-forward-impl` | `ggmlvirtgpu.Virtgpu-forward-impl` | 4 | 0 | 0 | 0 | `ggml-virtgpu/virtgpu-forward-impl.h` | `ggmlvirtgpu/Virtgpu-forward-impl.kt` |
| 182 | `htp.hvx-types` | `ggmlhexagon.htp.Hvx-types` | 4 | 0 | 0 | 0 | `ggml-hexagon/htp/hvx-types.h` | `ggmlhexagon/htp/Hvx-types.kt` |
| 183 | `ggml-metal.ggml-metal-impl` | `ggmlmetal.Ggml-metal-impl` | 2 | 0 | 0 | 0 | `ggml-metal/ggml-metal-impl.h` | `ggmlmetal/Ggml-metal-impl.kt` |
| 184 | `ggml-sycl.sycl_hw` | `ggmlsycl.SyclHw` | 2 | 0 | 0 | 0 | `ggml-sycl/sycl_hw.hpp` | `ggmlsycl/SyclHw.kt` |
| 185 | `ggml-cpu.arch-fallback` | `ggmlcpu.Arch-fallback` | 1 | 0 | 0 | 0 | `ggml-cpu/arch-fallback.h` | `ggmlcpu/Arch-fallback.kt` |
| 186 | `htp.hmx-profile` | `ggmlhexagon.htp.Hmx-profile` | 0 | 0 | 0 | 0 | `ggml-hexagon/htp/hmx-profile.h` | `ggmlhexagon/htp/Hmx-profile.kt` |
| 187 | `htp.hvx-utils` | `ggmlhexagon.htp.Hvx-utils` | 0 | 0 | 0 | 0 | `ggml-hexagon/htp/hvx-utils.h` | `ggmlhexagon/htp/Hvx-utils.kt` |
| 188 | `template-instances.fattn-tile-instance-dkq112-dv112` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq112-dv112` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq112-dv112.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq112-dv112.kt` |
| 189 | `template-instances.fattn-tile-instance-dkq128-dv128` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq128-dv128` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq128-dv128.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq128-dv128.kt` |
| 190 | `template-instances.fattn-tile-instance-dkq256-dv256` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq256-dv256` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq256-dv256.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq256-dv256.kt` |
| 191 | `template-instances.fattn-tile-instance-dkq40-dv40` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq40-dv40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq40-dv40.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq40-dv40.kt` |
| 192 | `template-instances.fattn-tile-instance-dkq512-dv512` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq512-dv512` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq512-dv512.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq512-dv512.kt` |
| 193 | `template-instances.fattn-tile-instance-dkq576-dv512` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq576-dv512` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq576-dv512.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq576-dv512.kt` |
| 194 | `template-instances.fattn-tile-instance-dkq64-dv64` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq64-dv64` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq64-dv64.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq64-dv64.kt` |
| 195 | `template-instances.fattn-tile-instance-dkq72-dv72` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq72-dv72` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq72-dv72.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq72-dv72.kt` |
| 196 | `template-instances.fattn-tile-instance-dkq80-dv80` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq80-dv80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq80-dv80.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq80-dv80.kt` |
| 197 | `template-instances.fattn-tile-instance-dkq96-dv96` | `ggmlsycl.templateinstances.Fattn-tile-instance-dkq96-dv96` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-tile-instance-dkq96-dv96.cpp` | `ggmlsycl/templateinstances/Fattn-tile-instance-dkq96-dv96.kt` |
| 198 | `template-instances.fattn-vec-instance-f16-f16` | `ggmlsycl.templateinstances.Fattn-vec-instance-f16-f16` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-f16-f16.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-f16-f16.kt` |
| 199 | `template-instances.fattn-vec-instance-f16-q4_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-f16-q40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-f16-q4_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-f16-q40.kt` |
| 200 | `template-instances.fattn-vec-instance-f16-q4_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-f16-q41` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-f16-q4_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-f16-q41.kt` |
| 201 | `template-instances.fattn-vec-instance-f16-q5_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-f16-q50` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-f16-q5_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-f16-q50.kt` |
| 202 | `template-instances.fattn-vec-instance-f16-q5_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-f16-q51` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-f16-q5_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-f16-q51.kt` |
| 203 | `template-instances.fattn-vec-instance-f16-q8_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-f16-q80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-f16-q8_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-f16-q80.kt` |
| 204 | `template-instances.fattn-vec-instance-q4_0-f16` | `ggmlsycl.templateinstances.Fattn-vec-instance-q40-f16` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_0-f16.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q40-f16.kt` |
| 205 | `template-instances.fattn-vec-instance-q4_0-q4_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q40-q40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_0-q4_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q40-q40.kt` |
| 206 | `template-instances.fattn-vec-instance-q4_0-q4_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q40-q41` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_0-q4_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q40-q41.kt` |
| 207 | `template-instances.fattn-vec-instance-q4_0-q5_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q40-q50` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_0-q5_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q40-q50.kt` |
| 208 | `template-instances.fattn-vec-instance-q4_0-q5_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q40-q51` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_0-q5_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q40-q51.kt` |
| 209 | `template-instances.fattn-vec-instance-q4_0-q8_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q40-q80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_0-q8_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q40-q80.kt` |
| 210 | `template-instances.fattn-vec-instance-q4_1-f16` | `ggmlsycl.templateinstances.Fattn-vec-instance-q41-f16` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_1-f16.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q41-f16.kt` |
| 211 | `template-instances.fattn-vec-instance-q4_1-q4_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q41-q40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_1-q4_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q41-q40.kt` |
| 212 | `template-instances.fattn-vec-instance-q4_1-q4_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q41-q41` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_1-q4_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q41-q41.kt` |
| 213 | `template-instances.fattn-vec-instance-q4_1-q5_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q41-q50` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_1-q5_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q41-q50.kt` |
| 214 | `template-instances.fattn-vec-instance-q4_1-q5_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q41-q51` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_1-q5_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q41-q51.kt` |
| 215 | `template-instances.fattn-vec-instance-q4_1-q8_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q41-q80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q4_1-q8_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q41-q80.kt` |
| 216 | `template-instances.fattn-vec-instance-q5_0-f16` | `ggmlsycl.templateinstances.Fattn-vec-instance-q50-f16` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_0-f16.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q50-f16.kt` |
| 217 | `template-instances.fattn-vec-instance-q5_0-q4_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q50-q40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_0-q4_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q50-q40.kt` |
| 218 | `template-instances.fattn-vec-instance-q5_0-q4_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q50-q41` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_0-q4_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q50-q41.kt` |
| 219 | `template-instances.fattn-vec-instance-q5_0-q5_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q50-q50` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_0-q5_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q50-q50.kt` |
| 220 | `template-instances.fattn-vec-instance-q5_0-q5_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q50-q51` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_0-q5_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q50-q51.kt` |
| 221 | `template-instances.fattn-vec-instance-q5_0-q8_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q50-q80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_0-q8_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q50-q80.kt` |
| 222 | `template-instances.fattn-vec-instance-q5_1-f16` | `ggmlsycl.templateinstances.Fattn-vec-instance-q51-f16` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_1-f16.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q51-f16.kt` |
| 223 | `template-instances.fattn-vec-instance-q5_1-q4_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q51-q40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_1-q4_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q51-q40.kt` |
| 224 | `template-instances.fattn-vec-instance-q5_1-q4_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q51-q41` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_1-q4_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q51-q41.kt` |
| 225 | `template-instances.fattn-vec-instance-q5_1-q5_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q51-q50` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_1-q5_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q51-q50.kt` |
| 226 | `template-instances.fattn-vec-instance-q5_1-q5_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q51-q51` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_1-q5_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q51-q51.kt` |
| 227 | `template-instances.fattn-vec-instance-q5_1-q8_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q51-q80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q5_1-q8_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q51-q80.kt` |
| 228 | `template-instances.fattn-vec-instance-q8_0-f16` | `ggmlsycl.templateinstances.Fattn-vec-instance-q80-f16` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q8_0-f16.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q80-f16.kt` |
| 229 | `template-instances.fattn-vec-instance-q8_0-q4_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q80-q40` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q8_0-q4_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q80-q40.kt` |
| 230 | `template-instances.fattn-vec-instance-q8_0-q4_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q80-q41` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q8_0-q4_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q80-q41.kt` |
| 231 | `template-instances.fattn-vec-instance-q8_0-q5_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q80-q50` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q8_0-q5_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q80-q50.kt` |
| 232 | `template-instances.fattn-vec-instance-q8_0-q5_1` | `ggmlsycl.templateinstances.Fattn-vec-instance-q80-q51` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q8_0-q5_1.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q80-q51.kt` |
| 233 | `template-instances.fattn-vec-instance-q8_0-q8_0` | `ggmlsycl.templateinstances.Fattn-vec-instance-q80-q80` | 0 | 0 | 0 | 0 | `ggml-sycl/template-instances/fattn-vec-instance-q8_0-q8_0.cpp` | `ggmlsycl/templateinstances/Fattn-vec-instance-q80-q80.kt` |
| 234 | `vendors.cuda` | `ggmlcuda.vendors.Cuda` | 0 | 0 | 0 | 0 | `ggml-cuda/vendors/cuda.h` | `ggmlcuda/vendors/Cuda.kt` |
| 235 | `vendors.musa` | `ggmlcuda.vendors.Musa` | 0 | 0 | 0 | 0 | `ggml-cuda/vendors/musa.h` | `ggmlcuda/vendors/Musa.kt` |

## Documentation Gaps

**Documentation coverage:** 3571 / 55 lines (6493%)

Documentation gaps (>20%), complete list:

No significant documentation gaps found.

