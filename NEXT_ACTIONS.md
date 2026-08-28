# Immediate Actions - High-Value Files

Based on AST analysis, here are the concrete next steps.

## Summary

- **Current Progress:** 8.2% (47/256 files)
- **Matched Files:** 21
- **Average Similarity:** 0.00
- **Critical Issues:** 21 files with <0.60 function similarity

## Priority 1: Fix Incomplete High-Dependency Files

### 1. ggml-impl
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 58
- **Priority Score:** 58106312.0
- **Functions:** 47/51 matched (target 77)
- **Missing functions:** `ggml_aligned_malloc`, `ggml_aligned_free`, `gguf_type_size`, `gguf_write_to_buf`
- **Types:** 6/12 matched (target 8)
- **Missing types:** `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order`, `ggml_cgraph`, `gguf_type`, `gguf_context`
- **Symbol Deficit:** 10 (functions: 4, types: 6)
- **Action:** Deep review - likely missing major functionality

### 2. ggml
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 50
- **Priority Score:** 50010112.0
- **Functions:** 0/1 matched (target 431)
- **Missing functions:** `ggml_uncaught_exception`
- **Types:** 0/0 matched (target 23)
- **Missing types:** _none_
- **Symbol Deficit:** 1 (functions: 1, types: 0)
- **Action:** Deep review - likely missing major functionality

### 3. ggml-backend-impl
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 47
- **Priority Score:** 47172608.0
- **Functions:** 7/8 matched (target 39)
- **Missing functions:** `ggml_backend_multi_buffer_set_usage`
- **Types:** 2/18 matched (target 17)
- **Missing types:** `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer`, `ggml_backend_buffer_usage`, `ggml_context`, `ggml_backend_i`, `ggml_cgraph`, `ggml_backend`, `ggml_backend_event`, `ggml_backend_device`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg`
- **Symbol Deficit:** 17 (functions: 1, types: 16)
- **Action:** Deep review - likely missing major functionality

### 4. ggml-cpu.common
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 47
- **Priority Score:** 47021208.0
- **Functions:** 8/8 matched (target 20)
- **Missing functions:** _none_
- **Types:** 2/4 matched (target 6)
- **Missing types:** `ggml_compute_params`, `ggml_tensor`
- **Symbol Deficit:** 2 (functions: 0, types: 2)
- **Action:** Deep review - likely missing major functionality

### 5. ggml-backend
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 20
- **Priority Score:** 20174710.0
- **Functions:** 128/131 matched (target 178)
- **Missing functions:** `ggml_backend_buffer_init`, `ggml_backend_buffer_copy_tensor`, `ggml_backend_buffer_is_multi_buffer`
- **Types:** 3/16 matched (target 7)
- **Missing types:** `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer_usage`, `ggml_cgraph`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_context`, `ggml_op`, `ggml_backend_sched`, `ggml_hash_set`, `ggml_init_params`, `ggml_backend_buffer_type`
- **Symbol Deficit:** 16 (functions: 3, types: 13)
- **Action:** Deep review - likely missing major functionality

### 6. ggml-common
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 16
- **Priority Score:** 16020210.0
- **Functions:** 0/2 matched (target 53)
- **Missing functions:** `union`, `GGML_TABLE_BEGIN`
- **Types:** 0/0 matched (target 28)
- **Missing types:** _none_
- **Symbol Deficit:** 2 (functions: 2, types: 0)
- **Action:** Deep review - likely missing major functionality

### 7. ggml-cpu.ggml-cpu
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 12
- **Priority Score:** 12154610.0
- **Functions:** 30/33 matched (target 159)
- **Missing functions:** `ggml_backend_cpu_device_get_extra_buffers_type`, `ggml_backend_cpu_device_context`, `ggml_backend_cpu_reg`
- **Types:** 1/13 matched (target 14)
- **Missing types:** `ggml_backend_cpu_context`, `ggml_backend_plan_cpu`, `ggml_cgraph`, `ggml_status`, `ggml_backend_i`, `ggml_backend_cpu_device_context`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_tensor`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg`
- **Symbol Deficit:** 15 (functions: 3, types: 12)
- **Action:** Deep review - likely missing major functionality

### 8. ggml-cpu.ggml-cpu-impl
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 11
- **Priority Score:** 11063910.0
- **Functions:** 31/32 matched (target 39)
- **Missing functions:** `ggml_vec_dot`
- **Types:** 2/7 matched (target 2)
- **Missing types:** `ggml_int16x8x2_t`, `ggml_uint8x16x2_t`, `ggml_uint8x16x4_t`, `ggml_int8x16x2_t`, `ggml_int8x16x4_t`
- **Symbol Deficit:** 6 (functions: 1, types: 5)
- **Action:** Deep review - likely missing major functionality

### 9. ggml-cpu.simd-mappings
- **Similarity:** 0.00 (needs 85% improvement)
- **Dependencies:** 10
- **Priority Score:** 10002010.0
- **Functions:** 20/20 matched (target 34)
- **Missing functions:** _none_
- **Types:** 0/0 matched (target 1)
- **Missing types:** _none_
- **Action:** Deep review - likely missing major functionality

## Priority 2: Port Missing High-Value Files

Critical missing files (>10 dependencies):

1. **ggml-sycl.fattn-vec** (37 deps)
   - Path: `ggml-sycl/fattn-vec.hpp`
   - Essential for 37 other files

2. **ggml-zdnn.utils** (23 deps)
   - Path: `ggml-zdnn/utils.hpp`
   - Essential for 23 other files

3. **openvino.node_context** (19 deps)
   - Path: `ggml-openvino/openvino/node_context.h`
   - Essential for 19 other files

4. **openvino.op_table** (18 deps)
   - Path: `ggml-openvino/openvino/op_table.h`
   - Essential for 18 other files

5. **ggml-sycl.convert** (17 deps)
   - Path: `ggml-sycl/convert.hpp`
   - Essential for 17 other files

6. **htp.hvx-base** (11 deps)
   - Path: `ggml-hexagon/htp/hvx-base.h`
   - Essential for 11 other files

7. **ggml-sycl.fattn-tile** (11 deps)
   - Path: `ggml-sycl/fattn-tile.hpp`
   - Essential for 11 other files

## Detailed Work Items

Every matched file is listed below with function and type symbol parity.

### 1. ggml-impl

- **Target:** `core.NumericConversions [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 58
- **Priority Score:** 58106312.0
- **Functions:** 47/51 matched (target 77)
- **Missing functions:** `ggml_aligned_malloc`, `ggml_aligned_free`, `gguf_type_size`, `gguf_write_to_buf`
- **Types:** 6/12 matched (target 8)
- **Missing types:** `ggml_tensor`, `ggml_op`, `ggml_cgraph_eval_order`, `ggml_cgraph`, `gguf_type`, `gguf_context`
- **Lint issues:** 8

### 2. ggml

- **Target:** `core.GGMLOps [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 50
- **Priority Score:** 50010112.0
- **Functions:** 0/1 matched (target 431)
- **Missing functions:** `ggml_uncaught_exception`
- **Types:** 0/0 matched (target 23)
- **Missing types:** _none_
- **Lint issues:** 90

### 3. ggml-backend-impl

- **Target:** `core.GGMLBackendImpl [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 47
- **Priority Score:** 47172608.0
- **Functions:** 7/8 matched (target 39)
- **Missing functions:** `ggml_backend_multi_buffer_set_usage`
- **Types:** 2/18 matched (target 17)
- **Missing types:** `ggml_backend_buffer_type_i`, `ggml_tensor`, `ggml_backend_buffer_type`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer`, `ggml_backend_buffer_usage`, `ggml_context`, `ggml_backend_i`, `ggml_cgraph`, `ggml_backend`, `ggml_backend_event`, `ggml_backend_device`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg`
- **Lint issues:** 49

### 4. ggml-cpu.common

- **Target:** `core.GGMLCpuCommon [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 47
- **Priority Score:** 47021208.0
- **Functions:** 8/8 matched (target 20)
- **Missing functions:** _none_
- **Types:** 2/4 matched (target 6)
- **Missing types:** `ggml_compute_params`, `ggml_tensor`

### 5. ggml-backend

- **Target:** `core.GGMLBackendUtils [STUB]`
- **Similarity:** 0.00
- **Dependents:** 20
- **Priority Score:** 20174710.0
- **Functions:** 128/131 matched (target 178)
- **Missing functions:** `ggml_backend_buffer_init`, `ggml_backend_buffer_copy_tensor`, `ggml_backend_buffer_is_multi_buffer`
- **Types:** 3/16 matched (target 7)
- **Missing types:** `ggml_tensor`, `ggml_backend_buffer_i`, `ggml_status`, `ggml_backend_buffer_usage`, `ggml_cgraph`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_context`, `ggml_op`, `ggml_backend_sched`, `ggml_hash_set`, `ggml_init_params`, `ggml_backend_buffer_type`
- **Lint issues:** 12

### 6. ggml-common

- **Target:** `core.GGMLCommon [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 16
- **Priority Score:** 16020210.0
- **Functions:** 0/2 matched (target 53)
- **Missing functions:** `union`, `GGML_TABLE_BEGIN`
- **Types:** 0/0 matched (target 28)
- **Missing types:** _none_

### 7. ggml-cpu.ggml-cpu

- **Target:** `core.GGMLCpuBackend [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 12
- **Priority Score:** 12154610.0
- **Functions:** 30/33 matched (target 159)
- **Missing functions:** `ggml_backend_cpu_device_get_extra_buffers_type`, `ggml_backend_cpu_device_context`, `ggml_backend_cpu_reg`
- **Types:** 1/13 matched (target 14)
- **Missing types:** `ggml_backend_cpu_context`, `ggml_backend_plan_cpu`, `ggml_cgraph`, `ggml_status`, `ggml_backend_i`, `ggml_backend_cpu_device_context`, `ggml_backend_dev_type`, `ggml_backend_dev_props`, `ggml_tensor`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_reg`
- **Lint issues:** 14

### 8. ggml-cpu.ggml-cpu-impl

- **Target:** `core.GGMLCpuImpl [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 11
- **Priority Score:** 11063910.0
- **Functions:** 31/32 matched (target 39)
- **Missing functions:** `ggml_vec_dot`
- **Types:** 2/7 matched (target 2)
- **Missing types:** `ggml_int16x8x2_t`, `ggml_uint8x16x2_t`, `ggml_uint8x16x4_t`, `ggml_int8x16x2_t`, `ggml_int8x16x4_t`

### 9. ggml-cpu.simd-mappings

- **Target:** `simd.GGMLSimd [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 10
- **Priority Score:** 10002010.0
- **Functions:** 20/20 matched (target 34)
- **Missing functions:** _none_
- **Types:** 0/0 matched (target 1)
- **Missing types:** _none_

### 10. ggml-cpu.traits

- **Target:** `core.GGMLCpuTraits [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 9
- **Priority Score:** 9051110.0
- **Functions:** 6/7 matched (target 6)
- **Missing functions:** `ggml_backend_cpu_get_extra_buffer_types`
- **Types:** 0/4 matched (target 2)
- **Missing types:** `ggml_compute_params`, `ggml_tensor`, `tensor_traits`, `extra_buffer_type`

### 11. ggml-quants

- **Target:** `core.GGMLQuantsRef [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 6
- **Priority Score:** 6017610.0
- **Functions:** 75/75 matched (target 122)
- **Missing functions:** _none_
- **Types:** 0/1 matched (target 3)
- **Missing types:** `ggml_type`
- **Lint issues:** 10

### 12. ggml-cpu.quants

- **Target:** `core.GGMLQuants [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 4
- **Priority Score:** 4277010.0
- **Functions:** 43/70 matched (target 75)
- **Missing functions:** `quantize_row_q8_0`, `quantize_row_q8_1`, `quantize_row_q8_K`, `ggml_vec_dot_q1_0_q8_0`, `ggml_vec_dot_q4_0_q8_0`, `ggml_vec_dot_q4_1_q8_1`, `ggml_vec_dot_q5_0_q8_0`, `ggml_vec_dot_q5_1_q8_1`, `ggml_vec_dot_q8_0_q8_0`, `ggml_vec_dot_mxfp4_q8_0`, `ggml_vec_dot_nvfp4_q8_0`, `ggml_vec_dot_q2_K_q8_K`, `ggml_vec_dot_q3_K_q8_K`, `ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q5_K_q8_K`, `ggml_vec_dot_q6_K_q8_K`, `ggml_vec_dot_tq1_0_q8_K`, `ggml_vec_dot_tq2_0_q8_K`, `ggml_vec_dot_iq2_xxs_q8_K`, `ggml_vec_dot_iq2_xs_q8_K`, `ggml_vec_dot_iq2_s_q8_K`, `ggml_vec_dot_iq3_xxs_q8_K`, `ggml_vec_dot_iq1_s_q8_K`, `ggml_vec_dot_iq1_m_q8_K`, `ggml_vec_dot_iq4_nl_q8_0`, `ggml_vec_dot_iq4_xs_q8_K`, `ggml_vec_dot_iq3_s_q8_K`
- **Types:** 0/0 matched (target 1)
- **Missing types:** _none_

### 13. ggml-sycl.type

- **Target:** `core.GGMLTypesTest [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 2
- **Priority Score:** 2040410.0
- **Functions:** 0/3 matched (target 9)
- **Missing functions:** `float_to_e4m3`, `e4m3_to_float`, `__nv_fp8_e4m3`
- **Types:** 0/1 matched
- **Missing types:** `__nv_fp8_e4m3`

### 14. ggml-sycl.backend

- **Target:** `core.GGMLBackendTest [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 2
- **Priority Score:** 2000010.0
- **Functions:** 0/0 matched (target 7)
- **Missing functions:** _none_
- **Types:** 0/0 matched (target 1)
- **Missing types:** _none_

### 15. ggml-threading

- **Target:** `core.GGMLScheduler [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 1
- **Priority Score:** 1000210.0
- **Functions:** 2/2 matched (target 18)
- **Missing functions:** _none_
- **Types:** 0/0 matched (target 7)
- **Missing types:** _none_
- **Lint issues:** 3

### 16. ggml-cpu.ops

- **Target:** `core.GGMLComputeOps [STUB]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 813010.0
- **Functions:** 150/220 matched (target 482)
- **Missing functions:** `ggml_compute_forward_dup_flt`, `ggml_compute_forward_dup_to_q`, `GGML_TENSOR_LOCALS`, `ggml_compute_forward_set_rows_f32`, `rotate_pairs`, `ggml_compute_forward_rope_flt`, `ggml_call_mul_mat`, `ggml_compute_forward_conv_transpose_2d_impl`, `ggml_compute_forward_upscale_f32`, `ggml_compute_forward_upscale`, `ggml_compute_forward_pad_f32`, `ggml_compute_forward_pad`, `ggml_compute_forward_pad_reflect_1d`, `for`, `ggml_wrap_index`, `ggml_compute_forward_roll_f32`, `ggml_compute_forward_roll`, `ggml_compute_forward_arange_f32`, `ggml_compute_forward_arange`, `ggml_compute_forward_timestep_embedding_f32`, `ggml_compute_forward_timestep_embedding`, `ggml_compute_forward_argsort_f32`, `ggml_compute_forward_argsort`, `ggml_compute_forward_top_k_f32`, `ggml_compute_forward_top_k`, `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, `ggml_compute_forward_flash_attn_ext_tiled`, `ggml_flash_attn_ext_reduce_partials`, `ggml_compute_forward_flash_attn_ext_f16`, `ggml_compute_forward_flash_attn_ext`, `ggml_compute_forward_flash_attn_back_f32`, `ggml_compute_forward_flash_attn_back`, `ggml_compute_forward_ssm_conv_f32`, `ggml_compute_forward_ssm_conv`, `ggml_compute_forward_ssm_scan_f32`, `ggml_compute_forward_ssm_scan`, `ggml_compute_forward_win_part_f32`, `ggml_compute_forward_win_part`, `ggml_compute_forward_win_unpart_f32`, `ggml_compute_forward_win_unpart`, `ggml_compute_forward_unary`, `ggml_compute_forward_glu`, `ggml_compute_forward_get_rel_pos_f16`, `ggml_compute_forward_get_rel_pos`, `ggml_compute_forward_add_rel_pos_f32`, `ggml_compute_forward_add_rel_pos`, `ggml_compute_forward_rwkv_wkv6_f32`, `ggml_compute_forward_rwkv_wkv6`, `ggml_compute_forward_gla_f32`, `ggml_compute_forward_gla`, `ggml_compute_forward_solve_tri_f32`, `ggml_compute_forward_solve_tri`, `ggml_compute_forward_gated_delta_net_one_chunk`, `ggml_compute_forward_gated_delta_net_f32`, `ggml_compute_forward_gated_delta_net`, `ggml_compute_forward_rwkv_wkv7_f32`, `ggml_compute_forward_rwkv_wkv7`, `ggml_compute_forward_map_custom1`, `ggml_compute_forward_map_custom2`, `ggml_compute_forward_map_custom3`, `ggml_compute_forward_custom`, `ggml_compute_forward_cross_entropy_loss_f32`, `ggml_compute_forward_cross_entropy_loss`, `ggml_compute_forward_cross_entropy_loss_back_f32`, `ggml_compute_forward_cross_entropy_loss_back`, `ggml_compute_forward_opt_step_adamw_f32`, `ggml_compute_forward_opt_step_adamw`, `ggml_compute_forward_opt_step_sgd_f32`, `ggml_compute_forward_opt_step_sgd`, `ggml_compute_forward_mul_mat`
- **Types:** 1/10 matched (target 15)
- **Missing types:** `ggml_tensor`, `ggml_sort_order`, `cmp_argsort`, `cmp_top_k`, `ggml_compute_params`, `ggml_map_custom1_op_params`, `ggml_map_custom2_op_params`, `ggml_map_custom3_op_params`, `ggml_custom_op_params`
- **Lint issues:** 64

### 17. ggml-openvino.utils

- **Target:** `core.GGMLTensorUtils [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 272710.0
- **Functions:** 0/22 matched (target 6)
- **Missing functions:** `graph_key`, `ov_runtime_context`, `ov_graph_compute`, `ov_graph_compute_dynamic`, `ov_graph_compute_static`, `checksum`, `print_input_tensor_info`, `print_output_tensor_info`, `pad_input`, `padded`, `set_zero_diagonal`, `get_inp_pos_tensor`, `get_is_prefill`, `get_ov_input_tensor`, `get_ov_input_tensor_static_decode`, `get_ov_input_tensor_static_prefill`, `create_ov_output_tensor`, `is_naive`, `naive_compute`, `key`, `new_state_tensor`, `convert_ggml_input_to_ov`
- **Types:** 0/5 matched (target 1)
- **Missing types:** `graph_key`, `graph_key_hash`, `ov_runtime_context`, `ggml_status`, `ggml_cgraph`

### 18. openvino.utils

- **Target:** `core.GGMLTestUtils [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 171710.0
- **Functions:** 0/17 matched (target 46)
- **Missing functions:** `getCurrentTime`, `dump_ov_model`, `num_inputs_check`, `non_cont_dim`, `argsort_descend`, `sorted_descend`, `is_permuted`, `permute`, `get_dimensions`, `rename_outputs_with_suffix`, `make_sin_cos`, `process_view_input`, `translate_1to1_match_2_inputs`, `rope_yarn_ramp_mix`, `dim_ids_vec`, `ggml_rope_yarn_corr_dim`, `ggml_rope_yarn_corr_dims`
- **Types:** 0/0 matched (target 13)
- **Missing types:** _none_
- **Lint issues:** 1

### 19. ggml-virtgpu.ggml-backend

- **Target:** `core.GGMLBackend [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 60610.0
- **Functions:** 0/6 matched (target 84)
- **Missing functions:** `ggml_backend_remoting_get_name`, `ggml_backend_remoting_free`, `ggml_backend_remoting_graph_compute`, `ggml_backend_remoting_graph_optimize`, `ggml_backend_remoting_guid`, `ggml_backend_remoting_device_init`
- **Types:** 0/0 matched (target 19)
- **Missing types:** _none_
- **Lint issues:** 22

### 20. backend.backend

- **Target:** `core.GGMLBackendIntegrationTest`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 60610.0
- **Functions:** 0/4 matched (target 3)
- **Missing functions:** `log_to_file_callback`, `apir_backend_deinit`, `apir_backend_initialize`, `apir_backend_dispatcher`
- **Types:** 0/2 matched (target 1)
- **Missing types:** `ggml_log_level`, `virgl_apir_callbacks`

### 21. ggml-sycl.quants

- **Target:** `core.GGMLCpuQuants [ZERO]`
- **Similarity:** 0.00
- **Dependents:** 0
- **Priority Score:** 50510.0
- **Functions:** 0/3 matched (target 45)
- **Missing functions:** `get_block_offset`, `get_d_offset`, `block_to_q8_1_ratio`
- **Types:** 0/2 matched (target 0)
- **Missing types:** `block_q_t`, `traits`

## Success Criteria

For each file to be considered "complete":
- **Similarity ≥ 0.85** (Excellent threshold)
- All public APIs ported
- All tests ported
- Documentation ported
- port-lint header present

## Next Commands

```bash
# Initialize task queue for systematic porting
cd tools/ast_distance
./ast_distance --init-tasks ../../ggml/src cpp ../../src/commonMain/kotlin/io.github.kotlinmania.llama.ore kotlin tasks.json ../../AGENTS.md

# Get next high-priority task
./ast_distance --assign tasks.json <agent-id>
```
