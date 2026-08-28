// port-lint: source llama.cpp/src/llama-arch.h llama.cpp/src/llama-arch.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// GGUF Metadata Key Identifiers
// Ported from: llama-arch.h  enum llm_kv
// =============================================================================

/**
 * GGUF metadata key identifiers.
 *
 * Each entry maps to a string pattern used in GGUF files. Patterns containing
 * `%s` are architecture-scoped — the skeleton is replaced by the architecture
 * name (e.g. `"llama"`) at lookup time via [LlmKvHelper]. Patterns without `%s`
 * are global keys shared across all architectures.
 *
 * @param pattern The GGUF key pattern. May contain `%s` for arch substitution.
 */
enum class LlmKv(val pattern: String) {
    // -- General --
    GENERAL_TYPE("general.type"),
    GENERAL_ARCHITECTURE("general.architecture"),
    GENERAL_QUANTIZATION_VERSION("general.quantization_version"),
    GENERAL_ALIGNMENT("general.alignment"),
    GENERAL_FILE_TYPE("general.file_type"),
    GENERAL_SAMPLING_SEQUENCE("general.sampling.sequence"),
    GENERAL_SAMPLING_TOP_K("general.sampling.top_k"),
    GENERAL_SAMPLING_TOP_P("general.sampling.top_p"),
    GENERAL_SAMPLING_MIN_P("general.sampling.min_p"),
    GENERAL_SAMPLING_XTC_PROBABILITY("general.sampling.xtc_probability"),
    GENERAL_SAMPLING_XTC_THRESHOLD("general.sampling.xtc_threshold"),
    GENERAL_SAMPLING_TEMP("general.sampling.temp"),
    GENERAL_SAMPLING_PENALTY_LAST_N("general.sampling.penalty_last_n"),
    GENERAL_SAMPLING_PENALTY_REPEAT("general.sampling.penalty_repeat"),
    GENERAL_SAMPLING_MIROSTAT("general.sampling.mirostat"),
    GENERAL_SAMPLING_MIROSTAT_TAU("general.sampling.mirostat_tau"),
    GENERAL_SAMPLING_MIROSTAT_ETA("general.sampling.mirostat_eta"),
    GENERAL_NAME("general.name"),
    GENERAL_AUTHOR("general.author"),
    GENERAL_VERSION("general.version"),
    GENERAL_URL("general.url"),
    GENERAL_DESCRIPTION("general.description"),
    GENERAL_LICENSE("general.license"),
    GENERAL_SOURCE_URL("general.source.url"),
    GENERAL_SOURCE_HF_REPO("general.source.huggingface.repository"),

    // -- Model hyperparameters --
    VOCAB_SIZE("%s.vocab_size"),
    CONTEXT_LENGTH("%s.context_length"),
    EMBEDDING_LENGTH("%s.embedding_length"),
    EMBEDDING_LENGTH_OUT("%s.embedding_length_out"),
    EMBEDDING_LENGTH_PER_LAYER("%s.embedding_length_per_layer_input"),
    FEATURES_LENGTH("%s.features_length"),
    BLOCK_COUNT("%s.block_count"),
    LEADING_DENSE_BLOCK_COUNT("%s.leading_dense_block_count"),
    FEED_FORWARD_LENGTH("%s.feed_forward_length"),
    EXPERT_FEED_FORWARD_LENGTH("%s.expert_feed_forward_length"),
    EXPERT_SHARED_FEED_FORWARD_LENGTH("%s.expert_shared_feed_forward_length"),
    EXPERT_CHUNK_FEED_FORWARD_LENGTH("%s.expert_chunk_feed_forward_length"),
    SWIGLU_CLAMP_EXP("%s.swiglu_clamp_exp"),
    SWIGLU_CLAMP_SHEXP("%s.swiglu_clamp_shexp"),
    USE_PARALLEL_RESIDUAL("%s.use_parallel_residual"),
    TENSOR_DATA_LAYOUT("%s.tensor_data_layout"),
    EXPERT_COUNT("%s.expert_count"),
    EXPERT_USED_COUNT("%s.expert_used_count"),
    EXPERT_SHARED_COUNT("%s.expert_shared_count"),
    EXPERT_GROUP_COUNT("%s.expert_group_count"),
    EXPERT_GROUP_USED_COUNT("%s.expert_group_used_count"),
    EXPERT_WEIGHTS_SCALE("%s.expert_weights_scale"),
    EXPERT_WEIGHTS_NORM("%s.expert_weights_norm"),
    EXPERT_GATING_FUNC("%s.expert_gating_func"),
    EXPERT_GROUP_SCALE("%s.expert_group_scale"),
    EXPERTS_PER_GROUP("%s.experts_per_group"),
    MOE_EVERY_N_LAYERS("%s.moe_every_n_layers"),
    MOE_LATENT_SIZE("%s.moe_latent_size"),
    NEXTN_PREDICT_LAYERS("%s.nextn_predict_layers"),
    NUM_DEEPSTACK_LAYERS("%s.n_deepstack_layers"),
    POOLING_TYPE("%s.pooling_type"),
    LOGIT_SCALE("%s.logit_scale"),
    DECODER_START_TOKEN_ID("%s.decoder_start_token_id"),
    DECODER_BLOCK_COUNT("%s.decoder_block_count"),
    ATTN_LOGIT_SOFTCAPPING("%s.attn_logit_softcapping"),
    ROUTER_LOGIT_SOFTCAPPING("%s.router_logit_softcapping"),
    FINAL_LOGIT_SOFTCAPPING("%s.final_logit_softcapping"),
    SWIN_NORM("%s.swin_norm"),
    RESCALE_EVERY_N_LAYERS("%s.rescale_every_n_layers"),
    TIME_MIX_EXTRA_DIM("%s.time_mix_extra_dim"),
    TIME_DECAY_EXTRA_DIM("%s.time_decay_extra_dim"),
    RESIDUAL_SCALE("%s.residual_scale"),
    EMBEDDING_SCALE("%s.embedding_scale"),
    TOKEN_SHIFT_COUNT("%s.token_shift_count"),
    INTERLEAVE_MOE_LAYER_STEP("%s.interleave_moe_layer_step"),
    FULL_ATTENTION_INTERVAL("%s.full_attention_interval"),

    // -- Attention --
    ATTENTION_HEAD_COUNT("%s.attention.head_count"),
    ATTENTION_HEAD_COUNT_KV("%s.attention.head_count_kv"),
    ATTENTION_MAX_ALIBI_BIAS("%s.attention.max_alibi_bias"),
    ATTENTION_CLAMP_KQV("%s.attention.clamp_kqv"),
    ATTENTION_KEY_LENGTH("%s.attention.key_length"),
    ATTENTION_VALUE_LENGTH("%s.attention.value_length"),
    ATTENTION_LAYERNORM_EPS("%s.attention.layer_norm_epsilon"),
    ATTENTION_LAYERNORM_RMS_EPS("%s.attention.layer_norm_rms_epsilon"),
    ATTENTION_GROUPNORM_EPS("%s.attention.group_norm_epsilon"),
    ATTENTION_GROUPNORM_GROUPS("%s.attention.group_norm_groups"),
    ATTENTION_CAUSAL("%s.attention.causal"),
    ATTENTION_Q_LORA_RANK("%s.attention.q_lora_rank"),
    ATTENTION_KV_LORA_RANK("%s.attention.kv_lora_rank"),
    ATTENTION_DECAY_LORA_RANK("%s.attention.decay_lora_rank"),
    ATTENTION_ICLR_LORA_RANK("%s.attention.iclr_lora_rank"),
    ATTENTION_VALUE_RESIDUAL_MIX_LORA_RANK("%s.attention.value_residual_mix_lora_rank"),
    ATTENTION_GATE_LORA_RANK("%s.attention.gate_lora_rank"),
    ATTENTION_RELATIVE_BUCKETS_COUNT("%s.attention.relative_buckets_count"),
    ATTENTION_SLIDING_WINDOW("%s.attention.sliding_window"),
    ATTENTION_SLIDING_WINDOW_PATTERN("%s.attention.sliding_window_pattern"),
    ATTENTION_SCALE("%s.attention.scale"),
    ATTENTION_OUTPUT_SCALE("%s.attention.output_scale"),
    ATTENTION_TEMPERATURE_LENGTH("%s.attention.temperature_length"),
    ATTENTION_TEMPERATURE_SCALE("%s.attention.temperature_scale"),
    ATTENTION_KEY_LENGTH_MLA("%s.attention.key_length_mla"),
    ATTENTION_VALUE_LENGTH_MLA("%s.attention.value_length_mla"),
    ATTENTION_KEY_LENGTH_SWA("%s.attention.key_length_swa"),
    ATTENTION_VALUE_LENGTH_SWA("%s.attention.value_length_swa"),
    ATTENTION_INDEXER_HEAD_COUNT("%s.attention.indexer.head_count"),
    ATTENTION_INDEXER_KEY_LENGTH("%s.attention.indexer.key_length"),
    ATTENTION_INDEXER_TOP_K("%s.attention.indexer.top_k"),
    ATTENTION_SHARED_KV_LAYERS("%s.attention.shared_kv_layers"),

    // -- RoPE --
    ROPE_DIMENSION_COUNT("%s.rope.dimension_count"),
    ROPE_DIMENSION_COUNT_SWA("%s.rope.dimension_count_swa"),
    ROPE_DIMENSION_SECTIONS("%s.rope.dimension_sections"),
    ROPE_FREQ_BASE("%s.rope.freq_base"),
    ROPE_FREQ_BASE_SWA("%s.rope.freq_base_swa"),
    ROPE_SCALE_LINEAR("%s.rope.scale_linear"),
    ROPE_SCALING_TYPE("%s.rope.scaling.type"),
    ROPE_SCALING_FACTOR("%s.rope.scaling.factor"),
    ROPE_SCALING_ATTN_FACTOR("%s.rope.scaling.attn_factor"),
    ROPE_SCALING_ORIG_CTX_LEN("%s.rope.scaling.original_context_length"),
    ROPE_SCALING_FINETUNED("%s.rope.scaling.finetuned"),
    ROPE_SCALING_YARN_LOG_MUL("%s.rope.scaling.yarn_log_multiplier"),
    ROPE_SCALING_YARN_EXT_FACTOR("%s.rope.scaling.yarn_ext_factor"),
    ROPE_SCALING_YARN_ATTN_FACTOR("%s.rope.scaling.yarn_attn_factor"),
    ROPE_SCALING_YARN_BETA_FAST("%s.rope.scaling.yarn_beta_fast"),
    ROPE_SCALING_YARN_BETA_SLOW("%s.rope.scaling.yarn_beta_slow"),

    // -- Split --
    SPLIT_NO("split.no"),
    SPLIT_COUNT("split.count"),
    SPLIT_TENSORS_COUNT("split.tensors.count"),

    // -- SSM --
    SSM_INNER_SIZE("%s.ssm.inner_size"),
    SSM_CONV_KERNEL("%s.ssm.conv_kernel"),
    SSM_STATE_SIZE("%s.ssm.state_size"),
    SSM_TIME_STEP_RANK("%s.ssm.time_step_rank"),
    SSM_GROUP_COUNT("%s.ssm.group_count"),
    SSM_DT_B_C_RMS("%s.ssm.dt_b_c_rms"),

    // -- KDA --
    KDA_HEAD_DIM("%s.kda.head_dim"),

    // -- WKV --
    WKV_HEAD_SIZE("%s.wkv.head_size"),

    // -- Tokenizer --
    TOKENIZER_MODEL("tokenizer.ggml.model"),
    TOKENIZER_PRE("tokenizer.ggml.pre"),
    TOKENIZER_LIST("tokenizer.ggml.tokens"),
    TOKENIZER_TOKEN_TYPE("tokenizer.ggml.token_type"),
    TOKENIZER_TOKEN_TYPE_COUNT("tokenizer.ggml.token_type_count"),
    TOKENIZER_SCORES("tokenizer.ggml.scores"),
    TOKENIZER_MERGES("tokenizer.ggml.merges"),
    TOKENIZER_BOS_ID("tokenizer.ggml.bos_token_id"),
    TOKENIZER_EOS_ID("tokenizer.ggml.eos_token_id"),
    TOKENIZER_EOT_ID("tokenizer.ggml.eot_token_id"),
    TOKENIZER_EOM_ID("tokenizer.ggml.eom_token_id"),
    TOKENIZER_UNK_ID("tokenizer.ggml.unknown_token_id"),
    TOKENIZER_SEP_ID("tokenizer.ggml.seperator_token_id"),
    TOKENIZER_PAD_ID("tokenizer.ggml.padding_token_id"),
    TOKENIZER_CLS_ID("tokenizer.ggml.cls_token_id"),
    TOKENIZER_MASK_ID("tokenizer.ggml.mask_token_id"),
    TOKENIZER_ADD_BOS("tokenizer.ggml.add_bos_token"),
    TOKENIZER_ADD_EOS("tokenizer.ggml.add_eos_token"),
    TOKENIZER_ADD_SEP("tokenizer.ggml.add_sep_token"),
    TOKENIZER_ADD_PREFIX("tokenizer.ggml.add_space_prefix"),
    TOKENIZER_REMOVE_EXTRA_WS("tokenizer.ggml.remove_extra_whitespaces"),
    TOKENIZER_PRECOMPILED_CHARSMAP("tokenizer.ggml.precompiled_charsmap"),
    TOKENIZER_HF_JSON("tokenizer.huggingface.json"),
    TOKENIZER_RWKV("tokenizer.rwkv.world"),
    TOKENIZER_CHAT_TEMPLATE("tokenizer.chat_template"),
    TOKENIZER_FIM_PRE_ID("tokenizer.ggml.fim_pre_token_id"),
    TOKENIZER_FIM_SUF_ID("tokenizer.ggml.fim_suf_token_id"),
    TOKENIZER_FIM_MID_ID("tokenizer.ggml.fim_mid_token_id"),
    TOKENIZER_FIM_PAD_ID("tokenizer.ggml.fim_pad_token_id"),
    TOKENIZER_FIM_REP_ID("tokenizer.ggml.fim_rep_token_id"),
    TOKENIZER_FIM_SEP_ID("tokenizer.ggml.fim_sep_token_id"),

    // -- Adapter --
    ADAPTER_TYPE("adapter.type"),
    ADAPTER_LORA_ALPHA("adapter.lora.alpha"),
    ADAPTER_LORA_TASK_NAME("adapter.lora.task_name"),
    ADAPTER_LORA_PROMPT_PREFIX("adapter.lora.prompt_prefix"),
    ADAPTER_ALORA_INVOCATION_TOKENS("adapter.alora.invocation_tokens"),

    // -- PosNet --
    POSNET_EMBEDDING_LENGTH("%s.posnet.embedding_length"),
    POSNET_BLOCK_COUNT("%s.posnet.block_count"),

    // -- ConvNext --
    CONVNEXT_EMBEDDING_LENGTH("%s.convnext.embedding_length"),
    CONVNEXT_BLOCK_COUNT("%s.convnext.block_count"),

    // -- Classifier --
    CLASSIFIER_OUTPUT_LABELS("%s.classifier.output_labels"),

    // -- ShortConv --
    SHORTCONV_L_CACHE("%s.shortconv.l_cache"),

    // -- XiELU --
    XIELU_ALPHA_N("xielu.alpha_n"),
    XIELU_ALPHA_P("xielu.alpha_p"),
    XIELU_BETA("xielu.beta"),
    XIELU_EPS("xielu.eps"),

    // -- Deprecated tokenizer keys --
    @Deprecated("Use TOKENIZER_FIM_PRE_ID instead")
    TOKENIZER_PREFIX_ID("tokenizer.ggml.prefix_token_id"),
    @Deprecated("Use TOKENIZER_FIM_SUF_ID instead")
    TOKENIZER_SUFFIX_ID("tokenizer.ggml.suffix_token_id"),
    @Deprecated("Use TOKENIZER_FIM_MID_ID instead")
    TOKENIZER_MIDDLE_ID("tokenizer.ggml.middle_token_id"),

    // -- Sentence-transformers dense layer features --
    DENSE_2_FEAT_IN("%s.dense_2_feat_in"),
    DENSE_2_FEAT_OUT("%s.dense_2_feat_out"),
    DENSE_3_FEAT_IN("%s.dense_3_feat_in"),
    DENSE_3_FEAT_OUT("%s.dense_3_feat_out"),
    ;
}

// =============================================================================
// Tensor Name Identifiers
// Ported from: llama-arch.h  enum llm_tensor  +  LLM_TENSOR_NAMES map
// =============================================================================

/**
 * Tensor name identifiers used in GGUF model files.
 *
 * Each entry carries a [ggufName] pattern that may contain `%d` templates
 * for the block (layer) index and/or expert index. Use [LlmTensorNameHelper]
 * to resolve these templates into concrete tensor names.
 *
 * @param ggufName Name pattern as stored in GGUF files. Empty string for
 *   entries that exist in the C++ enum but have no name mapping.
 */
enum class LlmTensor(val ggufName: String) {
    TOKEN_EMBD("token_embd"),
    TOKEN_EMBD_NORM("token_embd_norm"),
    TOKEN_TYPES("token_types"),
    POS_EMBD("position_embd"),
    DENSE_2_OUT("dense_2"),
    DENSE_3_OUT("dense_3"),
    OUTPUT("output"),
    OUTPUT_NORM("output_norm"),
    OUTPUT_NORM_LFM2("token_embd_norm"),  // fix for wrong tensor name in upstream
    ROPE_FREQS("rope_freqs"),
    ROPE_FACTORS_LONG("rope_factors_long"),
    ROPE_FACTORS_SHORT("rope_factors_short"),
    ATTN_Q("blk.%d.attn_q"),
    ATTN_K("blk.%d.attn_k"),
    ATTN_V("blk.%d.attn_v"),
    ATTN_QKV("blk.%d.attn_qkv"),
    ATTN_OUT("blk.%d.attn_output"),
    ATTN_NORM("blk.%d.attn_norm"),
    ATTN_NORM_2("blk.%d.attn_norm_2"),
    ATTN_OUT_NORM("blk.%d.attn_output_norm"),
    ATTN_POST_NORM("blk.%d.post_attention_norm"),
    ATTN_ROT_EMBD("blk.%d.attn_rot_embd"),
    ATTN_SINKS("blk.%d.attn_sinks"),
    ATTN_GATE("blk.%d.attn_gate"),
    FFN_GATE_INP("blk.%d.ffn_gate_inp"),
    FFN_GATE_INP_SHEXP("blk.%d.ffn_gate_inp_shexp"),
    FFN_NORM("blk.%d.ffn_norm"),
    FFN_POST_NORM("blk.%d.post_ffw_norm"),
    FFN_POST_NORM_1("blk.%d.post_ffw_norm_1"),
    FFN_POST_NORM_2("blk.%d.post_ffw_norm_2"),
    FFN_PRE_NORM_2("blk.%d.pre_ffw_norm_2"),
    FFN_GATE("blk.%d.ffn_gate"),
    FFN_DOWN("blk.%d.ffn_down"),
    FFN_UP("blk.%d.ffn_up"),
    FFN_ACT("blk.%d.ffn.act"),
    FFN_DOWN_EXP("blk.%d.ffn_down.%d"),
    FFN_GATE_EXP("blk.%d.ffn_gate.%d"),
    FFN_UP_EXP("blk.%d.ffn_up.%d"),
    FFN_NORM_EXPS("blk.%d.ffn_norm_exps"),
    FFN_DOWN_EXPS("blk.%d.ffn_down_exps"),
    FFN_GATE_EXPS("blk.%d.ffn_gate_exps"),
    FFN_UP_EXPS("blk.%d.ffn_up_exps"),
    FFN_GATE_UP_EXPS("blk.%d.ffn_gate_up_exps"),
    FFN_DOWN_SHEXP("blk.%d.ffn_down_shexp"),
    FFN_GATE_SHEXP("blk.%d.ffn_gate_shexp"),
    FFN_UP_SHEXP("blk.%d.ffn_up_shexp"),
    FFN_DOWN_CHEXPS("blk.%d.ffn_down_chexps"),
    FFN_GATE_CHEXPS("blk.%d.ffn_gate_chexps"),
    FFN_UP_CHEXPS("blk.%d.ffn_up_chexps"),
    FFN_EXP_PROBS_B("blk.%d.exp_probs_b"),
    FFN_LATENT_DOWN("blk.%d.ffn_latent_down"),
    FFN_LATENT_UP("blk.%d.ffn_latent_up"),
    ATTN_Q_NORM("blk.%d.attn_q_norm"),
    ATTN_K_NORM("blk.%d.attn_k_norm"),
    LAYER_OUT_NORM("blk.%d.layer_output_norm"),
    LAYER_OUT_SCALE("blk.%d.layer_output_scale"),
    POST_ATTN_NORM(""),  // in C++ enum but no name mapping
    POST_MLP_NORM(""),   // in C++ enum but no name mapping
    PER_LAYER_TOKEN_EMBD("per_layer_token_embd"),
    PER_LAYER_MODEL_PROJ("per_layer_model_proj"),
    PER_LAYER_INP_GATE("blk.%d.inp_gate"),
    PER_LAYER_PROJ("blk.%d.proj"),
    PER_LAYER_PROJ_NORM("per_layer_proj_norm"),
    PER_LAYER_POST_NORM("blk.%d.post_norm"),
    ALTUP_PROJ("altup_proj"),
    ALTUP_UNEMBD_PROJ("altup_unembd_proj"),
    ALTUP_CORRECT_COEF("blk.%d.altup_correct_coef"),
    ALTUP_CORRECT_SCALE("blk.%d.altup_correct_scale"),
    ALTUP_PREDICT_COEF("blk.%d.altup_predict_coef"),
    ALTUP_ROUTER("blk.%d.altup_router"),
    ALTUP_ROUTER_NORM("blk.%d.altup_router_norm"),
    LAUREL_L("blk.%d.laurel_l"),
    LAUREL_R("blk.%d.laurel_r"),
    LAUREL_POST_NORM("blk.%d.laurel_post_norm"),
    SSM_IN("blk.%d.ssm_in"),
    SSM_CONV1D("blk.%d.ssm_conv1d"),
    SSM_X("blk.%d.ssm_x"),
    SSM_DT("blk.%d.ssm_dt"),
    SSM_DT_NORM("blk.%d.ssm_dt_norm"),
    SSM_A("blk.%d.ssm_a"),
    SSM_A_NOSCAN("blk.%d.ssm_a"),  // uses same name; MUL instead of SSM_SCAN
    SSM_B_NORM("blk.%d.ssm_b_norm"),
    SSM_C_NORM("blk.%d.ssm_c_norm"),
    SSM_D("blk.%d.ssm_d"),
    SSM_NORM("blk.%d.ssm_norm"),
    SSM_OUT("blk.%d.ssm_out"),
    SSM_BETA_ALPHA("blk.%d.ssm_ba"),
    SSM_ALPHA("blk.%d.ssm_alpha"),
    SSM_CONV1D_Q("blk.%d.ssm_conv1d_q"),
    SSM_CONV1D_K("blk.%d.ssm_conv1d_k"),
    SSM_CONV1D_V("blk.%d.ssm_conv1d_v"),
    SSM_F_A("blk.%d.ssm_f_a"),
    SSM_F_B("blk.%d.ssm_f_b"),
    SSM_BETA("blk.%d.ssm_beta"),
    SSM_G_A("blk.%d.ssm_g_a"),
    SSM_G_B("blk.%d.ssm_g_b"),
    TIME_MIX_W0("blk.%d.time_mix_w0"),
    TIME_MIX_W1("blk.%d.time_mix_w1"),
    TIME_MIX_W2("blk.%d.time_mix_w2"),
    TIME_MIX_A0("blk.%d.time_mix_a0"),
    TIME_MIX_A1("blk.%d.time_mix_a1"),
    TIME_MIX_A2("blk.%d.time_mix_a2"),
    TIME_MIX_V0("blk.%d.time_mix_v0"),
    TIME_MIX_V1("blk.%d.time_mix_v1"),
    TIME_MIX_V2("blk.%d.time_mix_v2"),
    TIME_MIX_G1("blk.%d.time_mix_g1"),
    TIME_MIX_G2("blk.%d.time_mix_g2"),
    TIME_MIX_K_K("blk.%d.time_mix_k_k"),
    TIME_MIX_K_A("blk.%d.time_mix_k_a"),
    TIME_MIX_R_K("blk.%d.time_mix_r_k"),
    TIME_MIX_LERP_X("blk.%d.time_mix_lerp_x"),
    TIME_MIX_LERP_W("blk.%d.time_mix_lerp_w"),
    TIME_MIX_LERP_K("blk.%d.time_mix_lerp_k"),
    TIME_MIX_LERP_V("blk.%d.time_mix_lerp_v"),
    TIME_MIX_LERP_R("blk.%d.time_mix_lerp_r"),
    TIME_MIX_LERP_G("blk.%d.time_mix_lerp_g"),
    TIME_MIX_LERP_FUSED("blk.%d.time_mix_lerp_fused"),
    TIME_MIX_FIRST("blk.%d.time_mix_first"),
    TIME_MIX_DECAY("blk.%d.time_mix_decay"),
    TIME_MIX_DECAY_W1("blk.%d.time_mix_decay_w1"),
    TIME_MIX_DECAY_W2("blk.%d.time_mix_decay_w2"),
    TIME_MIX_KEY("blk.%d.time_mix_key"),
    TIME_MIX_VALUE("blk.%d.time_mix_value"),
    TIME_MIX_RECEPTANCE("blk.%d.time_mix_receptance"),
    TIME_MIX_GATE("blk.%d.time_mix_gate"),
    TIME_MIX_LN("blk.%d.time_mix_ln"),
    TIME_MIX_OUTPUT("blk.%d.time_mix_output"),
    CHANNEL_MIX_LERP_K("blk.%d.channel_mix_lerp_k"),
    CHANNEL_MIX_LERP_R("blk.%d.channel_mix_lerp_r"),
    CHANNEL_MIX_KEY("blk.%d.channel_mix_key"),
    CHANNEL_MIX_RECEPTANCE("blk.%d.channel_mix_receptance"),
    CHANNEL_MIX_VALUE("blk.%d.channel_mix_value"),
    ATTN_Q_A("blk.%d.attn_q_a"),
    ATTN_Q_B("blk.%d.attn_q_b"),
    ATTN_KV_A_MQA("blk.%d.attn_kv_a_mqa"),
    ATTN_KV_B("blk.%d.attn_kv_b"),
    ATTN_K_B("blk.%d.attn_k_b"),
    ATTN_V_B("blk.%d.attn_v_b"),
    ATTN_Q_A_NORM("blk.%d.attn_q_a_norm"),
    ATTN_KV_A_NORM("blk.%d.attn_kv_a_norm"),
    ATTN_SUB_NORM("blk.%d.attn_sub_norm"),
    FFN_SUB_NORM("blk.%d.ffn_sub_norm"),
    DEC_ATTN_NORM("dec.blk.%d.attn_norm"),
    DEC_ATTN_Q("dec.blk.%d.attn_q"),
    DEC_ATTN_K("dec.blk.%d.attn_k"),
    DEC_ATTN_V("dec.blk.%d.attn_v"),
    DEC_ATTN_OUT("dec.blk.%d.attn_o"),
    DEC_ATTN_REL_B("dec.blk.%d.attn_rel_b"),
    DEC_CROSS_ATTN_NORM("dec.blk.%d.cross_attn_norm"),
    DEC_CROSS_ATTN_Q("dec.blk.%d.cross_attn_q"),
    DEC_CROSS_ATTN_K("dec.blk.%d.cross_attn_k"),
    DEC_CROSS_ATTN_V("dec.blk.%d.cross_attn_v"),
    DEC_CROSS_ATTN_OUT("dec.blk.%d.cross_attn_o"),
    DEC_CROSS_ATTN_REL_B("dec.blk.%d.cross_attn_rel_b"),
    DEC_FFN_NORM("dec.blk.%d.ffn_norm"),
    DEC_FFN_GATE("dec.blk.%d.ffn_gate"),
    DEC_FFN_DOWN("dec.blk.%d.ffn_down"),
    DEC_FFN_UP("dec.blk.%d.ffn_up"),
    DEC_OUTPUT_NORM("dec.output_norm"),
    ENC_ATTN_NORM("enc.blk.%d.attn_norm"),
    ENC_ATTN_Q("enc.blk.%d.attn_q"),
    ENC_ATTN_K("enc.blk.%d.attn_k"),
    ENC_ATTN_V("enc.blk.%d.attn_v"),
    ENC_ATTN_OUT("enc.blk.%d.attn_o"),
    ENC_ATTN_REL_B("enc.blk.%d.attn_rel_b"),
    ENC_FFN_NORM("enc.blk.%d.ffn_norm"),
    ENC_FFN_GATE("enc.blk.%d.ffn_gate"),
    ENC_FFN_DOWN("enc.blk.%d.ffn_down"),
    ENC_FFN_UP("enc.blk.%d.ffn_up"),
    ENC_OUTPUT_NORM("enc.output_norm"),
    CLS("cls"),
    CLS_OUT("cls.output"),
    CLS_NORM("cls.norm"),
    CONV1D("conv1d"),
    CONVNEXT_DW("convnext.%d.dw"),
    CONVNEXT_NORM("convnext.%d.norm"),
    CONVNEXT_PW1("convnext.%d.pw1"),
    CONVNEXT_PW2("convnext.%d.pw2"),
    CONVNEXT_GAMMA("convnext.%d.gamma"),
    POS_NET_CONV1("posnet.%d.conv1"),
    POS_NET_CONV2("posnet.%d.conv2"),
    POS_NET_NORM("posnet.%d.norm"),
    POS_NET_NORM1("posnet.%d.norm1"),
    POS_NET_NORM2("posnet.%d.norm2"),
    POS_NET_ATTN_NORM("posnet.%d.attn_norm"),
    POS_NET_ATTN_Q("posnet.%d.attn_q"),
    POS_NET_ATTN_K("posnet.%d.attn_k"),
    POS_NET_ATTN_V("posnet.%d.attn_v"),
    POS_NET_ATTN_OUT("posnet.%d.attn_output"),
    SHORTCONV_CONV("blk.%d.shortconv.conv"),
    SHORTCONV_INPROJ("blk.%d.shortconv.in_proj"),
    SHORTCONV_OUTPROJ("blk.%d.shortconv.out_proj"),
    VISEXP_ATTN_QKV("blk.%d.vis_attn_qkv"),
    VISEXP_ATTN_OUT("blk.%d.vis_attn_output"),
    VISEXP_FFN_GATE("blk.%d.vis_gate"),
    VISEXP_FFN_DOWN("blk.%d.vis_down"),
    VISEXP_FFN_UP("blk.%d.vis_up"),
    INDEXER_K_NORM("blk.%d.indexer.k_norm"),
    INDEXER_PROJ("blk.%d.indexer.proj"),
    INDEXER_ATTN_K("blk.%d.indexer.attn_k"),
    INDEXER_ATTN_Q_B("blk.%d.indexer.attn_q_b"),
    NEXTN_EH_PROJ("blk.%d.nextn.eh_proj"),
    NEXTN_EMBED_TOKENS("blk.%d.nextn.embed_tokens"),
    NEXTN_ENORM("blk.%d.nextn.enorm"),
    NEXTN_HNORM("blk.%d.nextn.hnorm"),
    NEXTN_SHARED_HEAD_HEAD("blk.%d.nextn.shared_head_head"),
    NEXTN_SHARED_HEAD_NORM("blk.%d.nextn.shared_head_norm"),
    ;
}

// =============================================================================
// Tensor Layer Classification
// Ported from: llama-arch.h  enum llm_tensor_layer
// =============================================================================

/**
 * Classifies which part of the model graph a tensor belongs to.
 *
 * This is used by the allocator to assign the correct buffer type —
 * for example, input-layer tensors are typically placed on CPU/host memory.
 */
enum class LlmTensorLayer {
    INPUT,
    REPEATING,
    OUTPUT,
}

// =============================================================================
// Tensor Info
// Ported from: llama-arch.h  struct llm_tensor_info  +  LLM_TENSOR_INFOS map
// =============================================================================

/**
 * Metadata about a model weight tensor: which layer category it belongs to
 * and which GGML operation will consume it. The allocator uses this to pick
 * buffer types and to verify backend support.
 */
data class LlmTensorInfo(
    val layer: LlmTensorLayer,
    val op: GGMLOp,
)

/**
 * Maps every known [LlmTensor] to its [LlmTensorInfo].
 *
 * A mismatch between the declared layer/op and the actual usage can lead to
 * sub-optimal buffer assignment and extra overhead during computation.
 */
val LLM_TENSOR_INFOS: Map<LlmTensor, LlmTensorInfo> = mapOf(
    LlmTensor.TOKEN_EMBD             to LlmTensorInfo(LlmTensorLayer.INPUT,     GGMLOp.GET_ROWS),
    LlmTensor.POS_EMBD               to LlmTensorInfo(LlmTensorLayer.INPUT,     GGMLOp.GET_ROWS),
    LlmTensor.TOKEN_TYPES            to LlmTensorInfo(LlmTensorLayer.INPUT,     GGMLOp.GET_ROWS),
    LlmTensor.TOKEN_EMBD_NORM        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.OUTPUT                 to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.CLS                    to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.CLS_OUT                to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.CLS_NORM               to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    LlmTensor.DENSE_2_OUT            to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.DENSE_3_OUT            to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.OUTPUT_NORM            to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    LlmTensor.OUTPUT_NORM_LFM2       to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    LlmTensor.DEC_OUTPUT_NORM        to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    LlmTensor.ENC_OUTPUT_NORM        to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    LlmTensor.ROPE_FREQS             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ROPE),
    LlmTensor.ROPE_FACTORS_LONG      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ROPE),
    LlmTensor.ROPE_FACTORS_SHORT     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ROPE),
    LlmTensor.ATTN_Q                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_K                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_V                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_QKV               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_OUT               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_GATE              to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_GATE               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_DOWN               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_UP                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_DOWN_SHEXP         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_GATE_SHEXP         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_UP_SHEXP           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_Q_A               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_Q_B               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_KV_A_MQA          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_KV_B              to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_K_B               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_V_B               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ATTN_SINKS             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.SCALE),
    LlmTensor.DEC_ATTN_Q             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_ATTN_K             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_ATTN_V             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_ATTN_OUT           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_CROSS_ATTN_Q       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_CROSS_ATTN_K       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_CROSS_ATTN_V       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_CROSS_ATTN_OUT     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_FFN_GATE           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_FFN_DOWN           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.DEC_FFN_UP             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_ATTN_Q             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_ATTN_K             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_ATTN_V             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_ATTN_OUT           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_FFN_GATE           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_FFN_DOWN           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ENC_FFN_UP             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_GATE_INP_SHEXP     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_GATE_INP           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_IN                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_X                  to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_DT                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_OUT                to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_ALPHA              to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_BETA_ALPHA         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_W1            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_W2            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_A1            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_A2            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_V1            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_V2            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_G1            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_G2            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_DECAY_W1      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_DECAY_W2      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_KEY           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_VALUE         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_RECEPTANCE    to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_GATE          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_OUTPUT        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.CHANNEL_MIX_KEY        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.CHANNEL_MIX_RECEPTANCE to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.CHANNEL_MIX_VALUE      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.FFN_ACT                to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.DIV),
    LlmTensor.SSM_CONV1D             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.SSM_CONV),
    LlmTensor.SSM_A                  to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.SSM_SCAN),
    LlmTensor.SSM_A_NOSCAN           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_DT_NORM            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_B_NORM             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_C_NORM             to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_D                  to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_NORM               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_CONV1D_Q           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_CONV1D_K           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_CONV1D_V           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SSM_F_A                to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_F_B                to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_BETA               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_G_A                to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SSM_G_B                to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.TIME_MIX_LERP_X        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.TIME_MIX_LN            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.CHANNEL_MIX_LERP_K     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.CHANNEL_MIX_LERP_R     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.TIME_MIX_K_K           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.TIME_MIX_K_A           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.TIME_MIX_R_K           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.TIME_MIX_LERP_W        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_LERP_K        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_LERP_V        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_LERP_R        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_LERP_G        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_LERP_FUSED    to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_DECAY         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_W0            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_A0            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_V0            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    LlmTensor.TIME_MIX_FIRST         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.RWKV_WKV6),
    LlmTensor.ATTN_NORM              to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_NORM_2            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_OUT_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_POST_NORM         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_NORM               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_PRE_NORM_2         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_POST_NORM_1        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_POST_NORM_2        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_POST_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_NORM_EXPS          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_Q_NORM            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_K_NORM            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.LAYER_OUT_NORM         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.LAYER_OUT_SCALE        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_Q_A_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_KV_A_NORM         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ATTN_SUB_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_SUB_NORM           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.DEC_ATTN_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.DEC_CROSS_ATTN_NORM    to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.DEC_FFN_NORM           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ENC_ATTN_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ENC_FFN_NORM           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.DEC_ATTN_REL_B         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.GET_ROWS),
    LlmTensor.ENC_ATTN_REL_B         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.GET_ROWS),
    LlmTensor.FFN_DOWN_EXPS          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_GATE_EXPS          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_UP_EXPS            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_GATE_UP_EXPS       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_DOWN_CHEXPS        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_GATE_CHEXPS        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_UP_CHEXPS          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT_ID),
    LlmTensor.FFN_EXP_PROBS_B        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.ADD),
    // altup / laurel (gemma3n)
    LlmTensor.PER_LAYER_TOKEN_EMBD   to LlmTensorInfo(LlmTensorLayer.INPUT,     GGMLOp.GET_ROWS),
    LlmTensor.PER_LAYER_MODEL_PROJ   to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.PER_LAYER_PROJ_NORM    to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ALTUP_PROJ             to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.ALTUP_UNEMBD_PROJ      to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.PER_LAYER_INP_GATE     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.PER_LAYER_PROJ         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.PER_LAYER_POST_NORM    to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ALTUP_CORRECT_COEF     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ALTUP_CORRECT_SCALE    to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.ALTUP_PREDICT_COEF     to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ALTUP_ROUTER           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.ALTUP_ROUTER_NORM      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.LAUREL_L               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.LAUREL_R               to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.LAUREL_POST_NORM       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    // T5: loaded but never used
    LlmTensor.DEC_CROSS_ATTN_REL_B   to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.NONE),
    LlmTensor.CONV1D                 to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.IM2COL),
    LlmTensor.POS_NET_NORM           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.POS_NET_NORM1          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.POS_NET_NORM2          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.POS_NET_CONV1          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.IM2COL),
    LlmTensor.POS_NET_CONV2          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.IM2COL),
    LlmTensor.POS_NET_ATTN_NORM      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.POS_NET_ATTN_Q         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.POS_NET_ATTN_K         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.POS_NET_ATTN_V         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.POS_NET_ATTN_OUT       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.CONVNEXT_DW            to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.IM2COL),
    LlmTensor.CONVNEXT_NORM          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.CONVNEXT_PW1           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.CONVNEXT_PW2           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.CONVNEXT_GAMMA         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.SHORTCONV_CONV         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.SSM_CONV),
    LlmTensor.SHORTCONV_INPROJ       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.SHORTCONV_OUTPROJ      to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.VISEXP_ATTN_QKV        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.VISEXP_ATTN_OUT        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.VISEXP_FFN_GATE        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.VISEXP_FFN_DOWN        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.VISEXP_FFN_UP          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.INDEXER_K_NORM         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.INDEXER_PROJ           to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.INDEXER_ATTN_K         to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    LlmTensor.INDEXER_ATTN_Q_B       to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL_MAT),
    // NextN/MTP: output-layer tensors reserved for future MTP support
    LlmTensor.NEXTN_EH_PROJ          to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.NEXTN_EMBED_TOKENS     to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.GET_ROWS),
    LlmTensor.NEXTN_ENORM            to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.GET_ROWS),
    LlmTensor.NEXTN_HNORM            to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    LlmTensor.NEXTN_SHARED_HEAD_HEAD to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL_MAT),
    LlmTensor.NEXTN_SHARED_HEAD_NORM to LlmTensorInfo(LlmTensorLayer.OUTPUT,    GGMLOp.MUL),
    // Nemotron 3 Super
    LlmTensor.FFN_LATENT_DOWN        to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
    LlmTensor.FFN_LATENT_UP          to LlmTensorInfo(LlmTensorLayer.REPEATING, GGMLOp.MUL),
)

// =============================================================================
// Helper: KV Key Formatting
// Ported from: llama-arch.cpp  struct LLM_KV
// =============================================================================

/**
 * Resolves GGUF metadata key strings for a given architecture.
 *
 * Usage:
 * ```
 * val kv = LlmKvHelper(LlamaModelArch.LLAMA)
 * val key = kv(LlmKv.EMBEDDING_LENGTH)  // -> "llama.embedding_length"
 * ```
 *
 * @param arch  The model architecture whose name replaces `%s` in patterns.
 * @param suffix Optional suffix appended after a dot (e.g. for per-layer keys).
 */
class LlmKvHelper(
    val arch: LlamaModelArch,
    val suffix: String? = null,
) {
    /**
     * Format a [LlmKv] pattern into a concrete GGUF key string.
     * Replaces `%s` with the architecture's [LlamaModelArch.ggufName].
     */
    operator fun invoke(kv: LlmKv): String {
        var name = kv.pattern.replace("%s", arch.ggufName)
        if (suffix != null) {
            name += ".$suffix"
        }
        return name
    }
}

// =============================================================================
// Helper: Tensor Name Formatting
// Ported from: llama-arch.cpp  struct LLM_TN / LLM_TN_IMPL
// =============================================================================

/**
 * Resolves GGUF tensor name strings, substituting block and expert indices.
 *
 * Usage:
 * ```
 * val tn = LlmTensorNameHelper(LlamaModelArch.LLAMA)
 * val name = tn(LlmTensor.OUTPUT)                        // -> "output"
 * val name = tn(LlmTensor.ATTN_NORM, "weight", bid = 3)  // -> "blk.3.attn_norm.weight"
 * ```
 */
class LlmTensorNameHelper(val arch: LlamaModelArch) {

    /** Resolve a tensor name with optional suffix, block id, and expert id. */
    operator fun invoke(
        tensor: LlmTensor,
        suffix: String? = null,
        bid: Int = -1,
        xid: Int = -1,
    ): String {
        require(tensor.ggufName.isNotEmpty()) {
            "No GGUF name mapping for tensor ${tensor.name}"
        }
        var name = tensor.ggufName
        // Replace first %d with bid, second %d with xid
        if (bid >= 0) {
            name = name.replaceFirst("%d", bid.toString())
        }
        if (xid >= 0) {
            name = name.replaceFirst("%d", xid.toString())
        }
        if (suffix != null) {
            name += ".$suffix"
        }
        return name
    }
}

// =============================================================================
// Architecture Query Functions
// Ported from: llama-arch.cpp  free functions
// =============================================================================

/** Return all known architectures (excluding UNKNOWN). */
fun llmArchAll(): List<LlamaModelArch> = LlamaModelArch.entries.filter { it != LlamaModelArch.UNKNOWN }

/** Look up a [LlamaModelArch] by its GGUF name string, or [LlamaModelArch.UNKNOWN] if not found. */
fun llmArchFromString(name: String): LlamaModelArch {
    return LlamaModelArch.entries.find { it.ggufName == name } ?: LlamaModelArch.UNKNOWN
}

/** Get the GGUF name string for an architecture. */
fun llmArchName(arch: LlamaModelArch): String = arch.ggufName

/** Get the [LlmTensorInfo] for a tensor, or throw if unknown. */
fun llmTensorInfoFor(tensor: LlmTensor): LlmTensorInfo {
    return LLM_TENSOR_INFOS[tensor]
        ?: error("No tensor info for ${tensor.name}")
}

/** Returns true if the architecture is purely recurrent (Mamba, RWKV, etc.). */
fun llmArchIsRecurrent(arch: LlamaModelArch): Boolean = when (arch) {
    LlamaModelArch.MAMBA,
    LlamaModelArch.MAMBA2,
    LlamaModelArch.RWKV6,
    LlamaModelArch.RWKV6QWEN2,
    LlamaModelArch.RWKV7,
    LlamaModelArch.ARWKV7 -> true
    else -> false
}

/** Returns true if the architecture is a hybrid (attention + recurrent layers). */
fun llmArchIsHybrid(arch: LlamaModelArch): Boolean = when (arch) {
    LlamaModelArch.JAMBA,
    LlamaModelArch.FALCON_H1,
    LlamaModelArch.PLAMO2,
    LlamaModelArch.GRANITE_HYBRID,
    LlamaModelArch.LFM2,
    LlamaModelArch.LFM2MOE,
    LlamaModelArch.NEMOTRON_H,
    LlamaModelArch.NEMOTRON_H_MOE,
    LlamaModelArch.QWEN3NEXT,
    LlamaModelArch.KIMI_LINEAR,
    LlamaModelArch.QWEN35,
    LlamaModelArch.QWEN35MOE -> true
    else -> false
}

/** Returns true if the architecture uses diffusion-based generation. */
fun llmArchIsDiffusion(arch: LlamaModelArch): Boolean = when (arch) {
    LlamaModelArch.DREAM,
    LlamaModelArch.LLADA,
    LlamaModelArch.LLADA_MOE,
    LlamaModelArch.RND1 -> true
    else -> false
}

/** Returns true if the architecture supports split-mode tensors. */
fun llmArchSupportsSplitModeTensor(arch: LlamaModelArch): Boolean = when (arch) {
    LlamaModelArch.GROK,
    LlamaModelArch.MPT,
    LlamaModelArch.PLAMO2,
    LlamaModelArch.MINICPM3,
    LlamaModelArch.GEMMA3N,
    LlamaModelArch.MAMBA,
    LlamaModelArch.MAMBA2,
    LlamaModelArch.JAMBA,
    LlamaModelArch.FALCON_H1,
    LlamaModelArch.OLMO2,
    LlamaModelArch.OLMOE,
    LlamaModelArch.DEEPSEEK2,
    LlamaModelArch.GLM_DSA,
    LlamaModelArch.BITNET,
    LlamaModelArch.T5,
    LlamaModelArch.NEMOTRON_H,
    LlamaModelArch.NEMOTRON_H_MOE,
    LlamaModelArch.GRANITE_HYBRID,
    LlamaModelArch.LFM2,
    LlamaModelArch.LFM2MOE,
    LlamaModelArch.MINIMAX_M2,
    LlamaModelArch.MISTRAL4,
    LlamaModelArch.KIMI_LINEAR -> false
    else -> true
}
