// port-lint: source llama.cpp/src/llama-model.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*
import io.github.kotlinmania.llama.gguf.LlamaModelLoader
import io.github.kotlinmania.llama.gguf.TensorLoadFlags

// =============================================================================
// Model Architecture Enum
// Ported from: llama-arch.h  enum llm_arch
// =============================================================================

/**
 * Supported model architectures.
 *
 * Each variant maps 1-to-1 to a `LLM_ARCH_*` constant in the C++ codebase.
 * The [ggufName] string is the canonical key stored in GGUF metadata under
 * `general.architecture`.
 *
 * @param ggufName The architecture identifier as it appears in GGUF files.
 */
enum class LlamaModelArch(val ggufName: String) {
    CLIP("clip"),
    LLAMA("llama"),
    LLAMA4("llama4"),
    DECI("deci"),
    FALCON("falcon"),
    BAICHUAN("baichuan"),
    GROK("grok"),
    GPT2("gpt2"),
    GPTJ("gptj"),
    GPTNEOX("gptneox"),
    MPT("mpt"),
    STARCODER("starcoder"),
    REFACT("refact"),
    BERT("bert"),
    MODERN_BERT("modern_bert"),
    NOMIC_BERT("nomic_bert"),
    NOMIC_BERT_MOE("nomic_bert_moe"),
    NEO_BERT("neo_bert"),
    JINA_BERT_V2("jina_bert_v2"),
    JINA_BERT_V3("jina_bert_v3"),
    EUROBERT("eurobert"),
    BLOOM("bloom"),
    STABLELM("stablelm"),
    QWEN("qwen"),
    QWEN2("qwen2"),
    QWEN2MOE("qwen2moe"),
    QWEN2VL("qwen2vl"),
    QWEN3("qwen3"),
    QWEN3MOE("qwen3moe"),
    QWEN3NEXT("qwen3next"),
    QWEN3VL("qwen3vl"),
    QWEN3VLMOE("qwen3vlmoe"),
    QWEN35("qwen35"),
    QWEN35MOE("qwen35moe"),
    PHI2("phi2"),
    PHI3("phi3"),
    PHIMOE("phimoe"),
    PLAMO("plamo"),
    PLAMO2("plamo2"),
    PLAMO3("plamo3"),
    CODESHELL("codeshell"),
    ORION("orion"),
    INTERNLM2("internlm2"),
    MINICPM("minicpm"),
    MINICPM3("minicpm3"),
    GEMMA("gemma"),
    GEMMA2("gemma2"),
    GEMMA3("gemma3"),
    GEMMA3N("gemma3n"),
    GEMMA4("gemma4"),
    GEMMA_EMBEDDING("gemma_embedding"),
    STARCODER2("starcoder2"),
    MAMBA("mamba"),
    MAMBA2("mamba2"),
    JAMBA("jamba"),
    FALCON_H1("falcon_h1"),
    XVERSE("xverse"),
    COMMAND_R("command_r"),
    COHERE2("cohere2"),
    DBRX("dbrx"),
    OLMO("olmo"),
    OLMO2("olmo2"),
    OLMOE("olmoe"),
    OPENELM("openelm"),
    ARCTIC("arctic"),
    DEEPSEEK("deepseek"),
    DEEPSEEK2("deepseek2"),
    DEEPSEEK2OCR("deepseek2ocr"),
    CHATGLM("chatglm"),
    GLM4("glm4"),
    GLM4_MOE("glm4_moe"),
    GLM_DSA("glm_dsa"),
    BITNET("bitnet"),
    T5("t5"),
    T5ENCODER("t5encoder"),
    JAIS("jais"),
    JAIS2("jais2"),
    NEMOTRON("nemotron"),
    NEMOTRON_H("nemotron_h"),
    NEMOTRON_H_MOE("nemotron_h_moe"),
    EXAONE("exaone"),
    EXAONE4("exaone4"),
    EXAONE_MOE("exaone_moe"),
    RWKV6("rwkv6"),
    RWKV6QWEN2("rwkv6qwen2"),
    RWKV7("rwkv7"),
    ARWKV7("arwkv7"),
    GRANITE("granite"),
    GRANITE_MOE("granite_moe"),
    GRANITE_HYBRID("granite_hybrid"),
    CHAMELEON("chameleon"),
    WAVTOKENIZER_DEC("wavtokenizer_dec"),
    PLM("plm"),
    BAILINGMOE("bailingmoe"),
    BAILINGMOE2("bailingmoe2"),
    DOTS1("dots1"),
    ARCEE("arcee"),
    AFMOE("afmoe"),
    ERNIE4_5("ernie4_5"),
    ERNIE4_5_MOE("ernie4_5_moe"),
    HUNYUAN_MOE("hunyuan_moe"),
    HUNYUAN_DENSE("hunyuan_dense"),
    SMOLLM3("smollm3"),
    OPENAI_MOE("openai_moe"),
    LFM2("lfm2"),
    LFM2MOE("lfm2moe"),
    DREAM("dream"),
    SMALLTHINKER("smallthinker"),
    LLADA("llada"),
    LLADA_MOE("llada_moe"),
    SEED_OSS("seed_oss"),
    GROVEMOE("grovemoe"),
    APERTUS("apertus"),
    MINIMAX_M2("minimax_m2"),
    COGVLM("cogvlm"),
    RND1("rnd1"),
    PANGU_EMBED("pangu_embed"),
    MISTRAL3("mistral3"),
    MISTRAL4("mistral4"),
    PADDLEOCR("paddleocr"),
    MIMO2("mimo2"),
    STEP35("step35"),
    LLAMA_EMBED("llama_embed"),
    MAINCODER("maincoder"),
    KIMI_LINEAR("kimi_linear"),
    UNKNOWN("unknown");

    companion object {
        private val byGgufName: Map<String, LlamaModelArch> =
            entries.associateBy { it.ggufName }

        /** Look up an architecture from its GGUF metadata string. */
        fun fromGgufName(name: String): LlamaModelArch =
            byGgufName[name] ?: UNKNOWN
    }
}

// =============================================================================
// Model Type / Size Enum
// Ported from: llama-model.h  enum llm_type
// =============================================================================

/**
 * Known model size variants.
 *
 * These correspond to the `LLM_TYPE_*` constants in C++ and are used in
 * [LlamaHparams] to record which size variant was detected during model
 * loading.  The [displayName] is the human-readable label returned by
 * `llm_type_name()`.
 *
 * @param displayName Short human-readable label (e.g. "7B", "8x7B").
 */
enum class LlamaModelType(val displayName: String) {
    UNKNOWN("?B"),
    TYPE_14M("14M"),
    TYPE_17M("17M"),
    TYPE_22M("22M"),
    TYPE_33M("33M"),
    TYPE_47M("47M"),
    TYPE_60M("60M"),
    TYPE_70M("70M"),
    TYPE_80M("80M"),
    TYPE_109M("109M"),
    TYPE_137M("137M"),
    TYPE_140M("140M"),
    TYPE_149M("149M"),
    TYPE_160M("160M"),
    TYPE_190M("190M"),
    TYPE_220M("220M"),
    TYPE_250M("250M"),
    TYPE_256M("256M"),
    TYPE_270M("270M"),
    TYPE_335M("335M"),
    TYPE_350M("350M"),
    TYPE_360M("360M"),
    TYPE_395M("395M"),
    TYPE_410M("410M"),
    TYPE_450M("450M"),
    TYPE_475M("475M"),
    TYPE_558M("558M"),
    TYPE_700M("700M"),
    TYPE_770M("770M"),
    TYPE_780M("780M"),
    TYPE_950M("950M"),
    TYPE_0_3B("0.3B"),
    TYPE_0_5B("0.5B"),
    TYPE_0_6B("0.6B"),
    TYPE_0_8B("0.8B"),
    TYPE_1B("1B"),
    TYPE_1_2B("1.2B"),
    TYPE_1_3B("1.3B"),
    TYPE_1_4B("1.4B"),
    TYPE_1_5B("1.5B"),
    TYPE_1_6B("1.6B"),
    TYPE_1_7B("1.7B"),
    TYPE_1_8B("1.8B"),
    TYPE_2B("2B"),
    TYPE_2_6B("2.6B"),
    TYPE_2_8B("2.8B"),
    TYPE_2_9B("2.9B"),
    TYPE_3B("3B"),
    TYPE_4B("4B"),
    TYPE_6B("6B"),
    TYPE_6_9B("6.9B"),
    TYPE_7B("7B"),
    TYPE_8B("8B"),
    TYPE_9B("9B"),
    TYPE_11B("11B"),
    TYPE_12B("12B"),
    TYPE_13B("13B"),
    TYPE_14B("14B"),
    TYPE_15B("15B"),
    TYPE_16B("16B"),
    TYPE_20B("20B"),
    TYPE_26B("26B"),
    TYPE_27B("27B"),
    TYPE_30B("30B"),
    TYPE_31B("31B"),
    TYPE_32B("32B"),
    TYPE_34B("34B"),
    TYPE_35B("35B"),
    TYPE_36B("36B"),
    TYPE_40B("40B"),
    TYPE_65B("65B"),
    TYPE_70B("70B"),
    TYPE_120B("120B"),
    TYPE_142B("142B"),
    TYPE_236B("236B"),
    TYPE_290B("290B"),
    TYPE_314B("314B"),
    TYPE_405B("405B"),
    TYPE_671B("671B"),
    SMALL("0.1B"),
    MEDIUM("0.4B"),
    LARGE("0.8B"),
    XL("1.5B"),
    TYPE_A1_7B("A1.7B"),
    TYPE_A2_7B("A2.7B"),
    TYPE_8x7B("8x7B"),
    TYPE_8x22B("8x22B"),
    TYPE_16x12B("16x12B"),
    TYPE_16x3_8B("16x3.8B"),
    TYPE_10B_128x3_66B("10B+128x3.66B"),
    TYPE_57B_A14B("57B.A14B"),
    TYPE_17B_16E("17Bx16E (Scout)"),
    TYPE_17B_128E("17Bx128E (Maverick)"),
    TYPE_A13B("A13B"),
    TYPE_7B_A1B("7B.A1B"),
    TYPE_8B_A1B("8B.A1B"),
    TYPE_16B_A1B("16B.A1B"),
    TYPE_21B_A3B("21B.A3B"),
    TYPE_24B_A2B("24B.A2B"),
    TYPE_26B_A4B("26B.A4B"),
    TYPE_30B_A3B("30B.A3B"),
    TYPE_31B_A3_5B("31B.A3.5B"),
    TYPE_35B_A3B("35B.A3B"),
    TYPE_48B_A3B("48B.A3B"),
    TYPE_80B_A3B("80B.A3B"),
    TYPE_100B_A6B("100B.A6B"),
    TYPE_102B_A12B("102B.A12B"),
    TYPE_106B_A12B("106B.A12B"),
    TYPE_120B_A12B("120B.A12B"),
    TYPE_122B_A10B("122B.A10B"),
    TYPE_196B_A11B("196B.A11B"),
    TYPE_230B_A10B("230B.A10B"),
    TYPE_235B_A22B("235B.A22B"),
    TYPE_300B_A47B("300B.A47B"),
    TYPE_310B_A15B("310B.A15B"),
    TYPE_355B_A32B("355B.A32B"),
    TYPE_397B_A17B("397B.A17B"),
    TYPE_744B_A40B("744B.A40B"),
    TYPE_E2B("E2B"),
    TYPE_E4B("E4B"),
}

// =============================================================================
// RoPE Scaling Types
// Ported from: llama-model.cpp  LLAMA_ROPE_SCALING_TYPES
// =============================================================================

/** RoPE position-scaling strategy used during training and inference. */
enum class LlamaRopeScalingType(val displayName: String) {
    UNSPECIFIED("unspecified"),
    NONE("none"),
    LINEAR("linear"),
    YARN("yarn"),
    LONGROPE("longrope");

    companion object {
        /** Parse a rope-scaling type from the string stored in GGUF metadata. */
        fun fromName(name: String): LlamaRopeScalingType =
            entries.firstOrNull { it.displayName == name } ?: UNSPECIFIED
    }
}

/**
 * RoPE type — determines how positional encoding is applied.
 * Ported from: `llama_rope_type` in llama.h.
 */
enum class LlamaRopeType(val value: Int) {
    NONE(-1),
    NORM(0),
    NEOX(2),
    MROPE(8),
    IMROPE(24),
    VISION(32);

    companion object {
        fun fromValue(v: Int): LlamaRopeType = entries.firstOrNull { it.value == v } ?: NONE
    }
}

/** Expert gating function type. */
enum class LlamaExpertGatingFuncType(val id: Int) {
    NONE(0),
    SOFTMAX(1),
    SIGMOID(2),
    SOFTMAX_WEIGHT(3);

    companion object {
        fun fromId(value: Int): LlamaExpertGatingFuncType =
            entries.firstOrNull { it.id == value } ?: NONE
    }
}

// NOTE: LlamaSwaType is defined in KVCache.kt (NONE, STANDARD, CHUNKED, SYMMETRIC).

// =============================================================================
// Hyperparameters
// LlamaHparams is now defined in LlamaHparams.kt (port of llama-hparams.h/cpp)
// =============================================================================



// =============================================================================
// Layer Tensor Bundles
// Ported from: llama-model.h  struct llama_layer  (+ posnet / convnext / etc.)
// =============================================================================

/**
 * PosNet sub-layer tensors (used by WavTokenizer).
 *
 * Each field is nullable – `null` means the model does not use this tensor.
 */
data class LlamaLayerPosnet(
    // resnet
    var norm1: GGMLTensor? = null,
    var norm1B: GGMLTensor? = null,
    var conv1: GGMLTensor? = null,
    var conv1B: GGMLTensor? = null,
    var norm2: GGMLTensor? = null,
    var norm2B: GGMLTensor? = null,
    var conv2: GGMLTensor? = null,
    var conv2B: GGMLTensor? = null,
    // attention
    var attnNorm: GGMLTensor? = null,
    var attnNormB: GGMLTensor? = null,
    var attnQ: GGMLTensor? = null,
    var attnQB: GGMLTensor? = null,
    var attnK: GGMLTensor? = null,
    var attnKB: GGMLTensor? = null,
    var attnV: GGMLTensor? = null,
    var attnVB: GGMLTensor? = null,
    var attnO: GGMLTensor? = null,
    var attnOB: GGMLTensor? = null,
    // normalize
    var norm: GGMLTensor? = null,
    var normB: GGMLTensor? = null,
)

/** ConvNeXt sub-layer tensors. */
data class LlamaLayerConvnext(
    var dw: GGMLTensor? = null,
    var dwB: GGMLTensor? = null,
    var norm: GGMLTensor? = null,
    var normB: GGMLTensor? = null,
    var pw1: GGMLTensor? = null,
    var pw1B: GGMLTensor? = null,
    var pw2: GGMLTensor? = null,
    var pw2B: GGMLTensor? = null,
    var gamma: GGMLTensor? = null,
)

/** ShortConv sub-layer tensors. */
data class LlamaLayerShortconv(
    var inProj: GGMLTensor? = null,
    var conv: GGMLTensor? = null,
    var outProj: GGMLTensor? = null,
)

/** NextN sub-layer tensors. */
data class LlamaLayerNextn(
    var ehProj: GGMLTensor? = null,
    var embedTokens: GGMLTensor? = null,
    var enorm: GGMLTensor? = null,
    var hnorm: GGMLTensor? = null,
    var sharedHeadHead: GGMLTensor? = null,
    var sharedHeadNorm: GGMLTensor? = null,
)

/**
 * All weight tensors that belong to a single transformer layer.
 *
 * This is a faithful port of `struct llama_layer` from `llama-model.h`.
 * Nullable tensors default to `null` – a specific model architecture will only
 * populate the subset it needs.
 */
data class LlamaModelLayer(
    // ---- normalization ----
    var attnNorm: GGMLTensor? = null,
    var attnNormB: GGMLTensor? = null,
    var attnNorm2: GGMLTensor? = null,
    var attnNorm2B: GGMLTensor? = null,
    var attnQNorm: GGMLTensor? = null,
    var attnQNormB: GGMLTensor? = null,
    var attnKNorm: GGMLTensor? = null,
    var attnKNormB: GGMLTensor? = null,
    var attnOutNorm: GGMLTensor? = null,
    var attnOutNormB: GGMLTensor? = null,
    var attnQANorm: GGMLTensor? = null,
    var attnKvANorm: GGMLTensor? = null,
    var attnSubNorm: GGMLTensor? = null,
    var attnPostNorm: GGMLTensor? = null,
    var ffnSubNorm: GGMLTensor? = null,
    var attnNormCross: GGMLTensor? = null,
    var attnNormEnc: GGMLTensor? = null,
    var ssmNorm: GGMLTensor? = null,
    var ssmDtNorm: GGMLTensor? = null,
    var ssmBNorm: GGMLTensor? = null,
    var ssmCNorm: GGMLTensor? = null,

    // ---- attention ----
    var wq: GGMLTensor? = null,
    var wk: GGMLTensor? = null,
    var wv: GGMLTensor? = null,
    var wo: GGMLTensor? = null,
    var wqkv: GGMLTensor? = null,
    var wqA: GGMLTensor? = null,
    var wqB: GGMLTensor? = null,
    var wkvAMqa: GGMLTensor? = null,
    var wkvB: GGMLTensor? = null,
    var wkB: GGMLTensor? = null,
    var wvB: GGMLTensor? = null,
    var wqCross: GGMLTensor? = null,
    var wkCross: GGMLTensor? = null,
    var wvCross: GGMLTensor? = null,
    var woCross: GGMLTensor? = null,
    var wqEnc: GGMLTensor? = null,
    var wkEnc: GGMLTensor? = null,
    var wvEnc: GGMLTensor? = null,
    var woEnc: GGMLTensor? = null,
    var wqkvGate: GGMLTensor? = null,

    // ---- attention bias ----
    var bq: GGMLTensor? = null,
    var bk: GGMLTensor? = null,
    var bv: GGMLTensor? = null,
    var bo: GGMLTensor? = null,
    var bqkv: GGMLTensor? = null,

    // ---- relative position bias ----
    var attnRelB: GGMLTensor? = null,
    var attnRelBEnc: GGMLTensor? = null,
    var attnRelBCross: GGMLTensor? = null,

    // ---- FFN normalization ----
    var ffnNorm: GGMLTensor? = null,
    var ffnNormB: GGMLTensor? = null,
    var ffnPostNorm: GGMLTensor? = null,
    var ffnPostNorm1: GGMLTensor? = null,
    var ffnPostNorm2: GGMLTensor? = null,
    var ffnPreNorm2: GGMLTensor? = null,
    var layerOutNorm: GGMLTensor? = null,
    var layerOutNormB: GGMLTensor? = null,
    var ffnNormExps: GGMLTensor? = null,
    var ffnNormEnc: GGMLTensor? = null,

    // ---- feed-forward ----
    var ffnGate: GGMLTensor? = null,     // w1
    var ffnDown: GGMLTensor? = null,     // w2
    var ffnUp: GGMLTensor? = null,       // w3
    var ffnGateEnc: GGMLTensor? = null,
    var ffnDownEnc: GGMLTensor? = null,
    var ffnUpEnc: GGMLTensor? = null,

    // ---- MoE ----
    var ffnGateInp: GGMLTensor? = null,
    var ffnGateInpS: GGMLTensor? = null,
    var ffnGateExps: GGMLTensor? = null,
    var ffnDownExps: GGMLTensor? = null,
    var ffnUpExps: GGMLTensor? = null,
    var ffnGateUpExps: GGMLTensor? = null,
    var ffnGateInpB: GGMLTensor? = null,
    var ffnGateExpsB: GGMLTensor? = null,
    var ffnDownExpsB: GGMLTensor? = null,
    var ffnUpExpsB: GGMLTensor? = null,
    var ffnGateUpExpsB: GGMLTensor? = null,

    // ---- MoE per-expert scales ----
    var ffnGateExpsS: GGMLTensor? = null,
    var ffnDownExpsS: GGMLTensor? = null,
    var ffnUpExpsS: GGMLTensor? = null,

    // ---- MoE latent proj ----
    var ffnLatentDown: GGMLTensor? = null,
    var ffnLatentUp: GGMLTensor? = null,

    // ---- shared expert ----
    var ffnGateInpShexp: GGMLTensor? = null,
    var ffnGateShexp: GGMLTensor? = null,
    var ffnDownShexp: GGMLTensor? = null,
    var ffnUpShexp: GGMLTensor? = null,

    // ---- adjugate experts (chexps) ----
    var ffnGateChexps: GGMLTensor? = null,
    var ffnDownChexps: GGMLTensor? = null,
    var ffnUpChexps: GGMLTensor? = null,

    // ---- FFN bias ----
    var ffnGateB: GGMLTensor? = null,
    var ffnDownB: GGMLTensor? = null,
    var ffnUpB: GGMLTensor? = null,
    var ffnAct: GGMLTensor? = null,
    var ffnExpProbsB: GGMLTensor? = null,

    // ---- mamba proj ----
    var ssmIn: GGMLTensor? = null,
    var ssmX: GGMLTensor? = null,
    var ssmDt: GGMLTensor? = null,
    var ssmOut: GGMLTensor? = null,

    // ---- mamba ----
    var ssmConv1d: GGMLTensor? = null,
    var ssmA: GGMLTensor? = null,
    var ssmD: GGMLTensor? = null,

    // ---- mamba bias ----
    var ssmConv1dB: GGMLTensor? = null,
    var ssmDtB: GGMLTensor? = null,

    // ---- qwen3next ----
    var ssmBetaAlpha: GGMLTensor? = null,

    // ---- qwen3.5 ----
    var ssmAlpha: GGMLTensor? = null,

    // ---- RWKV ----
    var timeMixW1: GGMLTensor? = null,
    var timeMixW2: GGMLTensor? = null,
    var timeMixLerpX: GGMLTensor? = null,
    var timeMixLerpW: GGMLTensor? = null,
    var timeMixLerpK: GGMLTensor? = null,
    var timeMixLerpV: GGMLTensor? = null,
    var timeMixLerpR: GGMLTensor? = null,
    var timeMixLerpG: GGMLTensor? = null,
    var timeMixLerpFused: GGMLTensor? = null,
    var timeMixFirst: GGMLTensor? = null,
    var timeMixDecay: GGMLTensor? = null,
    var timeMixDecayW1: GGMLTensor? = null,
    var timeMixDecayW2: GGMLTensor? = null,
    var timeMixKey: GGMLTensor? = null,
    var timeMixKeyB: GGMLTensor? = null,
    var timeMixValue: GGMLTensor? = null,
    var timeMixValueB: GGMLTensor? = null,
    var timeMixReceptance: GGMLTensor? = null,
    var timeMixReceptanceB: GGMLTensor? = null,
    var timeMixGate: GGMLTensor? = null,

    // ---- RWKV7 ----
    var timeMixW0: GGMLTensor? = null,
    var timeMixA0: GGMLTensor? = null,
    var timeMixA1: GGMLTensor? = null,
    var timeMixA2: GGMLTensor? = null,
    var timeMixV0: GGMLTensor? = null,
    var timeMixV1: GGMLTensor? = null,
    var timeMixV2: GGMLTensor? = null,
    var timeMixG1: GGMLTensor? = null,
    var timeMixG2: GGMLTensor? = null,
    var timeMixKK: GGMLTensor? = null,
    var timeMixKA: GGMLTensor? = null,
    var timeMixRK: GGMLTensor? = null,
    var timeMixLn: GGMLTensor? = null,
    var timeMixLnB: GGMLTensor? = null,
    var timeMixOutput: GGMLTensor? = null,

    var channelMixLerpK: GGMLTensor? = null,
    var channelMixLerpR: GGMLTensor? = null,
    var channelMixKey: GGMLTensor? = null,
    var channelMixReceptance: GGMLTensor? = null,
    var channelMixValue: GGMLTensor? = null,

    // ---- long rope factors ----
    var ropeLong: GGMLTensor? = null,
    var ropeShort: GGMLTensor? = null,
    var ropeFreqs: GGMLTensor? = null,

    // ---- bitnet scale ----
    var wqS: GGMLTensor? = null,
    var wkS: GGMLTensor? = null,
    var wvS: GGMLTensor? = null,
    var woS: GGMLTensor? = null,
    var wqkvS: GGMLTensor? = null,
    var wqkvGateS: GGMLTensor? = null,
    var ffnGateS: GGMLTensor? = null,
    var ffnUpS: GGMLTensor? = null,
    var ffnDownS: GGMLTensor? = null,
    var ffnGateShexpS: GGMLTensor? = null,
    var ffnUpShexpS: GGMLTensor? = null,
    var ffnDownShexpS: GGMLTensor? = null,
    var ssmInS: GGMLTensor? = null,
    var ssmOutS: GGMLTensor? = null,
    var ssmAlphaS: GGMLTensor? = null,
    var ssmBetaS: GGMLTensor? = null,

    // ---- input scales ----
    var wqInS: GGMLTensor? = null,
    var wkInS: GGMLTensor? = null,
    var wvInS: GGMLTensor? = null,
    var woInS: GGMLTensor? = null,
    var wqkvInS: GGMLTensor? = null,
    var wqkvGateInS: GGMLTensor? = null,
    var ffnGateInS: GGMLTensor? = null,
    var ffnUpInS: GGMLTensor? = null,
    var ffnDownInS: GGMLTensor? = null,
    var ffnGateExpsInS: GGMLTensor? = null,
    var ffnDownExpsInS: GGMLTensor? = null,
    var ffnUpExpsInS: GGMLTensor? = null,
    var ffnGateShexpInS: GGMLTensor? = null,
    var ffnUpShexpInS: GGMLTensor? = null,
    var ffnDownShexpInS: GGMLTensor? = null,
    var ssmInInS: GGMLTensor? = null,
    var ssmOutInS: GGMLTensor? = null,
    var ssmAlphaInS: GGMLTensor? = null,
    var ssmBetaInS: GGMLTensor? = null,

    // ---- altup & laurel ----
    var perLayerInpGate: GGMLTensor? = null,
    var perLayerProj: GGMLTensor? = null,
    var perLayerPostNorm: GGMLTensor? = null,
    var altupCorrectCoef: GGMLTensor? = null,
    var altupCorrectScale: GGMLTensor? = null,
    var altupPredictCoef: GGMLTensor? = null,
    var altupRouter: GGMLTensor? = null,
    var altupRouterNorm: GGMLTensor? = null,
    var laurelL: GGMLTensor? = null,
    var laurelR: GGMLTensor? = null,
    var laurelPostNorm: GGMLTensor? = null,

    // ---- openai-moe ----
    var attnSinks: GGMLTensor? = null,

    // ---- cogvlm ----
    var visexpAttnWqkv: GGMLTensor? = null,
    var visexpAttnWo: GGMLTensor? = null,
    var visexpFfnGate: GGMLTensor? = null,
    var visexpFfnDown: GGMLTensor? = null,
    var visexpFfnUp: GGMLTensor? = null,

    // ---- xIELU activation parameters ----
    var ffnActAlphaN: GGMLTensor? = null,
    var ffnActAlphaP: GGMLTensor? = null,
    var ffnActBeta: GGMLTensor? = null,
    var ffnActEps: GGMLTensor? = null,

    // ---- Kimi Linear KDA ----
    var ssmQConv: GGMLTensor? = null,
    var ssmKConv: GGMLTensor? = null,
    var ssmVConv: GGMLTensor? = null,
    var ssmFA: GGMLTensor? = null,
    var ssmFB: GGMLTensor? = null,
    var ssmBeta: GGMLTensor? = null,
    var ssmGA: GGMLTensor? = null,
    var ssmGB: GGMLTensor? = null,
    var ssmONorm: GGMLTensor? = null,

    // ---- DSA ----
    var indexerKNorm: GGMLTensor? = null,
    var indexerKNormB: GGMLTensor? = null,
    var indexerProj: GGMLTensor? = null,
    var indexerAttnK: GGMLTensor? = null,
    var indexerAttnQB: GGMLTensor? = null,

    // ---- gemma4 layer output scale ----
    var outScale: GGMLTensor? = null,

    // ---- composite sub-layers ----
    val posnet: LlamaLayerPosnet = LlamaLayerPosnet(),
    val convnext: LlamaLayerConvnext = LlamaLayerConvnext(),
    val shortconv: LlamaLayerShortconv = LlamaLayerShortconv(),
    val nextn: LlamaLayerNextn = LlamaLayerNextn(),
)

// =============================================================================
// Top-Level Model Structure
// Ported from: llama-model.h  struct llama_model
// =============================================================================

/**
 * Top-level model state that owns every weight tensor and the hyperparameters.
 *
 * This is a direct port of `struct llama_model`.  The existing [LlamaModel]
 * class below continues to serve as the high-level "ready-to-run" model with
 * a forward-pass implementation.  This struct is the lower-level container
 * that the model loader populates.
 */
class LlamaModelData(
    var type: LlamaModelType = LlamaModelType.UNKNOWN,
    var arch: LlamaModelArch = LlamaModelArch.UNKNOWN,
    var name: String = "n/a",
) {
    val hparams: LlamaHparams = LlamaHparams()

    // ---- classifier labels (for classifier models) ----
    val classifierLabels: MutableList<String> = mutableListOf()

    // ---- global tensors ----
    var tokEmbd: GGMLTensor? = null
    var typeEmbd: GGMLTensor? = null
    var posEmbd: GGMLTensor? = null
    var tokNorm: GGMLTensor? = null
    var tokNormB: GGMLTensor? = null

    var outputNorm: GGMLTensor? = null
    var outputNormB: GGMLTensor? = null
    var output: GGMLTensor? = null
    var outputB: GGMLTensor? = null
    var outputNormEnc: GGMLTensor? = null

    // classifier
    var cls: GGMLTensor? = null
    var clsB: GGMLTensor? = null
    var clsOut: GGMLTensor? = null
    var clsOutB: GGMLTensor? = null
    var clsNorm: GGMLTensor? = null

    var conv1d: GGMLTensor? = null
    var conv1dB: GGMLTensor? = null

    // gemma3n altup
    var altupProj: GGMLTensor? = null
    var altupUnembdProj: GGMLTensor? = null
    var perLayerTokEmbd: GGMLTensor? = null
    var perLayerModelProj: GGMLTensor? = null
    var perLayerProjNorm: GGMLTensor? = null

    // sentence-transformers
    var dense2OutLayers: GGMLTensor? = null
    var dense2OutLayersB: GGMLTensor? = null
    var dense3OutLayers: GGMLTensor? = null

    // ---- layers ----
    val layers: MutableList<LlamaModelLayer> = mutableListOf()

    // ---- GGUF key-value metadata ----
    val ggufKv: MutableMap<String, String> = mutableMapOf()

    // ---- vocabulary (populated by model loader) ----
    var vocab: LlamaVocab? = null

    // ---- tensor index (for lookup by name) ----
    val tensorsByName: MutableList<Pair<String, GGMLTensor>> = mutableListOf()

    // ---- cached stats (populated by loadStats) ----
    var nElementsCached: Long = 0L
    var nBytesCached: Long = 0L

    // ---- timing ----
    var tLoadUs: Long = 0L
    var tStartUs: Long = 0L

    // -- query helpers --

    /** Return the architecture's GGUF name. */
    fun archName(): String = arch.ggufName

    /** Return the human-readable model type name (e.g. "7B"). */
    fun typeName(): String = type.displayName

    /** Short model description: "arch type". */
    fun desc(): String = "${archName()} ${typeName()}"

    /** Look up a weight tensor by its GGUF name, or `null` if not loaded. */
    fun getTensor(name: String): GGMLTensor? =
        tensorsByName.firstOrNull { it.first == name }?.second

    /** Total number of elements across all loaded tensors. */
    fun nElements(): Long =
        tensorsByName.sumOf { (_, t) ->
            var count = 1L
            for (i in 0 until GGML_MAX_DIMS) count *= t.ne[i]
            count
        }

    /** Total number of parameters (elements) across all loaded tensors. */
    fun nParams(): Long = nElements()

    /** Total byte size across all loaded tensors (accounts for quantization). */
    fun nBytes(): Long =
        tensorsByName.sumOf { (_, t) ->
            var count = 1L
            for (i in 0 until GGML_MAX_DIMS) count *= t.ne[i]
            val bs = t.type.byteSize.toLong()
            if (bs > 0) count * bs else count
        }

    /**
     * The model's total byte size as reported by the model loader.
     *
     * Returns the cached value from [loadStats] if available, otherwise falls
     * back to computing it from the loaded tensors.
     *
     * Ported from C++ `llama_model::size()` / `llama_model_size()`.
     */
    fun size(): Long = if (nBytesCached > 0) nBytesCached else nBytes()

    /** Total number of loaded tensors. Ported from C++ `llama_model::n_tensors()`. */
    fun nTensors(): Int = tensorsByName.size

    // -- architecture classification --

    /**
     * Whether this model has an encoder component.
     *
     * Ported from C++ `llama_model_has_encoder()`.
     */
    fun hasEncoder(): Boolean = when (arch) {
        LlamaModelArch.T5,
        LlamaModelArch.T5ENCODER -> true
        else -> false
    }

    /**
     * Whether this model has a decoder component.
     *
     * Ported from C++ `llama_model_has_decoder()`.
     */
    fun hasDecoder(): Boolean = when (arch) {
        LlamaModelArch.T5ENCODER -> false
        else -> true
    }

    /**
     * Whether this model uses a purely recurrent architecture (Mamba, RWKV, etc.).
     *
     * Ported from C++ `llama_model_is_recurrent()`.
     */
    fun isRecurrent(): Boolean = llmArchIsRecurrent(arch)

    /**
     * Whether this model uses a hybrid (attention + recurrent) architecture.
     *
     * Ported from C++ `llama_model_is_hybrid()`.
     */
    fun isHybrid(): Boolean = llmArchIsHybrid(arch)

    /**
     * Whether this model uses diffusion-based generation.
     *
     * Ported from C++ `llama_model_is_diffusion()`.
     */
    fun isDiffusion(): Boolean = llmArchIsDiffusion(arch)

    /**
     * The decoder start token ID, or -1 if not set.
     *
     * Ported from C++ `llama_model_decoder_start_token()`.
     */
    fun decoderStartToken(): Int = hparams.decStartTokenId

    // -- RoPE helpers --

    /**
     * Determine the RoPE type for this architecture.
     *
     * Ported from C++ `llama_model_rope_type()`. The mapping follows the
     * exhaustive switch in the original: architectures that don't use RoPE
     * return [LlamaRopeType.NONE], standard pair-wise RoPE returns
     * [LlamaRopeType.NORM], and NeoX-style (offset by nRot/2) returns
     * [LlamaRopeType.NEOX].
     */
    fun ropeType(): LlamaRopeType = when (arch) {
        // --- no RoPE ---
        LlamaModelArch.CLIP,
        LlamaModelArch.GPT2,
        LlamaModelArch.GPTJ,
        LlamaModelArch.MPT,
        LlamaModelArch.REFACT,
        LlamaModelArch.BLOOM,
        LlamaModelArch.MAMBA,
        LlamaModelArch.MAMBA2,
        LlamaModelArch.JAMBA,
        LlamaModelArch.JINA_BERT_V2,
        LlamaModelArch.T5,
        LlamaModelArch.T5ENCODER,
        LlamaModelArch.JAIS,
        LlamaModelArch.RWKV6,
        LlamaModelArch.RWKV6QWEN2,
        LlamaModelArch.RWKV7,
        LlamaModelArch.ARWKV7,
        LlamaModelArch.WAVTOKENIZER_DEC,
        LlamaModelArch.NEMOTRON_H,
        LlamaModelArch.NEMOTRON_H_MOE,
        LlamaModelArch.KIMI_LINEAR
            -> LlamaRopeType.NONE

        // --- normal (consecutive-pair) RoPE ---
        LlamaModelArch.LLAMA,
        LlamaModelArch.LLADA,
        LlamaModelArch.LLAMA4,
        LlamaModelArch.DECI,
        LlamaModelArch.BAICHUAN,
        LlamaModelArch.STARCODER,
        LlamaModelArch.INTERNLM2,
        LlamaModelArch.MINICPM,
        LlamaModelArch.XVERSE,
        LlamaModelArch.COMMAND_R,
        LlamaModelArch.COHERE2,
        LlamaModelArch.OLMO,
        LlamaModelArch.ARCTIC,
        LlamaModelArch.DEEPSEEK,
        LlamaModelArch.DEEPSEEK2,
        LlamaModelArch.DEEPSEEK2OCR,
        LlamaModelArch.PLM,
        LlamaModelArch.CHATGLM,
        LlamaModelArch.GRANITE,
        LlamaModelArch.GRANITE_MOE,
        LlamaModelArch.GRANITE_HYBRID,
        LlamaModelArch.CHAMELEON,
        LlamaModelArch.BAILINGMOE,
        LlamaModelArch.NEO_BERT,
        LlamaModelArch.SMOLLM3,
        LlamaModelArch.ARCEE,
        LlamaModelArch.ERNIE4_5,
        LlamaModelArch.ERNIE4_5_MOE,
        LlamaModelArch.MISTRAL3,
        LlamaModelArch.MISTRAL4,
        LlamaModelArch.LLAMA_EMBED,
        LlamaModelArch.MAINCODER,
        LlamaModelArch.GLM_DSA
            -> LlamaRopeType.NORM

        // --- NeoX (offset by nRot/2) RoPE ---
        LlamaModelArch.FALCON,
        LlamaModelArch.FALCON_H1,
        LlamaModelArch.GROK,
        LlamaModelArch.DBRX,
        LlamaModelArch.BERT,
        LlamaModelArch.JINA_BERT_V3,
        LlamaModelArch.MODERN_BERT,
        LlamaModelArch.NOMIC_BERT,
        LlamaModelArch.NOMIC_BERT_MOE,
        LlamaModelArch.EUROBERT,
        LlamaModelArch.STABLELM,
        LlamaModelArch.BITNET,
        LlamaModelArch.QWEN,
        LlamaModelArch.QWEN2,
        LlamaModelArch.DREAM,
        LlamaModelArch.QWEN2MOE,
        LlamaModelArch.QWEN3,
        LlamaModelArch.QWEN3MOE,
        LlamaModelArch.LLADA_MOE,
        LlamaModelArch.RND1,
        LlamaModelArch.OLMO2,
        LlamaModelArch.OLMOE,
        LlamaModelArch.PHI2,
        LlamaModelArch.PHI3,
        LlamaModelArch.PHIMOE,
        LlamaModelArch.PLAMO,
        LlamaModelArch.PLAMO2,
        LlamaModelArch.PLAMO3,
        LlamaModelArch.GEMMA,
        LlamaModelArch.GEMMA2,
        LlamaModelArch.GEMMA3,
        LlamaModelArch.GEMMA3N,
        LlamaModelArch.GEMMA4,
        LlamaModelArch.GEMMA_EMBEDDING,
        LlamaModelArch.STARCODER2,
        LlamaModelArch.OPENELM,
        LlamaModelArch.GPTNEOX,
        LlamaModelArch.CODESHELL,
        LlamaModelArch.ORION,
        LlamaModelArch.NEMOTRON,
        LlamaModelArch.EXAONE,
        LlamaModelArch.EXAONE4,
        LlamaModelArch.EXAONE_MOE,
        LlamaModelArch.MINICPM3,
        LlamaModelArch.BAILINGMOE2,
        LlamaModelArch.DOTS1,
        LlamaModelArch.HUNYUAN_MOE,
        LlamaModelArch.JAIS2,
        LlamaModelArch.OPENAI_MOE,
        LlamaModelArch.HUNYUAN_DENSE,
        LlamaModelArch.LFM2,
        LlamaModelArch.LFM2MOE,
        LlamaModelArch.SMALLTHINKER,
        LlamaModelArch.SEED_OSS,
        LlamaModelArch.GROVEMOE,
        LlamaModelArch.APERTUS,
        LlamaModelArch.MINIMAX_M2,
        LlamaModelArch.COGVLM,
        LlamaModelArch.PANGU_EMBED,
        LlamaModelArch.AFMOE,
        LlamaModelArch.QWEN3NEXT,
        LlamaModelArch.MIMO2,
        LlamaModelArch.STEP35
            -> LlamaRopeType.NEOX

        // --- multi-dimensional RoPE ---
        LlamaModelArch.QWEN2VL,
        LlamaModelArch.PADDLEOCR
            -> LlamaRopeType.MROPE

        // --- interleaved multi-dimensional RoPE ---
        LlamaModelArch.QWEN3VL,
        LlamaModelArch.QWEN3VLMOE,
        LlamaModelArch.QWEN35,
        LlamaModelArch.QWEN35MOE
            -> LlamaRopeType.IMROPE

        // --- conditional ---
        LlamaModelArch.GLM4 ->
            if (hparams.useMrope()) LlamaRopeType.MROPE else LlamaRopeType.NORM
        LlamaModelArch.GLM4_MOE ->
            if (hparams.useMrope()) LlamaRopeType.MROPE else LlamaRopeType.NEOX

        LlamaModelArch.UNKNOWN -> LlamaRopeType.NONE
    }

    /** RoPE frequency scale from training. Ported from C++ `llama_model_rope_freq_scale_train()`. */
    fun ropeFreqScaleTrain(): Float = hparams.ropeFreqScaleTrain

    /**
     * Get the effective RoPE frequency base for a given layer, accounting for
     * sliding-window attention layers that may use a different base.
     *
     * Ported from C++ `llama_model::get_rope_freq_base()`.
     */
    fun getRopeFreqBase(cparams: LlamaCParams, il: Int): Float =
        if (hparams.isSwa(il)) hparams.ropeFreqBaseTrainSwa else cparams.ropeFreqBase

    /**
     * Get the effective RoPE frequency scale for a given layer.
     *
     * Ported from C++ `llama_model::get_rope_freq_scale()`.
     */
    fun getRopeFreqScale(cparams: LlamaCParams, il: Int): Float =
        if (hparams.isSwa(il)) hparams.ropeFreqScaleTrainSwa else cparams.ropeFreqScale

    /**
     * Get the RoPE frequency-factor tensor for a given layer. Returns
     * `rope_freqs` if present, otherwise selects `rope_long` or `rope_short`
     * based on the sequence context length vs. the original training context.
     *
     * Ported from C++ `llama_model::get_rope_factors()`.
     */
    fun getRopeFactors(cparams: LlamaCParams, il: Int): GGMLTensor? {
        val layer = layers.getOrNull(il) ?: return null
        if (layer.ropeFreqs != null) return layer.ropeFreqs
        return if (cparams.nCtxSeq > hparams.nCtxOrigYarn) layer.ropeLong else layer.ropeShort
    }

    // -- metadata accessors --

    /**
     * Look up a GGUF metadata value by key.
     *
     * Ported from C++ `llama_model_meta_val_str()`.
     *
     * @return the value string, or `null` if the key is absent.
     */
    fun metaValStr(key: String): String? = ggufKv[key]

    /** Number of GGUF metadata key-value pairs. Ported from C++ `llama_model_meta_count()`. */
    fun metaCount(): Int = ggufKv.size

    /**
     * Get the key at a given index in the metadata map.
     *
     * Ported from C++ `llama_model_meta_key_by_index()`.
     *
     * @return the key string, or `null` if [index] is out of range.
     */
    fun metaKeyByIndex(index: Int): String? {
        if (index < 0 || index >= ggufKv.size) return null
        return ggufKv.keys.elementAtOrNull(index)
    }

    /**
     * Get the value at a given index in the metadata map.
     *
     * Ported from C++ `llama_model_meta_val_str_by_index()`.
     *
     * @return the value string, or `null` if [index] is out of range.
     */
    fun metaValStrByIndex(index: Int): String? {
        if (index < 0 || index >= ggufKv.size) return null
        return ggufKv.values.elementAtOrNull(index)
    }

    /**
     * Retrieve the chat template from GGUF metadata.
     *
     * Ported from C++ `llama_model_chat_template()`.
     *
     * @param templateName Optional template name suffix. When `null`, uses the
     *   default `tokenizer.chat_template` key.
     * @return the chat template string, or `null` if not found.
     */
    fun chatTemplate(templateName: String? = null): String? {
        val kv = LlmKvHelper(arch, templateName)
        val key = kv(LlmKv.TOKENIZER_CHAT_TEMPLATE)
        return ggufKv[key]
    }

    // -- tensor mutation --

    /**
     * Set the data of a tensor identified by name.
     *
     * Finds the tensor in [tensorsByName], validates that the provided data
     * size matches the tensor's byte requirements, and copies the data into
     * the tensor's backing storage.
     *
     * Ported from the concept of C++ `llama_model_set_tensor()` — in C++ this
     * copies into a backend buffer; here we copy into the ByteArray-backed
     * tensor data directly via [GGMLTensor.setData].
     *
     * @param name  GGUF tensor name (e.g. "blk.0.attn_q.weight").
     * @param data  Raw bytes to copy into the tensor.
     * @return `true` if the tensor was found and data was set, `false` otherwise.
     */
    fun setTensor(name: String, data: ByteArray): Boolean {
        val tensor = getTensor(name) ?: return false
        val expectedBytes = tensor.nBytes()
        require(data.size.toLong() == expectedBytes) {
            "setTensor('$name'): data size ${data.size} != expected $expectedBytes bytes"
        }
        tensor.setData(data, offset = 0, length = data.size)
        return true
    }

    // -- loading --

    /**
     * Cache the aggregate element/byte counts from the model loader.
     *
     * Ported from C++ `llama_model::load_stats()`.
     */
    fun loadStats(ml: LlamaModelLoader) {
        nElementsCached = ml.nElements
        nBytesCached = ml.nBytes
    }

    /**
     * Resolve the model architecture from the loader's GGUF metadata.
     *
     * Ported from C++ `llama_model::load_arch()`.
     *
     * @throws IllegalStateException if the architecture is unknown.
     */
    fun loadArch(ml: LlamaModelLoader) {
        arch = LlamaModelArch.fromGgufName(ml.archName)
        check(arch != LlamaModelArch.UNKNOWN) {
            "Unknown model architecture: '${ml.archName}'"
        }
    }

    /**
     * Populate [hparams] from GGUF metadata.
     *
     * Ported from C++ `llama_model::load_hparams()`.
     *
     * @param ml  The model loader that provides metadata access.
     */
    fun loadHParams(ml: LlamaModelLoader) {
        val kv = LlmKvHelper(arch)

        // ---- store GGUF key-value metadata ----
        ml.ggufContext.metadata.forEach { (k, v) ->
            ggufKv[k] = v.toString()
        }

        // ---- general.name ----
        ml.getKey<String>(kv(LlmKv.GENERAL_NAME), required = false)?.let { name = it }

        if (hparams.vocabOnly || arch == LlamaModelArch.CLIP) return

        // ---- basic shape ----
        hparams.nCtxTrain = ml.getKey<Int>(kv(LlmKv.CONTEXT_LENGTH)) ?: 0
        hparams.nEmbd     = ml.getKey<Int>(kv(LlmKv.EMBEDDING_LENGTH)) ?: 0
        hparams.nLayer    = ml.getKey<Int>(kv(LlmKv.BLOCK_COUNT)) ?: 0
        hparams.nExpert      = ml.getKey<Int>(kv(LlmKv.EXPERT_COUNT), required = false) ?: 0
        hparams.nExpertUsed  = ml.getKey<Int>(kv(LlmKv.EXPERT_USED_COUNT), required = false) ?: 0

        // ---- per-layer head counts and feed-forward sizes ----
        val nLayer = hparams.nLayer
        ml.getKeyOrArr(kv(LlmKv.FEED_FORWARD_LENGTH), hparams.nFfArr, nLayer, required = true)
        ml.getKeyOrArr(kv(LlmKv.ATTENTION_HEAD_COUNT), hparams.nHeadArr, nLayer, required = true)

        // n_head_kv defaults to n_head if absent
        val hasHeadKv = ml.getKeyOrArr(
            kv(LlmKv.ATTENTION_HEAD_COUNT_KV), hparams.nHeadKvArr, nLayer, required = false
        )
        if (!hasHeadKv) {
            for (i in 0 until nLayer) hparams.nHeadKvArr[i] = hparams.nHeadArr[i]
        }

        // ---- RoPE ----
        hparams.ropeFreqBaseTrain = ml.getKey<Float>(kv(LlmKv.ROPE_FREQ_BASE), required = false) ?: 10000.0f
        ml.getKey<String>(kv(LlmKv.ROPE_SCALING_TYPE), required = false)?.let {
            hparams.ropeScalingTypeTrain = LlamaRopeScalingType.fromName(it)
        }
        ml.getKey<Float>(kv(LlmKv.ROPE_SCALING_FACTOR), required = false)?.let { factor ->
            if (factor != 0.0f) hparams.ropeFreqScaleTrain = 1.0f / factor
        }
        ml.getKey<Boolean>(kv(LlmKv.ROPE_SCALING_FINETUNED), required = false)?.let {
            hparams.ropeFinetuned = it
        }
        ml.getKey<Float>(kv(LlmKv.ROPE_SCALING_ATTN_FACTOR), required = false)?.let {
            hparams.ropeAttnFactor = it
        }
        ml.getKey<Int>(kv(LlmKv.ROPE_SCALING_ORIG_CTX_LEN), required = false)?.let {
            hparams.nCtxOrigYarn = it
        }

        // ---- head dimensions ----
        val defaultHeadK = hparams.nEmbd / hparams.nHead()
        hparams.nEmbdHeadKFull = ml.getKey<Int>(kv(LlmKv.ATTENTION_KEY_LENGTH), required = false) ?: defaultHeadK
        hparams.nEmbdHeadVFull = ml.getKey<Int>(kv(LlmKv.ATTENTION_VALUE_LENGTH), required = false) ?: hparams.nEmbdHeadKFull

        // default SWA head dims to full dims
        hparams.nEmbdHeadKSwa = hparams.nEmbdHeadKFull
        hparams.nEmbdHeadVSwa = hparams.nEmbdHeadVFull

        // ---- RoPE dimension count ----
        hparams.nRotFull = ml.getKey<Int>(kv(LlmKv.ROPE_DIMENSION_COUNT), required = false)
            ?: hparams.nEmbdHeadKFull
        hparams.nRotSwa = hparams.nRotFull

        // ---- MoE params (optional) ----
        if (hparams.nExpert > 0) {
            hparams.nFfExp = ml.getKey<Int>(kv(LlmKv.EXPERT_FEED_FORWARD_LENGTH), required = false)
                ?: hparams.nFf()
            ml.getKey<Int>(kv(LlmKv.EXPERT_SHARED_COUNT), required = false)?.let {
                hparams.nExpertShared = it
            }
            ml.getKey<Int>(kv(LlmKv.EXPERT_GATING_FUNC), required = false)?.let {
                hparams.expertGatingFunc = it
            }
        }

        // ---- sliding window ----
        ml.getKey<Int>(kv(LlmKv.ATTENTION_SLIDING_WINDOW), required = false)?.let {
            hparams.nSwa = it
        }

        // ===================================================================
        // Architecture-specific hparams + model type determination
        // ===================================================================
        when (arch) {
            LlamaModelArch.LLAMA -> {
                hparams.fNormRmsEps = ml.getKey<Float>(kv(LlmKv.ATTENTION_LAYERNORM_RMS_EPS)) ?: 1e-5f
                hparams.nSwa = ml.getKey<Int>(kv(LlmKv.ATTENTION_SLIDING_WINDOW), required = false) ?: 0
                if (hparams.nSwa != 0) {
                    hparams.swaType = LlamaSwaType.STANDARD
                }

                type = when (hparams.nLayer) {
                    16  -> LlamaModelType.TYPE_1B
                    22  -> LlamaModelType.TYPE_1B
                    26  -> LlamaModelType.TYPE_3B
                    28  -> LlamaModelType.TYPE_3B
                    32  -> when {
                        hparams.nExpert == 8 -> LlamaModelType.TYPE_8x7B
                        hparams.nHead() >= 64 -> LlamaModelType.TYPE_8B
                        else -> LlamaModelType.TYPE_7B
                    }
                    40  -> LlamaModelType.TYPE_13B
                    48  -> LlamaModelType.TYPE_34B
                    56  -> if (hparams.nExpert == 8) LlamaModelType.TYPE_8x22B else LlamaModelType.TYPE_30B
                    60  -> LlamaModelType.TYPE_30B
                    80  -> when {
                        hparams.nExpert == 8 -> LlamaModelType.TYPE_8x22B
                        hparams.nHead() == hparams.nHeadKv() -> LlamaModelType.TYPE_65B
                        else -> LlamaModelType.TYPE_70B
                    }
                    else -> LlamaModelType.UNKNOWN
                }
            }

            LlamaModelArch.LLAMA4 -> {
                hparams.fNormRmsEps = ml.getKey<Float>(kv(LlmKv.ATTENTION_LAYERNORM_RMS_EPS)) ?: 1e-5f
                hparams.nSwa = ml.getKey<Int>(kv(LlmKv.ATTENTION_SLIDING_WINDOW), required = false) ?: 0
                hparams.swaType = if (hparams.nSwa != 0) LlamaSwaType.CHUNKED else LlamaSwaType.NONE
                hparams.nMoeLayerStep = ml.getKey<Int>(kv(LlmKv.INTERLEAVE_MOE_LAYER_STEP), required = false) ?: 0
                ml.getKey<Int>(kv(LlmKv.EXPERT_FEED_FORWARD_LENGTH), required = false)?.let {
                    hparams.nFfExp = it
                }
                ml.getKey<Int>(kv(LlmKv.EXPERT_SHARED_FEED_FORWARD_LENGTH), required = false)?.let {
                    hparams.nFfShexp = it
                }
                ml.getKey<Int>(kv(LlmKv.EXPERT_GATING_FUNC), required = false)?.let {
                    hparams.expertGatingFunc = it
                }

                type = when {
                    hparams.nExpert == 16 -> LlamaModelType.TYPE_17B_16E
                    hparams.nExpert == 128 -> LlamaModelType.TYPE_17B_128E
                    else -> LlamaModelType.UNKNOWN
                }
            }

            else -> {
                // LATER: implement hparams loading for arch ${arch.ggufName}
                println("WARNING: loadHParams not yet implemented for arch: ${arch.ggufName}")
            }
        }

        // Resolve the rope type from the architecture (C++ line 2949)
        hparams.ropeType = ropeType()
    }

    /**
     * Load weight tensors from a GGUF file into [layers] and global tensor fields.
     *
     * Ported from C++ `llama_model::load_tensors()`.
     *
     * @param ml  The model loader that provides tensor creation.
     * @return `true` on success.
     */
    fun loadTensors(ml: LlamaModelLoader): Boolean {
        val hp = hparams
        val tn = LlmTensorNameHelper(arch)

        val nEmbd      = hp.nEmbd.toLong()
        val nLayer     = hp.nLayer
        val nVocab     = ml.nTensors.toLong() // approximation; actual vocab may be in metadata
        // Attempt to read actual vocab size from metadata
        val kv = LlmKvHelper(arch)
        val nVocabActual: Long = ml.getKey<Int>(kv(LlmKv.VOCAB_SIZE), required = false)?.toLong()
            ?: run {
                // Infer from token_embd tensor shape
                val tokEmbdMeta = ml.getTensorMeta(tn(LlmTensor.TOKEN_EMBD, "weight"))
                tokEmbdMeta?.ne?.get(1) ?: 32000L
            }

        when (arch) {
            LlamaModelArch.LLAMA, LlamaModelArch.LLAMA4 -> {
                // ---- global tensors ----
                tokEmbd = ml.createTensor(
                    tn(LlmTensor.TOKEN_EMBD, "weight"),
                    longArrayOf(nEmbd, nVocabActual),
                    0
                )
                outputNorm = ml.createTensor(
                    tn(LlmTensor.OUTPUT_NORM, "weight"),
                    longArrayOf(nEmbd),
                    0
                )
                output = ml.createTensor(
                    tn(LlmTensor.OUTPUT, "weight"),
                    longArrayOf(nEmbd, nVocabActual),
                    TensorLoadFlags.NOT_REQUIRED
                )
                // If output is missing, it may be tied to token_embd
                if (output == null) {
                    output = ml.createTensor(
                        tn(LlmTensor.TOKEN_EMBD, "weight"),
                        longArrayOf(nEmbd, nVocabActual),
                        TensorLoadFlags.DUPLICATED
                    )
                }

                // ---- per-layer tensors ----
                for (i in 0 until nLayer) {
                    val layer = LlamaModelLayer()

                    val nHead     = hp.nHead(i).toLong()
                    val nHeadKv   = hp.nHeadKv(i).toLong()
                    val nFf       = hp.nFf(i).toLong()
                    val nEmbdHeadK = hp.nEmbdHeadK(i).toLong()
                    val nEmbdHeadV = hp.nEmbdHeadV(i).toLong()
                    val nEmbdKGqa = hp.nEmbdKGqa(i).toLong()
                    val nEmbdVGqa = hp.nEmbdVGqa(i).toLong()

                    layer.attnNorm = ml.createTensor(
                        tn(LlmTensor.ATTN_NORM, "weight", bid = i),
                        longArrayOf(nEmbd),
                        0
                    )

                    layer.wq = ml.createTensor(
                        tn(LlmTensor.ATTN_Q, "weight", bid = i),
                        longArrayOf(nEmbd, nHead * nEmbdHeadK),
                        0
                    )
                    layer.wk = ml.createTensor(
                        tn(LlmTensor.ATTN_K, "weight", bid = i),
                        longArrayOf(nEmbd, nEmbdKGqa),
                        0
                    )
                    layer.wv = ml.createTensor(
                        tn(LlmTensor.ATTN_V, "weight", bid = i),
                        longArrayOf(nEmbd, nEmbdVGqa),
                        0
                    )
                    layer.wo = ml.createTensor(
                        tn(LlmTensor.ATTN_OUT, "weight", bid = i),
                        longArrayOf(nHead * nEmbdHeadV, nEmbd),
                        0
                    )

                    // optional per-layer attention norms (qk norm)
                    layer.attnQNorm = ml.createTensor(
                        tn(LlmTensor.ATTN_Q_NORM, "weight", bid = i),
                        longArrayOf(nEmbdHeadK),
                        TensorLoadFlags.NOT_REQUIRED
                    )
                    layer.attnKNorm = ml.createTensor(
                        tn(LlmTensor.ATTN_K_NORM, "weight", bid = i),
                        longArrayOf(nEmbdHeadK),
                        TensorLoadFlags.NOT_REQUIRED
                    )

                    // optional rope factors
                    layer.ropeLong = ml.createTensor(
                        tn(LlmTensor.ROPE_FACTORS_LONG, "weight", bid = i),
                        longArrayOf(hp.nRot(i).toLong() / 2),
                        TensorLoadFlags.NOT_REQUIRED or TensorLoadFlags.DUPLICATED
                    )
                    layer.ropeShort = ml.createTensor(
                        tn(LlmTensor.ROPE_FACTORS_SHORT, "weight", bid = i),
                        longArrayOf(hp.nRot(i).toLong() / 2),
                        TensorLoadFlags.NOT_REQUIRED or TensorLoadFlags.DUPLICATED
                    )

                    layer.ffnNorm = ml.createTensor(
                        tn(LlmTensor.FFN_NORM, "weight", bid = i),
                        longArrayOf(nEmbd),
                        0
                    )

                    // Determine if this layer is MoE
                    val isMoe = when {
                        arch == LlamaModelArch.LLAMA4 && hp.nMoeLayerStep > 0 ->
                            (i + 1) % hp.nMoeLayerStep == 0
                        hp.nExpert > 0 -> true
                        else -> false
                    }

                    if (isMoe) {
                        val nExpert = hp.nExpert.toLong()
                        val nFfExpL = if (hp.nFfExp > 0) hp.nFfExp.toLong() else nFf

                        layer.ffnGateInp = ml.createTensor(
                            tn(LlmTensor.FFN_GATE_INP, "weight", bid = i),
                            longArrayOf(nEmbd, nExpert),
                            0
                        )

                        layer.ffnGateExps = ml.createTensor(
                            tn(LlmTensor.FFN_GATE_EXPS, "weight", bid = i),
                            longArrayOf(nEmbd, nFfExpL, nExpert),
                            TensorLoadFlags.NOT_REQUIRED
                        )
                        layer.ffnDownExps = ml.createTensor(
                            tn(LlmTensor.FFN_DOWN_EXPS, "weight", bid = i),
                            longArrayOf(nFfExpL, nEmbd, nExpert),
                            TensorLoadFlags.NOT_REQUIRED
                        )
                        layer.ffnUpExps = ml.createTensor(
                            tn(LlmTensor.FFN_UP_EXPS, "weight", bid = i),
                            longArrayOf(nEmbd, nFfExpL, nExpert),
                            TensorLoadFlags.NOT_REQUIRED
                        )

                        // shared expert (LLAMA4)
                        if (arch == LlamaModelArch.LLAMA4 && hp.nFfShexp > 0) {
                            val nFfShexpL = hp.nFfShexp.toLong()
                            layer.ffnGateShexp = ml.createTensor(
                                tn(LlmTensor.FFN_GATE_SHEXP, "weight", bid = i),
                                longArrayOf(nEmbd, nFfShexpL),
                                TensorLoadFlags.NOT_REQUIRED
                            )
                            layer.ffnDownShexp = ml.createTensor(
                                tn(LlmTensor.FFN_DOWN_SHEXP, "weight", bid = i),
                                longArrayOf(nFfShexpL, nEmbd),
                                TensorLoadFlags.NOT_REQUIRED
                            )
                            layer.ffnUpShexp = ml.createTensor(
                                tn(LlmTensor.FFN_UP_SHEXP, "weight", bid = i),
                                longArrayOf(nEmbd, nFfShexpL),
                                TensorLoadFlags.NOT_REQUIRED
                            )
                        }
                    } else {
                        // Dense FFN
                        layer.ffnGate = ml.createTensor(
                            tn(LlmTensor.FFN_GATE, "weight", bid = i),
                            longArrayOf(nEmbd, nFf),
                            0
                        )
                        layer.ffnDown = ml.createTensor(
                            tn(LlmTensor.FFN_DOWN, "weight", bid = i),
                            longArrayOf(nFf, nEmbd),
                            0
                        )
                        layer.ffnUp = ml.createTensor(
                            tn(LlmTensor.FFN_UP, "weight", bid = i),
                            longArrayOf(nEmbd, nFf),
                            0
                        )
                    }

                    layers.add(layer)
                }
            }

            else -> {
                println("WARNING: loadTensors not yet implemented for arch: ${arch.ggufName}")
                return true
            }
        }

        // Register all created tensors in tensorsByName
        fun register(name: String, tensor: GGMLTensor?) {
            if (tensor != null) tensorsByName.add(name to tensor)
        }

        register(tn(LlmTensor.TOKEN_EMBD, "weight"), tokEmbd)
        register(tn(LlmTensor.OUTPUT_NORM, "weight"), outputNorm)
        register(tn(LlmTensor.OUTPUT, "weight"), output)

        for ((i, layer) in layers.withIndex()) {
            register(tn(LlmTensor.ATTN_NORM, "weight", bid = i), layer.attnNorm)
            register(tn(LlmTensor.ATTN_Q, "weight", bid = i), layer.wq)
            register(tn(LlmTensor.ATTN_K, "weight", bid = i), layer.wk)
            register(tn(LlmTensor.ATTN_V, "weight", bid = i), layer.wv)
            register(tn(LlmTensor.ATTN_OUT, "weight", bid = i), layer.wo)
            register(tn(LlmTensor.ATTN_Q_NORM, "weight", bid = i), layer.attnQNorm)
            register(tn(LlmTensor.ATTN_K_NORM, "weight", bid = i), layer.attnKNorm)
            register(tn(LlmTensor.FFN_NORM, "weight", bid = i), layer.ffnNorm)
            register(tn(LlmTensor.FFN_GATE, "weight", bid = i), layer.ffnGate)
            register(tn(LlmTensor.FFN_DOWN, "weight", bid = i), layer.ffnDown)
            register(tn(LlmTensor.FFN_UP, "weight", bid = i), layer.ffnUp)
            register(tn(LlmTensor.FFN_GATE_INP, "weight", bid = i), layer.ffnGateInp)
            register(tn(LlmTensor.FFN_GATE_EXPS, "weight", bid = i), layer.ffnGateExps)
            register(tn(LlmTensor.FFN_DOWN_EXPS, "weight", bid = i), layer.ffnDownExps)
            register(tn(LlmTensor.FFN_UP_EXPS, "weight", bid = i), layer.ffnUpExps)
            register(tn(LlmTensor.FFN_GATE_SHEXP, "weight", bid = i), layer.ffnGateShexp)
            register(tn(LlmTensor.FFN_DOWN_SHEXP, "weight", bid = i), layer.ffnDownShexp)
            register(tn(LlmTensor.FFN_UP_SHEXP, "weight", bid = i), layer.ffnUpShexp)
            register(tn(LlmTensor.ROPE_FACTORS_LONG, "weight", bid = i), layer.ropeLong)
            register(tn(LlmTensor.ROPE_FACTORS_SHORT, "weight", bid = i), layer.ropeShort)
        }

        return true
    }

    /**
     * Print a comprehensive summary of the model to stdout.
     *
     * Ported from C++ `llama_model::print_info()`.
     */
    fun printInfo() {
        val hp = hparams
        println("llama_model_info: arch         = ${archName()}")
        println("llama_model_info: type         = ${typeName()}")
        println("llama_model_info: name         = $name")
        println("llama_model_info: nEmbd        = ${hp.nEmbd}")
        println("llama_model_info: nEmbdHeadK   = ${hp.nEmbdHeadKFull}")
        println("llama_model_info: nEmbdHeadV   = ${hp.nEmbdHeadVFull}")
        println("llama_model_info: nHead        = ${hp.nHead()}")
        println("llama_model_info: nHeadKv      = ${hp.nHeadKv()}")
        println("llama_model_info: nLayer       = ${hp.nLayer}")
        println("llama_model_info: nCtxTrain    = ${hp.nCtxTrain}")
        println("llama_model_info: nRot         = ${hp.nRotFull}")
        println("llama_model_info: nFf          = ${hp.nFf()}")
        if (hp.nExpert > 0) {
            println("llama_model_info: nExpert      = ${hp.nExpert}")
            println("llama_model_info: nExpertUsed  = ${hp.nExpertUsed}")
            if (hp.nFfExp > 0) {
                println("llama_model_info: nFfExp       = ${hp.nFfExp}")
            }
            if (hp.nFfShexp > 0) {
                println("llama_model_info: nFfShexp     = ${hp.nFfShexp}")
            }
        }
        println("llama_model_info: fNormEps     = ${hp.fNormEps}")
        println("llama_model_info: fNormRmsEps  = ${hp.fNormRmsEps}")
        println("llama_model_info: ropeFreqBase = ${hp.ropeFreqBaseTrain}")
        println("llama_model_info: ropeScaling  = ${hp.ropeScalingTypeTrain.displayName}")
        if (hp.nSwa > 0) {
            println("llama_model_info: nSwa         = ${hp.nSwa}")
            println("llama_model_info: swaType      = ${hp.swaType}")
        }
        val nParams = nParams()
        val modelSize = size()
        val bpw = if (nParams > 0) modelSize * 8.0 / nParams else 0.0
        println("llama_model_info: nParams      = $nParams")
        println("llama_model_info: nTensors     = ${nTensors()}")
        println("llama_model_info: size         = ${modelSize}")
        println("llama_model_info: BPW          = ${GGMLUtilities.formatDouble(bpw)}")
        println("llama_model_info: ropeType     = ${hp.ropeType}")
    }
}

// =============================================================================
// Tensor Name Mapping Helpers
// Ported from: llama-arch.h / llama-arch.cpp  tensor naming conventions
// =============================================================================

/**
 * Standard GGUF tensor name prefixes used across architectures.
 *
 * These constants mirror the `LLM_TENSOR_*` name strings used by
 * `llm_arch_from_string()` and tensor-loading code.  They are used when
 * building the expected tensor name for a given layer index.
 */
object LlamaTensorNames {
    const val TOKEN_EMBD        = "token_embd"
    const val TOKEN_EMBD_NORM   = "token_embd_norm"
    const val TOKEN_TYPES       = "token_types"
    const val POS_EMBD          = "position_embd"
    const val OUTPUT            = "output"
    const val OUTPUT_NORM       = "output_norm"
    const val ATTN_NORM         = "blk.%d.attn_norm"
    const val ATTN_NORM_2       = "blk.%d.attn_norm_2"
    const val ATTN_Q            = "blk.%d.attn_q"
    const val ATTN_K            = "blk.%d.attn_k"
    const val ATTN_V            = "blk.%d.attn_v"
    const val ATTN_QKV          = "blk.%d.attn_qkv"
    const val ATTN_OUT          = "blk.%d.attn_output"
    const val FFN_NORM          = "blk.%d.ffn_norm"
    const val FFN_GATE          = "blk.%d.ffn_gate"
    const val FFN_DOWN          = "blk.%d.ffn_down"
    const val FFN_UP            = "blk.%d.ffn_up"
    const val FFN_GATE_INP      = "blk.%d.ffn_gate_inp"
    const val FFN_GATE_EXPS     = "blk.%d.ffn_gate_exps"
    const val FFN_DOWN_EXPS     = "blk.%d.ffn_down_exps"
    const val FFN_UP_EXPS       = "blk.%d.ffn_up_exps"
    const val SSM_IN            = "blk.%d.ssm_in"
    const val SSM_CONV1D        = "blk.%d.ssm_conv1d"
    const val SSM_X             = "blk.%d.ssm_x"
    const val SSM_DT            = "blk.%d.ssm_dt"
    const val SSM_A             = "blk.%d.ssm_a"
    const val SSM_D             = "blk.%d.ssm_d"
    const val SSM_OUT           = "blk.%d.ssm_out"

    /** Build a per-layer tensor name by substituting the layer index. */
    fun forLayer(template: String, layerIndex: Int): String =
        template.replace("%d", layerIndex.toString())

    /**
     * Build the full tensor name with an optional weight/bias suffix.
     * Example: `forLayer(ATTN_Q, 3)` → `"blk.3.attn_q"`,
     * then appending `".weight"` → `"blk.3.attn_q.weight"`.
     */
    fun weight(template: String, layerIndex: Int): String =
        "${forLayer(template, layerIndex)}.weight"

    fun bias(template: String, layerIndex: Int): String =
        "${forLayer(template, layerIndex)}.bias"
}

// =============================================================================
// Existing Model Implementation (preserved)
// =============================================================================

/**
 * Utility function for linear transformation (matrix multiplication) with proper computation.
 * This function creates the result tensor, sets up the operation, and executes the computation.
 */
internal fun computeLinear(
    context: GGMLContext,
    graphAllocator: GGMLGraphAllocator,
    input: GGMLTensor,
    weight: GGMLTensor
): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne[0] = weight.ne[1] // output dimension
    result.ne[1] = input.ne[1]  // sequence length
    result.ne[2] = input.ne[2]  // batch size
    for (i in 3 until GGML_MAX_DIMS) result.ne[i] = 1L
    result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
    result.op = GGMLOp.MUL_MAT
    result.src[0] = weight
    result.src[1] = input
    graphAllocator.allocateTensor(result)
    
    // Execute the actual computation
    computeMatMul(graphAllocator, weight, input, result)
    return result
}

/**
 * Utility function for element-wise addition with proper computation.
 */
internal fun computeElementAdd(
    context: GGMLContext,
    graphAllocator: GGMLGraphAllocator,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    val result = GGMLTensor(type = GGMLType.F32)
    result.ne = a.ne.copyOf()
    result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
    result.op = GGMLOp.ADD
    result.src[0] = a
    result.src[1] = b
    graphAllocator.allocateTensor(result)
    
    // Execute the actual computation
    computeAdd(graphAllocator, a, b, result)
    return result
}

/**
 * Utility function for element-wise multiplication with proper computation.
 */
internal fun computeElementMul(
    context: GGMLContext,
    graphAllocator: GGMLGraphAllocator,
    a: GGMLTensor,
    b: GGMLTensor
): GGMLTensor {
    val result = GGMLTensor(type = a.type)
    result.ne = a.ne.copyOf()
    result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
    result.op = GGMLOp.MUL
    result.src[0] = a
    result.src[1] = b
    graphAllocator.allocateTensor(result)
    
    // Execute the actual computation  
    computeMul(graphAllocator, a, b, result)
    return result
}

/**
 * Utility function for SILU activation with proper computation.
 */
internal fun computeSilu(
    graphAllocator: GGMLGraphAllocator,
    input: GGMLTensor
): GGMLTensor {
    val result = GGMLTensor(type = input.type)
    result.ne = input.ne.copyOf()
    result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
    result.op = GGMLOp.SILU
    result.src[0] = input
    graphAllocator.allocateTensor(result)
    
    // Use existing SILU implementation
    computeSilu(graphAllocator, input, result)
    return result
}

/**
 * LLaMA model configuration.
 */
data class LlamaConfig(
    val vocabSize: Int = 32000,
    val hiddenSize: Int = 4096,
    val intermediateSize: Int = 11008,
    val numHiddenLayers: Int = 32,
    val numAttentionHeads: Int = 32,
    val numKeyValueHeads: Int = 32,
    val hiddenAct: String = "silu",
    val maxPositionEmbeddings: Int = 2048,
    val initializer_range: Float = 0.02f,
    val rmsNormEps: Float = 1e-6f,
    val useCache: Boolean = true,
    val ropeTheta: Float = 10000.0f
) {
    val headDim: Int = hiddenSize / numAttentionHeads
    
    init {
        require(hiddenSize % numAttentionHeads == 0) {
            "Hidden size must be divisible by number of attention heads"
        }
        require(numKeyValueHeads <= numAttentionHeads) {
            "Number of key-value heads must be <= number of attention heads"
        }
    }
}

/**
 * RMS (Root Mean Square) Layer Normalization.
 */
class RMSNorm(
    val normalizedShape: Int,
    val eps: Float = 1e-6f
) {
    // Weight parameter for scaling
    val weight: GGMLTensor = GGMLTensor(type = GGMLType.F32).apply {
        ne[0] = normalizedShape.toLong()
        for (i in 1 until GGML_MAX_DIMS) ne[i] = 1L
        nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
    }
    
    fun forward(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        input: GGMLTensor
    ): GGMLTensor {
        val result = GGMLTensor(type = input.type).apply {
            ne = input.ne.copyOf()
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
            op = GGMLOp.RMS_NORM
            src[0] = input
            graphAllocator.allocateTensor(this)
        }
        computeRMSNorm(graphAllocator, input, eps, result)
        return result
    }
}

/**
 * Multi-Layer Perceptron (Feed-Forward Network) for LLaMA.
 */
class LlamaMLP(
    val config: LlamaConfig
) {
    // Linear projections
    val gateProj: GGMLTensor = createLinearWeight(config.hiddenSize, config.intermediateSize)
    val upProj: GGMLTensor = createLinearWeight(config.hiddenSize, config.intermediateSize)
    val downProj: GGMLTensor = createLinearWeight(config.intermediateSize, config.hiddenSize)
    
    private fun createLinearWeight(inputSize: Int, outputSize: Int): GGMLTensor {
        return GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = inputSize.toLong()
            ne[1] = outputSize.toLong()
            for (i in 2 until GGML_MAX_DIMS) ne[i] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
    }
    
    fun forward(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        input: GGMLTensor
    ): GGMLTensor {
        // Gate path: gate_proj(x)
        val gate = computeLinear(context, graphAllocator, input, gateProj)
        val gateActivated = computeSilu(graphAllocator, gate)
        
        // Up path: up_proj(x)  
        val up = computeLinear(context, graphAllocator, input, upProj)
        
        // Element-wise multiply: gate * up
        val intermediate = computeElementMul(context, graphAllocator, gateActivated, up)
        
        // Down projection: down_proj(intermediate)
        return computeLinear(context, graphAllocator, intermediate, downProj)
    }
    
    private fun linear(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        input: GGMLTensor,
        weight: GGMLTensor
    ): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = weight.ne[1] // output dimension
        result.ne[1] = input.ne[1]  // sequence length
        result.ne[2] = input.ne[2]  // batch size
        for (i in 3 until GGML_MAX_DIMS) result.ne[i] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        result.op = GGMLOp.MUL_MAT
        result.src[0] = weight
        result.src[1] = input
        graphAllocator.allocateTensor(result)
        return result
    }
    
    private fun silu(context: GGMLContext, graphAllocator: GGMLGraphAllocator, input: GGMLTensor): GGMLTensor {
        val result = GGMLTensor(type = input.type)
        result.ne = input.ne.copyOf()
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        result.op = GGMLOp.SILU
        result.src[0] = input
        graphAllocator.allocateTensor(result)
        
        // Use existing SILU implementation
        computeSilu(graphAllocator, input, result)
        return result
    }
    
    private fun elementWiseMul(context: GGMLContext, graphAllocator: GGMLGraphAllocator, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
        val result = GGMLTensor(type = a.type)
        result.ne = a.ne.copyOf()
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        result.op = GGMLOp.MUL
        result.src[0] = a
        result.src[1] = b
        graphAllocator.allocateTensor(result)
        
        // Use existing MUL implementation
        computeMul(graphAllocator, a, b, result)
        return result
    }
}

/**
 * Single transformer layer (decoder block) for LLaMA.
 */
class LlamaDecoderLayer(
    val config: LlamaConfig,
    val layerIdx: Int
) {
    val selfAttention = LlamaAttention(
        hiddenSize = config.hiddenSize,
        numHeads = config.numAttentionHeads,
        headDim = config.headDim,
        maxPositionEmbeddings = config.maxPositionEmbeddings,
        ropeTheta = config.ropeTheta
    )
    
    val mlp = LlamaMLP(config)
    val inputLayernorm = RMSNorm(config.hiddenSize, config.rmsNormEps)
    val postAttentionLayernorm = RMSNorm(config.hiddenSize, config.rmsNormEps)
    
    // Linear projections for attention
    val qProj: GGMLTensor = createLinearWeight(config.hiddenSize, config.hiddenSize)
    val kProj: GGMLTensor = createLinearWeight(config.hiddenSize, config.hiddenSize) 
    val vProj: GGMLTensor = createLinearWeight(config.hiddenSize, config.hiddenSize)
    val oProj: GGMLTensor = createLinearWeight(config.hiddenSize, config.hiddenSize)
    
    private fun createLinearWeight(inputSize: Int, outputSize: Int): GGMLTensor {
        return GGMLTensor(type = GGMLType.F32).apply {
            ne[0] = inputSize.toLong()
            ne[1] = outputSize.toLong()
            for (i in 2 until GGML_MAX_DIMS) ne[i] = 1L
            nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
        }
    }
    
    fun forward(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        hiddenStates: GGMLTensor,
        attentionMask: GGMLTensor? = null,
        kvCache: KVCache? = null
    ): GGMLTensor {
        // Pre-attention layer norm
        val normedInput = inputLayernorm.forward(context, graphAllocator, hiddenStates)
        
        // Self-attention
        val query = linear(context, graphAllocator, normedInput, qProj)
        val key = linear(context, graphAllocator, normedInput, kProj)
        val value = linear(context, graphAllocator, normedInput, vProj)
        
        val attentionOutput = selfAttention.computeAttention(
            context, graphAllocator, query, key, value, attentionMask, kvCache
        )
        
        val attentionProjected = linear(context, graphAllocator, attentionOutput, oProj)
        
        // Add residual connection
        val afterAttention = add(context, graphAllocator, hiddenStates, attentionProjected)
        
        // Pre-MLP layer norm
        val normedAfterAttention = postAttentionLayernorm.forward(context, graphAllocator, afterAttention)
        
        // MLP
        val mlpOutput = mlp.forward(context, graphAllocator, normedAfterAttention)
        
        // Add residual connection
        return add(context, graphAllocator, afterAttention, mlpOutput)
    }
    
    private fun linear(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        input: GGMLTensor,
        weight: GGMLTensor
    ): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = weight.ne[1] // output dimension
        result.ne[1] = input.ne[1]  // sequence length
        result.ne[2] = input.ne[2]  // batch size
        for (i in 3 until GGML_MAX_DIMS) result.ne[i] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        result.op = GGMLOp.MUL_MAT
        result.src[0] = weight
        result.src[1] = input
        graphAllocator.allocateTensor(result)
        return result
    }
    
    private fun add(context: GGMLContext, graphAllocator: GGMLGraphAllocator, a: GGMLTensor, b: GGMLTensor): GGMLTensor {
        val result = GGMLTensor(type = a.type)
        result.ne = a.ne.copyOf()
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        result.op = GGMLOp.ADD
        result.src[0] = a
        result.src[1] = b
        graphAllocator.allocateTensor(result)
        return result
    }
}

/**
 * Complete LLaMA model.
 */
class LlamaModel(
    val config: LlamaConfig
) {
    // Token embedding table
    val embedTokens: GGMLTensor = GGMLTensor(type = GGMLType.F32).apply {
        ne[0] = config.hiddenSize.toLong()
        ne[1] = config.vocabSize.toLong()
        for (i in 2 until GGML_MAX_DIMS) ne[i] = 1L
        nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
    }
    
    // Transformer layers
    val layers = Array(config.numHiddenLayers) { i ->
        LlamaDecoderLayer(config, i)
    }
    
    // Final layer norm
    val norm = RMSNorm(config.hiddenSize, config.rmsNormEps)
    
    // Language modeling head (output projection)
    val lmHead: GGMLTensor = GGMLTensor(type = GGMLType.F32).apply {
        ne[0] = config.hiddenSize.toLong()
        ne[1] = config.vocabSize.toLong()
        for (i in 2 until GGML_MAX_DIMS) ne[i] = 1L
        nb = calculateContiguousStrides(ne, type, GGML_MAX_DIMS)
    }
    
    /**
     * Forward pass through the model.
     */
    fun forward(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        inputIds: IntArray,
        attentionMask: GGMLTensor? = null,
        kvCaches: Array<KVCache>? = null
    ): GGMLTensor {
        val batchSize = 1
        val seqLen = inputIds.size
        
        // Embed input tokens
        var hiddenStates = embedInput(context, graphAllocator, inputIds)
        
        // Pass through each transformer layer
        for (i in layers.indices) {
            val kvCache = kvCaches?.get(i)
            hiddenStates = layers[i].forward(
                context, graphAllocator, hiddenStates, attentionMask, kvCache
            )
        }
        
        // Final layer norm
        hiddenStates = norm.forward(context, graphAllocator, hiddenStates)
        
        // Language modeling head
        val logits = linear(context, graphAllocator, hiddenStates, lmHead)
        
        return logits
    }
    
    private fun embedInput(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        inputIds: IntArray
    ): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = config.hiddenSize.toLong()
        result.ne[1] = inputIds.size.toLong()
        result.ne[2] = 1L // batch size
        for (i in 3 until GGML_MAX_DIMS) result.ne[i] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        
        graphAllocator.allocateTensor(result)
        
        // Copy embeddings for each input token
        for (seqIdx in inputIds.indices) {
            val tokenId = inputIds[seqIdx]
            for (hiddenIdx in 0 until config.hiddenSize) {
                val embeddingValue = embedTokens.getFloat(graphAllocator, hiddenIdx, tokenId)
                result.setFloat(graphAllocator, embeddingValue, hiddenIdx, seqIdx, 0)
            }
        }
        
        return result
    }
    
    private fun linear(
        context: GGMLContext,
        graphAllocator: GGMLGraphAllocator,
        input: GGMLTensor,
        weight: GGMLTensor
    ): GGMLTensor {
        val result = GGMLTensor(type = GGMLType.F32)
        result.ne[0] = weight.ne[1] // output dimension (vocab size)
        result.ne[1] = input.ne[1]  // sequence length
        result.ne[2] = input.ne[2]  // batch size
        for (i in 3 until GGML_MAX_DIMS) result.ne[i] = 1L
        result.nb = calculateContiguousStrides(result.ne, result.type, GGML_MAX_DIMS)
        result.op = GGMLOp.MUL_MAT
        result.src[0] = weight
        result.src[1] = input
        graphAllocator.allocateTensor(result)
        return result
    }
    
    /**
     * Initialize model parameters (simplified).
     */
    fun initializeParameters(graphAllocator: GGMLGraphAllocator) {
        // Allocate tensors
        graphAllocator.allocateTensor(embedTokens)
        graphAllocator.allocateTensor(lmHead)
        graphAllocator.allocateTensor(norm.weight)
        
        for (layer in layers) {
            graphAllocator.allocateTensor(layer.qProj)
            graphAllocator.allocateTensor(layer.kProj)
            graphAllocator.allocateTensor(layer.vProj)
            graphAllocator.allocateTensor(layer.oProj)
            graphAllocator.allocateTensor(layer.inputLayernorm.weight)
            graphAllocator.allocateTensor(layer.postAttentionLayernorm.weight)
            graphAllocator.allocateTensor(layer.mlp.gateProj)
            graphAllocator.allocateTensor(layer.mlp.upProj)
            graphAllocator.allocateTensor(layer.mlp.downProj)
        }
        
        // Initialize with small random values (simplified)
        // In a real implementation, this would load from a model file
    }
}
