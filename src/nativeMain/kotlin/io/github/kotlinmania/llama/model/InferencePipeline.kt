// port-lint: source llama.cpp/src/models/models.h
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*


open class LlmBuildMambaBase(params: LlmGraphParams) : LlmGraphContext(params) {
    open fun buildMambaLayer(inp: LlmGraphInputRs, cur: GGMLTensor, model: LlamaModelData, ubatch: LlamaUBatch, il: Int): GGMLTensor = error("not yet ported")
    open fun buildMamba2Layer(inp: LlmGraphInputRs, cur: GGMLTensor, model: LlamaModelData, ubatch: LlamaUBatch, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildDeltaNetBase(params: LlmGraphParams) : LlmGraphContext(params) {
    open fun buildDeltaNetChunking(q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, g: GGMLTensor, b: GGMLTensor, s: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
    open fun buildDeltaNetAutoregressive(q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, g: GGMLTensor, b: GGMLTensor, s: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
    open fun buildDeltaNetFused(q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, g: GGMLTensor, b: GGMLTensor, s: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
    open fun buildDeltaNet(q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, g: GGMLTensor, b: GGMLTensor, s: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
}

open class LlmBuildRwkv6Base(open val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params) {
    open fun buildRwkv6ChannelMix(layer: Any, cur: GGMLTensor, xPrev: GGMLTensor, arch: Any): GGMLTensor = error("not yet ported")
    open fun buildRwkv6TimeMix(inp: LlmGraphInputRs, cur: GGMLTensor, xPrev: GGMLTensor, ubatch: LlamaUBatch, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildRwkv7Base(open val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params) {
    open fun buildRwkv7ChannelMix(layer: Any, cur: GGMLTensor, xPrev: GGMLTensor, arch: Any): GGMLTensor = error("not yet ported")
    open fun buildRwkv7TimeMix(inp: LlmGraphInputRs, cur: GGMLTensor, xPrev: GGMLTensor, firstLayerValue: GGMLTensor, ubatch: LlamaUBatch, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildAfmoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildApertus(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildArcee(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildArctic(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildArwkv7(override val model: LlamaModelData, params: LlmGraphParams) : LlmBuildRwkv7Base(model, params)

open class LlmBuildBaichuan(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildBailingmoe2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildBailingmoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildBert(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildBitnet(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildBloom(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildChameleon(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildChatglm(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildCodeshell(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildCogvlm(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildCohere2Iswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildCommandR(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildDbrx(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildDeci(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildDeepseek2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildDeepseek(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildDots1(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildDream(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildErnie45(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildErnie45Moe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPaddleocr(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildExaone4(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildExaone(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildExaoneMoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildFalcon(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildFalconH1(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildMambaBase(params)

open class LlmBuildGemma2Iswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGemma3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGemma3nIswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params) {
    val nEmbdHead: Long = 0L
    val nEmbdAltup: Long = 0L
    val nAltup: Long = 0L
    val iAltupAct: Int = 0
    val nLayerSparsity: Int = 10
    val fSparsityStdMul: Float = 1.6448533535003662f
    open fun calcMagnitude(x: GGMLTensor): GGMLTensor = error("not yet ported")
    open fun buildInpPerLayer(): GGMLTensor = error("not yet ported")
    open fun projectPerLayerInputs(inpBatch: GGMLTensor, inpPerLayer: GGMLTensor): GGMLTensor = error("not yet ported")
    open fun gaussianTopk(x: GGMLTensor): GGMLTensor = error("not yet ported")
    open fun altupComputeRouterModalities(x: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun altupPredict(cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun laurel(cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun altupCorrect(predictions: GGMLTensor, activated: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildGemma4Iswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params) {
    val nEmbdPerLayer: Long = 0L
    open fun buildInpPerLayer(): GGMLTensor = error("not yet ported")
    open fun projectPerLayerInputs(inpBatch: GGMLTensor, inpPerLayer: GGMLTensor): GGMLTensor = error("not yet ported")
}

open class LlmBuildGemmaEmbedding(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGemma(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGlm4(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGlm4Moe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGpt2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGptneox(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGranite(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params) {
    open fun buildAttentionLayer(cur: GGMLTensor, inpPos: GGMLTensor, inpAttn: LlmGraphInputAttnKv, model: LlamaModelData, nEmbdHead: Long, il: Int): GGMLTensor = error("not yet ported")
    open fun buildLayerFfn(cur: GGMLTensor, inpSA: GGMLTensor, model: LlamaModelData, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildGraniteHybrid(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildMambaBase(params) {
    open fun buildLayerFfn(cur: GGMLTensor, inpSA: GGMLTensor, model: LlamaModelData, il: Int): GGMLTensor = error("not yet ported")
    open fun buildAttentionLayer(cur: GGMLTensor, inpPos: GGMLTensor, inpAttn: LlmGraphInputAttnKv, model: LlamaModelData, nEmbdHead: Long, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildGrok(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildGrovemoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildHunyuanDense(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildHunyuanMoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildInternlm2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildJais(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildJais2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildJamba(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildMambaBase(params)

open class LlmBuildKimiLinear(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildDeltaNetBase(params) {
    open fun buildKdaAutoregressive(q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, gk: GGMLTensor, beta: GGMLTensor, state: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
    open fun buildKdaChunking(q: GGMLTensor, k: GGMLTensor, v: GGMLTensor, gk: GGMLTensor, beta: GGMLTensor, state: GGMLTensor, causalMask: GGMLTensor, identity: GGMLTensor, diagMask: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
}

open class LlmBuildLfm2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildLlada(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildLladaMoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildLlama(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildLlama4(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildMaincoder(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildMamba(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildMambaBase(params)

open class LlmBuildMimo2Iswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildMinicpm3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildMinimaxM2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildMistral3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildModernBert(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildMpt(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildNemotron(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildNemotronH(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildMambaBase(params) {
    open fun buildFfnLayer(cur: GGMLTensor, model: LlamaModelData, il: Int): GGMLTensor = error("not yet ported")
    open fun buildAttentionLayer(cur: GGMLTensor, inpAttn: LlmGraphInputAttnKv, model: LlamaModelData, nEmbdHead: Long, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildNeoBert(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildEurobert(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildOlmo2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildOlmoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildOlmo(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildOpenaiMoeIswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildOpenelm(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildOrion(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPanguEmbedded(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPhi2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPhi3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPlamo2(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildMambaBase(params) {
    open fun buildPlamo2AttnLayer(inp: LlmGraphInputAttnKv, inpPos: GGMLTensor, cur: GGMLTensor, model: LlamaModelData, il: Int): GGMLTensor = error("not yet ported")
}

open class LlmBuildPlamo(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPlamo3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildPlm(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen2moe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen2vl(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen3moe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen3vl(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen3vlmoe(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildQwen3next(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildDeltaNetBase(params) {
    open fun buildLayerAttnLinear(inp: LlmGraphInputRs, cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun buildLayerFfn(cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun buildNormGated(input: GGMLTensor, weights: GGMLTensor, gate: GGMLTensor, layer: Int): GGMLTensor = error("not yet ported")
    open fun buildQkvz(input: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
}

open class LlmBuildQwen35(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildDeltaNetBase(params) {
    open fun buildLayerAttnLinear(inp: LlmGraphInputRs, cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun buildLayerFfn(cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun buildNormGated(input: GGMLTensor, weights: GGMLTensor, gate: GGMLTensor, layer: Int): GGMLTensor = error("not yet ported")
    open fun buildQkvz(input: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
}

open class LlmBuildQwen35moe(val model: LlamaModelData, params: LlmGraphParams) : LlmBuildDeltaNetBase(params) {
    open fun buildLayerAttnLinear(inp: LlmGraphInputRs, cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun buildLayerFfn(cur: GGMLTensor, il: Int): GGMLTensor = error("not yet ported")
    open fun buildNormGated(input: GGMLTensor, weights: GGMLTensor, gate: GGMLTensor, layer: Int): GGMLTensor = error("not yet ported")
    open fun buildQkvz(input: GGMLTensor, il: Int): Pair<GGMLTensor, GGMLTensor> = error("not yet ported")
}

open class LlmBuildQwen(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildRefact(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildRnd1(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildRwkv6(override val model: LlamaModelData, params: LlmGraphParams) : LlmBuildRwkv6Base(model, params)

open class LlmBuildRwkv6qwen2(override val model: LlamaModelData, params: LlmGraphParams) : LlmBuildRwkv6Base(model, params)

open class LlmBuildRwkv7(override val model: LlamaModelData, params: LlmGraphParams) : LlmBuildRwkv7Base(model, params)

open class LlmBuildSeedOss(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildSmallthinker(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildSmollm3(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildStablelm(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildStarcoder2(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildStarcoder(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildStep35Iswa(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildT5(open val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildT5encoder(override val model: LlamaModelData, params: LlmGraphParams) : LlmBuildT5(model, params)

open class LlmBuildWavtokenizerDec(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)

open class LlmBuildXverse(val model: LlamaModelData, params: LlmGraphParams) : LlmGraphContext(params)
