// port-lint: source llama.cpp/src/llama-chat.h llama.cpp/src/llama-chat.cpp
package io.github.kotlinmania.llama.model

import io.github.kotlinmania.llama.ore.*

// =============================================================================
// Chat template types and application
// Ported from: llama-chat.h / llama-chat.cpp
// =============================================================================

/**
 * Enumeration of all built-in chat templates.
 *
 * Port of `enum llm_chat_template` in `llama-chat.h`.
 */
enum class LlmChatTemplate {
    CHATML,
    LLAMA_2,
    LLAMA_2_SYS,
    LLAMA_2_SYS_BOS,
    LLAMA_2_SYS_STRIP,
    MISTRAL_V1,
    MISTRAL_V3,
    MISTRAL_V3_TEKKEN,
    MISTRAL_V7,
    MISTRAL_V7_TEKKEN,
    PHI_3,
    PHI_4,
    FALCON_3,
    ZEPHYR,
    MONARCH,
    GEMMA,
    ORION,
    OPENCHAT,
    VICUNA,
    VICUNA_ORCA,
    DEEPSEEK,
    DEEPSEEK_2,
    DEEPSEEK_3,
    DEEPSEEK_OCR,
    COMMAND_R,
    LLAMA_3,
    CHATGLM_3,
    CHATGLM_4,
    GLMEDGE,
    MINICPM,
    EXAONE_3,
    EXAONE_4,
    EXAONE_MOE,
    RWKV_WORLD,
    GRANITE_3_X,
    GRANITE_4_0,
    GIGACHAT,
    MEGREZ,
    YANDEX,
    BAILING,
    BAILING_THINK,
    BAILING2,
    LLAMA4,
    SMOLVLM,
    DOTS1,
    HUNYUAN_MOE,
    OPENAI_MOE,
    HUNYUAN_DENSE,
    HUNYUAN_OCR,
    KIMI_K2,
    SEED_OSS,
    GROK_2,
    PANGU_EMBED,
    SOLAR_OPEN,
    UNKNOWN;
}

/**
 * A single chat message with role and content.
 *
 * Port of `struct llama_chat_message` used in the C++ chat API.
 */
data class LlamaChatMessage(
    val role: String,
    val content: String,
)

// ---------------------------------------------------------------------------
// Template name ↔ enum mapping
// Ported from: LLM_CHAT_TEMPLATES map in llama-chat.cpp
// ---------------------------------------------------------------------------

/** Bidirectional map between template string names and [LlmChatTemplate] values. */
private val TEMPLATE_NAME_MAP: Map<String, LlmChatTemplate> = mapOf(
    "chatml"            to LlmChatTemplate.CHATML,
    "llama2"            to LlmChatTemplate.LLAMA_2,
    "llama2-sys"        to LlmChatTemplate.LLAMA_2_SYS,
    "llama2-sys-bos"    to LlmChatTemplate.LLAMA_2_SYS_BOS,
    "llama2-sys-strip"  to LlmChatTemplate.LLAMA_2_SYS_STRIP,
    "mistral-v1"        to LlmChatTemplate.MISTRAL_V1,
    "mistral-v3"        to LlmChatTemplate.MISTRAL_V3,
    "mistral-v3-tekken" to LlmChatTemplate.MISTRAL_V3_TEKKEN,
    "mistral-v7"        to LlmChatTemplate.MISTRAL_V7,
    "mistral-v7-tekken" to LlmChatTemplate.MISTRAL_V7_TEKKEN,
    "phi3"              to LlmChatTemplate.PHI_3,
    "phi4"              to LlmChatTemplate.PHI_4,
    "falcon3"           to LlmChatTemplate.FALCON_3,
    "zephyr"            to LlmChatTemplate.ZEPHYR,
    "monarch"           to LlmChatTemplate.MONARCH,
    "gemma"             to LlmChatTemplate.GEMMA,
    "orion"             to LlmChatTemplate.ORION,
    "openchat"          to LlmChatTemplate.OPENCHAT,
    "vicuna"            to LlmChatTemplate.VICUNA,
    "vicuna-orca"       to LlmChatTemplate.VICUNA_ORCA,
    "deepseek"          to LlmChatTemplate.DEEPSEEK,
    "deepseek2"         to LlmChatTemplate.DEEPSEEK_2,
    "deepseek3"         to LlmChatTemplate.DEEPSEEK_3,
    "deepseek-ocr"      to LlmChatTemplate.DEEPSEEK_OCR,
    "command-r"         to LlmChatTemplate.COMMAND_R,
    "llama3"            to LlmChatTemplate.LLAMA_3,
    "chatglm3"          to LlmChatTemplate.CHATGLM_3,
    "chatglm4"          to LlmChatTemplate.CHATGLM_4,
    "glmedge"           to LlmChatTemplate.GLMEDGE,
    "minicpm"           to LlmChatTemplate.MINICPM,
    "exaone3"           to LlmChatTemplate.EXAONE_3,
    "exaone4"           to LlmChatTemplate.EXAONE_4,
    "exaone-moe"        to LlmChatTemplate.EXAONE_MOE,
    "rwkv-world"        to LlmChatTemplate.RWKV_WORLD,
    "granite"           to LlmChatTemplate.GRANITE_3_X,
    "granite-4.0"       to LlmChatTemplate.GRANITE_4_0,
    "gigachat"          to LlmChatTemplate.GIGACHAT,
    "megrez"            to LlmChatTemplate.MEGREZ,
    "yandex"            to LlmChatTemplate.YANDEX,
    "bailing"           to LlmChatTemplate.BAILING,
    "bailing-think"     to LlmChatTemplate.BAILING_THINK,
    "bailing2"          to LlmChatTemplate.BAILING2,
    "llama4"            to LlmChatTemplate.LLAMA4,
    "smolvlm"           to LlmChatTemplate.SMOLVLM,
    "hunyuan-moe"       to LlmChatTemplate.HUNYUAN_MOE,
    "gpt-oss"           to LlmChatTemplate.OPENAI_MOE,
    "hunyuan-dense"     to LlmChatTemplate.HUNYUAN_DENSE,
    "hunyuan-ocr"       to LlmChatTemplate.HUNYUAN_OCR,
    "kimi-k2"           to LlmChatTemplate.KIMI_K2,
    "seed_oss"          to LlmChatTemplate.SEED_OSS,
    "grok-2"            to LlmChatTemplate.GROK_2,
    "pangu-embedded"    to LlmChatTemplate.PANGU_EMBED,
    "solar-open"        to LlmChatTemplate.SOLAR_OPEN,
)

/** Reverse map: enum → name string. */
private val TEMPLATE_ENUM_TO_NAME: Map<LlmChatTemplate, String> =
    TEMPLATE_NAME_MAP.entries.associate { (k, v) -> v to k }

// =============================================================================
// Public API
// =============================================================================

/**
 * Look up a [LlmChatTemplate] by its string name.
 *
 * Port of `llm_chat_template_from_str()`.
 *
 * @throws NoSuchElementException if [name] is not a recognized template.
 */
fun llmChatTemplateFromStr(name: String): LlmChatTemplate {
    return TEMPLATE_NAME_MAP[name]
        ?: throw NoSuchElementException("Unknown chat template: $name")
}

/**
 * Detect a [LlmChatTemplate] from a Jinja-style template string.
 *
 * The function first tries an exact name lookup, then falls back to
 * heuristic substring matching — the same approach used in the C++ source.
 *
 * Port of `llm_chat_detect_template()`.
 */
fun llmChatDetectTemplate(tmpl: String): LlmChatTemplate {
    // Try exact name match first
    TEMPLATE_NAME_MAP[tmpl]?.let { return it }

    fun contains(s: String) = tmpl.contains(s)

    if (contains("<|im_start|>")) {
        return when {
            contains("<|im_sep|>")          -> LlmChatTemplate.PHI_4
            contains("<end_of_utterance>")   -> LlmChatTemplate.SMOLVLM
            else                             -> LlmChatTemplate.CHATML
        }
    } else if (tmpl.startsWith("mistral") || contains("[INST]")) {
        if (contains("[SYSTEM_PROMPT]")) {
            return LlmChatTemplate.MISTRAL_V7
        } else if (contains("' [INST] ' + system_message") || contains("[AVAILABLE_TOOLS]")) {
            return when {
                contains(" [INST]")    -> LlmChatTemplate.MISTRAL_V1
                contains("\"[INST]\"") -> LlmChatTemplate.MISTRAL_V3_TEKKEN
                else                   -> LlmChatTemplate.MISTRAL_V3
            }
        } else {
            val supportSystemMessage = contains("<<SYS>>")
            val addBosInsideHistory  = contains("bos_token + '[INST]")
            val stripMessage         = contains("content.strip()")
            return when {
                stripMessage         -> LlmChatTemplate.LLAMA_2_SYS_STRIP
                addBosInsideHistory  -> LlmChatTemplate.LLAMA_2_SYS_BOS
                supportSystemMessage -> LlmChatTemplate.LLAMA_2_SYS
                else                 -> LlmChatTemplate.LLAMA_2
            }
        }
    } else if (contains("<|assistant|>") && contains("<|end|>")) {
        return LlmChatTemplate.PHI_3
    } else if (contains("[gMASK]<sop>")) {
        return LlmChatTemplate.CHATGLM_4
    } else if (contains("<|assistant|>") && contains("<|user|>")) {
        if (contains("<|tool_declare|>")) return LlmChatTemplate.EXAONE_MOE
        return if (contains("</s>")) LlmChatTemplate.FALCON_3 else LlmChatTemplate.GLMEDGE
    } else if (contains("<|{{ item['role'] }}|>") && contains("<|begin_of_image|>")) {
        return LlmChatTemplate.GLMEDGE
    } else if (contains("<|user|>") && contains("<|endoftext|>")) {
        return LlmChatTemplate.ZEPHYR
    } else if (contains("bos_token + message['role']")) {
        return LlmChatTemplate.MONARCH
    } else if (contains("<start_of_turn>")) {
        return LlmChatTemplate.GEMMA
    } else if (contains("'\\n\\nAssistant: ' + eos_token")) {
        return LlmChatTemplate.ORION
    } else if (contains("GPT4 Correct ")) {
        return LlmChatTemplate.OPENCHAT
    } else if (contains("USER: ") && contains("ASSISTANT: ")) {
        return if (contains("SYSTEM: ")) LlmChatTemplate.VICUNA_ORCA else LlmChatTemplate.VICUNA
    } else if (contains("### Instruction:") && contains("<|EOT|>")) {
        return LlmChatTemplate.DEEPSEEK
    } else if (contains("<|START_OF_TURN_TOKEN|>") && contains("<|USER_TOKEN|>")) {
        return LlmChatTemplate.COMMAND_R
    } else if (contains("<|start_header_id|>") && contains("<|end_header_id|>")) {
        return LlmChatTemplate.LLAMA_3
    } else if (contains("[gMASK]sop")) {
        return LlmChatTemplate.CHATGLM_3
    } else if (contains("\u7528\u6237>")) { // <用户>
        return LlmChatTemplate.MINICPM
    } else if (contains("'Assistant: ' + message['content'] + eos_token")) {
        return LlmChatTemplate.DEEPSEEK_2
    } else if (contains("\uff5cAssistant\uff5c") && contains("\uff5cUser\uff5c") && contains("\uff5cend\u2581of\u2581sentence\uff5c")) {
        return LlmChatTemplate.DEEPSEEK_3
    } else if (contains("[|system|]") && contains("[|assistant|]") && contains("[|endofturn|]")) {
        return if (contains("[|tool|]")) LlmChatTemplate.EXAONE_4 else LlmChatTemplate.EXAONE_3
    } else if (contains("rwkv-world") || contains("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}")) {
        return LlmChatTemplate.RWKV_WORLD
    } else if (contains("<|start_of_role|>")) {
        return if (contains("<tool_call>") || contains("<tools>")) LlmChatTemplate.GRANITE_4_0
               else LlmChatTemplate.GRANITE_3_X
    } else if (contains("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]")) {
        return LlmChatTemplate.GIGACHAT
    } else if (contains("<|role_start|>")) {
        return LlmChatTemplate.MEGREZ
    } else if (contains(" \u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043D\u0442:")) { // Ассистент:
        return LlmChatTemplate.YANDEX
    } else if (contains("<role>ASSISTANT</role>") && contains("'HUMAN'")) {
        return LlmChatTemplate.BAILING
    } else if (contains("<role>ASSISTANT</role>") && contains("\"HUMAN\"") && contains("<think>")) {
        return LlmChatTemplate.BAILING_THINK
    } else if (contains("<role>ASSISTANT</role>") && contains("<role>HUMAN</role>") && contains("<|role_end|>")) {
        return LlmChatTemplate.BAILING2
    } else if (contains("<|header_start|>") && contains("<|header_end|>")) {
        return LlmChatTemplate.LLAMA4
    } else if (contains("<|endofuserprompt|>")) {
        return LlmChatTemplate.DOTS1
    } else if (contains("<|extra_0|>") && contains("<|extra_4|>")) {
        return LlmChatTemplate.HUNYUAN_MOE
    } else if (contains("<|start|>") && contains("<|channel|>")) {
        return LlmChatTemplate.OPENAI_MOE
    } else if (contains("\uff5chy_Assistant\uff5c") && contains("\uff5chy_begin\u2581of\u2581sentence\uff5c")) {
        return LlmChatTemplate.HUNYUAN_OCR
    } else if (contains("\uff5chy_Assistant\uff5c") && contains("\uff5chy_place\u2581holder\u2581no\u2581\u0033\uff5c")) {
        return LlmChatTemplate.HUNYUAN_DENSE
    } else if (contains("<|im_assistant|>assistant<|im_middle|>")) {
        return LlmChatTemplate.KIMI_K2
    } else if (contains("<seed:bos>")) {
        return LlmChatTemplate.SEED_OSS
    } else if (contains("'Assistant: '  + message['content'] + '<|separator|>")) {
        return LlmChatTemplate.GROK_2
    } else if (contains("[unused9]\u7CFB\u7EDF\uFF1A[unused10]")) { // 系统：
        return LlmChatTemplate.PANGU_EMBED
    } else if (contains("<|begin|>") && contains("<|end|>") && contains("<|content|>")) {
        return LlmChatTemplate.SOLAR_OPEN
    }
    return LlmChatTemplate.UNKNOWN
}

/**
 * Return the list of all built-in template names.
 *
 * Port of `llama_chat_builtin_templates()`.
 */
fun llamaChatBuiltinTemplates(): List<String> = TEMPLATE_NAME_MAP.keys.toList()

/**
 * Apply a chat template to a list of messages, producing a formatted prompt string.
 *
 * Port of `llm_chat_apply_template()` in `llama-chat.cpp`.
 *
 * @param tmpl   The template to apply.
 * @param chat   Ordered list of chat messages.
 * @param addAss If `true`, append the assistant prompt prefix at the end.
 * @return The formatted prompt string, or `null` if the template is unsupported.
 */
fun llmChatApplyTemplate(
    tmpl: LlmChatTemplate,
    chat: List<LlamaChatMessage>,
    addAss: Boolean,
): String? {
    val ss = StringBuilder()

    when (tmpl) {
        LlmChatTemplate.CHATML -> {
            for (msg in chat) {
                ss.append("<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n")
            }
            if (addAss) ss.append("<|im_start|>assistant\n")
        }

        LlmChatTemplate.MISTRAL_V7,
        LlmChatTemplate.MISTRAL_V7_TEKKEN -> {
            val trailingSpace = if (tmpl == LlmChatTemplate.MISTRAL_V7) " " else ""
            for (msg in chat) {
                when (msg.role) {
                    "system" -> ss.append("[SYSTEM_PROMPT]${trailingSpace}${msg.content}[/SYSTEM_PROMPT]")
                    "user"   -> ss.append("[INST]${trailingSpace}${msg.content}[/INST]")
                    else     -> ss.append("${trailingSpace}${msg.content}</s>")
                }
            }
        }

        LlmChatTemplate.MISTRAL_V1,
        LlmChatTemplate.MISTRAL_V3,
        LlmChatTemplate.MISTRAL_V3_TEKKEN -> {
            val leadingSpace = if (tmpl == LlmChatTemplate.MISTRAL_V1) " " else ""
            val trailingSpace = if (tmpl == LlmChatTemplate.MISTRAL_V3_TEKKEN) "" else " "
            val trimAssistant = tmpl == LlmChatTemplate.MISTRAL_V3
            var insideTurn = false
            for (msg in chat) {
                if (!insideTurn) {
                    ss.append("${leadingSpace}[INST]${trailingSpace}")
                    insideTurn = true
                }
                when (msg.role) {
                    "system" -> ss.append("${msg.content}\n\n")
                    "user"   -> ss.append("${msg.content}${leadingSpace}[/INST]")
                    else -> {
                        val content = if (trimAssistant) msg.content.trim() else msg.content
                        ss.append("${trailingSpace}${content}</s>")
                        insideTurn = false
                    }
                }
            }
        }

        LlmChatTemplate.LLAMA_2,
        LlmChatTemplate.LLAMA_2_SYS,
        LlmChatTemplate.LLAMA_2_SYS_BOS,
        LlmChatTemplate.LLAMA_2_SYS_STRIP -> {
            val supportSystem = tmpl != LlmChatTemplate.LLAMA_2
            val addBosInside = tmpl == LlmChatTemplate.LLAMA_2_SYS_BOS
            val stripMsg = tmpl == LlmChatTemplate.LLAMA_2_SYS_STRIP
            var insideTurn = true // skip BOS at beginning
            ss.append("[INST] ")
            for (msg in chat) {
                val content = if (stripMsg) msg.content.trim() else msg.content
                if (!insideTurn) {
                    insideTurn = true
                    ss.append(if (addBosInside) "<s>[INST] " else "[INST] ")
                }
                when (msg.role) {
                    "system" -> {
                        if (supportSystem) {
                            ss.append("<<SYS>>\n${content}\n<</SYS>>\n\n")
                        } else {
                            ss.append("${content}\n")
                        }
                    }
                    "user" -> ss.append("${content} [/INST]")
                    else -> {
                        ss.append("${content}</s>")
                        insideTurn = false
                    }
                }
            }
        }

        LlmChatTemplate.PHI_3 -> {
            for (msg in chat) {
                ss.append("<|${msg.role}|>\n${msg.content}<|end|>\n")
            }
            if (addAss) ss.append("<|assistant|>\n")
        }

        LlmChatTemplate.PHI_4 -> {
            for (msg in chat) {
                ss.append("<|im_start|>${msg.role}<|im_sep|>${msg.content}<|im_end|>")
            }
            if (addAss) ss.append("<|im_start|>assistant<|im_sep|>")
        }

        LlmChatTemplate.FALCON_3 -> {
            for (msg in chat) {
                ss.append("<|${msg.role}|>\n${msg.content}\n")
            }
            if (addAss) ss.append("<|assistant|>\n")
        }

        LlmChatTemplate.ZEPHYR -> {
            for (msg in chat) {
                ss.append("<|${msg.role}|>\n${msg.content}<|endoftext|>\n")
            }
            if (addAss) ss.append("<|assistant|>\n")
        }

        LlmChatTemplate.MONARCH -> {
            for ((i, msg) in chat.withIndex()) {
                val bos = if (i == 0) "" else "<s>"
                ss.append("${bos}${msg.role}\n${msg.content}</s>\n")
            }
            if (addAss) ss.append("<s>assistant\n")
        }

        LlmChatTemplate.GEMMA -> {
            var systemPrompt = ""
            for (msg in chat) {
                if (msg.role == "system") {
                    systemPrompt += msg.content.trim()
                    continue
                }
                val role = if (msg.role == "assistant") "model" else msg.role
                ss.append("<start_of_turn>${role}\n")
                if (systemPrompt.isNotEmpty() && role != "model") {
                    ss.append("${systemPrompt}\n\n")
                    systemPrompt = ""
                }
                ss.append("${msg.content.trim()}<end_of_turn>\n")
            }
            if (addAss) ss.append("<start_of_turn>model\n")
        }

        LlmChatTemplate.ORION -> {
            var systemPrompt = ""
            for (msg in chat) {
                when (msg.role) {
                    "system" -> systemPrompt += msg.content
                    "user" -> {
                        ss.append("Human: ")
                        if (systemPrompt.isNotEmpty()) {
                            ss.append("${systemPrompt}\n\n")
                            systemPrompt = ""
                        }
                        ss.append("${msg.content}\n\nAssistant: </s>")
                    }
                    else -> ss.append("${msg.content}</s>")
                }
            }
        }

        LlmChatTemplate.OPENCHAT -> {
            for (msg in chat) {
                if (msg.role == "system") {
                    ss.append("${msg.content}<|end_of_turn|>")
                } else {
                    val role = msg.role.replaceFirstChar { it.uppercase() }
                    ss.append("GPT4 Correct ${role}: ${msg.content}<|end_of_turn|>")
                }
            }
            if (addAss) ss.append("GPT4 Correct Assistant:")
        }

        LlmChatTemplate.VICUNA,
        LlmChatTemplate.VICUNA_ORCA -> {
            for (msg in chat) {
                when (msg.role) {
                    "system" -> {
                        if (tmpl == LlmChatTemplate.VICUNA_ORCA) {
                            ss.append("SYSTEM: ${msg.content}\n")
                        } else {
                            ss.append("${msg.content}\n\n")
                        }
                    }
                    "user"      -> ss.append("USER: ${msg.content}\n")
                    "assistant" -> ss.append("ASSISTANT: ${msg.content}</s>\n")
                }
            }
            if (addAss) ss.append("ASSISTANT:")
        }

        LlmChatTemplate.DEEPSEEK -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append(msg.content)
                    "user"      -> ss.append("### Instruction:\n${msg.content}\n")
                    "assistant" -> ss.append("### Response:\n${msg.content}\n<|EOT|>\n")
                }
            }
            if (addAss) ss.append("### Response:\n")
        }

        LlmChatTemplate.COMMAND_R -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>${msg.content.trim()}<|END_OF_TURN_TOKEN|>")
                    "user"      -> ss.append("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>${msg.content.trim()}<|END_OF_TURN_TOKEN|>")
                    "assistant" -> ss.append("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>${msg.content.trim()}<|END_OF_TURN_TOKEN|>")
                }
            }
            if (addAss) ss.append("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
        }

        LlmChatTemplate.LLAMA_3 -> {
            for (msg in chat) {
                ss.append("<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content.trim()}<|eot_id|>")
            }
            if (addAss) ss.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        }

        LlmChatTemplate.CHATGLM_3 -> {
            ss.append("[gMASK]sop")
            for (msg in chat) {
                ss.append("<|${msg.role}|>\n ${msg.content}")
            }
            if (addAss) ss.append("<|assistant|>")
        }

        LlmChatTemplate.CHATGLM_4 -> {
            ss.append("[gMASK]<sop>")
            for (msg in chat) {
                ss.append("<|${msg.role}|>\n${msg.content}")
            }
            if (addAss) ss.append("<|assistant|>\n")
        }

        LlmChatTemplate.GLMEDGE -> {
            for (msg in chat) {
                ss.append("<|${msg.role}|>\n${msg.content}")
            }
            if (addAss) ss.append("<|assistant|>")
        }

        LlmChatTemplate.MINICPM -> {
            for (msg in chat) {
                if (msg.role == "user") {
                    ss.append("<\u7528\u6237>${msg.content.trim()}<AI>")
                } else {
                    ss.append(msg.content.trim())
                }
            }
        }

        LlmChatTemplate.DEEPSEEK_2 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("${msg.content}\n\n")
                    "user"      -> ss.append("User: ${msg.content}\n\n")
                    "assistant" -> ss.append("Assistant: ${msg.content}\uff5cend\u2581of\u2581sentence\uff5c")
                }
            }
            if (addAss) ss.append("Assistant:")
        }

        LlmChatTemplate.DEEPSEEK_3 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("${msg.content}\n\n")
                    "user"      -> ss.append("\uff5cUser\uff5c${msg.content}")
                    "assistant" -> ss.append("\uff5cAssistant\uff5c${msg.content}\uff5cend\u2581of\u2581sentence\uff5c")
                }
            }
            if (addAss) ss.append("\uff5cAssistant\uff5c")
        }

        LlmChatTemplate.DEEPSEEK_OCR -> {
            for (msg in chat) {
                ss.append(msg.content)
            }
        }

        LlmChatTemplate.EXAONE_3 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("[|system|]${msg.content.trim()}[|endofturn|]\n")
                    "user"      -> ss.append("[|user|]${msg.content.trim()}\n")
                    "assistant" -> ss.append("[|assistant|]${msg.content.trim()}[|endofturn|]\n")
                }
            }
            if (addAss) ss.append("[|assistant|]")
        }

        LlmChatTemplate.EXAONE_4 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("[|system|]${msg.content.trim()}[|endofturn|]\n")
                    "user"      -> ss.append("[|user|]${msg.content.trim()}\n")
                    "assistant" -> ss.append("[|assistant|]${msg.content.trim()}[|endofturn|]\n")
                    "tool"      -> ss.append("[|tool|]${msg.content.trim()}[|endofturn|]\n")
                }
            }
            if (addAss) ss.append("[|assistant|]")
        }

        LlmChatTemplate.EXAONE_MOE -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("<|system|>\n${msg.content.trim()}<|endofturn|>\n")
                    "user"      -> ss.append("<|user|>\n${msg.content.trim()}<|endofturn|>\n")
                    "assistant" -> ss.append("<|assistant|>\n${msg.content.trim()}<|endofturn|>\n")
                    "tool"      -> ss.append("<|tool|>\n${msg.content.trim()}<|endofturn|>\n")
                }
            }
            if (addAss) ss.append("<|assistant|>\n")
        }

        LlmChatTemplate.RWKV_WORLD -> {
            for ((i, msg) in chat.withIndex()) {
                when (msg.role) {
                    "system"    -> ss.append("System: ${msg.content.trim()}\n\n")
                    "user" -> {
                        ss.append("User: ${msg.content.trim()}\n\n")
                        if (i == chat.size - 1) ss.append("Assistant:")
                    }
                    "assistant" -> ss.append("Assistant: ${msg.content.trim()}\n\n")
                }
            }
        }

        LlmChatTemplate.GRANITE_3_X -> {
            for (msg in chat) {
                ss.append("<|start_of_role|>${msg.role}<|end_of_role|>")
                if (msg.role == "assistant_tool_call") ss.append("<|tool_call|>")
                ss.append("${msg.content}<|end_of_text|>\n")
            }
            if (addAss) ss.append("<|start_of_role|>assistant<|end_of_role|>")
        }

        LlmChatTemplate.GRANITE_4_0 -> {
            for (msg in chat) {
                if (msg.role == "assistant_tool_call") {
                    ss.append("<|start_of_role|>assistant<|end_of_role|><|tool_call|>")
                } else {
                    ss.append("<|start_of_role|>${msg.role}<|end_of_role|>")
                }
                ss.append("${msg.content}<|end_of_text|>\n")
            }
            if (addAss) ss.append("<|start_of_role|>assistant<|end_of_role|>")
        }

        LlmChatTemplate.GIGACHAT -> {
            val hasSystem = chat.isNotEmpty() && chat[0].role == "system"
            if (hasSystem) {
                ss.append("<s>${chat[0].content}<|message_sep|>")
            } else {
                ss.append("<s>")
            }
            val startIdx = if (hasSystem) 1 else 0
            for (i in startIdx until chat.size) {
                when (chat[i].role) {
                    "user" -> {
                        ss.append("user<|role_sep|>${chat[i].content}<|message_sep|>")
                        ss.append("available functions<|role_sep|>[]<|message_sep|>")
                    }
                    "assistant" -> ss.append("assistant<|role_sep|>${chat[i].content}<|message_sep|>")
                }
            }
            if (addAss) ss.append("assistant<|role_sep|>")
        }

        LlmChatTemplate.MEGREZ -> {
            for (msg in chat) {
                ss.append("<|role_start|>${msg.role}<|role_end|>${msg.content}<|turn_end|>")
            }
            if (addAss) ss.append("<|role_start|>assistant<|role_end|>")
        }

        LlmChatTemplate.YANDEX -> {
            for (msg in chat) {
                when (msg.role) {
                    "user"      -> ss.append(" \u041F\u043E\u043B\u044C\u0437\u043E\u0432\u0430\u0442\u0435\u043B\u044C: ${msg.content}\n\n")
                    "assistant" -> ss.append(" \u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043D\u0442: ${msg.content}\n\n")
                }
            }
            if (addAss) ss.append(" \u0410\u0441\u0441\u0438\u0441\u0442\u0435\u043D\u0442:[SEP]")
        }

        LlmChatTemplate.BAILING,
        LlmChatTemplate.BAILING_THINK -> {
            for (msg in chat) {
                val role = if (msg.role == "user") "HUMAN" else msg.role.uppercase()
                ss.append("<role>${role}</role>${msg.content}")
            }
            if (addAss) {
                ss.append("<role>ASSISTANT</role>")
                if (tmpl == LlmChatTemplate.BAILING_THINK) ss.append("<think>")
            }
        }

        LlmChatTemplate.BAILING2 -> {
            val hasSystem = chat.isNotEmpty() && chat[0].role == "system"
            if (!hasSystem) {
                ss.append("<role>SYSTEM</role>detailed thinking off<|role_end|>")
            }
            for (msg in chat) {
                val role = if (msg.role == "user") "HUMAN" else msg.role.uppercase()
                ss.append("<role>${role}</role>${msg.content}<|role_end|>")
            }
            if (addAss) ss.append("<role>ASSISTANT</role>")
        }

        LlmChatTemplate.LLAMA4 -> {
            for (msg in chat) {
                ss.append("<|header_start|>${msg.role}<|header_end|>\n\n${msg.content.trim()}<|eot|>")
            }
            if (addAss) ss.append("<|header_start|>assistant<|header_end|>\n\n")
        }

        LlmChatTemplate.SMOLVLM -> {
            ss.append("<|im_start|>")
            for (msg in chat) {
                when (msg.role) {
                    "system" -> ss.append("${msg.content}\n\n")
                    "user"   -> ss.append("User: ${msg.content}<end_of_utterance>\n")
                    else     -> ss.append("Assistant: ${msg.content}<end_of_utterance>\n")
                }
            }
            if (addAss) ss.append("Assistant:")
        }

        LlmChatTemplate.DOTS1 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system" -> ss.append("<|system|>${msg.content}<|endofsystem|>")
                    "user"   -> ss.append("<|userprompt|>${msg.content}<|endofuserprompt|>")
                    else     -> ss.append("<|response|>${msg.content}<|endofresponse|>")
                }
            }
            if (addAss) ss.append("<|response|>")
        }

        LlmChatTemplate.HUNYUAN_MOE -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("<|startoftext|>${msg.content}<|extra_4|>")
                    "assistant" -> ss.append("${msg.content}<|eos|>")
                    else        -> ss.append("<|startoftext|>${msg.content}<|extra_0|>")
                }
            }
        }

        LlmChatTemplate.OPENAI_MOE -> {
            for (msg in chat) {
                ss.append("<|start|>${msg.role}<|message|>${msg.content}")
                ss.append(if (msg.role == "assistant") "<|return|>" else "<|end|>")
            }
            if (addAss) ss.append("<|start|>assistant")
        }

        LlmChatTemplate.HUNYUAN_DENSE -> {
            for ((i, msg) in chat.withIndex()) {
                if (i == 0 && msg.role == "system") {
                    ss.append("${msg.content}<\uff5chy_place\u2581holder\u2581no\u2581\u0033\uff5c>")
                }
                when (msg.role) {
                    "assistant" -> ss.append("<\uff5chy_Assistant\uff5c>${msg.content}<\uff5chy_place\u2581holder\u2581no\u2581\u0032\uff5c>")
                    "user"      -> ss.append("<\uff5chy_User\uff5c>${msg.content}<\uff5chy_Assistant\uff5c>")
                }
            }
        }

        LlmChatTemplate.HUNYUAN_OCR -> {
            ss.append("<\uff5chy_begin\u2581of\u2581sentence\uff5c>")
            for ((i, msg) in chat.withIndex()) {
                if (i == 0 && msg.role == "system") {
                    ss.append("${msg.content}<\uff5chy_place\u2581holder\u2581no\u2581\u0033\uff5c>")
                    continue
                }
                when (msg.role) {
                    "user"      -> ss.append("${msg.content}<\uff5chy_User\uff5c>")
                    "assistant" -> ss.append("${msg.content}<\uff5chy_Assistant\uff5c>")
                }
            }
        }

        LlmChatTemplate.KIMI_K2 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("<|im_system|>system<|im_middle|>")
                    "user"      -> ss.append("<|im_user|>user<|im_middle|>")
                    "assistant" -> ss.append("<|im_assistant|>assistant<|im_middle|>")
                    "tool"      -> ss.append("<|im_system|>tool<|im_middle|>")
                }
                ss.append("${msg.content}<|im_end|>")
            }
            if (addAss) ss.append("<|im_assistant|>assistant<|im_middle|>")
        }

        LlmChatTemplate.SEED_OSS -> {
            for (msg in chat) {
                val content = if (msg.role == "assistant") msg.content.trim() else msg.content
                ss.append("<seed:bos>${msg.role}\n${content}<seed:eos>")
            }
            if (addAss) ss.append("<seed:bos>assistant\n")
        }

        LlmChatTemplate.GROK_2 -> {
            for (msg in chat) {
                when (msg.role) {
                    "system"    -> ss.append("System: ${msg.content.trim()}<|separator|>\n\n")
                    "user"      -> ss.append("Human: ${msg.content.trim()}<|separator|>\n\n")
                    "assistant" -> ss.append("Assistant: ${msg.content}<|separator|>\n\n")
                }
            }
            if (addAss) ss.append("Assistant:")
        }

        LlmChatTemplate.PANGU_EMBED -> {
            for ((i, msg) in chat.withIndex()) {
                if (i == 0 && msg.role != "system") {
                    ss.append("[unused9]\u7CFB\u7EDF\uFF1A[unused10]")
                }
                when (msg.role) {
                    "system"   -> ss.append("[unused9]\u7CFB\u7EDF\uFF1A${msg.content}[unused10]")
                    "user"     -> ss.append("[unused9]\u7528\u6237\uFF1A${msg.content}[unused10]")
                    "assistant"-> ss.append("[unused9]\u52A9\u624B\uFF1A${msg.content}[unused10]")
                    "tool"     -> ss.append("[unused9]\u5DE5\u5177\uFF1A${msg.content}[unused10]")
                    "function" -> ss.append("[unused9]\u65B9\u6CD5\uFF1A${msg.content}[unused10]")
                }
            }
            if (addAss) ss.append("[unused9]\u52A9\u624B\uFF1A")
        }

        LlmChatTemplate.SOLAR_OPEN -> {
            for (msg in chat) {
                ss.append("<|begin|>${msg.role}<|content|>${msg.content}<|end|>")
            }
            if (addAss) ss.append("<|begin|>assistant")
        }

        LlmChatTemplate.UNKNOWN -> return null
    }

    return ss.toString()
}
