package ai.solace.llamakotlin.model

/**
 * Grammar-based constrained generation for LLaMA models.
 * Implements GBNF (Grammar-Based Next-token Filtering) for structured output.
 */

/**
 * Grammar element types based on GBNF specification.
 */
enum class GrammarElementType {
    END,          // End of rule
    ALT,          // Alternative (|)
    RULE_REF,     // Reference to another rule
    CHAR,         // Single character
    CHAR_NOT,     // Character not in set
    CHAR_RNG_UPPER, // Character range upper bound
    CHAR_ALT      // Character alternative
}

/**
 * A single element in a grammar rule.
 */
data class GrammarElement(
    val type: GrammarElementType,
    val value: Int = 0 // Character code or rule reference
)

/**
 * A complete grammar rule.
 */
data class GrammarRule(
    val elements: List<GrammarElement>
)

/**
 * State in the grammar parsing state machine.
 */
data class GrammarState(
    val ruleId: Int,
    val position: Int
)

/**
 * GBNF grammar parser and constraint engine.
 */
class GBNFGrammar(
    val rules: List<GrammarRule>,
    val rootRuleId: Int = 0
) {
    
    /**
     * Parse a GBNF grammar string into rules.
     */
    companion object {
        fun parse(grammarText: String): GBNFGrammar {
            val lines = grammarText.lines().map { it.trim() }.filter { it.isNotEmpty() && !it.startsWith("#") }
            val rules = mutableListOf<GrammarRule>()
            val ruleNames = mutableMapOf<String, Int>()
            
            var rootRuleId = 0
            
            for (line in lines) {
                if (!line.contains("::=")) continue
                
                val parts = line.split("::=", limit = 2)
                if (parts.size != 2) continue
                
                val ruleName = parts[0].trim()
                val ruleBody = parts[1].trim()
                
                // Assign rule ID
                val ruleId = rules.size
                ruleNames[ruleName] = ruleId
                
                if (rules.isEmpty()) {
                    rootRuleId = ruleId // First rule is root
                }
                
                // Parse rule body (simplified - full GBNF parsing is complex)
                val elements = parseRuleBody(ruleBody, ruleNames)
                rules.add(GrammarRule(elements))
            }
            
            return GBNFGrammar(rules, rootRuleId)
        }
        
        private fun parseRuleBody(body: String, ruleNames: Map<String, Int>): List<GrammarElement> {
            val elements = mutableListOf<GrammarElement>()
            
            // Simplified parsing - a full implementation would need a proper parser
            // This handles basic character ranges and rule references
            
            var i = 0
            while (i < body.length) {
                when {
                    body[i] == '[' -> {
                        // Character class
                        val end = body.indexOf(']', i)
                        if (end != -1) {
                            val charClass = body.substring(i + 1, end)
                            parseCharacterClass(charClass, elements)
                            i = end + 1
                        } else {
                            i++
                        }
                    }
                    body[i] == '"' -> {
                        // String literal
                        val end = body.indexOf('"', i + 1)
                        if (end != -1) {
                            val literal = body.substring(i + 1, end)
                            for (char in literal) {
                                elements.add(GrammarElement(GrammarElementType.CHAR, char.code))
                            }
                            i = end + 1
                        } else {
                            i++
                        }
                    }
                    body[i] == '\'' -> {
                        // Single character
                        if (i + 2 < body.length && body[i + 2] == '\'') {
                            elements.add(GrammarElement(GrammarElementType.CHAR, body[i + 1].code))
                            i += 3
                        } else {
                            i++
                        }
                    }
                    body[i].isLetter() -> {
                        // Rule reference
                        val start = i
                        while (i < body.length && (body[i].isLetterOrDigit() || body[i] == '_')) {
                            i++
                        }
                        val ruleName = body.substring(start, i)
                        val ruleId = ruleNames[ruleName] ?: 0
                        elements.add(GrammarElement(GrammarElementType.RULE_REF, ruleId))
                    }
                    body[i] == '|' -> {
                        // Alternative
                        elements.add(GrammarElement(GrammarElementType.ALT))
                        i++
                    }
                    else -> {
                        i++
                    }
                }
            }
            
            elements.add(GrammarElement(GrammarElementType.END))
            return elements
        }
        
        private fun parseCharacterClass(charClass: String, elements: MutableList<GrammarElement>) {
            var i = 0
            while (i < charClass.length) {
                when {
                    i + 2 < charClass.length && charClass[i + 1] == '-' -> {
                        // Character range
                        val start = charClass[i].code
                        val end = charClass[i + 2].code
                        for (c in start..end) {
                            elements.add(GrammarElement(GrammarElementType.CHAR, c))
                            if (c < end) {
                                elements.add(GrammarElement(GrammarElementType.CHAR_ALT))
                            }
                        }
                        i += 3
                    }
                    else -> {
                        // Single character
                        elements.add(GrammarElement(GrammarElementType.CHAR, charClass[i].code))
                        if (i + 1 < charClass.length) {
                            elements.add(GrammarElement(GrammarElementType.CHAR_ALT))
                        }
                        i++
                    }
                }
            }
        }
    }
    
    /**
     * Filter tokens based on grammar constraints.
     * Returns a list of valid token IDs given the current grammar state.
     */
    fun filterTokens(
        currentStates: List<GrammarState>,
        tokenizer: Tokenizer,
        candidates: Array<TokenData>
    ): Array<TokenData> {
        val validTokens = mutableSetOf<Int>()
        
        // For each current state, find which tokens can be accepted
        for (state in currentStates) {
            val validForState = getValidTokensForState(state, tokenizer)
            validTokens.addAll(validForState)
        }
        
        // Filter candidates to only include valid tokens
        return candidates.filter { it.id in validTokens }.toTypedArray()
    }
    
    /**
     * Get valid tokens for a specific grammar state.
     */
    private fun getValidTokensForState(state: GrammarState, tokenizer: Tokenizer): Set<Int> {
        val validTokens = mutableSetOf<Int>()
        
        if (state.ruleId >= rules.size) return validTokens
        
        val rule = rules[state.ruleId]
        if (state.position >= rule.elements.size) return validTokens
        
        val element = rule.elements[state.position]
        
        when (element.type) {
            GrammarElementType.CHAR -> {
                // Find tokens that start with this character
                val char = element.value.toChar()
                for (tokenId in 0 until tokenizer.getVocabSize()) {
                    val tokenText = tokenizer.decode(tokenId)
                    if (tokenText.startsWith(char)) {
                        validTokens.add(tokenId)
                    }
                }
            }
            
            GrammarElementType.RULE_REF -> {
                // Recursively check referenced rule
                val referencedRule = GrammarState(element.value, 0)
                validTokens.addAll(getValidTokensForState(referencedRule, tokenizer))
            }
            
            GrammarElementType.ALT -> {
                // Alternative - check both paths
                val nextState = GrammarState(state.ruleId, state.position + 1)
                validTokens.addAll(getValidTokensForState(nextState, tokenizer))
            }
            
            GrammarElementType.END -> {
                // End of rule - no more tokens needed for this rule
                return validTokens
            }
            
            else -> {
                // Other element types - simplified handling
                for (tokenId in 0 until tokenizer.getVocabSize()) {
                    validTokens.add(tokenId)
                }
            }
        }
        
        return validTokens
    }
    
    /**
     * Update grammar states after consuming a token.
     */
    fun updateStates(
        currentStates: List<GrammarState>,
        tokenId: Int,
        tokenizer: Tokenizer
    ): List<GrammarState> {
        val newStates = mutableListOf<GrammarState>()
        val tokenText = tokenizer.decode(tokenId)
        
        for (state in currentStates) {
            val nextStates = processToken(state, tokenText)
            newStates.addAll(nextStates)
        }
        
        return newStates.distinct()
    }
    
    /**
     * Process a token against a grammar state.
     */
    private fun processToken(state: GrammarState, tokenText: String): List<GrammarState> {
        val nextStates = mutableListOf<GrammarState>()
        
        if (state.ruleId >= rules.size) return nextStates
        
        val rule = rules[state.ruleId]
        if (state.position >= rule.elements.size) return nextStates
        
        val element = rule.elements[state.position]
        
        when (element.type) {
            GrammarElementType.CHAR -> {
                val expectedChar = element.value.toChar()
                if (tokenText.startsWith(expectedChar)) {
                    // Advance to next position
                    nextStates.add(GrammarState(state.ruleId, state.position + 1))
                }
            }
            
            GrammarElementType.RULE_REF -> {
                // Start processing the referenced rule
                nextStates.add(GrammarState(element.value, 0))
            }
            
            GrammarElementType.END -> {
                // Rule completed
                return nextStates
            }
            
            else -> {
                // Simplified - advance for other types
                nextStates.add(GrammarState(state.ruleId, state.position + 1))
            }
        }
        
        return nextStates
    }
    
    /**
     * Get initial grammar states (start from root rule).
     */
    fun getInitialStates(): List<GrammarState> {
        return listOf(GrammarState(rootRuleId, 0))
    }
    
    /**
     * Check if grammar parsing is complete.
     */
    fun isComplete(states: List<GrammarState>): Boolean {
        return states.any { state ->
            if (state.ruleId < rules.size) {
                val rule = rules[state.ruleId]
                state.position >= rule.elements.size || 
                (state.position < rule.elements.size && 
                 rule.elements[state.position].type == GrammarElementType.END)
            } else {
                false
            }
        }
    }
}

/**
 * Tokenizer interface for grammar constraints.
 * This is a simplified interface - full implementation would depend on the actual tokenizer used.
 */
interface Tokenizer {
    fun encode(text: String): List<Int>
    fun decode(tokenId: Int): String
    fun decode(tokenIds: List<Int>): String
    fun getVocabSize(): Int
}

/**
 * Simple tokenizer implementation for testing.
 */
class SimpleTokenizer : Tokenizer {
    private val vocab = mutableMapOf<String, Int>()
    private val reverseVocab = mutableMapOf<Int, String>()
    private var nextId = 0
    
    init {
        // Add basic vocabulary
        addToken(" ")
        addToken("\n")
        for (c in 'a'..'z') {
            addToken(c.toString())
        }
        for (c in 'A'..'Z') {
            addToken(c.toString())
        }
        for (c in '0'..'9') {
            addToken(c.toString())
        }
        addToken(".")
        addToken(",")
        addToken("!")
        addToken("?")
        addToken(":")
        addToken(";")
    }
    
    private fun addToken(token: String) {
        if (token !in vocab) {
            vocab[token] = nextId
            reverseVocab[nextId] = token
            nextId++
        }
    }
    
    override fun encode(text: String): List<Int> {
        val result = mutableListOf<Int>()
        for (char in text) {
            val token = char.toString()
            addToken(token) // Add new tokens dynamically
            result.add(vocab[token]!!)
        }
        return result
    }
    
    override fun decode(tokenId: Int): String {
        return reverseVocab[tokenId] ?: "<UNK>"
    }
    
    override fun decode(tokenIds: List<Int>): String {
        return tokenIds.joinToString("") { decode(it) }
    }
    
    override fun getVocabSize(): Int = nextId
}

/**
 * Grammar-constrained sampling context.
 */
class GrammarSamplingContext(
    private val grammar: GBNFGrammar,
    private val tokenizer: Tokenizer
) {
    private var currentStates = grammar.getInitialStates()
    
    /**
     * Filter tokens based on current grammar state.
     */
    fun filterCandidates(candidates: Array<TokenData>): Array<TokenData> {
        return grammar.filterTokens(currentStates, tokenizer, candidates)
    }
    
    /**
     * Update grammar state after sampling a token.
     */
    fun acceptToken(tokenId: Int) {
        currentStates = grammar.updateStates(currentStates, tokenId, tokenizer)
    }
    
    /**
     * Check if generation is complete according to grammar.
     */
    fun isComplete(): Boolean {
        return grammar.isComplete(currentStates)
    }
    
    /**
     * Reset grammar state.
     */
    fun reset() {
        currentStates = grammar.getInitialStates()
    }
    
    /**
     * Get current grammar states (for debugging).
     */
    fun getCurrentStates(): List<GrammarState> = currentStates
}