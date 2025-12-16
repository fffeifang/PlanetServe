package io.planetllm.verification;

import io.planetllm.config.VerificationConfig;
import io.planetllm.llm.LLMClient;
import io.planetllm.llm.LLMClient.Token;
import io.planetllm.llm.LLMClient.CompletionResult;
import io.planetllm.llm.LLMClient.LogProbEntry;
import io.planetllm.llm.LLMClient.LLMException;

import java.util.ArrayList;
import java.util.List;

/**
 * Calculates perplexity-based credibility scores for model responses.
 * 
 * The verification process evaluates responses token-by-token:
 * 1. Tokenize the response into individual tokens
 * 2. For each token position k, condition the local LLM on prompt + tokens[0..k]
 * 3. Get the probability distribution for the next token
 * 4. Look up the probability assigned to the actual token at position k+1
 * 5. Calculate perplexity: PPL = exp(-1/n * Σ log(p(t_i | t_<i)))
 * 6. Return credibility score as 1/PPL
 */
public class PerplexityCalculator {
    
    private final LLMClient llmClient;
    
    /**
     * Result of a perplexity calculation.
     */
    public static class PerplexityResult {
        public final double perplexity;
        public final double credibilityScore;
        public final int tokenCount;
        public final int matchedTokenCount;
        public final List<Double> tokenProbabilities;
        
        public PerplexityResult(double perplexity, double credibilityScore, 
                int tokenCount, int matchedTokenCount, List<Double> tokenProbabilities) {
            this.perplexity = perplexity;
            this.credibilityScore = credibilityScore;
            this.tokenCount = tokenCount;
            this.matchedTokenCount = matchedTokenCount;
            this.tokenProbabilities = tokenProbabilities;
        }
        
        @Override
        public String toString() {
            return String.format(
                "PerplexityResult{PPL=%.4f, credibility=%.4f, tokens=%d, matched=%d}",
                perplexity, credibilityScore, tokenCount, matchedTokenCount);
        }
    }
    
    public PerplexityCalculator(LLMClient llmClient) {
        this.llmClient = llmClient;
    }
    
    /**
     * Calculate the credibility score for a response given a prompt.
     * Uses token-by-token probability evaluation with perplexity.
     * 
     * @param prompt The original prompt sent to the model
     * @param response The response from the model node to verify
     * @return PerplexityResult containing perplexity and credibility score
     * @throws VerificationException if verification fails
     */
    public PerplexityResult calculateCredibility(String prompt, String response) 
            throws VerificationException {
        try {
            // First, tokenize the response
            List<Token> tokens = llmClient.tokenize(response);
            
            if (tokens.isEmpty()) {
                System.err.println("No tokens in response");
                return new PerplexityResult(Double.MAX_VALUE, 0.0, 0, 0, new ArrayList<>());
            }
            
            System.out.println("Tokenized response into " + tokens.size() + " tokens");
            
            // Track probabilities for each token
            List<Double> tokenProbabilities = new ArrayList<>();
            int matchedCount = 0;
            
            // Build context starting with the prompt
            StringBuilder currentContext = new StringBuilder(prompt);
            
            // For each token position, get the probability of the current token
            // At position i, we check if tokens[i] would be predicted by the LLM
            // given the prompt + tokens[0..i-1]
            for (int i = 0; i < tokens.size() - 1; i++) {
                Token currentToken = tokens.get(i);
                
                // Get completion with logprobs for what comes next
                CompletionResult completion = llmClient.complete(
                        currentContext.toString(), 
                        1,  // Generate one token
                        0   // Temperature 0 for deterministic
                );
                
                // Find the probability assigned to the actual token at position i
                Double logprob = llmClient.findLogprobForToken(
                        completion.topLogprobs, 
                        currentToken.id
                );
                
                double probability;
                if (logprob != null) {
                    // Convert logprob to probability: p = e^logprob
                    probability = Math.exp(logprob);
                    matchedCount++;
                    System.out.printf("Token '%s': logprob=%.4f, prob=%.6f%n", 
                            currentToken.piece, logprob, probability);
                } else {
                    // Token not in top-k, use epsilon as probability
                    probability = VerificationConfig.EPSILON;
                    System.out.printf("Token '%s': not in top-k, using epsilon%n", 
                            currentToken.piece);
                }
                
                tokenProbabilities.add(probability);
                
                // Append current token's text to context for next iteration
                currentContext.append(currentToken.piece);
            }
            
            // Calculate perplexity using geometric mean
            if (tokenProbabilities.isEmpty()) {
                return new PerplexityResult(Double.MAX_VALUE, 0.0, 
                        tokens.size(), 0, tokenProbabilities);
            }
            
            // PPL = exp(-1/n * Σ log(p_i))
            double sumLogProb = 0.0;
            for (double prob : tokenProbabilities) {
                sumLogProb += Math.log(prob > 0 ? prob : VerificationConfig.EPSILON);
            }
            
            double avgNegLogProb = -sumLogProb / tokenProbabilities.size();
            double perplexity = Math.exp(avgNegLogProb);
            
            // Credibility score is the reciprocal of perplexity (normalized)
            double credibilityScore = 1.0 / perplexity;
            
            // Clamp credibility score to [0, 1]
            credibilityScore = Math.min(1.0, Math.max(0.0, credibilityScore));
            
            System.out.printf("Perplexity: %.4f, Credibility: %.4f%n", 
                    perplexity, credibilityScore);
            
            return new PerplexityResult(
                    perplexity, 
                    credibilityScore, 
                    tokens.size(), 
                    matchedCount, 
                    tokenProbabilities
            );
            
        } catch (LLMException e) {
            throw new VerificationException("LLM error during verification: " + e.getMessage(), e);
        }
    }
    
    /**
     * Calculate the average credibility score across multiple challenge responses.
     * This represents C(T) for an epoch.
     * 
     * @param results List of perplexity results from multiple challenges
     * @return Average credibility score
     */
    public double calculateAverageCredibility(List<PerplexityResult> results) {
        if (results.isEmpty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (PerplexityResult result : results) {
            sum += result.credibilityScore;
        }
        
        return sum / results.size();
    }
    
    /**
     * Batch verification of multiple prompt-response pairs.
     * 
     * @param promptResponsePairs List of [prompt, response] pairs
     * @return List of perplexity results
     */
    public List<PerplexityResult> batchVerify(List<String[]> promptResponsePairs) {
        List<PerplexityResult> results = new ArrayList<>();
        
        for (String[] pair : promptResponsePairs) {
            if (pair.length != 2) {
                System.err.println("Invalid prompt-response pair, skipping");
                continue;
            }
            
            try {
                PerplexityResult result = calculateCredibility(pair[0], pair[1]);
                results.add(result);
            } catch (VerificationException e) {
                System.err.println("Verification failed for pair: " + e.getMessage());
                // Add a failed result with zero credibility
                results.add(new PerplexityResult(
                        Double.MAX_VALUE, 0.0, 0, 0, new ArrayList<>()));
            }
        }
        
        return results;
    }
    
    /**
     * Custom exception for verification errors.
     */
    public static class VerificationException extends Exception {
        public VerificationException(String message) {
            super(message);
        }
        
        public VerificationException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}

