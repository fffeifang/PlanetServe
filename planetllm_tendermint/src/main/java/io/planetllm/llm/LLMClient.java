package io.planetllm.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.planetllm.config.VerificationConfig;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Client for communicating with a local llama-server instance.
 * Provides tokenization and completion capabilities for perplexity-based verification.
 */
public class LLMClient {
    
    private final String baseUrl;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    /**
     * Represents a token with its ID and text piece.
     */
    public static class Token {
        public final int id;
        public final String piece;
        
        public Token(int id, String piece) {
            this.id = id;
            this.piece = piece;
        }
        
        @Override
        public String toString() {
            return String.format("Token{id=%d, piece='%s'}", id, piece);
        }
    }
    
    /**
     * Represents a logprob entry from the completion response.
     */
    public static class LogProbEntry {
        public final int tokenId;
        public final String token;
        public final double logprob;
        
        public LogProbEntry(int tokenId, String token, double logprob) {
            this.tokenId = tokenId;
            this.token = token;
            this.logprob = logprob;
        }
    }
    
    /**
     * Represents the result of a completion request.
     */
    public static class CompletionResult {
        public final String text;
        public final List<LogProbEntry> topLogprobs;
        
        public CompletionResult(String text, List<LogProbEntry> topLogprobs) {
            this.text = text;
            this.topLogprobs = topLogprobs;
        }
    }
    
    public LLMClient() {
        this(VerificationConfig.DEFAULT_LLM_HOST, VerificationConfig.DEFAULT_LLM_PORT);
    }
    
    public LLMClient(String host, int port) {
        this.baseUrl = String.format("http://%s:%d", host, port);
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(VerificationConfig.LLM_TIMEOUT_SECONDS))
                .build();
        this.objectMapper = new ObjectMapper();
    }
    
    /**
     * Check if the LLM server is healthy and ready to handle requests.
     * 
     * @return true if server is ready
     */
    public boolean checkHealth() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/health"))
                    .timeout(Duration.ofSeconds(5))
                    .GET()
                    .build();
            
            HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() == 200) {
                JsonNode json = objectMapper.readTree(response.body());
                return "ok".equals(json.path("status").asText());
            }
            return false;
            
        } catch (Exception e) {
            System.err.println("Health check failed: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Tokenize a text string into tokens with their IDs and pieces.
     * 
     * @param content The text to tokenize
     * @return List of tokens
     * @throws LLMException if tokenization fails
     */
    public List<Token> tokenize(String content) throws LLMException {
        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("content", content);
            requestBody.put("with_pieces", true);
            
            String jsonBody = objectMapper.writeValueAsString(requestBody);
            
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/tokenize"))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(VerificationConfig.LLM_TIMEOUT_SECONDS))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();
            
            HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() != 200) {
                throw new LLMException("Tokenization failed with status: " + response.statusCode());
            }
            
            JsonNode json = objectMapper.readTree(response.body());
            JsonNode tokensNode = json.path("tokens");
            
            List<Token> tokens = new ArrayList<>();
            for (JsonNode tokenNode : tokensNode) {
                int id = tokenNode.path("id").asInt();
                String piece = tokenNode.path("piece").asText("");
                tokens.add(new Token(id, piece));
            }
            
            return tokens;
            
        } catch (LLMException e) {
            throw e;
        } catch (Exception e) {
            throw new LLMException("Tokenization error: " + e.getMessage(), e);
        }
    }
    
    /**
     * Send a completion request and get logprobs for the generated token.
     * 
     * @param prompt The prompt to complete
     * @param maxTokens Maximum tokens to generate (typically 1 for verification)
     * @param temperature Temperature for sampling (0 for deterministic)
     * @return CompletionResult with text and logprobs
     * @throws LLMException if completion fails
     */
    public CompletionResult complete(String prompt, int maxTokens, double temperature) 
            throws LLMException {
        try {
            Map<String, Object> requestBody = new HashMap<>();
            requestBody.put("prompt", prompt);
            requestBody.put("max_tokens", maxTokens);
            requestBody.put("temperature", temperature);
            requestBody.put("logprobs", VerificationConfig.TOP_K_LOGPROBS);
            
            String jsonBody = objectMapper.writeValueAsString(requestBody);
            
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/v1/completions"))
                    .header("Content-Type", "application/json")
                    .timeout(Duration.ofSeconds(VerificationConfig.LLM_TIMEOUT_SECONDS))
                    .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
                    .build();
            
            HttpResponse<String> response = httpClient.send(request, 
                    HttpResponse.BodyHandlers.ofString());
            
            if (response.statusCode() != 200) {
                throw new LLMException("Completion failed with status: " + response.statusCode());
            }
            
            JsonNode json = objectMapper.readTree(response.body());
            
            // Parse the completion response
            JsonNode choices = json.path("choices");
            if (choices.isEmpty()) {
                throw new LLMException("No choices in completion response");
            }
            
            JsonNode firstChoice = choices.get(0);
            String text = firstChoice.path("text").asText("");
            
            // Parse logprobs
            List<LogProbEntry> topLogprobs = new ArrayList<>();
            JsonNode logprobsNode = firstChoice.path("logprobs").path("content");
            
            if (logprobsNode.isArray() && !logprobsNode.isEmpty()) {
                JsonNode topLogprobsNode = logprobsNode.get(0).path("top_logprobs");
                
                for (JsonNode entry : topLogprobsNode) {
                    int tokenId = entry.path("id").asInt(-1);
                    String token = entry.path("token").asText("");
                    double logprob = entry.path("logprob").asDouble(Double.NEGATIVE_INFINITY);
                    topLogprobs.add(new LogProbEntry(tokenId, token, logprob));
                }
            }
            
            return new CompletionResult(text, topLogprobs);
            
        } catch (LLMException e) {
            throw e;
        } catch (Exception e) {
            throw new LLMException("Completion error: " + e.getMessage(), e);
        }
    }
    
    /**
     * Find the logprob for a specific token ID in the top logprobs.
     * 
     * @param topLogprobs List of top logprob entries
     * @param targetTokenId The token ID to find
     * @return The logprob value, or null if not found in top-k
     */
    public Double findLogprobForToken(List<LogProbEntry> topLogprobs, int targetTokenId) {
        for (LogProbEntry entry : topLogprobs) {
            if (entry.tokenId == targetTokenId) {
                return entry.logprob;
            }
        }
        return null;  // Token not in top-k
    }
    
    /**
     * Custom exception for LLM-related errors.
     */
    public static class LLMException extends Exception {
        public LLMException(String message) {
            super(message);
        }
        
        public LLMException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}

