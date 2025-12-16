package io.planetllm;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.planetllm.config.VerificationConfig;
import io.planetllm.epoch.EpochManager;
import io.planetllm.epoch.EpochManager.ChallengeResponse;
import io.planetllm.epoch.EpochManager.EpochState;
import io.planetllm.epoch.EpochManager.ReputationVote;
import io.planetllm.llm.LLMClient;
import io.planetllm.reputation.ReputationManager;
import io.planetllm.reputation.ReputationManager.ReputationUpdate;
import io.planetllm.verification.PerplexityCalculator;
import io.planetllm.verification.PerplexityCalculator.PerplexityResult;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * PlanetServe Verification Application using Tendermint consensus.
 * 
 * Implements the verification protocol:
 * 1. Sampling-based verification with challenge prompts
 * 2. Perplexity-based credibility scoring
 * 3. Reputation with moving average and sliding window punishment
 * 4. BFT consensus for reputation updates (2n/3 + 1 votes)
 */
public class CredibilityApp {
    
    private final Map<String, String> storage = new ConcurrentHashMap<>();
    private final ObjectMapper objectMapper;
    
    // Core components
    private final LLMClient llmClient;
    private final PerplexityCalculator perplexityCalculator;
    private final ReputationManager reputationManager;
    private final EpochManager epochManager;
    
    // This node's identity
    private String nodeId;
    
    // Cache of verification results for current epoch
    private final Map<String, PerplexityResult> verificationCache = new ConcurrentHashMap<>();
    
    // Challenge prompts pool for sampling
    private final List<String> challengePromptsPool = new ArrayList<>();
    
    // User nodes registry for proxy selection
    private final Map<String, UserNodeInfo> userNodes = new ConcurrentHashMap<>();
    
    /**
     * User node information for proxy selection.
     */
    public static class UserNodeInfo {
        public final String ipAddress;
        public final int port;
        public final String rsaPublicKey;
        public final long registrationTime;
        
        public UserNodeInfo(String ipAddress, int port, String rsaPublicKey) {
            this.ipAddress = ipAddress;
            this.port = port;
            this.rsaPublicKey = rsaPublicKey;
            this.registrationTime = System.currentTimeMillis();
        }
    }
    
    public CredibilityApp() {
        this("verification_node_1");
    }
    
    public CredibilityApp(String nodeId) {
        this.nodeId = nodeId;
        this.objectMapper = new ObjectMapper();
        
        // Initialize LLM client
        this.llmClient = new LLMClient();
        this.perplexityCalculator = new PerplexityCalculator(llmClient);
        
        // Initialize reputation and epoch managers
        this.reputationManager = new ReputationManager();
        this.epochManager = new EpochManager();
        
        // Initialize challenge prompts pool
        initializeChallengePrompts();
        
        // Initialize with example verification nodes
        List<String> verificationNodes = Arrays.asList(
                "verification_node_1",
                "verification_node_2",
                "verification_node_3"
        );
        epochManager.registerVerificationNodes(verificationNodes);
        
        // Initialize with example model nodes
        registerDefaultModelNodes();
        
        System.out.println("CredibilityApp initialized with node ID: " + nodeId);
    }
    
    /**
     * Initialize the pool of challenge prompts.
     */
    private void initializeChallengePrompts() {
        challengePromptsPool.add("The United States Congress");
        challengePromptsPool.add("What is climate change?");
        challengePromptsPool.add("Explain the theory of relativity");
        challengePromptsPool.add("What is machine learning?");
        challengePromptsPool.add("Describe the water cycle");
        challengePromptsPool.add("What causes earthquakes?");
        challengePromptsPool.add("How do vaccines work?");
        challengePromptsPool.add("What is photosynthesis?");
        challengePromptsPool.add("Explain quantum computing");
        challengePromptsPool.add("What is blockchain technology?");
    }
    
    /**
     * Register default model nodes.
     */
    private void registerDefaultModelNodes() {
        reputationManager.registerModel("172.16.1.10", "public_key_1");
        reputationManager.registerModel("172.16.1.11", "public_key_2");
        reputationManager.registerModel("172.16.1.12", "public_key_3");
    }
    
    /**
     * Set this node's ID.
     */
    public void setNodeId(String nodeId) {
        this.nodeId = nodeId;
    }
    
    /**
     * Get this node's ID.
     */
    public String getNodeId() {
        return nodeId;
    }
    
    /**
     * Check if the LLM server is healthy.
     */
    public boolean checkLLMHealth() {
        return llmClient.checkHealth();
    }

    /**
     * Process transactions submitted to the application.
     * 
     * @param txData The transaction data as a byte array (JSON)
     * @return A response message
     */
    public String deliverTx(byte[] txData) {
        try {
            Map<String, Object> tx = objectMapper.readValue(txData, Map.class);
            String type = (String) tx.get("type");
            
            if (type == null) {
                return createErrorResponse("Invalid transaction format: missing type");
            }
            
            switch (type) {
                case "start_epoch":
                    return handleStartEpoch(tx);
                case "challenge":
                    return handleChallenge(tx);
                case "response":
                    return handleResponse(tx);
                case "vote":
                    return handleVote(tx);
                case "commit":
                    return handleCommit(tx);
                case "register_model":
                    return handleRegisterModel(tx);
                case "register_user":
                    return handleRegisterUser(tx);
                default:
                    return createErrorResponse("Unknown transaction type: " + type);
            }
            
        } catch (Exception e) {
            return createErrorResponse("Error processing transaction: " + e.getMessage());
        }
    }
    
    /**
     * Handle start_epoch transaction.
     */
    private String handleStartEpoch(Map<String, Object> tx) {
        long newEpoch = epochManager.startNewEpoch();
        verificationCache.clear();
        return createSuccessResponse("Started epoch " + newEpoch + 
                ", leader: " + epochManager.getCurrentLeader());
    }
    
    /**
     * Handle challenge transaction from the leader.
     */
    @SuppressWarnings("unchecked")
    private String handleChallenge(Map<String, Object> tx) {
        String sender = (String) tx.get("sender");
        Map<String, String> challenges = (Map<String, String>) tx.get("challenges");
        
        if (!epochManager.isLeader(sender)) {
            return createErrorResponse("Only the leader can submit challenges");
        }
        
        if (challenges == null || challenges.isEmpty()) {
            return createErrorResponse("No challenges provided");
        }
        
        boolean success = epochManager.submitChallenges(sender, challenges);
        if (success) {
            return createSuccessResponse("Submitted " + challenges.size() + " challenges");
        } else {
            return createErrorResponse("Failed to submit challenges");
        }
    }
    
    /**
     * Handle response transaction from a model node.
     */
    private String handleResponse(Map<String, Object> tx) {
        String modelId = (String) tx.get("model_id");
        String prompt = (String) tx.get("prompt");
        String response = (String) tx.get("response");
        String digest = (String) tx.get("digest");
        Object timestampObj = tx.get("timestamp");
        long timestamp = timestampObj instanceof Number ? 
                ((Number) timestampObj).longValue() : System.currentTimeMillis();
        
        // Validate required fields
        if (modelId == null || prompt == null || response == null) {
            return createErrorResponse("Missing required fields in response");
        }
        
        // Check if model is registered
        if (!reputationManager.isRegistered(modelId)) {
            return createErrorResponse("Unknown model: " + modelId);
        }
        
        // Check if model is trusted
        if (!reputationManager.isTrusted(modelId)) {
            return createErrorResponse("Model is marked as untrusted: " + modelId);
        }
        
        // Record the response
        ChallengeResponse challengeResponse = new ChallengeResponse(
                modelId, prompt, response, digest, timestamp);
        
        boolean recorded = epochManager.recordResponse(challengeResponse);
        if (!recorded) {
            return createErrorResponse("Failed to record response");
        }
        
        // Verify the response using perplexity calculation
        try {
            PerplexityResult result = perplexityCalculator.calculateCredibility(prompt, response);
            verificationCache.put(modelId + ":" + prompt, result);
            
            System.out.printf("Verified response from %s: credibility=%.4f, PPL=%.4f%n",
                    modelId, result.credibilityScore, result.perplexity);
            
            return createSuccessResponse(String.format(
                    "Response recorded and verified. Credibility: %.4f", 
                    result.credibilityScore));
                    
        } catch (PerplexityCalculator.VerificationException e) {
            // Record response but note verification failure
            System.err.println("Verification failed for " + modelId + ": " + e.getMessage());
            return createSuccessResponse("Response recorded but verification failed: " + 
                    e.getMessage());
        }
    }
    
    /**
     * Handle vote transaction for reputation updates.
     */
    private String handleVote(Map<String, Object> tx) {
        String voterId = (String) tx.get("voter_id");
        String modelId = (String) tx.get("model_id");
        Object scoreObj = tx.get("proposed_score");
        double proposedScore = scoreObj instanceof Number ? 
                ((Number) scoreObj).doubleValue() : 0.0;
        Boolean approve = (Boolean) tx.get("approve");
        String signature = (String) tx.get("signature");
        
        if (voterId == null || modelId == null || approve == null) {
            return createErrorResponse("Missing required fields in vote");
        }
        
        ReputationVote vote = new ReputationVote(
                voterId, modelId, proposedScore, approve, signature);
        
        boolean recorded = epochManager.submitVote(vote);
        if (!recorded) {
            return createErrorResponse("Failed to record vote");
        }
        
        // Check if consensus has been reached
        if (epochManager.hasReachedConsensus(modelId)) {
            Double consensusScore = epochManager.getConsensusScore(modelId);
            return createSuccessResponse(String.format(
                    "Vote recorded. Consensus reached for %s: %.4f", 
                    modelId, consensusScore));
        }
        
        return createSuccessResponse("Vote recorded");
    }
    
    /**
     * Handle commit transaction to finalize reputation updates.
     */
    private String handleCommit(Map<String, Object> tx) {
        // Commit all reputation updates that have reached consensus
        Map<String, Double> updates = epochManager.commitUpdates();
        
        // Apply updates through reputation manager
        List<ReputationUpdate> reputationUpdates = new ArrayList<>();
        for (Map.Entry<String, Double> entry : updates.entrySet()) {
            String modelId = entry.getKey();
            double challengeScore = entry.getValue();
            
            ReputationUpdate update = reputationManager.updateReputation(modelId, challengeScore);
            reputationUpdates.add(update);
            
            System.out.println("Updated reputation: " + update);
        }
        
        return createSuccessResponse("Committed " + updates.size() + " reputation updates");
    }
    
    /**
     * Handle register_model transaction.
     */
    private String handleRegisterModel(Map<String, Object> tx) {
        String modelId = (String) tx.get("model_id");
        String publicKey = (String) tx.get("public_key");
        
        if (modelId == null || publicKey == null) {
            return createErrorResponse("Missing model_id or public_key");
        }
        
        reputationManager.registerModel(modelId, publicKey);
        return createSuccessResponse("Registered model: " + modelId);
    }
    
    /**
     * Handle register_user transaction for user node registration.
     */
    private String handleRegisterUser(Map<String, Object> tx) {
        String ipAddress = (String) tx.get("ip_address");
        Object portObj = tx.get("port");
        int port = portObj instanceof Number ? ((Number) portObj).intValue() : 0;
        String rsaPublicKey = (String) tx.get("rsa_public_key");
        
        if (ipAddress == null || port <= 0 || rsaPublicKey == null) {
            return createErrorResponse("Missing ip_address, port, or rsa_public_key");
        }
        
        String nodeId = ipAddress + ":" + port;
        userNodes.put(nodeId, new UserNodeInfo(ipAddress, port, rsaPublicKey));
        
        System.out.println("Registered user node: " + nodeId);
        return createSuccessResponse("Registered user node: " + nodeId);
    }
    
    /**
     * Get all registered user nodes.
     */
    public Map<String, UserNodeInfo> getUserNodes() {
        return new HashMap<>(userNodes);
    }
    
    /**
     * Generate a local credibility score for a model based on cached verification results.
     * This calculates C(T) for voting.
     */
    public double calculateEpochCredibilityScore(String modelId) {
        List<PerplexityResult> results = new ArrayList<>();
        
        for (Map.Entry<String, PerplexityResult> entry : verificationCache.entrySet()) {
            if (entry.getKey().startsWith(modelId + ":")) {
                results.add(entry.getValue());
            }
        }
        
        if (results.isEmpty()) {
            return 0.0;
        }
        
        return perplexityCalculator.calculateAverageCredibility(results);
    }
    
    /**
     * Generate challenges for the current epoch.
     * Only the leader should call this.
     */
    public Map<String, String> generateChallenges() {
        Map<String, String> challenges = new HashMap<>();
        Set<String> modelIds = reputationManager.getTrustedModelIds();
        
        Random random = new Random();
        int challengeCount = Math.min(VerificationConfig.CHALLENGES_PER_EPOCH, 
                challengePromptsPool.size());
        
        List<String> selectedPrompts = new ArrayList<>(challengePromptsPool);
        Collections.shuffle(selectedPrompts, random);
        
        int promptIndex = 0;
        for (String modelId : modelIds) {
            if (promptIndex >= challengeCount) break;
            challenges.put(modelId, selectedPrompts.get(promptIndex));
            promptIndex++;
        }
        
        return challenges;
    }
    
    /**
     * Commit the current state to storage.
     */
    public byte[] commit() {
        try {
            // Save current state
            storage.put("epoch", String.valueOf(epochManager.getCurrentEpoch()));
            storage.put("epoch_state", epochManager.getEpochState().name());
            storage.put("reputations", objectMapper.writeValueAsString(
                    reputationManager.getAllReputations()));
            storage.put("committed_updates", objectMapper.writeValueAsString(
                    epochManager.getCommittedUpdates()));
            
            // Create a hash of the application state
            String appStateStr = objectMapper.writeValueAsString(storage);
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(appStateStr.getBytes(StandardCharsets.UTF_8));
            return Arrays.copyOf(hash, 8);
            
        } catch (Exception e) {
            System.err.println("Error committing state: " + e.getMessage());
            return new byte[8];
        }
    }

    /**
     * Handle queries to the application state.
     */
    public String query(String path, byte[] queryData) {
        try {
            Map<String, Object> queryParams = queryData != null && queryData.length > 0 ?
                    objectMapper.readValue(queryData, Map.class) : new HashMap<>();
            
            switch (path) {
                case "epoch":
                    return createQueryResponse(Map.of(
                            "epoch", epochManager.getCurrentEpoch(),
                            "state", epochManager.getEpochState().name(),
                            "leader", epochManager.getCurrentLeader()
                    ));
                    
                case "challenges":
                    return createQueryResponse(epochManager.getCurrentChallenges());
                    
                case "reputations":
                    return createQueryResponse(reputationManager.getAllReputations());
                    
                case "model_reputation":
                    String modelId = (String) queryParams.get("model_id");
                    if (modelId != null) {
                        return createQueryResponse(Map.of(
                                "model_id", modelId,
                                "reputation", reputationManager.getReputation(modelId),
                                "trusted", reputationManager.isTrusted(modelId),
                                "history", reputationManager.getScoreHistory(modelId)
                        ));
                    }
                    return createErrorResponse("Missing model_id parameter");
                    
                case "trusted_models":
                    return createQueryResponse(reputationManager.getTrustedModelIds());
                    
                case "untrusted_models":
                    return createQueryResponse(reputationManager.getUntrustedModelIds());
                    
                case "verification_nodes":
                    return createQueryResponse(Map.of(
                            "nodes", epochManager.getVerificationNodes(),
                            "count", epochManager.getVerificationNodeCount()
                    ));
                    
                case "health":
                    boolean llmHealthy = checkLLMHealth();
                    return createQueryResponse(Map.of(
                            "app_healthy", true,
                            "llm_healthy", llmHealthy,
                            "node_id", nodeId
                    ));
                
                case "user_nodes":
                    List<Map<String, Object>> userNodesList = new ArrayList<>();
                    for (UserNodeInfo info : userNodes.values()) {
                        userNodesList.add(Map.of(
                                "ip_address", info.ipAddress,
                                "port", info.port,
                                "rsa_public_key", info.rsaPublicKey
                        ));
                    }
                    return createQueryResponse(userNodesList);
                    
                default:
                    return createErrorResponse("Unknown query path: " + path);
            }
            
        } catch (Exception e) {
            return createErrorResponse("Error processing query: " + e.getMessage());
        }
    }
    
    /**
     * Create a success response.
     */
    private String createSuccessResponse(String message) {
        try {
            return objectMapper.writeValueAsString(Map.of(
                    "status", "success",
                    "message", message
            ));
        } catch (Exception e) {
            return "{\"status\":\"success\",\"message\":\"" + message + "\"}";
        }
    }
    
    /**
     * Create an error response.
     */
    private String createErrorResponse(String message) {
        try {
            return objectMapper.writeValueAsString(Map.of(
                    "status", "error",
                    "message", message
            ));
        } catch (Exception e) {
            return "{\"status\":\"error\",\"message\":\"" + message + "\"}";
        }
    }
    
    /**
     * Create a query response.
     */
    private String createQueryResponse(Object data) {
        try {
            return objectMapper.writeValueAsString(Map.of(
                    "status", "success",
                    "data", data
            ));
        } catch (Exception e) {
            return createErrorResponse("Failed to serialize response");
        }
    }
    
    // Getters for testing and monitoring
    
    public ReputationManager getReputationManager() {
        return reputationManager;
    }
    
    public EpochManager getEpochManager() {
        return epochManager;
    }
    
    public PerplexityCalculator getPerplexityCalculator() {
        return perplexityCalculator;
    }
    
    public LLMClient getLLMClient() {
        return llmClient;
    }
}
