package io.planetllm.reputation;

import io.planetllm.config.VerificationConfig;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages reputation scores for model nodes with moving average and sliding window punishment.
 * 
 * Reputation update formula:
 *   R(T) = α·R(T−1) + β·C(T)  (normal case)
 *   R(T) = α·R(T−1) + ((W+1)/(W+c/γ+2))·C(T)  (punishment case)
 * 
 * where:
 *   - R(T): Reputation at epoch T
 *   - C(T): Average challenge score at epoch T
 *   - α, β: Moving average weights (0.4, 0.6)
 *   - W: Sliding window size (5)
 *   - c: Count of abnormal values in window
 *   - γ: Punishment threshold (1/5)
 */
public class ReputationManager {
    
    // Model ID -> Current reputation score
    private final Map<String, Double> reputations = new ConcurrentHashMap<>();
    
    // Model ID -> Sliding window of past challenge scores
    private final Map<String, LinkedList<Double>> scoreHistory = new ConcurrentHashMap<>();
    
    // Model ID -> Trusted status
    private final Map<String, Boolean> trustedStatus = new ConcurrentHashMap<>();
    
    // Model ID -> Public key for signature verification
    private final Map<String, String> modelPublicKeys = new ConcurrentHashMap<>();
    
    /**
     * Represents the result of a reputation update.
     */
    public static class ReputationUpdate {
        public final String modelId;
        public final double previousReputation;
        public final double newReputation;
        public final double challengeScore;
        public final boolean punishmentApplied;
        public final int abnormalCount;
        public final boolean isTrusted;
        
        public ReputationUpdate(String modelId, double previousReputation, 
                double newReputation, double challengeScore, 
                boolean punishmentApplied, int abnormalCount, boolean isTrusted) {
            this.modelId = modelId;
            this.previousReputation = previousReputation;
            this.newReputation = newReputation;
            this.challengeScore = challengeScore;
            this.punishmentApplied = punishmentApplied;
            this.abnormalCount = abnormalCount;
            this.isTrusted = isTrusted;
        }
        
        @Override
        public String toString() {
            return String.format(
                "ReputationUpdate{model='%s', prev=%.4f, new=%.4f, C(T)=%.4f, " +
                "punishment=%b, abnormal=%d, trusted=%b}",
                modelId, previousReputation, newReputation, challengeScore,
                punishmentApplied, abnormalCount, isTrusted);
        }
    }
    
    /**
     * Register a new model node.
     * 
     * @param modelId Unique identifier for the model (e.g., IP address)
     * @param publicKey Public key for signature verification
     */
    public void registerModel(String modelId, String publicKey) {
        reputations.putIfAbsent(modelId, VerificationConfig.INITIAL_REPUTATION);
        scoreHistory.putIfAbsent(modelId, new LinkedList<>());
        trustedStatus.putIfAbsent(modelId, true);
        modelPublicKeys.put(modelId, publicKey);
        
        System.out.println("Registered model: " + modelId + 
                " with initial reputation: " + VerificationConfig.INITIAL_REPUTATION);
    }
    
    /**
     * Update the reputation for a model based on its challenge score.
     * 
     * @param modelId The model's identifier
     * @param challengeScore The average score C(T) from challenges in this epoch
     * @return ReputationUpdate with details of the update
     */
    public ReputationUpdate updateReputation(String modelId, double challengeScore) {
        // Get current reputation (or initialize if new)
        double previousReputation = reputations.getOrDefault(modelId, 
                VerificationConfig.INITIAL_REPUTATION);
        
        // Get or create score history
        LinkedList<Double> history = scoreHistory.computeIfAbsent(modelId, 
                k -> new LinkedList<>());
        
        // Count abnormal values in the sliding window
        int abnormalCount = countAbnormalValues(history);
        
        // Check if current score is abnormal and add to history
        if (VerificationConfig.isAbnormalScore(challengeScore)) {
            abnormalCount++;
        }
        
        // Add current score to history and maintain window size
        history.addLast(challengeScore);
        if (history.size() > VerificationConfig.WINDOW_SIZE) {
            history.removeFirst();
        }
        
        // Calculate new reputation
        double newReputation;
        boolean punishmentApplied = false;
        
        int windowCount = history.size();
        
        if (VerificationConfig.shouldApplyPunishment(abnormalCount, windowCount)) {
            // Apply punishment: R(T) = α·R(T−1) + punishmentFactor·C(T)
            double punishmentFactor = VerificationConfig.calculatePunishmentFactor(abnormalCount);
            newReputation = VerificationConfig.ALPHA * previousReputation + 
                           punishmentFactor * challengeScore;
            punishmentApplied = true;
            
            System.out.printf("Punishment applied to %s: factor=%.4f, abnormal=%d/%d%n",
                    modelId, punishmentFactor, abnormalCount, windowCount);
        } else {
            // Normal update: R(T) = α·R(T−1) + β·C(T)
            newReputation = VerificationConfig.ALPHA * previousReputation + 
                           VerificationConfig.BETA * challengeScore;
        }
        
        // Clamp reputation to valid range
        newReputation = Math.min(VerificationConfig.MAX_REPUTATION, 
                Math.max(VerificationConfig.MIN_REPUTATION, newReputation));
        
        // Update stored reputation
        reputations.put(modelId, newReputation);
        
        // Update trusted status
        boolean isTrusted = !VerificationConfig.isUntrusted(newReputation);
        trustedStatus.put(modelId, isTrusted);
        
        if (!isTrusted) {
            System.out.printf("WARNING: Model %s marked as UNTRUSTED (reputation=%.4f)%n",
                    modelId, newReputation);
        }
        
        return new ReputationUpdate(
                modelId, previousReputation, newReputation, challengeScore,
                punishmentApplied, abnormalCount, isTrusted);
    }
    
    /**
     * Count the number of abnormal values in the score history.
     */
    private int countAbnormalValues(LinkedList<Double> history) {
        int count = 0;
        for (Double score : history) {
            if (VerificationConfig.isAbnormalScore(score)) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * Get the current reputation for a model.
     * 
     * @param modelId The model's identifier
     * @return The current reputation score
     */
    public double getReputation(String modelId) {
        return reputations.getOrDefault(modelId, VerificationConfig.INITIAL_REPUTATION);
    }
    
    /**
     * Check if a model is currently trusted.
     * 
     * @param modelId The model's identifier
     * @return true if the model is trusted
     */
    public boolean isTrusted(String modelId) {
        return trustedStatus.getOrDefault(modelId, false);
    }
    
    /**
     * Get all registered model IDs.
     * 
     * @return Set of model identifiers
     */
    public Set<String> getAllModelIds() {
        return new HashSet<>(reputations.keySet());
    }
    
    /**
     * Get all trusted model IDs.
     * 
     * @return Set of trusted model identifiers
     */
    public Set<String> getTrustedModelIds() {
        Set<String> trusted = new HashSet<>();
        for (Map.Entry<String, Boolean> entry : trustedStatus.entrySet()) {
            if (entry.getValue()) {
                trusted.add(entry.getKey());
            }
        }
        return trusted;
    }
    
    /**
     * Get all untrusted model IDs.
     * 
     * @return Set of untrusted model identifiers
     */
    public Set<String> getUntrustedModelIds() {
        Set<String> untrusted = new HashSet<>();
        for (Map.Entry<String, Boolean> entry : trustedStatus.entrySet()) {
            if (!entry.getValue()) {
                untrusted.add(entry.getKey());
            }
        }
        return untrusted;
    }
    
    /**
     * Get the score history for a model.
     * 
     * @param modelId The model's identifier
     * @return List of historical challenge scores
     */
    public List<Double> getScoreHistory(String modelId) {
        LinkedList<Double> history = scoreHistory.get(modelId);
        return history != null ? new ArrayList<>(history) : new ArrayList<>();
    }
    
    /**
     * Get all reputations as a map.
     * 
     * @return Map of model ID to reputation score
     */
    public Map<String, Double> getAllReputations() {
        return new HashMap<>(reputations);
    }
    
    /**
     * Get the public key for a model.
     * 
     * @param modelId The model's identifier
     * @return The model's public key, or null if not registered
     */
    public String getPublicKey(String modelId) {
        return modelPublicKeys.get(modelId);
    }
    
    /**
     * Check if a model is registered.
     * 
     * @param modelId The model's identifier
     * @return true if the model is registered
     */
    public boolean isRegistered(String modelId) {
        return reputations.containsKey(modelId);
    }
    
    /**
     * Serialize the current state to a map for storage.
     * 
     * @return Map containing all reputation data
     */
    public Map<String, Object> serialize() {
        Map<String, Object> state = new HashMap<>();
        state.put("reputations", new HashMap<>(reputations));
        state.put("trustedStatus", new HashMap<>(trustedStatus));
        
        // Convert score histories to regular lists for serialization
        Map<String, List<Double>> historyMap = new HashMap<>();
        for (Map.Entry<String, LinkedList<Double>> entry : scoreHistory.entrySet()) {
            historyMap.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        state.put("scoreHistory", historyMap);
        
        return state;
    }
    
    /**
     * Restore state from a serialized map.
     * 
     * @param state The state map to restore from
     */
    @SuppressWarnings("unchecked")
    public void deserialize(Map<String, Object> state) {
        if (state.containsKey("reputations")) {
            reputations.clear();
            reputations.putAll((Map<String, Double>) state.get("reputations"));
        }
        
        if (state.containsKey("trustedStatus")) {
            trustedStatus.clear();
            trustedStatus.putAll((Map<String, Boolean>) state.get("trustedStatus"));
        }
        
        if (state.containsKey("scoreHistory")) {
            scoreHistory.clear();
            Map<String, List<Double>> historyMap = 
                    (Map<String, List<Double>>) state.get("scoreHistory");
            for (Map.Entry<String, List<Double>> entry : historyMap.entrySet()) {
                LinkedList<Double> history = new LinkedList<>(entry.getValue());
                scoreHistory.put(entry.getKey(), history);
            }
        }
    }
}

