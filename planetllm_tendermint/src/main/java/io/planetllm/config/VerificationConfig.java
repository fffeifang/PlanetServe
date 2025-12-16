package io.planetllm.config;

/**
 * Configuration constants for the PlanetServe verification protocol.
 * 
 * These parameters control the reputation scoring and punishment mechanisms:
 * - Alpha (α): Weight for previous epoch's reputation in moving average
 * - Beta (β): Weight for current epoch's challenge score
 * - Window Size (W): Number of past epochs to track for anomaly detection
 * - Tau (τ): Threshold below which a score is considered abnormal
 * - Gamma (γ): Fraction of abnormal values triggering punishment
 */
public class VerificationConfig {
    
    // Moving average weights for reputation calculation: R(T) = α·R(T−1) + β·C(T)
    public static final double ALPHA = 0.4;  // Weight for previous reputation
    public static final double BETA = 0.6;   // Weight for current challenge score
    
    // Sliding window parameters for anomaly detection
    public static final int WINDOW_SIZE = 5;  // Number of past epochs to track (W)
    public static final double TAU = 0.4;     // Threshold for abnormal score in single epoch (τ)
    public static final double GAMMA = 0.2;   // Fraction threshold for punishment (γ = 1/5)
    
    // Reputation thresholds
    public static final double INITIAL_REPUTATION = 0.5;  // Starting reputation for new models
    public static final double UNTRUSTED_THRESHOLD = 0.3; // Below this, model is marked untrusted
    public static final double MAX_REPUTATION = 1.0;      // Maximum possible reputation
    public static final double MIN_REPUTATION = 0.0;      // Minimum possible reputation
    
    // Perplexity calculation parameters
    public static final double EPSILON = 0.00001;  // Small value for numerical stability
    public static final int TOP_K_LOGPROBS = 50;   // Number of top logprobs to request (need more for accurate verification)
    
    // LLM server configuration
    public static final String DEFAULT_LLM_HOST = "localhost";
    public static final int DEFAULT_LLM_PORT = 8080;
    public static final int LLM_TIMEOUT_SECONDS = 30;
    
    // Epoch configuration
    public static final long EPOCH_DURATION_MS = 60000;  // 1 minute per epoch
    public static final int CHALLENGES_PER_EPOCH = 3;    // Number of challenges per epoch
    
    // Consensus configuration (Tendermint)
    public static final int MIN_VOTES_FRACTION_NUMERATOR = 2;
    public static final int MIN_VOTES_FRACTION_DENOMINATOR = 3;  // Need 2n/3 + 1 votes
    
    private VerificationConfig() {
        // Prevent instantiation
    }
    
    /**
     * Calculate the punishment factor based on abnormal count.
     * Punishment formula: (W + 1) / (W + c/γ + 2)
     * 
     * @param abnormalCount Number of abnormal values in the window
     * @return The punishment factor (less than 1.0 when punishment applies)
     */
    public static double calculatePunishmentFactor(int abnormalCount) {
        if (abnormalCount == 0) {
            return BETA;  // No punishment, use normal beta
        }
        return (WINDOW_SIZE + 1.0) / (WINDOW_SIZE + (abnormalCount / GAMMA) + 2.0);
    }
    
    /**
     * Check if the ratio of abnormal values exceeds the punishment threshold.
     * 
     * @param abnormalCount Number of abnormal values
     * @param windowCount Total values in the window
     * @return true if punishment should be applied
     */
    public static boolean shouldApplyPunishment(int abnormalCount, int windowCount) {
        if (windowCount == 0) return false;
        return ((double) abnormalCount / windowCount) > GAMMA;
    }
    
    /**
     * Check if a challenge score is considered abnormal.
     * 
     * @param score The challenge score C(T)
     * @return true if the score is below the threshold
     */
    public static boolean isAbnormalScore(double score) {
        return score < TAU;
    }
    
    /**
     * Check if a model should be marked as untrusted.
     * 
     * @param reputation The model's current reputation
     * @return true if the model should be marked untrusted
     */
    public static boolean isUntrusted(double reputation) {
        return reputation < UNTRUSTED_THRESHOLD;
    }
}

