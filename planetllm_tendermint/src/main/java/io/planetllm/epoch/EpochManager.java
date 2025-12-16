package io.planetllm.epoch;

import io.planetllm.config.VerificationConfig;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Manages epoch-based verification cycles and leader selection.
 * 
 * In each epoch:
 * 1. A leader is selected from the verification node committee
 * 2. The leader sends challenge requests to model nodes
 * 3. Model nodes respond to challenges
 * 4. All verification nodes independently verify responses
 * 5. Nodes vote on reputation updates
 * 6. Updates are committed when 2n/3 + 1 votes are reached
 */
public class EpochManager {
    
    private final AtomicLong currentEpoch = new AtomicLong(0);
    private final List<String> verificationNodes = new ArrayList<>();
    private String currentLeader;
    
    // Epoch state
    private EpochState epochState = EpochState.CHALLENGE;
    
    // Challenge data for current epoch
    private final Map<String, String> currentChallenges = new ConcurrentHashMap<>();
    
    // Collected responses: modelId -> (prompt -> response)
    private final Map<String, Map<String, ChallengeResponse>> epochResponses = new ConcurrentHashMap<>();
    
    // Votes for reputation updates: modelId -> (nodeId -> vote)
    private final Map<String, Map<String, ReputationVote>> reputationVotes = new ConcurrentHashMap<>();
    
    // Committed updates for this epoch
    private final Map<String, Double> committedUpdates = new ConcurrentHashMap<>();
    
    /**
     * Represents the state of an epoch.
     */
    public enum EpochState {
        CHALLENGE,    // Leader sending challenges
        RESPONSE,     // Waiting for model responses
        EVALUATION,   // Verification nodes evaluating responses
        VOTING,       // Voting on reputation updates
        COMMITTED     // Epoch completed, updates committed
    }
    
    /**
     * Represents a model's response to a challenge.
     */
    public static class ChallengeResponse {
        public final String modelId;
        public final String prompt;
        public final String response;
        public final String digest;
        public final long timestamp;
        
        public ChallengeResponse(String modelId, String prompt, String response, 
                String digest, long timestamp) {
            this.modelId = modelId;
            this.prompt = prompt;
            this.response = response;
            this.digest = digest;
            this.timestamp = timestamp;
        }
    }
    
    /**
     * Represents a vote on a reputation update.
     */
    public static class ReputationVote {
        public final String voterId;
        public final String modelId;
        public final double proposedScore;
        public final boolean approve;
        public final String signature;
        
        public ReputationVote(String voterId, String modelId, double proposedScore, 
                boolean approve, String signature) {
            this.voterId = voterId;
            this.modelId = modelId;
            this.proposedScore = proposedScore;
            this.approve = approve;
            this.signature = signature;
        }
    }
    
    /**
     * Result of epoch processing.
     */
    public static class EpochResult {
        public final long epochNumber;
        public final Map<String, Double> reputationUpdates;
        public final int totalResponses;
        public final int successfulVerifications;
        
        public EpochResult(long epochNumber, Map<String, Double> reputationUpdates,
                int totalResponses, int successfulVerifications) {
            this.epochNumber = epochNumber;
            this.reputationUpdates = reputationUpdates;
            this.totalResponses = totalResponses;
            this.successfulVerifications = successfulVerifications;
        }
    }
    
    /**
     * Register verification nodes for leader selection.
     * 
     * @param nodeIds List of verification node identifiers
     */
    public void registerVerificationNodes(List<String> nodeIds) {
        verificationNodes.clear();
        verificationNodes.addAll(nodeIds);
        Collections.sort(verificationNodes);  // Deterministic ordering
        
        if (!verificationNodes.isEmpty()) {
            selectLeader();
        }
        
        System.out.println("Registered " + verificationNodes.size() + 
                " verification nodes. Leader: " + currentLeader);
    }
    
    /**
     * Select the leader for the current epoch.
     * Uses round-robin selection based on epoch number.
     */
    private void selectLeader() {
        if (verificationNodes.isEmpty()) {
            currentLeader = null;
            return;
        }
        
        int leaderIndex = (int) (currentEpoch.get() % verificationNodes.size());
        currentLeader = verificationNodes.get(leaderIndex);
    }
    
    /**
     * Start a new epoch.
     * 
     * @return The new epoch number
     */
    public long startNewEpoch() {
        long newEpoch = currentEpoch.incrementAndGet();
        
        // Clear previous epoch data
        currentChallenges.clear();
        epochResponses.clear();
        reputationVotes.clear();
        committedUpdates.clear();
        
        // Select new leader
        selectLeader();
        
        // Set state to challenge
        epochState = EpochState.CHALLENGE;
        
        System.out.println("Started epoch " + newEpoch + " with leader: " + currentLeader);
        
        return newEpoch;
    }
    
    /**
     * Check if a node is the current leader.
     * 
     * @param nodeId The node to check
     * @return true if the node is the current leader
     */
    public boolean isLeader(String nodeId) {
        return nodeId != null && nodeId.equals(currentLeader);
    }
    
    /**
     * Get the current leader.
     * 
     * @return The current leader's node ID
     */
    public String getCurrentLeader() {
        return currentLeader;
    }
    
    /**
     * Get the current epoch number.
     * 
     * @return Current epoch number
     */
    public long getCurrentEpoch() {
        return currentEpoch.get();
    }
    
    /**
     * Get the current epoch state.
     * 
     * @return Current epoch state
     */
    public EpochState getEpochState() {
        return epochState;
    }
    
    /**
     * Submit challenges from the leader.
     * 
     * @param leaderId The leader submitting challenges
     * @param challenges Map of modelId to challenge prompt
     * @return true if challenges were accepted
     */
    public boolean submitChallenges(String leaderId, Map<String, String> challenges) {
        if (!isLeader(leaderId)) {
            System.err.println("Only the leader can submit challenges");
            return false;
        }
        
        if (epochState != EpochState.CHALLENGE) {
            System.err.println("Invalid state for challenge submission: " + epochState);
            return false;
        }
        
        currentChallenges.clear();
        currentChallenges.putAll(challenges);
        
        // Transition to response state
        epochState = EpochState.RESPONSE;
        
        System.out.println("Submitted " + challenges.size() + " challenges for epoch " + 
                currentEpoch.get());
        
        return true;
    }
    
    /**
     * Record a challenge response from a model node.
     * 
     * @param response The challenge response
     * @return true if the response was recorded
     */
    public boolean recordResponse(ChallengeResponse response) {
        if (epochState != EpochState.RESPONSE) {
            System.err.println("Invalid state for response recording: " + epochState);
            return false;
        }
        
        // Verify the challenge exists
        if (!currentChallenges.containsValue(response.prompt)) {
            System.err.println("Response prompt does not match any challenge");
            return false;
        }
        
        // Store the response
        epochResponses.computeIfAbsent(response.modelId, k -> new ConcurrentHashMap<>())
                .put(response.prompt, response);
        
        System.out.println("Recorded response from " + response.modelId + 
                " for prompt: " + response.prompt.substring(0, Math.min(50, response.prompt.length())) + "...");
        
        return true;
    }
    
    /**
     * Check if all expected responses have been received.
     * 
     * @param expectedModelIds Set of model IDs that should respond
     * @return true if all responses received
     */
    public boolean allResponsesReceived(Set<String> expectedModelIds) {
        for (String modelId : expectedModelIds) {
            String challengePrompt = currentChallenges.get(modelId);
            if (challengePrompt == null) continue;
            
            Map<String, ChallengeResponse> responses = epochResponses.get(modelId);
            if (responses == null || !responses.containsKey(challengePrompt)) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Transition to evaluation state.
     */
    public void startEvaluation() {
        if (epochState == EpochState.RESPONSE) {
            epochState = EpochState.EVALUATION;
            System.out.println("Epoch " + currentEpoch.get() + " entering evaluation phase");
        }
    }
    
    /**
     * Submit a vote for a reputation update.
     * 
     * @param vote The reputation vote
     * @return true if the vote was recorded
     */
    public boolean submitVote(ReputationVote vote) {
        if (epochState != EpochState.EVALUATION && epochState != EpochState.VOTING) {
            System.err.println("Invalid state for voting: " + epochState);
            return false;
        }
        
        if (epochState == EpochState.EVALUATION) {
            epochState = EpochState.VOTING;
        }
        
        // Verify voter is a valid verification node
        if (!verificationNodes.contains(vote.voterId)) {
            System.err.println("Invalid voter: " + vote.voterId);
            return false;
        }
        
        // Store the vote
        reputationVotes.computeIfAbsent(vote.modelId, k -> new ConcurrentHashMap<>())
                .put(vote.voterId, vote);
        
        return true;
    }
    
    /**
     * Check if a reputation update has reached consensus (2n/3 + 1 votes).
     * 
     * @param modelId The model to check
     * @return true if consensus reached
     */
    public boolean hasReachedConsensus(String modelId) {
        Map<String, ReputationVote> votes = reputationVotes.get(modelId);
        if (votes == null) return false;
        
        int totalNodes = verificationNodes.size();
        int requiredVotes = (2 * totalNodes) / 3 + 1;
        
        // Count approvals for the same proposed score
        Map<Double, Integer> scoreVotes = new HashMap<>();
        for (ReputationVote vote : votes.values()) {
            if (vote.approve) {
                double roundedScore = Math.round(vote.proposedScore * 10000.0) / 10000.0;
                scoreVotes.merge(roundedScore, 1, Integer::sum);
            }
        }
        
        // Check if any score has enough votes
        for (int count : scoreVotes.values()) {
            if (count >= requiredVotes) {
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * Get the consensus score for a model.
     * 
     * @param modelId The model to check
     * @return The consensus score, or null if no consensus
     */
    public Double getConsensusScore(String modelId) {
        Map<String, ReputationVote> votes = reputationVotes.get(modelId);
        if (votes == null) return null;
        
        int totalNodes = verificationNodes.size();
        int requiredVotes = (2 * totalNodes) / 3 + 1;
        
        // Count approvals for each proposed score
        Map<Double, Integer> scoreVotes = new HashMap<>();
        for (ReputationVote vote : votes.values()) {
            if (vote.approve) {
                double roundedScore = Math.round(vote.proposedScore * 10000.0) / 10000.0;
                scoreVotes.merge(roundedScore, 1, Integer::sum);
            }
        }
        
        // Find score with enough votes
        for (Map.Entry<Double, Integer> entry : scoreVotes.entrySet()) {
            if (entry.getValue() >= requiredVotes) {
                return entry.getKey();
            }
        }
        
        return null;
    }
    
    /**
     * Commit reputation updates that have reached consensus.
     * 
     * @return Map of modelId to committed scores
     */
    public Map<String, Double> commitUpdates() {
        Map<String, Double> updates = new HashMap<>();
        
        for (String modelId : reputationVotes.keySet()) {
            Double consensusScore = getConsensusScore(modelId);
            if (consensusScore != null) {
                updates.put(modelId, consensusScore);
                committedUpdates.put(modelId, consensusScore);
            }
        }
        
        if (!updates.isEmpty()) {
            epochState = EpochState.COMMITTED;
            System.out.println("Committed " + updates.size() + 
                    " reputation updates for epoch " + currentEpoch.get());
        }
        
        return updates;
    }
    
    /**
     * Get all responses for the current epoch.
     * 
     * @return Map of modelId to their responses
     */
    public Map<String, Map<String, ChallengeResponse>> getEpochResponses() {
        return new HashMap<>(epochResponses);
    }
    
    /**
     * Get the current challenges.
     * 
     * @return Map of modelId to challenge prompt
     */
    public Map<String, String> getCurrentChallenges() {
        return new HashMap<>(currentChallenges);
    }
    
    /**
     * Get committed updates for this epoch.
     * 
     * @return Map of modelId to committed scores
     */
    public Map<String, Double> getCommittedUpdates() {
        return new HashMap<>(committedUpdates);
    }
    
    /**
     * Get the number of verification nodes.
     * 
     * @return Number of nodes
     */
    public int getVerificationNodeCount() {
        return verificationNodes.size();
    }
    
    /**
     * Get the list of verification nodes.
     * 
     * @return List of node IDs
     */
    public List<String> getVerificationNodes() {
        return new ArrayList<>(verificationNodes);
    }
}

