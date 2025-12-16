#ifndef VERIFICATION_NODE_HPP
#define VERIFICATION_NODE_HPP

#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>

#include "../encrypt_p2p/network_handler.hpp"
#include "../encrypt_p2p/key_generation.hpp"
#include "../encrypt_p2p/crypto_utils.hpp"
#include "../llm/llama_server_client.hpp"
#include "tendermint_client.hpp"

#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace node {


class VerificationNode {
public:
    struct Config {
        std::string ip_address = "0.0.0.0";
        int http_port = 8080;
        std::string tendermint_host = "localhost";
        int tendermint_port = 26658;
        std::string node_id = "verification_node_1";
        // llama-server endpoint (single source of truth for perplexity)
        std::string llama_server_host = "localhost";
        int llama_server_port = 8080;
    };
    
    using VerificationResult = llm::PerplexityResult;

    struct SignedNodeList {
        std::vector<std::tuple<std::string, int, std::string>> user_nodes;  // <IP, port, pubkey>
        std::vector<std::tuple<std::string, int, std::string, double>> model_nodes;  // <IP, port, pubkey, reputation>
        std::string digest;
        std::string signature;  // This verifier's signature
        int64_t epoch;
        int64_t timestamp;
    };

    VerificationNode();
    VerificationNode(const Config& config);
    ~VerificationNode();

    // Initialize the verification node
    bool initialize();

    // Start the HTTP server and background tasks
    void start();

    // Stop the node gracefully
    void stop();

    // Check if Tendermint connection is healthy
    bool isTendermintHealthy();

        
    // Node Registration API
        

    // Register a new user node
    bool registerUserNode(const std::string& ip_address, int port, 
                          const std::string& rsa_public_key);

    // Register a new model node
    bool registerModelNode(const std::string& ip_address, int port,
                           const std::string& rsa_public_key);

    // Get the global user nodes list
    std::vector<std::tuple<std::string, int, std::string>> getUserNodesList() const;

    // Get the global model nodes list with reputations
    std::vector<std::tuple<std::string, int, std::string, double>> getModelNodesList() const;

        
    // Signed Node List API (BFT consensus)
        

    /**
     * Get a signed node list for distribution to users.
     * The list is signed by this verification node and should be aggregated
     * with signatures from other verifiers for BFT proof.
     */
    SignedNodeList getSignedNodeList();

    /**
     * Sign a node list digest with this verifier's private key.
     */
    std::string signDigest(const std::string& digest);

    /**
     * Verify a signature from another verification node.
     */
    bool verifySignature(const std::string& digest, const std::string& signature,
                         const std::string& public_key);

        
    // Verification Protocol API
        

    // Start a new verification epoch (leader only)
    bool startNewEpoch();

    // Submit challenges to model nodes (leader only)
    bool submitChallenges(const std::map<std::string, std::string>& challenges);

    // Connect to llama-server for perplexity verification
    bool connectLlamaServer(const std::string& host = "", int port = 0);
    bool isLlamaServerConnected() const { return llama_server_connected_; }
    
    // Calculate credibility score for a model response using perplexity
    // Uses llama-server as the SINGLE source of truth
    VerificationResult verifyResponse(const std::string& prompt, const std::string& response);
    
    // Legacy interface - returns credibility score only
    double calculateCredibility(const std::string& model_id, const std::string& prompt,
                                const std::string& response);

    // Submit a vote for reputation update
    bool submitReputationVote(const std::string& model_id, double proposed_score, bool approve);

    // Get current epoch information
    std::optional<TendermintClient::EpochInfo> getCurrentEpoch();

    // Query model reputation
    std::optional<TendermintClient::ReputationInfo> getModelReputation(const std::string& model_id);

        
    // Accessors
        

    std::string getNodeId() const { return config_.node_id; }
    std::string getPublicKey() const { return rsa_public_key_; }
    bool isRunning() const { return running_; }

private:
    Config config_;
    
    // RSA key pair for signing
    std::string rsa_public_key_;
    std::string rsa_private_key_;
    
    // Tendermint client for BFT consensus
    std::unique_ptr<TendermintClient> tendermint_client_;
    
    // llama-server client for perplexity calculation (SINGLE source of truth)
    std::unique_ptr<llm::LlamaServerClient> llama_server_;
    bool llama_server_connected_ = false;
    
    // Local node lists (synced with Tendermint)
    std::vector<std::tuple<std::string, int, std::string>> user_nodes_list_;
    std::vector<std::tuple<std::string, int, std::string>> model_nodes_list_;
    mutable std::mutex nodes_mutex_;
    
    // HTTP server state
    std::atomic<bool> running_{false};
    std::thread http_server_thread_;
    
    // Sync with Tendermint periodically
    std::thread sync_thread_;
    
    // Internal methods
    void runHttpServer();
    void syncWithTendermint();
    void handleHttpRequest(int client_socket);
    
    // Calculate digest of node list
    std::string calculateNodeListDigest() const;
};

} // namespace node

#endif 