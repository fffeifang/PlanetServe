/**
 * PlanetServe Demo - Verifier Node (with Tendermint Integration)
 * 
 * Verification node that:
 * 1. Sends challenge prompts to model nodes via S-IDA relay network
 * 2. Receives responses and verifies them using PERPLEXITY-BASED verification
 * 3. Submits verification results to Java Tendermint for BFT consensus
 * 4. Queries reputation updates from Tendermint
 * 
 * Integration Modes:
 * - Standalone: Uses local LLM for perplexity calculation (default)
 * - Tendermint: Submits responses to Java CredibilityApp for verification + consensus
 * 
 * Usage:
 *   ./demo_verifier_node [--port <port>] [--tendermint <host:port>] [--challenge <prompt>]
 */

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <map>
#include <mutex>
#include <fstream>
#include <vector>
#include <iomanip>
#include <memory>

#include "config.hpp"
#include "message_types.hpp"
#include "tendermint_client.hpp"
#include "../src/encrypt_p2p/network_handler.hpp"
#include "../src/encrypt_p2p/s_ida.hpp"
#include "../src/llm/llama_wrapper.hpp"

// Global flag for signal handling
std::atomic<bool> g_running{true};
bool verbose = true;

void signalHandler(int signal) {
    std::cout << "\n[VerifierNode] Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

class DemoVerifierNode {
public:
    DemoVerifierNode(const demo::DemoConfig& config)
        : config_(config),
          listener_("127.0.0.1", config.verifier_node_port),
          llm_(std::make_unique<llm::LlamaWrapper>()),
          tendermint_client_(std::make_unique<demo::TendermintClient>()) {
        // Load challenge prompts
        loadChallengePrompts();
    }
    
    ~DemoVerifierNode() {
        stop();
    }
    
    bool initialize() {
        LOG_INFO("VerifierNode", "Initializing...");
        
        // Bind listener for receiving responses
        if (!listener_.bind("0.0.0.0", config_.verifier_node_port)) {
            LOG_ERROR("VerifierNode", "Failed to bind to port " << config_.verifier_node_port);
            return false;
        }
        
        LOG_INFO("VerifierNode", "Listening on port " << config_.verifier_node_port);
        
        // Start receiver thread
        running_ = true;
        receiver_thread_ = std::thread(&DemoVerifierNode::receiverLoop, this);
        
        return true;
    }
    
    /**
     * Connect to Java Tendermint CredibilityApp.
     */
    bool connectTendermint(const std::string& host, int port) {
        tendermint_client_->setHost(host);
        tendermint_client_->setPort(port);
        
        LOG_INFO("VerifierNode", "Connecting to Tendermint at " << host << ":" << port);
        
        if (tendermint_client_->isConnected()) {
            tendermint_enabled_ = true;
            LOG_INFO("VerifierNode", "Connected to Tendermint successfully");
            
            // Query current epoch
            std::string epoch_info = tendermint_client_->queryEpoch();
            LOG_INFO("VerifierNode", "Tendermint epoch: " << epoch_info);
            
            return true;
        } else {
            LOG_WARN("VerifierNode", "Could not connect to Tendermint - running in standalone mode");
            tendermint_enabled_ = false;
            return false;
        }
    }
    
    /**
     * Submit a model response to Tendermint for verification and consensus.
     */
    std::string submitToTendermint(const std::string& model_id,
                                    const std::string& challenge,
                                    const std::string& response) {
        if (!tendermint_enabled_) {
            return "{\"status\":\"error\",\"message\":\"Tendermint not connected\"}";
        }
        
        LOG_INFO("VerifierNode", "Submitting response to Tendermint for model: " << model_id);
        
        std::string result = tendermint_client_->submitResponse(model_id, challenge, response);
        LOG_INFO("VerifierNode", "Tendermint response: " << result);
        
        return result;
    }
    
    /**
     * Query model reputation from Tendermint.
     */
    std::string queryReputation(const std::string& model_id) {
        if (!tendermint_enabled_) {
            return "{\"status\":\"error\",\"message\":\"Tendermint not connected\"}";
        }
        
        return tendermint_client_->queryModelReputation(model_id);
    }
    
    /**
     * Query all reputations from Tendermint.
     */
    std::string queryAllReputations() {
        if (!tendermint_enabled_) {
            return "{\"status\":\"error\",\"message\":\"Tendermint not connected\"}";
        }
        
        return tendermint_client_->queryReputations();
    }
    
    bool isTendermintEnabled() const { return tendermint_enabled_; }
    
    /**
     * The verifier MUST run the same model as the model node.
     */
    bool loadModel(const std::string& model_path) {
        LOG_INFO("VerifierNode", "Loading LLM for perplexity verification: " << model_path);
        
        if (!llm_->initialize()) {
            LOG_ERROR("VerifierNode", "Failed to initialize LLM backend");
            return false;
        }
        
        if (!llm_->loadModel(model_path, 2048, 4)) {
            LOG_ERROR("VerifierNode", "Failed to load model: " << model_path);
            return false;
        }
        
        model_loaded_ = true;
        LOG_INFO("VerifierNode", "LLM loaded successfully for verification");
        return true;
    }
    
    std::string sendChallenge(const std::string& challenge_prompt) {
        std::string request_id = "verif_" + demo::generateRequestId();
        
        LOG_INFO("VerifierNode", "Sending challenge with request_id: " << request_id);
        LOG_INFO("VerifierNode", "Challenge: " << 
                (challenge_prompt.size() > 100 ? challenge_prompt.substr(0, 100) + "..." : challenge_prompt));
        
        // Create response aggregator
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            response_cache_[request_id] = demo::ReplyAggregator(request_id, config_.n, config_.k);
        }
        
        // Split the challenge using S-IDA
        std::vector<encrypt_p2p::SIDA::Clove> cloves;
        try {
            cloves = encrypt_p2p::SIDA::split(challenge_prompt, config_.n, config_.k);
        } catch (const std::exception& e) {
            LOG_ERROR("VerifierNode", "Failed to split challenge: " << e.what());
            return "ERROR: Failed to split challenge";
        }
        
        LOG_INFO("VerifierNode", "Split challenge into " << cloves.size() << " cloves");
        
        // Send each clove through a different path
        auto first_hops = config_.getFirstHops();
        
        for (size_t i = 0; i < cloves.size() && i < first_hops.size(); i++) {
            demo::DemoMessage msg;
            msg.type = demo::MSG_VERIF_CHALLENGE_SHARE;  // Different type for verification
            msg.request_id = request_id;
            msg.path_id = static_cast<int>(i);
            msg.hop_id = 0;
            msg.share_index = static_cast<int>(cloves[i].fragmentIndex);
            msg.n = config_.n;
            msg.k = config_.k;
            msg.payload = encrypt_p2p::SIDA::serializeClove(cloves[i]);
            
            // Log the send
            demo::LogEntry log;
            log.request_id = request_id;
            log.msg_type = msg.type;
            log.path_id = msg.path_id;
            log.hop_id = msg.hop_id;
            log.share_index = msg.share_index;
            log.n = msg.n;
            log.k = msg.k;
            log.next_hop = first_hops[i].ip + ":" + std::to_string(first_hops[i].port);
            log.status = "SENDING";
            log.print("VerifierNode");
            
            // Send to the first relay of this path
            encrypt_p2p::NetworkHandler sender("127.0.0.1", config_.verifier_node_port + 100 + i);
            if (!sender.connect(first_hops[i].ip, first_hops[i].port)) {
                LOG_ERROR("VerifierNode", "Failed to connect to relay " << first_hops[i].ip 
                         << ":" << first_hops[i].port);
                continue;
            }
            
            if (!sender.sendData(msg.serialize())) {
                LOG_ERROR("VerifierNode", "Failed to send challenge clove to path " << i);
            } else {
                log.status = "SENT";
                log.print("VerifierNode");
            }
            
            sender.disconnect();
        }
        
        // Wait for response
        LOG_INFO("VerifierNode", "Waiting for response (timeout: 30s)...");
        
        auto start = std::chrono::steady_clock::now();
        const int timeout_seconds = 30;
        
        while (g_running) {
            {
                std::unique_lock<std::mutex> lock(cache_mutex_);
                auto it = response_cache_.find(request_id);
                if (it != response_cache_.end() && it->second.reconstructed) {
                    LOG_INFO("VerifierNode", "Received complete response!");
                    return it->second.result;
                }
            }
            
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= timeout_seconds) {
                LOG_ERROR("VerifierNode", "Response timeout after " << timeout_seconds << " seconds");
                return "ERROR: Response timeout";
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        return "ERROR: Interrupted";
    }
    
    /**
     * Verify a model response using PERPLEXITY-BASED verification.
     * 
     * This implements the core verification protocol:
     * 1. Tokenize the response
     * 2. For each token position k, compute P(token_k | prompt + tokens[0..k-1])
     * 3. PPL = exp(-1/n * Σ log(P(token_i)))
     * 4. Credibility = 1/PPL
     * 
     * A legitimate model node running the correct model will produce
     * responses with LOW perplexity (HIGH credibility).
     * A malicious node running a different/smaller model will produce
     * responses with HIGH perplexity (LOW credibility).
     */
    llm::PerplexityResult verifyResponse(const std::string& challenge, const std::string& response) {
        llm::PerplexityResult result;
        
        // Basic validation
        if (response.empty() || response.find("ERROR:") == 0) {
            result.perplexity = 1e10;
            result.credibilityScore = 0.0;
            LOG_ERROR("VerifierNode", "Invalid response for verification");
            return result;
        }
        
        // If model not loaded, fall back to heuristic
        if (!model_loaded_ || !llm_->isModelLoaded()) {
            LOG_WARN("VerifierNode", "LLM not loaded, using heuristic verification");
            result.perplexity = 10.0;  // Assume moderate perplexity
            result.credibilityScore = 0.1;
            
            // Simple heuristic fallback
            if (response.size() > challenge.size() / 4) {
                result.credibilityScore += 0.3;
            }
            if (response.find("I cannot") == std::string::npos &&
                response.find("Error") == std::string::npos) {
                result.credibilityScore += 0.2;
            }
            return result;
        }
        
        // === REAL PERPLEXITY-BASED VERIFICATION ===
        LOG_INFO("VerifierNode", "Computing perplexity for response (" 
                << response.size() << " chars)...");
        
        auto start = std::chrono::steady_clock::now();
        
        result = llm_->calculatePerplexity(challenge, response);
        
        auto end = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        LOG_INFO("VerifierNode", "Perplexity calculation complete in " << duration_ms << "ms");
        LOG_INFO("VerifierNode", "  Tokens: " << result.tokenCount 
                << ", Matched: " << result.matchedTokenCount);
        LOG_INFO("VerifierNode", "  PPL: " << std::fixed << std::setprecision(4) << result.perplexity);
        LOG_INFO("VerifierNode", "  Credibility: " << std::fixed << std::setprecision(4) 
                << result.credibilityScore);
        
        return result;
    }
    
    /**
     * Get simple credibility score (for backward compatibility)
     */
    double getCredibilityScore(const std::string& challenge, const std::string& response) {
        return verifyResponse(challenge, response).credibilityScore;
    }
    
    void stop() {
        running_ = false;
        if (receiver_thread_.joinable()) {
            receiver_thread_.join();
        }
    }
    
    const std::vector<std::string>& getChallengePrompts() const {
        return challenge_prompts_;
    }
    
private:
    void loadChallengePrompts() {
        // Try to load from file
        std::ifstream file("eval/verification/prompts.json");
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            file.close();
            
            // Simple JSON array parsing (prompts are just strings)
            size_t pos = 0;
            while ((pos = content.find("\"", pos)) != std::string::npos) {
                size_t end = content.find("\"", pos + 1);
                if (end != std::string::npos) {
                    std::string prompt = content.substr(pos + 1, end - pos - 1);
                    if (!prompt.empty() && prompt != "[" && prompt != "]" && prompt != ",") {
                        // Unescape basic sequences
                        size_t escape_pos;
                        while ((escape_pos = prompt.find("\\n")) != std::string::npos) {
                            prompt.replace(escape_pos, 2, "\n");
                        }
                        challenge_prompts_.push_back(prompt);
                    }
                    pos = end + 1;
                } else {
                    break;
                }
            }
        }
        
        // Default challenges if file not found or empty
        if (challenge_prompts_.empty()) {
            challenge_prompts_ = {
                "What is 2 + 2?",
                "Complete the sentence: The quick brown fox...",
                "What is the capital of France?",
                "Explain what a neural network is in one sentence.",
                "What color is the sky on a clear day?"
            };
        }
        
        LOG_INFO("VerifierNode", "Loaded " << challenge_prompts_.size() << " challenge prompts");
    }
    
    void receiverLoop() {
        LOG_DEBUG("VerifierNode", "Receiver thread started");
        
        while (running_ && g_running) {
            std::string sender_ip;
            int sender_port;
            
            std::string data = listener_.receiveData(sender_ip, sender_port, 100);
            
            if (data.empty()) {
                continue;
            }
            
            demo::DemoMessage msg = demo::DemoMessage::deserialize(data);
            
            if (!msg.isValid()) {
                LOG_ERROR("VerifierNode", "Received invalid message");
                continue;
            }
            
            demo::LogEntry log;
            log.request_id = msg.request_id;
            log.msg_type = msg.type;
            log.path_id = msg.path_id;
            log.share_index = msg.share_index;
            log.n = msg.n;
            log.k = msg.k;
            log.status = "RECEIVED";
            log.print("VerifierNode");
            
            processResponseClove(msg);
        }
        
        LOG_DEBUG("VerifierNode", "Receiver thread ended");
    }
    
    void processResponseClove(const demo::DemoMessage& msg) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        auto it = response_cache_.find(msg.request_id);
        if (it == response_cache_.end()) {
            LOG_ERROR("VerifierNode", "Received response for unknown request: " << msg.request_id);
            return;
        }
        
        auto& aggregator = it->second;
        
        if (aggregator.addShare(msg.share_index, msg.payload)) {
            LOG_INFO("VerifierNode", "Added response share " << msg.share_index 
                    << " (" << aggregator.getStatus() << ")");
        }
        
        if (aggregator.hasEnoughShares() && !aggregator.reconstructed) {
            std::vector<encrypt_p2p::SIDA::Clove> cloves;
            for (const auto& share : aggregator.received_shares) {
                try {
                    cloves.push_back(encrypt_p2p::SIDA::deserializeClove(share.second));
                } catch (const std::exception& e) {
                    LOG_ERROR("VerifierNode", "Failed to deserialize response clove: " << e.what());
                }
            }
            
            try {
                aggregator.result = encrypt_p2p::SIDA::combine(cloves, aggregator.threshold);
                aggregator.reconstructed = true;
                
                demo::LogEntry log;
                log.request_id = msg.request_id;
                log.msg_type = "VERIF_RESPONSE";
                log.n = msg.n;
                log.k = msg.k;
                log.status = "RECONSTRUCTED";
                log.print("VerifierNode");
                
            } catch (const std::exception& e) {
                LOG_ERROR("VerifierNode", "Failed to reconstruct response: " << e.what());
            }
        }
    }
    
    demo::DemoConfig config_;
    encrypt_p2p::NetworkHandler listener_;
    
    std::atomic<bool> running_{false};
    std::thread receiver_thread_;
    
    std::vector<std::string> challenge_prompts_;
    
    std::map<std::string, demo::ReplyAggregator> response_cache_;
    std::mutex cache_mutex_;
    
    // LLM for perplexity-based verification
    std::unique_ptr<llm::LlamaWrapper> llm_;
    bool model_loaded_ = false;
    
    // Tendermint integration for BFT consensus
    std::unique_ptr<demo::TendermintClient> tendermint_client_;
    bool tendermint_enabled_ = false;
};



int main(int argc, char* argv[]) {
    std::string config_file = "configs/demo_local.yaml";
    std::string challenge_prompt;
    std::string model_path;
    std::string tendermint_endpoint;
    int port = 0;
    bool load_model = true;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
        else if (arg == "--challenge" && i + 1 < argc) {
            challenge_prompt = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--tendermint" && i + 1 < argc) {
            tendermint_endpoint = argv[++i];
        }
        else if (arg == "--no-model") {
            load_model = false;
        }
    }
    
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    demo::DemoConfig config = demo::ConfigLoader::loadOrDefault(config_file);
    
    if (port > 0) {
        config.verifier_node_port = port;
    }
    
    // Use model path from config if not specified on command line
    if (model_path.empty()) {
        model_path = config.model_path;
    }
    
    std::cout << "=== PlanetServe Demo - Verifier Node ===" << std::endl;
    std::cout << "Port: " << config.verifier_node_port << std::endl;
    std::cout << "Model Node: " << config.model_node_ip << ":" << config.model_node_port << std::endl;
    std::cout << "S-IDA: n=" << config.n << ", k=" << config.k << std::endl;
    std::cout << "Verification: " << (load_model ? "Local LLM" : "Heuristic") << std::endl;
    std::cout << "Tendermint: " << (tendermint_endpoint.empty() ? "(not connected)" : tendermint_endpoint) << std::endl;
    std::cout << std::endl;
    
    try {
        DemoVerifierNode verifier_node(config);
        
        if (!verifier_node.initialize()) {
            return 1;
        }
        
        // Connect to Tendermint if specified
        if (!tendermint_endpoint.empty()) {
            std::string tm_host = "localhost";
            int tm_port = 8080;
            
            size_t colon = tendermint_endpoint.find(':');
            if (colon != std::string::npos) {
                tm_host = tendermint_endpoint.substr(0, colon);
                tm_port = std::stoi(tendermint_endpoint.substr(colon + 1));
            } else {
                tm_host = tendermint_endpoint;
            }
            
            if (verifier_node.connectTendermint(tm_host, tm_port)) {
                std::cout << "[OK] Connected to Tendermint at " << tm_host << ":" << tm_port << std::endl;
            }
        }
        
        // Load LLM for perplexity-based verification
        if (load_model) {
            std::cout << "Loading LLM for perplexity verification..." << std::endl;
            if (!verifier_node.loadModel(model_path)) {
                std::cerr << "[WARNING] Failed to load LLM, falling back to heuristic verification" << std::endl;
            }
        } else {
            std::cout << "Running in heuristic mode (no LLM verification)" << std::endl;
        }
        
        if (!challenge_prompt.empty()) {
            // Single challenge mode
            std::string response = verifier_node.sendChallenge(challenge_prompt);
            
            std::cout << "\n=== Response ===" << std::endl;
            std::cout << response << std::endl;
            std::cout << "================\n" << std::endl;
            
            // Verify the response using perplexity
            auto result = verifier_node.verifyResponse(challenge_prompt, response);
            std::cout << "\n=== Local Verification Result ===" << std::endl;
            std::cout << "Perplexity: " << std::fixed << std::setprecision(4) << result.perplexity << std::endl;
            std::cout << "Credibility Score: " << std::fixed << std::setprecision(4) << result.credibilityScore << std::endl;
            std::cout << "Tokens: " << result.tokenCount << " (matched: " << result.matchedTokenCount << ")" << std::endl;
            std::cout << "==================================" << std::endl;
            
            // Submit to Tendermint if connected
            if (verifier_node.isTendermintEnabled()) {
                std::cout << "\n=== Submitting to Tendermint ===" << std::endl;
                std::string model_id = config.model_node_ip + ":" + std::to_string(config.model_node_port);
                std::string tm_result = verifier_node.submitToTendermint(model_id, challenge_prompt, response);
                std::cout << "Tendermint Response: " << tm_result << std::endl;
                
                // Query updated reputation
                std::string reputation = verifier_node.queryReputation(model_id);
                std::cout << "Model Reputation: " << reputation << std::endl;
                std::cout << "==================================" << std::endl;
            }
            
        } else {
            // Interactive mode
            std::cout << "Interactive mode. Type 'quit' to exit, 'list' to see challenges.\n" << std::endl;
            
            while (g_running) {
                std::cout << "Enter challenge (or number 1-" << verifier_node.getChallengePrompts().size() 
                         << " for preset): ";
                std::string input;
                std::getline(std::cin, input);
                
                if (input == "quit" || input == "exit") {
                    break;
                }
                
                if (input == "list") {
                    const auto& prompts = verifier_node.getChallengePrompts();
                    for (size_t i = 0; i < prompts.size(); i++) {
                        std::cout << "  " << (i + 1) << ": " << prompts[i] << std::endl;
                    }
                    continue;
                }
                
                if (input.empty()) {
                    continue;
                }
                
                // Check if input is a number
                std::string actual_prompt;
                try {
                    int idx = std::stoi(input);
                    const auto& prompts = verifier_node.getChallengePrompts();
                    if (idx >= 1 && idx <= static_cast<int>(prompts.size())) {
                        actual_prompt = prompts[idx - 1];
                    } else {
                        actual_prompt = input;
                    }
                } catch (...) {
                    actual_prompt = input;
                }
                
                std::string response = verifier_node.sendChallenge(actual_prompt);
                
                std::cout << "\n=== Response ===" << std::endl;
                std::cout << response << std::endl;
                std::cout << "================\n" << std::endl;
                
                // Verify using perplexity
                auto result = verifier_node.verifyResponse(actual_prompt, response);
                std::cout << "\n=== Verification Result ===" << std::endl;
                std::cout << "Perplexity: " << std::fixed << std::setprecision(4) << result.perplexity << std::endl;
                std::cout << "Credibility Score: " << std::fixed << std::setprecision(4) << result.credibilityScore << std::endl;
                std::cout << "Tokens: " << result.tokenCount << " (matched: " << result.matchedTokenCount << ")" << std::endl;
                std::cout << "============================" << std::endl;
                std::cout << std::endl;
            }
        }
        
        verifier_node.stop();
        
    } catch (const std::exception& e) {
        std::cerr << "[VerifierNode] Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "[VerifierNode] Shutdown complete." << std::endl;
    return 0;
}
