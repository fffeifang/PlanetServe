#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <map>
#include <mutex>
#include <vector>

#include "config.hpp"
#include "message_types.hpp"
#include "../src/encrypt_p2p/network_handler.hpp"
#include "../src/encrypt_p2p/s_ida.hpp"
#include "../src/llm/llama_wrapper.hpp"

// Global flag for signal handling
std::atomic<bool> g_running{true};
bool verbose = true;

void signalHandler(int signal) {
    std::cout << "\n[ModelNode] Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

class DemoModelNode {
public:
    DemoModelNode(const demo::DemoConfig& config)
        : config_(config), 
          listener_("127.0.0.1", config.model_node_port),
          llm_(new llm::LlamaWrapper()) {
    }
    
    ~DemoModelNode() {
        stop();
    }
    
    bool initialize() {
        LOG_INFO("ModelNode", "Initializing...");
        
        // Initialize LLM backend
        if (!llm_->initialize()) {
            LOG_ERROR("ModelNode", "Failed to initialize LLM backend");
            return false;
        }
        
        // Bind listener
        if (!listener_.bind("0.0.0.0", config_.model_node_port)) {
            LOG_ERROR("ModelNode", "Failed to bind to port " << config_.model_node_port);
            return false;
        }
        
        LOG_INFO("ModelNode", "Listening on port " << config_.model_node_port);
        return true;
    }
    
    bool loadModel(const std::string& model_path) {
        LOG_INFO("ModelNode", "Loading model: " << model_path);
        
        if (!llm_->loadModel(model_path, 2048)) {
            LOG_ERROR("ModelNode", "Failed to load model from: " << model_path);
            return false;
        }
        
        LOG_INFO("ModelNode", "Model loaded successfully");
        model_loaded_ = true;
        return true;
    }
    
    void run() {
        LOG_INFO("ModelNode", "Starting main loop...");
        running_ = true;
        
        while (g_running && running_) {
            std::string sender_ip;
            int sender_port;
            
            // Receive message with timeout
            std::string data = listener_.receiveData(sender_ip, sender_port, 100);
            
            if (data.empty()) {
                continue;  // Timeout, check g_running
            }
            
            // Parse the message
            demo::DemoMessage msg = demo::DemoMessage::deserialize(data);
            msg.sender_ip = sender_ip;
            msg.sender_port = sender_port;
            
            if (!msg.isValid()) {
                LOG_ERROR("ModelNode", "Received invalid message from " 
                         << sender_ip << ":" << sender_port);
                continue;
            }
            
            // Log the received clove
            demo::LogEntry log;
            log.request_id = msg.request_id;
            log.msg_type = msg.type;
            log.path_id = msg.path_id;
            log.hop_id = msg.hop_id;
            log.share_index = msg.share_index;
            log.n = msg.n;
            log.k = msg.k;
            log.status = "RECEIVED";
            log.print("ModelNode");
            
            // Process the clove
            processClove(msg);
        }
        
        LOG_INFO("ModelNode", "Main loop ended.");
    }
    
    void stop() {
        running_ = false;
    }
    
private:
    void processClove(const demo::DemoMessage& msg) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        // Get or create aggregator for this request
        if (clove_cache_.find(msg.request_id) == clove_cache_.end()) {
            clove_cache_[msg.request_id] = demo::ReplyAggregator(msg.request_id, msg.n, msg.k);
            sender_info_[msg.request_id] = std::make_pair(msg.sender_ip, msg.sender_port);
        }
        
        auto& aggregator = clove_cache_[msg.request_id];
        
        // Deserialize and add the clove
        try {
            encrypt_p2p::SIDA::Clove clove = encrypt_p2p::SIDA::deserializeClove(msg.payload);
            
            // Store as serialized for later reconstruction
            if (aggregator.addShare(msg.share_index, msg.payload)) {
                LOG_INFO("ModelNode", "Added share " << msg.share_index 
                        << " for request " << msg.request_id
                        << " (" << aggregator.getStatus() << ")");
            }
        } catch (const std::exception& e) {
            LOG_ERROR("ModelNode", "Failed to deserialize clove: " << e.what());
            return;
        }
        
        // Check if we can reconstruct
        if (aggregator.hasEnoughShares() && !aggregator.reconstructed) {
            reconstructAndProcess(msg.request_id, msg);
        }
    }
    
    void reconstructAndProcess(const std::string& request_id, const demo::DemoMessage& orig_msg) {
        auto& aggregator = clove_cache_[request_id];
        
        // Collect cloves for reconstruction
        std::vector<encrypt_p2p::SIDA::Clove> cloves;
        for (const auto& share : aggregator.received_shares) {
            try {
                cloves.push_back(encrypt_p2p::SIDA::deserializeClove(share.second));
            } catch (const std::exception& e) {
                LOG_ERROR("ModelNode", "Failed to deserialize stored clove: " << e.what());
            }
        }
        
        if (static_cast<int>(cloves.size()) < aggregator.threshold) {
            LOG_ERROR("ModelNode", "Not enough valid cloves for reconstruction");
            return;
        }
        
        // Reconstruct the original message
        std::string reconstructed;
        try {
            reconstructed = encrypt_p2p::SIDA::combine(cloves, aggregator.threshold);
            aggregator.reconstructed = true;
            
            demo::LogEntry log;
            log.request_id = request_id;
            log.msg_type = orig_msg.type;
            log.n = orig_msg.n;
            log.k = orig_msg.k;
            log.status = "RECONSTRUCTED";
            log.print("ModelNode");
            
            LOG_INFO("ModelNode", "Reconstructed prompt (" << reconstructed.size() << " bytes): " 
                    << (reconstructed.size() > 100 ? reconstructed.substr(0, 100) + "..." : reconstructed));
            
        } catch (const std::exception& e) {
            LOG_ERROR("ModelNode", "Failed to reconstruct message: " << e.what());
            return;
        }
        
        // Generate LLM response
        std::string response;
        if (model_loaded_ && llm_->isModelLoaded()) {
            LOG_INFO("ModelNode", "Generating LLM response...");
            response = llm_->generate(reconstructed, 256, 0.7f);
            LOG_INFO("ModelNode", "Generated response (" << response.size() << " bytes): "
                    << (response.size() > 100 ? response.substr(0, 100) + "..." : response));
        } else {
            // Mock response for testing without model
            response = "[MOCK RESPONSE] You asked: " + 
                      (reconstructed.size() > 50 ? reconstructed.substr(0, 50) + "..." : reconstructed);
            LOG_INFO("ModelNode", "Generated mock response (model not loaded)");
        }
        
        // Store result
        aggregator.result = response;
        
        // Send response back through S-IDA
        sendResponse(request_id, response, orig_msg);
    }
    
    void sendResponse(const std::string& request_id, const std::string& response,
                      const demo::DemoMessage& orig_msg) {
        // Split response using S-IDA
        std::vector<encrypt_p2p::SIDA::Clove> cloves;
        try {
            cloves = encrypt_p2p::SIDA::split(response, orig_msg.n, orig_msg.k);
        } catch (const std::exception& e) {
            LOG_ERROR("ModelNode", "Failed to split response: " << e.what());
            return;
        }
        
        LOG_INFO("ModelNode", "Split response into " << cloves.size() << " cloves");
        
        // Determine destination based on message type
        std::string dest_ip;
        int dest_port;
        
        if (orig_msg.type == demo::MSG_USER_PROMPT_SHARE) {
            // Send directly to user node for demo simplicity
            dest_ip = config_.user_node_ip;
            dest_port = config_.user_node_port;
        } else if (orig_msg.type == demo::MSG_VERIF_CHALLENGE_SHARE) {
            // Send directly to verifier node
            dest_ip = config_.verifier_node_ip;
            dest_port = config_.verifier_node_port;
        } else {
            LOG_ERROR("ModelNode", "Unknown original message type for response");
            return;
        }
        
        // Send all cloves to the destination
        for (size_t i = 0; i < cloves.size(); i++) {
            demo::DemoMessage reply;
            reply.type = demo::MSG_MODEL_REPLY_SHARE;
            reply.request_id = request_id;
            reply.path_id = static_cast<int>(i);
            reply.hop_id = 0;
            reply.share_index = static_cast<int>(cloves[i].fragmentIndex);
            reply.n = orig_msg.n;
            reply.k = orig_msg.k;
            reply.payload = encrypt_p2p::SIDA::serializeClove(cloves[i]);
            
            // Log the send
            demo::LogEntry log;
            log.request_id = request_id;
            log.msg_type = reply.type;
            log.path_id = reply.path_id;
            log.share_index = reply.share_index;
            log.n = reply.n;
            log.k = reply.k;
            log.next_hop = dest_ip + ":" + std::to_string(dest_port);
            log.status = "SENDING";
            log.print("ModelNode");
            
            // Send the clove - use unique local port for each send
            encrypt_p2p::NetworkHandler sender("127.0.0.1", config_.model_node_port + 100 + static_cast<int>(i));
            if (!sender.connect(dest_ip, dest_port)) {
                LOG_ERROR("ModelNode", "Failed to connect to " << dest_ip << ":" << dest_port);
                continue;
            }
            
            if (!sender.sendData(reply.serialize())) {
                LOG_ERROR("ModelNode", "Failed to send response clove " << i);
            } else {
                log.status = "SENT";
                log.print("ModelNode");
            }
            
            sender.disconnect();
        }
    }
    
    demo::DemoConfig config_;
    encrypt_p2p::NetworkHandler listener_;
    std::unique_ptr<llm::LlamaWrapper> llm_;
    
    std::atomic<bool> running_{false};
    bool model_loaded_ = false;
    
    // Clove aggregation cache
    std::map<std::string, demo::ReplyAggregator> clove_cache_;
    std::map<std::string, std::pair<std::string, int>> sender_info_;
    std::mutex cache_mutex_;
};

void printUsage(const char* program_name) {
    std::cout << "PlanetServe Demo - Model Node\n\n"
              << "Usage:\n"
              << "  " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  --port <port>         Port to listen on (default: 9000)\n"
              << "  --model <path>        Path to model file (default: from config)\n"
              << "  --config <file>       Configuration file (default: configs/demo_local.yaml)\n"
              << "  --no-model            Run without loading LLM (for testing)\n"
              << "  --help                Show this help message\n\n"
              << "Example:\n"
              << "  " << program_name << " --model models/Llama-3.2-1B-Instruct-Q4_K_M.gguf\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    std::string config_file = "configs/demo_local.yaml";
    std::string model_path;
    int port = 0;
    bool load_model = true;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
        else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
        else if (arg == "--no-model") {
            load_model = false;
        }
    }
    
    // Set up signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Load configuration
    demo::DemoConfig config = demo::ConfigLoader::loadOrDefault(config_file);
    
    // Override with command line
    if (port > 0) {
        config.model_node_port = port;
    }
    if (!model_path.empty()) {
        config.model_path = model_path;
    }
    
    std::cout << "=== PlanetServe Demo - Model Node ===" << std::endl;
    std::cout << "Port: " << config.model_node_port << std::endl;
    std::cout << "Model: " << config.model_path << std::endl;
    std::cout << "S-IDA: n=" << config.n << ", k=" << config.k << std::endl;
    std::cout << std::endl;
    
    try {
        DemoModelNode model_node(config);
        
        if (!model_node.initialize()) {
            return 1;
        }
        
        if (load_model) {
            if (!model_node.loadModel(config.model_path)) {
                std::cerr << "[ModelNode] Warning: Running without LLM model" << std::endl;
            }
        } else {
            std::cout << "[ModelNode] Running in no-model mode (mock responses)" << std::endl;
        }
        
        std::cout << "\nModel node is running. Press Ctrl+C to stop.\n" << std::endl;
        
        model_node.run();
        
    } catch (const std::exception& e) {
        std::cerr << "[ModelNode] Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "[ModelNode] Shutdown complete." << std::endl;
    return 0;
}
