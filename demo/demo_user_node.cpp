#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <map>
#include <mutex>
#include <condition_variable>

#include "config.hpp"
#include "message_types.hpp"
#include "../src/encrypt_p2p/network_handler.hpp"
#include "../src/encrypt_p2p/s_ida.hpp"

// Global flag for signal handling
std::atomic<bool> g_running{true};
bool verbose = true;

void signalHandler(int signal) {
    std::cout << "\n[UserNode] Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

class DemoUserNode {
public:
    DemoUserNode(const demo::DemoConfig& config)
        : config_(config),
          listener_("127.0.0.1", config.user_node_port) {
    }
    
    ~DemoUserNode() {
        stop();
    }
    
    bool initialize() {
        LOG_INFO("UserNode", "Initializing...");
        
        // Bind listener for receiving responses
        if (!listener_.bind("0.0.0.0", config_.user_node_port)) {
            LOG_ERROR("UserNode", "Failed to bind to port " << config_.user_node_port);
            return false;
        }
        
        LOG_INFO("UserNode", "Listening on port " << config_.user_node_port);
        
        // Start receiver thread
        running_ = true;
        receiver_thread_ = std::thread(&DemoUserNode::receiverLoop, this);
        
        return true;
    }
    
    std::string sendPrompt(const std::string& prompt) {
        std::string request_id = demo::generateRequestId();
        
        LOG_INFO("UserNode", "Sending prompt with request_id: " << request_id);
        LOG_INFO("UserNode", "Prompt (" << prompt.size() << " bytes): " 
                << (prompt.size() > 100 ? prompt.substr(0, 100) + "..." : prompt));
        
        // Create response aggregator
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            response_cache_[request_id] = demo::ReplyAggregator(request_id, config_.n, config_.k);
        }
        
        // Split the prompt using S-IDA
        std::vector<encrypt_p2p::SIDA::Clove> cloves;
        try {
            cloves = encrypt_p2p::SIDA::split(prompt, config_.n, config_.k);
        } catch (const std::exception& e) {
            LOG_ERROR("UserNode", "Failed to split prompt: " << e.what());
            return "ERROR: Failed to split prompt";
        }
        
        LOG_INFO("UserNode", "Split prompt into " << cloves.size() << " cloves");
        
        // Send each clove through a different path
        auto first_hops = config_.getFirstHops();
        
        for (size_t i = 0; i < cloves.size() && i < first_hops.size(); i++) {
            demo::DemoMessage msg;
            msg.type = demo::MSG_USER_PROMPT_SHARE;
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
            log.print("UserNode");
            
            // Send to the first relay of this path
            encrypt_p2p::NetworkHandler sender("127.0.0.1", config_.user_node_port + 100 + i);
            if (!sender.connect(first_hops[i].ip, first_hops[i].port)) {
                LOG_ERROR("UserNode", "Failed to connect to relay " << first_hops[i].ip 
                         << ":" << first_hops[i].port);
                continue;
            }
            
            if (!sender.sendData(msg.serialize())) {
                LOG_ERROR("UserNode", "Failed to send clove to path " << i);
            } else {
                log.status = "SENT";
                log.print("UserNode");
            }
            
            sender.disconnect();
        }
        
        // Wait for response
        LOG_INFO("UserNode", "Waiting for response (timeout: 30s)...");
        
        auto start = std::chrono::steady_clock::now();
        const int timeout_seconds = 30;
        
        while (g_running) {
            {
                std::unique_lock<std::mutex> lock(cache_mutex_);
                auto it = response_cache_.find(request_id);
                if (it != response_cache_.end() && it->second.reconstructed) {
                    LOG_INFO("UserNode", "Received complete response!");
                    return it->second.result;
                }
            }
            
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() >= timeout_seconds) {
                LOG_ERROR("UserNode", "Response timeout after " << timeout_seconds << " seconds");
                return "ERROR: Response timeout";
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        return "ERROR: Interrupted";
    }
    
    void stop() {
        running_ = false;
        if (receiver_thread_.joinable()) {
            receiver_thread_.join();
        }
    }
    
private:
    void receiverLoop() {
        LOG_DEBUG("UserNode", "Receiver thread started");
        
        while (running_ && g_running) {
            std::string sender_ip;
            int sender_port;
            
            std::string data = listener_.receiveData(sender_ip, sender_port, 100);
            
            if (data.empty()) {
                continue;
            }
            
            // Parse the message
            demo::DemoMessage msg = demo::DemoMessage::deserialize(data);
            
            if (!msg.isValid()) {
                LOG_ERROR("UserNode", "Received invalid message");
                continue;
            }
            
            // Log received response clove
            demo::LogEntry log;
            log.request_id = msg.request_id;
            log.msg_type = msg.type;
            log.path_id = msg.path_id;
            log.share_index = msg.share_index;
            log.n = msg.n;
            log.k = msg.k;
            log.status = "RECEIVED";
            log.print("UserNode");
            
            // Process the response clove
            processResponseClove(msg);
        }
        
        LOG_DEBUG("UserNode", "Receiver thread ended");
    }
    
    void processResponseClove(const demo::DemoMessage& msg) {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        auto it = response_cache_.find(msg.request_id);
        if (it == response_cache_.end()) {
            LOG_ERROR("UserNode", "Received response for unknown request: " << msg.request_id);
            return;
        }
        
        auto& aggregator = it->second;
        
        // Add the share
        if (aggregator.addShare(msg.share_index, msg.payload)) {
            LOG_INFO("UserNode", "Added response share " << msg.share_index 
                    << " (" << aggregator.getStatus() << ")");
        }
        
        // Check if we can reconstruct
        if (aggregator.hasEnoughShares() && !aggregator.reconstructed) {
            // Reconstruct the response
            std::vector<encrypt_p2p::SIDA::Clove> cloves;
            for (const auto& share : aggregator.received_shares) {
                try {
                    cloves.push_back(encrypt_p2p::SIDA::deserializeClove(share.second));
                } catch (const std::exception& e) {
                    LOG_ERROR("UserNode", "Failed to deserialize response clove: " << e.what());
                }
            }
            
            try {
                aggregator.result = encrypt_p2p::SIDA::combine(cloves, aggregator.threshold);
                aggregator.reconstructed = true;
                
                demo::LogEntry log;
                log.request_id = msg.request_id;
                log.msg_type = "MODEL_REPLY";
                log.n = msg.n;
                log.k = msg.k;
                log.status = "RECONSTRUCTED";
                log.print("UserNode");
                
                LOG_INFO("UserNode", "Reconstructed response (" << aggregator.result.size() << " bytes)");
                
            } catch (const std::exception& e) {
                LOG_ERROR("UserNode", "Failed to reconstruct response: " << e.what());
            }
        }
    }
    
    demo::DemoConfig config_;
    encrypt_p2p::NetworkHandler listener_;
    
    std::atomic<bool> running_{false};
    std::thread receiver_thread_;
    
    // Response aggregation
    std::map<std::string, demo::ReplyAggregator> response_cache_;
    std::mutex cache_mutex_;
};

int main(int argc, char* argv[]) {
    std::string config_file = "configs/demo_local.yaml";
    std::string prompt;
    int port = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        }
        else if (arg == "--port" && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        }
        else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
    }
    
    // Set up signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Load configuration
    demo::DemoConfig config = demo::ConfigLoader::loadOrDefault(config_file);
    
    if (port > 0) {
        config.user_node_port = port;
    }
    
    std::cout << "=== PlanetServe Demo - User Node ===" << std::endl;
    std::cout << "Port: " << config.user_node_port << std::endl;
    std::cout << "Model Node: " << config.model_node_ip << ":" << config.model_node_port << std::endl;
    std::cout << "S-IDA: n=" << config.n << ", k=" << config.k << std::endl;
    std::cout << "Relay Paths: " << config.relay_paths.size() << std::endl;
    std::cout << std::endl;
    
    try {
        DemoUserNode user_node(config);
        
        if (!user_node.initialize()) {
            return 1;
        }
        
        if (prompt.empty()) {
            // Interactive mode
            std::cout << "Interactive mode. Type 'quit' to exit.\n" << std::endl;
            
            while (g_running) {
                std::cout << "Enter prompt: ";
                std::getline(std::cin, prompt);
                
                if (prompt == "quit" || prompt == "exit") {
                    break;
                }
                
                if (prompt.empty()) {
                    continue;
                }
                
                std::string response = user_node.sendPrompt(prompt);
                
                std::cout << "\n=== Response ===" << std::endl;
                std::cout << response << std::endl;
                std::cout << "================\n" << std::endl;
            }
        } else {
            // Single prompt mode
            std::string response = user_node.sendPrompt(prompt);
            
            std::cout << "\n=== Response ===" << std::endl;
            std::cout << response << std::endl;
            std::cout << "================" << std::endl;
        }
        
        user_node.stop();
        
    } catch (const std::exception& e) {
        std::cerr << "[UserNode] Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "[UserNode] Shutdown complete." << std::endl;
    return 0;
}
