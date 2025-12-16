#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <map>

#include "config.hpp"
#include "message_types.hpp"
#include "../src/encrypt_p2p/network_handler.hpp"

std::atomic<bool> g_running{true};

void signalHandler(int signal) {
    std::cout << "\n[Relay] Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

class RelayNode {
public:
    RelayNode(int port, const demo::DemoConfig& config)
        : port_(port), config_(config), 
          listener_("127.0.0.1", port) {
        
        // Get relay info from config
        if (!config_.getRelayInfo(port_, path_id_, hop_id_, next_hop_ip_, next_hop_port_)) {
            throw std::runtime_error("Port " + std::to_string(port) + " not found in relay config");
        }
        
        node_name_ = "Relay:P" + std::to_string(path_id_) + "H" + std::to_string(hop_id_);
    }
    
    bool initialize() {
        if (!listener_.bind("0.0.0.0", port_)) {
            LOG_ERROR(node_name_, "Failed to bind to port " << port_);
            return false;
        }
        
        LOG_INFO(node_name_, "Listening on port " << port_);
        LOG_INFO(node_name_, "Next hop: " << next_hop_ip_ << ":" << next_hop_port_);
        
        return true;
    }
    
    void run() {
        LOG_INFO(node_name_, "Starting relay loop...");
        
        while (g_running) {
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
                LOG_ERROR(node_name_, "Received invalid message from " 
                         << sender_ip << ":" << sender_port);
                continue;
            }
            
            // Log the received message
            demo::LogEntry log;
            log.request_id = msg.request_id;
            log.msg_type = msg.type;
            log.path_id = path_id_;
            log.hop_id = hop_id_;
            log.share_index = msg.share_index;
            log.n = msg.n;
            log.k = msg.k;
            log.status = "RECEIVED";
            log.print(node_name_);
            
            // Determine forwarding direction and destination
            std::string forward_ip;
            int forward_port;
            
            if (msg.type == demo::MSG_USER_PROMPT_SHARE || 
                msg.type == demo::MSG_VERIF_CHALLENGE_SHARE) {
                // Forward direction: to model node
                forward_ip = next_hop_ip_;
                forward_port = next_hop_port_;
                
                // Increment hop_id for next relay
                msg.hop_id = hop_id_ + 1;
            } else if (msg.type == demo::MSG_MODEL_REPLY_SHARE) {
                // Reverse direction: back to user/verifier
                // For simplicity, use the sender as the previous hop to track back
                // In real implementation, this would use stored path info
                forward_ip = msg.sender_ip;
                forward_port = msg.sender_port;
            } else {
                LOG_ERROR(node_name_, "Unknown message type: " << msg.type);
                continue;
            }
            
            // Forward the message
            log.status = "FORWARDING";
            log.next_hop = forward_ip + ":" + std::to_string(forward_port);
            log.print(node_name_);
            
            // Create a new connection for forwarding
            encrypt_p2p::NetworkHandler sender("127.0.0.1", port_ + 10000);
            if (!sender.connect(forward_ip, forward_port)) {
                LOG_ERROR(node_name_, "Failed to connect to " << forward_ip << ":" << forward_port);
                continue;
            }
            
            std::string serialized = msg.serialize();
            if (!sender.sendData(serialized)) {
                LOG_ERROR(node_name_, "Failed to send to " << forward_ip << ":" << forward_port);
                sender.disconnect();
                continue;
            }
            
            sender.disconnect();
            
            // Log successful forward
            log.status = "FORWARDED";
            log.print(node_name_);
        }
        
        LOG_INFO(node_name_, "Relay loop ended.");
    }
    
private:
    int port_;
    int path_id_;
    int hop_id_;
    std::string next_hop_ip_;
    int next_hop_port_;
    demo::DemoConfig config_;
    std::string node_name_;
    encrypt_p2p::NetworkHandler listener_;
};

void printUsage(const char* program_name) {
    std::cout << "PlanetServe Demo - Relay Node\n\n"
              << "Usage:\n"
              << "  " << program_name << " --port <port> [--config <config_file>]\n\n"
              << "Options:\n"
              << "  --port <port>         Port to listen on (required)\n"
              << "  --config <file>       Configuration file (default: configs/demo_local.yaml)\n"
              << "  --help                Show this help message\n\n"
              << "Example:\n"
              << "  " << program_name << " --port 9300\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    int port = 0;
    std::string config_file = "configs/demo_local.yaml";
    
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
        else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        }
    }
    
    if (port == 0) {
        std::cerr << "Error: --port is required\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Set up signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Load configuration
    demo::DemoConfig config = demo::ConfigLoader::loadOrDefault(config_file);
    
    try {
        // Create and run the relay node
        RelayNode relay(port, config);
        
        if (!relay.initialize()) {
            return 1;
        }
        
        relay.run();
        
    } catch (const std::exception& e) {
        std::cerr << "[Relay] Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
