/**
 * PlanetServe Verification Node - Main Entry Point
 * 
 * This starts a C++ verification node that integrates with the Java Tendermint
 * application for BFT consensus on reputation updates.
 * 
 * Usage:
 *   ./verification_node [options]
 * 
 * Options:
 *   --port <port>           HTTP server port (default: 8080)
 *   --node-id <id>          Node identifier (default: verification_node_1)
 *   --tendermint <host:port> Tendermint endpoint (default: localhost:26658)
 *   --help                  Show this help message
 */

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>

#include "verification_node.hpp"


std::atomic<bool> g_running{true};

void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

void printUsage(const char* program_name) {
    std::cout << "PlanetServe Verification Node\n\n"
              << "Usage:\n"
              << "  " << program_name << " [options]\n\n"
              << "Options:\n"
              << "  --port <port>             HTTP server port (default: 8080)\n"
              << "  --node-id <id>            Node identifier (default: verification_node_1)\n"
              << "  --tendermint <host:port>  Tendermint endpoint (default: localhost:26658)\n"
              << "  --llama-server <host:port> llama-server endpoint for perplexity (default: localhost:8080)\n"
              << "  --help                    Show this help message\n\n"
              << "Note: Uses llama-server as the SINGLE source of truth for perplexity calculation.\n"
              << "      Both Java Tendermint and C++ nodes share the same llama-server instance.\n\n"
              << "Example:\n"
              << "  " << program_name << " --port 8081 --llama-server localhost:8080\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    node::VerificationNode::Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--port" && i + 1 < argc) {
            config.http_port = std::stoi(argv[++i]);
        }
        else if (arg == "--node-id" && i + 1 < argc) {
            config.node_id = argv[++i];
        }
        else if (arg == "--tendermint" && i + 1 < argc) {
            std::string endpoint = argv[++i];
            size_t colon = endpoint.find(':');
            if (colon != std::string::npos) {
                config.tendermint_host = endpoint.substr(0, colon);
                config.tendermint_port = std::stoi(endpoint.substr(colon + 1));
            } else {
                config.tendermint_host = endpoint;
            }
        }
        else if (arg == "--llama-server" && i + 1 < argc) {
            std::string endpoint = argv[++i];
            size_t colon = endpoint.find(':');
            if (colon != std::string::npos) {
                config.llama_server_host = endpoint.substr(0, colon);
                config.llama_server_port = std::stoi(endpoint.substr(colon + 1));
            } else {
                config.llama_server_host = endpoint;
            }
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Print configuration
    std::cout << "=== PlanetServe Verification Node ===" << std::endl;
    std::cout << "Node ID: " << config.node_id << std::endl;
    std::cout << "HTTP Port: " << config.http_port << std::endl;
    std::cout << "Tendermint: " << config.tendermint_host << ":" << config.tendermint_port << std::endl;
    std::cout << "llama-server: " << config.llama_server_host << ":" << config.llama_server_port << std::endl;
    std::cout << std::endl;
    
    // Set up signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Create and initialize the verification node
    node::VerificationNode verification_node(config);
    
    if (!verification_node.initialize()) {
        std::cerr << "Failed to initialize verification node" << std::endl;
        return 1;
    }
    
    // Start the node
    verification_node.start();
    
    std::cout << "\nVerification node is running. Press Ctrl+C to stop.\n" << std::endl;
    std::cout << "API Endpoints:" << std::endl;
    std::cout << "  POST /register       - Register a user or model node" << std::endl;
    std::cout << "  GET  /nodes          - Get signed node list" << std::endl;
    std::cout << "  GET  /health         - Health check" << std::endl;
    std::cout << "  GET  /reputation     - Query reputations" << std::endl;
    std::cout << std::endl;
    
    // Wait for shutdown signal
    while (g_running && verification_node.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Stop the node
    verification_node.stop();
    
    std::cout << "Verification node stopped." << std::endl;
    return 0;
}

