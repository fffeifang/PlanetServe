#include "verification_node.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <openssl/sha.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/evp.h>
#include <openssl/err.h>

// For simple HTTP server
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

using json = nlohmann::json;

namespace node {

VerificationNode::VerificationNode() : VerificationNode(Config{}) {}

VerificationNode::VerificationNode(const Config& config) : config_(config) {
    // Generate RSA key pair
    auto key_pair = encrypt_p2p::generateRSAKeyPair();
    rsa_private_key_ = key_pair.first;
    rsa_public_key_ = key_pair.second;
    
    // Initialize Tendermint client
    tendermint_client_ = std::make_unique<TendermintClient>(
        config_.tendermint_host, config_.tendermint_port);
    
    // Initialize llama-server client (SINGLE source for perplexity)
    llama_server_ = std::make_unique<llm::LlamaServerClient>(
        config_.llama_server_host, config_.llama_server_port);
}

VerificationNode::~VerificationNode() {
    stop();
}

bool VerificationNode::initialize() {
    // Check Tendermint connection
    if (!isTendermintHealthy()) {
        std::cerr << "[VerificationNode] Tendermint not available at " 
                  << config_.tendermint_host << ":" << config_.tendermint_port << std::endl;
    }
    
    // Initial sync with Tendermint
    syncWithTendermint();
    
    // Connect to llama-server
    connectLlamaServer();
    
    return true;
}

bool VerificationNode::connectLlamaServer(const std::string& host, int port) {
    if (!host.empty()) {
        llama_server_->setEndpoint(host, port > 0 ? port : config_.llama_server_port);
    }
    
    llama_server_connected_ = llama_server_->isHealthy();
    return llama_server_connected_;
}

void VerificationNode::start() {
    if (running_) return;
    
    running_ = true;
    
    // Start HTTP server thread
    http_server_thread_ = std::thread(&VerificationNode::runHttpServer, this);
    
    // Start sync thread
    sync_thread_ = std::thread([this]() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (running_) {
                syncWithTendermint();
            }
        }
    });
}

void VerificationNode::stop() {
    if (!running_) return;
    
    running_ = false;
    
    // Wait for threads to finish
    if (http_server_thread_.joinable()) {
        http_server_thread_.join();
    }
    if (sync_thread_.joinable()) {
        sync_thread_.join();
    }
}

bool VerificationNode::isTendermintHealthy() {
    return tendermint_client_->checkHealth();
}

bool VerificationNode::registerUserNode(const std::string& ip_address, int port,
                                         const std::string& rsa_public_key) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    // Check if already registered
    for (const auto& node : user_nodes_list_) {
        if (std::get<0>(node) == ip_address && std::get<1>(node) == port) {
            // Update public key if changed
            if (std::get<2>(node) != rsa_public_key) {
                user_nodes_list_.erase(
                    std::remove(user_nodes_list_.begin(), user_nodes_list_.end(), node),
                    user_nodes_list_.end());
                user_nodes_list_.emplace_back(ip_address, port, rsa_public_key);
            }
            return true;
        }
    }
    
    // Add new node
    user_nodes_list_.emplace_back(ip_address, port, rsa_public_key);
    
    // Also register with Tendermint (for BFT consensus)
    tendermint_client_->registerUserNode(ip_address, port, rsa_public_key);
    
    return true;
}

bool VerificationNode::registerModelNode(const std::string& ip_address, int port,
                                          const std::string& rsa_public_key) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    // Check if already registered
    for (const auto& node : model_nodes_list_) {
        if (std::get<0>(node) == ip_address && std::get<1>(node) == port) {
            return true;
        }
    }
    
    // Add new model node
    model_nodes_list_.emplace_back(ip_address, port, rsa_public_key);
    
    // Register with Tendermint for reputation tracking
    std::string model_id = ip_address + ":" + std::to_string(port);
    tendermint_client_->registerModel(model_id, rsa_public_key);
    
    return true;
}

std::vector<std::tuple<std::string, int, std::string>> 
VerificationNode::getUserNodesList() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    return user_nodes_list_;
}

std::vector<std::tuple<std::string, int, std::string, double>>
VerificationNode::getModelNodesList() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    // Get reputations from Tendermint
    auto reputations = tendermint_client_->queryReputations();
    
    std::vector<std::tuple<std::string, int, std::string, double>> result;
    for (const auto& node : model_nodes_list_) {
        std::string model_id = std::get<0>(node) + ":" + std::to_string(std::get<1>(node));
        double reputation = 0.5;  // Default
        auto it = reputations.find(model_id);
        if (it != reputations.end()) {
            reputation = it->second;
        }
        result.emplace_back(std::get<0>(node), std::get<1>(node), std::get<2>(node), reputation);
    }
    
    return result;
}

VerificationNode::SignedNodeList VerificationNode::getSignedNodeList() {
    SignedNodeList result;
    
    // Get current node lists
    result.user_nodes = getUserNodesList();
    
    auto model_nodes = getModelNodesList();
    result.model_nodes = model_nodes;
    
    // Get epoch info
    auto epoch_info = getCurrentEpoch();
    result.epoch = epoch_info ? epoch_info->epoch_number : 0;
    
    result.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Calculate digest
    result.digest = calculateNodeListDigest();
    
    // Sign the digest with this verifier's private key
    result.signature = signDigest(result.digest);
    
    return result;
}

std::string VerificationNode::signDigest(const std::string& digest) {
    try {
        return encrypt_p2p::encryptRSA(digest.substr(0, 200), rsa_public_key_);
    } catch (...) {
        return "";
    }
}

bool VerificationNode::verifySignature(const std::string& digest, 
                                        const std::string& signature,
                                        const std::string& public_key) {
    try {
        std::string decrypted = encrypt_p2p::decryptRSA(signature, public_key);
        return decrypted == digest.substr(0, 200);
    } catch (...) {
        return false;
    }
}

bool VerificationNode::startNewEpoch() {
    return tendermint_client_->startEpoch();
}

bool VerificationNode::submitChallenges(const std::map<std::string, std::string>& challenges) {
    bool all_success = true;
    
    for (const auto& [model_id, prompt] : challenges) {
        // Generate response from model (in real scenario, model would respond via network)
        // Here we submit the challenge to Tendermint for tracking
        json payload = {
            {"type", "challenge"},
            {"sender", config_.node_id},
            {"model_id", model_id},
            {"prompt", prompt},
            {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}
        };
        
        auto result = tendermint_client_->httpPost("/tx", payload);
        if (!result || !tendermint_client_->checkResponse(*result)) {
            all_success = false;
        }
    }
    
    return all_success;
}

VerificationNode::VerificationResult VerificationNode::verifyResponse(
    const std::string& prompt, const std::string& response) {
    
    // Use llama-server as the SINGLE source of truth for perplexity
    return llama_server_->calculatePerplexity(prompt, response);
}

double VerificationNode::calculateCredibility(const std::string& model_id,
                                               const std::string& prompt,
                                               const std::string& response) {
    auto result = verifyResponse(prompt, response);
    
    if (result.success) {
        // Submit verification result to Tendermint
        std::string digest = encrypt_p2p::toHex(response.substr(0, std::min(size_t(64), response.size())));
        tendermint_client_->submitResponse(model_id, prompt, response, digest);
    }
    
    return result.credibilityScore;
}

bool VerificationNode::submitReputationVote(const std::string& model_id,
                                             double proposed_score, bool approve) {
    // Sign the vote
    std::string vote_data = model_id + ":" + std::to_string(proposed_score);
    std::string signature = signDigest(vote_data);
    
    return tendermint_client_->submitVote(
        config_.node_id, model_id, proposed_score, approve, signature);
}

std::optional<TendermintClient::EpochInfo> VerificationNode::getCurrentEpoch() {
    return tendermint_client_->queryEpoch();
}

std::optional<TendermintClient::ReputationInfo> 
VerificationNode::getModelReputation(const std::string& model_id) {
    return tendermint_client_->queryModelReputation(model_id);
}

void VerificationNode::syncWithTendermint() {
    // Get trusted models from Tendermint
    auto trusted_models = tendermint_client_->queryTrustedModels();
    auto reputations = tendermint_client_->queryReputations();
    
    // Update local model nodes with reputation data
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    for (const auto& model_id : trusted_models) {
        bool found = false;
        for (const auto& node : model_nodes_list_) {
            std::string node_id = std::get<0>(node) + ":" + std::to_string(std::get<1>(node));
            if (node_id == model_id) {
                found = true;
                break;
            }
        }
        // Trusted models are tracked in Tendermint
    }
}

std::string VerificationNode::calculateNodeListDigest() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    
    // Build JSON representation
    json list_json;
    
    for (const auto& node : user_nodes_list_) {
        list_json["user_nodes"].push_back({
            {"ip", std::get<0>(node)},
            {"port", std::get<1>(node)},
            {"public_key", std::get<2>(node)}
        });
    }
    
    for (const auto& node : model_nodes_list_) {
        list_json["model_nodes"].push_back({
            {"ip", std::get<0>(node)},
            {"port", std::get<1>(node)},
            {"public_key", std::get<2>(node)}
        });
    }
    
    // Calculate SHA-256
    std::string list_str = list_json.dump();
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(list_str.c_str()), 
           list_str.size(), hash);
    
    // Convert to hex
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setfill('0') << std::setw(2) << (int)hash[i];
    }
    
    return ss.str();
}

void VerificationNode::runHttpServer() {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) return;
    
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(config_.http_port);
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        close(server_fd);
        return;
    }
    
    if (listen(server_fd, 10) < 0) {
        close(server_fd);
        return;
    }
    
    // Set socket timeout for accept
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    
    while (running_) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        
        if (client_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;  // Timeout, check if still running
            }
            continue;
        }
        
        // Handle request in current thread (could spawn new thread for concurrency)
        handleHttpRequest(client_fd);
        close(client_fd);
    }
    
    close(server_fd);
}

void VerificationNode::handleHttpRequest(int client_socket) {
    char buffer[4096];
    ssize_t bytes_read = read(client_socket, buffer, sizeof(buffer) - 1);
    if (bytes_read <= 0) return;
    
    buffer[bytes_read] = '\0';
    std::string request(buffer);
    
    // Parse HTTP request (simplified)
    std::istringstream iss(request);
    std::string method, path, version;
    iss >> method >> path >> version;
    
    // Extract body for POST requests
    std::string body;
    size_t body_start = request.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        body = request.substr(body_start + 4);
    }
    
    // Route request
    json response;
    int status_code = 200;
    
    try {
        if (path == "/register" && method == "POST") {
            // Handle registration
            json req = json::parse(body);
            std::string ip = req.value("ip_address", "");
            int port = req.value("port", 0);
            std::string pubkey = req.value("rsa_public_key", "");
            std::string node_type = req.value("type", "user");
            
            bool success;
            if (node_type == "model") {
                success = registerModelNode(ip, port, pubkey);
            } else {
                success = registerUserNode(ip, port, pubkey);
            }
            
            if (success) {
                response["status"] = "success";
                response["message"] = "Registration successful";
                
                // Return node lists
                auto user_nodes = getUserNodesList();
                auto model_nodes = getModelNodesList();
                
                json user_json = json::array();
                for (const auto& node : user_nodes) {
                    user_json.push_back({
                        {"ip_address", std::get<0>(node)},
                        {"port", std::get<1>(node)},
                        {"rsa_public_key", std::get<2>(node)}
                    });
                }
                
                json model_json = json::array();
                for (const auto& node : model_nodes) {
                    model_json.push_back({
                        {"ip_address", std::get<0>(node)},
                        {"port", std::get<1>(node)},
                        {"rsa_public_key", std::get<2>(node)},
                        {"reputation", std::get<3>(node)}
                    });
                }
                
                response["user_nodes"] = user_json;
                response["model_nodes"] = model_json;
            } else {
                response["status"] = "error";
                response["message"] = "Registration failed";
                status_code = 400;
            }
            
        } else if (path == "/nodes" && method == "GET") {
            // Return signed node list
            auto signed_list = getSignedNodeList();
            
            json user_json = json::array();
            for (const auto& node : signed_list.user_nodes) {
                user_json.push_back({
                    {"ip_address", std::get<0>(node)},
                    {"port", std::get<1>(node)},
                    {"rsa_public_key", std::get<2>(node)}
                });
            }
            
            json model_json = json::array();
            for (const auto& node : signed_list.model_nodes) {
                model_json.push_back({
                    {"ip_address", std::get<0>(node)},
                    {"port", std::get<1>(node)},
                    {"rsa_public_key", std::get<2>(node)},
                    {"reputation", std::get<3>(node)}
                });
            }
            
            response["status"] = "success";
            response["user_nodes"] = user_json;
            response["model_nodes"] = model_json;
            response["digest"] = signed_list.digest;
            response["signature"] = signed_list.signature;
            response["verifier_id"] = config_.node_id;
            response["verifier_pubkey"] = rsa_public_key_;
            response["epoch"] = signed_list.epoch;
            response["timestamp"] = signed_list.timestamp;
            
        } else if (path == "/health" && method == "GET") {
            bool tendermint_healthy = isTendermintHealthy();
            response["status"] = "success";
            response["node_id"] = config_.node_id;
            response["tendermint_connected"] = tendermint_healthy;
            response["llama_server_connected"] = llama_server_connected_;
            
        } else if (path == "/verify" && method == "POST") {
            // Verify a model response using perplexity
            json req = json::parse(body);
            std::string model_id = req.value("model_id", "");
            std::string prompt = req.value("prompt", "");
            std::string model_response = req.value("response", "");
            
            if (prompt.empty() || model_response.empty()) {
                response["status"] = "error";
                response["message"] = "Missing prompt or response";
                status_code = 400;
            } else {
                auto result = verifyResponse(prompt, model_response);
                
                response["status"] = result.success ? "success" : "error";
                response["perplexity"] = result.perplexity;
                response["credibility"] = result.credibilityScore;
                response["token_count"] = result.tokenCount;
                response["matched_count"] = result.matchedTokenCount;
                
                // Submit to Tendermint if model_id provided
                if (!model_id.empty() && result.success) {
                    std::string digest = encrypt_p2p::toHex(model_response.substr(0, 64));
                    tendermint_client_->submitResponse(model_id, prompt, model_response, digest);
                    response["submitted_to_tendermint"] = true;
                }
            }
            
        } else if (path == "/reputation" && method == "GET") {
            // Get reputation for a model
            // Parse model_id from query string
            size_t query_start = path.find('?');
            std::string model_id;
            if (query_start != std::string::npos) {
                // Parse query params
                std::string query = path.substr(query_start + 1);
                // Simple parsing for model_id=xxx
                size_t eq = query.find('=');
                if (eq != std::string::npos) {
                    model_id = query.substr(eq + 1);
                }
            }
            
            if (!model_id.empty()) {
                auto rep_info = getModelReputation(model_id);
                if (rep_info) {
                    response["status"] = "success";
                    response["model_id"] = rep_info->model_id;
                    response["reputation"] = rep_info->reputation;
                    response["is_trusted"] = rep_info->is_trusted;
                } else {
                    response["status"] = "error";
                    response["message"] = "Model not found";
                    status_code = 404;
                }
            } else {
                // Return all reputations
                auto reputations = tendermint_client_->queryReputations();
                response["status"] = "success";
                response["reputations"] = reputations;
            }
            
        } else {
            response["status"] = "error";
            response["message"] = "Not found";
            status_code = 404;
        }
        
    } catch (const std::exception& e) {
        response["status"] = "error";
        response["message"] = std::string("Error: ") + e.what();
        status_code = 500;
    }
    
    // Send HTTP response
    std::string response_body = response.dump();
    std::ostringstream http_response;
    http_response << "HTTP/1.1 " << status_code << " OK\r\n"
                  << "Content-Type: application/json\r\n"
                  << "Content-Length: " << response_body.size() << "\r\n"
                  << "Connection: close\r\n"
                  << "\r\n"
                  << response_body;
    
    std::string response_str = http_response.str();
    write(client_socket, response_str.c_str(), response_str.size());
}

} // namespace node
