// src/model_node.hpp
#ifndef MODEL_NODE_HPP
#define MODEL_NODE_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <atomic>
#include "../encrypt_p2p/network_handler.hpp"
#include "../encrypt_p2p/crypto_utils.hpp"
#include "../encrypt_p2p/s_ida.hpp"
#include "../llm/llama_wrapper.hpp"


namespace node {

struct ModelSession {
    std::string sessionID;
    std::vector<std::string> history;  // Store conversation history
    std::vector<std::string> proxy_list;
    bool active = true;
};

class ModelNode {
public:
    ModelNode(const std::string& ip_address, int port);
    ~ModelNode();

    // Initialize the model node
    bool initialize();

    // Load a model using llama.cpp
    bool loadModel(const std::string &modelPath);

    // Process a message using the loaded model
    std::string processMessage(const std::string &sessionID, const std::string &message);
    
    // Start listening for messages
    void startListening();
    
    // Stop listening for messages
    void stopListening();

private:
    // LLM wrapper
    std::unique_ptr<llm::LlamaWrapper> llm;

    // Network handling
    std::string ip_address;
    int port;
    std::string RSA_PUBLIC_KEY;
    std::string RSA_PRIVATE_KEY;
    encrypt_p2p::NetworkHandler listener;
    std::atomic<bool> running{false};
    
    // Thread management
    std::thread receiverThread;
    std::thread processorThread;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<std::tuple<std::string, int, std::string>> messageQueue; // <sender_IP, sender_port, message>
    
    // Session management
    std::map<std::string, ModelSession> sessions;
    std::mutex sessionsMutex;
    
    // S-IDA message processing
    std::map<std::string, std::map<int, std::vector<encrypt_p2p::SIDA::Clove>>> sidaMessageCache; // sessionID -> seq_num -> cloves
    std::mutex sidaCacheMtx;
    
    // Helper methods
    void receiveLoop();
    void processMessages();
    bool processSIDAClove(const std::string& sessionID, int seq_num, const std::string& serializedClove);
    std::string reconstructSIDAMessage(const std::string& sessionID, int seq_num, int k);
    bool sendSIDAResponse(const std::string& sessionID, int seq_num, const std::string& message,
                          const std::string& recipient_ip, int recipient_port, int n, int k);
};

} // namespace node

#endif // MODEL_NODE_HPP