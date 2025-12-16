#include "model_node.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <openssl/sha.h>
#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>

namespace node {

ModelNode::ModelNode(const std::string& ip_address, int port) 
    : ip_address(ip_address), port(port), listener(ip_address, port), llm(new llm::LlamaWrapper()) {
    std::cout << "[ModelNode] Created with IP: " << ip_address << ", port: " << port << std::endl;
    std::pair<std::string, std::string> keyPair = encrypt_p2p::generateRSAKeyPair();
    RSA_PUBLIC_KEY = keyPair.first;
    RSA_PRIVATE_KEY = keyPair.second;
}

ModelNode::~ModelNode() {
    stopListening();
    std::cout << "[ModelNode] Destroyed." << std::endl;
}

bool ModelNode::initialize() {
    // Initialize llama.cpp backend
    if (!llm->initialize()) {
        std::cerr << "[ModelNode] Failed to initialize LLM backend" << std::endl;
        return false;
    }
    
    // Bind listener to port
    if (!listener.bind("0.0.0.0", port)) {
        std::cerr << "[ModelNode] Failed to bind listener socket to port " << port << std::endl;
        return false;
    }
    
    std::cout << "[ModelNode] Successfully initialized and bound to port " << port << std::endl;
    return true;
}

bool ModelNode::loadModel(const std::string &modelPath) {
    if (!llm->loadModel(modelPath, 2048)) {
        std::cerr << "[ModelNode] Failed to load model from: " << modelPath << std::endl;
        return false;
    }
    
    std::cout << "[ModelNode] Successfully loaded model from: " << modelPath << std::endl;
    return true;
}

std::string ModelNode::processMessage(const std::string &sessionID, const std::string &message) {
    std::lock_guard<std::mutex> lock(sessionsMutex);
    
    // Get or create session
    std::vector<std::string> proxy_list;
    {
        const std::string proxyKey = "\"proxy_list\":\"";
        size_t keyPos = message.find(proxyKey);
        if (keyPos != std::string::npos) {
            keyPos += proxyKey.length();
            size_t endPos = message.find("\"", keyPos);
            if (endPos != std::string::npos) {
                std::string proxies = message.substr(keyPos, endPos - keyPos);
                std::istringstream iss(proxies);
                std::string token;
                while (iss >> token) {
                    proxy_list.push_back(token);
                }
            }
        }
    }
    auto& session = sessions[sessionID];
    if (session.sessionID.empty()) {
        session.sessionID = sessionID;
        session.proxy_list = proxy_list;
    }
    
    // Add user message to history
    session.history.push_back("User: " + message);
    
    // Prepare the prompt with context from previous conversation
    std::string fullPrompt;
    if (!session.history.empty()) {
        // Use up to 5 previous exchanges for context
        const auto& history = session.history;
        int startIdx = std::max(0, static_cast<int>(history.size()) - 10);
        
        fullPrompt = "Previous conversation:\n";
        for (int i = startIdx; i < static_cast<int>(history.size()) - 1; ++i) {
            fullPrompt += history[i] + "\n";
        }
        fullPrompt += "\nCurrent message: " + message + "\n\nResponse:";
    } else {
        fullPrompt = message + "\n\nResponse:";
    }
    
    // Generate response
    std::string response = llm->generate(fullPrompt, 512, 0.7f);
    
    // Add response to history
    session.history.push_back("Assistant: " + response);
    
    return response;
}

void ModelNode::startListening() {
    if (running) {
        return;
    }
    
    running = true;
    receiverThread = std::thread(&ModelNode::receiveLoop, this);
    processorThread = std::thread(&ModelNode::processMessages, this);
    
    std::cout << "[ModelNode] Started listening for messages" << std::endl;
}

void ModelNode::stopListening() {
    if (!running) {
        return;
    }
    
    running = false;
    cv.notify_all();
    
    if (receiverThread.joinable()) {
        receiverThread.join();
    }
    
    if (processorThread.joinable()) {
        processorThread.join();
    }
    
    std::cout << "[ModelNode] Stopped listening for messages" << std::endl;
}

void ModelNode::receiveLoop() {
    while (running) {
        std::string sender_IP, msg;
        int sender_port;
        msg = listener.receiveData(sender_IP, sender_port, 100); // poll with 100ms timeout
        
        if (!msg.empty()) {
            std::lock_guard<std::mutex> lock(mtx);
            messageQueue.push(std::make_tuple(sender_IP, sender_port, msg));
            cv.notify_one();
        }
    }
}

void ModelNode::processMessages() {
    while (running) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !messageQueue.empty() || !running; });
        
        if (!running) {
            break;
        }
        
        while (!messageQueue.empty()) {
            std::string sender_IP, whole_message;
            int sender_port;
            std::tie(sender_IP, sender_port, whole_message) = messageQueue.front();
            messageQueue.pop();
            lock.unlock();
            
            // Parse the message
            std::istringstream iss(whole_message);
            std::string message_type, sessionID;
            iss >> message_type >> sessionID;
            
            // Process based on message type
            if (message_type == "R|") {
                // S-IDA message format: "R|sessionID|seq_num|clove"
                int seq_num;
                iss >> seq_num;
                
                // Read the rest of the stream as the serialized clove
                std::string serializedClove;
                std::getline(iss, serializedClove);
                
                // Process the S-IDA clove
                if (processSIDAClove(sessionID, seq_num, serializedClove)) {
                    // Check if we have enough cloves to reconstruct the message
                    std::string reconstructedMessage = reconstructSIDAMessage(sessionID, seq_num, 3); // Using k=3
                    
                    if (!reconstructedMessage.empty()) {
                        // Process the reconstructed message
                        std::string response = processMessage(sessionID, reconstructedMessage);
                        
                        // Send the response back using S-IDA
                        sendSIDAResponse(sessionID, seq_num, response, sender_IP, sender_port, 4, 3); // n=4, k=3
                    }
                }
            } else if (message_type == "DIRECT|") {
                // Direct message format (for testing): "DIRECT|sessionID|message"
                std::string message;
                std::getline(iss, message);
                
                // Process the message directly
                std::string response = processMessage(sessionID, message);
                
                // Send the direct response back
                std::stringstream ss;
                ss << "RESPONSE|" << sessionID << "|" << response;
                listener.sendData(ss.str());
            } else {
                std::cerr << "[ModelNode] Unknown message type: " << message_type << std::endl;
            }
            
            lock.lock();
        }
    }
}

bool ModelNode::processSIDAClove(const std::string& sessionID, int seq_num, const std::string& serializedClove) {
    // Deserialize the clove
    encrypt_p2p::SIDA::Clove clove;
    try {
        clove = encrypt_p2p::SIDA::deserializeClove(serializedClove);
    } catch (const std::exception& e) {
        std::cerr << "[ModelNode] Failed to deserialize S-IDA clove: " << e.what() << std::endl;
        return false;
    }
    
    // Add the clove to the cache
    {
        std::lock_guard<std::mutex> lock(sidaCacheMtx);
        sidaMessageCache[sessionID][seq_num].push_back(clove);
    }
    
    return true;
}

std::string ModelNode::reconstructSIDAMessage(const std::string& sessionID, int seq_num, int k) {
    std::vector<encrypt_p2p::SIDA::Clove> cloves;
    
    // Get the cloves from the cache
    {
        std::lock_guard<std::mutex> lock(sidaCacheMtx);
        if (sidaMessageCache.find(sessionID) == sidaMessageCache.end() ||
            sidaMessageCache[sessionID].find(seq_num) == sidaMessageCache[sessionID].end() ||
            sidaMessageCache[sessionID][seq_num].size() < static_cast<size_t>(k)) {
            
            // Not enough cloves to reconstruct
            return "";
        }
        
        cloves = sidaMessageCache[sessionID][seq_num];
    }
    
    // Reconstruct the message
    try {
        return encrypt_p2p::SIDA::combine(cloves, k);
    } catch (const std::exception& e) {
        std::cerr << "[ModelNode] Failed to reconstruct S-IDA message: " << e.what() << std::endl;
        return "";
    }
}

bool ModelNode::sendSIDAResponse(const std::string& sessionID, int seq_num, const std::string& message,
                                const std::string& recipient_ip, int recipient_port, int n, int k) {
    //add digest to the message and sign the message
    unsigned char hash[SHA256_DIGEST_LENGTH];

    // SHA-256 context
    std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string message_to_digest = message + "|" + timestamp;
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, message_to_digest.c_str(), message_to_digest.size());
    SHA256_Final(hash, &sha256);    

    std::string digest = std::string(reinterpret_cast<char*>(hash), SHA256_DIGEST_LENGTH);
    std::string signed_digest = encrypt_p2p::encryptRSA(digest, RSA_PRIVATE_KEY);
    std::string message = message + "|" + signed_digest;
    
    // Apply S-IDA to split the message
    std::vector<encrypt_p2p::SIDA::Clove> cloves;
    try {
        cloves = encrypt_p2p::SIDA::split(message, n, k);
    } catch (const std::exception& e) {
        std::cerr << "[ModelNode] S-IDA split failed: " << e.what() << std::endl;
        return false;
    }
    
    // Send each clove back to the same recipient
    // TODO: may need to send through different paths
    for (size_t i = 0; i < cloves.size(); i++) {
        // Serialize the clove
        std::string serializedClove = encrypt_p2p::SIDA::serializeClove(cloves[i]);
        
        // Create the message format: "RESPONSE|sessionID|seq_num|clove_index|total_cloves|clove"
        std::stringstream ss;
        ss << "RESPONSE|" << sessionID << "|" << seq_num << "|" << i << "|" << cloves.size() << "|" << serializedClove;
        std::string responseMsg = ss.str();
        
        // Send the clove
        encrypt_p2p::NetworkHandler handler(ip_address, port + 1 + i); // Use different ports for each clove
        if (!handler.connect(recipient_ip, recipient_port)) {
            std::cerr << "[ModelNode] Failed to connect to " << recipient_ip << ":" << recipient_port << std::endl;
            continue;
        }
        
        if (!handler.sendData(responseMsg)) {
            std::cerr << "[ModelNode] Failed to send S-IDA clove " << i << " of " << cloves.size() << std::endl;
        }
        
        handler.disconnect();
    }
    
    return true;
}

} 