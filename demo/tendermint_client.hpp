#ifndef TENDERMINT_CLIENT_HPP
#define TENDERMINT_CLIENT_HPP

#include <string>
#include <map>
#include <sstream>
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

namespace demo {

/**
 * Simple HTTP client for Tendermint ABCI application
 */
class TendermintClient {
public:
    TendermintClient(const std::string& host = "localhost", int port = 26658)
        : host_(host), port_(port) {}
    
    /**
     * Submit a model response for verification.
     * 
     * @param model_id The model node identifier
     * @param prompt The challenge prompt
     * @param response The model's response
     * @param digest Optional response digest/signature
     * @return JSON response from Tendermint
     */
    std::string submitResponse(const std::string& model_id,
                                const std::string& prompt,
                                const std::string& response,
                                const std::string& digest = "") {
        std::stringstream json;
        json << "{";
        json << "\"type\":\"response\",";
        json << "\"model_id\":\"" << escapeJson(model_id) << "\",";
        json << "\"prompt\":\"" << escapeJson(prompt) << "\",";
        json << "\"response\":\"" << escapeJson(response) << "\",";
        json << "\"digest\":\"" << escapeJson(digest) << "\",";
        json << "\"timestamp\":" << std::time(nullptr);
        json << "}";
        
        return sendTransaction(json.str());
    }
    
    /**
     * Start a new verification epoch.
     */
    std::string startEpoch() {
        return sendTransaction("{\"type\":\"start_epoch\"}");
    }
    
    /**
     * Commit reputation updates after voting.
     */
    std::string commitUpdates() {
        return sendTransaction("{\"type\":\"commit\"}");
    }
    
    /**
     * Register a model node.
     */
    std::string registerModel(const std::string& model_id, const std::string& public_key) {
        std::stringstream json;
        json << "{";
        json << "\"type\":\"register_model\",";
        json << "\"model_id\":\"" << escapeJson(model_id) << "\",";
        json << "\"public_key\":\"" << escapeJson(public_key) << "\"";
        json << "}";
        
        return sendTransaction(json.str());
    }
    
    /**
     * Query current epoch information.
     */
    std::string queryEpoch() {
        return sendQuery("epoch");
    }
    
    /**
     * Query all model reputations.
     */
    std::string queryReputations() {
        return sendQuery("reputations");
    }
    
    /**
     * Query a specific model's reputation.
     */
    std::string queryModelReputation(const std::string& model_id) {
        return sendQuery("model_reputation", "{\"model_id\":\"" + model_id + "\"}");
    }
    
    /**
     * Query trusted models list.
     */
    std::string queryTrustedModels() {
        return sendQuery("trusted_models");
    }
    
    /**
     * Query untrusted models list.
     */
    std::string queryUntrustedModels() {
        return sendQuery("untrusted_models");
    }
    
    /**
     * Health check.
     */
    std::string queryHealth() {
        return sendQuery("health");
    }
    
    /**
     * Check if Tendermint is reachable.
     */
    bool isConnected() {
        try {
            std::string result = queryHealth();
            return result.find("\"status\":\"success\"") != std::string::npos;
        } catch (...) {
            return false;
        }
    }
    
    void setHost(const std::string& host) { host_ = host; }
    void setPort(int port) { port_ = port; }
    std::string getEndpoint() const { return host_ + ":" + std::to_string(port_); }
    
private:
    std::string host_;
    int port_;
    
    /**
     * Send a transaction to the ABCI application.
     */
    std::string sendTransaction(const std::string& tx_json) {
        return httpPost("/tx", tx_json);
    }
    
    /**
     * Send a query to the ABCI application.
     */
    std::string sendQuery(const std::string& path, const std::string& data = "") {
        std::string url = "/query?path=" + path;
        if (!data.empty()) {
            return httpPost(url, data);
        }
        return httpGet(url);
    }
    
    /**
     * Simple HTTP GET request.
     */
    std::string httpGet(const std::string& path) {
        int sock = createSocket();
        if (sock < 0) {
            return "{\"status\":\"error\",\"message\":\"Failed to create socket\"}";
        }
        
        std::stringstream request;
        request << "GET " << path << " HTTP/1.1\r\n";
        request << "Host: " << host_ << ":" << port_ << "\r\n";
        request << "Connection: close\r\n";
        request << "\r\n";
        
        std::string response = sendRequest(sock, request.str());
        close(sock);
        
        return extractBody(response);
    }
    
    /**
     * Simple HTTP POST request.
     */
    std::string httpPost(const std::string& path, const std::string& body) {
        int sock = createSocket();
        if (sock < 0) {
            return "{\"status\":\"error\",\"message\":\"Failed to create socket\"}";
        }
        
        std::stringstream request;
        request << "POST " << path << " HTTP/1.1\r\n";
        request << "Host: " << host_ << ":" << port_ << "\r\n";
        request << "Content-Type: application/json\r\n";
        request << "Content-Length: " << body.length() << "\r\n";
        request << "Connection: close\r\n";
        request << "\r\n";
        request << body;
        
        std::string response = sendRequest(sock, request.str());
        close(sock);
        
        return extractBody(response);
    }
    
    /**
     * Create a TCP socket and connect.
     */
    int createSocket() {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "[TendermintClient] Socket creation failed" << std::endl;
            return -1;
        }
        
        struct hostent* server = gethostbyname(host_.c_str());
        if (!server) {
            std::cerr << "[TendermintClient] Host not found: " << host_ << std::endl;
            close(sock);
            return -1;
        }
        
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
        addr.sin_port = htons(port_);
        
        // Set timeout
        struct timeval timeout;
        timeout.tv_sec = 10;
        timeout.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cerr << "[TendermintClient] Connection failed to " << host_ << ":" << port_ << std::endl;
            close(sock);
            return -1;
        }
        
        return sock;
    }
    
    /**
     * Send request and receive response.
     */
    std::string sendRequest(int sock, const std::string& request) {
        if (send(sock, request.c_str(), request.length(), 0) < 0) {
            return "";
        }
        
        std::string response;
        char buffer[4096];
        int bytes;
        
        while ((bytes = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes] = '\0';
            response += buffer;
        }
        
        return response;
    }
    
    /**
     * Extract body from HTTP response.
     */
    std::string extractBody(const std::string& response) {
        size_t pos = response.find("\r\n\r\n");
        if (pos != std::string::npos) {
            return response.substr(pos + 4);
        }
        return response;
    }
    
    /**
     * Escape string for JSON.
     */
    std::string escapeJson(const std::string& s) {
        std::string result;
        for (char c : s) {
            switch (c) {
                case '"': result += "\\\""; break;
                case '\\': result += "\\\\"; break;
                case '\n': result += "\\n"; break;
                case '\r': result += "\\r"; break;
                case '\t': result += "\\t"; break;
                default: result += c;
            }
        }
        return result;
    }
};

} 

#endif 
