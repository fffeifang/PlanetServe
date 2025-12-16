#ifndef LLAMA_SERVER_CLIENT_HPP
#define LLAMA_SERVER_CLIENT_HPP

#include <string>
#include <vector>
#include <optional>
#include <cstring>
#include <sstream>
#include <cmath>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

namespace llm {

/**
 * Token information from llama-server
 */
struct TokenInfo {
    int id;
    std::string text;
    float logprob;
};

/**
 * Perplexity calculation result
 */
struct PerplexityResult {
    double perplexity = 0.0;
    double credibilityScore = 0.0;
    int tokenCount = 0;
    int matchedTokenCount = 0;
    std::vector<double> tokenProbabilities;
    bool success = false;
    std::string error;
};



class LlamaServerClient {
public:
    LlamaServerClient(const std::string& host = "localhost", int port = 8080)
        : host_(host), port_(port) {}
    
    void setEndpoint(const std::string& host, int port) {
        host_ = host;
        port_ = port;
    }
    
    std::string getEndpoint() const {
        return host_ + ":" + std::to_string(port_);
    }
    
    /**
     * Check if llama-server is healthy
     */
    bool isHealthy() {
        std::string response = httpGet("/health");
        return response.find("ok") != std::string::npos;
    }
    
    /**
     * Tokenize text into tokens
     */
    std::vector<TokenInfo> tokenize(const std::string& text) {
        std::vector<TokenInfo> tokens;
        
        std::string body = "{\"content\":\"" + escapeJson(text) + "\"}";
        std::string response = httpPost("/tokenize", body);
        
        // Parse JSON response: {"tokens":[{"id":123,"piece":"hello"},...]}
        size_t pos = response.find("\"tokens\"");
        if (pos == std::string::npos) return tokens;
        
        // Simple JSON array parsing
        size_t start = response.find('[', pos);
        size_t end = response.find(']', start);
        if (start == std::string::npos || end == std::string::npos) return tokens;
        
        std::string array = response.substr(start + 1, end - start - 1);
        
        // Parse each token object
        size_t objStart = 0;
        while ((objStart = array.find('{', objStart)) != std::string::npos) {
            size_t objEnd = array.find('}', objStart);
            if (objEnd == std::string::npos) break;
            
            std::string obj = array.substr(objStart, objEnd - objStart + 1);
            
            TokenInfo token;
            // Parse id
            size_t idPos = obj.find("\"id\":");
            if (idPos != std::string::npos) {
                token.id = std::stoi(obj.substr(idPos + 5));
            }
            // Parse piece/text
            size_t piecePos = obj.find("\"piece\":\"");
            if (piecePos != std::string::npos) {
                size_t pieceEnd = obj.find("\"", piecePos + 9);
                token.text = obj.substr(piecePos + 9, pieceEnd - piecePos - 9);
            }
            
            tokens.push_back(token);
            objStart = objEnd + 1;
        }
        
        return tokens;
    }
    
    /**
     * Calculate perplexity of a response given a prompt.
     * 
     * This is the SINGLE implementation of perplexity calculation.
     * Algorithm:
     * 1. Tokenize the response
     * 2. For each token position, get P(token | prompt + previous_tokens)
     * 3. PPL = exp(-1/n × Σ log(p_i))
     * 4. Credibility = min(1.0, 1/PPL)
     */
    PerplexityResult calculatePerplexity(const std::string& prompt, 
                                          const std::string& response) {
        PerplexityResult result;
        
        // Tokenize response
        auto tokens = tokenize(response);
        if (tokens.empty()) {
            result.error = "Failed to tokenize response";
            return result;
        }
        
        result.tokenCount = static_cast<int>(tokens.size());
        
        // Build context starting with prompt
        std::string context = prompt;
        double sumLogProb = 0.0;
        const double EPSILON = 1e-10;
        
        // For each token, get the probability
        for (size_t i = 0; i < tokens.size(); i++) {
            // Request completion with logprobs
            std::string body = "{\"prompt\":\"" + escapeJson(context) + "\","
                              "\"n_predict\":1,"
                              "\"temperature\":0,"
                              "\"logprobs\":true,"
                              "\"top_logprobs\":40}";
            
            std::string resp = httpPost("/completion", body);
            
            // Parse logprobs from response
            double tokenProb = EPSILON;
            
            // Look for the token's probability in top_logprobs
            // Format: "top_logprobs":[{"token_id":123,"logprob":-0.5},...]
            size_t logprobsPos = resp.find("\"top_logprobs\"");
            if (logprobsPos != std::string::npos) {
                // Find our token id
                std::string searchId = "\"token_id\":" + std::to_string(tokens[i].id);
                size_t tokenPos = resp.find(searchId, logprobsPos);
                if (tokenPos != std::string::npos) {
                    size_t lpPos = resp.find("\"logprob\":", tokenPos);
                    if (lpPos != std::string::npos) {
                        double logprob = std::stod(resp.substr(lpPos + 10));
                        tokenProb = std::exp(logprob);
                        result.matchedTokenCount++;
                    }
                }
            }
            
            // Clamp probability
            tokenProb = std::max(tokenProb, EPSILON);
            result.tokenProbabilities.push_back(tokenProb);
            sumLogProb += std::log(tokenProb);
            
            // Append token text to context
            context += tokens[i].text;
        }
        
        // Calculate perplexity
        if (!result.tokenProbabilities.empty()) {
            double avgNegLogProb = -sumLogProb / result.tokenProbabilities.size();
            result.perplexity = std::exp(avgNegLogProb);
            result.credibilityScore = std::min(1.0, 1.0 / result.perplexity);
            result.success = true;
        }
        
        return result;
    }
    
private:
    std::string host_;
    int port_;
    
    std::string httpGet(const std::string& path) {
        return httpRequest("GET", path, "");
    }
    
    std::string httpPost(const std::string& path, const std::string& body) {
        return httpRequest("POST", path, body);
    }
    
    std::string httpRequest(const std::string& method, 
                            const std::string& path,
                            const std::string& body) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) return "";
        
        struct hostent* server = gethostbyname(host_.c_str());
        if (!server) {
            close(sock);
            return "";
        }
        
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
        addr.sin_port = htons(port_);
        
        // Set timeout
        struct timeval timeout;
        timeout.tv_sec = 30;
        timeout.tv_usec = 0;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        
        if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock);
            return "";
        }
        
        // Build HTTP request
        std::stringstream req;
        req << method << " " << path << " HTTP/1.1\r\n";
        req << "Host: " << host_ << ":" << port_ << "\r\n";
        if (!body.empty()) {
            req << "Content-Type: application/json\r\n";
            req << "Content-Length: " << body.length() << "\r\n";
        }
        req << "Connection: close\r\n\r\n";
        req << body;
        
        std::string request = req.str();
        send(sock, request.c_str(), request.length(), 0);
        
        // Read response
        std::string response;
        char buffer[4096];
        int bytes;
        while ((bytes = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes] = '\0';
            response += buffer;
        }
        
        close(sock);
        
        // Extract body
        size_t bodyStart = response.find("\r\n\r\n");
        if (bodyStart != std::string::npos) {
            return response.substr(bodyStart + 4);
        }
        return response;
    }
    
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
