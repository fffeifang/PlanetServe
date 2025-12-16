#ifndef DEMO_MESSAGE_TYPES_HPP
#define DEMO_MESSAGE_TYPES_HPP

#include <string>
#include <sstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>

namespace demo {

      
// Message Type Constants
      

// Message type prefixes for protocol identification
constexpr const char* MSG_USER_PROMPT_SHARE = "USER_PROMPT_SHARE";
constexpr const char* MSG_VERIF_CHALLENGE_SHARE = "VERIF_CHALLENGE_SHARE";
constexpr const char* MSG_MODEL_REPLY_SHARE = "MODEL_REPLY_SHARE";
constexpr const char* MSG_DIRECT_PROMPT = "DIRECT_PROMPT";
constexpr const char* MSG_DIRECT_REPLY = "DIRECT_REPLY";

      
// Message Structures
      

/**
 * DemoMessage - Wire format for messages in the demo
 * 
 * Format: TYPE|request_id|path_id|hop_id|share_index|n|k|payload
 * 
 * TYPE: Message type identifier
 * request_id: Unique request identifier
 * path_id: Which path (0-3) this message is on
 * hop_id: Which hop in the path (0-2)
 * share_index: S-IDA share index (1-based, matches clove index)
 * n: Total number of shares
 * k: Threshold for reconstruction
 * payload: Serialized clove or message content
 */
struct DemoMessage {
    std::string type;
    std::string request_id;
    int path_id;
    int hop_id;
    int share_index;
    int n;
    int k;
    std::string payload;
    
    // Sender information (filled by receiver)
    std::string sender_ip;
    int sender_port = 0;
    
    // Serialize to wire format
    std::string serialize() const {
        std::stringstream ss;
        ss << type << "|"
           << request_id << "|"
           << path_id << "|"
           << hop_id << "|"
           << share_index << "|"
           << n << "|"
           << k << "|"
           << payload;
        return ss.str();
    }
    
    // Deserialize from wire format
    static DemoMessage deserialize(const std::string& data) {
        DemoMessage msg;
        std::istringstream iss(data);
        std::string token;
        
        // Parse each field
        if (std::getline(iss, token, '|')) msg.type = token;
        if (std::getline(iss, token, '|')) msg.request_id = token;
        if (std::getline(iss, token, '|')) msg.path_id = std::stoi(token);
        if (std::getline(iss, token, '|')) msg.hop_id = std::stoi(token);
        if (std::getline(iss, token, '|')) msg.share_index = std::stoi(token);
        if (std::getline(iss, token, '|')) msg.n = std::stoi(token);
        if (std::getline(iss, token, '|')) msg.k = std::stoi(token);
        
        // Everything else is payload
        std::getline(iss, msg.payload);
        
        return msg;
    }
    
    // Check if message is valid
    bool isValid() const {
        return !type.empty() && !request_id.empty() && 
               n > 0 && k > 0 && k <= n;
    }
};

      
// Request ID Generation
      

inline std::string generateRequestId() {
    // Format: timestamp_random
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    std::stringstream ss;
    ss << "req_" << millis << "_" << dis(gen);
    return ss.str();
}

      
// Reply Aggregation Helper
      

/**
 * ReplyAggregator - Collects response shares and tracks reconstruction status
 */
struct ReplyAggregator {
    std::string request_id;
    int expected_shares;
    int threshold;
    std::vector<std::pair<int, std::string>> received_shares;  // <share_index, serialized_clove>
    bool reconstructed = false;
    std::string result;
    
    ReplyAggregator() : expected_shares(4), threshold(3) {}
    
    ReplyAggregator(const std::string& req_id, int n, int k) 
        : request_id(req_id), expected_shares(n), threshold(k) {}
    
    // Add a received share
    bool addShare(int share_index, const std::string& serialized_clove) {
        // Check for duplicate
        for (const auto& share : received_shares) {
            if (share.first == share_index) {
                return false;  // Duplicate
            }
        }
        
        received_shares.emplace_back(share_index, serialized_clove);
        return true;
    }
    
    // Check if we have enough shares
    bool hasEnoughShares() const {
        return static_cast<int>(received_shares.size()) >= threshold;
    }
    
    // Get status string
    std::string getStatus() const {
        std::stringstream ss;
        ss << "received " << received_shares.size() << "/" << expected_shares 
           << " shares (need " << threshold << ")";
        return ss.str();
    }
};

} 

#endif 
