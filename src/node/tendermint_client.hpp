#ifndef TENDERMINT_CLIENT_HPP
#define TENDERMINT_CLIENT_HPP

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include "../encrypt_p2p/crypto_utils.hpp"
#include "../encrypt_p2p/key_generation.hpp"

namespace node {

class TendermintClient {
public:
    // Node information structure
    struct NodeInfo {
        std::string ip_address;
        int port;
        std::string rsa_public_key;
        double reputation;          // For model nodes
        bool is_trusted;            // For model nodes
    };

    // Signed node list with BFT consensus proof
    struct SignedNodeList {
        std::vector<NodeInfo> user_nodes;
        std::vector<NodeInfo> model_nodes;
        std::string digest;         // SHA-256 hash of the list
        std::vector<std::string> verifier_signatures;  // Signatures from verification nodes
        int64_t epoch;              // Epoch when this list was generated
        int64_t timestamp;          // Unix timestamp
        bool is_valid;              // Whether the list has >= 2n/3+1 signatures
    };

    // Epoch information
    struct EpochInfo {
        int64_t epoch_number;
        std::string state;          // CHALLENGE, RESPONSE, EVALUATION, VOTING, COMMITTED
        std::string leader;
    };

    // Reputation query result
    struct ReputationInfo {
        std::string model_id;
        double reputation;
        bool is_trusted;
        std::vector<double> score_history;
    };

    TendermintClient();
    TendermintClient(const std::string& tendermint_host, int tendermint_port);
    ~TendermintClient();

    // Configure the Tendermint server endpoint
    void setEndpoint(const std::string& host, int port);

    // Check if the Tendermint application is healthy
    bool checkHealth();

       
    // Transaction API (POST /tx)
       

    // Start a new verification epoch (leader only)
    bool startEpoch();

    bool registerModel(const std::string& model_id, const std::string& public_key);

    bool registerUserNode(const std::string& ip_address, int port, 
                          const std::string& rsa_public_key);

    bool submitResponse(const std::string& model_id, const std::string& prompt,
                        const std::string& response, const std::string& digest);

    bool submitVote(const std::string& voter_id, const std::string& model_id,
                    double proposed_score, bool approve, const std::string& signature);

    bool commitUpdates();

       
    // Query API (GET /query)
       

    // Get current epoch information
    std::optional<EpochInfo> queryEpoch();

    // Get all model reputations
    std::map<std::string, double> queryReputations();

    // Get reputation for a specific model
    std::optional<ReputationInfo> queryModelReputation(const std::string& model_id);

    // Get list of trusted model IDs
    std::vector<std::string> queryTrustedModels();

    // Get list of untrusted model IDs
    std::vector<std::string> queryUntrustedModels();

    // Get verification node information
    std::vector<std::string> queryVerificationNodes();

       
    // Signed Node List API (with BFT proof)
       

    SignedNodeList getSignedNodeList(int min_signatures = 0);

   
    bool verifySignedNodeList(const SignedNodeList& node_list,
                              const std::map<std::string, std::string>& verifier_public_keys);

    // Get the last error message
    std::string getLastError() const { return last_error_; }

    // Send HTTP POST request (public for custom transactions)
    std::optional<nlohmann::json> httpPost(const std::string& endpoint, 
                                           const nlohmann::json& payload);

    // Parse JSON response and check for errors
    bool checkResponse(const nlohmann::json& response);

private:
    std::string tendermint_host_;
    int tendermint_port_;
    std::string last_error_;

    // CURL callback for receiving response
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp);

    // Send HTTP GET request
    std::optional<nlohmann::json> httpGet(const std::string& endpoint,
                                          const std::map<std::string, std::string>& params = {});

    // Build URL from endpoint and parameters
    std::string buildUrl(const std::string& endpoint, 
                         const std::map<std::string, std::string>& params = {});
};

   
// Implementation
   

inline size_t TendermintClient::WriteCallback(void* contents, size_t size, 
                                              size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* response = static_cast<std::string*>(userp);
    response->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

inline TendermintClient::TendermintClient() 
    : tendermint_host_("localhost"), tendermint_port_(26658) {
}

inline TendermintClient::TendermintClient(const std::string& tendermint_host, int tendermint_port)
    : tendermint_host_(tendermint_host), tendermint_port_(tendermint_port) {
}

inline TendermintClient::~TendermintClient() {
}

inline void TendermintClient::setEndpoint(const std::string& host, int port) {
    tendermint_host_ = host;
    tendermint_port_ = port;
}

inline std::string TendermintClient::buildUrl(const std::string& endpoint,
                                              const std::map<std::string, std::string>& params) {
    std::string url = "http://" + tendermint_host_ + ":" + 
                      std::to_string(tendermint_port_) + endpoint;
    
    if (!params.empty()) {
        url += "?";
        bool first = true;
        for (const auto& [key, value] : params) {
            if (!first) url += "&";
            url += key + "=" + value;
            first = false;
        }
    }
    return url;
}

inline std::optional<nlohmann::json> TendermintClient::httpPost(
    const std::string& endpoint, const nlohmann::json& payload) {
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        last_error_ = "Failed to initialize CURL";
        return std::nullopt;
    }

    std::string url = buildUrl(endpoint);
    std::string payloadStr = payload.dump();
    std::string responseStr;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payloadStr.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseStr);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        last_error_ = "CURL error: " + std::string(curl_easy_strerror(res));
        return std::nullopt;
    }

    try {
        return nlohmann::json::parse(responseStr);
    } catch (const std::exception& e) {
        last_error_ = "JSON parse error: " + std::string(e.what());
        return std::nullopt;
    }
}

inline std::optional<nlohmann::json> TendermintClient::httpGet(
    const std::string& endpoint, const std::map<std::string, std::string>& params) {
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        last_error_ = "Failed to initialize CURL";
        return std::nullopt;
    }

    std::string url = buildUrl(endpoint, params);
    std::string responseStr;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseStr);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        last_error_ = "CURL error: " + std::string(curl_easy_strerror(res));
        return std::nullopt;
    }

    try {
        return nlohmann::json::parse(responseStr);
    } catch (const std::exception& e) {
        last_error_ = "JSON parse error: " + std::string(e.what());
        return std::nullopt;
    }
}

inline bool TendermintClient::checkResponse(const nlohmann::json& response) {
    if (response.contains("status") && response["status"] == "success") {
        return true;
    }
    if (response.contains("message")) {
        last_error_ = response["message"].get<std::string>();
    }
    return false;
}

inline bool TendermintClient::checkHealth() {
    auto response = httpGet("/query", {{"path", "health"}});
    if (!response) return false;
    
    if (response->contains("data") && (*response)["data"].contains("app_healthy")) {
        return (*response)["data"]["app_healthy"].get<bool>();
    }
    return false;
}

inline bool TendermintClient::startEpoch() {
    nlohmann::json payload = {{"type", "start_epoch"}};
    auto response = httpPost("/tx", payload);
    return response && checkResponse(*response);
}

inline bool TendermintClient::registerModel(const std::string& model_id, 
                                            const std::string& public_key) {
    nlohmann::json payload = {
        {"type", "register_model"},
        {"model_id", model_id},
        {"public_key", public_key}
    };
    auto response = httpPost("/tx", payload);
    return response && checkResponse(*response);
}

inline bool TendermintClient::registerUserNode(const std::string& ip_address, int port,
                                               const std::string& rsa_public_key) {
    nlohmann::json payload = {
        {"type", "register_user"},
        {"ip_address", ip_address},
        {"port", port},
        {"rsa_public_key", rsa_public_key}
    };
    auto response = httpPost("/tx", payload);
    return response && checkResponse(*response);
}

inline bool TendermintClient::submitResponse(const std::string& model_id,
                                             const std::string& prompt,
                                             const std::string& response_text,
                                             const std::string& digest) {
    nlohmann::json payload = {
        {"type", "response"},
        {"model_id", model_id},
        {"prompt", prompt},
        {"response", response_text},
        {"digest", digest},
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()}
    };
    auto response = httpPost("/tx", payload);
    return response && checkResponse(*response);
}

inline bool TendermintClient::submitVote(const std::string& voter_id,
                                         const std::string& model_id,
                                         double proposed_score, bool approve,
                                         const std::string& signature) {
    nlohmann::json payload = {
        {"type", "vote"},
        {"voter_id", voter_id},
        {"model_id", model_id},
        {"proposed_score", proposed_score},
        {"approve", approve},
        {"signature", signature}
    };
    auto response = httpPost("/tx", payload);
    return response && checkResponse(*response);
}

inline bool TendermintClient::commitUpdates() {
    nlohmann::json payload = {{"type", "commit"}};
    auto response = httpPost("/tx", payload);
    return response && checkResponse(*response);
}

inline std::optional<TendermintClient::EpochInfo> TendermintClient::queryEpoch() {
    auto response = httpGet("/query", {{"path", "epoch"}});
    if (!response || !checkResponse(*response)) return std::nullopt;
    
    try {
        auto& data = (*response)["data"];
        return EpochInfo{
            data["epoch"].get<int64_t>(),
            data["state"].get<std::string>(),
            data["leader"].get<std::string>()
        };
    } catch (...) {
        last_error_ = "Failed to parse epoch info";
        return std::nullopt;
    }
}

inline std::map<std::string, double> TendermintClient::queryReputations() {
    std::map<std::string, double> result;
    auto response = httpGet("/query", {{"path", "reputations"}});
    if (!response || !checkResponse(*response)) return result;
    
    try {
        auto& data = (*response)["data"];
        for (auto& [key, value] : data.items()) {
            result[key] = value.get<double>();
        }
    } catch (...) {
        last_error_ = "Failed to parse reputations";
    }
    return result;
}

inline std::optional<TendermintClient::ReputationInfo> 
TendermintClient::queryModelReputation(const std::string& model_id) {
    auto response = httpGet("/query", {
        {"path", "model_reputation"},
        {"data", "{\"model_id\":\"" + model_id + "\"}"}
    });
    if (!response || !checkResponse(*response)) return std::nullopt;
    
    try {
        auto& data = (*response)["data"];
        ReputationInfo info;
        info.model_id = data["model_id"].get<std::string>();
        info.reputation = data["reputation"].get<double>();
        info.is_trusted = data["trusted"].get<bool>();
        
        for (auto& score : data["history"]) {
            info.score_history.push_back(score.get<double>());
        }
        return info;
    } catch (...) {
        last_error_ = "Failed to parse model reputation";
        return std::nullopt;
    }
}

inline std::vector<std::string> TendermintClient::queryTrustedModels() {
    std::vector<std::string> result;
    auto response = httpGet("/query", {{"path", "trusted_models"}});
    if (!response || !checkResponse(*response)) return result;
    
    try {
        auto& data = (*response)["data"];
        for (auto& model_id : data) {
            result.push_back(model_id.get<std::string>());
        }
    } catch (...) {
        last_error_ = "Failed to parse trusted models";
    }
    return result;
}

inline std::vector<std::string> TendermintClient::queryUntrustedModels() {
    std::vector<std::string> result;
    auto response = httpGet("/query", {{"path", "untrusted_models"}});
    if (!response || !checkResponse(*response)) return result;
    
    try {
        auto& data = (*response)["data"];
        for (auto& model_id : data) {
            result.push_back(model_id.get<std::string>());
        }
    } catch (...) {
        last_error_ = "Failed to parse untrusted models";
    }
    return result;
}

inline std::vector<std::string> TendermintClient::queryVerificationNodes() {
    std::vector<std::string> result;
    auto response = httpGet("/query", {{"path", "verification_nodes"}});
    if (!response || !checkResponse(*response)) return result;
    
    try {
        auto& data = (*response)["data"]["nodes"];
        for (auto& node : data) {
            result.push_back(node.get<std::string>());
        }
    } catch (...) {
        last_error_ = "Failed to parse verification nodes";
    }
    return result;
}

inline TendermintClient::SignedNodeList TendermintClient::getSignedNodeList(int min_signatures) {
    SignedNodeList result;
    result.is_valid = false;
    
    // Query trusted models
    auto trustedModels = queryTrustedModels();
    auto reputations = queryReputations();
    
    // Build model nodes list
    for (const auto& model_id : trustedModels) {
        auto repInfo = queryModelReputation(model_id);
        if (repInfo) {
            NodeInfo node;
            node.ip_address = model_id;  // model_id is typically IP address
            node.port = 8080;  // Default model node port
            node.reputation = repInfo->reputation;
            node.is_trusted = repInfo->is_trusted;
            result.model_nodes.push_back(node);
        }
    }
    
    // Query epoch for timestamp
    auto epochInfo = queryEpoch();
    if (epochInfo) {
        result.epoch = epochInfo->epoch_number;
    }
    
    result.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Create digest of the node list
    nlohmann::json listJson;
    for (const auto& node : result.model_nodes) {
        listJson["model_nodes"].push_back({
            {"ip", node.ip_address},
            {"port", node.port},
            {"reputation", node.reputation}
        });
    }
    for (const auto& node : result.user_nodes) {
        listJson["user_nodes"].push_back({
            {"ip", node.ip_address},
            {"port", node.port},
            {"public_key", node.rsa_public_key}
        });
    }
    listJson["epoch"] = result.epoch;
    listJson["timestamp"] = result.timestamp;
    
    // Calculate SHA-256 digest
    std::string listStr = listJson.dump();
    result.digest = encrypt_p2p::toHex(listStr);  // Simplified - should use actual SHA-256
    
    // TODO: In a real implementation, query each verification node for their signature
    // and verify we have >= 2n/3+1 valid signatures
    // For now, mark as valid if we got the data
    result.is_valid = !result.model_nodes.empty();
    
    return result;
}

inline bool TendermintClient::verifySignedNodeList(
    const SignedNodeList& node_list,
    const std::map<std::string, std::string>& verifier_public_keys) {
    
    // TODO:  signature verification
    
    if (node_list.verifier_signatures.size() < verifier_public_keys.size() * 2 / 3) {
        last_error_ = "Not enough signatures for BFT consensus";
        return false;
    }
    
    // Simplified verification for now
    return node_list.is_valid;
}

} 

#endif 

