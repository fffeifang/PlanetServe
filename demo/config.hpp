#ifndef DEMO_CONFIG_HPP
#define DEMO_CONFIG_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <map>

namespace demo {

 
// Configuration Structures
 

struct RelayHop {
    std::string ip;
    int port;
};

struct RelayPath {
    int path_id;
    std::vector<RelayHop> hops;  // 3 hops per path
};

struct DemoConfig {
    // S-IDA parameters
    int n = 4;  // Number of paths/shares
    int k = 3;  // Threshold for reconstruction
    
    // Model node
    std::string model_node_ip = "127.0.0.1";
    int model_node_port = 9000;
    std::string model_path = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";
    
    // Verification node
    std::string verifier_node_ip = "127.0.0.1";
    int verifier_node_port = 9100;
    
    // User node
    std::string user_node_ip = "127.0.0.1";
    int user_node_port = 9200;
    
    // Relay paths (4 paths × 3 hops = 12 relays)
    std::vector<RelayPath> relay_paths;
    
    // Logging
    bool verbose = true;
    
    // Initialize with default relay configuration
    void initDefaultRelays() {
        relay_paths.clear();
        
        // Path 0: ports 9300, 9301, 9302
        RelayPath path0;
        path0.path_id = 0;
        path0.hops = {
            {"127.0.0.1", 9300},
            {"127.0.0.1", 9301},
            {"127.0.0.1", 9302}
        };
        relay_paths.push_back(path0);
        
        // Path 1: ports 9310, 9311, 9312
        RelayPath path1;
        path1.path_id = 1;
        path1.hops = {
            {"127.0.0.1", 9310},
            {"127.0.0.1", 9311},
            {"127.0.0.1", 9312}
        };
        relay_paths.push_back(path1);
        
        // Path 2: ports 9320, 9321, 9322
        RelayPath path2;
        path2.path_id = 2;
        path2.hops = {
            {"127.0.0.1", 9320},
            {"127.0.0.1", 9321},
            {"127.0.0.1", 9322}
        };
        relay_paths.push_back(path2);
        
        // Path 3: ports 9330, 9331, 9332
        RelayPath path3;
        path3.path_id = 3;
        path3.hops = {
            {"127.0.0.1", 9330},
            {"127.0.0.1", 9331},
            {"127.0.0.1", 9332}
        };
        relay_paths.push_back(path3);
    }
    
    // Get all relay ports as a flat list
    std::vector<std::pair<int, int>> getAllRelayPorts() const {
        std::vector<std::pair<int, int>> result;  // <path_id, port>
        for (const auto& path : relay_paths) {
            for (size_t hop = 0; hop < path.hops.size(); hop++) {
                result.emplace_back(path.path_id, path.hops[hop].port);
            }
        }
        return result;
    }
    
    // Get relay info for a specific port
    bool getRelayInfo(int port, int& path_id, int& hop_id, 
                      std::string& next_ip, int& next_port) const {
        for (const auto& path : relay_paths) {
            for (size_t hop = 0; hop < path.hops.size(); hop++) {
                if (path.hops[hop].port == port) {
                    path_id = path.path_id;
                    hop_id = static_cast<int>(hop);
                    
                    // Determine next hop
                    if (hop + 1 < path.hops.size()) {
                        // Forward to next relay in path
                        next_ip = path.hops[hop + 1].ip;
                        next_port = path.hops[hop + 1].port;
                    } else {
                        // Last hop forwards to model node
                        next_ip = model_node_ip;
                        next_port = model_node_port;
                    }
                    return true;
                }
            }
        }
        return false;
    }
    
    // Get the first hop of each path (for user node to send to)
    std::vector<RelayHop> getFirstHops() const {
        std::vector<RelayHop> result;
        for (const auto& path : relay_paths) {
            if (!path.hops.empty()) {
                result.push_back(path.hops[0]);
            }
        }
        return result;
    }
};

 
// YAML Config Parser (supports nested relay_paths)
 

class ConfigLoader {
public:
    static DemoConfig loadFromFile(const std::string& filename) {
        DemoConfig config;
        
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "[ConfigLoader] Warning: Cannot open " << filename 
                      << ", using defaults" << std::endl;
            config.initDefaultRelays();
            return config;
        }
        
        std::string line;
        std::string current_section;
        bool in_relay_paths = false;
        bool in_hops = false;
        RelayPath current_path;
        RelayHop current_hop;
        bool relay_paths_found = false;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            std::string raw_line = line;
            if (line.empty()) continue;
            
            // Count leading spaces for indentation
            size_t indent = line.find_first_not_of(" ");
            if (indent == std::string::npos) continue;
            if (line[indent] == '#') continue;
            
            // Trim the line
            line = line.substr(indent);
            if (line.empty()) continue;
            
            // Top-level sections (no indentation or minimal)
            if (indent == 0) {
                // Save current path if we were building one
                if (in_relay_paths && !current_path.hops.empty()) {
                    config.relay_paths.push_back(current_path);
                    current_path = RelayPath();
                }
                in_relay_paths = false;
                in_hops = false;
                
                if (line.find("sida:") == 0) {
                    current_section = "sida";
                } else if (line.find("model_node:") == 0) {
                    current_section = "model_node";
                } else if (line.find("verifier_node:") == 0) {
                    current_section = "verifier_node";
                } else if (line.find("user_node:") == 0) {
                    current_section = "user_node";
                } else if (line.find("relay_paths:") == 0) {
                    current_section = "relay_paths";
                    in_relay_paths = true;
                    relay_paths_found = true;
                    config.relay_paths.clear();  // Clear defaults, use YAML
                } else if (line.find("logging:") == 0) {
                    current_section = "logging";
                }
                continue;
            }
            
            // Parse relay_paths section with proper nesting
            if (current_section == "relay_paths" && in_relay_paths) {
                // New path entry: "- path_id: N"
                if (line.find("- path_id:") == 0) {
                    // Save previous path
                    if (!current_path.hops.empty()) {
                        config.relay_paths.push_back(current_path);
                    }
                    current_path = RelayPath();
                    in_hops = false;
                    
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string val = line.substr(colon + 1);
                        val.erase(0, val.find_first_not_of(" \t"));
                        val.erase(val.find_last_not_of(" \t\r\n") + 1);
                        current_path.path_id = std::stoi(val);
                    }
                    continue;
                }
                
                // Hops section
                if (line.find("hops:") == 0) {
                    in_hops = true;
                    continue;
                }
                
                // New hop entry: "- ip: X.X.X.X"
                if (in_hops && line.find("- ip:") == 0) {
                    // Save previous hop if exists
                    if (current_hop.port > 0) {
                        current_path.hops.push_back(current_hop);
                    }
                    current_hop = RelayHop();
                    
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string val = line.substr(colon + 1);
                        val.erase(0, val.find_first_not_of(" \t"));
                        val.erase(val.find_last_not_of(" \t\r\n") + 1);
                        current_hop.ip = val;
                    }
                    continue;
                }
                
                // Port for current hop
                if (in_hops && line.find("port:") == 0) {
                    size_t colon = line.find(':');
                    if (colon != std::string::npos) {
                        std::string val = line.substr(colon + 1);
                        val.erase(0, val.find_first_not_of(" \t"));
                        val.erase(val.find_last_not_of(" \t\r\n") + 1);
                        current_hop.port = std::stoi(val);
                    }
                    continue;
                }
                
                continue;
            }
            
            // Parse key-value pairs for other sections
            size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            
            std::string key = line.substr(0, colon);
            std::string value = line.substr(colon + 1);
            
            // Trim
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            if (!value.empty()) {
                size_t end = value.find_last_not_of(" \t\r\n");
                if (end != std::string::npos) value = value.substr(0, end + 1);
            }
            
            // Apply values based on section
            if (current_section == "sida") {
                if (key == "n") config.n = std::stoi(value);
                else if (key == "k") config.k = std::stoi(value);
            } else if (current_section == "model_node") {
                if (key == "ip") config.model_node_ip = value;
                else if (key == "port") config.model_node_port = std::stoi(value);
                else if (key == "model_path") config.model_path = value;
            } else if (current_section == "verifier_node") {
                if (key == "ip") config.verifier_node_ip = value;
                else if (key == "port") config.verifier_node_port = std::stoi(value);
            } else if (current_section == "user_node") {
                if (key == "ip") config.user_node_ip = value;
                else if (key == "port") config.user_node_port = std::stoi(value);
            } else if (current_section == "logging") {
                if (key == "verbose") config.verbose = (value == "true");
            }
        }
        
        // Save last hop and path (only if we have valid data)
        if (in_relay_paths) {
            if (current_hop.port > 0 && !current_hop.ip.empty()) {
                current_path.hops.push_back(current_hop);
            }
            if (!current_path.hops.empty() && current_path.path_id >= 0) {
                config.relay_paths.push_back(current_path);
            }
        }
        
        file.close();
        
        // Use defaults if no relay paths found in YAML
        if (config.relay_paths.empty()) {
            std::cerr << "[ConfigLoader] No relay_paths in YAML, using defaults" << std::endl;
            config.initDefaultRelays();
        } else {
            std::cout << "[ConfigLoader] Loaded " << config.relay_paths.size() 
                      << " relay paths from " << filename << std::endl;
        }
        
        return config;
    }
    
    // Load with defaults if file not found
    static DemoConfig loadOrDefault(const std::string& filename = "configs/demo_local.yaml") {
        DemoConfig config;
        try {
            config = loadFromFile(filename);
        } catch (const std::exception& e) {
            std::cerr << "[ConfigLoader] Error: " << e.what() << ", using defaults" << std::endl;
            config.initDefaultRelays();
        }
        
        // Ensure relay paths are initialized
        if (config.relay_paths.empty()) {
            config.initDefaultRelays();
        }
        
        return config;
    }
};

 
// Logging Utilities
 

#define LOG_INFO(node_type, msg) \
    std::cout << "[" << node_type << "] " << msg << std::endl

#define LOG_DEBUG(node_type, msg) \
    if (verbose) std::cout << "[" << node_type << "][DEBUG] " << msg << std::endl

#define LOG_ERROR(node_type, msg) \
    std::cerr << "[" << node_type << "][ERROR] " << msg << std::endl

#define LOG_WARN(node_type, msg) \
    std::cout << "[" << node_type << "][WARN] " << msg << std::endl

// Structured log entry for demo requirements
struct LogEntry {
    std::string request_id;
    std::string msg_type;        // USER_PROMPT_SHARE, VERIF_CHALLENGE_SHARE, MODEL_REPLY, RELAY_FORWARD
    int path_id = -1;
    int hop_id = -1;
    int share_index = -1;
    int n = 4;
    int k = 3;
    std::string next_hop;
    std::string status;          // RECEIVED, FORWARDED, RECONSTRUCTED, SENT
    
    std::string toString() const {
        std::stringstream ss;
        ss << "{";
        ss << "\"request_id\":\"" << request_id << "\"";
        ss << ",\"msg_type\":\"" << msg_type << "\"";
        if (path_id >= 0) ss << ",\"path_id\":" << path_id;
        if (hop_id >= 0) ss << ",\"hop_id\":" << hop_id;
        if (share_index >= 0) ss << ",\"share_index\":" << share_index;
        ss << ",\"n\":" << n << ",\"k\":" << k;
        if (!next_hop.empty()) ss << ",\"next_hop\":\"" << next_hop << "\"";
        if (!status.empty()) ss << ",\"status\":\"" << status << "\"";
        ss << "}";
        return ss.str();
    }
    
    void print(const std::string& node_type) const {
        std::cout << "[" << node_type << "][LOG] " << toString() << std::endl;
    }
};

} 

#endif 
