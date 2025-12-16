#ifndef IDA_HPP
#define IDA_HPP

#include <string>
#include <vector>
#include <random>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <set>
#include "crypto_utils.hpp"
#include "key_generation.hpp"

namespace encrypt_p2p {
// for testing purposes!!
class IDA {
public:
    struct Clove {
        std::vector<uint8_t> fragment;  // The encrypted data
        std::pair<uint8_t, std::vector<uint8_t>> keyShare;  // share of the AES key
        uint32_t originalDataSize;      // track original data size if needed
    };

    // Split into n cloves
    static std::vector<Clove> split(const std::string& message, int n, int k) {
        if (n < k || k <= 0 || n <= 0) {
            throw std::runtime_error("Invalid parameters: n >= k > 0 required");
        }

        // all zeros test
        std::string aesKey(32, '\0');
        

        if (RAND_bytes(reinterpret_cast<unsigned char*>(&aesKey[0]), aesKey.size()) != 1) {
            aesKey = std::string(32, 'A');
        }
        
        std::cerr << "[DEBUG] Original AES key = " << toHex(aesKey) << std::endl;
        
        std::string encryptedHex = encryptAES(message, aesKey);
        std::cerr << "Encrypted hex size: " << encryptedHex.size() << " bytes" << std::endl;
        
        // Create n shares with the same key
        std::vector<Clove> cloves;
        cloves.reserve(n);
        
        for (int i = 1; i <= n; i++) {
            Clove clove;
            
            std::vector<uint8_t> keyBytes(aesKey.begin(), aesKey.end());
            clove.keyShare = std::make_pair(static_cast<uint8_t>(i), keyBytes);
            
            std::vector<uint8_t> encryptedBytes(encryptedHex.begin(), encryptedHex.end());
            clove.fragment = encryptedBytes;
            clove.originalDataSize = static_cast<uint32_t>(encryptedHex.size());
            
            cloves.push_back(clove);
        }
        
        return cloves;
    }

    static std::string combine(const std::vector<Clove>& cloves, int k) {
        if ((int)cloves.size() < k || k <= 0) {
            throw std::runtime_error("Not enough cloves to combine");
        }
        
        const Clove& firstClove = cloves[0];
        
        const auto& keyShare = firstClove.keyShare;
        std::string aesKey(keyShare.second.begin(), keyShare.second.end());
        
        std::cerr << "[DEBUG] Reconstructed key = " << toHex(aesKey) << std::endl;
        
        std::string encryptedHex(firstClove.fragment.begin(), firstClove.fragment.end());
        
        try {
            std::string decrypted = decryptAES(encryptedHex, aesKey);
            return decrypted;
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to decrypt: " + std::string(e.what()));
        }
    }
    
    static std::string toHex(const std::string& input) {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (unsigned char c : input) {
            ss << std::setw(2) << static_cast<int>(c);
        }
        return ss.str();
    }
};

} 

#endif 