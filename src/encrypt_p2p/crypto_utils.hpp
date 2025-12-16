#ifndef CRYPTO_UTILS_HPP
#define CRYPTO_UTILS_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <vector>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/bio.h>
#include <openssl/rand.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace encrypt_p2p {
// Fixed IV for demonstration (16 bytes for AES-256-CBC)
static const unsigned char FIXED_IV[16] = {
    0x00, 0x01, 0x02, 0x03,
    0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B,
    0x0C, 0x0D, 0x0E, 0x0F
};

inline std::string toHex(const std::string& input) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (unsigned char c : input) {
        ss << std::setw(2) << static_cast<int>(c);
    }
    return ss.str();
}

inline std::string fromHex(const std::string& hex) {
    std::string result;
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byte = hex.substr(i, 2);
        char c = static_cast<char>(std::stoi(byte, nullptr, 16));
        result.push_back(c);
    }
    return result;
}

inline std::string generateIV() {
    unsigned char iv[16];
    if (RAND_bytes(iv, sizeof(iv)) != 1) {
        throw std::runtime_error("Failed to generate random IV");
    }
    return std::string(reinterpret_cast<char*>(iv), sizeof(iv));
}

inline std::string encryptAES(const std::string& plainText, const std::string& key) {
    // Generate IV
    std::string iv = generateIV();
    
    // Create and initialize the context
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        throw std::runtime_error("Failed to create cipher context");
    }
    
    std::string aesKey = key;
    if (aesKey.size() != 32) {
        if (aesKey.size() < 32) {
            aesKey.append(32 - aesKey.size(), '\0');  // Pad with null bytes
        } else {
            aesKey.resize(32);  // Truncate if too long
        }
    }
    
    // Initialize the encryption operation
    if (EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, 
                         reinterpret_cast<const unsigned char*>(aesKey.c_str()), 
                         reinterpret_cast<const unsigned char*>(iv.c_str())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Failed to initialize encryption");
    }
    
    std::vector<unsigned char> cipherText(plainText.size() + EVP_CIPHER_block_size(EVP_aes_256_cbc()));
    int len = 0;
    int cipherTextLen = 0;
    
    if (EVP_EncryptUpdate(ctx, cipherText.data(), &len, 
                       reinterpret_cast<const unsigned char*>(plainText.c_str()), 
                       static_cast<int>(plainText.size())) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Failed to encrypt data");
    }
    cipherTextLen = len;
    
    if (EVP_EncryptFinal_ex(ctx, cipherText.data() + len, &len) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("Failed to finalize encryption");
    }
    cipherTextLen += len;
    
    // Clean up
    EVP_CIPHER_CTX_free(ctx);
    
    std::string result = iv + std::string(reinterpret_cast<char*>(cipherText.data()), cipherTextLen);
    
    return toHex(result);
}

inline std::string decryptAES(const std::string& cipherTextHex, const std::string& key) {
    try {
        std::string cipherText = fromHex(cipherTextHex);
        
        if (cipherText.size() <= 16) {
            throw std::runtime_error("Invalid ciphertext: too short to contain IV");
        }
        
        std::string iv = cipherText.substr(0, 16);
        std::string actualCipherText = cipherText.substr(16);
        
        // Ensure key is 32 bytes (256 bits) for AES-256
        std::string aesKey = key;
        if (aesKey.size() != 32) {
            if (aesKey.size() < 32) {
                aesKey.append(32 - aesKey.size(), '\0');  // Pad with null bytes
            } else {
                aesKey.resize(32);  
            }
        }
        
        EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Failed to create cipher context");
        }
        
        if (EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, 
                             reinterpret_cast<const unsigned char*>(aesKey.c_str()), 
                             reinterpret_cast<const unsigned char*>(iv.c_str())) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to initialize decryption");
        }
        
        EVP_CIPHER_CTX_set_padding(ctx, 1);
        
        std::vector<unsigned char> plainText(actualCipherText.size() + AES_BLOCK_SIZE);
        int len = 0;
        int plainTextLen = 0;
        
        if (EVP_DecryptUpdate(ctx, plainText.data(), &len, 
                           reinterpret_cast<const unsigned char*>(actualCipherText.c_str()), 
                           static_cast<int>(actualCipherText.size())) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to decrypt data");
        }
        plainTextLen = len;
        
        if (EVP_DecryptFinal_ex(ctx, plainText.data() + len, &len) != 1) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Failed to finalize decryption (padding error - incorrect key or corrupted data)");
        }
        plainTextLen += len;
        
        EVP_CIPHER_CTX_free(ctx);
        
        return std::string(reinterpret_cast<char*>(plainText.data()), plainTextLen);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in decryptAES: " << e.what() << std::endl;
        throw;
    }
}

inline std::string encryptRSA(const std::string& plainText, const std::string& publicKeyPEM) {
    BIO* bio = BIO_new_mem_buf(publicKeyPEM.c_str(), -1);
    if (!bio) {
        throw std::runtime_error("Failed to create BIO for RSA public key");
    }
    
    RSA* rsa = PEM_read_bio_RSA_PUBKEY(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);
    
    if (!rsa) {
        ERR_print_errors_fp(stderr);
        throw std::runtime_error("Failed to load RSA public key");
    }
    
    // max 
    int keySize = RSA_size(rsa);
    int maxSize = keySize - 42; // For OAEP padding with SHA-1
    
    // too large
    if (plainText.size() > static_cast<size_t>(maxSize)) {
        RSA_free(rsa);
        throw std::runtime_error("Message too large for RSA encryption: " + 
                                std::to_string(plainText.size()) + " bytes (max: " + 
                                std::to_string(maxSize) + " bytes)");
    }

    std::vector<unsigned char> cipherText(keySize);
    int cipherTextLen = RSA_public_encrypt(
        static_cast<int>(plainText.size()), 
        reinterpret_cast<const unsigned char*>(plainText.c_str()), 
        cipherText.data(), 
        rsa, 
        RSA_PKCS1_OAEP_PADDING
    );
    
    // Clean up
    RSA_free(rsa);
    
    if (cipherTextLen == -1) {
        ERR_print_errors_fp(stderr);
        throw std::runtime_error("RSA encryption failed");
    }
    
    return toHex(std::string(reinterpret_cast<char*>(cipherText.data()), cipherTextLen));
}

inline std::string decryptRSA(const std::string& cipherTextHex, const std::string& privateKeyPEM) {
    try {
        std::string cipherText = fromHex(cipherTextHex);
        
        BIO* bio = BIO_new_mem_buf(privateKeyPEM.c_str(), -1);
        if (!bio) {
            throw std::runtime_error("Failed to create BIO for RSA private key");
        }
        
        RSA* rsa = PEM_read_bio_RSAPrivateKey(bio, nullptr, nullptr, nullptr);
        BIO_free(bio);
        
        if (!rsa) {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("Failed to load RSA private key");
        }
        
        int keySize = RSA_size(rsa);
        
        if (cipherText.size() > static_cast<size_t>(keySize)) {
            RSA_free(rsa);
            throw std::runtime_error("Invalid ciphertext size: " + 
                                    std::to_string(cipherText.size()) + 
                                    " (should be <= " + std::to_string(keySize) + ")");
        }
        
        std::vector<unsigned char> plainText(keySize);
        int plainTextLen = RSA_private_decrypt(
            static_cast<int>(cipherText.size()), 
            reinterpret_cast<const unsigned char*>(cipherText.c_str()), 
            plainText.data(), 
            rsa, 
            RSA_PKCS1_OAEP_PADDING
        );
        
        // Clean up
        RSA_free(rsa);
        
        if (plainTextLen == -1) {
            ERR_print_errors_fp(stderr);
            throw std::runtime_error("RSA decryption failed");
        }
        
        return std::string(reinterpret_cast<char*>(plainText.data()), plainTextLen);
    } catch (const std::exception& e) {
        std::cerr << "Error in decryptRSA: " << e.what() << std::endl;
        throw; 
    }
}

inline std::string decryptAES_no_padding(const std::string& encryptedHex, const std::string& key) {
    try {
        std::string cipherText = fromHex(encryptedHex);
        
        if (cipherText.size() <= 16) {
            throw std::runtime_error("Invalid ciphertext: too short");
        }
        
        unsigned char iv[16];
        memcpy(iv, cipherText.c_str(), 16);
        std::string actualCipherText = cipherText.substr(16);
        
        std::string paddedKey = key;
        if (paddedKey.size() < 32) {
            paddedKey.resize(32, 0);
        } else if (paddedKey.size() > 32) {
            paddedKey.resize(32);
        }
        
        // Create and initialize the context
        EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
        if (!ctx) {
            throw std::runtime_error("Could not create EVP context");
        }
        
        if (!EVP_DecryptInit_ex(ctx, EVP_aes_256_cbc(), nullptr, 
                            reinterpret_cast<const unsigned char*>(paddedKey.c_str()), 
                            iv)) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Could not initialize decryption operation");
        }
        
        EVP_CIPHER_CTX_set_padding(ctx, 0);
        
        std::vector<unsigned char> outbuf(actualCipherText.size());
        int outlen = 0;
        
        // Decrypt all the data
        if (!EVP_DecryptUpdate(ctx, outbuf.data(), &outlen, 
                            reinterpret_cast<const unsigned char*>(actualCipherText.c_str()), 
                            static_cast<int>(actualCipherText.size()))) {
            EVP_CIPHER_CTX_free(ctx);
            throw std::runtime_error("Error in decryption update");
        }
        
        // Finalize the decryption (may not add any bytes due to disabled padding)
        int finalLen = 0;
        if (!EVP_DecryptFinal_ex(ctx, outbuf.data() + outlen, &finalLen)) {
            // TODO: ignore?? 
        }
        
        EVP_CIPHER_CTX_free(ctx);
        
        std::string result(reinterpret_cast<char*>(outbuf.data()), outlen + finalLen);
        
        size_t lastPrintable = result.find_last_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>/?`~ \t\n\r");
        if (lastPrintable != std::string::npos) {
            result = result.substr(0, lastPrintable + 1);
        }
        
        bool hasPrintable = false;
        for (char c : result) {
            if (isprint(c) || c == '\n' || c == '\t' || c == '\r') {
                hasPrintable = true;
                break;
            }
        }
        
        if (!hasPrintable) {
            std::string cleanResult;
            for (char c : result) {
                if (isprint(c) || c == '\n' || c == '\t' || c == '\r') {
                    cleanResult += c;
                }
            }
            return cleanResult;
        }
        
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error in decryptAES_no_padding: " << e.what() << std::endl;
        throw;
    }
}
} 

#endif 