#ifndef KEY_GENERATION_HPP
#define KEY_GENERATION_HPP

#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <string>
#include <vector>
#include <memory>
#include <random>

namespace encrypt_p2p {

inline void handleErrors() {
    ERR_print_errors_fp(stderr);
    abort();
}

inline std::pair<std::string, std::string> generateRSAKeyPair(int keyLength = 2048) {
    RSA *rsa = RSA_new();
    if (!rsa) {
        handleErrors();
    }
    
    BIGNUM *e = BN_new();
    if (!e) {
        RSA_free(rsa);
        handleErrors();
    }
    
    if (BN_set_word(e, RSA_F4) != 1) { // RSA_F4 is 65537
        BN_free(e);
        RSA_free(rsa);
        handleErrors();
    }

    if (RSA_generate_key_ex(rsa, keyLength, e, nullptr) != 1) {
        BN_free(e);
        RSA_free(rsa);
        handleErrors();
    }

    BIO *privBio = BIO_new(BIO_s_mem());
    if (!privBio) {
        BN_free(e);
        RSA_free(rsa);
        handleErrors();
    }
    
    if (!PEM_write_bio_RSAPrivateKey(privBio, rsa, nullptr, nullptr, 0, nullptr, nullptr)) {
        BIO_free(privBio);
        BN_free(e);
        RSA_free(rsa);
        handleErrors();
    }

    BIO *pubBio = BIO_new(BIO_s_mem());
    if (!pubBio) {
        BIO_free(privBio);
        BN_free(e);
        RSA_free(rsa);
        handleErrors();
    }
    
    if (!PEM_write_bio_RSA_PUBKEY(pubBio, rsa)) { // X.509 SubjectPublicKeyInfo format
        BIO_free(pubBio);
        BIO_free(privBio);
        BN_free(e);
        RSA_free(rsa);
        handleErrors();
    }

    char *privKeyPtr = nullptr;
    long privKeySize = BIO_get_mem_data(privBio, &privKeyPtr);
    std::string privateKey(privKeyPtr, privKeySize);

    char *pubKeyPtr = nullptr;
    long pubKeySize = BIO_get_mem_data(pubBio, &pubKeyPtr);
    std::string publicKey(pubKeyPtr, pubKeySize);

    // Clean up
    BIO_free(privBio);
    BIO_free(pubBio);
    RSA_free(rsa);
    BN_free(e);

    return std::make_pair(privateKey, publicKey);
}

inline std::string generateAESKey() {
    const int KEY_SIZE = 32; // 256 bit key for AES-256
    std::vector<unsigned char> key(KEY_SIZE);
    
    if (RAND_bytes(key.data(), KEY_SIZE) != 1) {
        ERR_print_errors_fp(stderr);
        throw std::runtime_error("Failed to generate secure random AES key");
    }
    
    return std::string(reinterpret_cast<char*>(key.data()), KEY_SIZE);
}

} 

#endif 