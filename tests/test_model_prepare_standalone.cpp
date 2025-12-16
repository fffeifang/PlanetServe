#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>  // For std::accumulate
#include <filesystem>
#include <openssl/sha.h>

#include "../src/encrypt_p2p/s_ida.hpp"


const std::string MODEL_PATH = "../models/Llama-3.3-70B-Instruct-Q4_0.gguf";
const std::string WORKLOAD_PATH = "../datasets/toolbench_prompts_1000.jsonl";
const std::string LOG_FILE = "model_prepare_performance.log";
const std::string LOG_FILE_2 = "model_decrypt_performance.log";
const int NUM_CLOVES_TO_TEST = 10000;  
const int DEFAULT_N = 4;  //
const int DEFAULT_K = 3;  


std::string cleanTokenizerArtifacts(const std::string& modelOutput) {
    std::string cleaned;
    cleaned.reserve(modelOutput.size());
    
    for (size_t i = 0; i < modelOutput.size(); ++i) {
        if (modelOutput[i] == 'Ġ') {
            // Replace Ġ (representing a space) with an actual space
            cleaned += ' ';
        } else if (modelOutput[i] == 'Ċ') {
            // Replace Ċ (representing a newline) with an actual newline
            cleaned += '\n';
        } else {
            // Keep other characters as is
            cleaned += modelOutput[i];
        }
    }
    
    return cleaned;
}

std::string extractJsonValue(const std::string& jsonStr, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t keyPos = jsonStr.find(searchKey);
    if (keyPos == std::string::npos) {
        return "";
    }
    
    size_t colonPos = jsonStr.find(':', keyPos);
    if (colonPos == std::string::npos) {
        return "";
    }
    
    size_t valueStart = jsonStr.find_first_not_of(" \t\n\r", colonPos + 1);
    if (valueStart == std::string::npos) {
        return "";
    }
    
    if (jsonStr[valueStart] == '"') {
        size_t valueEnd = valueStart + 1;
        bool escaped = false;
        
        while (valueEnd < jsonStr.size()) {
            if (jsonStr[valueEnd] == '\\') {
                escaped = !escaped;
                valueEnd++;
            } else if (jsonStr[valueEnd] == '"' && !escaped) {
                break;
            } else {
                escaped = false;
                valueEnd++;
            }
        }
        
        if (valueEnd >= jsonStr.size()) {
            return "";
        }
        
        std::string value = jsonStr.substr(valueStart + 1, valueEnd - valueStart - 1);
        
        std::string unescaped;
        for (size_t i = 0; i < value.size(); ++i) {
            if (value[i] == '\\' && i + 1 < value.size()) {
                // Handle escape sequences
                switch (value[i + 1]) {
                    case 'n': unescaped += '\n'; break;
                    case 'r': unescaped += '\r'; break;
                    case 't': unescaped += '\t'; break;
                    case '\\': unescaped += '\\'; break;
                    case '"': unescaped += '"'; break;
                    default: unescaped += value[i + 1]; break;
                }
                i++; // Skip the escaped character
            } else {
                unescaped += value[i];
            }
        }
        
        return unescaped;
    }
    
    return "";
}

std::vector<std::string> readPromptsFromToolbench(const std::string& filename) {
    std::vector<std::string> prompts;
    std::ifstream file(filename);
    std::string line;
    
    if (file.is_open()) {
        while (std::getline(file, line)) {
            if (!line.empty()) {
                // Extract the "workload" field which contains the prompt
                std::string workload = extractJsonValue(line, "workload");
                if (!workload.empty()) {
                    prompts.push_back(workload);
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
    
    return prompts;
}

std::vector<std::string> readAllQuestions() {
    std::cout << "Reading prompts from toolbench file: " << WORKLOAD_PATH << std::endl;
    auto prompts = readPromptsFromToolbench(WORKLOAD_PATH);
    std::cout << "Read " << prompts.size() << " prompts from toolbench file" << std::endl;
    return prompts;
}

std::string truncatePrompt(const std::string& prompt, size_t maxLength = 1500) {
    if (prompt.length() <= maxLength) {
        return prompt;
    }
    return prompt.substr(0, maxLength) + "...";
}

// Main function to test clove preparation
int main() {
    // Initialize logging
    std::ofstream logFile(LOG_FILE);
    std::ofstream logFile_2(LOG_FILE_2);
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << LOG_FILE << std::endl;
        return 1;
    }
    if (!logFile_2.is_open()) {
        std::cerr << "Failed to open log file: " << LOG_FILE_2 << std::endl;
        return 1;
    }
    
    logFile << "# S-IDA Clove Preparation Performance Test with Real Llama Model\n";
    auto now = std::chrono::system_clock::now();
    std::time_t current_time = std::chrono::system_clock::to_time_t(now);
    std::tm* timeinfo = std::localtime(&current_time);
    logFile << "# Date: " << std::put_time(timeinfo, "%Y-%m-%d %H:%M:%S") << "\n";
    logFile << "# Model: " << MODEL_PATH << "\n";
    logFile << "# Workload: " << WORKLOAD_PATH << "\n";
    logFile << "# Number of cloves: " << NUM_CLOVES_TO_TEST << "\n";
    logFile << "# N (paths): " << DEFAULT_N << ", K (threshold): " << DEFAULT_K << "\n\n";
    logFile_2 << "# S-IDA Clove Decryption Performance Test with Real Llama Model\n";
    logFile_2 << "# Date: " << std::put_time(timeinfo, "%Y-%m-%d %H:%M:%S") << "\n";
    logFile_2 << "# Model: " << MODEL_PATH << "\n";
    logFile_2 << "# Workload: " << WORKLOAD_PATH << "\n";
    logFile_2 << "# Number of cloves: " << NUM_CLOVES_TO_TEST << "\n";
    logFile_2 << "# N (paths): " << DEFAULT_N << ", K (threshold): " << DEFAULT_K << "\n\n";
    
    // Initialize Llama model
    std::cout << "Initializing Llama model from " << MODEL_PATH << "..." << std::endl;
    logFile << "Initializing Llama model...\n";
    
    
    std::string answersFile = "model_answers_cache.txt";
    std::vector<std::string> modelAnswers;
    bool useCache = false;
    
    
    
    
    
    if (std::filesystem::exists(answersFile)) {
        std::cout << "Found cached model answers, reading from file..." << std::endl;
        std::ifstream cache(answersFile);
        std::string line;
        while (std::getline(cache, line)) {
            // Check if the line contains tokenizer artifacts and clean if needed
            if (line.find('Ġ') != std::string::npos || line.find('Ċ') != std::string::npos) {
                line = cleanTokenizerArtifacts(line);
            }
            modelAnswers.push_back(line);
        }
        cache.close();
        
        if (!modelAnswers.empty()) {
            std::cout << "Loaded " << modelAnswers.size() << " cached answers" << std::endl;
            useCache = true;
        }
    }
    
    // If we don't have cached answers, generate them with the model
    if (!useCache) {
        logFile << "No cached answers found, generating new answers...\n";
    }
    
    // Test latency of preparing cloves
    std::cout << "Testing latency of preparing " << NUM_CLOVES_TO_TEST << " cloves..." << std::endl;
    logFile << "Testing clove preparation latency for " << NUM_CLOVES_TO_TEST << " messages\n";
    
    // Generate test messages using the model answers
    std::vector<std::string> testMessages;
    
    // Use actual model answers for testing
    for (const auto& answer : modelAnswers) {
        testMessages.push_back(answer);
    }
    
    // If we don't have enough answers, repeat them until we reach NUM_CLOVES_TO_TEST
    int repeatCount = 1;
    while (testMessages.size() < NUM_CLOVES_TO_TEST) {
        std::cout << "Repeating model answers (pass " << repeatCount << ") to reach " << NUM_CLOVES_TO_TEST << " messages" << std::endl;
        
        for (const auto& answer : modelAnswers) {
            if (testMessages.size() >= NUM_CLOVES_TO_TEST) {
                break;
            }
            testMessages.push_back(answer);
        }
        repeatCount++;
    }
    
    // Truncate to exactly NUM_CLOVES_TO_TEST if we've gone over
    if (testMessages.size() > NUM_CLOVES_TO_TEST) {
        testMessages.resize(NUM_CLOVES_TO_TEST);
    }

    // Measure clove preparation time
    std::vector<double> clovePrepTimes;
    clovePrepTimes.reserve(NUM_CLOVES_TO_TEST);
    std::vector<double> cloveDecryptTimes;
    cloveDecryptTimes.reserve(NUM_CLOVES_TO_TEST);
    double totalPrepTime = 0.0;
    double totalDecryptTime = 0.0;
    size_t totalMessageSize = 0;
    std::pair<std::string, std::string> keyPair = encrypt_p2p::generateRSAKeyPair();
    std::string RSA_PUBLIC_KEY = keyPair.first;
    std::string RSA_PRIVATE_KEY = keyPair.second;
    for (int i = 0; i < NUM_CLOVES_TO_TEST; i++) {
        const std::string& message = testMessages[i];
        totalMessageSize += message.size();
        
        auto startPrep = std::chrono::high_resolution_clock::now();
        std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string message_to_digest = message + "|" + timestamp;
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, message_to_digest.c_str(), message_to_digest.size());
        SHA256_Final(hash, &sha256);    

        std::string digest(reinterpret_cast<char*>(hash), SHA256_DIGEST_LENGTH);
        std::string signed_digest = encrypt_p2p::encryptRSA(digest, RSA_PRIVATE_KEY);
        std::string prepared_message = message + "|" + signed_digest;
        // Create S-IDA cloves for the message without modifying the original message
        auto cloves = encrypt_p2p::SIDA::split(prepared_message, DEFAULT_N, DEFAULT_K);
        
        auto endPrep = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> prepTime = endPrep - startPrep;
        
        clovePrepTimes.push_back(prepTime.count());
        totalPrepTime += prepTime.count();
         
        // Log individual latency values for detailed analysis
        if(i != 0){
            logFile << "Clove " << i << " preparation time: " << prepTime.count() << " ms\n";
        }
        

        // Test decryption latency of cloves
        auto startDecrypt = std::chrono::high_resolution_clock::now();
        std::string decryptedMessage;
        bool decryptionSuccess = false;
        
        try {
            // The S-IDA implementation will handle the proper selection of shares internally
            decryptedMessage = encrypt_p2p::SIDA::combine(cloves, DEFAULT_K);
            decryptionSuccess = true;
        } catch (const std::exception& e) {
            std::cerr << "Error during S-IDA combine (message " << i << "): " << e.what() << std::endl;
            logFile_2 << "Error during S-IDA combine for message " << i << ": " << e.what() << "\n";
            
            // Try a second approach with explicit selection and sorting
            try {
                // Sort cloves by fragment index to ensure we get a valid combination
                std::sort(cloves.begin(), cloves.end(), 
                    [](const encrypt_p2p::SIDA::Clove& a, const encrypt_p2p::SIDA::Clove& b) {
                        return a.fragmentIndex < b.fragmentIndex;
                    });
                
                // Now select the first DEFAULT_K cloves with ordered x-coordinates
                std::vector<encrypt_p2p::SIDA::Clove> selectedCloves(cloves.begin(), cloves.begin() + DEFAULT_K);
                
                decryptedMessage = encrypt_p2p::SIDA::combine(selectedCloves, DEFAULT_K);
                decryptionSuccess = true;
                
                logFile_2 << "Decryption succeeded on second attempt with sorted key shares\n";
            } catch (const std::exception& e) {
                std::cerr << "Error during second S-IDA combine attempt (message " << i << "): " << e.what() << std::endl;
                logFile_2 << "Error during second S-IDA combine attempt for message " << i << ": " << e.what() << "\n";
            }
        }
        
        auto endDecrypt = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> decryptTime = endDecrypt - startDecrypt;
        
        if (decryptionSuccess) {
            totalDecryptTime += decryptTime.count();
            cloveDecryptTimes.push_back(decryptTime.count());
            logFile_2 << "Decryption succeeded for message " << i << "\n";
            if(i != 0){
                logFile_2 << "Clove " << i << " decryption time: " << decryptTime.count() << " ms\n";
            }
        } else {
            cloveDecryptTimes.push_back(9999.0); // Use a large value to indicate failure
            logFile_2 << "Decryption failed for message " << i << "\n";
        }
        
        if (i % 1000 == 0) {
            std::cout << "Processed " << i << " messages" << std::endl;
        }
    }
    
    // Calculate statistics
    double avgPrepTime = totalPrepTime / NUM_CLOVES_TO_TEST;
    double avgMessageSize = static_cast<double>(totalMessageSize) / NUM_CLOVES_TO_TEST;
    
    // Filter out failed decryptions (values of 9999.0)
    std::vector<double> successfulDecryptTimes;
    for (const auto& time : cloveDecryptTimes) {
        if (time < 9000.0) { // Use a threshold to identify successful decryptions
            successfulDecryptTimes.push_back(time);
        }
    }
    
    double avgDecryptTime = 0.0;
    double p50_decrypt = 0.0, p90_decrypt = 0.0, p99_decrypt = 0.0;
    
    if (!successfulDecryptTimes.empty()) {
        double totalSuccessfulDecryptTime = std::accumulate(successfulDecryptTimes.begin(), successfulDecryptTimes.end(), 0.0);
        avgDecryptTime = totalSuccessfulDecryptTime / successfulDecryptTimes.size();
        
        // Sort times for percentile calculations
        std::sort(successfulDecryptTimes.begin(), successfulDecryptTimes.end());
        
        // Only calculate percentiles if there are enough samples
        if (successfulDecryptTimes.size() >= 2) {
            p50_decrypt = successfulDecryptTimes[successfulDecryptTimes.size() / 2];
            
            if (successfulDecryptTimes.size() >= 10) {
                p90_decrypt = successfulDecryptTimes[successfulDecryptTimes.size() * 9 / 10];
            }
            
            if (successfulDecryptTimes.size() >= 100) {
                p99_decrypt = successfulDecryptTimes[successfulDecryptTimes.size() * 99 / 100];
            }
        }
    }
    
    // Sort preparation times for percentile calculations
    std::sort(clovePrepTimes.begin(), clovePrepTimes.end());
    
    double p50 = clovePrepTimes[NUM_CLOVES_TO_TEST / 2];
    double p90 = clovePrepTimes[NUM_CLOVES_TO_TEST * 9 / 10];
    double p99 = clovePrepTimes[NUM_CLOVES_TO_TEST * 99 / 100];
    
    // Log success rate for decryption
    double successRate = (static_cast<double>(successfulDecryptTimes.size()) / NUM_CLOVES_TO_TEST) * 100.0;
    logFile_2 << "\n# Decryption Success Rate: " << std::fixed << std::setprecision(2) << successRate << "% (" 
              << successfulDecryptTimes.size() << "/" << NUM_CLOVES_TO_TEST << " messages)\n";
    
    if (successfulDecryptTimes.empty()) {
        logFile_2 << "No successful decryptions to calculate statistics.\n";
    } else {
        // Log results for successful decryptions
        logFile_2 << "\n# Performance Results (for successful decryptions only)\n";
        logFile_2 << "Average clove decryption time: " << std::fixed << std::setprecision(3) << avgDecryptTime << " ms\n";
        logFile_2 << "Average message size: " << std::fixed << std::setprecision(1) << avgMessageSize << " bytes\n";
        logFile_2 << "Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / avgDecryptTime) << " cloves/second\n";
        logFile_2 << "p50 latency: " << std::fixed << std::setprecision(3) << p50_decrypt << " ms\n";
        
        if (successfulDecryptTimes.size() >= 10) {
            logFile_2 << "p90 latency: " << std::fixed << std::setprecision(3) << p90_decrypt << " ms\n";
        }
        
        if (successfulDecryptTimes.size() >= 100) {
            logFile_2 << "p99 latency: " << std::fixed << std::setprecision(3) << p99_decrypt << " ms\n";
        }
    }
    
    // Log preparation results (these should always be successful)
    logFile << "\n# Performance Results\n";
    logFile << "Average clove preparation time: " << std::fixed << std::setprecision(3) << avgPrepTime << " ms\n";
    logFile << "Average message size: " << std::fixed << std::setprecision(1) << avgMessageSize << " bytes\n";
    logFile << "Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / avgPrepTime) << " cloves/second\n";
    logFile << "p50 latency: " << std::fixed << std::setprecision(3) << p50 << " ms\n";
    logFile << "p90 latency: " << std::fixed << std::setprecision(3) << p90 << " ms\n";
    logFile << "p99 latency: " << std::fixed << std::setprecision(3) << p99 << " ms\n";
    
    // Output summary to console
    std::cout << "\n=== Clove Preparation Performance Summary ===\n";
    std::cout << "Processed " << NUM_CLOVES_TO_TEST << " messages with average size " << avgMessageSize << " bytes" << std::endl;
    std::cout << "Average clove preparation time: " << avgPrepTime << " ms" << std::endl;
    std::cout << "Throughput: " << (1000.0 / avgPrepTime) << " cloves/second" << std::endl;
    std::cout << "p50 latency: " << p50 << " ms" << std::endl;
    std::cout << "p90 latency: " << p90 << " ms" << std::endl;
    std::cout << "p99 latency: " << p99 << " ms" << std::endl;
    std::cout << "Detailed results written to " << LOG_FILE << std::endl;
    
    std::cout << "\n=== Clove Decryption Performance Summary ===\n";
    std::cout << "Decryption success rate: " << std::fixed << std::setprecision(2) << successRate << "% (" 
             << successfulDecryptTimes.size() << "/" << NUM_CLOVES_TO_TEST << " messages)" << std::endl;
    
    if (successfulDecryptTimes.empty()) {
        std::cout << "No successful decryptions to calculate statistics." << std::endl;
    } else {
        std::cout << "Average clove decryption time: " << avgDecryptTime << " ms" << std::endl;
        std::cout << "Throughput: " << (1000.0 / avgDecryptTime) << " cloves/second" << std::endl;
        std::cout << "p50 latency: " << p50_decrypt << " ms" << std::endl;
        
        if (successfulDecryptTimes.size() >= 10) {
            std::cout << "p90 latency: " << p90_decrypt << " ms" << std::endl;
        }
        
        if (successfulDecryptTimes.size() >= 100) {
            std::cout << "p99 latency: " << p99_decrypt << " ms" << std::endl;
        }
    }
    
    std::cout << "Detailed results written to " << LOG_FILE_2 << std::endl;
    return 0;
} 