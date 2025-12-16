#ifndef LLAMA_WRAPPER_HPP
#define LLAMA_WRAPPER_HPP

#include <string>
#include <vector>
#include <mutex>
#include "../../deps/llama.cpp/include/llama.h"

namespace llm {


struct TokenInfo {
    llama_token id;
    std::string text;
    float logprob;  // Log probability of this token
};


struct PerplexityResult {
    double perplexity;
    double credibilityScore;
    int tokenCount;
    int matchedTokenCount;
    std::vector<double> tokenProbabilities;
    
    PerplexityResult() : perplexity(0), credibilityScore(0), 
                         tokenCount(0), matchedTokenCount(0) {}
};

class LlamaWrapper {
public:
    LlamaWrapper();
    ~LlamaWrapper();

    // Init
    bool initialize();

    bool loadModel(const std::string& modelPath, int contextSize = 2048, int threads = 0);

    std::string generate(const std::string& prompt, int maxTokens = 512, float temperature = 0.7f);

    // Check loaded
    bool isModelLoaded() const;

    void cleanup();
    
    //credit calcualtion
    std::vector<TokenInfo> tokenizeWithInfo(const std::string& text);
    

    PerplexityResult calculatePerplexity(const std::string& prompt, const std::string& response);
    

    std::vector<std::pair<llama_token, float>> getNextTokenLogprobs(
        const std::string& context, int topK = 40);

private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    std::mutex mtx;  // For thread safety
    bool initialized = false;

    // Reset KV cache
    void resetContext();

    // Tokenize input text (internal)
    std::vector<llama_token> tokenize(const std::string& text);
    
    // Get token text
    std::string getTokenText(llama_token token);
};

} 

#endif 