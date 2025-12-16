#include "llama_wrapper.hpp"
#include <iostream>
#include <algorithm>
#include <thread>
#include <cmath>

namespace llm {

LlamaWrapper::LlamaWrapper() {
    std::cout << "Creating LlamaWrapper" << std::endl;
}

LlamaWrapper::~LlamaWrapper() {
    cleanup();
}

bool LlamaWrapper::initialize() {
    std::lock_guard<std::mutex> lock(mtx);
    
    if (initialized) {
        return true;
    }
    
    // Initialize llama.cpp backend
    llama_backend_init();
    
    initialized = true;
    std::cout << "Initialized llama backend" << std::endl;
    return true;
}

bool LlamaWrapper::loadModel(const std::string& modelPath, int contextSize, int threads) {
    std::lock_guard<std::mutex> lock(mtx);
    
    if (!initialized) {
        if (!initialize()) {
            return false;
        }
    }
    
    // Clean up existing model if any
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    
    // Set model parameters
    llama_model_params model_params = llama_model_default_params();
    
    // Load the model
    model = llama_model_load_from_file(modelPath.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model from: " << modelPath << std::endl;
        return false;
    }
    
    // Set context parameters
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize;
    ctx_params.n_batch = 512;  // Batch size for prompt processing
    
    // Set number of threads
    if (threads <= 0) {
        ctx_params.n_threads = std::min(8, static_cast<int>(std::thread::hardware_concurrency()));
    } else {
        ctx_params.n_threads = threads;
    }
    
    // Create context
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context for model" << std::endl;
        llama_model_free(model);
        model = nullptr;
        return false;
    }
    
    std::cout << "Successfully loaded model from: " << modelPath << std::endl;
    std::cout << "  Context size: " << contextSize << std::endl;
    std::cout << "  Threads: " << ctx_params.n_threads << std::endl;
    
    return true;
}

// Helper function to clean tokenizer output (空格为乱码)
static std::string cleanTokenizerOutput(const std::string& raw) {
    std::string result;
    result.reserve(raw.size());
    
    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(raw[i]);
        
        // Check for UTF-8 multi-byte sequences
        if (c == 0xC4) {  // Start of 2-byte UTF-8
            if (i + 1 < raw.size()) {
                unsigned char next = static_cast<unsigned char>(raw[i + 1]);
                if (next == 0xA0) {  // Ġ (U+0120) -> space
                    result += ' ';
                    i++;
                    continue;
                } else if (next == 0x8A) {  // Ċ (U+010A) -> newline
                    result += '\n';
                    i++;
                    continue;
                }
            }
        }
        
        // Check for common special token markers
        if (c == 0xC3) {  // UTF-8 continuation for special chars
            if (i + 1 < raw.size()) {
                unsigned char next = static_cast<unsigned char>(raw[i + 1]);
                // Ã followed by something - could be encoding issue, pass through
                result += raw[i];
                continue;
            }
        }
        
        result += raw[i];
    }
    
    return result;
}

std::string LlamaWrapper::generate(const std::string& prompt, int maxTokens, float temperature) {
    std::lock_guard<std::mutex> lock(mtx);
    
    if (!ctx || !model) {
        std::cerr << "Model not loaded" << std::endl;
        return "Error: Model not loaded";
    }
    
    // Reset context
    resetContext();
    
    // Tokenize the prompt
    std::vector<llama_token> tokens = tokenize(prompt);
    if (tokens.empty()) {
        std::cerr << "Failed to tokenize prompt" << std::endl;
        return "Error: Failed to tokenize prompt";
    }
    
    // Feed the prompt to the model
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Failed to process prompt" << std::endl;
        return "Error: Failed to process prompt";
    }
    
    // Generate completion
    std::string result;
    llama_token id = 0;
    
    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    for (int i = 0; i < maxTokens; ++i) {
        // Sample the next token
        id = llama_sampler_sample(smpl, ctx, -1);
        
        // Check for end of sequence
        if (id == llama_vocab_eos(llama_model_get_vocab(model))) {
            break;
        }
        
        // Decode the token to text
        const char* token_str = llama_vocab_get_text(llama_model_get_vocab(model), id);
        if (token_str) {
            result += token_str;
        }
        
        // Feed the token back for the next prediction
        batch = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, batch) != 0) {
            break;
        }
    }
    
    // Clean up sampler
    llama_sampler_free(smpl);
    
    // Clean tokenizer output (convert Ġ -> space, Ċ -> newline)
    return cleanTokenizerOutput(result);
}

bool LlamaWrapper::isModelLoaded() const {
    return (model != nullptr && ctx != nullptr);
}

void LlamaWrapper::cleanup() {
    std::lock_guard<std::mutex> lock(mtx);
    
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    
    if (initialized) {
        llama_backend_free();
        initialized = false;
    }
}

void LlamaWrapper::resetContext() {
    if (ctx) {
        llama_kv_cache_clear(ctx);
    }
}

std::vector<llama_token> LlamaWrapper::tokenize(const std::string& text) {
    if (!model) {
        return {};
    }
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.length() + 1);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), true, true);
    } else {
        tokens.resize(n_tokens);
    }
    return tokens;
}

std::string LlamaWrapper::getTokenText(llama_token token) {
    if (!model) return "";
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const char* text = llama_vocab_get_text(vocab, token);
    return text ? cleanTokenizerOutput(text) : "";
}

std::vector<TokenInfo> LlamaWrapper::tokenizeWithInfo(const std::string& text) {
    std::lock_guard<std::mutex> lock(mtx);
    
    std::vector<TokenInfo> result;
    if (!model) return result;
    
    std::vector<llama_token> tokens = tokenize(text);
    result.reserve(tokens.size());
    
    for (llama_token tok : tokens) {
        TokenInfo info;
        info.id = tok;
        info.text = getTokenText(tok);
        info.logprob = 0.0f;  // Not computed yet
        result.push_back(info);
    }
    
    return result;
}

std::vector<std::pair<llama_token, float>> LlamaWrapper::getNextTokenLogprobs(
        const std::string& context, int topK) {
    // Note: This method assumes the caller holds the mutex
    std::vector<std::pair<llama_token, float>> result;
    
    if (!ctx || !model) return result;
    
    // Get logits for the last position
    float* logits = llama_get_logits(ctx);
    if (!logits) return result;
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    
    // Convert logits to log probabilities using softmax
    // First find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    // Compute exp and sum
    std::vector<float> probs(n_vocab);
    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    
    // Normalize and convert to log probs
    std::vector<std::pair<float, llama_token>> sorted_probs;
    sorted_probs.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        float prob = probs[i] / sum;
        float logprob = std::log(prob + 1e-10f);
        sorted_probs.push_back({logprob, static_cast<llama_token>(i)});
    }
    
    // Sort by log probability (descending)
    std::partial_sort(sorted_probs.begin(), 
                      sorted_probs.begin() + std::min(topK, n_vocab),
                      sorted_probs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Return top-k
    for (int i = 0; i < std::min(topK, n_vocab); i++) {
        result.push_back({sorted_probs[i].second, sorted_probs[i].first});
    }
    
    return result;
}

PerplexityResult LlamaWrapper::calculatePerplexity(const std::string& prompt, 
                                                    const std::string& response) {
    std::lock_guard<std::mutex> lock(mtx);
    
    PerplexityResult result;
    
    if (!ctx || !model) {
        std::cerr << "[Perplexity] Model not loaded" << std::endl;
        result.perplexity = 1e10;
        result.credibilityScore = 0.0;
        return result;
    }
    
    // Tokenize the response
    std::vector<llama_token> response_tokens = tokenize(response);
    if (response_tokens.empty()) {
        std::cerr << "[Perplexity] Empty response" << std::endl;
        result.perplexity = 1e10;
        result.credibilityScore = 0.0;
        return result;
    }
    
    result.tokenCount = static_cast<int>(response_tokens.size());
    
    // Tokenize the prompt
    std::vector<llama_token> prompt_tokens = tokenize(prompt);
    
    // Reset context
    resetContext();
    
    // Process the prompt first
    if (!prompt_tokens.empty()) {
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "[Perplexity] Failed to process prompt" << std::endl;
            result.perplexity = 1e10;
            result.credibilityScore = 0.0;
            return result;
        }
    }
    
    // Now process response tokens one by one, computing perplexity
    double sum_log_prob = 0.0;
    int matched_count = 0;
    const float EPSILON = 1e-10f;
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    
    for (size_t i = 0; i < response_tokens.size(); i++) {
        llama_token current_token = response_tokens[i];
        
        // Get logits for current position
        float* logits = llama_get_logits(ctx);
        if (!logits) {
            result.tokenProbabilities.push_back(EPSILON);
            continue;
        }
        
        // Compute softmax probability for the actual token
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > max_logit) max_logit = logits[j];
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < n_vocab; j++) {
            sum_exp += std::exp(logits[j] - max_logit);
        }
        
        float token_prob = std::exp(logits[current_token] - max_logit) / sum_exp;
        
        if (token_prob > EPSILON) {
            matched_count++;
        }
        
        // Clamp probability
        token_prob = std::max(token_prob, EPSILON);
        result.tokenProbabilities.push_back(token_prob);
        
        // Accumulate log probability
        sum_log_prob += std::log(token_prob);
        
        // Feed this token for next prediction
        llama_batch batch = llama_batch_get_one(&current_token, 1);
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "[Perplexity] Failed to decode token " << i << std::endl;
            break;
        }
    }
    
    result.matchedTokenCount = matched_count;
    
    // Calculate perplexity: PPL = exp(-1/n * Σ log(p_i))
    if (!result.tokenProbabilities.empty()) {
        double avg_neg_log_prob = -sum_log_prob / result.tokenProbabilities.size();
        result.perplexity = std::exp(avg_neg_log_prob);
        
        // Credibility = 1/PPL, clamped to [0, 1]
        result.credibilityScore = std::min(1.0, 1.0 / result.perplexity);
    } else {
        result.perplexity = 1e10;
        result.credibilityScore = 0.0;
    }
    
    return result;
}

} 