#pragma once
#ifndef RABIN_IDA_HPP
#define RABIN_IDA_HPP

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace encrypt_p2p {
// from wiki
class RabinIDA {
public:
    // Fragment structure returned by split
    struct Fragment {
        uint8_t index;                  // Fragment index (1-based, corresponds to x value)
        std::vector<uint8_t> data;      // Fragment data
        uint32_t originalSize;          // Original data size before padding
    };

    class GF256 {
    public:
        static void initTables() {
            if (tablesInitialized) return;
            
            uint16_t x = 1;
            for (int i = 0; i < 255; i++) {
                EXP_TABLE[i] = static_cast<uint8_t>(x);
                LOG_TABLE[x] = static_cast<uint8_t>(i);
                x = multiply_no_table(static_cast<uint8_t>(x), 0x03);
            }
            EXP_TABLE[255] = EXP_TABLE[0];
            LOG_TABLE[0] = 0; // Undefined, but set to 0 for safety
            
            tablesInitialized = true;
        }

        static inline uint8_t add(uint8_t a, uint8_t b) {
            return a ^ b;
        }

        static inline uint8_t sub(uint8_t a, uint8_t b) {
            return a ^ b; // Same as add in GF(2^8)
        }

        static inline uint8_t mul(uint8_t a, uint8_t b) {
            if (a == 0 || b == 0) return 0;
            int sum = LOG_TABLE[a] + LOG_TABLE[b];
            if (sum >= 255) sum -= 255;
            return EXP_TABLE[sum];
        }

        static inline uint8_t div(uint8_t a, uint8_t b) {
            if (b == 0) throw std::runtime_error("RabinIDA: Division by zero in GF256");
            if (a == 0) return 0;
            int diff = LOG_TABLE[a] - LOG_TABLE[b];
            if (diff < 0) diff += 255;
            return EXP_TABLE[diff];
        }

        static inline uint8_t pow(uint8_t base, uint8_t exp) {
            if (base == 0) return (exp == 0) ? 1 : 0;
            if (exp == 0) return 1;
            int logResult = (static_cast<int>(LOG_TABLE[base]) * exp) % 255;
            return EXP_TABLE[logResult];
        }

        static inline uint8_t inv(uint8_t a) {
            if (a == 0) throw std::runtime_error("RabinIDA: Cannot invert zero in GF256");
            return EXP_TABLE[255 - LOG_TABLE[a]];
        }

    private:
        static uint8_t multiply_no_table(uint8_t a, uint8_t b) {
            uint8_t result = 0;
            uint8_t high_bit;
            for (int i = 0; i < 8; i++) {
                if (b & 1) result ^= a;
                high_bit = a & 0x80;
                a <<= 1;
                if (high_bit) a ^= 0x1B; // x^8 + x^4 + x^3 + x + 1
                b >>= 1;
            }
            return result;
        }

        static uint8_t EXP_TABLE[256];
        static uint8_t LOG_TABLE[256];
        static bool tablesInitialized;
    };

    //=========================================================================
    // Split data into n fragments, any k can reconstruct
    //=========================================================================
    static std::vector<Fragment> split(const std::vector<uint8_t>& data, int n, int k) {
        if (k <= 0 || n <= 0 || k > n) {
            throw std::runtime_error("RabinIDA: Invalid parameters (need 0 < k <= n)");
        }
        if (n > 255) {
            throw std::runtime_error("RabinIDA: n must be <= 255 for GF(256)");
        }
        if (data.empty()) {
            throw std::runtime_error("RabinIDA: Cannot split empty data");
        }

        GF256::initTables();

        uint32_t originalSize = static_cast<uint32_t>(data.size());
        
      
        size_t paddedSize = ((data.size() + k - 1) / k) * k;
        std::vector<uint8_t> paddedData = data;
        paddedData.resize(paddedSize, 0);

        size_t chunkSize = paddedSize / k;

        std::vector<std::vector<uint8_t>> vandermondeRows(n, std::vector<uint8_t>(k));
        for (int i = 0; i < n; i++) {
            uint8_t x = static_cast<uint8_t>(i + 1);
            uint8_t x_pow = 1;
            for (int j = 0; j < k; j++) {
                vandermondeRows[i][j] = x_pow;
                x_pow = GF256::mul(x_pow, x);
            }
        }

        // Encode: for each fragment i, compute fragment[pos] = Σ A[i][t] * chunk_t[pos]
        std::vector<Fragment> fragments(n);
        for (int i = 0; i < n; i++) {
            fragments[i].index = static_cast<uint8_t>(i + 1);
            fragments[i].originalSize = originalSize;
            fragments[i].data.resize(chunkSize, 0);

            for (size_t pos = 0; pos < chunkSize; pos++) {
                uint8_t sum = 0;
                for (int t = 0; t < k; t++) {
                    uint8_t chunkByte = paddedData[t * chunkSize + pos];
                    sum = GF256::add(sum, GF256::mul(vandermondeRows[i][t], chunkByte));
                }
                fragments[i].data[pos] = sum;
            }
        }

        return fragments;
    }

    static std::vector<Fragment> split(const std::string& data, int n, int k) {
        std::vector<uint8_t> bytes(data.begin(), data.end());
        return split(bytes, n, k);
    }

    // Combine k fragments to reconstruct original data
    static std::vector<uint8_t> combine(const std::vector<Fragment>& fragments, int k) {
        if (static_cast<int>(fragments.size()) < k) {
            throw std::runtime_error("RabinIDA: Not enough fragments to reconstruct (need " + 
                                     std::to_string(k) + ", have " + std::to_string(fragments.size()) + ")");
        }

        GF256::initTables();

        // Use first k distinct fragments
        std::vector<Fragment> selected;
        std::vector<bool> usedIndices(256, false);
        
        for (const auto& frag : fragments) {
            if (!usedIndices[frag.index] && static_cast<int>(selected.size()) < k) {
                selected.push_back(frag);
                usedIndices[frag.index] = true;
            }
        }

        if (static_cast<int>(selected.size()) < k) {
            throw std::runtime_error("RabinIDA: Not enough distinct fragments");
        }

        size_t fragSize = selected[0].data.size();
        uint32_t originalSize = selected[0].originalSize;
        for (const auto& frag : selected) {
            if (frag.data.size() != fragSize) {
                throw std::runtime_error("RabinIDA: Fragment size mismatch");
            }
        }

        std::vector<std::vector<uint8_t>> matrix(k, std::vector<uint8_t>(k));
        for (int i = 0; i < k; i++) {
            uint8_t x = selected[i].index;
            uint8_t x_pow = 1;
            for (int j = 0; j < k; j++) {
                matrix[i][j] = x_pow;
                x_pow = GF256::mul(x_pow, x);
            }
        }
        std::vector<std::vector<uint8_t>> inverse = invertMatrix(matrix, k);

        size_t chunkSize = fragSize;
        std::vector<uint8_t> reconstructed(k * chunkSize);

        for (size_t pos = 0; pos < chunkSize; pos++) {
            // Build fragment value vector for this position
            std::vector<uint8_t> fragVals(k);
            for (int i = 0; i < k; i++) {
                fragVals[i] = selected[i].data[pos];
            }

            // Multiply inverse matrix by fragment values
            for (int t = 0; t < k; t++) {
                uint8_t sum = 0;
                for (int i = 0; i < k; i++) {
                    sum = GF256::add(sum, GF256::mul(inverse[t][i], fragVals[i]));
                }
                reconstructed[t * chunkSize + pos] = sum;
            }
        }

        // Truncate to original size
        reconstructed.resize(originalSize);
        return reconstructed;
    }

    static std::string combineToString(const std::vector<Fragment>& fragments, int k) {
        std::vector<uint8_t> data = combine(fragments, k);
        return std::string(data.begin(), data.end());
    }

private:
    // Invert a k×k matrix in GF(256) using Gaussian elimination
    static std::vector<std::vector<uint8_t>> invertMatrix(
        const std::vector<std::vector<uint8_t>>& matrix, int k) 
    {
        // Create augmented matrix [A | I]
        std::vector<std::vector<uint8_t>> augmented(k, std::vector<uint8_t>(2 * k));
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                augmented[i][j] = matrix[i][j];
            }
            augmented[i][k + i] = 1; // Identity matrix on the right
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < k; col++) {
            // Find pivot row
            int pivotRow = -1;
            for (int row = col; row < k; row++) {
                if (augmented[row][col] != 0) {
                    pivotRow = row;
                    break;
                }
            }

            if (pivotRow == -1) {
                throw std::runtime_error("RabinIDA: Matrix is singular, cannot invert");
            }

            // Swap rows if needed
            if (pivotRow != col) {
                std::swap(augmented[col], augmented[pivotRow]);
            }

            // Scale pivot row to make pivot = 1
            uint8_t pivotVal = augmented[col][col];
            uint8_t pivotInv = GF256::inv(pivotVal);
            for (int j = 0; j < 2 * k; j++) {
                augmented[col][j] = GF256::mul(augmented[col][j], pivotInv);
            }

            // Eliminate column in other rows
            for (int row = 0; row < k; row++) {
                if (row != col && augmented[row][col] != 0) {
                    uint8_t factor = augmented[row][col];
                    for (int j = 0; j < 2 * k; j++) {
                        augmented[row][j] = GF256::sub(
                            augmented[row][j],
                            GF256::mul(factor, augmented[col][j])
                        );
                    }
                }
            }
        }

        // Extract inverse from right half
        std::vector<std::vector<uint8_t>> inverse(k, std::vector<uint8_t>(k));
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                inverse[i][j] = augmented[i][k + j];
            }
        }

        return inverse;
    }
};

// Static member definitions
uint8_t RabinIDA::GF256::EXP_TABLE[256] = {0};
uint8_t RabinIDA::GF256::LOG_TABLE[256] = {0};
bool RabinIDA::GF256::tablesInitialized = false;

}

#endif 

