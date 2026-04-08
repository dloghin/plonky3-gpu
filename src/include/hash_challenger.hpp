#pragma once

#include "challenger_traits.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace p3_challenger {

template<typename T, typename Hasher, size_t OUT_SIZE>
class HashChallenger {
public:
    explicit HashChallenger(Hasher hasher, std::vector<T> initial_state = {})
        : input_buffer_(std::move(initial_state)), hasher_(std::move(hasher)) {}

    void observe(T value) {
        output_buffer_.clear();
        input_buffer_.push_back(std::move(value));
    }

    void observe_slice(const std::vector<T>& values) {
        if (values.empty()) return;
        output_buffer_.clear();
        input_buffer_.insert(input_buffer_.end(), values.begin(), values.end());
    }

    T sample() {
        if (output_buffer_.empty()) {
            flush();
        }
        T out = output_buffer_.back();
        output_buffer_.pop_back();
        return out;
    }

    const std::vector<T>& input_buffer() const { return input_buffer_; }
    const std::vector<T>& output_buffer() const { return output_buffer_; }

private:
    std::vector<T> input_buffer_;
    std::vector<T> output_buffer_;
    Hasher hasher_;

    void flush() {
        const auto out = hasher_.hash_iter(input_buffer_);
        input_buffer_.assign(out.begin(), out.end());
        output_buffer_.assign(out.begin(), out.end());
    }
};

} // namespace p3_challenger
