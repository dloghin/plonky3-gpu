#pragma once

#include <vector>
#include <utility>
#include <functional>
#include <optional>

namespace p3_util {

/// A simple key-value map backed by a std::vector<std::pair<K,V>>.
///
/// Lookup is O(n) linear scan -- suitable only for small key sets.
/// Mirrors Rust's p3_util::LinearMap<K,V>.
///
/// Used in the FRI verifier for mapping opening points to inverse denominators.
template<typename K, typename V>
class LinearMap {
public:
    using value_type = std::pair<K, V>;
    using container_type = std::vector<value_type>;
    using iterator = typename container_type::iterator;
    using const_iterator = typename container_type::const_iterator;

    LinearMap() = default;

    // -----------------------------------------------------------------------
    // Lookup
    // -----------------------------------------------------------------------

    /// Returns a pointer to the value associated with key, or nullptr if not found.
    V* get(const K& key) {
        for (auto& [k, v] : data_) {
            if (k == key) return &v;
        }
        return nullptr;
    }

    const V* get(const K& key) const {
        for (const auto& [k, v] : data_) {
            if (k == key) return &v;
        }
        return nullptr;
    }

    /// Alias for get() -- returns a mutable pointer or nullptr.
    V* get_mut(const K& key) { return get(key); }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Inserts or overwrites the value for key.
    void insert(const K& key, V value) {
        for (auto& [k, v] : data_) {
            if (k == key) {
                v = std::move(value);
                return;
            }
        }
        data_.emplace_back(key, std::move(value));
    }

    /// Returns a reference to the value for key.
    /// If the key is absent, inserts the value produced by calling make_value().
    /// Mirrors Rust's LinearMap::get_or_insert_with.
    template<typename F>
    V& get_or_insert_with(const K& key, F make_value) {
        for (auto& [k, v] : data_) {
            if (k == key) return v;
        }
        data_.emplace_back(key, make_value());
        return data_.back().second;
    }

    // -----------------------------------------------------------------------
    // Iteration helpers
    // -----------------------------------------------------------------------

    /// Returns a range over all values (mutable).
    std::vector<V*> values() {
        std::vector<V*> result;
        result.reserve(data_.size());
        for (auto& [k, v] : data_) result.push_back(&v);
        return result;
    }

    /// Returns a range over all keys (const).
    std::vector<const K*> keys() const {
        std::vector<const K*> result;
        result.reserve(data_.size());
        for (const auto& [k, v] : data_) result.push_back(&k);
        return result;
    }

    /// Iterator access to the underlying key-value pairs.
    iterator begin() { return data_.begin(); }
    iterator end()   { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end()   const { return data_.end(); }
    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend()   const { return data_.cend(); }

    /// iter() returns a const reference to the underlying storage for range-for.
    const container_type& iter() const { return data_; }

    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

private:
    container_type data_;
};

} // namespace p3_util
