#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace p3_challenger {

template<typename C, typename T, typename = void>
struct can_observe : std::false_type {};
template<typename C, typename T>
struct can_observe<C, T, std::void_t<decltype(std::declval<C&>().observe(std::declval<T>()))>>
    : std::true_type {};

template<typename C, typename T, typename = void>
struct can_observe_slice : std::false_type {};
template<typename C, typename T>
struct can_observe_slice<C, T,
    std::void_t<decltype(std::declval<C&>().observe_slice(std::declval<const std::vector<T>&>()))>>
    : std::true_type {};

template<typename C, typename T, typename = void>
struct can_sample : std::false_type {};
template<typename C, typename T>
struct can_sample<C, T, std::void_t<decltype(std::declval<C&>().sample())>>
    : std::is_same<decltype(std::declval<C&>().sample()), T> {};

template<typename C, typename T, typename = void>
struct can_sample_bits : std::false_type {};
template<typename C, typename T>
struct can_sample_bits<C, T, std::void_t<decltype(std::declval<C&>().sample_bits(std::declval<size_t>()))>>
    : std::is_same<decltype(std::declval<C&>().sample_bits(std::declval<size_t>())), T> {};

template<typename C, typename T, size_t N>
inline std::array<T, N> sample_array(C& challenger) {
    std::array<T, N> out{};
    for (size_t i = 0; i < N; ++i) out[i] = challenger.sample();
    return out;
}

template<typename C, typename F>
struct is_field_challenger : std::integral_constant<bool,
    can_observe<C, F>::value &&
    can_sample<C, F>::value &&
    can_sample_bits<C, size_t>::value> {};

} // namespace p3_challenger
