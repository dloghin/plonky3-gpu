#pragma once

#include "bn254.hpp"
#include "goldilocks.hpp"
#include "koala_bear.hpp"
#include "mersenne31.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace p3_field {

/**
 * @brief Serialization utilities for field elements.
 *
 * Implements byte-level serialization/deserialization to match Rust's
 * RawDataSerializable trait, ensuring cross-language compatibility.
 */
class Serialization {
public:
  static std::vector<uint8_t>
  encode_goldilocks(const std::vector<Goldilocks> &elements) {
    std::vector<uint8_t> bytes;
    bytes.reserve(elements.size() * 8);
    for (const auto &el : elements) {
      const uint64_t val = el.as_canonical_u64();
      for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<uint8_t>((val >> (8 * i)) & 0xFFu));
      }
    }
    return bytes;
  }

  static std::vector<Goldilocks>
  decode_goldilocks(const std::vector<uint8_t> &bytes) {
    if (bytes.size() % 8 != 0) {
      throw std::runtime_error(
          "Invalid byte size for Goldilocks deserialization");
    }

    std::vector<Goldilocks> elements;
    elements.reserve(bytes.size() / 8);
    for (size_t i = 0; i < bytes.size(); i += 8) {
      uint64_t val = 0;
      for (int j = 0; j < 8; ++j) {
        val |= (static_cast<uint64_t>(bytes[i + j]) << (8 * j));
      }
      elements.emplace_back(val);
    }
    return elements;
  }

  static std::vector<uint8_t>
  encode_koalabear(const std::vector<KoalaBear> &elements) {
    std::vector<uint8_t> bytes;
    bytes.reserve(elements.size() * 4);
    for (const auto &el : elements) {
      const uint32_t val = static_cast<uint32_t>(el.as_canonical_u64());
      for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>((val >> (8 * i)) & 0xFFu));
      }
    }
    return bytes;
  }

  static std::vector<KoalaBear>
  decode_koalabear(const std::vector<uint8_t> &bytes) {
    if (bytes.size() % 4 != 0) {
      throw std::runtime_error("Invalid byte size for KoalaBear deserialization");
    }

    std::vector<KoalaBear> elements;
    elements.reserve(bytes.size() / 4);
    for (size_t i = 0; i < bytes.size(); i += 4) {
      uint32_t val = 0;
      for (int j = 0; j < 4; ++j) {
        val |= (static_cast<uint32_t>(bytes[i + j]) << (8 * j));
      }
      elements.emplace_back(val);
    }
    return elements;
  }

  static std::vector<uint8_t>
  encode_mersenne31(const std::vector<Mersenne31> &elements) {
    std::vector<uint8_t> bytes;
    bytes.reserve(elements.size() * 4);
    for (const auto &el : elements) {
      const uint32_t val = static_cast<uint32_t>(el.as_canonical_u64());
      for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>((val >> (8 * i)) & 0xFFu));
      }
    }
    return bytes;
  }

  static std::vector<Mersenne31>
  decode_mersenne31(const std::vector<uint8_t> &bytes) {
    if (bytes.size() % 4 != 0) {
      throw std::runtime_error("Invalid byte size for Mersenne31 deserialization");
    }

    std::vector<Mersenne31> elements;
    elements.reserve(bytes.size() / 4);
    for (size_t i = 0; i < bytes.size(); i += 4) {
      uint32_t val = 0;
      for (int j = 0; j < 4; ++j) {
        val |= (static_cast<uint32_t>(bytes[i + j]) << (8 * j));
      }
      elements.emplace_back(val);
    }
    return elements;
  }

  static std::vector<uint8_t> encode_bn254(const std::vector<Bn254> &elements) {
    std::vector<uint8_t> bytes;
    bytes.reserve(elements.size() * 32);
    for (const auto &el : elements) {
      uint64_t limbs[4];
      el.as_canonical(limbs);
      for (int limb = 0; limb < 4; ++limb) {
        const uint64_t v = limbs[limb];
        for (int i = 0; i < 8; ++i) {
          bytes.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFFu));
        }
      }
    }
    return bytes;
  }

  static std::vector<Bn254> decode_bn254(const std::vector<uint8_t> &bytes) {
    if (bytes.size() % 32 != 0) {
      throw std::runtime_error("Invalid byte size for Bn254 deserialization");
    }

    std::vector<Bn254> elements;
    elements.reserve(bytes.size() / 32);
    for (size_t i = 0; i < bytes.size(); i += 32) {
      uint64_t limbs[4] = {0, 0, 0, 0};
      for (int limb = 0; limb < 4; ++limb) {
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) {
          v |= (static_cast<uint64_t>(bytes[i + (8 * limb) + j]) << (8 * j));
        }
        limbs[limb] = v;
      }
      elements.push_back(Bn254::from_canonical(limbs));
    }
    return elements;
  }
};

} // namespace p3_field
