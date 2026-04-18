#pragma once

#include "air.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_air {

template<typename F>
struct VirtualColumnTerm {
    enum class Source {
        Preprocessed,
        Main
    };

    Source source = Source::Main;
    size_t column;
    F weight;
};

template<typename F>
class VirtualColumn {
public:
    VirtualColumn() = default;

    explicit VirtualColumn(F constant) : constant_(constant) {}

    VirtualColumn(std::vector<VirtualColumnTerm<F>> terms, F constant = F{})
        : terms_(std::move(terms)), constant_(constant) {}

    static VirtualColumn single_main(size_t column) {
        return VirtualColumn({VirtualColumnTerm<F>{VirtualColumnTerm<F>::Source::Main, column, F{1}}});
    }

    static VirtualColumn single_preprocessed(size_t column) {
        return VirtualColumn({VirtualColumnTerm<F>{VirtualColumnTerm<F>::Source::Preprocessed, column, F{1}}});
    }

    const std::vector<VirtualColumnTerm<F>>& terms() const {
        return terms_;
    }

    F constant() const {
        return constant_;
    }

    void add_main_term(size_t column, const F& weight) {
        terms_.push_back(VirtualColumnTerm<F>{VirtualColumnTerm<F>::Source::Main, column, weight});
    }

    void add_preprocessed_term(size_t column, const F& weight) {
        terms_.push_back(VirtualColumnTerm<F>{VirtualColumnTerm<F>::Source::Preprocessed, column, weight});
    }

    template<typename T>
    T evaluate(const std::vector<T>& preprocessed_row, const std::vector<T>& main_row) const {
        T result = T(constant_);
        for (const auto& term : terms_) {
            if (term.source == VirtualColumnTerm<F>::Source::Main) {
                if (term.column >= main_row.size()) {
                    throw std::out_of_range("VirtualColumn main column out of bounds");
                }
                result = result + main_row[term.column] * T(term.weight);
            } else {
                if (term.column >= preprocessed_row.size()) {
                    throw std::out_of_range("VirtualColumn preprocessed column out of bounds");
                }
                result = result + preprocessed_row[term.column] * T(term.weight);
            }
        }
        return result;
    }

private:
    std::vector<VirtualColumnTerm<F>> terms_;
    F constant_ = F{};
};

} // namespace p3_air
