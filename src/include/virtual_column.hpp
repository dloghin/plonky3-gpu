#pragma once

#include "air.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace p3_air {

template<typename F>
struct VirtualColumnTerm {
    size_t row_offset;
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

    static VirtualColumn single(size_t column, size_t row_offset = 0) {
        return VirtualColumn({VirtualColumnTerm<F>{row_offset, column, F{1}}});
    }

    const std::vector<VirtualColumnTerm<F>>& terms() const {
        return terms_;
    }

    F constant() const {
        return constant_;
    }

    void add_term(size_t row_offset, size_t column, const F& weight) {
        terms_.push_back(VirtualColumnTerm<F>{row_offset, column, weight});
    }

    template<typename T>
    T evaluate(const RowWindow<T>& main_window) const {
        T result = T(constant_);
        for (const auto& term : terms_) {
            if (term.row_offset == 0) {
                result = result + main_window.current(term.column) * T(term.weight);
            } else if (term.row_offset == 1) {
                result = result + main_window.next(term.column) * T(term.weight);
            } else {
                throw std::out_of_range("VirtualColumn row_offset must be 0 or 1");
            }
        }
        return result;
    }

private:
    std::vector<VirtualColumnTerm<F>> terms_;
    F constant_ = F{};
};

} // namespace p3_air
