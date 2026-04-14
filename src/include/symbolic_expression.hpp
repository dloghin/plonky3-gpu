#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace p3_air {

struct SymbolicVariable {
    size_t row_offset = 0;
    size_t column = 0;

    bool operator==(const SymbolicVariable& other) const {
        return row_offset == other.row_offset && column == other.column;
    }

    bool operator!=(const SymbolicVariable& other) const {
        return !(*this == other);
    }
};

template<typename F>
class SymbolicExpression {
public:
    enum class Kind {
        Constant,
        Variable,
        Add,
        Mul,
        Sub,
        Neg
    };

private:
    struct Node {
        Kind kind;
        F constant{};
        SymbolicVariable variable{};
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;

        explicit Node(const F& c) : kind(Kind::Constant), constant(c) {}
        explicit Node(const SymbolicVariable& v) : kind(Kind::Variable), variable(v) {}
        Node(Kind k, std::shared_ptr<Node> lhs, std::shared_ptr<Node> rhs)
            : kind(k), left(std::move(lhs)), right(std::move(rhs)) {}
        Node(Kind k, std::shared_ptr<Node> child) : kind(k), left(std::move(child)) {}
    };

public:
    SymbolicExpression() : root_(std::make_shared<Node>(F{})) {}
    explicit SymbolicExpression(const F& constant) : root_(std::make_shared<Node>(constant)) {}
    explicit SymbolicExpression(const SymbolicVariable& variable)
        : root_(std::make_shared<Node>(variable)) {}

    static SymbolicExpression constant(const F& value) {
        return SymbolicExpression(value);
    }

    static SymbolicExpression variable(size_t row_offset, size_t column) {
        return SymbolicExpression(SymbolicVariable{row_offset, column});
    }

    Kind kind() const {
        return root_->kind;
    }

    const F& constant_value() const {
        if (root_->kind != Kind::Constant) {
            throw std::logic_error("Expression node is not a constant");
        }
        return root_->constant;
    }

    const SymbolicVariable& variable_value() const {
        if (root_->kind != Kind::Variable) {
            throw std::logic_error("Expression node is not a variable");
        }
        return root_->variable;
    }

    template<typename Resolver>
    F evaluate(Resolver&& resolve_variable) const {
        return evaluate_node(root_, std::forward<Resolver>(resolve_variable));
    }

    SymbolicExpression operator+(const SymbolicExpression& other) const {
        return from_node(std::make_shared<Node>(Kind::Add, root_, other.root_));
    }

    SymbolicExpression operator-(const SymbolicExpression& other) const {
        return from_node(std::make_shared<Node>(Kind::Sub, root_, other.root_));
    }

    SymbolicExpression operator*(const SymbolicExpression& other) const {
        return from_node(std::make_shared<Node>(Kind::Mul, root_, other.root_));
    }

    SymbolicExpression operator-() const {
        return from_node(std::make_shared<Node>(Kind::Neg, root_));
    }

private:
    std::shared_ptr<Node> root_;

    explicit SymbolicExpression(std::shared_ptr<Node> node) : root_(std::move(node)) {}

    static SymbolicExpression from_node(std::shared_ptr<Node> node) {
        return SymbolicExpression(std::move(node));
    }

    template<typename Resolver>
    static F evaluate_node(const std::shared_ptr<Node>& node, Resolver&& resolve_variable) {
        switch (node->kind) {
            case Kind::Constant:
                return node->constant;
            case Kind::Variable:
                return resolve_variable(node->variable);
            case Kind::Add:
                return evaluate_node(node->left, std::forward<Resolver>(resolve_variable)) +
                       evaluate_node(node->right, std::forward<Resolver>(resolve_variable));
            case Kind::Mul:
                return evaluate_node(node->left, std::forward<Resolver>(resolve_variable)) *
                       evaluate_node(node->right, std::forward<Resolver>(resolve_variable));
            case Kind::Sub:
                return evaluate_node(node->left, std::forward<Resolver>(resolve_variable)) -
                       evaluate_node(node->right, std::forward<Resolver>(resolve_variable));
            case Kind::Neg:
                return -evaluate_node(node->left, std::forward<Resolver>(resolve_variable));
            default:
                throw std::logic_error("Unsupported symbolic expression kind");
        }
    }
};

template<typename F>
SymbolicExpression<F> operator+(const F& lhs, const SymbolicExpression<F>& rhs) {
    return SymbolicExpression<F>(lhs) + rhs;
}

template<typename F>
SymbolicExpression<F> operator-(const F& lhs, const SymbolicExpression<F>& rhs) {
    return SymbolicExpression<F>(lhs) - rhs;
}

template<typename F>
SymbolicExpression<F> operator*(const F& lhs, const SymbolicExpression<F>& rhs) {
    return SymbolicExpression<F>(lhs) * rhs;
}

} // namespace p3_air
