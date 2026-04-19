#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

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

    enum class EvalPhase : unsigned char { Start, AfterLeft, AfterRight };

    struct EvalFrame {
        std::shared_ptr<Node> node;
        EvalPhase phase = EvalPhase::Start;
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
        auto&& resolver = std::forward<Resolver>(resolve_variable);

        std::vector<EvalFrame> stack;
        std::vector<F> values;
        stack.push_back(EvalFrame{node, EvalPhase::Start});

        while (!stack.empty()) {
            EvalFrame& top = stack.back();
            switch (top.phase) {
                case EvalPhase::Start:
                    switch (top.node->kind) {
                        case Kind::Constant:
                            values.push_back(top.node->constant);
                            stack.pop_back();
                            break;
                        case Kind::Variable:
                            values.push_back(resolver(top.node->variable));
                            stack.pop_back();
                            break;
                        case Kind::Neg:
                            top.phase = EvalPhase::AfterLeft;
                            stack.push_back(EvalFrame{top.node->left, EvalPhase::Start});
                            break;
                        case Kind::Add:
                        case Kind::Mul:
                        case Kind::Sub:
                            top.phase = EvalPhase::AfterLeft;
                            stack.push_back(EvalFrame{top.node->left, EvalPhase::Start});
                            break;
                        default:
                            throw std::logic_error("Unsupported symbolic expression kind");
                    }
                    break;
                case EvalPhase::AfterLeft:
                    switch (top.node->kind) {
                        case Kind::Neg: {
                            F v = values.back();
                            values.pop_back();
                            values.push_back(-v);
                            stack.pop_back();
                            break;
                        }
                        case Kind::Add:
                        case Kind::Mul:
                        case Kind::Sub:
                            top.phase = EvalPhase::AfterRight;
                            stack.push_back(EvalFrame{top.node->right, EvalPhase::Start});
                            break;
                        default:
                            throw std::logic_error("Invalid symbolic evaluation state");
                    }
                    break;
                case EvalPhase::AfterRight: {
                    F right = values.back();
                    values.pop_back();
                    F left = values.back();
                    values.pop_back();
                    F out{};
                    switch (top.node->kind) {
                        case Kind::Add:
                            out = left + right;
                            break;
                        case Kind::Mul:
                            out = left * right;
                            break;
                        case Kind::Sub:
                            out = left - right;
                            break;
                        default:
                            throw std::logic_error("Invalid symbolic evaluation state");
                    }
                    values.push_back(out);
                    stack.pop_back();
                    break;
                }
            }
        }

        if (values.size() != 1) {
            throw std::logic_error("Symbolic evaluation internal error");
        }
        return values.back();
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
