//
// Created by Clytie on 2018/11/3.
//

#ifndef DEEP_LEARNING_ACTIVEFUN_H
#define DEEP_LEARNING_ACTIVEFUN_H

#include <cmath>
#include "../Matrix.h"

static const float MAX_EXP = 50;

template <typename T>
class ActiveFun {
public:
    virtual matrix::Matrix<T> Forward(const matrix::Matrix<T> & inputs_) = 0;
    virtual matrix::Matrix<T> grad_() = 0;
};

template <typename T>
class Sigmoid : public ActiveFun<T> {
public:
    Sigmoid() : __grads(0, 0) {}
    ~Sigmoid() = default;

    matrix::Matrix<T> Forward(const matrix::Matrix<T> & inputs_) {
        matrix::Matrix<T> outputs_(inputs_.nrow, inputs_.ncol);
        __grads = matrix::Matrix<T>(inputs_.nrow, inputs_.ncol);
        for (size_t i = 0; i < inputs_.nrow; ++i) {
            for (size_t j = 0; j < inputs_.ncol; ++j) {
                if (inputs_(i, j) < -MAX_EXP) {
                    outputs_(i, j) = 0;
                    __grads(i, j) = 0;
                } else {
                    auto fExp = std::exp(-inputs_(i, j));
                    auto fTmp = 1.0 + fExp;
                    outputs_(i, j) = 1.0 / fTmp;
                    __grads(i, j) = fExp / (fTmp * fTmp);
                }
            }
        }
        return outputs_;
    }

    matrix::Matrix<T> grad_() {
        return __grads;
    }

private:
    matrix::Matrix<T> __grads;
};

template <typename T>
class Tanh : public ActiveFun<T> {
public:
    Tanh() : __grads(0, 0) {}
    ~Tanh() = default;

    matrix::Matrix<T> Forward(const matrix::Matrix<T> & inputs_) {
        matrix::Matrix<T> outputs_(inputs_.nrow, inputs_.ncol);
        __grads = matrix::Matrix<T>(inputs_.nrow, inputs_.ncol);
        for (size_t i = 0; i < inputs_.nrow; ++i) {
            for (size_t j = 0; j < inputs_.ncol; ++j) {
                if (inputs_(i, j) < -(MAX_EXP / 2.0)) {
                    outputs_(i, j) = 0;
                    __grads(i, j) = 0;
                } else {
                    auto fExp = std::exp(-2.0 * inputs_(i, j));
                    auto fTmp = 1.0f + fExp;
                    outputs_(i, j) = 2.0 / fTmp - 1;
                    __grads(i, j) = 4.0 * fExp / (fTmp * fTmp);
                }
            }
        }
        return outputs_;
    }

    matrix::Matrix<T> grad_() {
        return __grads;
    }

private:
    matrix::Matrix<T> __grads;
};

template <typename T>
class ReLU : public ActiveFun<T> {
public:
    ReLU() : __grads(0, 0) {}
    ~ReLU() = default;

    matrix::Matrix<T> Forward(const matrix::Matrix<T> & inputs_) {
        matrix::Matrix<T> outputs_(inputs_.nrow, inputs_.ncol);
        __grads = matrix::Matrix<T>(inputs_.nrow, inputs_.ncol);
        for (size_t i = 0; i < inputs_.nrow; ++i) {
            for (size_t j = 0; j < inputs_.ncol; ++j) {
                if (inputs_(i, j) <= 0) {
                    outputs_(i, j) = 0;
                    __grads(i, j) = 0;
                } else {
                    outputs_(i, j) = inputs_(i, j);
                    __grads(i, j) = 1;
                }
            }
        }
        return outputs_;
    }

    matrix::Matrix<T> grad_() {
        return __grads;
    }

private:
    matrix::Matrix<T> __grads;

};

#endif //DEEP_LEARNING_ACTIVEFUN_H
