//
// Created by Clytie on 2018/11/3.
//

#ifndef DEEP_LEARNING_LOSS_H
#define DEEP_LEARNING_LOSS_H

#include <cmath>
#include "../Matrix.h"

template <typename T>
class SoftMaxLoss {
public:
    SoftMaxLoss() : __grads(0, 0) {}
    ~SoftMaxLoss() = default;

    size_t Forward(matrix::Matrix<T> & inputs_, size_t label, T & loss) {
        assert(inputs_.ncol == 1);
        T inputs_max = inputs_.max_element();
        matrix::Matrix<T> exps(inputs_.nrow, 1);
        for (size_t i = 0; i < inputs_.nrow; ++i) {
            exps(i, 0) = std::exp(inputs_(i, 0) - inputs_max);
        }
        auto Sum = 0.0;
        for (size_t i = 0; i < inputs_.nrow; ++i) {
            Sum += exps(i, 0);
        }
        exps /= Sum;

        loss = -log(exps(label, 0) + 1e-10);

        __grads = exps;
        __grads(label, 0) -= 1;

        return exps.max_index().first;
    }

    matrix::Matrix<T> grad_() {
        return __grads;
    }

private:
    matrix::Matrix<T> __grads;

};
#endif //DEEP_LEARNING_LOSS_H
