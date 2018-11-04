//
// Created by Clytie on 2018/11/3.
//

#ifndef DEEP_LEARNING_OPTIMIZATION_H
#define DEEP_LEARNING_OPTIMIZATION_H

#include "../Matrix.h"

template <typename T>
class GradientDescent {
public:
    void Update(matrix::Matrix<T> & weights_,
                matrix::Matrix<T> & bias_,
                matrix::Matrix<T> & weights_grads_,
                matrix::Matrix<T> & bias_grads_,
                T learning_rate) {
        weights_ -=  weights_grads_ * learning_rate;
        bias_ -= bias_grads_ * learning_rate;
    }
};

#endif //DEEP_LEARNING_OPTIMIZATION_H
