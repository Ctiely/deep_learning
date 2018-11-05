//
// Created by Clytie on 2018/11/3.
//

#ifndef DEEP_LEARNING_DENSELAYER_H
#define DEEP_LEARNING_DENSELAYER_H

#include <iostream>
#include "../Matrix.h"

template <typename T>
class DenseLayer {
public:
    DenseLayer(size_t last_n_neurons,
               size_t n_neurons)
            : last_n_neurons(last_n_neurons),
              n_neurons(n_neurons),
              weights_(n_neurons, last_n_neurons),
              bias_(n_neurons, 1),
              outputs_(0, 0),
              grads_(0, 0) {}

    template <typename __Gen>
    DenseLayer(size_t last_n_neurons,
               size_t n_neurons,
               __Gen generator)
            : last_n_neurons(last_n_neurons),
              n_neurons(n_neurons),
              weights_(n_neurons, last_n_neurons, generator),
              bias_(n_neurons, 1, generator),
              outputs_(0, 0),
              grads_(0, 0) {}

    DenseLayer(size_t last_n_neurons,
               size_t n_neurons,
               std::vector<std::vector<float> > & initialize_weight,
               std::vector<std::vector<float> > & initialize_bias)
            : last_n_neurons(last_n_neurons),
              n_neurons(n_neurons),
              weights_(0, 0),
              bias_(0, 0),
              outputs_(0, 0),
              grads_(0, 0) {
        if (last_n_neurons) {
            weights_ = matrix::Matrix<T>(initialize_weight);
            bias_ = matrix::Matrix<T>(initialize_bias);
        }
    }

    void Forward(const matrix::Matrix<T> & inputs_) {
        outputs_ = weights_.dot(inputs_) + bias_;
    }

    void Backward(const matrix::Matrix<T> & input_weights_, matrix::Matrix<T> & input_grads_, matrix::Matrix<T> & active_grads_) {
        if (input_weights_.isEmpty()) {
            grads_ = input_grads_ * active_grads_;
        } else {
            grads_ = input_weights_.t().dot(input_grads_) * active_grads_;
        }
    }

    size_t last_n_neurons;
    size_t n_neurons;
    matrix::Matrix<T> weights_; //W
    matrix::Matrix<T> bias_; //b
    matrix::Matrix<T> outputs_; //z
    matrix::Matrix<T> grads_; //\delta
};


#endif //DEEP_LEARNING_DENSELAYER_H
