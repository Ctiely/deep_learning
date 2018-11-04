//
// Created by Clytie on 2018/10/31.
//

#ifndef DEEP_LEARNING_MATRIX_H
#define DEEP_LEARNING_MATRIX_H

#include <utility>
#include <Eigen/Eigen>

namespace matrix {
    template <typename T>
    class Matrix {
    public:
        Matrix(size_t nrow, size_t ncol) : nrow(nrow), ncol(ncol), __data(nrow, ncol) {
            __data.setZero();
        };
        explicit Matrix(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> & data_)
                : nrow((size_t)data_.rows()), ncol((size_t)data_.cols()), __data(std::move(data_)) {}
        explicit Matrix(std::vector<std::vector<T> > & data_)
                : nrow(data_.size()), ncol(data_.front().size()), __data(nrow, ncol) {
            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    __data(i, j) = data_[i][j];
                }
            }
        }
        template <typename __Generator>
        Matrix(size_t nrow, size_t ncol, __Generator generator)
                : nrow(nrow), ncol(ncol), __data(nrow, ncol) {
            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    __data(i, j) = generator();
                }
            }
        }
        explicit Matrix(std::vector<T> & data_, bool rowVec=true) {
            if (rowVec) {
                nrow = 1;
                ncol = data_.size();
            } else {
                nrow = data_.size();
                ncol = 1;
            }
            __data.resize(nrow, ncol);
            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    __data(i, j) = data_[i + j];
                }
            }
        }
        ~Matrix() = default;

        inline T operator()(size_t i, size_t j) const {
            return __data(i, j);
        }

        inline T &operator()(size_t i, size_t j) {
            return __data(i, j);
        }

        inline Matrix<T> operator()(size_t i) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> row = __data.row(i);
            return Matrix<T>(row);
        }

        Matrix<T> & operator=(const Matrix<T> & other) {
            if (this != &other) {
                nrow = other.nrow;
                ncol = other.ncol;
                __data = other.__data;
            }
            return *this;
        }

        Matrix<T> dot(const Matrix<T> & other) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> dot = __data * other.__data;
            return Matrix<T>(dot);
        }

        Matrix<T> operator*(const Matrix<T> & other) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> product = __data.cwiseProduct(other.__data);
            return Matrix<T>(product);
        }

        Matrix<T> operator+(const Matrix<T> & other) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> add = __data + other.__data;
            return Matrix<T>(add);
        }

        Matrix<T> operator-(const Matrix<T> & other) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> sub = __data - other.__data;
            return Matrix<T>(sub);
        }

        Matrix<T> operator-() const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> negative = -__data;
            return Matrix<T>(negative);
        }

        void operator+=(const Matrix<T> & other) {
            __data += other.__data;
        }

        void operator-=(const Matrix<T> & other) {
            __data -= other.__data;
        }

        void operator*=(const Matrix<T> & other) {
            __data = __data.cwiseProduct(other.__data);
        }

        Matrix<T> operator*(T scalar) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> multiply = __data * scalar;
            return Matrix<T>(multiply);
        }

        inline Matrix<T> operator/(T scalar) const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> divide = __data / scalar;
            return Matrix<T>(divide);
        }

        void operator*=(T scalar) {
            __data *= scalar;
        }

        void operator/=(T scalar) {
            __data /= scalar;
        }

        Matrix<T> transpose() const {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> t = __data.transpose();
            return Matrix<T>(t);
        }

        inline Matrix<T> t() const {
            return transpose();
        }

        bool isEmpty() const {
            return __data.size() == 0;
        }

        void setRandom() {
            __data.setRandom();
        }

        void setZero() {
            __data.setZero();
        }

        void setOnes() {
            __data.setOnes();
        }

        void resize(size_t rows, size_t cols) {
            nrow = rows;
            ncol = cols;
            __data.resize(rows, cols);
        }

        T max_element() {
            return __data.maxCoeff();
        }

        std::pair<size_t, size_t> max_index() {
            size_t i, j;
            __data.maxCoeff(&i, &j);
            return std::make_pair(i, j);
        }

        bool operator==(const Matrix<T> & other) const {
            return __data == other.__data;
        }

        bool operator!=(const Matrix<T> & other) const {
            return !(*this == other);
        }

        void print() const {
            printf("[");
            for (size_t i = 0; i < nrow; ++i) {
                printf("[");
                for (size_t j = 0; j < ncol; ++j) {
                    if (j < ncol - 1) {
                        printf("%f,", __data(i, j));
                    } else {
                        printf("%f", __data(i, j));
                    }
                }
                if (i < nrow - 1) {
                    printf("],\n");
                } else {
                    printf("]");
                }
            }
            printf("]\n");
        }

        size_t nrow, ncol;
    private:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> __data;
    };
}

#endif //DEEP_LEARNING_MATRIX_H
