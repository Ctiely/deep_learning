//
// Created by Clytie on 2018/10/31.
//

#ifndef DEEP_LEARNING_MATRIX_H
#define DEEP_LEARNING_MATRIX_H

#include <cstdio>
#include <utility>

namespace matrix {
    template <typename T>
    class Matrix {
    public:
        Matrix(size_t nrow, size_t ncol, bool initialize=true) : nrow(nrow), ncol(ncol), size(nrow * ncol) {
            //__data = new T[size];
            if (initialize) {
                setZero();
            }
        }
        explicit Matrix(std::vector<std::vector<T> > & data_)
                : nrow(data_.size()), ncol(data_.front().size()), size(nrow * ncol) {
            //__data = new T[size];
            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    //__data[j + i * ncol] = data_[i][j];
                    __data.push_back(data_[i][j]);
                }
            }
        }
        template <typename __Generator>
        Matrix(size_t nrow, size_t ncol, __Generator generator)
                : nrow(nrow), ncol(ncol), size(nrow * ncol) {
            //__data = new T[size];
            for (size_t i = 0; i < size; ++i) {
                //__data[i] = generator();
                __data.push_back(generator());
            }
        }
        explicit Matrix(std::vector<T> & data_, bool rowVec=true)
                : size(data_.size()) {
            if (rowVec) {
                nrow = 1;
                ncol = size;
            } else {
                nrow = size;
                ncol = 1;
            }
            //__data = new T[size];
            for (size_t i = 0; i < size; ++i) {
                //__data[i] = data_[i];
                __data.push_back(data_[i]);
            }
        }

        inline T operator()(size_t i, size_t j) const {
            return __data[j + i * ncol];
        }

        inline T &operator()(size_t i, size_t j) {
            return __data[j + i * ncol];
        }

        inline Matrix<T> operator()(size_t i) const {
            Matrix<T> row(1, ncol, false);
            for (size_t j = 0; j < ncol; ++j) {
                //row.__data[j] = __data[i * ncol + j];
                row.__data.push_back(__data[i * ncol + j]);
            }
            return row;
        }

        Matrix<T> & operator=(const Matrix<T> & other) {
            if (this != &other) {
                nrow = other.nrow;
                ncol = other.ncol;
                size = nrow * ncol;
                //__data = new T[size];
                //memcpy(__data, other.__data, sizeof(__data));
                __data = other.__data;
            }
            return *this;
        }

        Matrix<T> dot(const Matrix<T> & other) const {
            assert(ncol == other.nrow);
            const Matrix<T> & self = *this;
            Matrix<T> res(nrow, other.ncol, false);

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t k = 0; k < other.ncol; ++k) {
                    res.__data.push_back(0);
                    for (size_t j = 0; j < ncol; ++j) {
                        res(i, k) += self(i, j) * other(j, k);
                    }
                }
            }
            return res;
        }

        Matrix<T> operator*(const Matrix<T> & other) const {
            assert(nrow == other.nrow && ncol == other.ncol);
            Matrix<T> res(nrow, ncol, false);

            for (size_t i = 0; i < size; ++i) {
                //res.__data[i] = __data[i] * other.__data[i];
                res.__data.push_back(__data[i] * other.__data[i]);
            }
            return res;
        }

        Matrix<T> operator+(const Matrix<T> & other) const {
            assert(nrow == other.nrow && ncol == other.ncol);
            Matrix<T> res(nrow, ncol, false);

            for (size_t i = 0; i < size; ++i) {
                //res.__data[i] = __data[i] + other.__data[i];
                res.__data.push_back(__data[i] + other.__data[i]);
            }
            return res;
        }

        Matrix<T> operator-(const Matrix<T> & other) const {
            assert(nrow == other.nrow && ncol == other.ncol);
            Matrix<T> res(nrow, ncol, false);

            for (size_t i = 0; i < size; ++i) {
                //res.__data[i] = __data[i] - other.__data[i];
                res.__data.push_back(__data[i] - other.__data[i]);
            }
            return res;
        }

        Matrix<T> operator-() const {
            Matrix<T> res(nrow, ncol, false);

            for (size_t i = 0; i < size; ++i) {
                //res.__data[i] = -__data[i];
                res.__data.push_back(-__data[i]);
            }
            return res;
        }

        inline void operator+=(const Matrix<T> & other) {
            assert(nrow == other.nrow && ncol == other.ncol);
            for (size_t i = 0; i < size; ++i) {
                __data[i] += other.__data[i];
            }
        }

        inline void operator-=(const Matrix<T> & other) {
            assert(nrow == other.nrow && ncol == other.ncol);
            for (size_t i = 0; i < size; ++i) {
                __data[i] -= other.__data[i];
            }
        }

        inline void operator*=(const Matrix<T> & other) {
            assert(nrow == other.nrow && ncol == other.ncol);
            for (size_t i = 0; i < size; ++i) {
                __data[i] *= other.__data[i];
            }
        }

        inline Matrix<T> operator*(T scalar) const {
            Matrix<T> res(nrow, ncol, false);

            for (size_t i = 0; i < size; ++i) {
                //res.__data[i] = __data[i] * scalar;
                res.__data.push_back(__data[i] * scalar);
            }
            return res;
        }

        inline Matrix<T> operator/(T scalar) const {
            assert(scalar != 0);
            Matrix<T> res(nrow, ncol, false);

            for (size_t i = 0; i < size; ++i) {
                //res.__data[i] = __data[i] / scalar;
                res.__data.push_back(__data[i] / scalar);
            }
            return res;
        }

        inline void operator*=(T scalar) {
            for (size_t i = 0; i < size; ++i) {
                __data[i] *= scalar;
            }
        }

        inline void operator/=(T scalar) {
            assert(scalar != 0);
            for (size_t i = 0; i < size; ++i) {
                __data[i] /= scalar;
            }
        }

        Matrix<T> transpose() const {
            Matrix<T> res(ncol, nrow);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(j, i) = self(i, j);
                }
            }
            return res;
        }

        inline Matrix<T> t() const {
            return transpose();
        }

        inline bool isEmpty() const {
            return size == 0;
        }

        inline void setZero() {
            std::vector<T>(size).swap(__data);
            //for (size_t i = 0; i < size; ++i) {
            //    __data[i] = 0;
            //}
        }

        inline void setOnes() {
            std::vector<T>(size, 1).swap(__data);
            //for (size_t i = 0; i < size; ++i) {
            //    __data[i] = 1;
            //}
        }

        T max_element() {
            assert(size != 0);
            T max = __data[0];
            for (size_t i = 0; i < size; ++i) {
                if (__data[i] > max) {
                    max = __data[i];
                }
            }
            return max;
        }

        std::pair<size_t, size_t> max_index() {
            assert(size != 0);
            T max = __data[0];
            size_t max_index = 0;
            for (size_t i = 0; i < size; ++i) {
                if (__data[i] > max) {
                    max = __data[i];
                    max_index = i;
                }
            }
            return std::make_pair(max_index / ncol, max_index % ncol);
        }

        void print() const {
            printf("[");
            for (size_t i = 0; i < nrow; ++i) {
                printf("[");
                for (size_t j = 0; j < ncol; ++j) {
                    if (j < ncol - 1) {
                        printf("%f,", (*this)(i, j));
                    } else {
                        printf("%f", (*this)(i, j));
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

        size_t nrow, ncol, size;
    private:
        //T* __data;
        std::vector<T> __data;
    };
}

#endif //DEEP_LEARNING_MATRIX_H
