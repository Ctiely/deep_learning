//
// Created by Clytie on 2018/10/31.
//

#ifndef DEEP_LEARNING_MATRIX_H
#define DEEP_LEARNING_MATRIX_H

#include <vector>

namespace matrix {
    template <typename T, size_t M, size_t N>
    class Matrix {
    public:
        Matrix() {
            initialize();
        }

        explicit Matrix(const std::vector<T> & data_) {
            initialize();
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    auto index = i * N + j;
                    if (index >= data_.size()) {
                        break;
                    }
                    _data[i][j] = data_[index];
                }
            }
        }

        explicit Matrix(const std::vector<std::vector<T> > & data_) {
            _setData(data_);
        }

        Matrix(const Matrix & other) {
            *this = other;
        }

        ~Matrix() = default;

        inline T operator()(size_t i, size_t j) const {
            assert(i < M && j < N);
            return _data[i][j];
        }

        inline T &operator()(size_t i, size_t j) {
            assert(i < M && j < N);
            return _data[i][j];
        }

        Matrix<T, M, N> & operator=(const Matrix<T, M, N> & other) {
            if (this != &other) {
                _setData(other._data);
            }
            return *this;
        }

        template<size_t P>
        Matrix<T, M, P> dot(const Matrix<T, N, P> & other) const {
            Matrix<T, M, P> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < P; ++k) {
                    for (size_t j = 0; j < N; ++j) {
                        res(i, k) += self(i, j) * other(j, k);
                    }
                }
            }
            return res;
        }

        Matrix<T, M, N> operator*(const Matrix<T, M, N> & other) const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(i, j) = self(i, j) * other(i, j);
                }
            }
            return res;
        }

        Matrix<T, M, N> operator/(const Matrix<T, M, N> & other) const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    assert(other(i, j) != 0);
                    res(i, j) = self(i, j) / other(i, j);
                }
            }
            return res;
        }

        Matrix<T, M, N> operator+(const Matrix<T, M, N> & other) const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(i, j) = self(i, j) + other(i, j);
                }
            }
            return res;
        }

        Matrix<T, M, N> operator-(const Matrix<T, M, N> & other) const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(i, j) = self(i, j) - other(i, j);
                }
            }
            return res;
        }

        Matrix<T, M, N> operator-() const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(i, j) = -self(i, j);
                }
            }
            return res;
        }

        void operator+=(const Matrix<T, M, N> & other) {
            *this = *this + other;
        }

        void operator-=(const Matrix<T, M, N> & other) {
            *this = *this - other;
        }

        void operator*=(const Matrix<T, M, N> & other) {
            *this = *this * other;
        }

        void operator/=(const Matrix<T, M, N> & other) {
            *this = *this / other;
        }

        Matrix<T, M, N> operator*(T scalar) const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(i, j) = self(i, j) * scalar;
                }
            }
            return res;
        }

        inline Matrix<T, M, N> operator/(T scalar) const {
            return *this * (1.0 / scalar);
        }

        Matrix<T, M, N> operator+(T scalar) const {
            Matrix<T, M, N> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(i, j) = self(i, j) + scalar;
                }
            }
            return res;
        }

        inline Matrix<T, M, N> operator-(T scalar) const {
            return *this + (-1 * scalar);
        }

        void operator*=(T scalar) {
            *this = *this * scalar;
        }

        void operator/=(T scalar) {
            assert(scalar != 0);
            *this = *this * (1.0 / scalar);
        }

        inline void operator+=(T scalar) {
            *this = *this + scalar;
        }

        inline void operator-=(T scalar) {
            *this = *this - scalar;
        }

        bool operator==(const Matrix<T, M, N> &other) const {
            const Matrix<T, M, N> &self = *this;

            static constexpr float eps = 1e-10f;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    if (T(self(i, j) - other(i, j)) > eps) {
                        return false;
                    }
                }
            }
            return true;
        }

        bool operator!=(const Matrix<T, M, N> & other) const {
            return !(*this == other);
        }

        Matrix<T, N, M> transpose() const {
            Matrix<T, N, M> res;
            const Matrix<T, M, N> & self = *this;

            for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    res(j, i) = self(i, j);
                }
            }
            return res;
        }

        inline Matrix<T, N, M> t() const {
            return transpose();
        }

        void print() const {
            printf("[");
            for (int i = 0; i < M; ++i) {
                printf("[");
                for (int j = 0; j < N; ++j) {
                    if (j < N - 1) {
                        printf("%f,", _data[i][j]);
                    } else {
                        printf("%f", _data[i][j]);
                    }
                }
                if (i < M - 1) {
                    printf("],\n");
                } else {
                    printf("]");
                }
            }
            printf("]\n");
        }

        void initialize() {
            _data = std::vector<std::vector<T> >(M, std::vector<T>(N, T()));
        }

    private:
        std::vector<std::vector<T> > _data;

        void _setData(const std::vector<std::vector<T> > & data_) {
            initialize();
            auto m = std::min(data_.size(), M);
            auto n = std::min(data_.front().size(), N);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    _data[i][j] = data_[i][j];
                }
            }
        }
    };

}

#endif //DEEP_LEARNING_MATRIX_H
