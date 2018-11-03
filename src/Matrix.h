//
// Created by Clytie on 2018/10/31.
//

#ifndef DEEP_LEARNING_MATRIX_H
#define DEEP_LEARNING_MATRIX_H

#include <vector>
#include <utility>

namespace matrix {
    template <typename T>
    class Matrix {
    public:
        Matrix(size_t nrow, size_t ncol)
                : nrow(nrow), ncol(ncol) {
            initialize();
        }

        template <typename __Generator>
        Matrix(size_t nrow, size_t ncol, __Generator generator)
                : nrow(nrow), ncol(ncol) {
            initialize();
            for (int i = 0; i < _data.size(); ++i) {
                std::generate(_data[i].begin(), _data[i].end(), generator);
            }
        }

        Matrix(size_t nrow, size_t ncol, const std::vector<T> & data_)
                : nrow(nrow), ncol(ncol) {
            initialize();
            for (int i = 0; i < nrow; ++i) {
                for (int j = 0; j < ncol; ++j) {
                    auto index = i * ncol + j;
                    if (index >= data_.size()) {
                        break;
                    }
                    _data[i][j] = data_[index];
                }
            }
        }

        Matrix(size_t nrow, size_t ncol, const std::vector<std::vector<T> > & data_)
                : nrow(nrow), ncol(ncol) {
            _setData(data_);
        }

        Matrix(const Matrix<T> & other) {
            nrow = other.nrow;
            ncol = other.ncol;
            _setData(other._data);
        }

        ~Matrix() = default;

        inline T operator()(size_t i, size_t j) const {
            assert(i < nrow && j < ncol);
            return _data[i][j];
        }

        inline T &operator()(size_t i, size_t j) {
            assert(i < nrow && j < ncol);
            return _data[i][j];
        }

        Matrix<T> & operator=(const Matrix<T> & other) {
            if (this != &other) {
                nrow = other.nrow;
                ncol = other.ncol;
                _setData(other._data);
            }
            return *this;
        }

        Matrix<T> dot(const Matrix<T> & other) const {
            assert(ncol == other.nrow);
            Matrix<T> res(nrow, other.ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t k = 0; k < other.ncol; ++k) {
                    for (size_t j = 0; j < ncol; ++j) {
                        res(i, k) += self(i, j) * other(j, k);
                    }
                }
            }
            return res;
        }

        Matrix<T> operator*(const Matrix<T> & other) const {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(i, j) = self(i, j) * other(i, j);
                }
            }
            return res;
        }

        Matrix<T> operator/(const Matrix<T> & other) const {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    assert(other(i, j) != 0);
                    res(i, j) = self(i, j) / other(i, j);
                }
            }
            return res;
        }

        Matrix<T> operator+(const Matrix<T> & other) const {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(i, j) = self(i, j) + other(i, j);
                }
            }
            return res;
        }

        Matrix<T> operator-(const Matrix<T> & other) const {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(i, j) = self(i, j) - other(i, j);
                }
            }
            return res;
        }

        Matrix<T> operator-() const {
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(i, j) = -self(i, j);
                }
            }
            return res;
        }

        void operator+=(const Matrix<T> & other) {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            *this = *this + other;
        }

        void operator-=(const Matrix<T> & other) {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            *this = *this - other;
        }

        void operator*=(const Matrix<T> & other) {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            *this = *this * other;
        }

        void operator/=(const Matrix<T> & other) {
            assert(nrow == other.nrow);
            assert(ncol == other.ncol);
            *this = *this / other;
        }

        Matrix<T> operator*(T scalar) const {
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(i, j) = self(i, j) * scalar;
                }
            }
            return res;
        }

        inline Matrix<T> operator/(T scalar) const {
            return *this * (1.0 / scalar);
        }

        Matrix<T> operator+(T scalar) const {
            Matrix<T> res(nrow, ncol);
            const Matrix<T> & self = *this;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    res(i, j) = self(i, j) + scalar;
                }
            }
            return res;
        }

        inline Matrix<T> operator-(T scalar) const {
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

        bool operator==(const Matrix<T> & other) const {
            if (nrow != other.nrow || ncol != other.ncol) {
                return false;
            }

            const Matrix<T> & self = *this;

            static constexpr float eps = 1e-10f;

            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    if (T(self(i, j) - other(i, j)) > eps) {
                        return false;
                    }
                }
            }
            return true;
        }

        bool operator!=(const Matrix<T> & other) const {
            return !(*this == other);
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

        void print() const {
            printf("[");
            for (size_t i = 0; i < nrow; ++i) {
                printf("[");
                for (size_t j = 0; j < ncol; ++j) {
                    if (j < ncol - 1) {
                        printf("%f,", _data[i][j]);
                    } else {
                        printf("%f", _data[i][j]);
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

        bool isEmpty() const {
            return nrow == 0;
        }

        T max_element() const {
            const Matrix<T> & self = *this;
            T __max = self(0, 0);
            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    if (self(i, j) > __max) {
                        __max = self(i, j);
                    }
                }
            }
            return __max;
        }

        std::pair<size_t, size_t> max_index() const {
            assert(nrow > 0 && ncol > 0);
            const Matrix<T> & self = *this;
            T __max_element = self(0, 0);
            std::pair<size_t, size_t> __max = std::make_pair(0, 0);
            for (size_t i = 0; i < nrow; ++i) {
                for (size_t j = 0; j < ncol; ++j) {
                    if (self(i, j) > __max_element) {
                        __max_element = self(i, j);
                        __max.first = i;
                        __max.second = j;
                    }
                }
            }
            return __max;
        }

        size_t nrow;
        size_t ncol;
        std::vector<std::vector<T> > _data;
    private:
        void initialize() {
            _data = std::vector<std::vector<T> >(nrow, std::vector<T>(ncol, T()));
        }

        void _setData(const std::vector<std::vector<T> > & data_) {
            initialize();
            auto m = std::min(data_.size(), nrow);
            auto n = std::min(data_.front().size(), ncol);
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    _data[i][j] = data_[i][j];
                }
            }
        }
    };
}

#endif //DEEP_LEARNING_MATRIX_H
