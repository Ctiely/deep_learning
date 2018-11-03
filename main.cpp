#include <iostream>
#include "src/Matrix.h"
#include <random>

using namespace std;

void test_constructor() {
    matrix::Matrix<int> matrix1(3, 2);
    matrix1.print();

    matrix::Matrix<float> matrix2(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    matrix2.print();

    matrix::Matrix<double> matrix3(4, 3, {{1, 2, 3}, {3, 4, 5}, {5, 6, 7}});
    matrix3.print();

    matrix::Matrix<double> matrix4(matrix3);
    matrix4.print();

    matrix::Matrix<float> matrix5 = matrix2;
    matrix5.print();
}

void test_index() {
    matrix::Matrix<float> matrix1(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    cout << matrix1(1, 1) << " " << matrix1(0, 2) << endl;
}

void test_operator() {
    matrix::Matrix<double> matrix1(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    matrix::Matrix<double> matrix2(4, 3, {{1, 2, 3}, {3, 4, 5}, {5, 6, 7}});
    matrix::Matrix<double> matrix3 = matrix2.dot(matrix1);
    matrix3.print();

    matrix::Matrix<double> matrix4 = matrix1.dot(matrix1);
    matrix4.print();

    matrix::Matrix<double> matrix5 = matrix1 * matrix1;
    matrix5.print();

    matrix::Matrix<double> matrix6 = matrix1 / matrix5;
    matrix6.print();

    matrix::Matrix<double> matrix7 = matrix1 + matrix1;
    matrix7.print();

    matrix::Matrix<double> matrix8 = matrix1 - matrix7;
    matrix8.print();

    matrix::Matrix<double> matrix9 = -matrix1;
    matrix9.print();

    matrix1 += matrix5;
    matrix1.print();

    matrix1 -= matrix5;
    matrix1.print();

    matrix1 *= matrix5;
    matrix1.print();

    matrix1 /= matrix5;
    matrix1.print();

    matrix::Matrix<double> matrix10 = matrix1 * 2;
    matrix10.print();

    matrix::Matrix<double> matrix11 = matrix1 / 2;
    matrix11.print();

    matrix::Matrix<double> matrix12 = matrix1 + 2;
    matrix12.print();

    matrix::Matrix<double> matrix13 = matrix1 - 2;
    matrix13.print();

    matrix1 += 2;
    matrix1.print();

    matrix1 -= 2;
    matrix1.print();

    matrix1 *= 2;
    matrix1.print();

    matrix1 /= 2;
    matrix1.print();

    matrix2.transpose().print();
    matrix2.t().t().print();

    cout << (matrix1 == matrix1.t()) << endl;
    cout << (matrix1 == matrix1.t().t()) << endl;
}

void test_random_initialize() {
    std::random_device rd;
    std::mt19937 rg(rd());
    std::normal_distribution<float> normDist(0, 0.1);
    auto genNormRand = [&]() { return normDist(rg); };

    matrix::Matrix<float> matrix1(5, 4, genNormRand), matrix2(5, 4, genNormRand);
    matrix1.print();
    matrix2.print();
}

int main() {
    cout << "test_constructor:" << endl;
    test_constructor();
    cout << "test_index:" << endl;
    test_index();
    cout << "test_operator:" << endl;
    test_operator();
    cout << "test_random_initialize:" << endl;
    test_random_initialize();
    return 0;
}