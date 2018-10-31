#include <iostream>
#include "src/Matrix.h"

using namespace std;

void test_constructor() {
    matrix::Matrix<int, 3, 2> matrix1;
    matrix1.print();

    matrix::Matrix<float, 3, 3> matrix2({1, 2, 3, 4, 5, 6, 7, 8, 9});
    matrix2.print();

    matrix::Matrix<double, 4, 3> matrix3({{1, 2, 3}, {3, 4, 5}, {5, 6, 7}});
    matrix3.print();

    matrix::Matrix<double, 4, 3> matrix4(matrix3);
    matrix4.print();

    matrix::Matrix<float, 3, 3> matrix5 = matrix2;
    matrix5.print();
}

void test_index() {
    matrix::Matrix<float, 3, 3> matrix1({1, 2, 3, 4, 5, 6, 7, 8, 9});
    cout << matrix1(1, 1) << " " << matrix1(0, 2) << endl;
}

void test_operator() {
    matrix::Matrix<double, 3, 3> matrix1({1, 2, 3, 4, 5, 6, 7, 8, 9});
    matrix::Matrix<double, 4, 3> matrix2({{1, 2, 3}, {3, 4, 5}, {5, 6, 7}});
    matrix::Matrix<double, 4, 3> matrix3 = matrix2.dot(matrix1);
    matrix3.print();

    matrix::Matrix<double, 3, 3> matrix4 = matrix1.dot(matrix1);
    matrix4.print();

    matrix::Matrix<double, 3, 3> matrix5 = matrix1 * matrix1;
    matrix5.print();

    matrix::Matrix<double, 3, 3> matrix6 = matrix1 / matrix5;
    matrix6.print();

    matrix::Matrix<double, 3, 3> matrix7 = matrix1 + matrix1;
    matrix7.print();

    matrix::Matrix<double, 3, 3> matrix8 = matrix1 - matrix7;
    matrix8.print();

    matrix::Matrix<double, 3, 3> matrix9 = -matrix1;
    matrix9.print();

    matrix1 += matrix5;
    matrix1.print();

    matrix1 -= matrix5;
    matrix1.print();

    matrix1 *= matrix5;
    matrix1.print();

    matrix1 /= matrix5;
    matrix1.print();

    matrix::Matrix<double, 3, 3> matrix10 = matrix1 * 2;
    matrix10.print();

    matrix::Matrix<double, 3, 3> matrix11 = matrix1 / 2;
    matrix11.print();

    matrix::Matrix<double, 3, 3> matrix12 = matrix1 + 2;
    matrix12.print();

    matrix::Matrix<double, 3, 3> matrix13 = matrix1 - 2;
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

int main() {
    cout << "test_constructor:" << endl;
    test_constructor();
    cout << "test_index:" << endl;
    test_index();
    cout << "test_operator:" << endl;
    test_operator();
    return 0;
}