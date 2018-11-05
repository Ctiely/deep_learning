#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "src/Matrix.h"
#include "src/Loss/Loss.h"
#include "data/preprocess.h"
#include "src/Layer/DenseLayer.h"
#include "src/ActiveFunc/ActiveFun.h"
#include "src/Optimization/Optimization.h"
#include <Eigen/Eigen>

using namespace std;

void test_eigen() {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix1(10, 10);
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix2(10, 10);
    matrix1.setRandom();
    cout << matrix1(0, 0) << endl;
    cout << matrix2(0, 0) << endl;
    matrix2(0, 0) = 0;
    cout << matrix1(0, 0) << endl;
    cout << matrix2(0, 0) << endl;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix3 = matrix1 * matrix2;
    cout << matrix3.size() << endl;
}


void test_constructor() {
    matrix::Matrix<int> matrix1(3, 2);
    matrix1.print();

    //matrix::Matrix<float> matrix2(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    //matrix2.print();
    vector<vector<double> > vec = {{1, 2, 3}, {3, 4, 5}, {5, 6, 7}};
    matrix::Matrix<double> matrix3(vec);
    matrix3.print();

    matrix::Matrix<double> matrix4(matrix3);
    matrix4.print();

    matrix::Matrix<double> matrix5 = matrix3;
    matrix5.print();
}

void test_index() {
    std::random_device rd;
    std::mt19937 rg(rd());
    std::normal_distribution<float> normDist(0, 0.1);
    auto genNormRand = [&]() { return normDist(rg); };

    matrix::Matrix<float> matrix1(5, 3, genNormRand);
    matrix1.print();
    cout << matrix1(1, 1) << " " << matrix1(0, 2) << endl;
    matrix1(1).print();
    cout << matrix1.max_element() << endl;
    cout << matrix1.max_index().first << " " << matrix1.max_index().second << endl;
}

void test_operator() {
    vector<vector<double> > vec1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix::Matrix<double> matrix1(vec1);
    vector<vector<double> > vec2 = {{1, 2, 3}, {3, 4, 5}, {5, 6, 7}, {0, 0, 0}};
    matrix::Matrix<double> matrix2(vec2);
    matrix::Matrix<double> matrix3 = matrix2.dot(matrix1);
    matrix3.print();

    matrix::Matrix<double> matrix4 = matrix1.dot(matrix1);
    matrix4.print();

    matrix::Matrix<double> matrix5 = matrix1 * matrix1;
    matrix5.print();

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

    matrix::Matrix<double> matrix10 = matrix1 * 2;
    matrix10.print();

    matrix::Matrix<double> matrix11 = matrix1 / 2;
    matrix11.print();

    matrix1 *= 2;
    matrix1.print();

    matrix1 /= 2;
    matrix1.print();

    matrix2.transpose().print();
    matrix2.t().t().print();
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

void test_dense_layer() {
    DenseLayer<float> layer(10, 100);

    cout << layer.weights_.nrow << " " << layer.weights_.ncol << endl;
    cout << layer.bias_.nrow << " " << layer.bias_.ncol << endl;
    layer.weights_.print();
    layer.bias_.print();
}

void test_softmax() {
    SoftMaxLoss<float> softmaxloss;
    vector<vector<float> > vec = {{1}, {2}, {3}, {4}, {5}};
    matrix::Matrix<float> inputs_(vec);
    inputs_.print();
    float loss;
    cout << softmaxloss.Forward(inputs_, 0, loss) << endl;
    softmaxloss.grad_().print();
    cout << "loss: " << loss << endl;
}

void test_load_data() {
    size_t nImgRows, nImgCols;

    vector<vector<float> > trainImgs, testImgs;
    vector<float> trainLabels;
    //if you want to load from a file, use std::ifstream to open your file
    //	and replace the std::cin with the file object.

    std::ifstream trainFileStram("../data/train_2000a.txt");

    LoadData(trainFileStram, nImgRows, nImgCols, trainImgs, trainLabels, testImgs);
    cout << nImgRows << " " << nImgCols << endl;
    cout << "train size: " << trainImgs.size() << endl;
    cout << "test size: " << testImgs.size() << endl;
    cout << "feature size: " << trainImgs.front().size() << endl;
    matrix::Matrix<float> x_train(trainImgs);
    matrix::Matrix<float> y_train(trainLabels, false);
    cout << 1 << endl;
    matrix::Matrix<float> x_test(testImgs);
    cout << x_train.nrow << " " << x_train.ncol << endl;
    //x_train.print();
    y_train.print();
    y_train.t().print();
    //x_test.print();
}

void test_dnn() {
    clock_t start = clock();
    size_t nImgRows, nImgCols;
    vector<vector<float> > trainImgs, testImgs;
    vector<float> trainLabels;
    ifstream trainFileStram("../data/train_2000a.txt");
    LoadData(trainFileStram, nImgRows, nImgCols, trainImgs, trainLabels, testImgs);

    matrix::Matrix<float> x_train(trainImgs);
    matrix::Matrix<float> y_train(trainLabels, false);
    matrix::Matrix<float> x_test(testImgs);

    size_t nImgArea = nImgCols * nImgRows;

    size_t fc1In = 28;
    size_t fc2In = 10;
    size_t maxIter = 4;
    float lr = 0.05;
    size_t nBatchSize = 64;

    std::random_device rd;
    std::mt19937 rg(rd());
    std::normal_distribution<float> normDist(0, 0.1);
    auto genNormRand = [&]() { return normDist(rg); };
    DenseLayer<float> fc1(nImgArea, fc1In, genNormRand), fc2(fc1In, fc2In, genNormRand);
    Tanh<float> act;
    SoftMaxLoss<float> loss;
    GradientDescent<float> opt;

    matrix::Matrix<float> loss_weights_(0, 0);
    size_t iter;
    for (iter = 0; iter < maxIter; ++iter) {
        for (size_t iImgdx = 0; iImgdx < x_train.nrow; iImgdx += nBatchSize) {
            float fLossSum = 0.0f, fLoss;
            size_t nCorrected = 0;

            matrix::Matrix<float> fc1WeightsGrads(fc1In, nImgArea);
            matrix::Matrix<float> fc1BiasGrads(fc1In, 1);
            matrix::Matrix<float> fc2WeightsGrads(fc2In, fc1In);
            matrix::Matrix<float> fc2BiasGrads(fc2In, 1);

            for (size_t iBatch = 0; iBatch < nBatchSize; ++iBatch) {
                // Random SGD
                size_t iImg = random() % trainImgs.size();
                //cout << "img: " << iImg << endl;

                // forward
                matrix::Matrix<float> img = x_train(iImg);
                fc1.Forward(img.t());

                matrix::Matrix<float> outputs_ = act.Forward(fc1.outputs_); //a_fc1
                fc2.Forward(outputs_);
                size_t nPred = loss.Forward(fc2.outputs_, (size_t)y_train(iImg, 0), fLoss);

                // backward

                matrix::Matrix<float> loss_grads_ = loss.grad_();

                matrix::Matrix<float> fc2_active_grads_(fc2In, 1, false);
                fc2_active_grads_.setOnes();
                fc2.Backward(loss_weights_, loss_grads_, fc2_active_grads_);
                matrix::Matrix<float> act_grads = act.grad_();
                fc1.Backward(fc2.weights_, fc2.grads_, act_grads);

                fLossSum += fLoss;
                nCorrected += (nPred == y_train(iImg, 0));

                fc1WeightsGrads += fc1.grads_.dot(img);
                fc1BiasGrads += fc1.grads_;
                fc2WeightsGrads += fc2.grads_.dot(outputs_.t());
                fc2BiasGrads += fc2.grads_;
            }

            cout << "loss = " << fLossSum / (float)nBatchSize << "\tprecision = " << nCorrected / (float)nBatchSize << endl;

            opt.Update(fc1.weights_, fc1.bias_, fc1WeightsGrads, fc1BiasGrads, lr / (float)nBatchSize);
            opt.Update(fc2.weights_, fc2.bias_, fc2WeightsGrads, fc2BiasGrads, lr / (float)nBatchSize);
        }
    }

    std::ifstream testLabelFileStream("../data/label_2000a.txt");
    vector<float> testLabel;
    testLabel.resize(500);
    for (uint32_t i = 0; i < 500; i++) {
        testLabelFileStream >> testLabel[i];
    }

    float fLossSum = 0.0f, fLoss;
    uint32_t nCorrected = 0;
    for (uint32_t i = 0; i < testImgs.size(); i++) {
        fc1.Forward(x_test(i).t());
        matrix::Matrix<float> outputs_ = act.Forward(fc1.outputs_); //a_fc1
        fc2.Forward(outputs_);
        auto label = (size_t)testLabel[i];
        size_t nPred = loss.Forward(fc2.outputs_, label, fLoss);

        fLossSum += fLoss;
        nCorrected += (nPred == label);
    }
    std::cout << "[test] " << "loss = " << fLossSum / testImgs.size() << " accuracy = "
              << (float)nCorrected / testImgs.size() << std::endl;

    std::cout << "Duration: " << (clock() - start) / (double)CLOCKS_PER_SEC << "s " << "for " << iter
              << " times durations" << std::endl;
}

void test_speed_matrix() {
    std::random_device rd;
    std::mt19937 rg(rd());
    std::normal_distribution<float> normDist(0, 0.1);
    auto genNormRand = [&]() { return normDist(rg); };
    vector<float> arr(1000000);
    generate(arr.begin(), arr.end(), genNormRand);
    matrix::Matrix<float> matrix1(arr);
    clock_t start = clock();
    max_element(arr.begin(), arr.end());
    cout << (clock() - start) / (double)CLOCKS_PER_SEC << endl;
    start = clock();
    matrix1.max_element();
    cout << (clock() - start) / (double)CLOCKS_PER_SEC << endl;
}

int main() {
    //cout << "test_constructor:" << endl;
    //test_constructor();
    //cout << "test_index:" << endl;
    //test_index();
    //cout << "test_operator:" << endl;
    //test_operator();
    //cout << "test_random_initialize:" << endl;
    //test_random_initialize();
    //cout << "test_dense_layer:" << endl;
    //test_dense_layer();
    //test_softmax();
    //test_load_data();
    test_dnn();
    //test_speed_matrix();
    //test_eigen();
    return 0;
}