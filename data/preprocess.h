//
// Created by Clytie on 2018/11/4.
//

#ifndef DEEP_LEARNING_PREPROCESS_H
#define DEEP_LEARNING_PREPROCESS_H

#include <vector>
#include <string>
#include <iostream>


template<typename _IS>
void LoadData(_IS &inStream,
              size_t & pImgRows,
              size_t & pImgCols,
              std::vector<std::vector<float> > & trainImages,
              std::vector<float> & trainLabels,
              std::vector<std::vector<float>> & testImages) {
    size_t nTrainCnt, nTestCnt;
    inStream >> nTrainCnt >> nTestCnt >> pImgRows >> pImgCols;
    size_t nImgArea = pImgRows * pImgCols, n = 41;
    trainImages.resize(nTrainCnt);
    trainLabels.resize(nTrainCnt);
    testImages.resize(nTestCnt);
    for (size_t i = 0; i < nTrainCnt + nTestCnt; ++i) {
        std::string strLine;
        inStream >> strLine;
        std::vector<float> fltBuf(nImgArea);
        for (size_t j = 0; j < nImgArea / 2; ++j) {
            const char *p = strLine.c_str() + j * 3;
            size_t rawCode = (uint16_t) (p[0] - '0') * n * n;
            rawCode += (size_t) (p[1] - '0') * n;
            rawCode += (size_t) (p[2] - '0');
            fltBuf[j * 2 + 0] = ((rawCode & 0xFF) - 128.0f) / 255.0f;
            fltBuf[j * 2 + 1] = ((rawCode >> 8) - 128.0f) / 255.0f;
        }
        if (i < nTrainCnt) {
            fltBuf.swap(trainImages[i]);
            inStream >> trainLabels[i];
        } else {
            fltBuf.swap(testImages[i - nTrainCnt]);
        }
    }
}

#endif //DEEP_LEARNING_PREPROCESS_H
