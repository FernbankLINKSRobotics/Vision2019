#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/core/core.hpp>

using namespace cv;

class Calibration {
public: 
    struct Data {
        Mat camera;
        Mat distance;
    };

    Calibration(const char* file);
    Calibration(const char* file, const Size& boardSize, const size_t frameNum);

    bool isCalibrated();
    Data read();
    Data write(VideoCapture& vid);

private:
    const char* f_;
    size_t num_ = 15;
    int board_w = 9;
    int board_h = 6;
    int board_n = board_w * board_h;
    Size board_sz {board_w, board_h};

    FileStorage fs_;
};

#endif // CALIBRATION_H