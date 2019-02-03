#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include "calibration.h"

using namespace std;

const bool calibrate = true;
Calibration::Data config;

int main(){
    VideoCapture cap(0);

    if constexpr (calibrate) {
        Calibration c("calib.yml");
        config = (c.isCalibrated()) ? c.read() : c.write(cap);
        //cout << "CM:\n" << config.camera << "\n";
        //cout << "D:\n" << config.distance << "\n";
    }

    cout << "done\n";
}