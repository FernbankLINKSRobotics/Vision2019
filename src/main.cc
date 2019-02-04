#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <numeric>

#include "calibration.h"
#include "parallelCamera.h"


using namespace std;
using namespace cv;

const bool calibrate = false;
Calibration::Data config;

vector<int> v;

int main(){
    VideoCapture cap(0);
    ParallelCamera cam(0);

    if constexpr (calibrate) {
        Calibration c("calib.yml");
        config = (c.isCalibrated()) ? c.read() : c.write(cap);
        //cout << "CM:\n" << config.camera << "\n";
        //cout << "D:\n" << config.distance << "\n";
    }

    cam.start();
    while(waitKey(10) != 27){
        auto m = cam.get();
        if(m.empty()){ continue; }
        imshow("test", m);
    }

    cam.stop();
}