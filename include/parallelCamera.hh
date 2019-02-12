#ifndef PARALLEL_CAMERA_H
#define PARALLEL_CAMERA_H

#include <mutex>
#include <thread>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

class ParallelCamera {
public:
    ParallelCamera(int ind);
    void start();
    void stop();
    void pause();
    bool frame();
    Mat get();

private:
    VideoCapture cap_;
    std::mutex mFrame_;
    Mat frame_;
    std::mutex mNew_;
    bool new_ = false;

    bool running_;
    thread thread_;

    void update();
};

#endif // PARALLEL_CAMERA_H