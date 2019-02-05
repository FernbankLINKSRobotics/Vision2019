#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <numeric>
#include <queue>

#include "calibration.h"
#include "parallelCamera.h"


using namespace std;
using namespace cv;

const bool calibrate = false;
Calibration::Data config;

const double tol = 30;
const Scalar lo(70 - tol,50,50);
const Scalar hi(70 + tol,255,255);

vector<int> v;

vector<vector<Point>> targets(Mat& frame){
    Mat hsv, thres;
    vector<vector<Point>> cont;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, lo, hi, thres);
    findContours(thres, cont, RETR_TREE, CHAIN_APPROX_SIMPLE);
    return cont;
}

vector<double> largestN(vector<vector<Point>>& v, size_t n){
    size_t num = (n < v.size()) ? n : v.size();
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> q;
    for (size_t i = 0; i < v.size(); ++i) {
        double a = contourArea(v.at(i));
        if(q.size()<num)
            q.push(std::pair<double, int>(a, i));
        else if(q.top().first < a){
            q.pop();
            q.push(std::pair<double, int>(a, i));
        }
    }
    num = q.size();
    vector<double> res(num);
    for (size_t i = 0; i < num; ++i) {
        res[num - i - 1] = q.top().second;
        q.pop();
    }
    return res;
}

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
        auto f = cam.get();
        if(f.empty()){ continue; }
        auto cnt = targets(f);

        
        auto top = largestN(cnt, 1);
        vector<RotatedRect> recs;
        for(size_t e: top){
            recs.push_back(minAreaRect(cnt.at(e)));
        }


        for(size_t i=0; i<recs.size(); ++i){
            auto rec = recs.at(i);
            Point2f corners[4];
            rec.points(corners);
            for( int j = 0; j < 4; j++){
                line(f, corners[j], corners[(j+1)%4], Scalar(255,0,0), 10, 8 );
            }

            double angle = 0;
            if(rec.angle < -25){
                angle = -90 - rec.angle;
            } else {
                angle = -rec.angle;
            }
            cout << "Angle " << i << " :" << angle << '\n';
            cout << "Size " << i <<  " :" << rec.size.width * rec.size.height << '\n';

            int x = 0;
            if(angle < -15){
                x = corners[0].x;
            }

            line(f, Point(x, 0), Point(x, 1080), Scalar(0,255,0), 2);
        }
        

        drawContours(f,cnt,-1,Scalar(255, 191, 0), 2);
        imshow("test", f);
    }

    cam.stop();
}