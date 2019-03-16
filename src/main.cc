#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>

#include "calibration.hh"
#include "parallelCamera.hh"
#include "pipeline.hh"
#include "tracker.hh"

//#include "cscore"
#include <networktables/NetworkTable.h>

using namespace cv;
using namespace std;
using namespace pipeline;

const bool calibrate = false;
Calibration::Data config;

const double tol = 35;
const Scalar lo(70 - tol,30,30);
const Scalar hi(70 + tol,255,255);

vector<int> v;

constexpr double radius = 125;
Tracker t(radius);

const double HEIGHT = 26.6;
const double WIDTH = 19;

const double Ihc = ((HEIGHT / 2) - 0.5);
const double Iwc = ((WIDTH / 2) - 0.5);
const double f   = 300; // Focal length mm

const double hs = 10; // height of sensor in mm
const double hr = 25; // real object height
const double hi = 360; // height of frame in pixels

vector<Point> pts;
Point pp1, pp2;
contour c1, c2;

double yaw(int x){
    return atan((x - Lwc)/f);
}

double distance(int h){
    return (f * hr * hi)/(hs * h);
}

int main(){
    ParallelCamera cam(0);

    cam.start();
    while(waitKey(10) != 27){
        if(!cam.frame()){ continue; }
        Mat f = cam.get();
        
        contours cnt = thres(f, lo, hi);
        contours large = largestN(cnt, 1);

        /*
        if(large.size() < 1){ continue; }
        RotatedRect rect1 = minAreaRect(large[0]);
        double angle1 = rect1.angle;
        double height = max(rect1.size.height, rect1.size.width);
        if (height == rect1.size.width)
        {
            angle1 += 90;
            cout << "ADJUSTED\n";
        }
        cout << "Angle 1: " << angle1 << '\n';
        */

        if(large.size() < 2){ continue; }
        if(!t.tracking()){
            cout << "NOT\n";
            if(waitKey(1) == 13) {
                cout << "TRACKING\n";
                c1 = large.at(0);
                c2 = large.at(1);
                t.start(c1, c2);
            }
        }
        if(t.tracking()){
            order(large);
            t.update(large);
            c1 = t.left();
            c2 = t.right();

            RotatedRect rect1 = minAreaRect(c1);
            RotatedRect rect2 = minAreaRect(c2);
            double length1 = distance(hypot(rect1.size.height, rect1.size.width));
            double length2 = distance(hypot(rect2.size.height, rect2.size.width));
            Point p1 = centroid(rect1);
            Point p2 = centroid(rect2);
            double length = (length1 + length2) / 2;
            double angle = yaw((p1.x + p2.x) / 2);
            cout << "Length: " << length << " Angle: " << angle << '\n';

            circle(f, pp1, 10, Scalar(255, 0, 255));
            circle(f, centroid(c1), 5, Scalar(255, 255, 0));
            putText(f, "Target 1", centroid(c1), FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 200, 200));
            circle(f, pp2, 10, Scalar(255, 0, 255));
            circle(f, centroid(c2), 5, Scalar(255, 255, 0));
            putText(f, "Target 2", centroid(c2), FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 200, 200));
        }
        drawContours(f, cnt, -1, Scalar(255, 191, 0), 2);
        imshow("test", f);
    }
    cam.stop();
}