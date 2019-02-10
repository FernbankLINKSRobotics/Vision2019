#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>


#include "calibration.hh"
#include "parallelCamera.hh"
#include "pipeline.hh"


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
bool tracking = false;

vector<Point> pts;
Point pp1, pp2;
contour c1, c2;

Point centroid(Moments m){
    int x =0, y=0;
    if(m.m00 != 0){
        x = (int) m.m10 / m.m00;
        y = (int) m.m01 / m.m00;
    }
    return Point{x, y};
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
        if(!cam.frame()){ continue; }
        Mat f = cam.get();
        
        contours cnt = thres(f, lo, hi);
        contours large = largestN(cnt, 2);

        if(large.size() < 2){ continue; }
        if(!tracking){
            cout << "NOT\n";
            if(waitKey(1) == 13) {
                cout << "TRACKING\n";
                tracking = true;
                c1 = large.at(0);
                c2 = large.at(1);
                pp1 = centroid(moments(c1));
                pp2 = centroid(moments(c2));
            }
        }
        if(tracking){
            pts.clear();
            for(contour c: large){
                pts.push_back(centroid(moments(c)));
            }
            for(int i=0; i<large.size(); i++){
                double d1 = norm(pts[i] - pp1);
                double d2 = norm(pts[i] - pp2);
                if(d1 < radius){
                    pp1 = pts[i];
                    c1 = large[i];
                }
                if(d2 < radius){
                    pp2 = pts[i];
                    c2 = large[i];
                } else {
                    tracking = false;
                }
            }
            
            circle(f, pp1, 10, Scalar(255, 0, 255));
            circle(f, centroid(moments(c1)), 5, Scalar(255, 255, 0));
            putText(f, "Target 1", centroid(moments(c1)), FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 200, 200));
            circle(f, pp2, 10, Scalar(255, 0, 255));
            circle(f, centroid(moments(c2)), 5, Scalar(255, 255, 0));
            putText(f, "Target 2", centroid(moments(c2)), FONT_HERSHEY_COMPLEX, 0.8, Scalar(255, 200, 200));
        }
        

        //drawContours(f,top,-1,Scalar(255, 255, 0), 2);
        drawContours(f,cnt,-1,Scalar(255, 191, 0), 2);
        imshow("test", f);
    }
    cam.stop();
}

/*
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
*/