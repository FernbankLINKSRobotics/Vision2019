#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <numeric>

#include <boost/asio.hpp> 

#include "pipeline.hh"
#include "calibration.hh"
#include "parallelCamera.hh"


using namespace cv;
using namespace std;
using namespace pipeline;

const bool calibrate = false;
Calibration::Data config;

const double tol = 30;
const Scalar lo(60 - tol,50,50);
const Scalar hi(60 + tol,255,255);

// Temp Constats
const int bins = 140;
const int chan = 0;
Rect rioSize(0, 0, 500, 500);
Rect window(0, 0, 1000, 1000);


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
        auto f = cam.get();
        if(f.empty()){ continue; }

        MatND hist(1, &bins, CV_8UC1);
        Mat mask, hsv, hue, proj;
        float range[] = {0, 180};
        const float* ranges = range;

        cvtColor(f, hsv, COLOR_RGB2HSV);
        inRange(hsv, lo, hi, mask);

        imshow("test", mask);

        calcHist(&hsv, 1, &chan, mask, hist, 1, &bins, &ranges);
        normalize(hist, hist, 0, 255, NORM_MINMAX);
        hue.create(hsv.size(), hsv.depth());

        if( window.area() <= 1 ){
            int cols = proj.cols, rows = proj.rows, r = (MIN(cols, rows) + 5)/6;
            window = Rect(window.x - r, window.y - r,
                            window.x + r, window.y + r) &
                            Rect(0, 0, cols, rows);
        }
        calcBackProject(&hue, 1, 0, hist, proj, &ranges);
        proj &= mask;
        Mat rio(hue, rioSize);
        
        RotatedRect trackBox = CamShift(proj, window,
                                        TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 200, 1 ));

        //rectangle(f,trackBox,Scalar(255,0,0),1,8,0);
        Point2f corners[4];
        trackBox.points(corners);
        for( int j = 0; j < 4; j++){
            line(f, corners[j], corners[(j+1)%4], Scalar(255,0,0), 10, 8 );
        }

        imshow("HIST", hist);

        imshow("test", f);
    }

    cam.stop();
}

/*
        auto cnt = targets(f, lo, hi);
        auto lct = largestN(cnt, 3);
        vector<RotatedRect> recs;
        for(auto e: lct){
            recs.push_back(minAreaRect(e));
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
        */