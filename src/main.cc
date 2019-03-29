#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <cmath>
#include <limits>

#include "parallelCamera.hh"
#include "calibration.hh"
#include "pipeline.hh"
#include "tracker.hh"

#include <networktables/NetworkTable.h>
#include <networktables/NetworkTableInstance.h>
#include <cameraserver/CameraServer.h>
//#include "cscore/cscore.h"
//#include "cscore/cscore_oo.h"

using namespace cv;
using namespace std;
using namespace pipeline;

const double tol = 35;
const Scalar lo(70 - tol,30,30);
const Scalar hi(70 + tol,255,255);

vector<int> v;

constexpr double radius = 125;
Tracker t(radius);

const double HEIGHT = 720;
const double WIDTH = 1280;

const double fov = 68.5;
const double s_h = 4; //mm
const double s_w = 6; //mm
const double Ihc = ((HEIGHT / 2) - 0.5);
const double Iwc = ((WIDTH / 2) - 0.5);
const double fp = (WIDTH / (2 * math.tan(fov / 2))); // Focal length pix
const double fm = (fp * WIDTH) / s_w; 

const double hs = 10; // height of sensor in mm
const double hr = 25; // real object height
const double hp = 360; // height of frame in pixels

vector<Point> pts;
Point pp1, pp2;
contour c1, c2;


const bool NETWORK_TABLES = true;
nt::NetworkTableInstance inst;
std::shared_ptr<NetworkTable> table;

double yaw(int x){
    return atan((x - Iwc)/fp);
}

double distance(int h){
    return (f * hr * hp)/(hs * h);
}

int main(){
    ParallelCamera cam(0);

    if(NETWORK_TABLES){ //Set up Network Tables stuff
        inst = nt::NetworkTableInstance::GetDefault();
        inst.StartClientTeam(5332);
        table = inst.GetTable("vision_table");

        while(!inst.IsConnected()){
            cout << "NOT\n";
        }
    }

    auto mCamera0 = frc::CameraServer::GetInstance()->StartAutomaticCapture(0);
    mCamera0.SetExposureManual(40);

   if(NETWORK_TABLES){ //Set up Network Tables stuff
        inst = nt::NetworkTableInstance::GetDefault();
        inst.StartClientTeam(5332);
        table = inst.GetTable("vision_table");
        while(!inst.IsConnected()){
            cout << "NOT\n";
        }
   }

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
            Point p1 = centroid(c1);
            Point p2 = centroid(c2);
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
