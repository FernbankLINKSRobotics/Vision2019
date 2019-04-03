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

Calibration::Data config;

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

contour c1, c2;
Mat rvec = Mat::zeros(cv::Size(1, 49), CV_64FC1);
Mat tvec = Mat::zeros(cv::Size(1, 49), CV_64FC1);

std::vector<cv::Point3d> model_points; // ALL MEASUREMENTS IN CM
model_points.push_back(cv::Point3d(0.0f, -18.4f, -4.8f));
model_points.push_back(cv::Point3d(0.0f, -14.8f, 7.4f));
model_points.push_back(cv::Point3d(0.0f, -13.6f, -7.4f));
model_points.push_back(cv::Point3d(0.0f, -10.0f, 4.8f));
model_points.push_back(cv::Point3d(0.0f, 10.0f, 4.8f));
model_points.push_back(cv::Point3d(0.0f, 13.6f, -7.4f));
model_points.push_back(cv::Point3d(0.0f, 14.8f, 7.4f));
model_points.push_back(cv::Point3d(0.0f, 18.4f, -4.8f));

const bool NETWORK_TABLES = false;
nt::NetworkTableInstance inst;
std::shared_ptr<NetworkTable> table;

const bool STREAM_OUTPUT = false;
cs::CvSource cvSource{"cvSource", cs::VideoMode::kMJPEG, 265, 144, 30)};
cs::MjpegServer outputStreamServer{"outputStreamServer", 5800};

double yaw(int x){
    return atan((x - Iwc)/fp);
}

double distance(int h){
    return (f * hr * hp)/(hs * h);
}

int main(){
    ParallelCamera cam(0);

    auto mCamera0 = frc::CameraServer::GetInstance()->StartAutomaticCapture(0);
    mCamera0.SetExposureManual(40);

    inst = nt::NetworkTableInstance::GetDefault();
    inst.StartClientTeam(4468);
    table = inst.GetTable("vision_table");
    while(!inst.IsConnected()){
        cout << "NOT\n";
    }

    if (!table->GetBoolean("online", false)){
        table->PutBoolean("online", true);
    }

    if(STREAM_OUTPUT){
       outputStreamServer.SetSource(cvSource);
    }

    Calibration c("calib.yml");
    config = (c.isCalibrated()) ? c.read() : c.write(cap);

    cam.start();
    while (!table->getBoolean("stop", false)){
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
            if(table->getBoolean("track", false)) {
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

            auto p1 = corners(c1); // Polygons made of the corners
            auto p2 = corners(c2); // approxPolyDP
            std::sort(p1.begin(), p1.end(), [](Point2f &p1, Point2f &p2) {
                return p1.y > p2.y;
            });
            std::sort(p2.begin(), p2.end(), [](Point2f &pt1, Point2f &pt2) {
                return pt1.y > pt2.y;
            });
            auto image_points = concat(p1, p2);

            solvePnPRansac(model_points, image_points, config.camera, config.distance, rvec, tvec);

            /*
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
            */
        }
        drawContours(f, cnt, -1, Scalar(255, 191, 0), 2);
        imshow("test", f);

        if (STREAM_OUTPUT){
            auto out = f.clone();
            //cv::circle(outputFrame, cv::Point(cx, cy), 10, cv::Scalar(0, 0, 255), 10);
            cvSource.PutFrame(out);
        }

        table->PutNumber("Angle", rvec.at<double>(0,1));
        table->PutNumber("X", tvec.at<double>(0,0));
        table->PutNumber("Z", tvec.at<double>(0,2));
        table->PutBoolean("Visible", t.tracking());
    }
    cam.stop();
}
