#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "calibration.hh"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

Calibration::Calibration(const char* file){
  f_ = file;
  fs_ = FileStorage(file, FileStorage::READ);
}

Calibration::Calibration(const char* file, const Size& boardSize, const size_t frameNum){
  f_ = file;
  board_sz = boardSize;
  num_ = frameNum;
  fs_ = FileStorage(file, FileStorage::READ);
  board_w = board_sz.width;
  board_h = board_sz.height;
  board_n = board_w * board_h;
}

bool Calibration::isCalibrated(){ return fs_.isOpened(); }

Calibration::Data Calibration::read(){
  FileStorage fs(f_, FileStorage::READ);
  Mat C, D;
  fs["CM1"] >> C;
  fs["D1"] >> D;
  return Data {C, D};
}

Calibration::Data Calibration::write(VideoCapture& cap) {
  vector<vector<Point3f>> objectPoints;
  vector<vector<Point2f>> imagePoints;
  vector<Point2f> corners;

  vector<Point3f> obj;
  for (int j=0; j<board_n; j++) {
      obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
  }

  Mat img, gray;

  size_t success = 0;
  int k = 0;
  bool found = false;

  while(success < num_){
    cap >> img;

    cvtColor(img, gray, COLOR_BGR2GRAY);
    found = findChessboardCorners(gray, board_sz, corners, CALIB_CB_ADAPTIVE_THRESH 
                                                         | CALIB_CB_FAST_CHECK 
                                                         | CALIB_CB_NORMALIZE_IMAGE);

    if (found) {
      drawChessboardCorners(gray, board_sz, corners, found);
    }

    imshow("corners", gray);
  
    k = waitKey(1);
    if (found) {
      k = waitKey(0);
    }

    if(k == 27){ break; }
    if(k == 13 && found){
      imagePoints.push_back(corners);
      objectPoints.push_back(obj);
      cout << "Corners stored\n";
      success++;
    }
    if(k == ' '){ continue; }
  }
  
  destroyAllWindows();
  cout << "Starting Calibration\n";

  Mat intrinsic(3,3, CV_32FC1);
  Mat discoeffs;

  vector<Mat> rvecs, tvecs;
  intrinsic.at<float>(0,0) = 1;
  intrinsic.at<float>(1,1) = 1;

  calibrateCamera(objectPoints, imagePoints, img.size(), intrinsic, discoeffs, rvecs, tvecs);
  FileStorage fs;
  fs.open(f_, FileStorage::WRITE);

  fs << "CM1" << intrinsic;
  fs << "D1" << discoeffs;
  cout << "Done\n";
  return Data {intrinsic, discoeffs};
}