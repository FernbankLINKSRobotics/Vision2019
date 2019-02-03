#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "calibration.h"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/*
const int numBoards = 15;
const int board_w = 9;
const int board_h = 6;
const Size board_sz = Size(board_w, board_h);
const int board_n = board_w * board_h;
const char* file = "calib.yml";
*/

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
  Mat C, D;

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
    cap.read(img);
  //  cout << "captured\n";
  ///  imshow("test", img);
  //}
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
  
  //return Data {C, D};
}

/*
void calibrate(VideoCapture& cap){
  vector<vector<Point3f>> objectPoints;
  vector<vector<Point2f>> imagePoints;
  vector<Point2f> corners;

  vector<Point3f> obj;
  for (int j=0; j<board_n; j++) {
      obj.push_back(Point3f(j/board_w, j%board_w, 0.0f));
  }

  Mat img, gray;

  int success = 0;
  int k = 0;
  bool found = false;

  while(success < numBoards){
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
  fs.open(file, FileStorage::WRITE);

  fs << "CM1" << intrinsic;
  fs << "D1" << discoeffs;
  cout << "Done\n";
}

int main(){
  VideoCapture cap(1);

  FileStorage f(file, FileStorage::READ);
  if(!f.isOpened()){
    calibrate(cap);
  }

  f.open(file, FileStorage::READ);
  Mat C, D;
  f["CM1"] >> C;
  f["D1"] >> D;

  Mat img, imgU;
  while(waitKey(10) != 27){
    cap >> img;

    undistort(img, imgU, C, D);

    imshow("Distorted", img);
    imshow("Undistorted", imgU);
  }
}
*/