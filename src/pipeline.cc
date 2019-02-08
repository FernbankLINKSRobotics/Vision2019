#include "pipeline.hh"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>
#include <queue>

vector<vector<Point>> pipeline::targets(Mat& frame, const Scalar& lo, const Scalar& hi){
    Mat hsv, thres;
    vector<vector<Point>> cont;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, lo, hi, thres);
    findContours(thres, cont, RETR_TREE, CHAIN_APPROX_SIMPLE);
    return cont;
}

vector<double> pipeline::largestNindex(vector<vector<Point>>& v, size_t n){
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

vector<vector<Point>> pipeline::largestN(vector<vector<Point>>& v, size_t n){
    vector<vector<Point>> conts;
    auto ind = largestNindex(v, n);
    for(auto i: ind){
        conts.push_back(v.at(i));
    }
    return conts;
}

