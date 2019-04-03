#include "pipeline.hh"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <functional>
#include <vector>
#include <queue>

using namespace pipeline;

contours pipeline::thres(Mat& frame, const Scalar& lo, const Scalar& hi){
    Mat hsv, thres;
    contours cont;
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    inRange(hsv, lo, hi, thres);
    findContours(thres, cont, RETR_TREE, CHAIN_APPROX_SIMPLE);
    return cont;
}

vector<double> pipeline::largestNindex(contours& v, size_t n){
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

contours pipeline::largestN(contours& v, size_t n){
    contours conts;
    auto ind = largestNindex(v, n);
    for(auto i: ind){
        conts.push_back(v.at(i));
    }
    return conts;
}

Point pipeline::centroid(Moments& m){
    int x =0, y=0;
    if(m.m00 != 0){
        x = (int) m.m10 / m.m00;
        y = (int) m.m01 / m.m00;
    }
    return Point{x, y};
}

Point pipeline::centroid(contour& c){
    int x =0, y=0;
    Moments m = moments(c);
    if(m.m00 != 0){
        x = (int) m.m10 / m.m00;
        y = (int) m.m01 / m.m00;
    }
    return Point{x, y};
}

void pipeline::order(contours& c){
    sort(c.begin(), c.end(), [](contour& lhs, contour& rhs){
        Point l = centroid(lhs);
        Point r = centroid(rhs);
        return l.x < r.x;
    });
}

std::vector<Point2f> pipeline::corners(contour& c){
    contour ret;
    double eps = .05 * arcLength(c, true);
    approxPolyDP(c, ret, eps, true);
    return ret;
}

/*
vector<target> targets(contours& v){
    vector<target> ret;
    for(size_t i=0; i < v.size()-1; i+=2){
        target t = make_pair(v.at(i), v.at(i+1));
        ret.push_back(t);
    }
    return ret;
}
*/
