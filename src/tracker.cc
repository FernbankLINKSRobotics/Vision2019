#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "tracker.hh"

Tracker::Tracker(double r){
    radius_ = r;
}

void Tracker::start(contour& c1, contour& c2){
    Point p1 = centroid(c1);
    Point p2 = centroid(c2);
    if(p1.x < p2.x){
        left_ = c1;
        pLeft_ = p1;
        right_ = c2;
        pRight_ = p2;
    } else {
        left_ = c2;
        pLeft_ = p2;
        right_ = c1;
        pRight_ = p1;
    }
    tracking_ = true;
}

void Tracker::update(contours& cont){
    for(contour c: cont){
        Point p = centroid(c);
        double dl = norm(p - pLeft_);
        double dr = norm(p - pRight_);
        tracking_ = false;
        if(dl < radius_){
            pLeft_ = p;
            left_ = c;
            tracking_ = true;
        }
        if(dr < radius_){
            pRight_ = p;
            right_ = c;
            tracking_ = true;
        }
    }
}

contour Tracker::left() { return left_;  }
contour Tracker::right(){ return right_; }
bool Tracker::tracking(){ return tracking_; }
