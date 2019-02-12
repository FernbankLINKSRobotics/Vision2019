#ifndef TRACKER_H
#define TRACKER_H

#include "pipeline.hh"

using namespace pipeline;

class Tracker{
public:
    Tracker(){}
    Tracker(double r);
    
    void start(contour& c1, contour& c2);
    void update(contours& cont);
    
    bool tracking();
    contour left();
    contour right();

private:
    contour left_;
    Point pLeft_;
    contour right_;
    Point pRight_;
    bool tracking_ = false;
    double radius_ = 100;
};

#endif // TRACKER_H