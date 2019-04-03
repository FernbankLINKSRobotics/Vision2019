#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/core/core.hpp>
#include <functional>
#include <queue>
#include <vector>

using namespace std;
using namespace cv;


namespace pipeline {
    typedef vector<Point> contour;
    typedef vector<contour> contours;
    /*
    typedef struct {
        contour c1;
        contour c2;
    } target;
    */
    //typedef pair<contour, contour> target;
    
    contours thres(Mat& frame, const Scalar& lo, const Scalar& hi);
    contours largestN(contours& v, size_t n);
    //vector<target> targets(contours& v);
    vector<double> largestNindex(contours& v, size_t n);

    Point centroid(Moments& m);
    Point centroid(contour& c);

    void order(contours& c);
    contour corners(contour& c);

    template <typename T>
    std::vector<T> concat(const std::vector<T> &A, const std::vector<T> &B){
        std::vector<T> AB;
        AB.reserve(A.size() + B.size());         // preallocate memory
        AB.insert(AB.end(), A.begin(), A.end()); // add A;
        AB.insert(AB.end(), B.begin(), B.end()); // add B;
        return AB;
    }
}

#endif // PIPELINE_H