#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;

namespace pipeline {
    vector<vector<Point>> targets(Mat& frame, const Scalar& lo, const Scalar& hi);
    vector<vector<Point>> largestN(vector<vector<Point>>& v, size_t n);
    vector<double> largestNindex(vector<vector<Point>>& v, size_t n);


    template <typename Cont, typename Pred>
    Cont filter(const Cont &container, Pred predicate) {
        Cont result;
        std::copy_if(container.begin(), container.end(), std::back_inserter(result), predicate);
        return result;
    }
    template <typename Cont, typename Pred>
    Cont apply(const Cont &container, Pred predicate) {
        Cont result;
        std::for_each(container.begin(), container.end(), predicate);
        return result;
    }
    /*
    template<typename T, typename N>
    vector<N> pipeline::change(vector<T>& v, function<N(T)>& fn){
        vector<N> ret;
        for(T e: v){
            ret.push_back(fn(e));
        }
        return ret;
    }
    */
}

#endif // PIPELINE_H