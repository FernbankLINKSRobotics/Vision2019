#include "parallelCamera.h"

ParallelCamera::ParallelCamera(int ind){
    running_ = false;
    cap_ = VideoCapture(ind);
}

void ParallelCamera::start(){
    running_ = true;
    thread_ = thread(&ParallelCamera::update, this);
}

void ParallelCamera::stop(){
    running_ = false;
    if(thread_.joinable()) { thread_.join(); }
}

Mat ParallelCamera::get(){
    Mat img;
    mFrame_.lock();
    frame_.copyTo(img);
    mFrame_.unlock();
    return img;
}

void ParallelCamera::update(){
    int frameTimeout = 0;
    while(running_){
        frameTimeout = 0;
        while(cap_.grab() && frameTimeout++ < 1) {}
        mFrame_.lock();
        cap_.retrieve(frame_);
        mFrame_.unlock();
    }
}