#ifndef IMG_AUG_HPP_
#define IMG_AUG_HPP_

#include <opencv2/opencv.hpp>


void ImgIllumTrans(cv::Mat &img, uint8_t nNewMean);

bool CompatibleMat(const cv::Mat &m1, const cv::Mat &m2);

cv::Mat GetMotionKernel(int sz, int d, float theta);

void ImgJpegComp(cv::Mat &img, int quality);

void ImgResChange(cv::Mat &img, float ratio);

void ImgHsvAdjust(cv::Mat &img, const int h, const int s, const int v);


#endif
