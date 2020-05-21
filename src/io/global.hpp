#ifndef GLOBAL_HPP_
#define GLOBAL_HPP_

#include <opencv2/opencv.hpp>
enum FACE_PART
{
    FP_FULL_FACE    = 0,
    FP_LEFT_EYE     = 1,
    FP_RIGHT_EYE    = 2,
    FP_NOSE         = 3,
    FP_MOUTH        = 4,
    FP_MIDDLE_EYE   = 5,
    FP_LEFT_MOUTH   = 6, // How many fucking mouths does one man has?
    FP_RIGHT_MOUTH  = 7, // Bai Kou Mo Bian
    FP_RIMLESS_FACE = 8,
    FP_LEFT_BROW    = 9,
    FP_RIGHT_BROW   = 10,
    FP_MIDDLE_BROW  = 11,
    FP_GUARD        = 12, // ‡THIS IS A BOUNDARY GUARD‡
};

enum AUG_METHOD
{
    AM_ILLUMIOUS    = 0,
    AM_GAUSS_BLUR   = 1,
    AM_MOTION_BLUR  = 2,
    AM_JPEG_COMP    = 3,
    AM_RESOLUTION   = 4,
    AM_HSV_ADJ      = 5,
    AM_GUARD        = 6, // ‡THIS IS A BOUNDARY GUARD‡
};

enum SHUFFLE_METHOD
{
    SM_NOCHANGE     = 0,
    SM_ONEBYONE     = 1,
    SM_TWOBYTWO     = 2,
    SM_GUARD        = 3 // ‡THIS IS A BOUNDARY GUARD‡
};

enum CONST_UINT_MISCELLANEOUS
{
    IMAGE_CHANNELS      = 3,
    LANDMARK_LENGTH     = 72,
};

typedef std::vector<cv::Point2f> LANDMARK;

static std::mt19937     m_rg;

template <typename Dtype>
inline cv::Point_<Dtype> MiddelPoint(
        const cv::Point_<Dtype> &p1,
        const cv::Point_<Dtype> &p2
        )       
{               
    return (p1 + p2) / 2;
}               
        
template <typename Dtype>
inline cv::Rect_<Dtype> RectOfCenter(
        const cv::Point_<Dtype> &center,
        const cv::Size_<Dtype> &size
        )
{
    return cv::Rect_<Dtype>(
            center.x - size.width / 2,
            center.y - size.height / 2,
            size.width,
            size.height
            );
}   
    
template <typename Dtype>
inline cv::Point_<Dtype> CenterOfSize(const cv::Size_<Dtype> &rectSize)
{       
    return cv::Point_<Dtype>(rectSize.width / 2, rectSize.height / 2);
}       



#endif
