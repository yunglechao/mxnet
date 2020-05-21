/*!
 *  Copyright (c) 2015 by Contributors
 * \file face_image_aug.cc
 * \brief Face augmenter.
 */
#include <mxnet/base.h>
#include <utility>
#include <string>
#include <algorithm>
#include <vector>
#include "./image_augmenter.h"
#include "../common/utils.h"
#include "./img_aug.hpp"
#include "./global.hpp"

#if MXNET_USE_OPENCV
// Registers
namespace dmlc {
DMLC_REGISTRY_ENABLE(::mxnet::io::ImageAugmenterReg);
}  // namespace dmlc
#endif

namespace mxnet {
namespace io {

/*! \brief image augmentation parameters*/
struct FaceImageAugmentParam : public dmlc::Parameter<FaceImageAugmentParam> {
  /*! \brief resize shorter edge to size before applying other augmentations */
  int resize_height;
  int resize_width;
  int patch_size;
  int patch_idx;
  bool do_aug;
  int FacePatchSize_Main;
  int FacePatchSize_Other;
  int PatchFullSize;
  int PatchCropSize;
  float illum_trans_prob;
  float gauss_blur_prob;
  float motion_blur_prob;
  float jpeg_comp_prob;
  float res_change_prob;
  float hsv_adjust_prob;
  float rgb2gray_prob;
  bool color_mode;
  int aug_seed;

  // declare parameters
  DMLC_DECLARE_PARAMETER(FaceImageAugmentParam) {
    DMLC_DECLARE_FIELD(resize_height).set_default(-1)
        .describe("resize height.");
    DMLC_DECLARE_FIELD(resize_width).set_default(-1)
        .describe("resize width.");
    DMLC_DECLARE_FIELD(patch_size).set_default(7)
        .describe("patch_size.");
    DMLC_DECLARE_FIELD(patch_idx).set_default(0)
        .describe("patch_idx.");
    DMLC_DECLARE_FIELD(do_aug).set_default(true)
        .describe("do_aug.");
    DMLC_DECLARE_FIELD(FacePatchSize_Main).set_default(267)
        .describe("facepatchsize_main.");
    DMLC_DECLARE_FIELD(FacePatchSize_Other).set_default(128)
        .describe("facepatchsize_other.");
    DMLC_DECLARE_FIELD(PatchFullSize).set_default(128)
        .describe("patchfullsize.");
    DMLC_DECLARE_FIELD(PatchCropSize).set_default(108)
        .describe("patchcropsize.");
    DMLC_DECLARE_FIELD(illum_trans_prob).set_default(0.5)
        .describe("illum_trans_prob.");
    DMLC_DECLARE_FIELD(gauss_blur_prob).set_default(0.5)
        .describe("gauss_blur_prob.");
    DMLC_DECLARE_FIELD(motion_blur_prob).set_default(0.5)
        .describe("motion_blur_prob.");
    DMLC_DECLARE_FIELD(jpeg_comp_prob).set_default(0.5)
        .describe("jpeg_comp_prob.");
    DMLC_DECLARE_FIELD(res_change_prob).set_default(0.5)
        .describe("res_change_prob.");
    DMLC_DECLARE_FIELD(hsv_adjust_prob).set_default(0.5)
        .describe("hsv_adjust_prob.");
    DMLC_DECLARE_FIELD(rgb2gray_prob).set_default(0.5)
        .describe("rgb2gray_prob.");
    DMLC_DECLARE_FIELD(color_mode).set_default(true)
        .describe("color_mode.");
    DMLC_DECLARE_FIELD(aug_seed).set_default(0)
        .describe("aug_seed.");
  }
};

DMLC_REGISTER_PARAMETER(FaceImageAugmentParam);

std::vector<dmlc::ParamFieldInfo> ListFaceAugmentParams() {
  return FaceImageAugmentParam::__FIELDS__();
}

#if MXNET_USE_OPENCV

#ifdef _MSC_VER
#define M_PI CV_PI
#endif
/*! \brief helper class to do image augmentation */
class FaceImageAugmenter : public ImageAugmenter {
 public:
  // contructor
  FaceImageAugmenter() {
  }


  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    kwargs_left = param_.InitAllowUnknown(kwargs);
    std::srand(param_.aug_seed);
  }

  std::vector<cv::Mat> FaceProcess(const cv::Mat &src, std::vector<float> *label,
                  common::RANDOM_ENGINE *prnd) override {
    //get landmarks from *label
    std::vector<float> face_label = *label;
    LANDMARK lm;
    assert(face_label.size() == 145);
    for(int i = 1; i < face_label.size(); i+=2) {
       // skip first label
       float x = face_label[i];
       float y = face_label[i+1];
       cv::Point2f pt; pt.x = x; pt.y = y;
       assert(pt.x <= 400 && pt.y <=400);
       lm.push_back(pt);
    }
    //do resize based on src
    cv::Mat res;
    if (param_.resize_height != -1 && param_.resize_width != -1) {
      int new_width = param_.resize_height; // 400
      int new_height = param_.resize_width; // 400
      if (new_width != src.cols || new_height != src.rows) {
        cv::resize(src, res, cv::Size(new_width, new_height));
      } else {
        res = src;
      }
    } else {
      res = src;
    }
    //do augmentation
    std::vector<cv::Mat> patches;

    int patch_size = param_.patch_size; // 1
    int patch_idx = param_.patch_idx; // 0
    bool do_aug = param_.do_aug;
    LoadPatches(res, lm, patches, patch_size, patch_idx, do_aug);

    //return Mat vectors
    return patches;
  }

  bool ImageAugmentation(
      cv::Mat &img,
      AUG_METHOD augMethod,
      bool do_aug
      )
  {
      assert(img.type() == CV_8UC3);
      assert(augMethod < AM_GUARD);

      auto UNIPROBTEST = [](float fProb) -> bool
      {
          return (std::rand() / (RAND_MAX + 1.0f)) < fProb;
      };

      float illum_trans_prob = param_.illum_trans_prob;
      float gauss_blur_prob = param_.gauss_blur_prob;
      float motion_blur_prob = param_.motion_blur_prob;
      float jpeg_comp_prob = param_.jpeg_comp_prob;
      float res_change_prob = param_.res_change_prob;
      float hsv_adjust_prob = param_.hsv_adjust_prob;
      float rgb2gray_prob = param_.rgb2gray_prob;
      if (!do_aug) {
          illum_trans_prob = 0;
          gauss_blur_prob = 0;
          motion_blur_prob = 0;
          jpeg_comp_prob = 0;
          res_change_prob = 0;
          hsv_adjust_prob = 0;
          rgb2gray_prob = 0;
      }

          if (UNIPROBTEST(illum_trans_prob))
          {
              std::uniform_int_distribution<> rMean(50, 200);
              int nNewMean = rMean(m_rg);
              ImgIllumTrans(img, nNewMean);
          }
          if (UNIPROBTEST(gauss_blur_prob))
          {
              std::uniform_int_distribution<> rSize(1, 2);
              int nKernelSize = rSize(m_rg) * 2 + 1; // must be 3 or 5
              cv::GaussianBlur(img, img, cv::Size(nKernelSize, nKernelSize), 0, 0);
          }
          if (UNIPROBTEST(motion_blur_prob))
          {
              const int sz = 10;
              std::uniform_int_distribution<> rSize(0, sz - 3);
              std::uniform_real_distribution<float> rTheta(0.0f, 180.0f);
              cv::Mat kernel = GetMotionKernel(sz, rSize(m_rg) + 1, rTheta(m_rg));
              cv::filter2D(img, img, img.depth(), kernel);
              //std::cout << "motion " << t.Reset() << std::endl;
          }
          if (UNIPROBTEST(jpeg_comp_prob))
          {
              //CMyTimer t;
              std::uniform_int_distribution<> rQuality(30, 60);
              ImgJpegComp(img, rQuality(m_rg));
              //std::cout << "jpeg " << t.Reset() << std::endl;
          }
          if (UNIPROBTEST(res_change_prob))
          {
              //CMyTimer t;
              std::uniform_real_distribution<float> rScale(0.2f, 1.0f);
              ImgResChange(img, rScale(m_rg));
              //std::cout << "res " << t.Reset() << std::endl;
          }
          if (UNIPROBTEST(hsv_adjust_prob))
          {
              //CMyTimer t;
              std::uniform_int_distribution<> rh(-10, 10);
              std::uniform_int_distribution<> rs(-20, 20);
              ImgHsvAdjust(img, rh(m_rg), rs(m_rg), 0);
              //std::cout << "hsv " << t.Reset() << std::endl;
          }
          if (UNIPROBTEST(rgb2gray_prob))
          {
              cv::Mat gray;
              cv::cvtColor(img, gray, CV_RGB2GRAY);
              cv::cvtColor(gray, img, CV_GRAY2RGB);
          }
      return true;
  }


  cv::Rect2f GetFacePartRect(
          const cv::Size &imgSize,
          const LANDMARK &lm,
          FACE_PART facePart
          )
  {
      cv::Size_<float> m_FaceFullSize = cv::Size_<float>(267, 267);
      cv::Size_<float> m_FaceRimlessSize = cv::Size_<float>(200, 200);
      cv::Size_<float> m_FacePartSize = cv::Size_<float>(128, 128);

      assert(facePart < FP_GUARD);
      switch (facePart)
      {
      case FP_FULL_FACE:
          return RectOfCenter(CenterOfSize(cv::Size2f(imgSize)), m_FaceFullSize);
      case FP_LEFT_EYE:
          return RectOfCenter(lm[21], m_FacePartSize);
      case FP_RIGHT_EYE:
          return RectOfCenter(lm[38], m_FacePartSize);
      case FP_NOSE:
          return RectOfCenter(lm[57], m_FacePartSize);
      case FP_MOUTH:
          return RectOfCenter(MiddelPoint(lm[58], lm[62]), m_FacePartSize);
      case FP_MIDDLE_EYE:
          return RectOfCenter(MiddelPoint(lm[21], lm[38]), m_FacePartSize);
      case FP_LEFT_MOUTH:
          return RectOfCenter(lm[58], m_FacePartSize);
      case FP_RIGHT_MOUTH:
          return RectOfCenter(lm[62], m_FacePartSize);
      case FP_RIMLESS_FACE:
          return RectOfCenter(CenterOfSize(cv::Size2f(imgSize)), m_FaceRimlessSize);
      case FP_LEFT_BROW:
          return RectOfCenter(lm[24], m_FacePartSize);
      case FP_RIGHT_BROW:
          return RectOfCenter(lm[41], m_FacePartSize);
      case FP_MIDDLE_BROW:
          return RectOfCenter(MiddelPoint(lm[24], lm[41]), m_FacePartSize);
      default:
          return cv::Rect2f();
      }
  }


  void LoadPatches(
          cv::Mat & img_in,
          LANDMARK & lm,
          std::vector<cv::Mat> &patches,
          int patch_size,
          int patch_idx,
          bool do_aug
          )
  {
      assert(!img.empty());

      //Do data augmentation
      std::uniform_int_distribution<> rAugM(0, AM_GUARD - 1);
      AUG_METHOD augMethod = (AUG_METHOD)rAugM(m_rg);
      if (!ImageAugmentation(img_in, augMethod, do_aug))
      {
          augMethod = (AUG_METHOD)-1;
      }

      cv::Mat img;

      if (!param_.color_mode) {
          cv::cvtColor(img_in, img, CV_RGB2GRAY);
      } else {
          img = img_in;
      }

      std::vector<int> face_parts;
      face_parts.push_back(0);
      face_parts.push_back(1);
      face_parts.push_back(2);
      face_parts.push_back(3);
      face_parts.push_back(6);
      face_parts.push_back(7);

      if (patch_size == 1) {
          face_parts[0] = patch_idx;
      }

      int szw = param_.PatchFullSize;
      int szh = param_.PatchFullSize;
      cv::Size m_PatchFullSize = cv::Size(szw, szh);
      int imgw = param_.PatchCropSize;
      int imgh = param_.PatchCropSize;
      cv::Size m_PatchCropSize = cv::Size(imgw, imgh);
      bool patch_rand_crop = true;
      if (!do_aug) patch_rand_crop = false;

      for (int i = 0; i < patch_size; ++i)
      {
          FACE_PART iPart = (FACE_PART)face_parts[i];
          cv::Rect2f partRect(GetFacePartRect(img.size(), lm, iPart));

          // srcRect is the intersection of image rect and partRect
          cv::Rect2f srcRect(cv::Rect2f(cv::Point2f(), img.size()) & partRect);

          // Transform the coordination system of srcRect from image to part
          cv::Rect2f dstRect(srcRect.tl() - partRect.tl(), srcRect.size());

          // build a part image with black background
          cv::Mat part = cv::Mat::zeros(partRect.size(), img.type());
          // Take the copy from image and do resize
          img(srcRect).copyTo(part(dstRect));
          if (face_parts[i] == 0) {
              cv::resize(part, part, m_PatchFullSize);
          } else {
              assert(part.rows == 128 && part.cols == 128);
          }

          // Make crop of the patch
          cv::Size border = m_PatchFullSize - m_PatchCropSize;
          cv::Point cropTL(border / 2);
          if (patch_rand_crop)
          {
              cropTL = cv::Point(rand() % border.width, rand() % border.height);
          }

          // Add the patch of cropsize into results
          patches.push_back(part(cv::Rect(cropTL, m_PatchCropSize)).clone());
      }
  }

 private:
  // parameters
  FaceImageAugmentParam param_;
};

ImageAugmenter* ImageAugmenter::Create(const std::string& name) {
  return dmlc::Registry<ImageAugmenterReg>::Find(name)->body();
}

MXNET_REGISTER_IMAGE_AUGMENTER(aug_face)
.describe("face augmenter")
.set_body([]() {
    return new FaceImageAugmenter();
  });
#endif  // MXNET_USE_OPENCV
}  // namespace io
}  // namespace mxnet
