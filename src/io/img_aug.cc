#include "img_aug.hpp"

#include <assert.h>

#include <random>


void ImgIllumTrans(cv::Mat &img, uint8_t nNewMean)
{
	cv::cvtColor(img, img, CV_BGR2HLS);
	std::vector<cv::Mat> hls(3);
	cv::split(img, hls);
	double dMinL, dMaxL;
	cv::minMaxLoc(hls[1], &dMinL, &dMaxL);

	int nNewRange = 255;
	int nNewMinL = nNewMean - nNewRange / 2;

	double dBeta = (double)nNewRange / (double)(dMaxL - dMinL);
	// newPixel = (oldPixel - nMinL) * dBeta + nNewMinL
	// alpha = nMaxL, beta = -nMinL * dBeta + nNewMinL
	hls[1].convertTo(
		hls[1],
		hls[1].type(),
		dBeta,
		nNewMinL - dMinL * dBeta
		);
	cv::merge(hls, img);
	cv::cvtColor(img, img, CV_HLS2BGR);
}

bool CompatibleMat(const cv::Mat &m1, const cv::Mat &m2)
{
	return (
			m1.size() == m2.size() &&
			m1.channels() == m2.channels() &&
			m1.type() == m2.type()
			);
}

cv::Mat GetMotionKernel(int sz, int d, float theta)
{
	const float c = std::cos(theta / 180.0f * (float)M_PI);
	const float s = std::sin(theta / 180.0f * (float)M_PI);

	cv::Mat A = cv::Mat::zeros(cv::Size(3, 2), CV_32FC1);
	A.at<float>(0, 0) = c;
	A.at<float>(0, 1) = -s;
	A.at<float>(1, 0) = s;
	A.at<float>(1, 1) = c;

	cv::Mat vec = cv::Mat::zeros(cv::Size(1, 2), CV_32FC1);
	vec.at<float>(0, 0) = (d - 1) / 2;

	cv::Mat B = A(cv::Rect(0, 0, 2, 2)).clone();
	cv::Mat tmp = B * vec;
	A.at<float>(0, 2) = sz / 2 - tmp.at<float>(0, 0);
	A.at<float>(1, 2) = sz / 2 - tmp.at<float>(1, 0);

	cv::Mat kernel;
	cv::warpAffine(
			cv::Mat::ones(cv::Size(d, 1), CV_32FC1),
			kernel,
			A,
			cv::Size(sz, sz),
			cv::INTER_CUBIC
			);

	kernel /= cv::sum(kernel)[0];
	return kernel;
}

void ImgJpegComp(cv::Mat &img, int quality)
{
	std::vector<int> params;
	params.push_back(CV_IMWRITE_JPEG_QUALITY);
	params.push_back(quality); //jpeg quality 0-100, the higher the better

	std::vector<uchar> buffer;
	if (cv::imencode(".jpg", img, buffer, params))
	{
		img = cv::imdecode(buffer, CV_LOAD_IMAGE_COLOR);
	}
}

void ImgResChange(cv::Mat &img, float ratio)
{
	assert(ratio <= 1 && ratio > 0);
	cv::Size orgSize = img.size();
	cv::Size smallSize(
			std::max(1, int(img.cols * ratio)),
			std::max(1, int(img.rows * ratio))
			);
	cv::resize(img, img, smallSize);	//down sampling
	cv::resize(img, img, orgSize);		//up sampling
}

void ImgHsvAdjust(cv::Mat &img, const int h, const int s, const int v)
{
	cv::Mat hsv;
	assert(img.channels() != 1);
	if (img.channels() == 4)
	{
		cv::cvtColor(img, hsv, cv::COLOR_RGBA2BGR);
	}
	cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

	for (int i = 0; i < hsv.rows; i++)
	{
		cv::Vec3b *pRow = hsv.ptr<cv::Vec3b>(i);
		for (int j = 0; j < hsv.cols; j++)
		{
			cv::Vec3b &pixel = pRow[j];
			int cPointH = pixel[0] + h;
			int cPointS = pixel[1] + s;
			int cPointV = pixel[2] + v;
			// hue
			if (cPointH < 0)
			{
				pixel[0] = 0;
			}
			else if (cPointH > 179)
			{
				pixel[0] = 179;
			}
			else
			{
				pixel[0] = cPointH;
			}
			// saturation
			if (cPointS < 0)
			{
				pixel[1] = 0;
			}
			else if (cPointS > 255)
			{
				pixel[1] = 255;
			}
			else
			{
				pixel[1] = cPointS;
			}
			// value
			if (cPointV < 0)
			{
				pixel[2] = 0;
			}
			else if (cPointV > 255)
			{
				pixel[2] = 255;
			}
			else
			{
				pixel[2] = cPointV;
			}
		}
	}
	cv::cvtColor(hsv, img, cv::COLOR_HSV2BGR);
}
