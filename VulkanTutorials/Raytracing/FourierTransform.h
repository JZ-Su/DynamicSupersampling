#pragma once
#include <opencv2/opencv.hpp>

namespace NCL::Rendering::Vulkan {
	class FourierTransform {
	public:
		static cv::Mat FFTConvolve2D(const cv::Mat& image, const cv::Mat& kernel);
		static cv::Mat Spectralisation(const std::string& uri);

		static void ShowImage(cv::Mat image, const std::string& title = "", const std::string& writeTitle = "");

	protected:
		FourierTransform() {};
		~FourierTransform() {};
	};
}