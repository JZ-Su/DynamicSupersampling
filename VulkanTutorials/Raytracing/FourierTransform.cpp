#include "FourierTransform.h"

using namespace NCL;
using namespace Rendering;
using namespace Vulkan;

cv::Mat FourierTransform::FFTConvolve2D(const cv::Mat& image, const cv::Mat& kernel) {
	// Get the input image size and the convolution kernel size
	int M = image.rows;
	int N = image.cols;
	int K = kernel.rows;
	int L = kernel.cols;

	// Calculate the expanded size
	int P = M + K - 1;
	int Q = N + L - 1;

	// Zero-padding of inputs and convolution kernels
	cv::Mat A_pad(P, Q, CV_32F, cv::Scalar::all(0));
	cv::Mat B_pad(P, Q, CV_32F, cv::Scalar::all(0));
	cv::Mat C_pad(P, Q, CV_32F, cv::Scalar::all(0));

	image.copyTo(A_pad(cv::Rect(0, 0, image.cols, image.rows)));
	kernel.copyTo(B_pad(cv::Rect(0, 0, kernel.cols, kernel.rows)));

	// Calculate FFT
	cv::Mat A_fft, B_fft, C_fft;
	cv::dft(A_pad, A_fft, cv::DFT_COMPLEX_OUTPUT);
	cv::dft(B_pad, B_fft, cv::DFT_COMPLEX_OUTPUT);

	// Multiply point by point
	mulSpectrums(A_fft, B_fft, C_fft, 0);

	// Calculate inverse FFT
	cv::idft(C_fft, C_pad, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
	
	// Trimming results
	cv::Mat C = C_pad(cv::Rect(0, 0, image.cols, image.rows));

	return C;
}

cv::Mat FourierTransform::Spectralisation(const std::string& uri) {
	cv::Mat image = cv::imread(uri, cv::IMREAD_GRAYSCALE);
	if (image.empty()) {
		std::cout << "Error: Cannot find the file!" << std::endl;
		return image;
	}

	// Get image size
	int rows = image.rows;
	int cols = image.cols;

	cv::Mat padded;
	int m = cv::getOptimalDFTSize(image.rows);
	int n = cv::getOptimalDFTSize(image.cols);
	cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// Allocate space to the complex plane
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);

	// Calculate Fourier Transform
	cv::dft(complexI, complexI);

	// Calculate magnitude spectrum
	cv::split(complexI, planes);
	cv::magnitude(planes[0], planes[1], planes[0]);
	cv::Mat magnitudeImage = planes[0];

	// Switch to logarithmic scale
	magnitudeImage += cv::Scalar::all(1);
	cv::log(magnitudeImage, magnitudeImage);

	// Shear and redistributed magnitude map quadrant
	magnitudeImage = magnitudeImage(cv::Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;

	cv::Mat q0(magnitudeImage, cv::Rect(0, 0, cx, cy));   // Upper left  - lower left
	cv::Mat q1(magnitudeImage, cv::Rect(cx, 0, cx, cy));  // Upper right - lower right
	cv::Mat q2(magnitudeImage, cv::Rect(0, cy, cx, cy));  // Lower left  - upper left
	cv::Mat q3(magnitudeImage, cv::Rect(cx, cy, cx, cy)); // Lower right - upper right

	// Switch quadrant
	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);

	return magnitudeImage;
}

/*
	Normalize and show the image
	@param image: the output image mat
	@param title: output window's title
	@param outputFilename: If the string is not "", output the image file with it.
*/
void FourierTransform::ShowImage(cv::Mat image, const std::string& title, const std::string& outputFilename) {
	cv::Mat m;
	image.convertTo(m, CV_32F);
	cv::normalize(m, m, 0, 1, cv::NORM_MINMAX);
	cv::imshow(title, m);
	if (outputFilename != "") {
		cv::imwrite(outputFilename, m);
	}
}