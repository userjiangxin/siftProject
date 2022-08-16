#include <iostream>
#include <cmath>
#include "MySift.h"
int main()
{
	cv::Mat src = cv::imread("D:/opencv/pictures/lena.jpeg");
	cv::Mat src1;
	cv::resize(src, src1, cv::Size(src.cols * 2, src.rows * 2));
	MySift st; 
	cv::Mat BaseImg;
	std::vector<std::vector<cv::Mat>> pyr;
	std::vector<std::vector<cv::Mat>> dogpyr;
	std::vector<cv::KeyPoint> keypoints;
	//st.createBaseImg(src, BaseImg);
	cv::Ptr<cv::SiftFeatureDetector> sift = cv::SiftFeatureDetector::create();
	sift->detect(src, keypoints);
	cv::drawKeypoints(src, keypoints, src, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	/*int Octaves = st.getOctaveNum(BaseImg, 4);
	std::cout << Octaves << std::endl;*/
	//st.buildGaussianPyramid(BaseImg, pyr, 7);
	//st.buildDoGPyramid(pyr, dogpyr);
	//st.findScaleSpaceExtremum(pyr, dogpyr, keypoints);
	//for (int i = 0; i < keypoints.size(); i++)
	//{
	//	std::cout << keypoints[i].pt.x << " " << keypoints[i].pt.y << std::endl;
	//	std::cout << keypoints[i].size << std::endl;
	//	std::cout << keypoints[i].angle << std::endl;
	//	std::cout << keypoints[i].response << std::endl;
	//}
	////cv::drawKeypoints(src1, keypoints, src1, cv::Scalar::all(-1),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cv::imshow("src1", src1);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}