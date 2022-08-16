#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
/*************************定义常量*****************************/

//高斯核大小和标准差关系，size=2*(GAUSS_KERNEL_RATIO*sigma)+1,经常设置GAUSS_KERNEL_RATIO=2-3之间
const double GAUSS_KERNEL_RATIO = 3;

const int MAX_OCTAVES = 8;//金字塔最大组数

const float CONTR_THR = 0.04f;//默认是的对比度阈值(D(x))

const float CURV_THR = 10.0f;//关键点主曲率阈值

const float INIT_SIGMA = 0.5f;//输入图像的初始尺度

const int IMG_BORDER = 2;//图像边界忽略的宽度

const int MAX_INTERP_STEPS = 5;//关键点精确插值次数

const int ORI_HIST_BINS = 36;//计算特征点方向直方图的BINS个数

const float ORI_SIG_FCTR = 1.5f;//计算特征点主方向时候，高斯窗口的标准差因子

const float ORI_RADIUS = 3 * ORI_SIG_FCTR;//计算特征点主方向时，窗口半径因子

const float ORI_PEAK_RATIO = 0.8f;//计算特征点主方向时，直方图的峰值比

const int DESCR_WIDTH = 4;//描述子直方图的网格大小(4x4)

const int DESCR_HIST_BINS = 8;//每个网格中直方图角度方向的维度

const float DESCR_MAG_THR = 0.2f;//描述子幅度阈值

const float DESCR_SCL_FCTR = 3.0f;//计算描述子时，每个网格的大小因子

class MySift
{
	public:
		MySift(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
			double edgeThreshold = 10, double sigma = 1.6):nfeatures(nfeatures),
			nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
			edgeThreshold(edgeThreshold), sigma(sigma) {};//默认构造函数
		int getOctaveNum(const cv::Mat &image, int t);//根据图像宽高和最顶层的图像最小维数的对数值计算金字塔的层数
		void createBaseImg(const cv::Mat &image,cv::Mat &BaseImg);//BaseImg为加入上采样后被初始sigma模糊后的图像
		void buildGaussianPyramid(const cv::Mat &BaseImg, std::vector<std::vector<cv::Mat>> &pyr, int nOctaves);//构建高斯金字塔
		void buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr);//构建高斯差分金字塔
		void findScaleSpaceExtremum(std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint>&keypoints);//极值点检测

	private:
		int nfeatures;//检测的特征点个数
		int nOctaveLayers;//构建nOctaves组，每组nOctaveLayers层
		double contrastThreshold;//极值点检测的阈值，去除不稳定点
		double edgeThreshold;//消除边缘响应的阈值
		double sigma;//第0层进行高斯模糊的尺度因子

};

