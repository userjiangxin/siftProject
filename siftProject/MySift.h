#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
/*************************���峣��*****************************/

//��˹�˴�С�ͱ�׼���ϵ��size=2*(GAUSS_KERNEL_RATIO*sigma)+1,��������GAUSS_KERNEL_RATIO=2-3֮��
const double GAUSS_KERNEL_RATIO = 3;

const int MAX_OCTAVES = 8;//�������������

const float CONTR_THR = 0.04f;//Ĭ���ǵĶԱȶ���ֵ(D(x))

const float CURV_THR = 10.0f;//�ؼ�����������ֵ

const float INIT_SIGMA = 0.5f;//����ͼ��ĳ�ʼ�߶�

const int IMG_BORDER = 2;//ͼ��߽���ԵĿ��

const int MAX_INTERP_STEPS = 5;//�ؼ��㾫ȷ��ֵ����

const int ORI_HIST_BINS = 36;//���������㷽��ֱ��ͼ��BINS����

const float ORI_SIG_FCTR = 1.5f;//����������������ʱ�򣬸�˹���ڵı�׼������

const float ORI_RADIUS = 3 * ORI_SIG_FCTR;//����������������ʱ�����ڰ뾶����

const float ORI_PEAK_RATIO = 0.8f;//����������������ʱ��ֱ��ͼ�ķ�ֵ��

const int DESCR_WIDTH = 4;//������ֱ��ͼ�������С(4x4)

const int DESCR_HIST_BINS = 8;//ÿ��������ֱ��ͼ�Ƕȷ����ά��

const float DESCR_MAG_THR = 0.2f;//�����ӷ�����ֵ

const float DESCR_SCL_FCTR = 3.0f;//����������ʱ��ÿ������Ĵ�С����

class MySift
{
	public:
		MySift(int nfeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
			double edgeThreshold = 10, double sigma = 1.6):nfeatures(nfeatures),
			nOctaveLayers(nOctaveLayers), contrastThreshold(contrastThreshold),
			edgeThreshold(edgeThreshold), sigma(sigma) {};//Ĭ�Ϲ��캯��
		int getOctaveNum(const cv::Mat &image, int t);//����ͼ���ߺ�����ͼ����Сά���Ķ���ֵ����������Ĳ���
		void createBaseImg(const cv::Mat &image,cv::Mat &BaseImg);//BaseImgΪ�����ϲ����󱻳�ʼsigmaģ�����ͼ��
		void buildGaussianPyramid(const cv::Mat &BaseImg, std::vector<std::vector<cv::Mat>> &pyr, int nOctaves);//������˹������
		void buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr);//������˹��ֽ�����
		void findScaleSpaceExtremum(std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint>&keypoints);//��ֵ����

	private:
		int nfeatures;//�������������
		int nOctaveLayers;//����nOctaves�飬ÿ��nOctaveLayers��
		double contrastThreshold;//��ֵ�������ֵ��ȥ�����ȶ���
		double edgeThreshold;//������Ե��Ӧ����ֵ
		double sigma;//��0����и�˹ģ���ĳ߶�����

};

