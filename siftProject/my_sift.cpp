#include "MySift.h"
int MySift::getOctaveNum(const cv::Mat &image, int t)
{
	int ans;
	float size = (float)std::min(image.rows, image.cols);
	ans = cvRound(log2f(size) - t);
	return ans;
}

void MySift::createBaseImg(const cv::Mat &image, cv::Mat &BaseImg)
{
	cv::Mat gray_image;
	if (image.channels() != 1) {
		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
	}
	else {
		image.clone();
	}
	cv::Mat float_image, temp_image;//float_image为了之后的高斯
	gray_image.convertTo(float_image, CV_32FC1, 1.0 / 255.0);
	cv::resize(float_image, temp_image, cv::Size(2 * float_image.cols, 2 * float_image.rows), 0, 0, cv::INTER_LINEAR);
	double sigma_diff = sqrt(sigma*sigma - 2 * INIT_SIGMA * 2 * INIT_SIGMA);
	cv::GaussianBlur(temp_image, BaseImg, cv::Size(0, 0), sigma_diff, sigma_diff);

}
void MySift::buildGaussianPyramid(const cv::Mat &BaseImg, std::vector<std::vector<cv::Mat>> &pyr, int nOctaves)
{
	//产生高斯滤波核
	std::vector<double>sigmas;
	sigmas.push_back(sigma);
	double k = pow(2.0, 1.0 / nOctaveLayers);
	double pre_sig, cur_sig;
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		pre_sig = pow(k, i - 1)*sigma;
		cur_sig = k * pre_sig;
		sigmas.push_back(sqrt(cur_sig*cur_sig - pre_sig * pre_sig));
	}
	for (int i = 0; i < sigmas.size(); i++)
	{
		std::cout << sigmas[i] << std::endl;
	}
	//------------------------------------------------------------//
	pyr.resize(nOctaves, std::vector<cv::Mat>(nOctaveLayers + 3));//外层、内层
		//如果指定外层和内层向量的大小，就可用operator[]进行读和写；
		//如果只指定外层向量大小，就能用push_back()函数进行写，不能用operator[]进行读和写。
	for (int i = 0; i < nOctaves; i++)
	{
		for (int j = 0; j < nOctaveLayers + 3; j++)
		{
			if (i == 0 && j == 0)//第一组第一层
				pyr[i][j] = BaseImg;
			else if (j == 0) {//非第一组第一层
				resize(pyr[i - 1][nOctaveLayers], pyr[i][0], cv::Size(pyr[i - 1][nOctaveLayers].cols / 2,
					pyr[i - 1][nOctaveLayers].rows / 2), 0, 0, cv::INTER_LINEAR);

			}
			else
			{
				cv::GaussianBlur(pyr[i][j - 1], pyr[i][j], cv::Size(0, 0), sigmas[j], sigmas[j]);
			}
		}
	}
	for (int i = 0; i < pyr.size(); i++)
	{
		for (int j = 0; j < pyr[i].size(); j++)
		{
			cv::Mat temp;
			temp = pyr[i][j];
		}
	}
}
void MySift::buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr)
{
	int nOctaves = pyr.size();
	dogpyr.resize(nOctaves, std::vector<cv::Mat>(nOctaveLayers + 2));
	for (int i = 0; i < dogpyr.size(); i++)
	{
		for (int j = 0; j < dogpyr[i].size(); j++)
		{
			dogpyr[i][j] = pyr[i][j + 1] - pyr[i][j];
		}
	}
	for (int i = 0; i < dogpyr.size(); i++)
	{
		for (int j = 0; j < dogpyr[i].size(); j++)
		{
			cv::Mat temp = dogpyr[i][j];
		}
	}
}
static bool isExtremum(std::vector<std::vector<cv::Mat>> &dogpyr, int o, int l, int r, int c,float threshold)//极值点初步检测，搜索上下三层的差分金字塔
{
	float val = dogpyr[o][l].ptr<float>(r)[c];
	if (abs(val) > threshold)
	{
		if (val > 0)
		{
			for (int i = -1; i <= 1; i++)//层
			{
				for (int j = -1; j <= 1; j++)//行
				{
					for (int k = -1; k <= 1; k++)//列
					{
						if (val < dogpyr[o][l + i].ptr<float>(r + j)[c + k])
							return false;
					}
				}
			}
		}
		else
		{
			for (int i = -1; i <= 1; i++)//层
			{
				for (int j = -1; j <= 1; j++)//行
				{
					for (int k = -1; k <= 1; k++)//列
					{
						if (val > dogpyr[o][l + i].ptr<float>(r + j)[c + k])
							return false;
					}
				}
			}
		}
		return true;
	}
	return false;
}
/*
子像素插值，曲线拟合，利用已知的离散空间的点插值得到连续空间的极值点
这里因为取值σ是离散的为了模拟距离远近变化，所以需要对连续的空间找到极值点
@dogpyr:高斯差分金字塔
@kpt:关键点
------------------------
在之前初步寻找极值得到的极值点所在的组、层、行、列
@octave:组
@layer:层
@row:行
@col:列
------------------------
@contrastThreshold:对比阈值0.04
@edgeThreshold:边缘阈值10
@sigma:高斯尺度空间最底层图像尺度1.6
*/
static bool adjustLocExtermum(const std::vector<std::vector<cv::Mat>> &dogpyr,
	cv::KeyPoint &kpt, int octave, int &layer, int &row, int &col, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	float xr, xc, xl;
	int num = 0;
	for (; num < MAX_INTERP_STEPS; num++)//最大迭代次数
	{
		cv::Mat img = dogpyr[octave][layer];
		cv::Mat pre = dogpyr[octave][layer - 1];
		cv::Mat nex = dogpyr[octave][layer + 1];

		//有限差分求导x,y,σ方向一阶偏导
		float dx = (img.ptr<float>(row)[col + 1] - img.ptr<float>(row)[col - 1]) / 2.0;
		float dy = (img.ptr<float>(row + 1)[col] - img.ptr<float>(row - 1)[col]) / 2.0;
		float dz = (nex.ptr<float>(row)[col] - pre.ptr<float>(row)[col]) / 2.0;
		//二阶偏导
		float val = img.ptr<float>(row)[col];
		float dxx = (img.ptr<float>(row)[col + 1] + img.ptr<float>(row)[col - 1]) - 2 * val;
		float dyy = (img.ptr<float>(row + 1)[col] + img.ptr<float>(row - 1)[col]) - 2 * val;
		float dzz = (nex.ptr<float>(row)[col] + pre.ptr<float>(row)[col]) - 2 * val;

		//混合二阶偏导
		float dxy = ((img.ptr<float>(row + 1)[col + 1] + img.ptr<float>(row - 1)[col - 1]) -
			(img.ptr<float>(row + 1)[col - 1] + img.ptr<float>(row - 1)[col + 1])) / 4.0;

		float dxz = ((nex.ptr<float>(row)[col + 1] + pre.ptr<float>(row)[col - 1]) -
			(nex.ptr<float>(row)[col - 1] + pre.ptr<float>(row)[col + 1])) / 4.0;

		float dyz = ((nex.ptr<float>(row + 1)[col] + pre.ptr<float>(row - 1)[col]) -
			(nex.ptr<float>(row - 1)[col] + pre.ptr<float>(row + 1)[col])) / 4.0;

		cv::Matx33f H(dx, dy, dz,
			dxy, dyy, dyz,
			dxz, dyz, dzz);
		cv::Vec3f dD(dx, dy, dz);

		cv::Vec3f X_;
		cv::solve(H, dD, X_, cv::DECOMP_LU);//求解线性方程组的解
		//cv::Vec3f X_ = (H.inv()*dD);
		xc = -X_[0];//x方向偏移量
		xr = -X_[1];//y方向偏移量
		xl = -X_[2];//σ方向偏移量

		if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xl) < 0.5f)//偏移小于0.5，极值点正确
			break;
		col = col + cvRound(xc);
		row = row + cvRound(xr);
		layer = layer + cvRound(xl);

		if (layer<1 || layer>nOctaveLayers || col<IMG_BORDER || col>img.cols - IMG_BORDER
			|| row<IMG_BORDER || row>img.rows - IMG_BORDER)//关键点在边界区域删除
			return false;//表示极值点越界
	}
	if (num >= MAX_INTERP_STEPS)//大于迭代次数删除
		return false;
	//------------------------舍弃低对比度点也就是极值小于CONTR_THR / nOctaveLayers; CONTR_THR=0.04
	//需要重新计算调整后的
	cv::Mat image = dogpyr[octave][layer];
	cv::Mat prev = dogpyr[octave][layer - 1];
	cv::Mat next = dogpyr[octave][layer + 1];

	float dx = (image.ptr<float>(row)[col + 1] - image.ptr<float>(row)[col - 1]) / 2.0;
	float dy = (image.ptr<float>(row + 1)[col] - image.ptr<float>(row - 1)[col]) / 2.0;
	float dz = (next.ptr<float>(row)[col] - prev.ptr<float>(row)[col]) / 2.0;

	cv::Matx31f dD(dx, dy, dz);
	float t = dD.dot(cv::Matx31f(xc, xr, xl));
	float value = image.ptr<float>(row)[col] + t * 0.5;
	if (abs(value) < CONTR_THR / nOctaveLayers)
		return false;//去除低对比度的点
	//------------------去除边缘响应点---------------
	float val = image.ptr<float>(row)[col];//求hessian矩阵行列式值和迹
	float dxx = (image.ptr<float>(row)[col + 1] + image.ptr<float>(row)[col - 1]) - 2 * val;
	float dyy = (image.ptr<float>(row + 1)[col] + image.ptr<float>(row - 1)[col]) - 2 * val;
	float dxy = ((image.ptr<float>(row + 1)[col + 1] + image.ptr<float>(row - 1)[col - 1]) -
		(image.ptr<float>(row + 1)[col - 1] + image.ptr<float>(row - 1)[col + 1])) / 4.0;
	float det = dxx * dyy - dxy * dxy;
	float trace = dxx + dyy;
	if (det < 0 || (trace * trace*edgeThreshold >= det * (edgeThreshold + 1)*(edgeThreshold + 1)))
		return false;
	//通过对比度检查和边缘响应，返回keypoint
	//kpt.pt = cv::Point2f((float)(col + xc)*powf(2.0, octave)), ((float)row + xr)*powf(2.0, octave)));
	kpt.pt.x = ((float)col + xc)*(1 << octave);//*powf(2.0, octave);//最底层图像x坐标
	kpt.pt.y = ((float)row + xr)*(1 << octave);//*powf(2.0, octave);//y坐标
	kpt.octave = octave + (layer << 8);//组号保存在低字节，层号保存在高字节layer<<8==layer*2^8
	kpt.size = sigma * powf(2.f, (layer + xl) / nOctaveLayers)*(1 << octave);//*powf(2.0, octave);
	kpt.response = abs(value);
	return true;
}
static double* calculateHist(cv::Mat pyrImg,cv::Point pt,int radius,float sigma)//计算梯度直方图
{
	double *hist = new double[ORI_HIST_BINS];
	//每次清除数据
	for (int k = 0; k < ORI_HIST_BINS; k++)
	{
		hist[k] = 0.f;
	}
	float exp_scale = -1.f / (2 * sigma*sigma);
	float mag, ori, weight;
	int bin;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			int y = pt.y + i;
			int x = pt.x + j;
			if (x > 0 && x < pyrImg.cols - 1 && y>0 && pyrImg.rows - 1)
			{
				float dx = pyrImg.ptr<float>(y)[x + 1] - pyrImg.ptr<float>(y)[x - 1];
				float dy = pyrImg.ptr<float>(y + 1)[x] - pyrImg.ptr<float>(y - 1)[x];
				mag = sqrt(dx*dx + dy * dy);
				ori = cv::fastAtan2(dy, dx);//返回的是角度，在0-360度之间
				weight = exp((i*i + j * j)*exp_scale);
				bin = cvRound(ORI_HIST_BINS / 360.f*ori);//约束在0-36之间
				bin = bin < ORI_HIST_BINS ? bin : 0;
				hist[bin] += weight * mag;
			}
		}
	}
	return hist;
}
static double DominantDirection(double *hist, int n,int &maxi)
{
	double maxd = hist[0];
	maxi = 0;
	for (int i = 1; i < n; i++)
	{
		if (hist[i] > maxd)
		{
			maxd = hist[i];
			maxi = i;
		}
			
	}
	return maxd;
}
//关键点方向分配
static void computeKeypointsOrientations(cv::Mat pyrImg,std::vector<cv::KeyPoint> &keypoints,cv::KeyPoint &keypoint,int octave,cv::Point pt)
{
	double *hist = new double[ORI_HIST_BINS];
	////每次清除数据
	//for (int k = 0; k < ORI_HIST_BINS; k++)
	//{
	//	hist[k] = 0.f;
	//}
	float scale = keypoint.size/powf(2.0, octave);
	float sigma = ORI_SIG_FCTR * scale;//特征点邻域高斯权重标准差(1.5*scale)
	int radius = cvRound(ORI_RADIUS*scale);//3*1.5*σ
	int len = (2 * radius + 1)*(2 * radius + 1);
	//遍历梯度方向直方图
	/*-----------------1.计算梯度直方图----------------------*/
	hist = calculateHist(pyrImg, pt, radius, sigma);
	/*for (int i = 0; i < ORI_HIST_BINS; i++)
	{
		std::cout << hist[i] << " ";
	}
	std::cout << "---------------" << std::endl;*/
	//---------------------------------------------------------
	//2.平滑直方图
	double *shist = new double[ORI_HIST_BINS];
	for (int i = 0; i < ORI_HIST_BINS; i++)
	{
		shist[i] = (6 * hist[i] + 4 * (hist[(i - 1+ORI_HIST_BINS) % ORI_HIST_BINS] + hist[(i + 1) % ORI_HIST_BINS]) + (hist[(i - 2+ORI_HIST_BINS) % ORI_HIST_BINS] + hist[(i + 2) % ORI_HIST_BINS])) / 16.0;
	}
	/*for (int i = 0; i < ORI_HIST_BINS; i++)
	{
		std::cout << shist[i] << " " ;
	}
	std::cout <<"---------------"<< std::endl;*/
	//3.确定关键点主方向
	int maxi;//主方向的最大值的索引
	double maxhist = DominantDirection(shist, ORI_HIST_BINS,maxi);//获得平滑后的直方图的主方向的最大值
	//keypoint.angle = 360.f / ORI_HIST_BINS * maxi;
	//keypoints.push_back(keypoint);
	/*4.方向直方图的峰值则代表了该特征点的方向，以直方图中的最大值作为该关键点的主方向。为了增强匹配的鲁棒性，只保留峰值大于主
	方向峰值80%的方向作为改关键点的辅方向。因此，对于同一梯度值得多个峰值的关键点位置，在相同位置和尺度将会有多个关键点被
	创建但方向不同。仅有15%的关键点被赋予多个方向，但是可以明显的提高关键点的稳定性*/
	double sec_maxhist = ORI_PEAK_RATIO * maxhist;
	for (int i = 0; i < ORI_HIST_BINS; i++)
	{
		int left = 0, right = 0;
		left = (i - 1+ORI_HIST_BINS)% ORI_HIST_BINS;
		right = (i + 1) % ORI_HIST_BINS;
		if (shist[i] > shist[left] && shist[i] > shist[right] && shist[i] >= sec_maxhist)
		{
			float bin = i + 0.5f*(shist[left] - shist[right]) / (shist[left] + shist[right] - 2 * shist[i]);
			if (bin < 0)
				bin = bin + ORI_HIST_BINS;
			if (bin >= ORI_HIST_BINS)
				bin = bin - ORI_HIST_BINS;
			//cv::KeyPoint newkeypoint;
			//newkeypoint = keypoint;
			keypoint.angle = 360.f / ORI_HIST_BINS * bin;//特征点辅方向
			keypoints.push_back(keypoint);
		}
	}
	
}
void MySift::findScaleSpaceExtremum(std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint>&keypoints)
{
	int nOctaves = (int)dogpyr.size();
	float threshold = 0.5*contrastThreshold / nOctaveLayers;
	cv::KeyPoint keypoint;
	for (int i = 0; i < nOctaves; i++)
	{
		for (int j = 1; j <= nOctaveLayers; j++)
		{
			cv::Mat cur_img = dogpyr[i][j];
			cv::Mat pre_img = dogpyr[i][j - 1];
			cv::Mat nex_img = dogpyr[i][j + 1];
			int num_row = cur_img.rows;
			int num_col = cur_img.cols;
			for (int r = IMG_BORDER; r < num_row - IMG_BORDER; r++)
			{
				for (int c = IMG_BORDER; c < num_col - IMG_BORDER; c++)
				{
					int octave = i, layer = j, row = r, col = c;
					if (isExtremum(dogpyr, octave, layer, row, col, threshold))//26个区域找离散空间极值点
					{
						if (adjustLocExtermum(dogpyr, keypoint, octave, layer, row, col, nOctaveLayers, contrastThreshold, edgeThreshold, sigma))
						{
							//到这里的所有点都是关键点，接下来是确定关键点方向//
							computeKeypointsOrientations(pyr[octave][layer],keypoints,keypoint, octave, cv::Point(col,row));
							//keypoints.push_back(keypoint);
							//std::cout << layer << std::endl;
							//std::cout << col*pow(2,i) << " " << row*pow(2,i)<< std::endl;
							//std::cout << keypoint.pt.x << " " << keypoint.pt.y << std::endl;
							//keypoints.push_back(keypoint);
						}
					}
				}
			}
		}
	}
	
}