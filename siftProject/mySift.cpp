//#include "MySift.h"
//
//
//int MySift::getOctaveNum(const cv::Mat &image, int t)
//{
//	int ans;
//	float size = (float)std::min(image.rows, image.cols);
//	ans = cvRound(log2f(size) - t);
//	return ans;
//}
//
//void MySift::createBaseImg(const cv::Mat &image, cv::Mat &BaseImg)
//{
//	cv::Mat gray_image;
//	if (image.channels() != 1){
//		cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
//	}
//	else{
//		image.clone();
//	}
//	cv::Mat float_image,temp_image;//float_image为了之后的高斯
//	gray_image.convertTo(float_image, CV_32FC1,1.0/255.0);
//	cv::resize(float_image, temp_image, cv::Size(2 * float_image.cols, 2 * float_image.rows),0,0,cv::INTER_LINEAR);
//	double sigma_diff = sqrt(sigma*sigma - 2 * INIT_SIGMA * 2 * INIT_SIGMA);
//	cv::GaussianBlur(temp_image, BaseImg, cv::Size(0, 0), sigma_diff, sigma_diff);
//
//}
//void MySift::buildGaussianPyramid(const cv::Mat &BaseImg, std::vector<std::vector<cv::Mat>> &pyr, int nOctaves)
//{
//
//	std::vector<double>sigmas;
//	sigmas.push_back(sigma);
//	double k = pow(2.0, 1.0 / nOctaveLayers);
//	double pre_sig, cur_sig;
//	for (int i = 1; i < nOctaveLayers + 3; i++)
//	{
//		pre_sig = pow(k, i - 1)*sigma;
//		cur_sig = k * pre_sig;
//		sigmas.push_back(sqrt(cur_sig*cur_sig - pre_sig * pre_sig));
//	}
//	/*for (int i = 0; i < sigmas.size(); i++)
//	{
//		std::cout << sigmas[i] << std::endl;
//	}*/
//	pyr.resize(nOctaves,std::vector<cv::Mat>(nOctaveLayers + 3));//外层、内层
//		//如果指定外层和内层向量的大小，就可用operator[]进行读和写；
//		//如果只指定外层向量大小，就能用push_back()函数进行写，不能用operator[]进行读和写。
//	for (int i = 0; i < nOctaves; i++)
//	{
//		for (int j = 0; j < nOctaveLayers + 3; j++)
//		{
//			if (i == 0 && j == 0)//第一组第一层
//				pyr[i][j] = BaseImg;
//			else if (j == 0) {//非第一组第一层
//				resize(pyr[i - 1][nOctaveLayers], pyr[i][0], cv::Size(pyr[i - 1][nOctaveLayers].cols / 2, 
//					pyr[i - 1][nOctaveLayers].rows / 2), 0, 0, cv::INTER_LINEAR);
//				
//			}
//			else
//			{
//				cv::GaussianBlur(pyr[i][j - 1], pyr[i][j], cv::Size(0,0),sigmas[j], sigmas[j]);
//			}
//		}
//	}
//	for (int i = 0; i < pyr.size(); i++)
//	{
//		for (int j = 0; j < pyr[i].size(); j++)
//		{
//			cv::Mat temp;
//			temp = pyr[i][j];
//		}
//	}
//}
//void MySift::buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr)
//{
//	int nOctaves = pyr.size();
//	dogpyr.resize(nOctaves, std::vector<cv::Mat>(nOctaveLayers + 2));
//	for (int i = 0; i < dogpyr.size(); i++)
//	{
//		for (int j = 0; j < dogpyr[i].size(); j++)
//		{
//			dogpyr[i][j] = pyr[i][j+1] - pyr[i][j];
//		}
//	}
//	for (int i = 0; i < dogpyr.size(); i++)
//	{
//		for (int j = 0; j < dogpyr[i].size(); j++)
//		{
//			cv::Mat temp = dogpyr[i][j];
//		}
//	}
//}
//static bool isExtremum(std::vector<std::vector<cv::Mat>> &dogpyr,int o,int l,int r,int c)//极值点初步检测，搜索上下三层的差分金字塔
//{
//
//	float val = dogpyr[o][l].ptr<float>(r)[c];
//	//if (abs(val) > threshold) {
//
//		if (val > 0)
//		{
//			for (int i = -1; i <= 1; i++)//层
//			{
//				for (int j = -1; j <= 1; j++)//行
//				{
//					for (int k = -1; k <= 1; k++)//列
//					{
//						if (val < dogpyr[o][l + i].ptr<float>(r + j)[c + k])
//							return false;
//					}
//				}
//			}
//		}
//		else
//		{
//			for (int i = -1; i <= 1; i++)//层
//			{
//				for (int j = -1; j <= 1; j++)//行
//				{
//					for (int k = -1; k <= 1; k++)//列
//					{
//						if (val > dogpyr[o][l + i].ptr<float>(r + j)[c + k])
//							return false;
//					}
//				}
//			}
//		}
//		return true;
//	//}
//	//return false;
//}
///*
//子像素插值，曲线拟合，利用已知的离散空间的点插值得到连续空间的极值点
//这里因为取值σ是离散的为了模拟距离远近变化，所以需要对连续的空间找到极值点
//@dogpyr:高斯差分金字塔
//@kpt:关键点
//------------------------
//在之前初步寻找极值得到的极值点所在的组、层、行、列
//@octave:组
//@layer:层
//@row:行
//@col:列
//------------------------
//@contrastThreshold:对比阈值0.04
//@edgeThreshold:边缘阈值10
//@sigma:高斯尺度空间最底层图像尺度1.6
//*/
//static bool adjustLocExtermum(const std::vector<std::vector<cv::Mat>> &dogpyr,
//	cv::KeyPoint &kpt, int octave, int &layer, int &row, int &col, int nOctaveLayers,
//	float contrastThreshold, float edgeThreshold, float sigma)
//{
//	float xr, xc, xo;
//	int num = 0;
//	for (; num < MAX_INTERP_STEPS;num++)//最大迭代次数
//	{
//		cv::Mat img = dogpyr[octave][layer];
//		cv::Mat pre = dogpyr[octave][layer - 1];
//		cv::Mat nex = dogpyr[octave][layer + 1];
//
//		//有限差分求导x,y,σ方向一阶偏导
//		float dx = (img.ptr<float>(row)[col + 1] - img.ptr<float>(row)[col - 1]) / 2.0;
//		float dy = (img.ptr<float>(row + 1)[col] - img.ptr<float>(row - 1)[col]) / 2.0;
//		float dz = (nex.ptr<float>(row)[col] - pre.ptr<float>(row)[col]) / 2.0;
//		//二阶偏导
//		float val = img.ptr<float>(row)[col];
//		float dxx = (img.ptr<float>(row)[col + 1] + img.ptr<float>(row)[col - 1]) - 2 * val;
//		float dyy = (img.ptr<float>(row + 1)[col] + img.ptr<float>(row - 1)[col]) - 2 * val;
//		float dzz = (nex.ptr<float>(row)[col] + pre.ptr<float>(row)[col]) - 2 * val;
//
//		//混合二阶偏导
//		float dxy = ((img.ptr<float>(row + 1)[col + 1] + img.ptr<float>(row - 1)[col - 1]) -
//			(img.ptr<float>(row + 1)[col - 1] + img.ptr<float>(row - 1)[col + 1])) / 4.0;
//
//		float dxz = ((nex.ptr<float>(row)[col + 1] + pre.ptr<float>(row)[col - 1]) -
//			(nex.ptr<float>(row)[col - 1] + pre.ptr<float>(row)[col + 1])) / 4.0;
//
//		float dyz = ((nex.ptr<float>(row + 1)[col] + pre.ptr<float>(row - 1)[col]) -
//			(nex.ptr<float>(row - 1)[col] + pre.ptr<float>(row + 1)[col])) / 4.0;
//
//		cv::Matx33f H(dx, dy, dz,
//			dxy, dyy, dyz,
//			dxz, dyz, dzz);
//		cv::Vec3f dD(dx, dy, dz);
//
//		cv::Vec3f X_;
//		cv::solve(H, dD, X_, cv::DECOMP_LU);//求解线性方程组的解
//		//cv::Vec3f X_ = (H.inv()*dD);
//		xc = -X_[0];//x方向偏移量
//		xr = -X_[1];//y方向偏移量
//		xo = -X_[2];//σ方向偏移量
//
//		if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xo) < 0.5f)//偏移小于0.5，极值点正确
//			break;
//		row = row + cvRound(xr);
//		col = col + cvRound(xc);
//		layer = layer + cvRound(xo);
//
//		if (layer<1 || layer>nOctaveLayers || col<IMG_BORDER || col>img.cols - IMG_BORDER
//			|| row<IMG_BORDER || row>img.rows - IMG_BORDER)//关键点在边界区域删除
//			return false;//表示极值点越界
//	}
//	if (num >= MAX_INTERP_STEPS-1)//大于迭代次数删除
//		return false;
//	//------------------------舍弃低对比度点也就是极值小于
//	//需要重新计算调整后的
//	cv::Mat image = dogpyr[octave][layer];
//	cv::Mat prev = dogpyr[octave][layer - 1];
//	cv::Mat next = dogpyr[octave][layer + 1];
//
//	float dx = (image.ptr<float>(row)[col + 1] - image.ptr<float>(row)[col - 1]) / 2.0;
//	float dy = (image.ptr<float>(row + 1)[col] - image.ptr<float>(row - 1)[col]) / 2.0;
//	float dz = (next.ptr<float>(row)[col] - prev.ptr<float>(row)[col]) / 2.0;
//
//	cv::Matx31f dD(dx, dy, dz);
//	float t = dD.dot(cv::Matx31f(xc, xr, xo));
//	float value = image.ptr<float>(row)[col] + t * 0.5;
//	if (abs(value) < CONTR_THR / nOctaveLayers)
//		return false;//去除低对比度的点
//	//------------------去除边缘响应点---------------
//	float val = image.ptr<float>(row)[col];//求hessian矩阵行列式值和迹
//	float dxx = (image.ptr<float>(row)[col + 1] + image.ptr<float>(row)[col - 1]) - 2 * val;
//	float dyy = (image.ptr<float>(row + 1)[col] + image.ptr<float>(row - 1)[col]) - 2 * val;
//	float dxy = ((image.ptr<float>(row + 1)[col + 1] + image.ptr<float>(row - 1)[col - 1]) -
//		(image.ptr<float>(row + 1)[col - 1] + image.ptr<float>(row - 1)[col + 1])) / 4.0;
//	float det = dxx * dyy - dxy * dxy;
//	float trace = dxx + dyy;
//	if (det < 0 || (trace * trace*edgeThreshold >= det * (edgeThreshold + 1)*(edgeThreshold + 1)))
//		return false;
//	//通过对比度检查和边缘响应，返回keypoint
//	kpt.pt.x = float((col + xc)*powf(2.0, octave));//最底层图像x坐标
//	kpt.pt.y = float((row + xr)*powf(2.0, octave));//y坐标
//	kpt.octave = octave + (layer << 8);//组号保存在低字节，层号保存在高字节
//	kpt.size = sigma * powf(2.0, (layer + xo) / nOctaveLayers)*powf(2.0, octave);
//	kpt.response = abs(value);
//
//	return true;
//}
///*
//获取关键点的方向
///*
//	采集关键点所在高斯金字塔
//	1.计算领域梯度方向和幅值
//	2.计算梯度方向直方图
//	3.确定特征点方向
//	@pyrImg离求出的特征点尺度最接近的高斯金字塔中的高斯图像
//
//*/
//static float calculateGridentHist(cv::Mat &image,cv::Point pt,float scale,int n,float *hist)//计算梯度方向直方图
//{
//	int radius = cvRound(ORI_RADIUS*scale);//特征点邻域半径(3*1.5*scale)
//	int len = (2 * radius + 1)*(2 * radius + 1);//特征点邻域像素总个数（最大值）
//	float sigma = ORI_SIG_FCTR * scale; //特征点邻域高斯权重标准差(1.5*scale)
//	float exp_scale = -1.f / (2 * sigma*sigma);
//	//使用AutoBuffer分配一段内存，这里多出四个空间的目的是为了方便后面平滑直方图的需要。
//	cv::AutoBuffer<float> buffer(4 * len + n + 4);
//	//X保存水平差分，Y保存数值差分，Mag保存梯度幅度，Ori保存梯度角度，W保存高斯权重
//	float *X = buffer, *Y = buffer + len, *Mag = Y, *Ori = Y + len, *W = Ori + len;
//	float *temp_hist = W + len + 2;//临时保存直方图数据
//	for (int i = 0; i < n; ++i)
//		temp_hist[i] = 0.f;//数据清零
//
//	//计算邻域像素的水平差分和竖直差分
//	int k = 0;
//	for (int i = -radius; i < radius; ++i)
//	{
//		int y = pt.y + i;
//		if (y < 0 || y > image.rows - 1)
//			continue;
//		for (int j = -radius; j < radius; ++j)
//		{
//			int x = pt.x + j;
//			if (x < 0 || x > image.cols - 1)
//				continue;
//
//			float dx = image.at<float>(y, x + 1) - image.at<float>(y, x - 1);
//			float dy = image.at<float>(y + 1, x) - image.at<float>(y - 1, x);
//			X[k] = dx; Y[k] = dy; W[k] = (i*i + j * j)*exp_scale;
//			++k;
//		}
//	}
//	len = k;
//	//计算邻域像素的梯度幅度,梯度方向，高斯权重
//	for (int i = 0; i < k; i++)
//	{
//		W[i] = exp(W[i]);
//		Ori[i] = cv::fastAtan2(Y[i], X[i]);
//		Mag[i] = sqrt(X[i] * X[i] + Y[i] * Y[i]);
//	}
//	for (int i = 0; i < len; ++i)
//	{
//		int bin = cvRound((n / 360.f)*Ori[i]);//bin的范围约束在[0,(n-1)]
//		if (bin >= n)
//			bin = bin - n;
//		if (bin < 0)
//			bin = bin + n;
//		temp_hist[bin] = temp_hist[bin] + Mag[i] * W[i];
//	}
//
//	//平滑直方图
//	temp_hist[-1] = temp_hist[n - 1];
//	temp_hist[-2] = temp_hist[n - 2];
//	temp_hist[n] = temp_hist[0];
//	temp_hist[n + 1] = temp_hist[1];
//	for (int i = 0; i < n; ++i)
//	{
//		hist[i] = (temp_hist[i - 2] + temp_hist[i + 2])*(1.f / 16.f) +
//			(temp_hist[i - 1] + temp_hist[i + 1])*(4.f / 16.f) +
//			temp_hist[i] * (6.f / 16.f);
//	}
//
//	//获得直方图中最大值
//	float max_value = hist[0];
//	for (int i = 1; i < n; ++i)
//	{
//		if (hist[i] > max_value)
//			max_value = hist[i];
//	}
//	return max_value;
//}
////尺度空间极值点检测
//void MySift::findScaleSpaceExtremum(std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint>&keypoints)//极值点检测
//{
//	int count = 0;
//	int nOctaves = (int)dogpyr.size();
//	const int n = ORI_HIST_BINS;
//	float hist[n];
//	float threshold = 0.5*contrastThreshold / nOctaveLayers;
//	cv::KeyPoint keypoint;
//	keypoints.clear();
//	for (int i = 0; i < nOctaves; i++)
//	{
//		for (int j = 1; j < (nOctaveLayers+2)-1; j++)
//		{
//			cv::Mat cur_img = dogpyr[i][j];
//			cv::Mat pre_img = dogpyr[i][j - 1];
//			cv::Mat nex_img = dogpyr[i][j + 1];
//			int num_row = cur_img.rows;
//			int num_col = cur_img.cols;
//			for (int r = IMG_BORDER; r < num_row-IMG_BORDER; r++)
//			{
//				for (int c = IMG_BORDER; c < num_col - IMG_BORDER; c++)
//				{
//					float val = cur_img.ptr<float>(r)[c];
//					int octave = i, layer = j, row = r, col = c;
//					if (abs(val)>threshold&&isExtremum(dogpyr, octave, layer, row, col))//离散空间极值点&&极值点检测阈值
//					{
//						if (adjustLocExtermum(dogpyr, keypoint, octave, layer, row,col, nOctaveLayers, contrastThreshold,edgeThreshold,sigma))//检测通过
//						{
//							//count++;
//							//std::cout << keypoint.pt.x << " " << keypoint.pt.y << std::endl;
//							float scale = keypoint.size / powf(2.0, octave);//特征点相对于本组的尺度？？？
//							float max_hist = calculateGridentHist(pyr[octave][layer], cv::Point(col, row), scale, n, hist);
//							float mag_thr = max_hist * ORI_PEAK_RATIO;
//							//抛物线插值
//							for (int i = 0; i < n; ++i)
//							{
//								int left = 0, right = 0;
//								if (i == 0)
//									left = n - 1;
//								else
//									left = i - 1;
//
//								if (i == n - 1)
//									right = 0;
//								else
//									right = i + 1;
//
//								if (hist[i] > hist[left] && hist[i] > hist[right] && hist[i] >= mag_thr)
//								{
//									float bin = i + 0.5f*(hist[left] - hist[right]) / (hist[left] + hist[right] - 2 * hist[i]);
//									if (bin < 0)
//										bin = bin + n;
//									if (bin >= n)
//										bin = bin - n;
//
//									keypoint.angle = (360.f / n)*bin;//特征点的主方向0-360度
//									keypoints.push_back(keypoint);//保存该特征点
//
//								}
//
//							}
//						}
//
//					}
//				}
//			}
//		}
//	}
//	std::cout << count << std::endl;
//}