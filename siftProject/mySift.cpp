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
//	cv::Mat float_image,temp_image;//float_imageΪ��֮��ĸ�˹
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
//	pyr.resize(nOctaves,std::vector<cv::Mat>(nOctaveLayers + 3));//��㡢�ڲ�
//		//���ָ�������ڲ������Ĵ�С���Ϳ���operator[]���ж���д��
//		//���ָֻ�����������С��������push_back()��������д��������operator[]���ж���д��
//	for (int i = 0; i < nOctaves; i++)
//	{
//		for (int j = 0; j < nOctaveLayers + 3; j++)
//		{
//			if (i == 0 && j == 0)//��һ���һ��
//				pyr[i][j] = BaseImg;
//			else if (j == 0) {//�ǵ�һ���һ��
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
//static bool isExtremum(std::vector<std::vector<cv::Mat>> &dogpyr,int o,int l,int r,int c)//��ֵ�������⣬������������Ĳ�ֽ�����
//{
//
//	float val = dogpyr[o][l].ptr<float>(r)[c];
//	//if (abs(val) > threshold) {
//
//		if (val > 0)
//		{
//			for (int i = -1; i <= 1; i++)//��
//			{
//				for (int j = -1; j <= 1; j++)//��
//				{
//					for (int k = -1; k <= 1; k++)//��
//					{
//						if (val < dogpyr[o][l + i].ptr<float>(r + j)[c + k])
//							return false;
//					}
//				}
//			}
//		}
//		else
//		{
//			for (int i = -1; i <= 1; i++)//��
//			{
//				for (int j = -1; j <= 1; j++)//��
//				{
//					for (int k = -1; k <= 1; k++)//��
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
//�����ز�ֵ��������ϣ�������֪����ɢ�ռ�ĵ��ֵ�õ������ռ�ļ�ֵ��
//������Ϊȡֵ������ɢ��Ϊ��ģ�����Զ���仯��������Ҫ�������Ŀռ��ҵ���ֵ��
//@dogpyr:��˹��ֽ�����
//@kpt:�ؼ���
//------------------------
//��֮ǰ����Ѱ�Ҽ�ֵ�õ��ļ�ֵ�����ڵ��顢�㡢�С���
//@octave:��
//@layer:��
//@row:��
//@col:��
//------------------------
//@contrastThreshold:�Ա���ֵ0.04
//@edgeThreshold:��Ե��ֵ10
//@sigma:��˹�߶ȿռ���ײ�ͼ��߶�1.6
//*/
//static bool adjustLocExtermum(const std::vector<std::vector<cv::Mat>> &dogpyr,
//	cv::KeyPoint &kpt, int octave, int &layer, int &row, int &col, int nOctaveLayers,
//	float contrastThreshold, float edgeThreshold, float sigma)
//{
//	float xr, xc, xo;
//	int num = 0;
//	for (; num < MAX_INTERP_STEPS;num++)//����������
//	{
//		cv::Mat img = dogpyr[octave][layer];
//		cv::Mat pre = dogpyr[octave][layer - 1];
//		cv::Mat nex = dogpyr[octave][layer + 1];
//
//		//���޲����x,y,�ҷ���һ��ƫ��
//		float dx = (img.ptr<float>(row)[col + 1] - img.ptr<float>(row)[col - 1]) / 2.0;
//		float dy = (img.ptr<float>(row + 1)[col] - img.ptr<float>(row - 1)[col]) / 2.0;
//		float dz = (nex.ptr<float>(row)[col] - pre.ptr<float>(row)[col]) / 2.0;
//		//����ƫ��
//		float val = img.ptr<float>(row)[col];
//		float dxx = (img.ptr<float>(row)[col + 1] + img.ptr<float>(row)[col - 1]) - 2 * val;
//		float dyy = (img.ptr<float>(row + 1)[col] + img.ptr<float>(row - 1)[col]) - 2 * val;
//		float dzz = (nex.ptr<float>(row)[col] + pre.ptr<float>(row)[col]) - 2 * val;
//
//		//��϶���ƫ��
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
//		cv::solve(H, dD, X_, cv::DECOMP_LU);//������Է�����Ľ�
//		//cv::Vec3f X_ = (H.inv()*dD);
//		xc = -X_[0];//x����ƫ����
//		xr = -X_[1];//y����ƫ����
//		xo = -X_[2];//�ҷ���ƫ����
//
//		if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xo) < 0.5f)//ƫ��С��0.5����ֵ����ȷ
//			break;
//		row = row + cvRound(xr);
//		col = col + cvRound(xc);
//		layer = layer + cvRound(xo);
//
//		if (layer<1 || layer>nOctaveLayers || col<IMG_BORDER || col>img.cols - IMG_BORDER
//			|| row<IMG_BORDER || row>img.rows - IMG_BORDER)//�ؼ����ڱ߽�����ɾ��
//			return false;//��ʾ��ֵ��Խ��
//	}
//	if (num >= MAX_INTERP_STEPS-1)//���ڵ�������ɾ��
//		return false;
//	//------------------------�����ͶԱȶȵ�Ҳ���Ǽ�ֵС��
//	//��Ҫ���¼���������
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
//		return false;//ȥ���ͶԱȶȵĵ�
//	//------------------ȥ����Ե��Ӧ��---------------
//	float val = image.ptr<float>(row)[col];//��hessian��������ʽֵ�ͼ�
//	float dxx = (image.ptr<float>(row)[col + 1] + image.ptr<float>(row)[col - 1]) - 2 * val;
//	float dyy = (image.ptr<float>(row + 1)[col] + image.ptr<float>(row - 1)[col]) - 2 * val;
//	float dxy = ((image.ptr<float>(row + 1)[col + 1] + image.ptr<float>(row - 1)[col - 1]) -
//		(image.ptr<float>(row + 1)[col - 1] + image.ptr<float>(row - 1)[col + 1])) / 4.0;
//	float det = dxx * dyy - dxy * dxy;
//	float trace = dxx + dyy;
//	if (det < 0 || (trace * trace*edgeThreshold >= det * (edgeThreshold + 1)*(edgeThreshold + 1)))
//		return false;
//	//ͨ���Աȶȼ��ͱ�Ե��Ӧ������keypoint
//	kpt.pt.x = float((col + xc)*powf(2.0, octave));//��ײ�ͼ��x����
//	kpt.pt.y = float((row + xr)*powf(2.0, octave));//y����
//	kpt.octave = octave + (layer << 8);//��ű����ڵ��ֽڣ���ű����ڸ��ֽ�
//	kpt.size = sigma * powf(2.0, (layer + xo) / nOctaveLayers)*powf(2.0, octave);
//	kpt.response = abs(value);
//
//	return true;
//}
///*
//��ȡ�ؼ���ķ���
///*
//	�ɼ��ؼ������ڸ�˹������
//	1.���������ݶȷ���ͷ�ֵ
//	2.�����ݶȷ���ֱ��ͼ
//	3.ȷ�������㷽��
//	@pyrImg�������������߶���ӽ��ĸ�˹�������еĸ�˹ͼ��
//
//*/
//static float calculateGridentHist(cv::Mat &image,cv::Point pt,float scale,int n,float *hist)//�����ݶȷ���ֱ��ͼ
//{
//	int radius = cvRound(ORI_RADIUS*scale);//����������뾶(3*1.5*scale)
//	int len = (2 * radius + 1)*(2 * radius + 1);//���������������ܸ��������ֵ��
//	float sigma = ORI_SIG_FCTR * scale; //�����������˹Ȩ�ر�׼��(1.5*scale)
//	float exp_scale = -1.f / (2 * sigma*sigma);
//	//ʹ��AutoBuffer����һ���ڴ棬�������ĸ��ռ��Ŀ����Ϊ�˷������ƽ��ֱ��ͼ����Ҫ��
//	cv::AutoBuffer<float> buffer(4 * len + n + 4);
//	//X����ˮƽ��֣�Y������ֵ��֣�Mag�����ݶȷ��ȣ�Ori�����ݶȽǶȣ�W�����˹Ȩ��
//	float *X = buffer, *Y = buffer + len, *Mag = Y, *Ori = Y + len, *W = Ori + len;
//	float *temp_hist = W + len + 2;//��ʱ����ֱ��ͼ����
//	for (int i = 0; i < n; ++i)
//		temp_hist[i] = 0.f;//��������
//
//	//�����������ص�ˮƽ��ֺ���ֱ���
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
//	//�����������ص��ݶȷ���,�ݶȷ��򣬸�˹Ȩ��
//	for (int i = 0; i < k; i++)
//	{
//		W[i] = exp(W[i]);
//		Ori[i] = cv::fastAtan2(Y[i], X[i]);
//		Mag[i] = sqrt(X[i] * X[i] + Y[i] * Y[i]);
//	}
//	for (int i = 0; i < len; ++i)
//	{
//		int bin = cvRound((n / 360.f)*Ori[i]);//bin�ķ�ΧԼ����[0,(n-1)]
//		if (bin >= n)
//			bin = bin - n;
//		if (bin < 0)
//			bin = bin + n;
//		temp_hist[bin] = temp_hist[bin] + Mag[i] * W[i];
//	}
//
//	//ƽ��ֱ��ͼ
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
//	//���ֱ��ͼ�����ֵ
//	float max_value = hist[0];
//	for (int i = 1; i < n; ++i)
//	{
//		if (hist[i] > max_value)
//			max_value = hist[i];
//	}
//	return max_value;
//}
////�߶ȿռ伫ֵ����
//void MySift::findScaleSpaceExtremum(std::vector<std::vector<cv::Mat>> &pyr, std::vector<std::vector<cv::Mat>> &dogpyr, std::vector<cv::KeyPoint>&keypoints)//��ֵ����
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
//					if (abs(val)>threshold&&isExtremum(dogpyr, octave, layer, row, col))//��ɢ�ռ伫ֵ��&&��ֵ������ֵ
//					{
//						if (adjustLocExtermum(dogpyr, keypoint, octave, layer, row,col, nOctaveLayers, contrastThreshold,edgeThreshold,sigma))//���ͨ��
//						{
//							//count++;
//							//std::cout << keypoint.pt.x << " " << keypoint.pt.y << std::endl;
//							float scale = keypoint.size / powf(2.0, octave);//����������ڱ���ĳ߶ȣ�����
//							float max_hist = calculateGridentHist(pyr[octave][layer], cv::Point(col, row), scale, n, hist);
//							float mag_thr = max_hist * ORI_PEAK_RATIO;
//							//�����߲�ֵ
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
//									keypoint.angle = (360.f / n)*bin;//�������������0-360��
//									keypoints.push_back(keypoint);//�����������
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