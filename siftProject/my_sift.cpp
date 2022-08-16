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
	cv::Mat float_image, temp_image;//float_imageΪ��֮��ĸ�˹
	gray_image.convertTo(float_image, CV_32FC1, 1.0 / 255.0);
	cv::resize(float_image, temp_image, cv::Size(2 * float_image.cols, 2 * float_image.rows), 0, 0, cv::INTER_LINEAR);
	double sigma_diff = sqrt(sigma*sigma - 2 * INIT_SIGMA * 2 * INIT_SIGMA);
	cv::GaussianBlur(temp_image, BaseImg, cv::Size(0, 0), sigma_diff, sigma_diff);

}
void MySift::buildGaussianPyramid(const cv::Mat &BaseImg, std::vector<std::vector<cv::Mat>> &pyr, int nOctaves)
{
	//������˹�˲���
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
	pyr.resize(nOctaves, std::vector<cv::Mat>(nOctaveLayers + 3));//��㡢�ڲ�
		//���ָ�������ڲ������Ĵ�С���Ϳ���operator[]���ж���д��
		//���ָֻ�����������С��������push_back()��������д��������operator[]���ж���д��
	for (int i = 0; i < nOctaves; i++)
	{
		for (int j = 0; j < nOctaveLayers + 3; j++)
		{
			if (i == 0 && j == 0)//��һ���һ��
				pyr[i][j] = BaseImg;
			else if (j == 0) {//�ǵ�һ���һ��
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
static bool isExtremum(std::vector<std::vector<cv::Mat>> &dogpyr, int o, int l, int r, int c,float threshold)//��ֵ�������⣬������������Ĳ�ֽ�����
{
	float val = dogpyr[o][l].ptr<float>(r)[c];
	if (abs(val) > threshold)
	{
		if (val > 0)
		{
			for (int i = -1; i <= 1; i++)//��
			{
				for (int j = -1; j <= 1; j++)//��
				{
					for (int k = -1; k <= 1; k++)//��
					{
						if (val < dogpyr[o][l + i].ptr<float>(r + j)[c + k])
							return false;
					}
				}
			}
		}
		else
		{
			for (int i = -1; i <= 1; i++)//��
			{
				for (int j = -1; j <= 1; j++)//��
				{
					for (int k = -1; k <= 1; k++)//��
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
�����ز�ֵ��������ϣ�������֪����ɢ�ռ�ĵ��ֵ�õ������ռ�ļ�ֵ��
������Ϊȡֵ������ɢ��Ϊ��ģ�����Զ���仯��������Ҫ�������Ŀռ��ҵ���ֵ��
@dogpyr:��˹��ֽ�����
@kpt:�ؼ���
------------------------
��֮ǰ����Ѱ�Ҽ�ֵ�õ��ļ�ֵ�����ڵ��顢�㡢�С���
@octave:��
@layer:��
@row:��
@col:��
------------------------
@contrastThreshold:�Ա���ֵ0.04
@edgeThreshold:��Ե��ֵ10
@sigma:��˹�߶ȿռ���ײ�ͼ��߶�1.6
*/
static bool adjustLocExtermum(const std::vector<std::vector<cv::Mat>> &dogpyr,
	cv::KeyPoint &kpt, int octave, int &layer, int &row, int &col, int nOctaveLayers,
	float contrastThreshold, float edgeThreshold, float sigma)
{
	float xr, xc, xl;
	int num = 0;
	for (; num < MAX_INTERP_STEPS; num++)//����������
	{
		cv::Mat img = dogpyr[octave][layer];
		cv::Mat pre = dogpyr[octave][layer - 1];
		cv::Mat nex = dogpyr[octave][layer + 1];

		//���޲����x,y,�ҷ���һ��ƫ��
		float dx = (img.ptr<float>(row)[col + 1] - img.ptr<float>(row)[col - 1]) / 2.0;
		float dy = (img.ptr<float>(row + 1)[col] - img.ptr<float>(row - 1)[col]) / 2.0;
		float dz = (nex.ptr<float>(row)[col] - pre.ptr<float>(row)[col]) / 2.0;
		//����ƫ��
		float val = img.ptr<float>(row)[col];
		float dxx = (img.ptr<float>(row)[col + 1] + img.ptr<float>(row)[col - 1]) - 2 * val;
		float dyy = (img.ptr<float>(row + 1)[col] + img.ptr<float>(row - 1)[col]) - 2 * val;
		float dzz = (nex.ptr<float>(row)[col] + pre.ptr<float>(row)[col]) - 2 * val;

		//��϶���ƫ��
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
		cv::solve(H, dD, X_, cv::DECOMP_LU);//������Է�����Ľ�
		//cv::Vec3f X_ = (H.inv()*dD);
		xc = -X_[0];//x����ƫ����
		xr = -X_[1];//y����ƫ����
		xl = -X_[2];//�ҷ���ƫ����

		if (abs(xc) < 0.5f && abs(xr) < 0.5f && abs(xl) < 0.5f)//ƫ��С��0.5����ֵ����ȷ
			break;
		col = col + cvRound(xc);
		row = row + cvRound(xr);
		layer = layer + cvRound(xl);

		if (layer<1 || layer>nOctaveLayers || col<IMG_BORDER || col>img.cols - IMG_BORDER
			|| row<IMG_BORDER || row>img.rows - IMG_BORDER)//�ؼ����ڱ߽�����ɾ��
			return false;//��ʾ��ֵ��Խ��
	}
	if (num >= MAX_INTERP_STEPS)//���ڵ�������ɾ��
		return false;
	//------------------------�����ͶԱȶȵ�Ҳ���Ǽ�ֵС��CONTR_THR / nOctaveLayers; CONTR_THR=0.04
	//��Ҫ���¼���������
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
		return false;//ȥ���ͶԱȶȵĵ�
	//------------------ȥ����Ե��Ӧ��---------------
	float val = image.ptr<float>(row)[col];//��hessian��������ʽֵ�ͼ�
	float dxx = (image.ptr<float>(row)[col + 1] + image.ptr<float>(row)[col - 1]) - 2 * val;
	float dyy = (image.ptr<float>(row + 1)[col] + image.ptr<float>(row - 1)[col]) - 2 * val;
	float dxy = ((image.ptr<float>(row + 1)[col + 1] + image.ptr<float>(row - 1)[col - 1]) -
		(image.ptr<float>(row + 1)[col - 1] + image.ptr<float>(row - 1)[col + 1])) / 4.0;
	float det = dxx * dyy - dxy * dxy;
	float trace = dxx + dyy;
	if (det < 0 || (trace * trace*edgeThreshold >= det * (edgeThreshold + 1)*(edgeThreshold + 1)))
		return false;
	//ͨ���Աȶȼ��ͱ�Ե��Ӧ������keypoint
	//kpt.pt = cv::Point2f((float)(col + xc)*powf(2.0, octave)), ((float)row + xr)*powf(2.0, octave)));
	kpt.pt.x = ((float)col + xc)*(1 << octave);//*powf(2.0, octave);//��ײ�ͼ��x����
	kpt.pt.y = ((float)row + xr)*(1 << octave);//*powf(2.0, octave);//y����
	kpt.octave = octave + (layer << 8);//��ű����ڵ��ֽڣ���ű����ڸ��ֽ�layer<<8==layer*2^8
	kpt.size = sigma * powf(2.f, (layer + xl) / nOctaveLayers)*(1 << octave);//*powf(2.0, octave);
	kpt.response = abs(value);
	return true;
}
static double* calculateHist(cv::Mat pyrImg,cv::Point pt,int radius,float sigma)//�����ݶ�ֱ��ͼ
{
	double *hist = new double[ORI_HIST_BINS];
	//ÿ���������
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
				ori = cv::fastAtan2(dy, dx);//���ص��ǽǶȣ���0-360��֮��
				weight = exp((i*i + j * j)*exp_scale);
				bin = cvRound(ORI_HIST_BINS / 360.f*ori);//Լ����0-36֮��
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
//�ؼ��㷽�����
static void computeKeypointsOrientations(cv::Mat pyrImg,std::vector<cv::KeyPoint> &keypoints,cv::KeyPoint &keypoint,int octave,cv::Point pt)
{
	double *hist = new double[ORI_HIST_BINS];
	////ÿ���������
	//for (int k = 0; k < ORI_HIST_BINS; k++)
	//{
	//	hist[k] = 0.f;
	//}
	float scale = keypoint.size/powf(2.0, octave);
	float sigma = ORI_SIG_FCTR * scale;//�����������˹Ȩ�ر�׼��(1.5*scale)
	int radius = cvRound(ORI_RADIUS*scale);//3*1.5*��
	int len = (2 * radius + 1)*(2 * radius + 1);
	//�����ݶȷ���ֱ��ͼ
	/*-----------------1.�����ݶ�ֱ��ͼ----------------------*/
	hist = calculateHist(pyrImg, pt, radius, sigma);
	/*for (int i = 0; i < ORI_HIST_BINS; i++)
	{
		std::cout << hist[i] << " ";
	}
	std::cout << "---------------" << std::endl;*/
	//---------------------------------------------------------
	//2.ƽ��ֱ��ͼ
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
	//3.ȷ���ؼ���������
	int maxi;//����������ֵ������
	double maxhist = DominantDirection(shist, ORI_HIST_BINS,maxi);//���ƽ�����ֱ��ͼ������������ֵ
	//keypoint.angle = 360.f / ORI_HIST_BINS * maxi;
	//keypoints.push_back(keypoint);
	/*4.����ֱ��ͼ�ķ�ֵ������˸�������ķ�����ֱ��ͼ�е����ֵ��Ϊ�ùؼ����������Ϊ����ǿƥ���³���ԣ�ֻ������ֵ������
	�����ֵ80%�ķ�����Ϊ�Ĺؼ���ĸ�������ˣ�����ͬһ�ݶ�ֵ�ö����ֵ�Ĺؼ���λ�ã�����ͬλ�úͳ߶Ƚ����ж���ؼ��㱻
	����������ͬ������15%�Ĺؼ��㱻���������򣬵��ǿ������Ե���߹ؼ�����ȶ���*/
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
			keypoint.angle = 360.f / ORI_HIST_BINS * bin;//�����㸨����
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
					if (isExtremum(dogpyr, octave, layer, row, col, threshold))//26����������ɢ�ռ伫ֵ��
					{
						if (adjustLocExtermum(dogpyr, keypoint, octave, layer, row, col, nOctaveLayers, contrastThreshold, edgeThreshold, sigma))
						{
							//����������е㶼�ǹؼ��㣬��������ȷ���ؼ��㷽��//
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