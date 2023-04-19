# Measurement-of-key-parameters-of-wing-vibration-modulus
The project goal is to use binocular vision and non-contact measurement methods to identify modal parameters of large antennas, solar wings, etc. in orbit. 
The rar compressed file contains two folders. One folder has a chessboard for camera calibration, and the other folder has feature points, split into left and right images, for measurement purposes.

# 深空无人器翼展振动模量关键参数测量技术
rar的压缩文件包中包含了两个文件夹，一个文件夹中是棋盘，用来进行相机标定；另外一个文件夹是特征点，分为左右图像，进行测量用的。
课程训练项目要求（清晰写出本课程训练要做的内容和目标）
本项目目标：利用双目视觉非接触式测量手段实现大型天线、太阳翼等模态参数在轨辨识方法，具体在天线或太阳翼表面布设合作目标回光反射标志点作为测点，通过标志点提取技术及三角测量原理实现合作标记点处的振动位移信息计算。
为此我们做了简化研究，通过对黑白棋盘格格子的标定实现双目摄像机的标定，再通过对特征点的提取实现对振动位移信息的分析。




# 研究报告内容
# 一．对黑白棋盘格格子进行标定，从而实现对双目摄像机的标定
1、	原理
通过对两幅图像视差的计算，直接对前方景物（图像所拍摄到的范围）进行距离测量，而无需判断前方出现的是什么类型的障碍物。所以对于任何类型的障碍物，都能根据距离信息的变化，进行必要的预警或制动。双目摄像头的原理与人眼相似。人眼能够感知物体的远近，是由于两只眼睛对同一个物体呈现的图像存在差异，也称“视差”。物体距离越远，视差越小；反之，视差越大。视差的大小对应着物体与眼睛之间距离的远近，这也是3D电影能够使人有立体层次感知的原因。

![双目标定的原理图](../figure/figure1.png "figure1")
 
假设有一个点p，沿着垂直于相机中心连线方向上下移动，则其在左右相机上的成像点的位置会不断变化，即d=x1-x2的大小不断变化，并且点p和相机之间的距离Z跟视差d存在着反比关系。上式中视差d可以通过两个相机中心距T减去p点分别在左右图像上的投影点偏离中心点的值获得，所以只要获取到了两个相机的中心距T,就可以评估出p点距离相机的距离，这个中心距T也是双目标定中需要确立的参数之一。
2、过程
第1步：准备一张棋盘格，粘贴于墙面，并用直尺测量黑白方格的真实物理长度。
第2步：调用双目摄像头，分别从不同角度拍摄得到一系列棋盘格图像。
        前两步采用已有图像解决

第3步：采用工具箱TOOLBOX_calib，利用左目图片数据集，进行左目相机标定，得到左目内参矩阵K1、左目畸变系数向量D1，并点击save，得到mat文件。 


第4步：同上，利用右目图片数据集，进行右目相机标定，得到右目内参矩阵K2、右目畸变系数向量D2，

第5步：命令行输入stereo_gui，导入之前得到的文件夹，实现双目标定，点击show extrinsi可以显示照片与摄像头的关系图，以及双目摄像头之间的距离
 


下载工具箱并解压到toolbox文件夹，并在matlab中添加功能工具箱路径，在MATLAB命令行窗口输入calib_gui，弹出工具框，点击第一个选项输入图片前缀（right_或left_）和图片格式（本次采用jpg格式），之后选择第一行第三个选项并依次标定即可。

值得注意的是标定过程必须一次标定正确，如果有错误只能重新开始，所以需要比较谨慎。


3、用到的算法和代码
``` matlab

Calib_gui
Stereo_gui

``` 

4、结论、总结和拓展
在我们进行实验的时候发现一些缺陷和问题。
第一，	利用本次的标定工具箱，只要标错一个点就必须从头再来。工作量显著增加。
第二，	部分照片存在着较大的畸变，有些照片即使调整了Kc即一阶镜头畸变系数得到的结果仍然不够理想。为此我们尝试了大量的Kc值，发现部分图像确实利用一阶畸变系数无法校准。
于是我们开始另辟蹊径发现Matlab的机器视觉工具箱中正好有相机标定和双目相机的工具箱，并且我们尝试了应用。发现结果比标定工具箱要好得不少，但也出现了新的问题。
 


Matlab 的机器视觉工具箱能够很好地对单个相机进行标定，得到了近乎完美的结果，比标定工具箱好上不少，且实现了全自动化操作。
但是当我们期望利用两个相机分别的标定结果进行双目相机标定时出现了问题。具体来说，我们本来想的是利用机器视觉工具箱app的结果来进行双目标定，但是发现两个result的格式完全不一样无法使用。我们转而使用了matlab的双目标定app，但是这次出现了报错且无法解决。仔细分析后是发现我们本次标定的棋盘是23x23的正方形，而matlab的机器视觉app不能进行正方形棋盘的标定，因为有四个顶点都能成为原点o，于是产生了数组大小不一的错误，无法继续下去。所以我们最终结合了二者取其精华。

# 二、进行编码特征点提取并且利用双目相机标定的结果来计算三维空间真实坐标
1、特征点提取
1）特征点提取：利用imfindcircles函数，发现采用圆形Hough变换圆。
三维坐标测量：根据相机的标定结果对校正后的图像进行像素点匹配，之后根据匹配结果计算每个像素的深度，从而得到具体的三维坐标。
 
2）、过程
第一步： 对文件中图像进行读取处理；
第二步：利用imfindcircles函数为核心进行循环结构设计，同时打印出来圆中心的坐标结果；
 
第三步：对坐标结果进行处理分析，剔除无效坐标、缺失坐标；
 第四步：进行三维空间真实坐标的计算。

3）、用到的算法和代码
```
clear
zuobiao=zeros(126,2);
%%×ó±ßÏà»ú
for(m1=1:42)
m2=m1;
name=strcat('left_',num2str(m2),'.jpg');
A=imread(name);
 
 
 
MyYuanLaiPic = imread(name);%¶ÁÈ¡RGB¸ñÊ½µÄÍ¼Ïñ  
MyFirstGrayPic = rgb2gray(MyYuanLaiPic);%ÓÃÒÑÓÐµÄº¯Êý½øÐÐRGBµ½»Ò¶ÈÍ¼ÏñµÄ×ª»»
[rows , cols , colors] = size(MyYuanLaiPic);%µÃµ½Ô­À´Í¼ÏñµÄ¾ØÕóµÄ²ÎÊý  
MidGrayPic = zeros(rows , cols);%ÓÃµÃµ½µÄ²ÎÊý´´½¨Ò»¸öÈ«ÁãµÄ¾ØÕó£¬Õâ¸ö¾ØÕóÓÃÀ´´æ´¢ÓÃÏÂÃæµÄ·½·¨²úÉúµÄ»Ò¶ÈÍ¼Ïñ  
MidGrayPic = uint8(MidGrayPic);%½«´´½¨µÄÈ«Áã¾ØÕó×ª»¯Îªuint8¸ñÊ½£¬ÒòÎªÓÃÉÏÃæµÄÓï¾ä´´½¨Ö®ºóÍ¼ÏñÊÇdoubleÐÍµÄ  
   
for i = 1:rows  
    for j = 1:cols  
        sum = 0;  
        for k = 1:colors  
            sum = sum + MyYuanLaiPic(i , j , k) / 3;%½øÐÐ×ª»¯µÄ¹Ø¼ü¹«Ê½£¬sumÃ¿´Î¶¼ÒòÎªºóÃæµÄÊý×Ö¶ø²»ÄÜ³¬¹ý255  
        end  
        MidGrayPic(i , j) = sum;  
    end  
end  
imshow(MyYuanLaiPic); 
imshow(MidGrayPic);
Rmin = 6;
Rmax = 15;
[centersBright, radiiBright] = imfindcircles(MidGrayPic,[Rmin Rmax],'ObjectPolarity','bright');
[centersDark, radiiDark] = imfindcircles(MidGrayPic,[Rmin Rmax],'ObjectPolarity','dark');
viscircles(centersBright, radiiBright,'Color','b');
viscircles(centersDark, radiiDark,'LineStyle','--');
centersBright;
centersDark;
B1=3*(m2-1)+1;
B2=3*m2;
disp(centersDark)
disp(m2)
outname=strcat('out-left_',num2str(m2),'.jpg')
print(outname,'-dpng')
end
 
%%ÓÒ±ßÏà»ú
for(m1=1:42)
m2=m1;
name=strcat('left_',num2str(m2),'.jpg');
A=imread(name);
 
 
MyYuanLaiPic = imread(name);%¶ÁÈ¡RGB¸ñÊ½µÄÍ¼Ïñ  
MyFirstGrayPic = rgb2gray(MyYuanLaiPic);%ÓÃÒÑÓÐµÄº¯Êý½øÐÐRGBµ½»Ò¶ÈÍ¼ÏñµÄ×ª»»
[rows , cols , colors] = size(MyYuanLaiPic);%µÃµ½Ô­À´Í¼ÏñµÄ¾ØÕóµÄ²ÎÊý  
MidGrayPic = zeros(rows , cols);%ÓÃµÃµ½µÄ²ÎÊý´´½¨Ò»¸öÈ«ÁãµÄ¾ØÕó£¬Õâ¸ö¾ØÕóÓÃÀ´´æ´¢ÓÃÏÂÃæµÄ·½·¨²úÉúµÄ»Ò¶ÈÍ¼Ïñ  
MidGrayPic = uint8(MidGrayPic);%½«´´½¨µÄÈ«Áã¾ØÕó×ª»¯Îªuint8¸ñÊ½£¬ÒòÎªÓÃÉÏÃæµÄÓï¾ä´´½¨Ö®ºóÍ¼ÏñÊÇdoubleÐÍµÄ  
   
for i = 1:rows  
    for j = 1:cols  
        sum = 0;  
        for k = 1:colors  
            sum = sum + MyYuanLaiPic(i , j , k) / 3;%½øÐÐ×ª»¯µÄ¹Ø¼ü¹«Ê½£¬sumÃ¿´Î¶¼ÒòÎªºóÃæµÄÊý×Ö¶ø²»ÄÜ³¬¹ý255  
        end  
        MidGrayPic(i , j) = sum;  
    end  
end  
imshow(MyYuanLaiPic); 
imshow(MidGrayPic);
Rmin = 6;
Rmax = 15;
[centersBright, radiiBright] = imfindcircles(MidGrayPic,[Rmin Rmax],'ObjectPolarity','bright');
[centersDark, radiiDark] = imfindcircles(MidGrayPic,[Rmin Rmax],'ObjectPolarity','dark');
viscircles(centersBright, radiiBright,'Color','b');
viscircles(centersDark, radiiDark,'LineStyle','--');
centersBright;
centersDark;
B1=3*(m2-1)+1;
B2=3*m2;
disp(centersDark)
disp(m2)
outname=strcat('out-left_',num2str(m2),'.jpg')
print(outname,'-dpng')
end
```


```opencv部分：
Mat jiaozheng( Mat image )
{
    Size image_size = image.size();
    float intrinsic[3][3] = {589.2526583947847,0,321.8607532099886,0,585.7784771038199,251.0338528599469,0,0,1};
    float distortion[1][5] = {-0.5284205687061442, 0.3373615384253201, -0.002133029981628697, 0.001511983002864886, -0.1598661778309496};
    Mat intrinsic_matrix = Mat(3,3,CV_32FC1,intrinsic);
    Mat distortion_coeffs = Mat(1,5,CV_32FC1,distortion);
    Mat R = Mat::eye(3,3,CV_32F);       
    Mat mapx = Mat(image_size,CV_32FC1);
    Mat mapy = Mat(image_size,CV_32FC1);    
    initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
    Mat t = image.clone();
    cv::remap( image, t, mapx, mapy, INTER_LINEAR);
    return t;
}

//opencv2.4.9 vs2012
#include <opencv2\opencv.hpp>
#include <fstream>
#include <iostream>
 
using namespace std;
using namespace cv;
 
Point2f xyz2uv(Point3f worldPoint,float intrinsic[3][3],float translation[1][3],float rotation[3][3]);
Point3f uv2xyz(Point2f uvLeft,Point2f uvRight);
 
//图片对数量
int PicNum = 14;
 
//左相机内参数矩阵
float leftIntrinsic[3][3] = {4037.82450,			 0,		947.65449,
									  0,	3969.79038,		455.48718,
									  0,			 0,				1};
//左相机畸变系数
float leftDistortion[1][5] = {0.18962, -4.05566, -0.00510, 0.02895, 0};
//左相机旋转矩阵
float leftRotation[3][3] = {0.912333,		-0.211508,		 0.350590, 
							0.023249,		-0.828105,		-0.560091, 
							0.408789,		 0.519140,		-0.750590};
//左相机平移向量
float leftTranslation[1][3] = {-127.199992, 28.190639, 1471.356768};
 
//右相机内参数矩阵
float rightIntrinsic[3][3] = {3765.83307,			 0,		339.31958,
										0,	3808.08469,		660.05543,
										0,			 0,				1};
//右相机畸变系数
float rightDistortion[1][5] = {-0.24195, 5.97763, -0.02057, -0.01429, 0};
//右相机旋转矩阵
float rightRotation[3][3] = {-0.134947,		 0.989568,		-0.050442, 
							  0.752355,		 0.069205,		-0.655113, 
							 -0.644788,		-0.126356,		-0.753845};
//右相机平移向量
float rightTranslation[1][3] = {50.877397, -99.796492, 1507.312197};
 
 
int main()
{
	//已知空间坐标求成像坐标
	Point3f point(700,220,530);
	cout<<"左相机中坐标："<<endl;
	cout<<xyz2uv(point,leftIntrinsic,leftTranslation,leftRotation)<<endl;
	cout<<"右相机中坐标："<<endl;
	cout<<xyz2uv(point,rightIntrinsic,rightTranslation,rightRotation)<<endl;
 
	//已知左右相机成像坐标求空间坐标
	Point2f l = xyz2uv(point,leftIntrinsic,leftTranslation,leftRotation);
	Point2f r = xyz2uv(point,rightIntrinsic,rightTranslation,rightRotation);
	Point3f worldPoint;
	worldPoint = uv2xyz(l,r);
	cout<<"空间坐标为:"<<endl<<uv2xyz(l,r)<<endl;
 
	system("pause");
 
	return 0;
}
 
 
//************************************
// Description: 根据左右相机中成像坐标求解空间坐标
// Method:    uv2xyz
// FullName:  uv2xyz
// Access:    public 
// Parameter: Point2f uvLeft
// Parameter: Point2f uvRight
// Returns:   cv::Point3f
//************************************
Point3f uv2xyz(Point2f uvLeft,Point2f uvRight)
{
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	Mat mLeftRotation = Mat(3,3,CV_32F,leftRotation);
	Mat mLeftTranslation = Mat(3,1,CV_32F,leftTranslation);
	Mat mLeftRT = Mat(3,4,CV_32F);//左相机M矩阵
	hconcat(mLeftRotation,mLeftTranslation,mLeftRT);
	Mat mLeftIntrinsic = Mat(3,3,CV_32F,leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;
 
	Mat mRightRotation = Mat(3,3,CV_32F,rightRotation);
	Mat mRightTranslation = Mat(3,1,CV_32F,rightTranslation);
	Mat mRightRT = Mat(3,4,CV_32F);//右相机M矩阵
	hconcat(mRightRotation,mRightTranslation,mRightRT);
	Mat mRightIntrinsic = Mat(3,3,CV_32F,rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;
 
	//最小二乘法A矩阵
	Mat A = Mat(4,3,CV_32F);
	A.at<float>(0,0) = uvLeft.x * mLeftM.at<float>(2,0) - mLeftM.at<float>(0,0);
	A.at<float>(0,1) = uvLeft.x * mLeftM.at<float>(2,1) - mLeftM.at<float>(0,1);
	A.at<float>(0,2) = uvLeft.x * mLeftM.at<float>(2,2) - mLeftM.at<float>(0,2);
 
	A.at<float>(1,0) = uvLeft.y * mLeftM.at<float>(2,0) - mLeftM.at<float>(1,0);
	A.at<float>(1,1) = uvLeft.y * mLeftM.at<float>(2,1) - mLeftM.at<float>(1,1);
	A.at<float>(1,2) = uvLeft.y * mLeftM.at<float>(2,2) - mLeftM.at<float>(1,2);
 
	A.at<float>(2,0) = uvRight.x * mRightM.at<float>(2,0) - mRightM.at<float>(0,0);
	A.at<float>(2,1) = uvRight.x * mRightM.at<float>(2,1) - mRightM.at<float>(0,1);
	A.at<float>(2,2) = uvRight.x * mRightM.at<float>(2,2) - mRightM.at<float>(0,2);
 
	A.at<float>(3,0) = uvRight.y * mRightM.at<float>(2,0) - mRightM.at<float>(1,0);
	A.at<float>(3,1) = uvRight.y * mRightM.at<float>(2,1) - mRightM.at<float>(1,1);
	A.at<float>(3,2) = uvRight.y * mRightM.at<float>(2,2) - mRightM.at<float>(1,2);
 
	//最小二乘法B矩阵
	Mat B = Mat(4,1,CV_32F);
	B.at<float>(0,0) = mLeftM.at<float>(0,3) - uvLeft.x * mLeftM.at<float>(2,3);
	B.at<float>(1,0) = mLeftM.at<float>(1,3) - uvLeft.y * mLeftM.at<float>(2,3);
	B.at<float>(2,0) = mRightM.at<float>(0,3) - uvRight.x * mRightM.at<float>(2,3);
	B.at<float>(3,0) = mRightM.at<float>(1,3) - uvRight.y * mRightM.at<float>(2,3);
 
	Mat XYZ = Mat(3,1,CV_32F);
	//采用SVD最小二乘法求解XYZ
	solve(A,B,XYZ,DECOMP_SVD);
 
	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;
 
	//世界坐标系中坐标
	Point3f world;
	world.x = XYZ.at<float>(0,0);
	world.y = XYZ.at<float>(1,0);
	world.z = XYZ.at<float>(2,0);
 
	return world;
}
 
//************************************
// Description: 将世界坐标系中的点投影到左右相机成像坐标系中
// Method:    xyz2uv
// FullName:  xyz2uv
// Access:    public 
// Parameter: Point3f worldPoint
// Parameter: float intrinsic[3][3]
// Parameter: float translation[1][3]
// Parameter: float rotation[3][3]
// Returns:   cv::Point2f
// Author:    小白
// Date:      2017/01/10
// History:
//************************************
Point2f xyz2uv(Point3f worldPoint,float intrinsic[3][3],float translation[1][3],float rotation[3][3])
{
	//    [fx s x0]							[Xc]		[Xw]		[u]	  1		[Xc]
	//K = |0 fy y0|       TEMP = [R T]		|Yc| = TEMP*|Yw|		| | = —*K *|Yc|
	//    [ 0 0 1 ]							[Zc]		|Zw|		[v]	  Zc	[Zc]
	//													[1 ]
	Point3f c;
	c.x = rotation[0][0]*worldPoint.x + rotation[0][1]*worldPoint.y + rotation[0][2]*worldPoint.z + translation[0][0]*1;
	c.y = rotation[1][0]*worldPoint.x + rotation[1][1]*worldPoint.y + rotation[1][2]*worldPoint.z + translation[0][1]*1;
	c.z = rotation[2][0]*worldPoint.x + rotation[2][1]*worldPoint.y + rotation[2][2]*worldPoint.z + translation[0][2]*1;
 
	Point2f uv;
	uv.x = (intrinsic[0][0]*c.x + intrinsic[0][1]*c.y + intrinsic[0][2]*c.z)/c.z;
	uv.y = (intrinsic[1][0]*c.x + intrinsic[1][1]*c.y + intrinsic[1][2]*c.z)/c.z;
 
	return uv;
}
```


2、	计算对应特征点的三维坐标
通过查阅题目中给出的” stereo_triangulation”帮助文档可以知道，该函数内部参数主要有
输入部分：
xL:左图像素坐标的2xN矩阵
xR:右侧图像像素坐标的2xN矩阵
om,T:左右相机之间的旋转矢量和平移矢量(立体标定输出)
fc_left, cc_left,…:左相机固有参数(立体标定输出)
fc_right, cc_right,…:右侧相机的固有参数(立体标定输出)
输出部分：
XL:左侧相机参考系中点坐标的3xN矩阵 
XR:右相机参考系中点坐标的3xN矩阵
在第一问中通过stereo_gui工具箱，我们可以得出相机的内参矩阵，将得到的内参矩阵带入函数，同时将提取的左右特征点坐标分别用矩阵xL和矩阵xR表示，代码如下：
```
clc;
clear;
xL = readtable ('xL.txt');
xL = table2array(xL)
xR = readtable('xR.txt');
xR = table2array(xR)
load('Calib_Results_stereo.mat');
[XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
同时我们对其中内置的“stereo_triangulation”再次进行分析，并做了部分注释：
function [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right),

% [XL,XR] = stereo_triangulation(xL,xR,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right),
%
% Function that computes the position of a set on N points given the left and right image projections.
% The cameras are assumed to be calibrated, intrinsically, and extrinsically.
%
% Input:
%           xL: 2xN matrix of pixel coordinates in the left image
%           xR: 2xN matrix of pixel coordinates in the right image
%           om,T: rotation vector and translation vector between right and left cameras (output of stereo calibration)
%           fc_left,cc_left,...: intrinsic parameters of the left camera  (output of stereo calibration)
%           fc_right,cc_right,...: intrinsic parameters of the right camera (output of stereo calibration)
%
% Output:
%
%           XL: 3xN matrix of coordinates of the points in the left camera reference frame
%           XR: 3xN matrix of coordinates of the points in the right camera reference frame
%
% Note: XR and XL are related to each other through the rigid motion equation: XR = R * XL + T, where R = rodrigues(om)
% For more information, visit http://www.vision.caltech.edu/bouguetj/calib_doc/htmls/example5.html
%
%
% (c) Jean-Yves Bouguet - Intel Corporation - April 9th, 2003



%--- Normalize the image projection according to the intrinsic parameters of the left and right cameras
%相机标定内参中的γ (alpha_c)若为零，表示像素坐标系u、v轴成直角，若不为零像素坐标系u、v轴不成直角
%在引用alpha_c计算时，需要注意alpha_c=alpha_c/fc(1),即 γ=γ/fx

alpha_c_left = alpha_c_left/fc_left(1);
alpha_c_right = alpha_c_right/fc_right(1);

%option 1 Common camera model 没有径向畸变K3,即K3=0 
xt = normalize_pixel(xL,fc_left,cc_left,kc_left,alpha_c_left);
xtt = normalize_pixel(xR,fc_right,cc_right,kc_right,alpha_c_right);
%option 2 Special camera model 带有有径向畸变K3 
% xt = normalize_pixel_fisheye(xL,fc_left,cc_left,kc_left,alpha_c_left);
% xtt = normalize_pixel_fisheye(xR,fc_right,cc_right,kc_right,alpha_c_right);

%--- Extend the normalized projections in homogeneous coordinates
xt = [xt;ones(1,size(xt,2))];
xtt = [xtt;ones(1,size(xtt,2))];

%--- Number of points:
N = size(xt,2);

%--- Rotation matrix corresponding to the rigid motion between left and right cameras:
R = rodrigues(om);


%--- Triangulation of the rays in 3D space:

u = R * xt;

n_xt2 = dot(xt,xt);
n_xtt2 = dot(xtt,xtt);

T_vect = repmat(T, [1 N]);

DD = n_xt2 .* n_xtt2 - dot(u,xtt).^2;

dot_uT = dot(u,T_vect);
dot_xttT = dot(xtt,T_vect);
dot_xttu = dot(u,xtt);

NN1 = dot_xttu.*dot_xttT - n_xtt2 .* dot_uT;
NN2 = n_xt2.*dot_xttT - dot_uT.*dot_xttu;

Zt = NN1./DD;
Ztt = NN2./DD;

X1 = xt .* repmat(Zt,[3 1]);
X2 = R'*(xtt.*repmat(Ztt,[3,1])  - T_vect);


%--- Left coordinates:
XL = 1/2 * (X1 + X2);

%--- Right coordinates:
XR = R*XL + T_vect;
```
以上就是代码部分的内容，最终得到输出
XL:左侧相机参考系中点坐标的3xN矩阵；
XR:右相机参考系中点坐标的3xN矩阵；


4）、结论、总结和拓展
在进行第二部分的时候是最为困难且遇到最大问题的部分。同时有一些总结和拓展，具体来说有如下一些部分：
1.	圆提取不全且容易提取到边缘目标：imfindcircles函数掌握不精，理论上来说我们能通过优化参数来进行更为细致的提取，但是时间有些且经验不足走了很多弯路；
2.	圆提取的时候，根据其他组的成功经验尝试了surf算法，但是不熟悉，不过也拓展了思路，但是surf算法存在以下的缺点：第一，圆提取过多无法进行有效筛选，具体表现为三个特征点的都会提取出众多坐标，且调参的效果不好；第二，无法实现自动化，必须手动提取，因为surf算法提取的圆很多，需要手动一张一张图片进行手动三个点分别提取，工作量巨大，84张图片总共252个点，只能手动记录，效率低下；
3.	在计算三维坐标的时候没有发现calib_toolbox的自带的三维坐标计算，所以利用了opencv的三维坐标计算代码。之后跑通了之后又用回了matlab的calib——toolbox工具箱的算法
4.	圆提取可以利用matlab来调用OpenCV来进行，我们借鉴了别人成功的思路，利用opencv复现了Arc-support Line Segments Revisited: An Efficient High-quality Ellipse Detection的椭圆检测算法，可做了尝试后发现效果不好后放弃。

