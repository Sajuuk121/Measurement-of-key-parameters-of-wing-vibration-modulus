%读取一张图片，并显示
original_picture=imread('.\pics\left_1.jpg');
figure(1);
imshow(original_picture);
title('原始RGB图像')

%把图像转换成灰度图像
GrayPic=rgb2gray(original_picture);%把RGB图像转化成灰度图像
figure(2)
imshow(GrayPic);
title('RGB图像转化为灰度图像')

%对图像进行二值化处理
thresh=graythresh(original_picture);%graythresh为自动确定二值化阈值函数，大于该阈值的就变成白色，小于该阈值的就变成黑色，所以阈值越大越黑，阈值越小越白
Pic2=im2bw(original_picture,thresh);%如果想要自己设定阈值，那么就可以这样写Pic2=im2bw(original_picture,value);,value=[0,1]中间的任何数值
figure(3);
imshow(Pic2);
title('RGB图像转化为二值化图像')

thresh=graythresh(GrayPic);
Pic2_=im2bw(GrayPic,thresh);
figure(4);
imshow(Pic2_);
title('灰度图像转化为二值化图像')