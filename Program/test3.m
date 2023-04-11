clear
zuobiao=zeros(126,2);
%%左边相机
for(m1=1:42)
m2=m1;
name=strcat('left_',num2str(m2),'.jpg');
A=imread(name);



MyYuanLaiPic = imread(name);%读取RGB格式的图像  
MyFirstGrayPic = rgb2gray(MyYuanLaiPic);%用已有的函数进行RGB到灰度图像的转换
[rows , cols , colors] = size(MyYuanLaiPic);%得到原来图像的矩阵的参数  
MidGrayPic = zeros(rows , cols);%用得到的参数创建一个全零的矩阵，这个矩阵用来存储用下面的方法产生的灰度图像  
MidGrayPic = uint8(MidGrayPic);%将创建的全零矩阵转化为uint8格式，因为用上面的语句创建之后图像是double型的  
   
for i = 1:rows  
    for j = 1:cols  
        sum = 0;  
        for k = 1:colors  
            sum = sum + MyYuanLaiPic(i , j , k) / 3;%进行转化的关键公式，sum每次都因为后面的数字而不能超过255  
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

%%右边相机
for(m1=1:42)
m2=m1;
name=strcat('left_',num2str(m2),'.jpg');
A=imread(name);




MyYuanLaiPic = imread(name);%读取RGB格式的图像  
MyFirstGrayPic = rgb2gray(MyYuanLaiPic);%用已有的函数进行RGB到灰度图像的转换
[rows , cols , colors] = size(MyYuanLaiPic);%得到原来图像的矩阵的参数  
MidGrayPic = zeros(rows , cols);%用得到的参数创建一个全零的矩阵，这个矩阵用来存储用下面的方法产生的灰度图像  
MidGrayPic = uint8(MidGrayPic);%将创建的全零矩阵转化为uint8格式，因为用上面的语句创建之后图像是double型的  
   
for i = 1:rows  
    for j = 1:cols  
        sum = 0;  
        for k = 1:colors  
            sum = sum + MyYuanLaiPic(i , j , k) / 3;%进行转化的关键公式，sum每次都因为后面的数字而不能超过255  
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
