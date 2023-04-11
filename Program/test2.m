clear

im = imread('.\pics\left_15.jpg');
    minsize = [1,3,5,7];
    for k = 1:4
        % 找到所有椭圆
        [ellipses, L, posi] = ellipseDetectionByArcSupportLSs(im, 100, 0.01, -1);
        % 找到尺寸较小的
        b = ellipses(:,4); % 短轴长
        idx = find(b<minsize(k));
        % 剔除
        ellipses(idx,:) = [];
        % 显示
        subplot(2,2,k)
        drawEllipses(ellipses',im); % 为了使用方便进行了修改，可自定义颜色
        title(['\fontsize{14}minsize = ',num2str(minsize(k))])
    end 
