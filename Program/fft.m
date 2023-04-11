clear
            		%输入时域序列向量
yn=rand(20);
xn=[1 2 1 4];
Xk16 = fft(xn, 1);         		% 计算xn的1点fft
Xk32 = fft(xn, 1500);         		% 计算xn的32点fft

% 以下为绘图部分
k =0; 
wk = 2*k/1;            			%计算200点DFT对应的采样点频率
subplot(1,2,1);     
stem(wk, abs(Xk16), '.');      		%绘制200点DFT的幅频特性图
title('（a）200点DFT的幅频特性图');  
 xlabel('w/π');    
 ylabel(' 幅度 ');

subplot(1,2,2);     
stem(wk, angle(Xk16), '.');     	%绘制200点DFT的相频特性图
line([0,2], [0,0]);     
title('（b）200点DFT的相频特性图');
xlabel('w/π');    
ylabel(' 相位 ');

