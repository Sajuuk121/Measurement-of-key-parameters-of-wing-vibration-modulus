clear
M1=[1 0 0 0;0 1 0 0;0 0 1 0];
R=[0.993608682374763	0.000438553061203568	0.112878669290080;
-0.00105179960879089	0.999985010580374	0.00537329807021926;
-0.112874620828022	-0.00545768135575756	0.993594250027218
];
T=[-118.300952317316
1.29106386036938
-0.326848184175309
];
p1=[802.0465;269.3119;1];
p2=[569.0000;267.8220;1];
p1x=[0 -p1(3,1) p1(2,1);
    p1(3,1) 0 -p1(1,1);
    -p1(2,1) p1(1,1) 0];
p2x=[0 -p2(3,1) p2(2,1);
    p2(3,1) 0 -p2(1,1);
    -p2(2,1) p2(1,1) 0];
Hresult_c2_c1=[0.993608682	0.000438553	0.112878669	-118.3009523;
-0.0010518	0.999985011	0.005373298	1.29106386;
-0.112874621	-0.005457681	0.99359425	-0.326848184;
0	0	0	1]

for i=1:4
    k=i
    Hresult_c1_c2=inv(Hresult_c2_c1(:,:,k));
    M2=Hresult_c1_c2(1:3,1:4);
    A=[p1x*M1;p2x*M2];
    [U,D,V]=svd(A);
    P=V(:,4);
    P1est=P/P(4);
    P2est=Hresult_c1_c2*P1est;
    if P1est(3)>0 && P2est(3)>0
        Hest_c2_c1=Hresult_c2_c1(:,:,k);
        break;
    end
end
