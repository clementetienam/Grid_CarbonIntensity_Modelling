load Yes2.out;
load sgsim.out;
sgsim=reshape(sgsim,72000,100);
test=sgsim(21601:28800,1:100);
%X=sgsim(14401:50400,1:100);
X=test;
b=X;
D=reshape(Yes2,7200,2100);
A=D;
H=50;
[y] = AOmp(H,A,b);