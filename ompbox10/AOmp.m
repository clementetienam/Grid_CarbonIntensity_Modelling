function [y] = AOmp(H,A,b);
% H:number of iterations, A: matrix, b: signal
n=size(A);
A1=zeros(n);
R=b;
if(H<=0)
error('The number of iterations needs to be greater then 0')
end;
for k=1:1:H,
[c,d] = max(abs(A'*R));
A1(:,d)=A(:,d);
A(:,d)=0;
y = A1 \ b;
R = b-A1*y;
end;
