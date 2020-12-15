
clc;
clear all;
close all;
addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ompbox10'))
%addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ksvdbox13'))

%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  April 2009


disp(' ');
disp('  **********  OMP code  **********');
disp(' ');
% disp('  This demo generates a random mixture of cosines and spikes, adds noise,');
% disp('  and uses OMP to recover the mixture and the original signal. The graphs');
% disp('  show the original, noisy and recovered signal with the corresponding SNR');
% disp('  values. The true and recovered coefficients are shown in the bar graphs.');
disp(' ');

load permdic.out;
load sgsim.out;
sgsim=reshape(sgsim,72000,100);
test=sgsim(14401:50400,1:100);
%X=sgsim(14401:50400,1:100);
X=test;
D=reshape(permdic,36000,9000);
% dad=D\X;
% dad(dad<=50)=50;
% error=X-D*dad;
 G=D'*D;
  T=900; %70 is the best
% %EPSILON=0.00001;
  gamma = omp(D,X,G,T);
%  %gamma=omp2(D,X,G,EPSILON);
  joy=D*gamma;
  e1 = sqrt( sum( (D*gamma-X).^2, 1 )/7200 );
  overalle1=sum(e1);
  joy(joy<=50)=50;
%  err = X-joy;
%  error=sum(err);
%  tunde=sum(error(:,1:100));
%  tunde=abs(tunde);
%  
% %  a=reshape(X,36000*2000,1);
% %  b=reshape(joy,36000*2000,1);
% %  plot(a,b);
% %  file4 = fopen('sgsim1.out','w+'); %print to a file
% %  [i,j,val] = find(gamma);
% %  data_dump = [i,j,val];
% % % %fprintf(file,' %4.4f \n',data_dump(:,3) );             
% %  for k=1:numel(data_dump);
% %  fprintf(file4,' %4.4f \n',data_dump(k,3) );             
% %  end
% 
% % show results %
% % 
% % v=[1 n -max(abs([x;y]))*1.1 max(abs([x;y]))*1.1];
% % figure; plot(x); axis(v); title('Original signal');
% % figure; plot(y); axis(v); title(sprintf('Noisy signal, SNR=%.1fdB', 10*log10((x'*x)/(r'*r))));
% % figure; plot(D*gamma); axis(v); title(sprintf('Reconstructed signal, SNR=%.1fdB', 10*log10((x'*x)/(err'*err))));
% % 
% % v = [1 2*n -max(abs([g;gamma]))*1.1 max(abs([g;gamma]))*1.1];
% % figure; bar(full(g)); axis(v); title('True signal decomposition');
% % figure; bar(full(gamma)); axis(v); title('Decomposition recovered by OMP');
% % 
% % return;
% 
% 
% % random matrix with +/-1
