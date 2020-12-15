clc;
clear all;
close all;
disp('  Load files  ');
load Realizations.out;
Dictionary=Realizations;
 load rossmary.GRDECL;
 load rossmaryporo.GRDECL;
usebig=zeros(36000,1500);
usebig(:,1:1499)=Dictionary(:,1:1499);
rossmary=reshape(rossmary,120,60,10);
rossmaryperm=rossmary(:,:,3:7);
rossmaryperm=reshape(rossmaryperm,36000,1);
usebig(:,1500)=rossmaryperm;
Dataporo=(0.00002.*Dictionary)+0.1785;
usebigporo=zeros(36000,1500);
usebigporo(:,1:1499)=Dataporo(:,1:1499);
rossmaryporo=reshape(rossmaryporo,120,60,10);
rossmaryporo2=rossmaryporo(:,:,3:7);
rossmaryporo2=reshape(rossmaryporo2,36000,1);
usebigporo(:,1500)=rossmaryporo2;
use=usebig;
disp('  truncate to binary representation  ');
use(use<50)=50;
use=log10(use);
% use(use<50)=0;
% use(use>0)=1;
% use(use==1)=5000;
% use(use==0)=50;

 useporo=usebigporo;
%  useporo(useporo<0.1805)=0;
%  useporo(useporo>0)=1;
%  useporo(useporo==1)=0.3;
%  useporo(useporo==0)=0.1;


disp('  Normalise the columns  ');
 
%  for i=1:1500
%  use(:,i)= use(:,i)/norm(use(:,i));
%  end
%  for i=1:1500
%  useporo(:,i)=useporo(:,i)/norm(useporo(:,i));
%  end
use=normcols(use);
useporo=normcols(useporo);
% 
  %jafarpourperm = repmat(joy,[1 3]);
  %jafarpourporo = repmat(joyporo,[1 3]);
  
  jafarpourperm=use;
  jafarpourporo=useporo;
% 
disp('  output the files  ');
 file2 = fopen('rolandperm.out','w+'); %output the dictionary
for k=1:numel(jafarpourperm)                                                                       
fprintf(file2,' %4.6f \n',jafarpourperm(k) );             
end
% % load project.mat
% 
 file1 = fopen('rolandporo.out','w+'); %output the dictionary
for k=1:numel(jafarpourporo)                                                                       
fprintf(file1,' %4.6f \n',jafarpourporo(k) );             
end
disp('  Programme ended  ');