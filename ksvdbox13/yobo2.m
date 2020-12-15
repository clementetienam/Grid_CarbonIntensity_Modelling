%Clement etienam
% PhD supervisor: Dr Rossmary Villegas
%Co-supervisor: Dr Masoud Babei
%KSVDDEMO K-SVD dictionary learning.
%
clc;
clear;
close all;
disp('add the matlab paths ');
%  See also KSVDDENOISEDEMO.
addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ompbox10'))
addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ksvdbox13'))
%addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM SPE 10 SPARSITY'))



load Realizations.out;
poro=(0.00002.*Realizations)+0.1785;

clement2 = DCTsigned(Realizations, 2000);

Realizations=normcols((clement2));
yes2=reshape(Realizations,9000,2000);

dicperm=yes2(:,1:1500);
clement2poro = DCTsigned(poro, 2000);
poro=normcols(clement2poro);
dicporo=poro(:,1:1500);
  file1 = fopen('Yes2m.out','w+'); 
 for k=1:numel(dicperm)                                                                       
 fprintf(file1,' %4.6f \n',dicperm(k) );             
 end
   file2 = fopen('Yes2porom.out','w+'); 
 for k=1:numel(dicporo)                                                                       
 fprintf(file2,' %4.6f \n',dicporo(k) );             
 end



% load dctrealizations.out;
% Dataperm=reshape(dctrealizations,9000,2000);
% X=(Dataperm);
% 
% disp(' ');
% disp('  **********  K-SVD Dictionary learning  **********');
% %[m,n] = size(Dataporo);
% % dictionary dimensions
% 
% 
% 
% 
% %% generate normalized dictionary and data %%
% 
%  X = normcols(X);