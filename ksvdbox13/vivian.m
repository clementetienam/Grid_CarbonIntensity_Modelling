clc;
clear all;
close all;
addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ompbox10'))
disp(' ');
disp('  **********  K-SVD Denoising Demo  **********');
disp(' ');
N=100;
load sharon.out;
sgsim=reshape(sharon,72000,100);
for i=1:N
sgsimuse=reshape(sgsim(:,i),120,60,10);
sgs=sgsimuse(:,:,3:7);
ex=reshape(sgs,36000,1);
sg(:,i)=ex;
end
sigma=20;
unie=sg(1:1500,:);
load Yes2.out;
Yes2=reshape(Yes2,36000,1500);
dic=Yes2(:,1:100);
%% set parameters %%

params.x = unie;
params.blocksize = 15;
params.dict = Yes2;
params.sigma = sigma;
params.maxval = 20000;
%params.trainnum = 40000;
%params.iternum = 20;
params.memusage = 'high';



% denoise!

[y,nz] = ompdenoise(params,'tr');





