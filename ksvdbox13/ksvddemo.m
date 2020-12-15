%Clement etienam
% PhD supervisor: Dr Rossmary Villegas
%Co-supervisor: Dr Masoud Babei
%KSVDDEMO K-SVD dictionary learning.
%
clc;
clear all;
close all;
disp('add the matlab paths ');
%  See also KSVDDENOISEDEMO.
addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ompbox10'))
addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM answers SPE 10\ksvdbox13'))
%addpath(genpath('C:\Work\GSLIB\sgsim\ETIENAM SPE 10 SPARSITY'))



%   load Dperm.out;
%   load Dporo.out;
%    load Dpermsigned.out;
%    load Dporosigned.out;
%Dataperm=reshape(abs(Dperm),36000,2000);
%Dataporo=(0.000203147.*X)+0.0841322;
% Dataporo=reshape(abs(Dporo),36000,2000);
% Dpermsigned=reshape( Dpermsigned,36000,2000);
% Dporosigned=reshape( Dporosigned,36000,2000);
load dctrealizations.out;
Dataperm=reshape(dctrealizations,9000,2000);
X=(Dataperm);

disp(' ');
disp('  **********  K-SVD Dictionary learning  **********');
%[m,n] = size(Dataporo);
% dictionary dimensions




%% generate normalized dictionary and data %%

 X = normcols(X);
 %Dataporo=normcols(Dataporo);
% Dpermsigned=normcols(Dpermsigned);
% Dporosigned=normcols(Dporosigned);



%% run k-svd training %%
disp('  permeability KSVD training  ');
 params.data = X;
%Edata=10;
params.Tdata=30;
%params.Tdata = k;
params.dictsize = 1500;
params.iternum = 50;
params.memusage = 'high';
params.codemode='sparsity';
[Dksvd,g,err] = ksvd(params,'tr');

errperm = X-Dksvd*g;
sparseperm=full(g);

% disp('  porosity KSVD training  ');
% params.dictsize = 1500;
% params.iternum = 100;
% params.memusage = 'high';
% params.codemode='sparsity';
% params.data = Dataporo;
% params.Tdata=30;
% [Dksvd2,g2,err2] = ksvd(params,'tr');
% errporo = Dataporo-Dksvd2*g2;
% sparseporo=full(g2);
% 
% disp('  permeability signed distance  KSVD training  ');
%  params.data = Dpermsigned;
% %Edata=10;
% params.Tdata=30;
% %params.Tdata = k;
% params.dictsize = 1500;
% params.iternum = 100;
% params.memusage = 'high';
% params.codemode='sparsity';
% [Dksvds,gs,errs] = ksvd(params,'tr');
% 
% errpermsigned = Dpermsigned-Dksvds*gs;
% sparsepermsigned=full(gs);
% 
% disp('  porosity signed distance  KSVD training  ');
%  params.data = Dporosigned;
% %Edata=10;
% params.Tdata=30;
% %params.Tdata = k;
% params.dictsize = 1500;
% params.iternum = 100;
% params.memusage = 'high';
% params.codemode='sparsity';
% [Dksvdp,gp,errp] = ksvd(params,'tr');
% 
% errporosigned = Dporosigned-Dksvdp*gp;
% sparseporosigned=full(gp);
%% show results %%

figure;
%subplot(2,2,1)
plot(err);
title('K-SVD error convergence of permeability','FontName','Helvetica', 'Fontsize', 13);
xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13); 
ylabel('RMSE','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')


% subplot(2,2,2)
% plot(errs);
% title('K-SVD error convergence of permeability LS','FontName','Helvetica', 'Fontsize', 13);
% xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13); 
% ylabel('RMSE','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% 
% 
% subplot(2,2,4)
% plot(errp);
% title('K-SVD error convergence of porosity LS','FontName','Helvetica', 'Fontsize', 13);
% xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13); 
% ylabel('RMSE','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(2,2,3)
% plot(err2);
% title('K-SVD error convergence of porosity','FontName','Helvetica', 'Fontsize', 13);
% xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13); 
% ylabel('RMSE','FontName','Helvetica', 'Fontsize', 13);
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')

file = fopen('Yes2machine.out','w+'); %output the dictionary
for k=1:numel(Dksvd)                                                                       
fprintf(file,' %4.6f \n',Dksvd(k) );             
end

% file2 = fopen('Yes2poro.out','w+'); %output the dictionary
% for k=1:numel(Dksvd2)                                                                       
% fprintf(file2,' %4.4f \n',Dksvd2(k) );             
% end
% 
% file3 = fopen('Yes2signed.out','w+'); %output the sparse coefficients
% for k=1:numel(Dksvds)                                                                      
% fprintf(file3,' %4.4f \n',Dksvds(k) );             
% end
% file4 = fopen('Yes2signedporo.out','w+'); %output the sparse coefficients
% for k=1:numel(Dksvdp)                                                                       
% fprintf(file4,' %4.4f \n',Dksvdp(k) );             
% end