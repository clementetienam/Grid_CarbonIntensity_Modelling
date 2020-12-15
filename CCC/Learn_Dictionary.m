function [Dksvd,g]=Learn_Dictionary(Tdata,X,dictsize,iterr,oldfolder,folder,ii)
%X = normcols(X);

%% run k-svd training %%
disp(' KSVD training  ');
 params.data = X;
%Edata=10;
params.Tdata=Tdata;
params.dictsize = dictsize;
params.iternum = iterr;
params.memusage = 'high';
params.codemode='sparsity';
[Dksvd,g,err] = ksvd(params,'tr');

errperm = X-Dksvd*g;
figure;
%subplot(2,2,1)
plot(err);
title('K-SVD error convergence of permeability','FontName','Helvetica', 'Fontsize', 13);
xlabel('Iteration','FontName','Helvetica', 'Fontsize', 13); 
ylabel('RMSE','FontName','Helvetica', 'Fontsize', 13);
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')

filename= strcat('Block_', num2str(ii));
cd(folder)
saveas(gcf,filename,'fig')
cd(oldfolder)

end