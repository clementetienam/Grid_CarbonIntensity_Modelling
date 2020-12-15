% clc;
% clear all;
% close all;
ne=100;
% load joy.out;
%load sgsimmaps.out;
%load sgsim1output.out;


Pnew = reshape(joy,36000,ne);
%Pnew = reshape(sgsim1output,72000,ne);
Pnew=log10(Pnew);
 
 %run Assimilate.m
 
%  load rossmary.GRDECL;
%  Trueperm=reshape(rossmary,120,60,10);
%  Ssim = ones(1,ne);
 %% SSIM INDEX COMPARISON
% COMPARISON OF THE ENSEMBLES USING SSIM
 for i=1:ne

        % reshaping each member in 3D
        for j=1:5
  
        P1 = reshape(Pnew(:,i),120,60,5);

        
        
        [X,Y] = meshgrid(1:120,1:60);




figure()

surf(X',Y',P1(:,:,j))
shading flat
axis([1 120 1 60 ])
title('True Layer 3','FontName','Helvetica', 'Fontsize', 13);
ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
 caxis([1 5])
h = colorbar;
set(h, 'ylim', [1 5])
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white')
       

     
    end
 end
    % multiplication of P and S indices
    
%     index = Ssim; 
%     index=index';
%     
%     % choosing between two best members
%     
%     bestssim = find(index == max(index));
% 	Pssim = Pnew(:,bestssim); %best due to ssim
% 	
% 	Pssim=reshape(Pssim,120,60,10);
% 	
% 	
% 	
% 	
% 	
% 	%[X,Y] = meshgrid(1:120,1:60);
% 
% 
% 	
% 	
% Trueperm=log10(Trueperm);
% Pssim=log10(Pssim);
% 
% 
% 
% 
% 
% [X,Y] = meshgrid(1:120,1:60);
% 
% 
% 
% 
% figure()
% subplot(2,5,1);
% surf(X',Y',Trueperm(:,:,3))
% shading flat
% axis([1 120 1 60 ])
% title('True Layer 3','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% 
% subplot(2,5,2);
% surf(X',Y',Trueperm(:,:,4))
% shading flat
% 
% axis([1 120 1 60 ])
% grid off
% title('True Layer 4','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(2,5,3);
% surf(X',Y',Trueperm(:,:,5))
% shading flat
% axis([1 120 1 60 ])
% title('True Layer 5','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(2,5,4);
% surf(X',Y',Trueperm(:,:,6))
% shading flat
% axis([1 120 1 60 ])
% title('True Layer 6','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(2,5,5);
% surf(X',Y',Trueperm(:,:,7))
% shading flat
% axis([1 120 1 60 ])
% title('True Layer 7','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 	
% 	
% 
% subplot(2,5,6);
% surf(X',Y',Pssim(:,:,3))
% shading flat
% axis([1 120 1 60 ])
% title('Layer 3','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')	
% 
% 
% subplot(2,5,7);
% surf(X',Y',Pssim(:,:,4))
% shading flat
% axis([1 120 1 60 ])
% title('Layer 4','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')	
% 
% 
% subplot(2,5,8);
% surf(X',Y',Pssim(:,:,5))
% shading flat
% axis([1 120 1 60 ])
% title('Layer 5','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(2,5,9);
% surf(X',Y',Pssim(:,:,6))
% shading flat
% axis([1 120 1 60 ])
% title('Layer 6','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% subplot(2,5,10);
% surf(X',Y',Pssim(:,:,7))
% shading flat
% axis([1 120 1 60 ])
% title('Layer 7','FontName','Helvetica', 'Fontsize', 13);
% ylabel('Y', 'FontName','Helvetica', 'Fontsize', 13);
% xlabel('X', 'FontName','Helvetica', 'Fontsize', 13);
% caxis([1 5])
% h = colorbar;
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% 
% 
% 	
% figure()
%  subplot(1,2,1);
% slice(Trueperm,[1 60],[1 120],[3 7])
% axis equal on
% %axis([1 27 1 84 1 4 ])
% % xlim([1 84])
% % ylim([1 27])
% % zlim([0 4])
% shading flat
% grid off
% title('3D true permeability','FontName','Helvetica', 'Fontsize', 18);
% caxis([1 5])
% h = colorbar;
% xlabel('X','FontName','Helvetica', 'Fontsize', 18)
% ylabel('Y','FontName','Helvetica', 'Fontsize', 18)
% zlabel('Z','FontName','Helvetica', 'Fontsize', 18)
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% hold on
% plot3([55 55],[30 30],[1 50],'k','Linewidth',2);
% hold on
% plot3([18 18],[58 58],[1 50],'k','Linewidth',2);
% hold on
% plot3([6 6],[90 90],[1 50],'k','Linewidth',2);
% hold on
% plot3([39 39],[101 101],[1 50],'k','Linewidth',2);
% hold on
% plot3([25 25],[14 14],[1 50],'r','Linewidth',2);
% hold on
% plot3([39 39],[38 38],[1 50],'r','Linewidth',2);
% hold on
% plot3([23 23],[96 96],[1 50],'r','Linewidth',2);
% hold on
% plot3([41 41],[67 67],[1 50],'r','Linewidth',2);
% 
% subplot(1,2,2);
% slice(Pssim,[1 60],[1 120],[3 7])
% axis equal on
% %axis([1 27 1 84 1 4 ])
% % xlim([1 84])
% % ylim([1 27])
% % zlim([0 4])
% shading flat
% grid off
% title('SSIM best','FontName','Helvetica', 'Fontsize', 18);
% caxis([1 5])
% h = colorbar;
% xlabel('X','FontName','Helvetica', 'Fontsize', 18)
% ylabel('Y','FontName','Helvetica', 'Fontsize', 18)
% zlabel('Z','FontName','Helvetica', 'Fontsize', 18)
% set(h, 'ylim', [1 5])
% set(gca, 'FontName','Helvetica', 'Fontsize', 13)
% set(gcf,'color','white')
% hold on
% plot3([55 55],[30 30],[1 50],'k','Linewidth',2);
% hold on
% plot3([18 18],[58 58],[1 50],'k','Linewidth',2);
% hold on
% plot3([6 6],[90 90],[1 50],'k','Linewidth',2);
% hold on
% plot3([39 39],[101 101],[1 50],'k','Linewidth',2);
% hold on
% plot3([25 25],[14 14],[1 50],'r','Linewidth',2);
% hold on
% plot3([39 39],[38 38],[1 50],'r','Linewidth',2);
% hold on
% plot3([23 23],[96 96],[1 50],'r','Linewidth',2);
% hold on
% plot3([41 41],[67 67],[1 50],'r','Linewidth',2);

