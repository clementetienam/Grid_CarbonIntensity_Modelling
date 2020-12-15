clc;
clear;
close all;
disp('@Author: Dr Clement Etienam')
%% data
oldfolder=cd;
addpath('ksvdbox13')
addpath('ompbox10')
addpath('CCC')
Resultss = 'Results';
mkdir(Resultss);
rng(1);
titles=cell(1,6);
titles{1,2}='Fossil Fuel';
titles{1,3}='Interconnectors';
titles{1,4}='Nuclear';
titles{1,5}='Wind';
titles{1,6}='Other Renewables';
titles{1,1}='Bio mass';
cd('data')
numdd = xlsread("clems2.xlsx","GenerationHH(TWh)");
True_track=xlsread("clems2.xlsx","GenerationAnnualTotal");
cd(oldfolder)
output=numdd;
True_tr=True_track; %True_average to track
out = True_tr(all(~isnan(True_tr),2),:); % for nan - rows
%% Divide into 4 scenarios for 2017-2020
v1 =out;
b1 = 6; % block size
n1 = size(v1,1);
True_track = mat2cell(v1,diff([0:b1:n1-1,n1]));

%% Learn KSVD dictionary for the 4 blocks (Protoyping) and also get the sparse coefficients
b1 = 17520; % block size
matrix1=output(1:b1,:);
matrix2=output(b1+1:2*b1,:);
matrix3=output((2*b1)+1:3*b1,:);
matrix4=output((3*b1)+1:end,:);

matrix=cell(4,1);
matrix{1,1}=matrix1;
matrix{2,1}=matrix2;
matrix{3,1}=matrix3;
matrix{4,1}=matrix4;
dictionary=cell(4,1);
sparse_coeff=cell(4,1);

dictsize=1;
iterr=100;
Tdata=30;

parfor iiy=1:4
[dictionary{iiy,1},sparse_coeff{iiy,1}]=Learn_Dictionary(Tdata,matrix{iiy,1},dictsize,iterr,oldfolder,Resultss,iiy);
end
% Abig=[];
% for iy=1:4
%     r=zeros(1,6);
% parfor iw=1:6
%  r(:,iw) = normrnd(True_tr(iy,iw),0.001*True_tr(iy,iw),1,1);   
% end
% Abig=[Abig;r];
% end

Abig1=True_track{1,:};
Abig=Abig1(:,1:4)';
Ain=[];
parfor ik=1:4
    Ain=[Ain;sparse_coeff{ik,1}]
end
Ain=full(Ain);
%% Deep Neural Network between the inputs and the outputs
X_use=Ain;
y_use=Abig;
input_count = size( X_use , 2 );
output_count = size( y_use , 2 );
epoch=20000;
batch_size=5;
layers = [ ...
    sequenceInputLayer(input_count)
    fullyConnectedLayer(200)
    reluLayer
    fullyConnectedLayer(80)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(output_count)
    regressionLayer
    ];

options = trainingOptions('adam', ...
    'MaxEpochs',epoch, ...
    'MiniBatchSize', batch_size , ...
    'ValidationFrequency',10, ...
    'ValidationPatience',5, ...
    'Verbose',true, ...
    'Plots','training-progress');
Model_DNN = trainNetwork(X_use',y_use',layers,options);
cd(Resultss)
save ('Model_DNN.mat', 'Model_DNN');
cd(oldfolder)
%% Learn Time series Model
Bt1=output; 

data=Bt1;
cd(Resultss)
save('data.mat','data');
cd(oldfolder)

numTimeStepsTrain = 3*b1;% floor(0.75*length(data));
dataTrain = data(1:numTimeStepsTrain,:);
dataTest = data(numTimeStepsTrain+1:end,:);
mu = mean(dataTrain,1);
sig = std(dataTrain,1);
dataTrainStandardized = (dataTrain - mu) ./ sig;
shift=7; 
XTrain = dataTrainStandardized(1:end-shift,:);
YTrain = dataTrainStandardized(shift+1:end,:);

%%
numFeatures = size(XTrain,2);
numResponses = size(YTrain,2);
numHiddenUnits = 300;
choice=input('Enter the desired architecture for the LSTM 1 or 2: ');
if choice==1
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',10000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'MiniBatchSize', 1 , ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',1, ...
    'Plots','training-progress');
else
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(200,'OutputMode','sequence') 
    lstmLayer(150,'OutputMode','sequence')
    lstmLayer(70,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer]; 
fprintf('\nLSTM Network Architecture is defined successfully \n') 
%% Specify the training options 
options = trainingOptions('adam', ...
    'MaxEpochs',19950, ...
    'MiniBatchSize', 1 , ...
    'ExecutionEnvironment','auto', ...
    'Plots','training-progress'); 
end
net = trainNetwork(XTrain',YTrain',layers,options);
cd(Resultss)
save ('net.mat', 'net');
save ('shift.mat', 'shift');
save ('XTrain.mat', 'XTrain');
save ('numTimeStepsTrain.mat', 'numTimeStepsTrain');
transs=[mu;sig];
save('transs.mat','transs');
cd(oldfolder)
net2=net;
net3=net;

dataTestStandardized = (dataTest - mu) ./ sig;
XTest = dataTestStandardized(1:end-shift,:);
YTest = dataTestStandardized(shift+1:end,:);
net2 = predictAndUpdateState(net2,XTrain(1:end-shift,:)');

totalLength=length(data);
trainLength=numTimeStepsTrain;
%% 
%%
lastSteps = zeros(size(XTest,1)+shift,size(data,2)) ;
lastSteps(1:shift,:) = XTrain(end-shift+1:end,:); %The last shift elements of training data
lastSteps=lastSteps';
for i=1:size(XTest,1)
[net2,lastSteps(:,i+shift)] =predictAndUpdateState(net2,lastSteps(:,i));
end

forecastFromSelf = lastSteps(:,1:size(XTest,1));
forecastFromSelf=forecastFromSelf';
forecastFromSelf = sig.*forecastFromSelf+ mu;
forecastFromSelf=abs(forecastFromSelf);


XTest2=sig.*XTest+ mu;
YTest2=sig.*YTest+ mu;
YTrain2=sig.*YTrain+ mu;
XTrain2=sig.*XTrain+ mu;

%%
figure(1)
for i=1:size(data,2)
subplot(2,3,i)    
plot(XTest2(:,i) ,'b','LineWidth',1)
hold on
plot(forecastFromSelf(:,i) ,'r','LineWidth',1)
hold on
xlabel('Time','FontName','Helvetica', 'Fontsize', 13);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 13);
title([titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True test','ForecastSelf test 1',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 13)
set(gcf,'color','white') 
end
cd(Resultss)
saveas(gcf,'Self_1','fig')
cd(oldfolder)

%%
%%
figure(2)
for i=1:size(data,2)
subplot(2,3,i)
plot(YTest2(:,i),'r','LineWidth',1)
hold on
plot(forecastFromSelf(:,i) ,'b','LineWidth',1)
hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title([titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Self_2','fig')
cd(oldfolder)

figure(3)
for i=1:size(data,2)
subplot(2,3,i)
hist(XTest2(:,i)-forecastFromSelf(:,i))
xlabel('Frequency','FontName','Helvetica', 'Fontsize', 9);
ylabel('Diffrence','FontName','Helvetica', 'Fontsize', 9);
title(['Change in:',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Diffrence_from_true','fig')
cd(oldfolder)
parfor i=1:size(data,2)
 rmsee(:,i) = sqrt(mean((XTest2(:,i)-forecastFromSelf(:,i)).^2))./size(XTest,1);   
end

%% Correct predictions from Forecast from Self for the 4th Block (just to see)
% options2=optimset('Display','iter','MaxIter',...
% 1000000,'TolX',10^-200000,'TolFun',10^-200000,'MaxFunEvals',...
% 1000000,'PlotFcns',@optimplotfval,'UseParallel',true);  
% hyp_updated=fminsearch('Optimize_clement',reshape(sparse_coeff{4,1},1,6),...
%    options2,Model_DNN,reshape(Abig(end,:),1,6));


mean_uchee=sum(forecastFromSelf);
divideit=(mean_uchee./(Abig(end,:)));

forecastFromSelf_corrected=[];
parfor i=1:size(forecastFromSelf,1)
forecastFromSelf_corrected(i,:)=forecastFromSelf(i,:)./divideit;
end


parfor i=1:size(data,2)
 rmseee(:,i) = sqrt(mean((XTest2(:,i)-check(:,i)).^2))./size(XTest,1);   
end

% Recover full matrix
% forecastFromSelf_corrected=abs(full(dictionary{4,1}*reshape(Abig(end,:),1,6)));
figure(4)
for i=1:size(data,2)
subplot(2,3,i)
plot(YTest2(:,i),'r','LineWidth',1)
hold on
plot(forecastFromSelf_corrected(:,i) ,'b','LineWidth',1)
hold off
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Corrected :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Corrected_Self_4','fig')
cd(oldfolder)

figure(5)
for i=1:size(data,2)
subplot(2,3,i)
hist(XTest2(:,i)-forecastFromSelf_corrected(:,i))
xlabel('Frequency','FontName','Helvetica', 'Fontsize', 9);
ylabel('Diffrence','FontName','Helvetica', 'Fontsize', 9);
title(['Corrected Change in: ',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Corrected_diffrence_true','fig')
cd(oldfolder)

%%
% figure(6)
% for i=1:size(data,2)
% subplot(2,3,i)
% plot(Abig(4,i),'r','LineWidth',1)
% hold on
% plot(sum(forecastFromSelf(:,i)) ,'b','LineWidth',1)
% hold on
% plot(sum(forecastFromSelf_corrected(:,i)) ,'k','LineWidth',1)
% hold off
% xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
% ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
% title(['Corrected :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
% legend('True model','raw LSTM','corrected LSTM',...
%     'location','northeast');
% set(gca, 'FontName','Helvetica', 'Fontsize', 9)
% set(gcf,'color','white')  
% end
% cd(Resultss)
% saveas(gcf,'compare_correction','fig')
% cd(oldfolder)

%% Long Predict from 4/12/2020  to 31/12/2050 for a 30 minutes duration

t1={'04-Dec-2020 16:30:00'};
t2={'31-Dec-2050 00:00:00'};
t11=datevec(datenum(t1));
t22=datevec(datenum(t2));
time_interval_in_minutes = etime(t22,t11)/60;
Block_30=time_interval_in_minutes/30; % true size including leap years
Block_30_req=525600;

clementend=data(end-shift+1:end,:);
clementend=(clementend - mu) ./ sig;
netuse=net;
lastSteps3 = zeros(Block_30+shift,size(data,2)) ;
lastSteps3(1:shift,:) = clementend; %The last shift elements of training data
lastSteps3=lastSteps3';
for i=1:Block_30
[netuse,lastSteps3(:,i+shift)] =predictAndUpdateState(netuse,lastSteps3(:,i));
end
forecastFromSelf3 = lastSteps3(:,shift+1:end);
forecastFromSelf3=forecastFromSelf3';
forecastFromSelf3 = sig.*forecastFromSelf3+ mu;

usethis=2021:2050;
for i=1:30
A_cell(:,i) = cellstr(num2str(usethis(:,i)));
end
%% Divide into blocks of 17520
forecastFromSelf3(1:Block_30-Block_30_req,:)=[];
clemanswer=forecastFromSelf3;
v1 =forecastFromSelf3;
b1 = 17520; % block size
n1 = size(v1,1);
c1 = mat2cell(v1,diff([0:b1:n1-1,n1]));
tola = 1:30;
%%
raw_1=zeros(30,6);
parfor i=1:30
    aa=c1{i,:}
 raw_1(i,:)=sum(aa);   
end

%% Scenario 1
Abig1=True_track{1,:};
Abig=Abig1(:,5:end)';
figure(6)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,raw_1(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time(years)','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Raw :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Consumer_Transformation_raw','fig')
cd(oldfolder)
%% Scenario 2
Abig2=True_track{2,:};
Abig=Abig2(:,5:end)';
figure(7)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,raw_1(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Raw :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'System_Transformation_raw','fig')
cd(oldfolder)

%% Scenario 3
Abig3=True_track{3,:};
Abig=Abig3(:,5:end)';
figure(8)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,raw_1(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Raw :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Leading_the_way_raw','fig')
cd(oldfolder)

%% Scenario 4
Abig4=True_track{4,:};
Abig=Abig4(:,5:end)';
figure(9)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,raw_1(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Raw:',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Steady_progression_raw','fig')
cd(oldfolder)

%%

cd(Resultss)
 save('Future_Raw_Predictions_LSTM.mat','c1');
cd(oldfolder)

for j=1:6
figure(j+9)
for i=1:30
    use_this=c1{i,:};
    use_now=use_this(:,j);
subplot(5,6,i)
plot(use_now,'r','LineWidth',1)
hold on
xlabel('Time(years)','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
aee=i+2020;
title( [strcat('Year: ', sprintf('%d',aee),' for- '),titles{:,j}],...
    'FontName','Helvetica', 'Fontsize', 9)
% legend('True model','Self Forecast',...
%     'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end 
cd(Resultss)
filename= strcat('LSTM_Predicted ',titles{:,j}, '_Until 2050');
saveas(gcf,filename,'fig')
cd(oldfolder)
end
%% Now Maintain and track this signal going forward. This is the correction for the long prediction
% parfor ii=1:30
% [dictionary2{ii,1},sparse_coeff2{ii,1}]=Learn_Dictionary(Tdata,c1{ii,1},dictsize,iterr,oldfolder,folder,ii);
% end
% Read the true 30*6 signal to correct

%% Scenario 1- Consumer Transformation
hyp_updateds=cell(30,1);
scene_1=cell(30,1);
Abig1=True_track{1,:};
Abig=Abig1(:,5:end)';
parfor iclement=1:30
% hyp_updateds{iclement,1}=fminsearch('Optimize_clement',full(reshape(sparse_coeff2{iclement,1},1,6)),...
%    options2,Model_DNN,Abig(iclement,:));
% % Recover full matrix
%  ouut=Tackit(c1{iclement,1}, Abig(iclement,:))
scene_1{iclement,1}=Tackit(c1{iclement,1}, Abig(iclement,:))%full(dictionary{iclement,1})*hyp_updateds{iclement,1};   
end
%% Scenario 2- System Transformation
hyp_updateds2=cell(30,1);
scene_2=cell(30,1);
Abig2=True_track{2,:};
Abig=Abig2(:,5:end)';
parfor iclement1=1:30
% hyp_updateds2{iclement1,1}=fminsearch('Optimize_clement',full(reshape(sparse_coeff2{iclement1,1},1,6)),...
%    options2,Model_DNN,Abig(iclement1,:));
% Recover full matrix
scene_2{iclement1,1}=Tackit(c1{iclement1,1}, Abig(iclement1,:))%full(dictionary{iclement1,1})*hyp_updateds2{iclement1,1};   
end
%% Scenario 3- Leading the way
hyp_updateds3=cell(30,1);
scene_3=cell(30,1);
Abig3=True_track{3,:};
Abig=Abig3(:,5:end)';
parfor iclement2=1:30
% hyp_updateds3{iclement2,1}=fminsearch('Optimize_clement',full(reshape(sparse_coeff2{iclement2,1},1,6)),...
%    options2,Model_DNN,Abig(iclement2,:));
% Recover full matrix
scene_3{iclement2,1}=Tackit(c1{iclement2,1}, Abig(iclement2,:))%full(dictionary{iclement2,1})*hyp_updateds3{iclement2,1};   
end

%% Scenario 4- Steady progression
hyp_updateds4=cell(30,1);
scene_4=cell(30,1);
Abig4=True_track{4,:};
Abig=Abig4(:,5:end)';
parfor iclement3=1:30
% hyp_updateds4{iclement3,1}=fminsearch('Optimize_clement',full(reshape(sparse_coeff2{iclement3,1},1,6)),...
%    options2,Model_DNN,Abig(iclement3,:));
% Recover full matrix
scene_4{iclement3,1}=Tackit(c1{iclement3,1}, Abig(iclement3,:))%full(dictionary{iclement3,1})*hyp_updateds4{iclement3,1};   
end

%%
disp('Save the results')
cd(Resultss)
save ('scene_1.mat', 'scene_1');
save ('scene_2.mat', 'scene_2');
save ('scene_3.mat', 'scene_3');
save ('scene_4.mat', 'scene_4');
cd(oldfolder)
%% Get Average for the 4 scenario
%Scenario 1
Average_1=zeros(30,6);
parfor i=1:30
    aa=scene_1{i,:}
 Average_1(i,:)=sum(aa);   
end

%Scenario 2
Average_2=zeros(30,6);
parfor i=1:30
    aa=scene_2{i,:}
 Average_2(i,:)=sum(aa);   
end
%Scenario 3
Average_3=zeros(30,6);
parfor i=1:30
    aa=scene_3{i,:}
 Average_3(i,:)=sum(aa);   
end
%Scenario 4
Average_4=zeros(30,6);
parfor i=1:30
    aa=scene_4{i,:}
 Average_4(i,:)=sum(aa);   
end

cd(Resultss)
save ('Average_1.mat', 'Average_1');
save ('Average_2.mat', 'Average_2');
save ('Average_3.mat', 'Average_3');
save ('Average_4.mat', 'Average_4');
cd(oldfolder)

%%

%%
%% Scenario 1
Abig1=True_track{1,:};
Abig=Abig1(:,5:end)';
figure(16)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,Average_1(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Corrected :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Consumer_Transformation_corrected','fig')
cd(oldfolder)
%% Scenario 2
Abig2=True_track{2,:};
Abig=Abig2(:,5:end)';
figure(17)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,Average_2(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Corrected :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'System_Transformation_corrected','fig')
cd(oldfolder)

%% Scenario 3
Abig3=True_track{3,:};
Abig=Abig3(:,5:end)';
figure(18)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,Average_3(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Corrected :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Leading_the_way_corrected','fig')
cd(oldfolder)

%% Scenario 4
Abig4=True_track{4,:};
Abig=Abig4(:,5:end)';
figure(19)
for i=1:size(data,2)
subplot(2,3,i)
plot(tola,Abig(:,i),'r','LineWidth',1)
hold on
plot(tola,Average_4(:,i) ,'b','LineWidth',1)
hold off
xticks([1:30])
set(gca,'XTickLabel',A_cell,'XTickLabelRotation',45,'FontName','Helvetica', 'Fontsize', 9);
xlabel('Time','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
title(['Corrected :',titles{:,i}],'Interpreter','none','FontName','Helvetica', 'Fontsize', 10);
legend('True model','Self Forecast',...
    'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end
cd(Resultss)
saveas(gcf,'Steady_progression_corrected','fig')
cd(oldfolder)
%% Scenario 1
for j=1:6
figure(19+j)
for i=1:30
    use_this=scene_1{i,:};
    use_now=use_this(:,j);
subplot(5,6,i)
plot(use_now,'r','LineWidth',1)
hold on
xlabel('Time(years)','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
aee=i+2020;
title( [strcat('Year: ', sprintf('%d',aee),'for '),titles{:,j}],...
    'FontName','Helvetica', 'Fontsize', 9)
% legend('True model','Self Forecast',...
%     'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end 
cd(Resultss)
filename= strcat('LSTM_Predicted_corrected_Consumer_Transformation ',titles{:,j}, '_Until 2050');
saveas(gcf,filename,'fig')
cd(oldfolder)
end
%% Snenario 2
for j=1:6
figure(26+j)
for i=1:30
    use_this=scene_2{i,:};
    use_now=use_this(:,j);
subplot(5,6,i)
plot(use_now,'r','LineWidth',1)
hold on
xlabel('Time(years)','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
aee=i+2020;
title( [strcat('Year: ', sprintf('%d',aee),' for- '),titles{:,j}],...
    'FontName','Helvetica', 'Fontsize', 9)
% legend('True model','Self Forecast',...
%     'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end 
cd(Resultss)
filename= strcat('LSTM_Predicted_corrected_System_Transformation ',titles{:,j}, '_Until 2050');
saveas(gcf,filename,'fig')
cd(oldfolder)
end

%% Snenario 3
for j=1:6
figure(32+j)
for i=1:30
    use_this=scene_3{i,:};
    use_now=use_this(:,j);
subplot(5,6,i)
plot(use_now,'r','LineWidth',1)
hold on
xlabel('Time(years)','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
aee=i+2020;
title( [strcat('Year: ', sprintf('%d',aee),' for- '),titles{:,j}],...
    'FontName','Helvetica', 'Fontsize', 9)
% legend('True model','Self Forecast',...
%     'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end 
cd(Resultss)
filename= strcat('LSTM_Predicted_corrected_Leading_the_way ',titles{:,j}, '_Until 2050');
saveas(gcf,filename,'fig')
cd(oldfolder)
end

%% Snenario 4
for j=1:6
figure(37+j)
for i=1:30
    use_this=scene_4{i,:};
    use_now=use_this(:,j);
subplot(5,6,i)
plot(use_now,'r','LineWidth',1)
hold on
xlabel('Time(years)','FontName','Helvetica', 'Fontsize', 9);
ylabel('Temperature','FontName','Helvetica', 'Fontsize', 9);
aee=i+2020;
title( [strcat('Year: ', sprintf('%d',aee),' for- '),titles{:,j}],...
    'FontName','Helvetica', 'Fontsize', 9)
% legend('True model','Self Forecast',...
%     'location','northeast');
set(gca, 'FontName','Helvetica', 'Fontsize', 9)
set(gcf,'color','white')  
end 
cd(Resultss)
filename= strcat('LSTM_Predicted_corrected_Steady_progression',titles{:,j}, '_Until 2050');
saveas(gcf,filename,'fig')
cd(oldfolder)
end

rmpath('ksvdbox13')
rmpath('ompbox10')
rmpath('CCC')