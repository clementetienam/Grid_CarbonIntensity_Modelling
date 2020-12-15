%NAME: CoSaMP Tester
%PURPOSE: Tests sparse signals on CoSaMP
%AUTHOR: Deanna Needell
%OUTSIDE FUNCTIONS: None
%Testing Parameters
clc;
clear all;
load Yes2.out;
load sgsim.out;
Phi=reshape(Yes2,7200,2100);
sVals=[1:1:55]; % Sparsity levels
mVals=[1:7200]; %Measurement levels
dVals=[2100]; %dimension
numTrials=500; %Number of trials per parameter set
%Set Variable lengths and Data Collection
nums=length(sVals);
numm=length(mVals);
numd=length(dVals);
numCorrect = zeros(nums, numm, numd);
trend99 = zeros(numm, 1);
for is=1:nums
%A.4. Compressive Sampling Matching Pursuit 157
for im=1:numm
for id=1:numd
%Set Parameters
s = sVals(is);
m = mVals(im);
d = dVals(id);

%Start a trial
for trial=1:numTrials
%Generate Measurement matrix
%Phi = randn(m,d);
%Generate sparse signal
z = randperm(d);
x = zeros(d, 1);
x(z(1:s)) = sign(randn(s,1));
%Generate measurements
sgsim=reshape(sgsim,72000,100);
test=sgsim(21601:28800,1:100);
%X=sgsim(14401:50400,1:100);
u=test;
% u = Phi*x;
%Begin CoSaMP
%Initialize
a = zeros(d,1);
v = u;
it=0;
stop = 0;
while ~stop
%Signal Proxy
y = Phi'*v;
[tmp, ix] = sort(abs(y), 'descend');
Omega = ix(1:2*s);
[tmp, ix] = sort(abs(a), 'descend');
T = union(Omega, ix(1:s));
%Signal Estimation
b = zeros(d, 1);
b(T) = Phi(:, T) \ u;
%A.5. Reweighted L1 Minimization 158
%Prune
[tmp, ix] = sort(abs(b), 'descend');
a = zeros(d, 1);
a(ix(1:s), 1) = b(ix(1:s), 1);
%Sample Update
v = u - Phi*a;
%Iteration counter
it = it + 1;
%Check Halting Condition
if norm(a-x) <= 10^(-4) || it > max(8*s, 60)
stop = 1;
end
end %End CoSaMP iteration
%Collect Data
%if norm(a-x) <= 10^(-4)
numCorrect(is, im, id) = numCorrect(is, im, id) + 1;
%end
end % End trial
end %d
if trend99(im) == 0 && numCorrect(is, im, id) >= 0.99*numTrials
trend99(im) = s;
end
end %m
end %s
