%% Test 2: CoSaMP
%{
Cocommments:
    This is sensitive to K_target
    If the signal is noiseless, then K_target should be an overestimate
    of the true sparsity level.
    But if the signal is noisy, then the optimal performance usually
    happens if K_target is an underestimate of the true sparsity level.
    (This rule-of-thumb is also true for OMP )

    Also, convergence may be very slow if K_target is large.
    Furthermore, each iteration is slower if K_target is large.

    The "HSS" mode may help for noisy vectors.

    So, summary of recommendations for the case with noisy data:
    "HSS" and "two_solves" mode should be "true"
    "K_target" should be small, like 5% of N
    "addK" should be like 1*K_target or less, not 2*K_target.

%}

load Yes2.out;
load sgsim.out;
sgsim=reshape(sgsim,72000,100);
test=sgsim(21601:28800,1:100);
%X=sgsim(14401:50400,1:100);
b=test;
A=reshape(Yes2,7200,2100);
% Af  = @(x) A*x;
% At  = @(x) A'*x;

opts            = [];
opts.maxiter    = 50;
opts.tol        = 1e-8;
opts.HSS        = true;
opts.two_solves = true; % this can help, but no longer always works "perfectly" on noiseless data
opts.printEvery = 10;
% K_target    = round(length(b)/3)-1; opts.normTol = 2.0;
%K_target        = 50;   % When extremely noisy, this is best; when no noise, this is sub-optimal

%     K_target        = 100;  % This doesn't work "perfectly" but is OK
%     K_target        = 102;  % This works "perfectly" with noiseless data
    K_target        = 150;  % Slower, but works "perfectly" with noiseless data


% opts.addK       = 2*K_target; % default
opts.addK       = K_target; % this seems to work a bit better
% opts.addK       = 5;    % make this smaller and CoSaMP behaves more like OMP 
                        % (and does better for the correlated measurement matrix)

% opts.support_tol    = 1e-2;
disp(   '---------------------------------------');
fprintf('CoSaMP, -------------------------------\n\n');
[xk,r,normR,residHist] = CoSaMP( A, b, K_target, [], opts);
