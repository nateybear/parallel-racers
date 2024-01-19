clear; clc; 

% number of individuals 
N = 100000; 
% number of products 
J = 10; 
% number of simulation draws 
S = 100;

% observable demographics 
W = rand(N,1); 
% product char 
X  = rand(J,1);
% simulation draws, for simulated ln lik 
Draws = randn(1,S);

% beta vector
beta = randn(4,1);

% simulate choices, y (NOTE: THIS IS FAKE AND NOTE DONE GIVEN THE DGP)
y = floor(J*rand(N,1))+1; %agent choices (fake)

%--- SERIAL IMPLEMNTATION ---%

serial_tic = tic; 

% holder for the exp(utils) given a simulation draw
expUtils = ones(J, N, S);
% loop over draws
for i = 1:S
    expUtils(:,:,i) = exp(beta(1) + X*(beta(2) + beta(3)*W' + beta(4)*Draws(i)));
end
% calculate CCPs
CCP = mean(expUtils ./ sum(expUtils, 1), 3);
% calculate lnlik
lnlik_v = ones(N,1); 
for i = 1:N
    lnlik_v(i) = log(CCP(y(i), i));
end
% return negative sum of log like
lnlik = -sum(lnlik_v);

serial_time = toc(serial_tic); 

%--- PARALLEL IMPLEMNTATION ---%

% set the pool size 
P = 20;
% call the pool of size P, takes a hot sec to set up so excluding for runtime measures 
pool = parpool('local', P);

par_tic = tic; 

% holder for the exp(utils) given a simulation draw
expUtils = ones(J, N, S);
% loop over draws
parfor i = 1:S
    expUtils(:,:,i) = exp(beta(1) + X*(beta(2) + beta(3)*W' + beta(4)*Draws(i)));
end
% calculate CCPs
CCP = mean(expUtils ./ sum(expUtils, 1), 3);
% calculate lnlik
lnlik_v = ones(N,1); 
parfor i = 1:N
    lnlik_v(i) = log(CCP(y(i), i));
end
% return negative sum of log like
lnlik = -sum(lnlik_v); 

par_time = toc(par_tic); 

fprintf('Serial Time Elapsed (seconds): %f\n', serial_time)
fprintf('Parallel Time Elapsed (seconds): %f\n', par_time)

% close pool before exiting 
delete(pool);
