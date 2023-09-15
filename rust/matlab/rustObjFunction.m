clear; clc;


% ------------------------------------------------ %
% --- Constants And Objects Used in Estimation --- %
% ------------------------------------------------ %

% Size of pool 
P = 16;

% number of runs 
N = 10;

% set discount factor 
beta = 0.9;

% tranisition kernel for i = 0
T_0 = zeros(5,5);
T_0(1,2) =1;
T_0(2,3) =1;
T_0(3,4) =1;
T_0(4,5) =1;
T_0(5,5) =1;

% transition kernel for i = 1
T_1 = zeros(5,5); 
T_1(1,1) = 1; 
T_1(2,1) = 1;
T_1(3,1) = 1;
T_1(4,1) = 1;
T_1(5,1) = 1;

% state space 
a_t = (1:5)';

% guess of parameters 
theta1_hat = -1.1; 
R_hat = -5;

% start pool 
pool = parpool('local', P);

% ----------- %
% --- VFI --- %
% ----------- %


% initalize Cond-VF 
V_0_new = ones(size(a_t));
V_1_new = ones(size(a_t));
% initalize tolerence 
tol_0   = 1000;
tol_1   = 1000;
% VFI 
while (tol_0 > 1e-16) && (tol_1 > 1e-16)
    % update value function 
    V_0_old = V_0_new; 
    V_1_old = V_1_new; 
    % get new choice specific value functions 
    V_0_new = theta1_hat*a_t + beta*(0.5772 + log(exp(T_0*V_0_old) + exp(T_0*V_1_old)));
    V_1_new = R_hat + beta*(0.5772 + log(exp(T_1*V_0_old) + exp(T_1*V_1_old)));
    % update tol 
    tol_0 = max(abs(V_0_new - V_0_old));
    tol_1 = max(abs(V_1_new - V_1_old));
end


% ------------------- %
% --- Evaluate LL --- %
% ------------------- %

data  = load("..\data.asc");
a_obs_raw = data(:,1); 
i_obs_raw = data(:,2);

% list of duplicated data frame 
dupVector = [1, 10, 100, 1000, 10000, 100000]; 
% storage of mean run time 
t_out = zeros(size(dupVector, 2), 3);

for j = 1:size(dupVector,2)
    a_obs = kron(ones(dupVector(j), 1), a_obs_raw);
    i_obs = kron(ones(dupVector(j), 1), i_obs_raw);
    % run many times to get a mean of run times 
    t_mtx = zeros(N, 3);
    for k = 1:N
        %%% first draw noise for CCP calculations 
        eps0 = rand(size(a_obs));
        eps1 = rand(size(a_obs));
        %%% vectorize implementation 
        tic
        % recover CCPs 
        CCP_1 = exp(V_1_new)./(exp(V_0_new) +(exp(V_1_new)));
        CCP_0 = exp(V_0_new)./(exp(V_0_new) +(exp(V_1_new)));
        nLL_v  = (-1)*sum(log((i_obs).*CCP_1(a_obs) + (1 - i_obs).*CCP_1(a_obs)));
        t_mtx(k,1) = toc; 
        %%% serial implementation 
        nLL_s = zeros(size(i_obs));
        tic
        for i = 1:size(i_obs)
            CCP_1 = exp(V_1_new + eps1(i))./(exp(V_0_new + eps0(i)) +(exp(V_1_new + eps1(i))));
            CCP_0 = exp(V_0_new + eps1(i))./(exp(V_0_new + eps0(i)) +(exp(V_1_new + eps1(i))));
            nLL_s(i) = (-1)*sum(log((i_obs(i)).*CCP_1(a_obs(i)) + (1 - i_obs(i)).*CCP_1(a_obs(i))));
        end 
        nLL_s = sum(nLL_s);     
        t_mtx(k,2) = toc;
        %%% par for implementation 
        nLL_p = zeros(size(i_obs));
        tic
        parfor i = int32(1:size(i_obs))
            CCP_1 = exp(V_1_new + eps1(i))./(exp(V_0_new + eps0(i)) +(exp(V_1_new + eps1(i))));
            CCP_0 = exp(V_0_new + eps1(i))./(exp(V_0_new + eps0(i)) +(exp(V_1_new + eps1(i))));
            nLL_p(i) = (-1)*sum(log((i_obs(i)).*CCP_1(a_obs(i)) + (1 - i_obs(i)).*CCP_1(a_obs(i))));
        end
        t_mtx(k,3) = toc;
    end 
    t_out(j,:) = mean(t_mtx);
end

% pull together output to nice csv for plots, etc.
out = [];
for i = 1:3
    out = [out; horzcat(ones(size(dupVector, 2), 1)*i, dupVector'*size(a_obs_raw,1), t_out(:,i))];
end 
outTbl = array2table(out);
outTbl.Properties.VariableNames = {'Method', 'Size', 'Elapsed'}; 
outName = strcat('..\out\matlab_', 'P', string(P), '_N', string(N), '.csv');
writetable(outTbl, outName)

% close out pool 
delete(pool);












