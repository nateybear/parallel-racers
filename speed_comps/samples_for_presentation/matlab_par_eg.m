% matlab parallel processing example for presentation

clear; clc; 

a = 4;

% set the pool size 
P = 4;
% call the pool of size P
pool = parpool('local', P);

% loop
parfor i = 1:100
    mean(randn(a,1)) * i
end

% close pool before exiting 
delete(pool);