using Statistics
using LinearAlgebra

capN = 100000;
capJ = 10;
capS = 1000;
w = rand(capN, 1, 1);  #agent characteristic
x = rand(1, capJ, 1); # product characteristic
draws = rand(1, 1, capS); #sim draws
y = floor.(Int, capJ.*rand(capN)).+1; #agent choices (fake - Im not actually simulating choices according to model)


function L(b)
    CCPs = @. exp(b[1] + x * (b[2] + b[3] * w + b[4] * draws))
    CCPs = CCPs ./ sum(CCPs, dims = 2)
    CCPs = mean(CCPs, dims = 3)
    choices = reshape(1:capJ, 1, :) .== y
    sum(@. choices * log(CCPs) + (1 - choices) * log(1 - CCPs))
end
