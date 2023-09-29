using Distributed
@everywhere begin
    using Statistics
    using LinearAlgebra

    capN = 100000;
    capJ = 10;
    capS = 1000;
    x = rand(capJ, 1); # product characteristic
    draws = rand(1, capS); #sim draws
end

ws = rand(capN);  #agent characteristic
ys = ceil.(Int, capJ.*rand(capN)); #agent choices (fake - Im not actually simulating choices according to model)

mat = [ ws ys ]

function L(b)
    @sync @distributed (+) for (w, y) in eachrow(mat)
        CCPs = @. exp(b[1] + x * (b[2] + b[3] * w + b[4] * draws))
        CCPs = CCPs ./ sum(CCPs, dims = 1)
        CCPs = mean(CCPs, dims = 2)
        choices = 1:capJ .== y
        sum(@. choices * log(CCPs) + (1 - choices) * log(1 - CCPs))
    end
end
