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

function L_all_for(b)
    @sync @distributed (+) for (w, y) in eachrow(mat)
        CCPs = zeros(capJ, capS)
        for (s, draw) in enumerate(draws)
            sum_exponents = 0.0
            for (j, xj) in enumerate(x)
                exponent = exp(b[1] + b[2]*xj + b[3]*w + b[4]*draw)
                CCPs[j, s] = exponent
                sum_exponents += exponent
            end
            for j in 1:capJ
                CCPs[j, s] /= sum_exponents
                CCPs[j, s] = y == j ? log(CCPs[j, s]) : log(1 - CCPs[j, s])
            end
        end
        sum(mean(CCPs, dims=2))    
    end
end

function L(b, on_main = false)
    @sync @distributed (+) for (w, y) in eachrow(mat)
        product_chars = if on_main
            @fetchfrom 1 x
        else
            x
        end
        CCPs = @. exp(b[1] + product_chars * (b[2] + b[3] * w + b[4] * draws))
        CCPs = CCPs ./ sum(CCPs, dims = 1)
        CCPs = mean(CCPs, dims = 2)
        choices = 1:capJ .== y
        sum(@. choices * log(CCPs) + (1 - choices) * log(1 - CCPs))
    end
end
