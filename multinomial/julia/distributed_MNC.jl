function L(b)
    @sync @distributed (+) for (w, y) in eachrow(mat)
        CCPs = @. exp(b[1] + x * (b[2] + b[3] * w + b[4] * draws))
        CCPs = CCPs ./ sum(CCPs, dims = 2)
        CCPs = reshape(mean(CCPs, dims = 3), :)
        choices = 1:capJ .== y
        sum(@. choices * log(CCPs) + (1 - choices) * log(1 - CCPs))
    end
end
