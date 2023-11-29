function L(b)
    CCPs = @. exp(b[1] + x * (b[2] + b[3] * w + b[4] * draws))
    CCPs = CCPs ./ sum(CCPs, dims = 2)
    CCPs = mean(CCPs, dims = 3)
    choices = reshape(1:capJ, 1, :) .== y
    sum(@. choices * log(CCPs) + (1 - choices) * log(1 - CCPs))
end
