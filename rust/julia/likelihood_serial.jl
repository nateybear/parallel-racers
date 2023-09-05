using Distributions

@noinline function likelihood(params)
    out = 0.0
    for (a,i) in zip(params...)
        p = cdf(Logistic(), v₁[a] - v₀[a])
        out += i * log(p) + (1 - i) * log(1 - p)
    end

    return out
end
