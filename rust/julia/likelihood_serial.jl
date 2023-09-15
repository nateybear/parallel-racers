using Distributions

function likelihood((as, is))
    out = 0.0
    @noinline for (a,i) in zip(as, is)
        p = cdf(Logistic(), v₁[a] - v₀[a] + rand())
        out += i * log(p) + (1 - i) * log(1 - p)
    end

    return out
end
