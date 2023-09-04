using Gadfly, DataFrames, CSV, Chain, DataFramesMeta, Statistics
import Cairo, Fontconfig

df = @chain stdin begin
    CSV.read(DataFrame)
    groupby([:cores, :multiple])
    @combine(
        :elapsed_time = mean(:elapsed_time),
        :ymin = mean(:elapsed_time) - std(:elapsed_time),
        :ymax = mean(:elapsed_time) + std(:elapsed_time)
    )
end

p = plot(
        df,
        x=:multiple,
        y=:elapsed_time,
        ymin=:ymin,
        ymax=:ymax,
        color=:cores,
        Geom.line,
        Geom.point,
        Geom.errorbar,
        Scale.color_discrete(),
        Scale.x_log10(),
    )

draw(PNG("rust_out_$(ENV["MACHINE"])_julia.png", 16inch, 9inch), p)
