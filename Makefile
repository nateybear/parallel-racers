.PHONY: rust
.ALL: rust

# LHS is filename of graph you create, RHS is all the scripts that go into making it
# Second line is whatever command you run that does the thing
rust_out_$(MACHINE)_julia.png: rust_likelihood.jl rust_plot.jl rust_main_julia.sh
	./rust_main_julia.sh | julia --project=. rust_plot.jl

# add all rust targets here on the same line
rust: rust_out_$(MACHINE)_julia.png
