.PHONY: rust
.ALL: rust

# LHS is filename of graph you create, RHS is all the scripts that go into making it
# Second line is whatever command you run that does the thing
rust/out/$(MACHINE)_julia_output.csv: rust/julia/*.jl
	julia -t auto --project=. rust/julia/main.jl > rust/out/$(MACHINE)_julia_output.csv

# add all rust targets here on the same line
rust: rust/out/$(MACHINE)_julia_output.csv
