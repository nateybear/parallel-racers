#!/bin/bash

for p in 2 4 8 16
do
    echo "Running Julia with $p processes"
    julia --project=. -p $p rust/julia/main.jl > rust/out/julia_P$p.csv
done
