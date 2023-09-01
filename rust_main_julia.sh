#!/bin/bash

echo "threads,multiple,elapsed_time" 
# Function to run Julia command and convert output to CSV
run_julia() {
    local threads=$1
    local multiple=$2

    # Run Julia command and redirect output to CSV file
    local elapsed_time=`julia --project=. -t $threads rust_likelihood.jl $multiple | awk '{print $1}'`

    echo "$threads,$multiple,$elapsed_time"
}

for i in {1..20}; do
    for multiple in 1 10 100 1000 10000 100000; do
        for threads in 1 4 8 16 32; do
            run_julia $threads $multiple
        done
    done
done
