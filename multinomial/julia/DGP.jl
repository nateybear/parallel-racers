using Distributed

@everywhere begin
    using Statistics
    capJ = 10;
    capS = 1000;
    x = rand(1, capJ, 1); # product characteristic
    draws = rand(1, 1, capS); #sim draws
end

function gen_data(capN)
    w = rand(capN, 1, 1);  #agent characteristic
    y = floor.(Int, capJ.*rand(capN)).+1; #agent choices (fake - Im not actually simulating choices according to model)
    w, y
end
