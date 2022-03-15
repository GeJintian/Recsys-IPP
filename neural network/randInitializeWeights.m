function W = randInitializeWeights(L_in, L_out)
    epsilon_init = 6^0.5/(L_out + L_in)^0.5;
    W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
end
