function [X, Y] = data_gen_res_relu(A_g, B_g, N, noise_std)

[d, ~] = size(A_g);

X = randn(d, N);
Y = B_g * (max(A_g * X, 0) + X);

end