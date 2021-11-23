d = 12;

cnt = 0;
for i = 1 : 1000
N = 100000;
[A_g, B_g] = params_gen_res_relu(d, d);
[X, Y] = data_gen_res_relu(A_g, B_g, N, 0);

[A, B, X_n, X_p] = relulr(X, Y);
[~, N_n] = size(X_n);
[~, N_p] = size(X_p);
if (N_n >= d) && (N_p >= d)
  cnt = cnt + 1;
end
end