lambda = 2e-5;

cond_thr = inf;

d = 10;

N = 1000;
N_test = 100;

while 1
  [A_g, B_g] = params_gen_res_relu(d);
  if (cond(A_g) < cond_thr && cond(B_g) < cond_thr)
    break;
  end
end

T = 1;
while T <= 1
  [X, Y] = data_gen_res_relu(A_g, B_g, N, 0);
  [B, Xi] = reluqp2_layer2(X, Y);
  %[X_test, Y_test] = data_gen_res_relu(A_g, B_g, N_test, 0);
  %Y_pred = B * (max(A * X_test, 0) + X_test);
  %disp(mean(vecnorm(Y_pred - Y_test) ./ vecnorm(Y_test)));
  T = T + 1;
end