addpath('..');

d = 10;
N = 10;

[A_g, B_g] = params_gen_res_relu(d);

T = 1;

while T <= 100
  [X, Y] = data_gen_res_relu(A_g, B_g, N, 0);
  C_1 = randn(d, d);
  xi_1 = randn(d, N);
  C_2 = randn(d, d);
  xi_2 = randn(d, N);
  T_seg = 1;
  while T_seg <= 1000
    alpha = rand();
    C = alpha * C_1 + (1 - alpha) * C_2;
    xi = alpha * xi_1 + (1 - alpha) * xi_2;
    e_1 = obj_layer2(C_1, xi_1, X, Y);
    e_2 = obj_layer2(C_2, xi_2, X, Y);
    
    lin_val = alpha * e_1 + (1 - alpha) * e_2;
    obj_val = obj_layer2(C, xi, X, Y);
    
    if lin_val > obj_val
      fprintf('Test passed on T = %d, alpha = %.3f\n', T, alpha);
    else
      fprintf('Test failed on T = %d, alpha = %.3f', T, alpha);
      pause
    end
    
    T_seg = T_seg + 1;
  end
  
  T = T + 1;
end