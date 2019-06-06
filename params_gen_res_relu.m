function [A, B] = params_gen_res_relu(d)
  A = randn(d, d);
  B = randn(d, d);
  A = abs(A);
end