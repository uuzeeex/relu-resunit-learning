function [B, xi] = reluqp2_layer2(X, Y)

[n, N] = size(X);

cvx_begin
  cvx_precision medium
  variable C(n, n)
  variable xi(n, N)
  obj = sum(sum_square(xi + X - C * Y)) / N;
  minimize(obj)
  subject to
    xi >= 0
cvx_end

B = inv(C);

end