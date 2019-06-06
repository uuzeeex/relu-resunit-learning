function [A, B, xi] = reluqp2(X, Y, lambda)

[n, N] = size(X);

cvx_begin
  variable A(n, n)
  variable C(n, n)
  variable xi(n, N)
  obj = sum(sum_square(C * Y - xi - X)) / N + lambda * sum(sum(xi - A * X)) / N;
  minimize(obj)
  subject to
    xi >= 0
    xi >= A * X
cvx_end

B = inv(C);

end