function [A, B, Xi] = qp_toy(X, Y, lambda)

[n, N] = size(X);

cvx_begin
  variable A(n, n)
  variable C(n, n)
  variable Xi(n, N)
  obj = sum(sum_square(C * Y - Xi - X)) / N + lambda * sum(sum(Xi - A * X)) / N;
  minimize(obj)
  subject to
    Xi >= 0
    Xi >= A * X
cvx_end

B = inv(C);

end