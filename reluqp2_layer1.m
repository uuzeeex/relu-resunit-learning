function [A, eta] = reluqp2_layer1(X, Xi)

[n, N] = size(X);

cvx_begin
  variable A(n, n)
  variable eta(n, N)
  obj = sum(sum_square(A * X + eta - Xi)) / N;
  minimize(obj)
  subject to
    eta >= 0
cvx_end

end