N = 1000;
m = 10;
n = 10;
 
rng(2333);

X = randn(n, N);
A_g = -abs(randn(m, n));
Y = max(A_g * X, 0);

cvx_begin
  variable A(m, n)
  variable xi(m, N)
  obj = sum(sum_square(Y - xi)) / N + 1e-5 * sum(sum(A));%  + sum(sum_square(xi)) + sum(sum_square(A * X - xi))) / N;
  minimize(obj)
  subject to
    xi >= 0;
    xi >= A * X;
cvx_end

Y_pred = max(A * X, 0);
norm(Y - Y_pred) / norm(Y)
norm(A_g - A) / norm(A_g)


norm(Y) / sqrt(N)