N = 100;
m = 10;
n = 10;
k = 10;

%rng(2333);
T = 0;
A_errs_qp = [];
A_errs_bp = [];
B_errs_qp = [];
B_errs_bp = [];

while T < 100
X = randn(n, N);
A_g = randn(m, n);
B_g = randn(k, m);
A_g = abs(A_g);
Y = B_g * (max(A_g * X, 0) + X);

cvx_begin
  variable A(m, n)
  variable C(m, k)
  variable xi(m, N)
  obj = sum(sum_square(C * Y - xi - X)) / N + 2e-5 * sum(sum(xi - A * X)) / N;
  minimize(obj)
  subject to
    xi >= 0;
    xi >= A * X;
cvx_end
B = inv(C);
Y_pred = B * (max(A * X, 0) + X);
%norm(Y - Y_pred) / norm(Y);
A_errs_qp = [A_errs_qp; norm(A_g - A) / norm(A_g)];
B_errs_qp = [B_errs_qp; norm(B_g - B) / norm(B_g)];
[A_bp, B_bp] = backprop2(X, Y, 1e-3);
A_errs_bp = [A_errs_bp; norm(A_g - A_bp) / norm(A_g)];
B_errs_bp = [B_errs_bp; norm(B_g - B_bp) / norm(B_g)];
T = T + 1;
end

histfit(A_errs_qp);
hold on;
histfit(A_errs_bp);
%hold on;
histfit(B_errs_qp);
hold on;
histfit(B_errs_bp);

%norm(Y) / sqrt(N)