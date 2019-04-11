lambda = 2e-5;

A_errs_qp = zeros(8, 8);
B_errs_qp = zeros(8, 8);
x_axis = zeros(8, 1);
y_axis = zeros(8, 1);

for i = 1 : 8
  for j = 1 : 8
    A_errs = zeros(8, 1);
    B_errs = zeros(8, 1);
    n = 8  + i * 2;
    N = 80 + j * 20;
    x_axis(i) = N;
    y_axis(j) = n;
    
    A_g = randn(n, n);
    B_g = randn(n, n);
    A_g = abs(A_g);
    
    T = 1;
    while T <= 8
      X = randn(n, N);
      Y = B_g * (max(A_g * X, 0) + X);
      
      [A, B] = reluqp2(X, Y, lambda);
      
      A_errs(T) = norm(A_g - A) / norm(A_g);
      B_errs(T) = norm(B_g - B) / norm(B_g);
      
      T = T + 1;
    end
    A_errs_qp(i, j) = mean(A_errs);
    B_errs_qp(i, j) = mean(B_errs);
    
  end
end

heatmap(x_axis, y_axis, A_errors_qp);
heatmap(x_axis, y_axis, B_errors_qp);