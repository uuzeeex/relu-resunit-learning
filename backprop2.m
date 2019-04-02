function [A, B] = backprop2(X, Y, eta)

A = randn(10, 10);
B = randn(10, 10);

i = 0;

while i < 1000
  
  x = X(:, randi(size(X, 2)));
  y = Y(:, randi(size(Y, 2)));
  
  % numerical test
  row = randi(size(A, 1));
  col = randi(size(A, 2));
  A_t = A;
  A_t(row, col) = A_t(row, col) + 1e-6;
  B_t = B;
  B_t(row, col) = B_t(row, col) + 1e-6;
  
  % forwardprop
  s = A * x;
  h = max(s, 0) + x;
  y_pred = B * h;
  
  L = sum_square(y_pred - y) / 2.;
  
  % forwardprop for numerical test
  %y_pred_A_t = B * (max(A_t * x, 0) + x);
  %L_A_t = sum_square(y_pred_A_t - y) / 2.;
  
  %y_pred_B_t = B_t * (max(A * x, 0) + x);
  %L_B_t = sum_square(y_pred_B_t - y) / 2.;

  % backprop
  L_y = y_pred - y;
  
  L_B = L_y * h.';
  L_h = B.' * L_y;
  
  L_s = (s > 0) .* L_h;
  L_A = L_s * x.';
  
  B = B - eta * L_B;
  A = A - eta * L_A;
  
  i = i + 1;
  
  %fprintf('A: %.6f %.6f\n', L_A(row, col), (L_A_t - L) / 1e-6);
  %fprintf('B: %.6f %.6f\n', L_B(row, col), (L_B_t - L) / 1e-6);

end
