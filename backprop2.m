function [A, B] = backprop2(X, Y, batch_size, eta_0, decay_rate, iter)

[n, N] = size(X);

A = randn(n, n);
B = randn(n, n);

i = 0;

while i < iter
  
  p = randperm(size(X, 2), batch_size);
  
  %x = X;
  %y = Y;
  
  x = X(:, p);
  y = Y(:, p);
  
  % numerical test
  %row = randi(size(A, 1));
  %col = randi(size(A, 2));
  %A_t = A;
  %A_t(row, col) = A_t(row, col) + 1e-6;
  %B_t = B;
  %B_t(row, col) = B_t(row, col) + 1e-6;
  
  % forwardprop
  s = A * x;
  h = max(s, 0) + x;
  y_pred = B * h;
  
  L = sum(sum_square(y_pred - y)) / (2 * batch_size);
  %if i > 99900
  disp(L);
  %end
  
  % forwardprop for numerical test
  %y_pred_A_t = B * (max(A_t * x, 0) + x);
  %L_A_t = sum_square(y_pred_A_t - y) / 2.;
  
  %y_pred_B_t = B_t * (max(A * x, 0) + x);
  %L_B_t = sum_square(y_pred_B_t - y) / 2.;

  % backprop
  L_y = (y_pred - y) / batch_size;
  
  if i < 5
   %disp(L_y);
  end
  
  L_B = L_y * h.';
  L_h = B.' * L_y;
  
  L_s = (s > 0) .* L_h;
  L_A = L_s * x.';
  
  eta = eta_0 / (1 + decay_rate * i);
  
  B = B - eta * L_B;
  A = A - eta * L_A;
  
  % forwardprop again
  s = A * x;
  h = max(s, 0) + x;
  y_pred = B * h;
  L_new = sum(sum_square(y_pred - y)) / (2 * batch_size);
  
  if i > 0
    %disp(L_A);
    %disp(L_B);
    %disp(L_new - L);
  end
  
  i = i + 1;
  
  %if i < 20
  %fprintf('A: %.6f %.6f\n', L_A(row, col), (L_A_t - L) / 1e-6);
  %fprintf('B: %.6f %.6f\n', L_B(row, col), (L_B_t - L) / 1e-6);
  %end

end
