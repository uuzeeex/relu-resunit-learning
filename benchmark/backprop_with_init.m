function [A, B, L_new, ii, losses] = backprop_with_init(X, Y, X_test, Y_test, batch_size, eta_0, decay_rate, epoch, lambda1, lambda2, A, B, myinit)

[n, N] = size(X);
[~, N_test] = size(X_test);

L_old = 0;

if myinit > 0
	A = randn(n, n);
	B = randn(n, n);
	fprintf('doing Gaussian init\n');
else
	fprintf('doing my own init\n');
end

losses = zeros(1,epoch);

i = 0;

while i < epoch
  ii = i;
  
  p = randperm(size(X, 2));
  
  %x = X;
  %y = Y;
  
  X_train = X(:, p);
  Y_train = Y(:, p);
  
  k = 0;
  while k < N / batch_size
    
    x = X_train(:, k * batch_size + 1 : min( (k + 1) * batch_size, N));
    y = Y_train(:, k * batch_size + 1 : min( (k + 1) * batch_size, N));
    
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
  
    %L = sum(sum_square(y_pred - y)) / (2 * batch_size);
    %if i > 99900
    %disp(L);
     %end
  
    % forwardprop for numerical test
    %y_pred_A_t = B * (max(A_t * x, 0) + x);
    %L_A_t = sum_square(y_pred_A_t - y) / 2.;
  
    %y_pred_B_t = B_t * (max(A * x, 0) + x);
    %L_B_t = sum_square(y_pred_B_t - y) / 2.;

    % backprop
    L_y = (y_pred - y) / batch_size;
  
    L_B = L_y * h.';
    L_h = B.' * L_y;
  
    L_s = (s > 0) .* L_h;
    L_A = L_s * x.';
  
    eta = eta_0; % / (1 + decay_rate * i);
  
    B = B - eta * L_B - eta * lambda1 * B;
    A = A - eta * L_A - eta * lambda2 * A;
    k = k + 1;
  end
  
  % forwardprop test
  s = A * X_test;
  h = max(s, 0) + X_test;
  Y_pred = B * h;
  L_new = sum(sum_square(Y_pred - Y_test)) / (2 * N_test);
  
  i = i + 1;
  losses(i) = L_new;

  rel = (abs(L_new - L_old) / L_old);

  if (i > 3)
  	if ((abs(L_new - L_old) / L_old) < 1/(10000.0))
      i = epoch + 1;
    end
  end
  L_old = L_new;
  
  % if i < 20
  %   fprintf('A: %.6f %.6f\n', L_A(row, col), (L_A_t - L) / 1e-6);
  %   fprintf('B: %.6f %.6f\n', L_B(row, col), (L_B_t - L) / 1e-6);
  % end

	fprintf('Ended iteration %d with loss %.5f  rel %.5f\n', i, L_new, rel)
end

% plot(losses)

disp('losses')
disp(losses(1 : ii))
