Y_errs_2d_lp = zeros(8, 8);
Y_errs_2d_bp = zeros(8, 8);

x_axis = zeros(8, 1);
y_axis = zeros(8, 1);

N_test = 128;

for i = 1 : 8 
  d = 6 + i * 2;
  y_axis(i) = d;
  [A_g, B_g] = params_gen_res_relu(d, d);
  [X_test, Y_test] = data_gen_res_relu(A_g, B_g, N_test, 0);
  for j = 1 : 8
    n = 128 + j * 32;
    x_axis(j) = n;
    
    Y_errs_lp = zeros(16, 1);
    Y_errs_bp = zeros(16, 1);
    T = 1;
    while T <= 16
      [X, Y] = data_gen_res_relu(A_g, B_g, n, 0);
      
      % A_errs(T) = norm(A_g - A) / norm(A_g);
      % B_errs(T) = norm(B_g - B) / norm(B_g);
      
      % LP
      C = lp_layer_2(X, Y);
      H = C * Y - X;
      A_unscaled = lp_layer_1(X, H);
      A_lp = rescale_layer_1(X, H, A_unscaled);
      Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
      Y_errs_lp(T) = mean(vecnorm(Y_pred_lp - Y_test) ./ vecnorm(Y_test));
      
      % BP
      [A_bp, B_bp, ~, ~] = backprop(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 256);
      Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
      Y_errs_bp(T) = mean(vecnorm(Y_pred_bp - Y_test) ./ vecnorm(Y_test));
      
      T = T + 1;
    end
    % A_errs_qp(i, j) = mean(A_errs);
    % B_errs_qp(i, j) = mean(B_errs);
    Y_errs_2d_lp(i, j) = mean(Y_errs_lp);
    Y_errs_2d_bp(i, j) = mean(Y_errs_bp);
    
  end
end

data = Y_errs_2d_bp;

% heatmap(x_axis, y_axis, Y_errs_2d_lp);
% heatmap(x_axis, y_axis, Y_errs_2d_bp);

% Define integer grid of coordinates for the above data
[H,V] = meshgrid(1:size(data,2), 1:size(data,1));

% Define a finer grid of points
[H2,V2] = meshgrid(1:0.01:size(data,2), 1:0.01:size(data,1));

% Interpolate the data and show the output
outData = interp2(H, V, data, H2, V2, 'linear');
imagesc(outData);

% Cosmetic changes for the axes
set(gca, 'XTick', linspace(1,size(H2,2),size(H,2))); 
set(gca, 'YTick', linspace(1,size(H2,1),size(H,1)));
set(gca, 'XTickLabel', x_axis);
set(gca, 'YTickLabel', y_axis);

% Add colour bar

xlabel('sample size $n$', 'Interpreter', 'latex');
ylabel('number of dimensions $d$', 'Interpreter', 'latex');
title('SGD');

colorbar;