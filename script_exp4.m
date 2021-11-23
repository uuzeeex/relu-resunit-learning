N_test = 128;
T_gt = 5;%64;

d = 8; % [8, 16, 32]
time_bp = 2; % [2, 5, 10]
n = 512;
%n = 16;

Y_errs_lp = zeros(T_gt, 1);

Y_errs_bp = zeros(T_gt, 1);

times_lp = zeros(T_gt, 1);
errors_lp = zeros(T_gt, 1);

legend_names_bp = "SGD-GT_" + string(1 : T_gt);
legend_names_lp = "LP-GT_" + string(1 : T_gt);
markers_lp = ['o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'];

i = 1;

for T_gt_it = 1 : T_gt
  [A_g, B_g] = params_gen_res_relu(d, d);
  [X_test, Y_test] = data_gen_res_relu(A_g, B_g, N_test, 0);
  [X, Y] = data_gen_res_relu(A_g, B_g, n, 0);
  
  % lp
  tic
  C = relulp2_layer2(X, Y);
  B_lp = inv(C);
  H = C * Y - X;
  A_unscaled = relulp2_layer1(X, H);
  A_lp = rescale_layer1(X, H, A_unscaled);
  timestamp_lp_done = toc;
  
  Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
  Y_err_lp = mean(vecnorm(Y_pred_lp - Y_test) ./ vecnorm(Y_test));
    
  % bp
  [A_bp, B_bp, timestamps, errors] = backprop2(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 1000000, time_bp);
  
  Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
  
  Y_err_bp = mean(vecnorm(Y_pred_bp - Y_test) ./ vecnorm(Y_test));
  
  plot(timestamps, log10(errors), 'LineWidth', 2);
  hold on
  times_lp(T_gt_it) = timestamp_lp_done;
  errors_lp(T_gt_it) = log10(Y_err_lp);
end

for T_gt_it = 1 : T_gt
  plot(times_lp(T_gt_it), errors_lp(T_gt_it), markers_lp(T_gt_it), 'MarkerSize', 8, 'LineWidth', 2);
  hold on
end

legend(reshape([legend_names_bp, legend_names_lp], [T_gt * 2, 1]), 'NumColumns', 2);
xlabel('Running Time (s)', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Output Error (in $\log_{10}$)', 'Interpreter', 'latex', 'FontSize', 16);
title(['$d = ', num2str(d), '$'], 'Interpreter', 'latex', 'FontSize', 16);
grid on