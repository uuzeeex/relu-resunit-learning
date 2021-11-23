N_test = 128;
T_gt = 32; %64;

d = 16;
d_idx = 2;
n = 512;
%n = 16;

A_errs_lp = zeros(T_gt, 1);
B_errs_lp = zeros(T_gt, 1);
Y_errs_lp = zeros(T_gt, 1);

A_errs_bp = zeros(T_gt, 1);
B_errs_bp = zeros(T_gt, 1);
Y_errs_bp = zeros(T_gt, 1);

cond_nums = [1e0, 3e0, 1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5];
lambdas = cond_nums .^ (1 / d);

num_conds = size(lambdas, 2);

% B_errs_lp_by_cond = zeros(num_conds, 3);
% B_errs_bp_by_cond = zeros(num_conds, 3);

for i = 1 : num_conds
  eigs = zeros(d, 1);
  lambda = lambdas(i);
  for p = 1 : d
    eigs(p) = lambda ^ (-p);
  end

  for T_gt_it = 1 : T_gt
    [A_g, ~] = params_gen_res_relu(d, d);
    B_g = rand_orth_mat(d) * diag(eigs) * rand_orth_mat(d).';
    [X_test, Y_test] = data_gen_res_relu(A_g, B_g, N_test, 0);

    [X, Y] = data_gen_res_relu(A_g, B_g, n, 0);

    % lp
    C = relulp2_layer2(X, Y);
    B_lp = inv(C);
    H = C * Y - X;
    A_unscaled = relulp2_layer1(X, H);
    A_lp = rescale_layer1(X, H, A_unscaled);
    Y_pred_lp = C \ (max(A_lp * X_test, 0) + X_test);
  
    A_errs_lp(T_gt_it) = norm(A_lp - A_g) / norm(A_g);
    B_errs_lp(T_gt_it) = norm(B_lp - B_g) / norm(B_g);
    Y_errs_lp(T_gt_it) = mean(vecnorm(Y_pred_lp - Y_test) ./ vecnorm(Y_test));
  
    % bp
    [A_bp, B_bp, ~, ~] = backprop2(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);

    A_errs_bp(T_gt_it) = norm(A_bp - A_g) / norm(A_g);
    B_errs_bp(T_gt_it) = norm(B_bp - B_g) / norm(B_g);
    Y_errs_bp(T_gt_it) = mean(vecnorm(Y_pred_bp - Y_test) ./ vecnorm(Y_test));
  end
  B_errs_lp_by_cond(i, d_idx) = mean(B_errs_lp);
  B_errs_bp_by_cond(i, d_idx) = mean(B_errs_bp);
end

% semilogx(cond_nums, B_errs_bp_by_cond(:, d_idx), '-o', 'LineWidth', 2);
% 
% semilogx(cond_nums, B_errs_bp_by_cond(:, 1), '-o', 'LineWidth', 2);
% hold on
% semilogx(cond_nums, B_errs_bp_by_cond(:, 2), '-s', 'LineWidth', 2);
% hold on
% semilogx(cond_nums, B_errs_bp_by_cond(:, 3), '-x', 'LineWidth', 2);
% hold on
% xlabel('$\kappa(${\boldmath${B}$}$^\ast)$', 'Interpreter', 'latex', 'FontSize', 16);
% ylabel('Layer 2 Error', 'Interpreter', 'latex', 'FontSize', 16);
% legend('$d = 8$', '$d = 16$', '$d = 32$', 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'northwest');
% title('SGD', 'FontSize', 16);
% grid on

loglog(cond_nums, B_errs_lp_by_cond(:, 1), '-o', 'LineWidth', 2);
hold on
loglog(cond_nums, B_errs_lp_by_cond(:, 2), '-s', 'LineWidth', 2);
hold on
loglog(cond_nums, B_errs_lp_by_cond(:, 3), '-x', 'LineWidth', 2);
hold on
xlabel('$\kappa(${\boldmath${B}$}$^\ast)$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Layer 2 Error', 'Interpreter', 'latex', 'FontSize', 16);
legend('$d = 8$', '$d = 16$', '$d = 32$', 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'northwest');
title('LP', 'FontSize', 16);
grid on