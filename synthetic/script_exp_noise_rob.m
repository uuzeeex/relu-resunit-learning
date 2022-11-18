d = 10;
m = 10;

N = 512;
N_test = 128;

[A_g, B_g] = params_gen_res_relu(d, m);
[X_test, Y_test] = data_gen_res_relu(A_g, B_g, N_test, 0);

T = 1;
noise_str = [0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0];
risks_u_qp = zeros(9, 1);
risks_u_lp = zeros(9, 1);
risks_u_bp = zeros(9, 1);

A_err_u_qp = zeros(9, 1);
A_err_u_lp = zeros(9, 1);
A_err_u_bp = zeros(9, 1);

B_err_u_qp = zeros(9, 1);
B_err_u_lp = zeros(9, 1);
B_err_u_bp = zeros(9, 1);

while T <= 9
  trials = 1;
  risks_t_qp = zeros(8, 1);
  risks_t_lp = zeros(8, 1);
  risks_t_bp = zeros(8, 1);
  
  A_errs_t_qp = zeros(8, 1);
  A_errs_t_lp = zeros(8, 1);
  A_errs_t_bp = zeros(8, 1);
  
  B_errs_t_qp = zeros(8, 1);
  B_errs_t_lp = zeros(8, 1);
  B_errs_t_bp = zeros(8, 1);
  
  while trials <= 16
    [X, Y] = data_gen_res_relu(A_g, B_g, N, noise_str(T));
    
    % QP
    [C_qp, H_qp] = qp_layer_2(X, Y);
    B_qp = inv(C_qp);
    A_unscaled = qp_layer_1(X, H_qp);
    A_qp = rescale_layer_1(X, H_qp, A_unscaled);
    Y_pred_qp = C_qp \ (max(A_qp * X_test, 0) + X_test);
    risks_t_qp(trials) = mean(vecnorm(Y_pred_qp - Y_test) ./ vecnorm(Y_test));
    A_errs_t_qp(trials) = norm(A_qp - A_g) / norm(A_g);
    B_errs_t_qp(trials) = norm(B_qp - B_g) / norm(B_g);
    
    % LP
    C_lp = lp_noise_layer_2(X, Y);
    B_lp = inv(C_lp);
    H_lp = C_lp * Y - X;
    A_unscaled = lp_noise_layer_1(X, H_lp);
    A_lp = rescale_layer_1(X, H_lp, A_unscaled);
    Y_pred_lp = C_lp \ (max(A_lp * X_test, 0) + X_test);
    risks_t_lp(trials) = mean(vecnorm(Y_pred_lp - Y_test) ./ vecnorm(Y_test));
    A_errs_t_lp(trials) = norm(A_lp - A_g) / norm(A_g);
    B_errs_t_lp(trials) = norm(B_lp - B_g) / norm(B_g);
    
    % BP
    [A_bp, B_bp, ~, ~] = backprop(X, Y, X_test, Y_test, 32, 1e-3, 1e-5, 256);
    Y_pred_bp = B_bp * (max(A_bp * X_test, 0) + X_test);
    risks_t_bp(trials) = mean(vecnorm(Y_pred_bp - Y_test) ./ vecnorm(Y_test));
    A_errs_t_bp(trials) = norm(A_bp - A_g) / norm(A_g);
    B_errs_t_bp(trials) = norm(B_bp - B_g) / norm(B_g);
    
    trials = trials + 1;
  end
  risks_u_qp(T) = mean(risks_t_qp);
  risks_u_lp(T) = mean(risks_t_lp);
  risks_u_bp(T) = mean(risks_t_bp);
  
  A_err_u_qp(T) = mean(A_errs_t_qp);
  A_err_u_lp(T) = mean(A_errs_t_lp);
  A_err_u_bp(T) = mean(A_errs_t_bp);
  
  B_err_u_qp(T) = mean(B_errs_t_qp);
  B_err_u_lp(T) = mean(B_errs_t_lp);
  B_err_u_bp(T) = mean(B_errs_t_bp);
  % disp(mean(vecnorm(Y_pred - Y_test) ./ vecnorm(Y_test)));
  T = T + 1;
end

plot(noise_str, A_err_u_bp, '-o', 'LineWidth', 2);
hold on
plot(noise_str, A_err_u_lp, '-s', 'LineWidth', 2);
hold on
plot(noise_str, A_err_u_qp, '-x', 'LineWidth', 2);

legend('SGD', 'LP', 'QP');

plot(noise_str, B_err_u_bp, '-o', 'LineWidth', 2);
hold on
plot(noise_str, B_err_u_lp, '-s', 'LineWidth', 2);
hold on
plot(noise_str, B_err_u_qp, '-x', 'LineWidth', 2);

legend('SGD', 'LP', 'QP');

plot(noise_str, risks_u_bp, '-o', 'LineWidth', 2);
hold on
plot(noise_str, risks_u_lp, '-s', 'LineWidth', 2);
hold on
plot(noise_str, risks_u_qp, '-x', 'LineWidth', 2);

legend('SGD', 'LP', 'QP');