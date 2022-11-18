function [A, B, X_n, X_p] = lr(X, Y)

[~, N] = size(X);

X_n = X(:, all(X(:, 1 : floor(N/2)) <= 0));
Y_n = Y(:, all(X(:, 1 : floor(N/2)) <= 0));

B = 1;%Y_n / X_n;
X_p = X(:, floor(N/2) + 1 : end);
Y_p = Y(:, floor(N/2) + 1 : end);
idx = all(X_p >= 0);
X_p = X_p(:, idx);
Y_p = Y_p(:, idx);

A = 1;%(B \ Y_p - X_p) / X_p;

end