function obj = obj_layer_1(A, Eta, X, Xi, lambda)

[~, N] = size(X);
obj = sum(sum_square(A * X + Eta - Xi)) / N + lambda * sum(sum_square(A)); 

end
