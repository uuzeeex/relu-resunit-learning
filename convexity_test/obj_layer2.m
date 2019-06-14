function obj = obj_layer2(C, xi, X, Y)

[~, N] = size(X);
obj = sum(sum_square(C * Y - xi - X)) / N;

end
