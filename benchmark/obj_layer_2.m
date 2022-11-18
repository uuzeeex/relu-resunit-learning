function obj = obj_layer_2(C, Xi, X, Y)

[~, N] = size(X);
obj = sum(sum_square(C * Y - Xi - X)) / N; 

end
