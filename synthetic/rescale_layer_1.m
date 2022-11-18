function A_ = rescale_layer_1(X, Y, A)

[d, ~] = size(X);
A_ = zeros(size(A));

for j = 1 : d
  k = (A(j, :) * X(:, A(j, :) * X > 0)) / Y(j, A(j, :) * X > 0);
  A_(j, :) = A(j, :) / k;
end

end