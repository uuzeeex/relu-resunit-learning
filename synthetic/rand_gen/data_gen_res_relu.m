function [X, Y] = data_gen_res_relu(A_g, B_g, N, noise_std)

[l, d] = size(B_g);

%X = randn(d, N);
X = zeros(d, N);
for i = 1 : N
  for j = 1 : d
    p = rand;
    if p < 0.5
      X(j, i) = randn - 1 / 10;
    else
      X(j, i) = rand * 2 - 9 / 10;
    end
  end
end
Y = B_g * (max(A_g * X, 0) + X) + randn(l, N) * noise_std;
%Y = B_g * (max(A_g * X, 0));

end