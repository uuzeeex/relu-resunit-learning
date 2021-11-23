function C = relulp2_layer2(X, Y)

[d, N] = size(X);
[m, ~] = size(Y);

cvx_begin
  variable C(d, m)
  minimize(0)
  subject to
    C * Y - X >= 0
cvx_end

%B = inv(C);

end