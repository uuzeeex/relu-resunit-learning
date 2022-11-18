function C = lp_layer_2(X, Y)

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