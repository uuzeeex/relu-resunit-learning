function A = lp_layer_1(X, Y)

[d, N] = size(X);

cvx_begin
  variable A(d, d)
  obj = 0;
  minimize(obj)
  subject to
    Y - A * X >= 0
cvx_end

end