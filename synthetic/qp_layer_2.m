function [C, Xi] = qp_layer_2(X, Y)

[n, N] = size(X);
[l, ~] = size(Y);

cvx_begin
  cvx_precision medium
  variable C(n, l)
  variable Xi(n, N)
  obj = obj_layer_2(C, Xi, X, Y);
  minimize(obj)
  subject to
    Xi >= 0
cvx_end

%B = inv(C);

end