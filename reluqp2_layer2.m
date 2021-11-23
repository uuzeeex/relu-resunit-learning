function [C, xi] = reluqp2_layer2(X, Y)

[n, N] = size(X);
[l, ~] = size(Y);

cvx_begin
  cvx_precision medium
  variable C(n, l)
  variable xi(n, N)
  obj = obj_layer2(C, xi, X, Y);
  minimize(obj)
  subject to
    xi >= 0
cvx_end

%B = inv(C);

end