function [A, Eta] = qp_layer_1(X, Xi, lambda)

[n, N] = size(X);

cvx_begin
  variable A(n, n)
  variable Eta(n, N)
  obj = obj_layer_1(A, Eta, X, Xi, lambda);
  minimize(obj)
  subject to
    Eta >= 0
    A >= 0
cvx_end

end
