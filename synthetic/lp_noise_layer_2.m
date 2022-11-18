function C = lp_noise_layer_2(X, Y)

[d, N] = size(X);
[m, ~] = size(Y);

cvx_begin
  variable C(d, m)
  variable Zeta(d, N)
  obj = sum(sum(Zeta)) / N;
  minimize(obj)
  subject to
    C * Y - X >= -Zeta
    Zeta >= 0
cvx_end

%disp(Zeta)
%disp(sum(sum(Zeta)))
%B = inv(C);

end