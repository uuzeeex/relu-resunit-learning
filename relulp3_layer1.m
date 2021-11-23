function A = relulp3_layer1(X, Y)

[d, N] = size(X);
[m, ~] = size(Y);

cvx_begin
  variable A(d, d)
  variable Zeta(d, N)
  obj = sum(sum(Zeta)) / N;
  minimize(obj)
  subject to
    Y - A * X >= -Zeta
    Zeta >= 0
cvx_end

%disp(Zeta)
%disp(sum(sum(Zeta)))
%B = inv(C);

end