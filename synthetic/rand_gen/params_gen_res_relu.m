function [A, B] = params_gen_res_relu(m, d)
  A = abs(randn(d, d));
  %A = randi([0 3], d, d);
  %A(A >= 2) = 0;
  %A = eye(d);% + abs(randn(d, d) * 1e-2);
  %A = zeros(d);
  %A(1, 1) = 1;
  %A(1, 2) = 1;
  %A(1, 3) = 1;
  %A(2, 3) = 1;
  %A(3, 2) = 1;
  %A(3, 1) = 1;
  %A(4, 3) = 1;
  %A(4, 5) = 1;
  %A(3, 3) = 1;
  %A(7, 2) = 1;
  %A(7, 6) = 1;
  %A(7, 8) = 1;
  %A(2, 10) = 1;
  %A(1, d) = 1;
  %A(d, 1) = 1;
  %A = triu(A);
  %A(1, 1) = 100;
  %for i = 1 : d
  %  A(1, i) = 1;
  %end
  %A(d, 1) = 1;
  %B = randn(l, d);
  B = randn(m, d);
  %B = eye(l, d)
end