function [err, tr_err, err_bp, tr_err_bp, my_seed, c] = test_data_init(dataset_name, X, y_orig, y_class, typeof_y, normalize_data, bpparams)

%% Assume X is n x d, y_orig is n x yd
%% typeof can be 0 - regression, 1 - classification

total_err_bp = 0;
total_err_qp = 0;

MX = max(y_orig);
MN = min(y_orig);

my_seed = randi([0, 2000], 1);

rng(my_seed);

%% k for cross-validation
k = 5;

%% this is adding bias, and what's special about rectb.
X = [X'; ones(1, size(X,1))]';

[n, d] = size(X);
[n2, yd] = size(y_orig);

if (yd > d)
  error('yd > d not tolerated - dimension of y must be smaller or equal to dimension of x')
end

v_ = randperm(n);
env{6}.v_ = v_;

b = floor(n / k);
v = v_(1 : (b * k))';
t = numel(v);
c = mat2cell(v, diff([0 : b : t - 1, t]));

c{k} = [c{k}; (v_((b * k + 1) : end))'];

%% c now contains a random k-fold indices for the test set -- the rest is the training set

%% duplicate the Y if needed, and take the subset so that we have Y of size n x d
Y = repmat(y_orig, 1, floor((d / yd) + 2));
Y = Y(:, 1 : d);
fprintf('Y size %d \n', size(Y, 2));

if (yd > 1)
	k_ = 0;

	for i = (yd+1):size(Y,2)
		k_ = k_ + 1;
		if (k_ > yd)
			k_ = 1;
		end
		%Y(:,i) = Y(:, i) + normrnd(0, 0.1*std(Y(:,k_)), n, 1);
		%Y(:,i) = Y(:, i); % + normrnd(0, 1.0, n, 1);
	end
	for i = 1 : yd
		fprintf('[%s] %d  mean of Y: %.5f     std of Y: %.5f\n', i, dataset_name, mean(Y(:,i)), std(Y(:,i)));
	end
else
	Y(:, (yd + 1) : end) = Y(:, (yd + 1) : end) + normrnd(0, 0.1 * std(Y(:, 1)), n, d - yd);
	fprintf('[%s]   mean of Y: %.5f     std of Y: %.5f\n', dataset_name, mean(Y(:, 1)), std(Y(:, 1)));
end

err = zeros(k, yd);
tr_err = zeros(k, yd);
err_bp = zeros(k, yd);
tr_err_bp = zeros(k, yd);

env{6}.X = X;
env{6}.Y = Y;

disp('Starting')

for i=1:k
%for i=1:1
	fprintf('**** Fold %d\n', i);

	fold_tr = [];

	for j=1:k
		if i ~= j
			fold_tr = [c{j}; fold_tr];
		end
	end

	fold_ts = c{i};
	lx = (length(fold_ts));
	half = ceil(lx / 2);

	fold_dv = fold_ts(1 : half);
	fold_ts = fold_ts(half + 1 : end);

	env{i}.fold_tr = fold_tr;
	env{i}.fold_ts = fold_ts;

	%% now fold_tr includes 4/5 of the data, and fold_ts 1/5 -- according to the current fold

	X_old = X;

	mn = mean(X(fold_tr, 1 : end - 1));
	mn2 = mean(Y(fold_tr, 1 : yd));
	st = std(X(fold_tr, 1 : end - 1));
	st2 = std(Y(fold_tr, 1 : yd));
	st(st == 0) = 1;
	MX = max(y_orig);
	MN = min(y_orig);

	if (normalize_data == 1)
		X(:, 1 : end - 1) = X(:, 1 : end - 1) - repmat(mn, size(X, 1), 1);
		X(:, 1 : end - 1) = X(:, 1 : end - 1) ./ repmat(st, size(X, 1), 1);
	end

  % QP
  lambda1 = 0;
  lambda2 = 0;

  [C_qp, H_qp] = qp_layer_2(X(fold_tr, :)', Y(fold_tr, :)', lambda1);
  B_qp1 = pinv(full(C_qp));
  A_unscaled = qp_layer_1(X(fold_tr, :)', H_qp, lambda2);
  A_qp1 = rescale_layer_1(X(fold_tr, :)', H_qp, A_unscaled);
  Y_pred = B_qp1 * (max(A_qp1 * X', 0) + X');
  %% was C_qp \

	% LP
    %%C_lp = lp_noise_layer_2(X(fold_tr,:)', Y(fold_tr,:)');
    %%B_lp = inv(C_lp);
    %%H_lp = C_lp * Y(fold_tr,:)' - X(fold_tr,:)';
    %%A_unscaled = lp_noise_layer_1(X(fold_tr,:)', H_lp);
    %%A_lp = rescale_layer_1(X(fold_tr,:)', H_lp, A_unscaled);

    %%A_qp1 = A_lp;
    %%B_qp1 = B_lp;
    %%C_qp = C_lp;
    %%Y_pred = C_qp \ (max(A_qp1 * X', 0) + X');


	%Y_pred = B_qp1 * (max(A_qp1 * X',0) + X');
	env{i}.A_qp = A_qp1;
	env{i}.B_qp = B_qp1;
	env{i}.A_unscaled_qp = A_unscaled;
	env{i}.Y_pred_qp = Y_pred';

	if typeof_y == 0
		relevant_Y_pred = Y_pred(1:yd, :)';
		relevant_Y_pred = min(relevant_Y_pred, MX);
		relevant_Y_pred = max(relevant_Y_pred, MN);
		err(i, :) = sqrt((1/length(fold_ts)) * sum((relevant_Y_pred(fold_ts, :) - Y(fold_ts,1:yd)).^2));
		tr_err(i, :) = sqrt((1/length(fold_tr)) * sum((relevant_Y_pred(fold_tr, :) - Y(fold_tr,1:yd)).^2));
		total_err_qp = total_err_qp + err(i, 1);
	end

	if (typeof_y == 1) || (typeof_y == 2)
		if yd == 1
      relevant_Y_pred = Y_pred(1:yd, :)';
		  % err(i, :) = ((1/length(fold_ts))  * sum((relevant_Y_pred(fold_ts, 1)>0 - Y(fold_ts,1)).^2));
		  % err(i, :) = mean(abs(relevant_Y_pred(fold_ts, 1)>0 - Y(fold_ts,1)))
		  % tr_err(i, :) = ((1/length(fold_tr)) * sum((relevant_Y_pred(fold_tr, 1)>0 - Y(fold_tr,1)).^2));
      err(i, :) = ((1/length(fold_ts)) * sum((relevant_Y_pred(fold_ts, 1) > 0 ~= y_class(fold_ts))));
      tr_err(i, :) = ((1/length(fold_tr)) * sum((relevant_Y_pred(fold_tr, 1) > 0 ~= y_class(fold_tr))));
		else
		[tt, mx] = max(Y_pred(1 : yd, :));
		for u = fold_ts'
			err(i, :) = err(i, :) + Y(u, mx(u));
		end
		err(i, :) = 1 - (err(i, :) / length(fold_ts));
		for u = fold_tr'
			tr_err(i, :) = tr_err(i, :) + Y(u, mx(u));
		end
		tr_err(i, :) = 1-(tr_err(i, :) / length(fold_tr));

		%err(i, :) = ((1/length(fold_ts)) * sum((relevant_Y_pred(fold_ts, :) - Y(fold_ts,1:yd)).^2)));
		%tr_err(i, :) = ((1/length(fold_tr)) * 0.5 * sum(sum((relevant_Y_pred(fold_tr, :) - Y(fold_tr,1:yd)).^2)));
		end
	end
	%end %%% delete
	toc
	fprintf('[%s fold %d]   # examples: %d    qp_err %.5f     qp_train_err %.5f    lambda1 %.5f   lambda2 %.5f    mean: %s      std: %s\n', dataset_name, i, length(fold_ts), err(i,1), tr_err(i,1), lambda1, lambda2, mat2str(mn2), mat2str(st2));

	tic
	%% backprop - don't stop based on losses from test set, that would be "cheating"

	mn_err = sqrt((1 / length(fold_ts)) * sum((mn2 - Y(fold_ts, 1 : yd)) .^ 2));
	fprintf('[%s fold %d]   # examples: %d    mn_err %.5f      mean: %s      std: %s\n', dataset_name, i, length(fold_ts), mn_err, mat2str(mn2), mat2str(st2));

	for myinit = 0 : 5
    lambda1 = 0;
    lambda2 = 0;
    [A_bp1_, B_bp1_, L, iternum, losses] = backprop_with_init(X(fold_tr, :)', Y(fold_tr, :)', X(fold_dv, :)', Y(fold_dv, :)', 500, bpparams(1), bpparams(2), bpparams(3), lambda1, lambda2, A_qp1, B_qp1, myinit);

    L_old = L;
    A_bp1 = A_bp1_;
    B_bp1 = B_bp1_;
    Y_pred_bp = B_bp1 * (max(A_bp1 * X',0) + X');
    relevant_Y_pred_bp = Y_pred_bp(1 : yd, :)';
    Y_pred_bp = B_bp1 * (max(A_bp1 * X',0) + X');

    env{i}.bp{myinit + 1}.Y_pred_bp = Y_pred_bp';
    env{i}.bp{myinit + 1}.A_bp = A_bp1;
    env{i}.bp{myinit + 1}.B_bp = B_bp1;
    env{i}.bp{myinit + 1}.losses = losses;

	%% calculate error, 0 for regression, 1 for classification (assuming one-hot encoding for Y)
    if typeof_y == 0
      relevant_Y_pred_bp = Y_pred_bp(1:yd, :)';
      relevant_Y_pred_bp = min(relevant_Y_pred_bp, MX);
      relevant_Y_pred_bp = max(relevant_Y_pred_bp, MN);
      err_bp(i, :) = sqrt((1/length(fold_ts)) * sum((relevant_Y_pred_bp(fold_ts, :) - Y(fold_ts,1:yd)).^2));
      tr_err_bp(i, :) = sqrt((1/length(fold_tr)) * sum((relevant_Y_pred_bp(fold_tr, :) - Y(fold_tr,1:yd)).^2));
      total_err_bp = total_err_bp + err_bp(i, 1);
    end

    if (typeof_y == 1) || (typeof_y == 2)
      %% we assume Y is represented a one-hot encoding of the class
      if yd == 1
        relevant_Y_pred_bp = Y_pred_bp(1:yd, :)';
        %err_bp(i, :) = ((1/length(fold_ts))  * sum((relevant_Y_pred_bp(fold_ts, 1)>0 - Y(fold_ts,1)).^2));
        %err_bp(i, :) = mean(abs(relevant_Y_pred_bp(fold_ts, 1)>0 - Y(fold_ts,1)))
        %tr_err_bp(i, :) = ((1/length(fold_tr)) * sum((relevant_Y_pred_bp(fold_tr, 1)>0 - Y(fold_tr,1)).^2));
        err_bp(i, :) = ((1/length(fold_ts)) * sum((relevant_Y_pred_bp(fold_ts, 1)>0 ~= y_class(fold_ts))));
        tr_err_bp(i, :) = ((1/length(fold_tr)) * sum((relevant_Y_pred_bp(fold_tr, 1)>0 ~= y_class(fold_tr))));
      else
        [tt, mx] = max(Y_pred_bp(1 : yd, :));
        for u = fold_ts'
          err_bp(i, :) = err_bp(i, :) + Y(u, mx(u));
        end
        err_bp(i, :) = 1-(err_bp(i, :) / length(fold_ts));
        for u = fold_tr'
          tr_err_bp(i, :) = tr_err_bp(i, :) + Y(u, mx(u));
        end
        tr_err_bp(i, :) = 1-(tr_err_bp(i, :) / length(fold_tr));
      end
    end
    fprintf('[%s fold %d with %d epochs]   # examples: %d    myinit: %d (0 is qp)    bp_err %.5f     bp_train_err %.5f    lambda1 %.5f   lambda2 %.5f       mean: %s      std: %s\n', dataset_name, i, iternum, length(fold_ts), myinit, err_bp(i,1), tr_err_bp(i,1), lambda1, lambda2, mat2str(mn2), mat2str(st2));
    toc
  end
	X = X_old;
end

disp('size Y')
size(Y)
disp('size X')
size(X)
