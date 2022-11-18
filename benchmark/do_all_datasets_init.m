%% Supporting code Nonparametric Learning of Two-Layer ReLU Residual Units, https://openreview.net/forum?id=YiOI0vqJ0n.

cd cvx

cvx_setup

cvx_solver mosek
cvx_solver_settings('MSK_IPAR_NUM_THREADS', 25)
cvx_save_prefs

cd ..

clear;

diary('output_of_do_all_datasets_init.txt');

load('datasets/combined');

%% learning rate
LR = 0.000001;

bpparams = [LR, LR, 2000000];


X = delta_ailerons(:, delta_ailerons_X_atts);
Y = delta_ailerons(:, delta_ailerons_Y_atts);

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('delta-ailerons', X, Y, [], 0, 0, bpparams);

save('i2-delta_ailerons');

X = redwine(:, redwine_X_atts);
Y = redwine(:, redwine_Y_atts);

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('redwine', X, Y, [], 0, 1, bpparams);

save('i2-redwine');


load('datasets/matlab');

X = whitewine(:, whitewine_X_atts);
Y = whitewine(:, whitewine_Y_atts);

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('whitewine', X, Y, [], 0, 1, bpparams);

save('i2-whitewine');


X = ailerons(:, ailerons_X_atts);
Y = ailerons(:, ailerons_Y_atts);

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('ailerons', X, Y, [], 0, 1, bpparams);
save('i2-ailerons');


X = elev(:, elev_X_atts);
Y = elev(:, elev_Y_atts);

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('delta-elevators', X, Y, [], 0, 0, bpparams);
save('i2-elev');

X = housing(:, housing_X_atts);
Y = housing(:, housing_Y_atts);

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('housing', X, Y, [], 0, 1, bpparams);

save('i2-housing');


load('datasets/fasttext-jigsaw-100D');

[err_qp, tr_err_qp, err_bp, tr_err_bp, env, my_seed, c] = test_data_init('fasttext-jigsaw-100d', X, Y, [], 0, 0, bpparams);

save('i2-fasttext-results');

diary off;
