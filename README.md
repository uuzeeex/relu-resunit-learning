# Two-Layer ReLU Residual Unit Learning

Two-layer ReLU residual block solvers using QP/LP with baselines, as code repository for TMLR paper *Nonparametric Learning of Two-Layer ReLU Residual Units* ([link](https://openreview.net/forum?id=YiOI0vqJ0n)).

## Before you run

- Install `cvx` MATLAB software, guide [here](http://web.cvxr.com/cvx/doc/install.html).
- Make sure the repository path to MATLAB environment is added **recursively**.

## Scripts

- Under `synthetic` (for main paper Section 7.1 unless specified):
  - `script_exp_sample_eff.m`, `script_exp_sample_eff_heatmap_gen.m`: Sample efficiency experiments.
  - `script_exp_weight_rob.m`: Network weight robustness experiments.
  - `script_exp_noise_rob.m`: Noise robustness experiments.
  - `script_exp_running_time.m` : Running time efficiency versus SGD experiments.
  - `script_exp_cond.m`: Layer 2 weights condition number robustness experiments.
  - `vanilla_lr/`: Vanilla linear regression discussed in main paper Section 3.
  - `rand_gen/`: Random matrix and data generator used in the experiments.

- Under `benchmark`:
  - Run `do_all_datasets_init.m` followed by `make_table.py` to recreate the table in main paper Section 7.2.
