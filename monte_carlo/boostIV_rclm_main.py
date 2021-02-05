# ======================================================================================================================

# boost IV: application to the RCLM

# ======================================================================================================================

# suppress Deprecation warnings from sklearn
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=DeprecationWarning)

# extend the path to load DeepGMM
import sys
sys.path.extend(['~/boostIV/DeepGMM'])  # add path to DeepGMM auxiliary files

import pandas as pd
import numpy as np
import time
from sklearn import preprocessing

from KIV import get_KIV
from aux_functions import deepGMM_fit
from rclm_data_gen import gen_data, data_transform
from boostIV.methods import post_boostIV_tuned, boostIV_tuned, post_boostIV_cv_kfold, boostIV_cv_kfold
from boostIV.fit import get_boostIV_fit, get_post_boostIV_fit
from boostIV.primitives import cubic_spline, sigmoid_wl


def run_sim(seed, data_spec, Ms, wl_model=sigmoid_wl, iv_model='Ridge', opt_iv=True, n_process=1, nu=0.1, ls=True, cf=True, n_split=2, n_folds=2):
    """
    This function executes one simulation and evaluates the performance for an out of sample dataset
    :param seed: specify the seed
    :param data_spec: DGP parameters
    :param Ms: grid of number of boosting iterations
    :param wl_model: type of weak learner: sigmoid or spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param opt_iv: use optimal IVs if True
    :param n_process: if > 1, then performs CV for boostIV in parallel
    :param nu: shrinkage parameter
    :param ls: line search step
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :param n_split: number of folds for cross fitting
    :return: Out of sample MSEs
    """
    ### 1. Generate data ###
    data_raw = gen_data(seed, data_spec)
    data = data_transform(data_raw)
    data_train, data_test = data['train'], data['test']

    ### 2. Get estimates: time all the estimators ###

    # 2.1. Boosting
    start_boost = time.time()
    # boost = boostIV_cv_kfold_v2(data_train, n_split, Ms, wl_model, iv_model, nu, ls, cf=cf, n_folds=n_folds, n_process=n_process)
    boost = boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv, nu, ls, cf, n_process=n_process)
    end_boost = time.time() - start_boost
    print('Boosting took ' + str(end_boost / 60) + ' minutes.')

    # 2.2. Post-Boosting
    start_pboost = time.time()
    # post_boost = post_boostIV_cv_kfold_v2(data_train, n_split, Ms, wl_model, iv_model, nu, ls, cf, n_folds, n_process)
    post_boost = post_boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv, nu, ls, cf, n_process=n_process)
    end_pboost = time.time() - start_pboost
    print('Post-boosting took ' + str(end_pboost / 60) + ' minutes.')

    ### 3. Out of sample data ###
    X_oos = data_test['X']
    h_oos = data_test['g']
    n_obs_oos = len(h_oos)

    ### 4. Evaluate out of sample performance ###

    # 4.1. KIV fit
    start_kiv = time.time()
    h_kiv_oos = get_KIV(data_train, X_oos)
    end_kiv = time.time() - start_kiv
    print('KIV took ' + str(end_kiv / 60) + ' minutes.')
    mse_kiv = np.mean((h_kiv_oos - h_oos) ** 2)
    bias_kiv = np.mean(h_kiv_oos - h_oos)

    # 4.2. deep GMM fit
    start_deep_gmm = time.time()
    h_deepGMM_oos = deepGMM_fit(data)
    end_deep_gmm = time.time() - start_deep_gmm
    print('Deep GMM took ' + str(end_deep_gmm / 60) + ' minutes.')
    mse_deep_gmm = ((h_deepGMM_oos.detach().numpy() - h_oos) ** 2).mean()
    bias_deep_gmm = (h_deepGMM_oos.detach().numpy() - h_oos).mean()

    # 4.3. Boosting fit
    x_scaler = preprocessing.StandardScaler().fit(data_train['X'])
    Xmat_oos = np.c_[np.ones(n_obs_oos), x_scaler.transform(X_oos)]
    h_boost_oos = get_boost_fit(Xmat_oos, boost['param_opt'], wl_model, boost['M_opt'], n_split)
    mse_boost = np.mean((h_boost_oos - h_oos) ** 2)
    bias_boost = np.mean(h_boost_oos - h_oos)

    # 4.4. Post-Boosting fit
    # h_post_boost_oos = post_boostIV_fit_v2(post_boost['param_cv'], wl_model, M=post_boost['M_cv'], X=Xmat_oos, n_split=n_split, cf=cf)  # CV version
    h_post_boost_oos = post_boostIV_fit_v2(post_boost['param_opt'], wl_model, M=post_boost['M_opt'], X=Xmat_oos, n_split=n_split, cf=cf)
    mse_post_boost = np.mean((h_post_boost_oos - h_oos) ** 2)
    bias_post_boost = np.mean(h_post_boost_oos - h_oos)

    # check
    print('')
    print('=====================================')
    print('Simulation run finished: ' + str(seed))
    print('=====================================')
    print('')

    # consolidate results
    mse_out = [float(bias_kiv), float(bias_deep_gmm), float(bias_boost), float(bias_post_boost),
               mse_kiv, mse_deep_gmm, mse_boost, mse_post_boost]
    fit_out = np.vstack([h_kiv_oos, h_deepGMM_oos.detach().numpy().flatten(), h_boost_oos, h_post_boost_oos]).T
    return {'mse': mse_out, 'fit': fit_out}


def run_mc(n_runs, data_spec, Ms, wl_model=sigmoid_wl, iv_model='Ridge', opt_iv=True, n_process=2, nu=0.1, ls=True, cf=True, n_split=2, n_folds=2):
    # run MC loop
    out_list = []
    fit_list = []
    out_mat = np.zeros((n_runs, 12))
    for n in range(n_runs):
        out_list.append(run_sim(n, data_spec, Ms, wl_model, iv_model, opt_iv, n_process, nu, ls, cf, n_split, n_folds))
        out_mat[n, :] = out_list[n]['mse']
        fit_list.append(out_list[n]['fit'])
    # average fit across simulations
    fit_mc = np.mean(fit_list, axis=0)
    # convert into a pandas df
    out_mse = pd.DataFrame(out_mat, columns=['Bias KIV', 'Bias DeepGMM', 'Bias boostIV', 'Bias post-boostIV',
                                             'MSE KIV', 'MSE DeepGMM', 'MSE boostIV', 'MSE post-boostIV'])
    out_fit = pd.DataFrame(fit_mc, columns=['KIV', 'DeepGMM', 'boostIV', 'post-boostIV'])
    return {'mse': out_mse, 'fit': out_fit}


if __name__ == "__main__":
    # specify parameters
    n_runs = 1
    T = 100
    J = 10
    K_2 = 10
    data_spec = {'distr': 'indep-normal', 'design': 'Normal', 'T': T, 'J': J,
                 'N': 2000, 'beta_true': np.append([-1, 1], np.ones(K_2)), 'K_1': 1, 'K_2': K_2, 'K_w': 1, 'price_rc': True}
    # grid for boosting iterations
    Ms = [15, 25, 35, 45]
    # run and save output
    mc_mse = run_mc(n_runs, data_spec, Ms, opt_iv=False)

