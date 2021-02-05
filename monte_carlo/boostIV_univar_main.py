# ======================================================================================================================

# boostIV MC: Univariate design

# ======================================================================================================================

import pandas as pd
import numpy as np
import time
from sklearn import preprocessing

# suppress Deprecation warnings from sklearn
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=DeprecationWarning)

import sys
sys.path.extend(['~/boostIV/DeepGMM'])  # add path to DeepGMM auxiliary files

from KIV import get_KIV
from aux_functions import npiv_fit, get_npiv_fit, deepiv_fit, deepGMM_fit
from boostIV.methods import post_boostIV_tuned, boostIV_tuned, post_boostIV_cv_kfold, boostIV_cv_kfold
from boostIV.fit import get_boostIV_fit, get_post_boostIV_fit
from boostIV.primitives import cubic_spline, sigmoid_wl


############################################################
# Generate Data
############################################################

# Simulation design is based on the DeepGMM paper (p.8)
def gen_data(data_spec, seed):
    """
    Univariate function
    :param rho: controls the degree of endogeneity
    :param fun_type: type of the structural function: log or sin
    """
    # DGP parameters
    n_obs = data_spec['n_obs']
    rho = data_spec['rho']
    fun_type = data_spec['fun_type']
    # set a seed
    np.random.seed(seed)
    # draw (e, V, W)
    Z = np.random.uniform(low=-3, high=3, size=(n_obs, 2))
    e = np.random.normal(size=n_obs)
    delta = np.random.normal(loc=0, scale=np.sqrt(0.1), size=n_obs)
    gamma = np.random.normal(loc=0, scale=np.sqrt(0.1), size=n_obs)
    X = np.sum(Z, axis=1) + e + gamma
    # structural function
    if fun_type == 'log':
        h_fun = np.log(np.abs(16 * X - 8) + 1) * np.sign(X - 0.5)
    elif fun_type == 'sin':
        h_fun = np.sin(X)
    elif fun_type == 'step':
        h_fun = 1. * (X < 0) + 2.5 * (X >= 0)
    elif fun_type == 'abs':
        h_fun = np.abs(X)
    else:
        sys.exit('No structural function matched: pick log, sin, step, or abs.')
    Y = h_fun + rho * e + delta
    df_out = {'X': X, 'Y': Y, 'Z': Z, 'h': h_fun}
    return df_out


def data_transform(data):
    ### 1. stransform data ###
    # extract the data
    X = data['X']
    Y = data['Y']
    Z = data['Z']
    g = data['h']

    ### 2. split data into train, dev, and test subsamples ###
    n = len(Y)
    # 50% train, 25% dev, 25% test
    n_train = n // 2
    n_dev = (n - n_train) // 2

    # indices for splits
    train_id = range(n_train)
    dev_id = range(n_train, n_train + n_dev)
    test_id = range(n_train + n_dev, n)

    # do splits
    X_train, Z_train, Y_train, g_train = X[train_id], Z[train_id], Y[train_id], g[train_id]
    X_dev, Z_dev, Y_dev, g_dev = X[dev_id], Z[dev_id], Y[dev_id], g[dev_id]
    X_test, Z_test, Y_test, g_test = X[test_id], Z[test_id], Y[test_id], g[test_id]

    # store as dictionaries
    data_train = {'X': X_train, 'Y': Y_train, 'Z': Z_train, 'g': g_train}
    data_dev = {'X': X_dev, 'Y': Y_dev, 'Z': Z_dev, 'g': g_dev}
    data_test = {'X': X_test, 'Y': Y_test, 'Z': Z_test, 'g': g_test}

    return {'train': data_train, 'dev': data_dev, 'test': data_test}


############################################################
# MC main files
############################################################

def run_sim(seed, data_spec, Ms, wl_model=sigmoid_wl, iv_model='Ridge', opt_iv=True, n_process=2, nu=0.1, ls=True, cf=True, n_split=2, n_folds=2):
    """
    This function executes one simulation
    :param seed: specify the seed
    :param M: number of boosting iterations
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
    data_raw = gen_data(data_spec, seed)
    data = data_transform(data_raw)
    data_train, data_test = data['train'], data['test']

    ### 2. Get estimates: time all the estimators ###
    # 2.1. NPIV
    start_npiv = time.time()
    beta_npiv = npiv_fit(data_train, basis='POLY', degree=3)
    end_npiv = time.time() - start_npiv
    print('NPIV took ' + str(end_npiv / 60) + ' minutes.')

    # 2.2. DeepIV
    start_deepiv = time.time()
    deepiv_param = deepiv_fit(data_train)
    end_deepiv = time.time() - start_deepiv
    print('DeepIV took ' + str(end_deepiv / 60) + ' minutes.')

    # 2.3. Boosting
    start_boost = time.time()
    Ms_boost = [2000, 5000]  # requires more iterations than post-boostIV
    # boost = boostIV_cv_kfold_v2(data_train, n_split, Ms, wl_model, iv_model, nu, ls, cf=cf, n_folds=n_folds, n_process=n_process)
    boost = boostIV_tuned(data, n_split, Ms_boost, wl_model, iv_model, opt_iv, nu, ls, cf, n_process=n_process)
    end_boost = time.time() - start_boost
    print('Boosting took ' + str(end_boost / 60) + ' minutes.')

    # 2.4. Post-Boosting
    start_pboost = time.time()
    # post_boost = post_boostIV_cv_kfold_v2(data_train, n_split, Ms, wl_model, iv_model, nu, ls, cf, n_folds, n_process)
    post_boost = post_boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv, nu, ls, cf, n_process=n_process)
    end_pboost = time.time() - start_pboost
    print('Post-boosting took ' + str(end_pboost / 60) + ' minutes.')

    ### 3. Out of sample data ###
    # Generate oos data
    n_obs_oos = 1000
    X_oos = np.linspace(start=-3, stop=3, num=n_obs_oos)
    # structural function
    if fun_type == 'log':
        h_oos = np.log(np.abs(16 * X_oos - 8) + 1) * np.sign(X_oos - 0.5)
    elif fun_type == 'sin':
        h_oos = np.sin(X_oos)
    elif fun_type == 'step':
        h_oos = 1 * (X_oos < 0) + 2.5 * (X_oos >= 0)
    elif fun_type == 'abs':
        h_oos = np.abs(X_oos)
    else:
        sys.exit('No structural function matched: pick log, sin, step, or abs.')
    data['test']['X'], data['test']['g'] = X_oos, h_oos

    ### 4. Evaluate out of sample performance ###

    # 4.1. NPIV fit
    h_npiv_oos = get_npiv_fit(X_oos.reshape(len(X_oos), 1), beta_npiv, basis='POLY', degree=3)
    mse_npiv = np.mean((h_npiv_oos - h_oos) ** 2)
    bias_npiv = np.mean(h_npiv_oos - h_oos)

    # 4.2. KIV fit: time it
    start_kiv = time.time()
    h_kiv_oos = get_KIV(data_train, X_oos.reshape(len(X_oos), 1))
    end_kiv = time.time() - start_kiv
    print('KIV took ' + str(end_kiv / 60) + ' minutes.')
    mse_kiv = np.mean((h_kiv_oos - h_oos) ** 2)
    bias_kiv = np.mean(h_kiv_oos - h_oos)

    # 4.3. deep iv fit
    h_deep_iv_oos = deepiv_param.predict(X_oos, np.ones((X_oos.shape[0], 1)))
    mse_deepiv = np.mean((h_deep_iv_oos - h_oos) ** 2)
    bias_deepiv = np.mean(h_deep_iv_oos - h_oos)

    # 4.4. deep GMM fit
    start_deep_gmm = time.time()
    h_deepGMM_oos = deepGMM_fit(data, model='toy')
    end_deep_gmm = time.time() - start_deep_gmm
    print('Deep GMM took ' + str(end_deep_gmm / 60) + ' minutes.')
    mse_deep_gmm = ((h_deepGMM_oos.detach().numpy() - h_oos) ** 2).mean()
    bias_deep_gmm = (h_deepGMM_oos.detach().numpy() - h_oos).mean()

    # 4.5. Boosting fit
    x_scaler = preprocessing.StandardScaler().fit(data_train['X'].reshape(-1, 1))
    Xmat_oos = np.c_[np.ones(n_obs_oos), x_scaler.transform(X_oos.reshape(-1, 1))]
    # h_boost_oos = get_boostIV_fit(Xmat_oos, boost['param_cv'], wl_model, boost['M_cv'], n_split)  # CV version
    h_boost_oos = get_boostIV_fit(Xmat_oos, boost['param_opt'], wl_model, boost['M_opt'], n_split)
    mse_boost = np.mean((h_boost_oos - h_oos) ** 2)
    bias_boost = np.mean(h_boost_oos - h_oos)

    # 4.6. Post boosting fit
    # h_post_boost_oos = get_post_boostIV_fit(post_boost['param_cv'], wl_model, M=post_boost['M_cv'], X=Xmat_oos, n_split=n_split, cf=cf)  # CV version
    h_post_boost_oos = get_post_boostIV_fit(post_boost['param_opt'], wl_model, M=post_boost['M_opt'], X=Xmat_oos, n_split=n_split, cf=cf)
    mse_post_boost = np.mean((h_post_boost_oos - h_oos) ** 2)
    bias_post_boost = np.mean(h_post_boost_oos - h_oos)

    # counter
    print('')
    print('=====================================')
    print('Simulation run finished: ' + str(seed))
    print('=====================================')
    print('')

    # consolidate results
    mse_out = [float(bias_npiv), float(bias_kiv), float(bias_deepiv), float(bias_deep_gmm), float(bias_boost), float(bias_post_boost),
               mse_npiv, mse_kiv, mse_deepiv, mse_deep_gmm, mse_boost, mse_post_boost]
    fit_out = np.vstack([h_npiv_oos, h_kiv_oos, h_deep_iv_oos, h_deepGMM_oos.detach().numpy().flatten(), h_boost_oos, h_post_boost_oos]).T
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
    out_mse = pd.DataFrame(out_mat, columns=['Bias NPIV', 'Bias KIV', 'Bias DeepIV', 'Bias DeepGMM', 'Bias boostIV', 'Bias post-boostIV',
                                             'MSE NPIV', 'MSE KIV', 'MSE DeepIV', 'MSE DeepGMM', 'MSE boostIV', 'MSE post-boostIV'])
    out_fit = pd.DataFrame(fit_mc, columns=['NPIV', 'KIV', 'DeepIV', 'DeepGMM', 'boostIV', 'post-boostIV'])
    return {'mse': out_mse, 'fit': out_fit}


if __name__ == "__main__":
    # specify the parameters
    n_runs = 1
    n_obs = 2000
    rho = 0.5
    Ms = [4 + 3 * j for j in range(33)]
    fun_type = 'abs'
    data_spec = {'n_obs': n_obs, 'rho': rho, 'fun_type': fun_type}
    mc_mse = run_mc(n_runs, data_spec, Ms, opt_iv=False, nu=1)


