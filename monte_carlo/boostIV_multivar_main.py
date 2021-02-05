# ======================================================================================================================

# boostIV MC: Multivariate non-linear design

# ======================================================================================================================

import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
from scipy.stats import multivariate_normal

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

def gen_data(data_spec, seed):
    """
    Design:
        Y = h(X) + e, E[e|Z] = 0
        X_k = g_k(Z) + v, E[v|Z] = 0
        e is correlated with all components of v
        h(X) = exp{-0.5 * X'X}
        g(Z) can take different forms
    Parameters:
        :param n_obs: number of observations
        :param dx: number of endogebous variables
        :param dz: number of instruments
        :param rho: controls the degree of ensdogeneity
        :param iv_fun: defines the reduced form function g: lin, nonlin, and sparse
        :param srt_fun: defines the structural function h: exp or sin
    """
    # set a seed
    np.random.seed(seed)
    # DGP parameters
    n_obs = data_spec['n_obs']
    dx = data_spec['dx']
    dz = data_spec['dz']
    rho = data_spec['rho']
    str_fun = data_spec['str_fun']
    iv_fun = data_spec['iv_fun']
    # draw Z
    Sigma_z = np.zeros((dz, dz))
    for j in range(dz):
        for k in range(dz):
            Sigma_z[j, k] = 0.5 ** np.abs(j - k)
    Z = np.random.multivariate_normal(mean=np.zeros(dz), cov=Sigma_z, size=n_obs)
    # Z = np.random.normal(0, 1, size=(n_obs, dz))
    # draw e ~ N(0,1) and then v|e ~ N(rho * e, I - rho^2) so that the unconditional cov of v is identity
    e = np.random.normal(0, 1, size=n_obs)
    v = np.vstack([np.random.multivariate_normal(mean=rho * np.ones(dx) * eps, cov=np.eye(dx) * (1 - rho ** 2), size=1) for eps in e])
    # reduced form equation
    if iv_fun == 'lin':
        """ follows the exponential design of Belloni et al. (2012) """
        pi = np.array([0.7 ** x for x in range(dz)])
        Pi = np.repeat(pi.reshape(1, len(pi)), dx, axis=0)
        X = Z @ Pi.T + v
    elif iv_fun == 'nonlin':
        """
        g_k(Z) = G(Z, theta_k), where G is a multivariate normal density parameterized by theta_k
        theta_k = (mu_k, sigma_k), where for simplicity sigma_k = np.eye, and only mu_k differs across X's
        we fix mu_k for all Z's, so that mu_k = mu[k] * np.ones(dz)
        """
        mu = np.linspace(start=-2, stop=2, num=dx)
        g_fun = lambda m: multivariate_normal(mean=m, cov=np.ones(dz)).pdf(Z)
        X = np.array([g_fun(mu[k] * np.ones(dz)) for k in range(dx)]).T + v
    elif iv_fun == 'sparse':
        """
        g_1(Z) = Z_1
        g_2(Z) = Z_1 + Z_2
        .
        .
        .
        g_K(Z) = Z_1 + ... + Z_K
        """
        Pi = np.zeros((dx, dz))
        ltr_id = np.tril_indices(dx)
        Pi[ltr_id] = 1
        X = Z @ Pi.T + v
    else:
        sys.exit('No iv_fun matched: pick lin, nonlin, or sparse.')
    # structural equation
    if str_fun == 'exp':
        h_fun = np.array([np.exp(-X[i, :] @ X[i, :].T / 2) for i in range(n_obs)])
    elif str_fun == 'sin':
        h_fun = np.sum(np.array([np.sin(10 * X[:, j]) for j in range(dx)]), axis=0)
    else:
        sys.exit('No str_fun matched: pick exp or sin.')
    # add noise
    Y = h_fun + e

    return {'X': X, 'Y': Y, 'Z': Z, 'h': h_fun}


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
    X_train, Z_train, Y_train, g_train = X[train_id, :], Z[train_id, :], Y[train_id], g[train_id]
    X_dev, Z_dev, Y_dev, g_dev = X[dev_id, :], Z[dev_id, :], Y[dev_id], g[dev_id]
    X_test, Z_test, Y_test, g_test = X[test_id, :], Z[test_id, :], Y[test_id], g[test_id]

    # store as dictionaries
    data_train = {'X': X_train, 'Y': Y_train, 'Z': Z_train, 'g': g_train}
    data_dev = {'X': X_dev, 'Y': Y_dev, 'Z': Z_dev, 'g': g_dev}
    data_test = {'X': X_test, 'Y': Y_test, 'Z': Z_test, 'g': g_test}

    return {'train': data_train, 'dev': data_dev, 'test': data_test}

############################################################
# MC main files
############################################################

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
    data_raw = gen_data(data_spec, seed)
    data = data_transform(data_raw)
    data_train, data_test = data['train'], data['test']

    ### 2. Get estimates: time all the estimators ###
    # 2.1. NPIV
    start_npiv = time.time()
    beta_npiv = npiv_fit(data_train, basis='SPL', degree=4)
    end_npiv = time.time() - start_npiv
    print('NPIV took ' + str(end_npiv / 60) + ' minutes.')

    # 2.2. Boosting
    start_boost = time.time()
    # boost = boostIV_cv_kfold_v2(data_train, n_split, Ms, wl_model, iv_model, nu, ls, cf=cf, n_folds=n_folds, n_process=n_process)
    boost = boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv, nu, ls, cf, early=True, n_process=n_process)
    end_boost = time.time() - start_boost
    print('Boosting took ' + str(end_boost / 60) + ' minutes.')

    # 2.3. DeepIV
    start_deepiv = time.time()
    deepiv_param = deepiv_fit(data_train)
    end_deepiv = time.time() - start_deepiv
    print('DeepIV took ' + str(end_deepiv / 60) + ' minutes.')

    # 2.4. Post-Boosting
    start_pboost = time.time()
    # post_boost = post_boostIV_cv_kfold_v2(data_train, n_split, Ms, wl_model, iv_model, nu, ls, cf, n_folds, n_process)
    post_boost = post_boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv, nu, ls, cf=cf, early=True, n_process=n_process)
    end_pboost = time.time() - start_pboost
    print('Post-boosting took ' + str(end_pboost / 60) + ' minutes.')

    ### 3. Out of sample data ###
    X_oos = data_test['X']
    h_oos = data_test['g']
    n_obs_oos = len(h_oos)

    ### 4. Evaluate out of sample performance ###

    # 4.1. NPIV fit
    h_npiv_oos = get_npiv_fit(X_oos, beta_npiv, basis='SPL', degree=4)
    mse_npiv = np.mean((h_npiv_oos - h_oos) ** 2)
    bias_npiv = np.mean(h_npiv_oos - h_oos)

    # 4.2. KIV fit: time it
    start_kiv = time.time()
    h_kiv_oos = get_KIV(data_train, X_oos)
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
    h_deepGMM_oos = deepGMM_fit(data, model='simple')
    end_deep_gmm = time.time() - start_deep_gmm
    print('Deep GMM took ' + str(end_deep_gmm / 60) + ' minutes.')
    mse_deep_gmm = ((h_deepGMM_oos.detach().numpy() - h_oos) ** 2).mean()
    bias_deep_gmm = (h_deepGMM_oos.detach().numpy() - h_oos).mean()

    # 4.5. Boosting fit
    x_scaler = preprocessing.StandardScaler().fit(data_train['X'])
    Xmat_oos = np.c_[np.ones(n_obs_oos), x_scaler.transform(X_oos)]
    # h_boost_oos = get_boostIV_fit(Xmat_oos, boost['param_cv'], wl_model, boost['M_cv'], n_split)  # CV version
    h_boost_oos = get_boostIV_fit(Xmat_oos, boost['param_opt'], wl_model, boost['M_opt'], n_split)
    mse_boost = np.mean((h_boost_oos - h_oos) ** 2)
    bias_boost = np.mean(h_boost_oos - h_oos)

    # 4.6. Post boosting fit
    # h_post_boost_oos = get_post_boostIV_fit_v2(post_boost['param_cv'], wl_model, M=post_boost['M_cv'], X=Xmat_oos, n_split=n_split, cf=cf)  # CV version
    h_post_boost_oos = get_post_boostIV_fit(post_boost['param_opt'], wl_model, M=post_boost['M_opt'], X=Xmat_oos, n_split=n_split, cf=cf)
    mse_post_boost = np.mean((h_post_boost_oos - h_oos) ** 2)
    bias_post_boost = np.mean(h_post_boost_oos - h_oos)

    # check
    print('')
    print('=====================================')
    print('Simulation run finished: ' + str(seed))
    print('=====================================')
    print('')

    # consolidate results
    mse_out = [float(bias_npiv), float(bias_kiv), float(bias_deepiv), float(bias_deep_gmm), float(bias_boost), float(bias_post_boost),
               mse_npiv, mse_kiv, mse_deepiv, mse_deep_gmm, mse_boost, mse_post_boost]
    return mse_out


############################################################
# Run MC
############################################################

def run_mc_multivar(n_runs, data_spec, Ms, wl_model=sigmoid_wl, iv_model='Ridge', opt_iv=True, n_process=2, nu=0.1, ls=True, cf=True, n_split=2, n_folds=2):
    """
    This function funs the whole MC
    :param n_runs: number of simulations
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
    :return: bias and mse stats across simulations
    """
    # run MC loop
    out_mat = np.zeros((n_runs, 12))
    for n in range(n_runs):
        out_mat[n, :] = run_sim(n, data_spec, Ms, wl_model, iv_model, opt_iv, n_process, nu, ls, cf, n_split, n_folds)
    # convert into a pandas df
    out_df = pd.DataFrame(out_mat, columns=['Bias NPIV', 'Bias KIV', 'Bias DeepIV', 'Bias DeepGMM', 'Bias boostIV', 'Bias post-boostIV',
                                            'MSE NPIV', 'MSE KIV', 'MSE DeepIV', 'MSE DeepGMM', 'MSE boostIV', 'MSE post-boostIV'])
    return out_df

if __name__ == "__main__":
    # specify parameters
    n_runs = 1
    n_obs = 2000  # only half goes for training, a quarter for testing
    dx = 5
    dz = 7
    rho = 0.75  # no decimals in terminal
    Ms = [10 + 10 * j for j in range(10)]
    iv_fun = 'nonlin'
    str_fun = 'exp'
    data_spec = {'n_obs': n_obs, 'dx': dx, 'dz': dz, 'rho': rho, 'iv_fun': iv_fun, 'str_fun': str_fun}
    mc_mse = run_mc_multivar(n_runs, data_spec, Ms, opt_iv=True)

