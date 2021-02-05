# ======================================================================================================================

# boostIV functions

# ======================================================================================================================

import numpy as np
import sys
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import preprocessing
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from joblib import Parallel, delayed
from boostIV.primitives import sigmoid_wl, cubic_spline, fsam_ls_iv, sigmoid_wl_jac, cubic_spline_jac
from boostIV.fit import get_boostIV_fit, get_post_boostIV_fit, get_bf_hat


###################################################

### boostIV: not tuned ###

###################################################

def boostIV_no_crossfit(data, M, wl_model, iv_model, nu, ls, opt_iv=True):
    """
    :param data: a dictionary, which is a tuple (X, Y, Z); ! since it's a data slice, it must be pre-scaled !
    :param M: number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param nu: shrinkage parameter
    :param ls: line search step
    :param opt_iv: if True, use optimal Chamberlain IVs
    :return: estimated boosting parameters
    """

    # IV estimators
    if iv_model == 'Ridge':
        iv_est = linear_model.RidgeCV(cv=2, fit_intercept=True)
    elif iv_model == 'Elnet':
        iv_est = linear_model.ElasticNetCV(cv=2, fit_intercept=True)
    elif iv_model == 'Lasso':
        iv_est = linear_model.LassoCV(cv=2, fit_intercept=True)
    elif iv_model == 'RandomForest':
        iv_est = RandomForestRegressor(n_estimators=100, random_state=42)
    elif iv_model == 'NN':
        iv_est = MLPRegressor(hidden_layer_sizes=(5, 3), solver='adam', learning_rate='invscaling', random_state=1, max_iter=400)
    else:
        sys.exit('No IV model found. Pick one of the following: Ridge, Elnet, Lasso, Random Forest, or NN')

    # upload variables from the data
    X, Y, Z = data['X'], data['Y'], data['Z']

    # number of parameters for a base learner
    if wl_model == sigmoid_wl:
        n_param = 1 + X.shape[1]
    elif wl_model == cubic_spline:
        n_param = ((3 + 2) * (X.shape[1] - 1) + 1) + 1
    else:
        sys.exit('No weak learner type matched: pick sigmoid_wl or cubic_spline.')

    # get jacobians
    if wl_model == sigmoid_wl:
        wl_model_jac = sigmoid_wl_jac
    elif wl_model == cubic_spline:
        wl_model_jac = cubic_spline_jac
    else:
        sys.exit('No weak learner type matched: pick sigmoid_wl or cubic_spline.')

    # number of observations
    n_obs = X.shape[0]

    # initialize base learners
    param_mat = np.zeros((n_param, M))
    # matrix of fitted weak learners: the first column of mean(Y) is the initialization
    b_mat = np.zeros((n_obs, M + 1))
    b_mat[:, 0] = np.ones(n_obs) * np.mean(Y)

    # start the algo
    m = 1
    while m <= M:
        """ 1st stage: learn IVs """
        if opt_iv:
            if m == 1:
                # initialize IVs
                Phi = b_mat[:, m - 1]
                iv_est.fit(Z, Phi)
                iv_opt = iv_est.predict(Z)
            else:
                # optimal IVs: based on previous step parameter estimates
                Phi = wl_model(X, param_mat[1:, m - 2])
                Phi_jac = wl_model_jac(X, param_mat[1:, m - 2])
                # IVs for beta: weight on the basis function
                iv_est.fit(Z, -Phi)  # minus sign
                iv_1 = iv_est.predict(Z)
                # IVs for gamma: wl parameters
                xi_hat = -param_mat[0, m - 2] * Phi_jac  # minus sign
                iv_fit = [iv_est.fit(Z, xi_hat[:, j]) for j in range(xi_hat.shape[1])]
                iv_2 = np.vstack([iv_fit[j].predict(Z) for j in range(xi_hat.shape[1])]).T
                # drop repeating columns if any
                iv_2 = np.unique(iv_2, axis=1)
                # concatenate and weigh by the variance from the m-1 step
                eps_hat = np.mean((Y - b_mat[:, m - 1]) ** 2)
                iv_opt = np.c_[iv_1, iv_2] / eps_hat
        else:
            if m == 1:
                Phi = b_mat[:, m - 1]
            else:
                Phi = wl_model(X, param_mat[1:, m - 2])
            iv_est.fit(Z, Phi)
            iv_opt = iv_est.predict(Z)

        """ 2nd stage: Solve the minimization problem """
        resid = Y - b_mat[:, m - 1]
        fit = least_squares(fsam_ls_iv, x0=np.ones(n_param), args=(resid, X, iv_opt, wl_model))

        """ Line search: fit the optimal weight using OLS """
        if ls:
            Phi_hat = wl_model(X, fit.x[1:]).reshape(-1, 1)
            gamma_hat = np.linalg.inv(Phi_hat.T @ Phi_hat) @ Phi_hat.T @ np.array(resid)

        """ Update the matrices """
        if ls:
            # multiply the weight by the shrinkage parameter and store
            param_mat[:, m - 1] = np.append(nu * gamma_hat, fit.x[1:])
            b_mat[:, m] = b_mat[:, m - 1] + nu * gamma_hat * wl_model(X, fit.x[1:])
        else:
            param_mat[:, m - 1] = fit.x
            b_mat[:, m] = b_mat[:, m - 1] + nu * fit.x[0] * wl_model(X, fit.x[1:])

        """ Update the counter """
        print('Boosting iteration finished: ' + str(m))
        m += 1

    return np.c_[np.ones(n_param) * Y.mean(), param_mat]


def boostIV_crossfit(data, M, wl_model, iv_model, nu, ls, opt_iv=True, n_split=2):
    """
    :param data: a dictionary, which is a tuple (X, Y, Z); ! since it's a data slice, it must be pre-scaled !
    :param n_split: number of folds for cross fitting
    :param M: number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso, NN
    :param nu: shrinkage parameter
    :param ls: line search step
    :param opt_iv: if True, use optimal Chamberlain IVs
    :return: estimated boosting parameters
    """

    # IV estimators
    if iv_model == 'Ridge':
        iv_est = linear_model.RidgeCV(cv=2, fit_intercept=True)
    elif iv_model == 'Elnet':
        iv_est = linear_model.ElasticNetCV(cv=2, fit_intercept=True)
    elif iv_model == 'Lasso':
        iv_est = linear_model.LassoCV(cv=2, fit_intercept=True)
    elif iv_model == 'RandomForest':
        iv_est = RandomForestRegressor(n_estimators=100, random_state=42)
    elif iv_model == 'NN':
        iv_est = MLPRegressor(hidden_layer_sizes=(5, 3), solver='adam', learning_rate='invscaling', random_state=1, max_iter=400)
    else:
        sys.exit('No IV model found. Pick one of the following: Ridge, Elnet, Lasso, Random Forest, or NN')

    # upload variables from the data
    X, Y, Z = data['X'], data['Y'], data['Z']

    # number of parameters for a base learner
    if wl_model == sigmoid_wl:
        n_param = 1 + X.shape[1]
    elif wl_model == cubic_spline:
        n_param = ((3 + 2) * (X.shape[1] - 1) + 1) + 1
    else:
        sys.exit('No weak learner type matched: pick sigmoid_wl or cubic_spline.')

    # get jacobians
    if wl_model == sigmoid_wl:
        wl_model_jac = sigmoid_wl_jac
    elif wl_model == cubic_spline:
        wl_model_jac = cubic_spline_jac
    else:
        sys.exit('No weak learner type matched: pick sigmoid_wl or cubic_spline.')

    # number of observations
    n_obs = X.shape[0]

    # Partition the data into n_split folds
    # set up indices for partitions
    kf = KFold(n_splits=n_split)
    Y_train, Y_test = [], []
    X_train, X_test = [], []
    Z_train, Z_test = [], []

    # splits are made in a consecutive fashion
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        Y_train.append(Y[train_index])
        Y_test.append(Y[test_index])
        X_train.append(X[train_index, :])
        X_test.append(X[test_index, :])
        Z_train.append(Z[train_index, :])
        Z_test.append(Z[test_index, :])

    # initialize base learners
    param_mat = np.zeros((n_param * n_split, M))
    # matrix of fitted weak learners: the first column of mean(Y) is the initialization
    b_mat = np.zeros((n_obs, M + 1))
    b_mat[:, 0] = np.ones(n_obs) * np.mean(Y)  # actually not used in the code

    # Outer Loop
    m = 1
    while m <= M:

        # partition the basis functions (make updates every iteration)
        b_mat_train, b_mat_test = [], []
        for train_index, test_index in kf.split(X):
            b_mat_train.append(b_mat[train_index, :])
            b_mat_test.append(b_mat[test_index, :])

        b_mat_update = []
        # Inner Loop
        for k in range(n_split):
            """ Prepare data partitions """
            Y_train_k, Y_test_k = Y_train[k], Y_test[k]
            X_train_k, X_test_k = X_train[k], X_test[k]
            Z_train_k, Z_test_k = Z_train[k], Z_test[k]
            b_mat_train_k, b_mat_test_k = b_mat_train[k][:, m - 1], b_mat_test[k][:, m - 1]

            """ 1st stage: learn IVs """
            if opt_iv:
                # initialize IVs
                if m == 1:
                    # Phi_train_k = b_mat_train_k  # this one assigns mean for the whole data
                    Phi_train_k = Y_train_k * np.ones(len(Y_train_k))  # mean for each fold
                    iv_est.fit(Z_train_k, Phi_train_k)
                    # apply the learnt transformation to the k-th fold (test)
                    iv_opt_k = iv_est.predict(Z_test_k)
                else:
                    # optimal IVs: based on previous step parameter estimates
                    Phi_train_k = wl_model(X_train_k, param_mat[1 + k * n_param: (k + 1) * n_param, m - 2])
                    Phi_jac_train_k = wl_model_jac(X_train_k, param_mat[1 + k * n_param: (k + 1) * n_param, m - 2])
                    # IVs for beta: weight on the basis function
                    iv_est.fit(Z_train_k, -Phi_train_k)  # minus sign
                    iv_1 = iv_est.predict(Z_test_k)
                    # IVs for gamma: wl parameters
                    xi_hat = -param_mat[k * n_param, m - 2] * Phi_jac_train_k  # minus sign
                    iv_fit = [iv_est.fit(Z_train_k, xi_hat[:, j]) for j in range(xi_hat.shape[1])]
                    iv_2 = np.vstack([iv_fit[j].predict(Z_test_k) for j in range(xi_hat.shape[1])]).T
                    # # drop repeating columns if any
                    iv_2 = np.unique(iv_2, axis=1)
                    # concatenate and weigh by the variance from the m-1 step
                    eps_hat = np.mean((Y_train_k - b_mat_train_k) ** 2)
                    iv_opt_k = np.c_[iv_1, iv_2] / eps_hat
            else:
                if m == 1:
                    # Phi_train_k = b_mat_train_k  # this one assigns mean for the whole data
                    Phi_train_k = Y_train_k * np.ones(len(Y_train_k))  # mean for each fold
                else:
                    Phi_train_k = wl_model(X_train_k, param_mat[1 + k * n_param: (k + 1) * n_param, m - 2])
                iv_est.fit(Z_train_k, Phi_train_k)
                # apply the learnt transformation to the k-th fold (test)
                iv_opt_k = iv_est.predict(Z_test_k)

            """ 2nd stage: Solve the minimization problem """
            # learn basis functions using boosting for the k-th fold (test)
            resid = Y_test_k - b_mat_test_k
            fit = least_squares(fsam_ls_iv, x0=0.1 * np.ones(n_param), args=(resid, X_test_k, iv_opt_k, wl_model))

            """ Line search: fit the optimal weight using OLS """
            if ls:
                Phi_hat_k = wl_model(X_test_k, fit.x[1:]).reshape(-1, 1)
                gamma_hat_k = np.linalg.inv(Phi_hat_k.T @ Phi_hat_k) @ Phi_hat_k.T @ np.array(resid)

            """ Update the matrices """
            if ls:
                # multiply the weight by the shrinkage parameter and store
                param_mat[k * n_param: (k + 1) * n_param, m - 1] = np.append(nu * gamma_hat_k, fit.x[1:])
                b_mat_update_k = b_mat_test_k + nu * gamma_hat_k * wl_model(X_test_k, fit.x[1:])
            else:
                param_mat[k * n_param: (k + 1) * n_param, m - 1] = fit.x
                b_mat_update_k = b_mat_test_k + nu * fit.x[0] * wl_model(X_test_k, fit.x[1:])
            b_mat_update = np.r_[b_mat_update, b_mat_update_k]

        # due to consecutive splits, the ordering is preserved
        b_mat[:, m] = b_mat_update

        """ Update the counter """
        print('Boosting iteration finished: ' + str(m))
        m += 1

    return np.c_[np.ones(n_param * n_split) * Y.mean(), param_mat]


def post_boostIV_crossfit(data, M, wl_model, iv_model, cf, nu, ls, opt_iv=True, n_split=2, scale=True, post_type='LS'):
    """
    Performs post-boosting with crossfiting
    :param data: a dictionary, which is a tuple (X, Y, Z)
    :param n_split: number of data splits
    :param M: number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :param nu: shrinkage parameter
    :param ls: line search step
    :param opt_iv: if True, use optimal Chamberlain IVs
    :param scale: standardize the data if true
    :param post_type: model to use in the post-processing step
    :return: post-boosting parameters
    """

    # upload variables from the data
    X, Y, Z = data['X'], np.array(data['Y']), data['Z']
    # scale the variables
    if scale == True:
        X_scaled = preprocessing.scale(X)
        Z_scaled = preprocessing.scale(Z)
        Xmat = np.c_[np.ones(len(X)), X_scaled]  # add an intercept
        Zmat = np.c_[np.ones(len(Z)), Z_scaled]  # add an intercept
    else:
        # if scale = False, pass data with an intercept
        Xmat = X
        Zmat = Z

    # split the data for cross fitting
    # set up indices for partitions
    kf = KFold(n_splits=n_split)
    Y_train, Y_test = [], []
    X_train, X_test = [], []
    Z_train, Z_test = [], []

    # splits are made in a consecutive fashion
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        Y_train.append(Y[train_index])
        Y_test.append(Y[test_index])
        X_train.append(Xmat[train_index, :])
        X_test.append(Xmat[test_index, :])
        Z_train.append(Zmat[train_index, :])
        Z_test.append(Zmat[test_index, :])

    param_mat = []
    post_boost_fit = []
    # do cross-fitting
    for k in range(n_split):

        print('#================================#')
        print('post-boostIV: outer CF fold {}'.format(k + 1))
        print('#================================#')

        """ Prepare data partitions """
        Y_train_k, Y_test_k = Y_train[k], Y_test[k]
        X_train_k, X_test_k = X_train[k], X_test[k]
        Z_train_k, Z_test_k = Z_train[k], Z_test[k]

        """ Boosting on train data """
        data_train = {'X': X_train_k, 'Z': Z_train_k, 'Y': Y_test_k}
        if cf:
            param_mat_k = boostIV_crossfit(data_train, M, wl_model, iv_model, nu, ls, opt_iv, n_split=2)
        else:
            param_mat_k = boostIV_no_crossfit(data_train, M, wl_model, iv_model, nu, ls, opt_iv)
        param_mat.append(param_mat_k)

        """ Post-boosting on test data """
        if cf:
            bfs_fit_k = get_bf_hat(X_test_k, param_mat_k, wl_model, M, n_split=2)
        else:
            bfs_fit_k = np.vstack([wl_model(X_test_k, param_mat_k[1:, k + 1]) for k in range(M)]).T

        if post_type == 'LS':
            # linear regression
            lin_fit_k = linear_model.LinearRegression()
            lin_fit_k.fit(bfs_fit_k, Y_test_k)
            post_boost_fit.append(lin_fit_k)
        elif post_type == 'NN':
            # NN
            nn_fit_k = MLPRegressor(hidden_layer_sizes=(32, 16,), solver='adam', random_state=1)
            nn_fit_k.fit(bfs_fit_k, Y_test_k)
            post_boost_fit.append(nn_fit_k)
        elif post_type == 'Ridge':
            # Ridge
            ridge_fit_k = RidgeCV(cv=3, fit_intercept=True)
            ridge_fit_k.fit(bfs_fit_k, Y_test_k)
            post_boost_fit.append(ridge_fit_k)
        elif post_type == 'Lasso':
            # Lasso
            lasso_fit_k = LassoCV(fit_intercept=True, cv=3)
            lasso_fit_k.fit(bfs_fit_k, Y_test_k)
            post_boost_fit.append(lasso_fit_k)
        else:
            sys.exit('No post_type matched: pick LS, NN, Ridge or Lasso')

    return {'bf_param': param_mat, 'bf_weights': post_boost_fit}


###################################################

### Cross-Validation ###

###################################################

# post-boostIV CV
def post_boostIV_cv_kfold(data, n_split, Ms, wl_model, iv_model, opt_iv=True, nu=0.1, ls=True, cf=True, n_folds=2, n_process=1, early=True):
    """
    This function performs a 2-fold CV for boosting IV
    :param data: a dictionary, which is a tuple (X, Y, Z)
    :param Ms: grid of number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param opt_iv: if True, use optimal Chamberlain IVs
    :param early:  if True, apply early stopping, search along the whole space otherwise
    :param n_process: if > 1, then performs CV in parallel
    :param n_split: number of folds for cross fitting
    :param nu: shrinkage parameter
    :param ls: line search step
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :param n_folds: number of folds for CV
    :return: estimated boosting parameters, optimal number of iterations, MSE path for the number of iterations
    """
    # 1. split the data into the training and validation sets
    X, Y, Z = data['X'], data['Y'], data['Z']

    # 2. scale data
    X_scaled = preprocessing.scale(X)
    Z_scaled = preprocessing.scale(Z)
    Xmat = np.c_[np.ones(len(X)), X_scaled]  # add an intercept
    Zmat = np.c_[np.ones(len(Z)), Z_scaled]

    # set up indices for partitions
    kf = KFold(n_splits=n_folds)
    Y_train, Y_test = [], []
    X_train, X_test = [], []
    Z_train, Z_test = [], []

    # splits are made in a consecutive fashion
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        Y_train.append(Y[train_index])
        Y_test.append(Y[test_index])
        X_train.append(Xmat[train_index])
        X_test.append(Xmat[test_index])
        Z_train.append(Zmat[train_index])
        Z_test.append(Zmat[test_index])

    # auxiliary function evaluates test MSE for one fold
    # for parallelization across folds in the early stop specification
    def fold_fit(data_train, data_test, M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf):
        # fit using the train data
        param_mats = post_boostIV_crossfit(data_train, M, wl_model, iv_model, nu, ls, opt_iv, cf, n_split, scale=False)
        # calculate the test MSE
        h_boost_oos = get_post_boostIV_fit(param_mats, wl_model, M, data_test['X'], n_split, cf)
        mse_boost = np.mean((h_boost_oos - data_test['Y']) ** 2)
        return mse_boost

    if early:
        # early stopping: as soon as MSE starts going up, stop
        i = 1
        tol = 1e-4  # for numerical errors
        mse_boost_cv = []
        while i <= len(Ms):
            # run in parallel across folds
            mse_boost_i = Parallel(n_jobs=n_process, verbose=20)(delayed(fold_fit)
                                                                 ({'X': X_train[k], 'Y': Y_train[k], 'Z': Z_train[k]},
                                                                  {'X': X_test[k], 'Y': Y_test[k], 'Z': Z_test[k]},
                                                                  Ms[i - 1], wl_model, iv_model, opt_iv, n_split, nu, ls, cf) for k in range(n_folds))
            mse_boost_cv.append(np.mean(mse_boost_i))

            # early stopping rule
            if i > 1 and mse_boost_cv[i - 2] < (mse_boost_cv[i - 1] - tol):
                break
            i += 1
        M_optimal = int(Ms[i - 2])
    else:
        # initialize the matrix of MSEs
        mse_boost = []
        for k in range(n_folds):
            # run in parallel across Ms
            mse_boost_k = Parallel(n_jobs=n_process, verbose=20)(delayed(fold_fit)
                                                                 ({'X': X_train[k], 'Y': Y_train[k], 'Z': Z_train[k]},
                                                                  {'X': X_test[k], 'Y': Y_test[k], 'Z': Z_test[k]},
                                                                  M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf) for M in Ms)
            mse_boost.append(mse_boost_k)
        #  pick the optimal number of iterations
        mse_boost_cv = np.mean(mse_boost, axis=0)
        M_optimal = Ms[np.argmin(mse_boost_cv)]

    # optimal parameteres
    param_cv = post_boostIV_crossfit(data, M_optimal, wl_model, iv_model, nu, ls, opt_iv, cf, n_split, scale=True)

    return {'param_cv': param_cv, 'M_cv': M_optimal, 'MSE_path': mse_boost_cv}


# boostIV CV
def boostIV_cv_kfold(data, n_split, Ms, wl_model, iv_model, opt_iv=True, nu=0.1, ls=True, cf=True, n_folds=2, n_process=1, early=True):
    """
    This function performs a 2-fold CV for boosting IV
    :param data: a dictionary, which is a tuple (X, Y, Z)
    :param Ms: grid of number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param opt_iv: if True, use optimal Chamberlain IVs
    :param early: if True, apply early stopping, search along the whole space otherwise
    :param n_process: if > 1, then performs CV in parallel
    :param n_split: number of folds for cross fitting
    :param nu: shrinkage parameter
    :param ls: line search step
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :param n_folds: number of folds for CV
    :return: estimated boosting parameters, optimal number of iterations, MSE path for the number of iterations
    """
    # 1. split the data into the training and validation sets
    X, Y, Z = data['X'], data['Y'], data['Z']

    # 2. scale data
    X_scaled = preprocessing.scale(X)
    Z_scaled = preprocessing.scale(Z)
    Xmat = np.c_[np.ones(len(X)), X_scaled]  # add an intercept
    Zmat = np.c_[np.ones(len(Z)), Z_scaled]

    # set up indices for partitions
    kf = KFold(n_splits=n_folds)
    Y_train, Y_test = [], []
    X_train, X_test = [], []
    Z_train, Z_test = [], []

    # splits are made in a consecutive fashion
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        Y_train.append(Y[train_index])
        Y_test.append(Y[test_index])
        X_train.append(Xmat[train_index])
        X_test.append(Xmat[test_index])
        Z_train.append(Zmat[train_index])
        Z_test.append(Zmat[test_index])

    # auxiliary function evaluates test MSE for one fold
    # for parallelization across folds in the early step specification
    def fold_fit(data_train, data_test, M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf):
        # fit using the train data
        if cf:
            param_mats = boostIV_crossfit(data_train, M, wl_model, iv_model, nu, ls, opt_iv, n_split=n_split)
        else:
            param_mats = boostIV_no_crossfit(data_train, M, wl_model, iv_model, nu, ls, opt_iv)
        # calculate the test MSE
        h_boost_oos = get_boostIV_fit(data_test['X'], param_mats, wl_model, M, n_split)
        mse_boost = np.mean((h_boost_oos - data_test['Y']) ** 2)
        return mse_boost

    if early:
        # early stopping: as soon as MSE starts going up, stop
        i = 1
        tol = 1e-4  # for numerical errors
        mse_boost_cv = []
        while i <= len(Ms):
            # run in parallel across folds
            mse_boost_i = Parallel(n_jobs=n_process, verbose=20)(delayed(fold_fit)
                                                                 ({'X': X_train[k], 'Y': Y_train[k], 'Z': Z_train[k]},
                                                                  {'X': X_test[k], 'Y': Y_test[k], 'Z': Z_test[k]},
                                                                  Ms[i - 1], wl_model, iv_model, opt_iv, n_split, nu, ls, cf) for k in range(n_folds))
            mse_boost_cv.append(np.mean(mse_boost_i))

            # early stopping rule
            if i > 1 and mse_boost_cv[i - 2] < (mse_boost_cv[i - 1] - tol):
                break
            i += 1
        M_optimal = int(Ms[i - 2])
    else:
        # initialize the matrix of MSEs
        mse_boost = []
        for k in range(n_folds):
            # run in parallel across Ms
            mse_boost_k = Parallel(n_jobs=n_process, verbose=20)(delayed(fold_fit)
                                                                 ({'X': X_train[k], 'Y': Y_train[k], 'Z': Z_train[k]},
                                                                  {'X': X_test[k], 'Y': Y_test[k], 'Z': Z_test[k]},
                                                                  M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf) for M in Ms)
            mse_boost.append(mse_boost_k)
        # pick the optimal number of iterations
        mse_boost_cv = np.mean(mse_boost, axis=0)
        M_optimal = Ms[np.argmin(mse_boost_cv)]

    # optimal parameters
    data_scaled = {'X': Xmat, 'Y': Y, 'Z': Zmat}
    if cf:
        param_cv = boostIV_crossfit(data_scaled, M_optimal, wl_model, iv_model, nu, ls, opt_iv, n_split=n_split)
    else:
        param_cv = boostIV_no_crossfit(data_scaled, M_optimal, wl_model, iv_model, nu, ls, opt_iv)

    return {'param_cv': param_cv, 'M_cv': M_optimal, 'MSE_path': mse_boost_cv}


###################################################

### Tuning using the validation set ###

###################################################

# boostIV tuning
def boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv=True, nu=0.1, ls=True, cf=True, early=True, n_process=1):
    """
    This function performs tuning of the hyperparameters on the validation set
    :param data: includes both training and validation sets
    :param Ms: grid of number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param opt_iv: if True, use optimal Chamberlain IVs
    :param early:  if True, apply early stopping, search along the whole space otherwise
    :param n_split: number of folds for cross fitting
    :param nu: shrinkage parameter
    :param ls: line search step
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :param n_process: if > 1, runs in parallel under no early stopping
    :return: estimated boosting parameters, optimal number of iterations, MSE path for the number of iterations
    """
    # 1. split the data into the training and validation sets
    data_train, data_dev = data['train'], data['dev']
    X_train, Y_train, Z_train = data_train['X'], data_train['Y'], data_train['Z']
    X_dev, Y_dev, Z_dev = data_dev['X'], data_dev['Y'], data_dev['Z']

    # 2. scale data according to the train set
    # check inputs
    if len(X_train.shape) == 1:
        x_scaler = preprocessing.StandardScaler().fit(X_train.reshape(-1, 1))
        X_train_scaled, X_dev_scaled = x_scaler.transform(X_train.reshape(-1, 1)), x_scaler.transform(X_dev.reshape(-1, 1))
    else:
        x_scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled, X_dev_scaled = x_scaler.transform(X_train), x_scaler.transform(X_dev)

    if len(Z_train.shape) == 1:
        z_scaler = preprocessing.StandardScaler().fit(Z_train.reshape(-1, 1))
        Z_train_scaled, Z_dev_scaled = z_scaler.transform(Z_train.reshape(-1, 1)), z_scaler.transform(Z_dev.reshape(-1, 1))
    else:
        z_scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_train_scaled, Z_dev_scaled = z_scaler.transform(Z_train), z_scaler.transform(Z_dev)

    Xmat_train, Xmat_dev = np.c_[np.ones(len(X_train)), X_train_scaled], np.c_[np.ones(len(X_dev)), X_dev_scaled]
    Zmat_train, Zmat_dev = np.c_[np.ones(len(Z_train)), Z_train_scaled], np.c_[np.ones(len(Z_dev)), Z_dev_scaled]
    data_train_scaled = {'X': Xmat_train, 'Y': Y_train, 'Z': Zmat_train}
    data_dev_scaled = {'X': Xmat_dev, 'Y': Y_dev, 'Z': Zmat_dev}

    # 3. use the validation set to tune the number of iterations

    # auxiliary function evaluates test MSE for m iterations
    def validation_fit(data_dev, M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf):
        # fit using the train data
        if cf:
            param_mats = boostIV_crossfit(data_dev, M, wl_model, iv_model, nu, ls, opt_iv, n_split=n_split)
        else:
            param_mats = boostIV_no_crossfit(data_dev, M, wl_model, iv_model, nu, ls, opt_iv)
        # calculate the test MSE
        h_boost_oos = get_boostIV_fit(data_dev['X'], param_mats, wl_model, M, n_split)
        mse_boost = np.mean((h_boost_oos - data_dev['Y']) ** 2)
        return mse_boost

    if early:
        # early stopping: as soon as MSE starts going up, stop
        i = 1
        tol = 1e-4  # for numerical errors
        mse_boost = []
        while i <= len(Ms):
            # run in parallel across folds
            mse_boost_i = validation_fit(data_dev_scaled, Ms[i - 1], wl_model, iv_model, opt_iv, n_split, nu, ls, cf)
            mse_boost.append(mse_boost_i)

            # early stopping rule
            if i > 1 and mse_boost[i - 2] < (mse_boost[i - 1] - tol):
                break
            i += 1
        M_optimal = int(Ms[i - 2])
    else:
        # run in parallel across Ms
        mse_boost = Parallel(n_jobs=n_process, verbose=20)(delayed(validation_fit)
                                                           (data_dev_scaled, M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf) for M in Ms)
        #  pick the optimal number of iterations
        M_optimal = Ms[np.argmin(mse_boost)]

    # optimal parameters
    if cf:
        param_opt = boostIV_crossfit(data_train_scaled, M_optimal, wl_model, iv_model, nu, ls, opt_iv, n_split=n_split)
    else:
        param_opt = boostIV_no_crossfit(data_train_scaled, M_optimal, wl_model, iv_model, nu, ls, opt_iv)

    return {'param_opt': param_opt, 'M_opt': M_optimal, 'MSE_path': mse_boost}


# post-boostIV tuning
def post_boostIV_tuned(data, n_split, Ms, wl_model, iv_model, opt_iv=True, nu=0.1, ls=True, cf=True, early=True, n_process=1):
    """
    This function performs tuning of the hyperparameters on the validation set
    :param data: includes both training and validation sets
    :param Ms: grid of number of boosting iterations
    :param wl_model: type of weak learner as a function: sigmoid_wl or cubic_spline
    :param iv_model: first stage fit: Ridge, Elnet, RandomForest, Lasso
    :param opt_iv: if True, use optimal Chamberlain IVs
    :param early:  if True, apply early stopping, search along the whole space otherwise
    :param n_split: number of folds for cross fitting
    :param nu: shrinkage parameter
    :param ls: line search step
    :param cf: if True, add a cross-fitting step for the boosting step (I and II stages)
    :param n_process: if > 1, runs in parallel under no early stopping
    :return: estimated boosting parameters, optimal number of iterations, MSE path for the number of iterations
    """
    # 1. split the data into the training and validation sets
    data_train, data_dev = data['train'], data['dev']
    X_train, Y_train, Z_train = data_train['X'], data_train['Y'], data_train['Z']
    X_dev, Y_dev, Z_dev = data_dev['X'], data_dev['Y'], data_dev['Z']

    # 2. scale data according to the train set
    # check inputs
    if len(X_train.shape) == 1:
        x_scaler = preprocessing.StandardScaler().fit(X_train.reshape(-1, 1))
        X_train_scaled, X_dev_scaled = x_scaler.transform(X_train.reshape(-1, 1)), x_scaler.transform(X_dev.reshape(-1, 1))
    else:
        x_scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_scaled, X_dev_scaled = x_scaler.transform(X_train), x_scaler.transform(X_dev)

    if len(Z_train.shape) == 1:
        z_scaler = preprocessing.StandardScaler().fit(Z_train.reshape(-1, 1))
        Z_train_scaled, Z_dev_scaled = z_scaler.transform(Z_train.reshape(-1, 1)), z_scaler.transform(Z_dev.reshape(-1, 1))
    else:
        z_scaler = preprocessing.StandardScaler().fit(Z_train)
        Z_train_scaled, Z_dev_scaled = z_scaler.transform(Z_train), z_scaler.transform(Z_dev)

    Xmat_train, Xmat_dev = np.c_[np.ones(len(X_train)), X_train_scaled], np.c_[np.ones(len(X_dev)), X_dev_scaled]
    Zmat_train, Zmat_dev = np.c_[np.ones(len(Z_train)), Z_train_scaled], np.c_[np.ones(len(Z_dev)), Z_dev_scaled]
    data_train_scaled = {'X': Xmat_train, 'Y': Y_train, 'Z': Zmat_train}
    data_dev_scaled = {'X': Xmat_dev, 'Y': Y_dev, 'Z': Zmat_dev}

    # 3. use the validation set to tune the number of iterations

    # auxiliary function evaluates test MSE for m iterations
    def validation_fit(data_dev, M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf):
        # fit using the validation set
        param_mats = post_boostIV_crossfit(data_dev, M, wl_model, iv_model, nu, ls, opt_iv, cf, n_split, scale=False)
        # calculate the test MSE
        h_boost_oos = get_post_boostIV_fit(param_mats, wl_model, M, data_dev['X'], n_split, cf)
        mse_boost = np.mean((h_boost_oos - data_dev['Y']) ** 2)
        return mse_boost

    if early:
        # early stopping: as soon as MSE starts going up, stop
        i = 1
        tol = 1e-4  # for numerical errors
        mse_boost = []
        while i <= len(Ms):
            # run in parallel across folds
            mse_boost_i = validation_fit(data_dev_scaled, Ms[i - 1], wl_model, iv_model, opt_iv, n_split, nu, ls, cf)
            mse_boost.append(np.mean(mse_boost_i))

            # early stopping rule
            if i > 1 and mse_boost[i - 2] < (mse_boost[i - 1] - tol):
                break
            i += 1
        M_optimal = int(Ms[i - 2])
    else:
        # run in parallel across Ms
        mse_boost = Parallel(n_jobs=n_process, verbose=20)(delayed(validation_fit)
                                                           (data_dev_scaled, M, wl_model, iv_model, opt_iv, n_split, nu, ls, cf) for M in Ms)
        #  pick the optimal number of iterations
        M_optimal = Ms[np.argmin(mse_boost)]

    # optimal parameters
    param_opt = post_boostIV_crossfit(data_train_scaled, M_optimal, wl_model, iv_model, nu, ls, opt_iv, cf, n_split, scale=False)

    return {'param_opt': param_opt, 'M_opt': M_optimal, 'MSE_path': mse_boost}
